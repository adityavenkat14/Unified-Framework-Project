"""
adapter.py — Lightweight residual MLP adapters for frozen CLIP features.

Same pattern as CLIP-Adapter / Tip-Adapter: CLIP itself stays completely
frozen. Only these small adapters learn, via a residual connection so the
adapter can only *refine* the frozen embedding, never replace it outright
early in training.

Two adapters: one for image-region features, one for text/description
features. Kept as separate modules (not shared weights) since image and
text embeddings, while in the same joint space, come from different
encoders and empirically benefit from independent adapters in this
literature (CLIP-Adapter ablates this).
"""

import torch
import torch.nn as nn


class ResidualMLPAdapter(nn.Module):
    """
    x_adapted = ratio * mlp(x) + (1 - ratio) * x

    `ratio` starts low (default 0.2) so the adapter begins close to a no-op
    (i.e. close to raw frozen CLIP features) and the residual blend lets
    training gradually lean on the adapter only as much as it helps —
    standard CLIP-Adapter practice, avoids destroying the pretrained
    embedding space early in few-shot training.
    """

    def __init__(self, dim: int, hidden_ratio: float = 0.25, ratio: float = 0.2, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(1, int(dim * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
        )
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.mlp(x)
        out = self.ratio * residual + (1 - self.ratio) * x
        return out / out.norm(dim=-1, keepdim=True)


class GoalAdapterPair(nn.Module):
    """
    Holds the image-side and text-side adapters together, since they're
    always trained and checkpointed jointly for this project's contrastive
    objective.
    """

    def __init__(self, dim: int, hidden_ratio: float = 0.25, ratio: float = 0.2, dropout: float = 0.1):
        super().__init__()
        self.image_adapter = ResidualMLPAdapter(dim, hidden_ratio, ratio, dropout)
        self.text_adapter = ResidualMLPAdapter(dim, hidden_ratio, ratio, dropout)

    def forward_image(self, image_features: torch.Tensor) -> torch.Tensor:
        """image_features: [N, D], already L2-normalized frozen CLIP features."""
        return self.image_adapter(image_features)

    def forward_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """text_features: [N, D], already L2-normalized frozen CLIP features."""
        return self.text_adapter(text_features)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
