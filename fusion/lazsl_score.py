"""
Computes S_LaZSL for every class using LaZSL's optimal-transport module,
reusing the patch embeddings ABS already extracts (no separate LaZSL data
pipeline needed).

Caveat: OP_d.get_OP_distence scores one class's descriptions against the
batch at a time (Sinkhorn is run per class), so this is O(num_classes)
Sinkhorn solves per batch. Fine for CUB (~200 classes) / OxfordPets (~37);
for ImageNet-scale label spaces, chunk classes and/or raise --lazsl_max_iter
down before scaling this up.
"""
import torch
from tqdm import tqdm
from .lazsl_op import OP_d


def build_op_d(max_iter: int = 100, gama: float = 0.1, theta: float = 0.0,
                constrain_type: str = "const"):
    return OP_d(max_iter=max_iter, gama=gama, theta=theta, constrain_type=constrain_type)


@torch.no_grad()
def compute_lazsl_scores(patch_embeds: torch.Tensor, zeroshot_weights: torch.Tensor,
                          op_d: OP_d, class_chunk: int = 1, batch_chunk: int = 64) -> torch.Tensor:
    """
    patch_embeds:      [B, crop_num, D]  (already L2-normalized, as ABS produces)
    zeroshot_weights:  [num_classes, num_descriptions, D]  (already L2-normalized)
    returns:            [B, num_classes] S_LaZSL similarity scores
    """
    device = patch_embeds.device
    B, crop_num, D = patch_embeds.shape
    num_classes = zeroshot_weights.shape[0]

    all_scores = torch.zeros(B, num_classes, device=device)

    for b_start in tqdm(range(0, B, batch_chunk), desc="LaZSL scoring (batches)"):
        b_end = min(b_start + batch_chunk, B)
        # OP_d expects image_features as [M patches, b batch, D]
        img_feats = patch_embeds[b_start:b_end].permute(1, 0, 2).contiguous()

        for c in range(num_classes):
            text_feats = zeroshot_weights[c]  # [num_descriptions, D]
            score = op_d.get_OP_distence(img_feats, text_feats)  # [b]
            if score is None:
                # Sinkhorn hit NaN for this (batch, class) pair -- leave as 0
                # rather than aborting the whole run.
                continue
            all_scores[b_start:b_end, c] = score

    return all_scores
