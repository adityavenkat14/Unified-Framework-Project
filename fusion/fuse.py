"""
S_c = alpha * S_global + beta * S_WCA + gamma * S_LaZSL
Predicted class = argmax S_c

Each score tensor is [B, num_classes]. Because the three scores live on
different scales (raw cosine sim, log-softmax-weighted sim, 1-Wasserstein
similarity), z-normalize each per-image before combining -- otherwise
whichever score has the largest raw magnitude will dominate regardless of
alpha/beta/gamma.
"""
import itertools
import torch


def _znorm(s: torch.Tensor) -> torch.Tensor:
    mean = s.mean(dim=1, keepdim=True)
    std = s.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (s - mean) / std


def fuse_scores(s_global: torch.Tensor, s_wca: torch.Tensor, s_lazsl: torch.Tensor,
                 alpha: float, beta: float, gamma: float, normalize: bool = True) -> torch.Tensor:
    if normalize:
        s_global = _znorm(s_global)
        s_wca = _znorm(s_wca)
        s_lazsl = _znorm(s_lazsl)
    return alpha * s_global + beta * s_wca + gamma * s_lazsl


def accuracy_from_scores(scores: torch.Tensor, target: torch.Tensor) -> float:
    pred = scores.argmax(dim=1)
    return (pred == target).float().mean().item() * 100.0


def grid_search_weights(s_global, s_wca, s_lazsl, target, steps=5):
    """Simple grid search over alpha/beta/gamma in [0,1] (normalized to sum to 1).
    Run this on a held-out validation split, not the test split you report on."""
    best = {"acc": -1.0, "alpha": None, "beta": None, "gamma": None}
    grid = [i / (steps - 1) for i in range(steps)]
    for a, b, g in itertools.product(grid, repeat=3):
        if a + b + g == 0:
            continue
        norm = a + b + g
        a_, b_, g_ = a / norm, b / norm, g / norm
        scores = fuse_scores(s_global, s_wca, s_lazsl, a_, b_, g_)
        acc = accuracy_from_scores(scores, target)
        if acc > best["acc"]:
            best.update({"acc": acc, "alpha": a_, "beta": b_, "gamma": g_})
    return best
