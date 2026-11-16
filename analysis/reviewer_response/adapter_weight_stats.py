"""
adapter_weight_stats.py -- REAL analysis of the two saved adapter checkpoints.
No data, no CLIP. Only the weights are inspected.

For each adapter (image / text) of each checkpoint:
  * Frobenius norm of each layer, and the ratio to the expected norm of
    PyTorch's default nn.Linear initialisation (U(-1/sqrt(fan_in), 1/sqrt(fan_in)),
    expected squared norm = rows * cols / (3 * fan_in)). Ratio ~1 means the layer
    stayed near initialisation scale; it does NOT measure distance from the
    actual (unrecorded) initial weights.
  * singular-value spectrum of the composed map W2 @ W1 (the adapter is
    W2 relu(W1 x), so this is only an upper-bound proxy) and its effective rank
    (exp of the entropy of normalised singular values).
  * largest singular value x lambda (0.2): an upper bound on how much the
    residual branch can add to a unit-norm input before re-normalisation
    (ReLU and dropout can only reduce it).

Usage: python adapter_weight_stats.py <repo_root> <out_dir>
"""
import csv
import hashlib
import json
import os
import sys

import torch

repo, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
LAMBDA = 0.2
rows = []
for ds in ["cub", "oxford_pet"]:
    path = os.path.join(repo, f"adapters_{ds}_16shot.pt")
    sd = torch.load(path, map_location="cpu")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    for side in ["image_adapter", "text_adapter"]:
        w1 = sd[f"{side}.mlp.0.weight"].double()  # [128, 512]
        w2 = sd[f"{side}.mlp.3.weight"].double()  # [512, 128]
        exp1 = (w1.numel() / (3 * w1.shape[1])) ** 0.5
        exp2 = (w2.numel() / (3 * w2.shape[1])) ** 0.5
        s = torch.linalg.svdvals(w2 @ w1)
        pnorm = s / s.sum()
        eff_rank = float(torch.exp(-(pnorm * torch.log(pnorm + 1e-300)).sum()))
        rows.append({
            "checkpoint": f"adapters_{ds}_16shot.pt", "sha256_prefix": sha[:16], "adapter": side,
            "W1_fro": round(float(w1.norm()), 4), "W1_fro_over_init_expected": round(float(w1.norm()) / exp1, 4),
            "W2_fro": round(float(w2.norm()), 4), "W2_fro_over_init_expected": round(float(w2.norm()) / exp2, 4),
            "W2W1_sigma_max": round(float(s[0]), 4),
            "W2W1_sigma_max_x_lambda": round(float(s[0]) * LAMBDA, 4),
            "W2W1_effective_rank": round(eff_rank, 2),
            "W2W1_top10_energy_pct": round(float((s[:10] ** 2).sum() / (s ** 2).sum() * 100), 2),
        })

with open(os.path.join(out, "adapter_weight_stats.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(json.dumps(r))
