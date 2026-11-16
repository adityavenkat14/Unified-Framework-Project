# Reviewer-response code (ICLR 2027 rebuttal)

This code was added for the rebuttal. It does not change the UVA method or any existing script.

## Evaluation with the saved checkpoints (repository root)
| File | What it does | Needs |
|---|---|---|
| `eval_reviewer.py` | See the run order below. | Test-split crop caches (`features/<ds>/*-dino2.pkl`) + CLIP ViT-B/16 |
| `low_compute_baselines.py` | Training-free Tip-Adapter and the global-feature adapter (UVA's adapters, loss and settings on the CLS embedding), both on UVA's exact 16-shot draw (seed 0). | Train and test caches (`*-dino2-train.pkl`, `*-dino2.pkl`); CLIP text encoder only |
| `notebooks/uva_reviewer_eval_kaggle.ipynb` | Kaggle notebook that runs both scripts (Run All). | GPU session with internet; `features.tar` as input |

`eval_reviewer.py` runs in this order:
1. **Reproduction check.** It recomputes the paper's CLIP, zero-shot ABS+WCA and UVA numbers, and stops a dataset if any is more than 0.5 points off.
2. **The reviewer experiments:**
   - global-feature control;
   - ABS+WCA with BiFTA-DR and no adapters;
   - test-time single-adapter variants;
   - WCA/LaZSL agreement;
   - difficulty bins with a split-half design.
3. **Outputs and self-check.** It saves predictions and runs a numpy self-check.

Both scripts write `results/*.csv`, `results/*_meta.json` (command, hashes, versions) and image-level predictions.

## Analyses that need no model or data (this folder)
| File | What it does |
|---|---|
| `description_audit.py <repo> <out>` | Pool sizes, BiFTA-DR source shares, overlap with the first 50 CuPL descriptions, exact de-duplication counts |
| `adapter_weight_stats.py <repo> <out>` | Weight norms, singular values and effective rank of the released adapters |
| `audit_consistency_checks.py` | Integer-count consistency of logged accuracies; reconstruction of the Pets agreement confusion matrix |
| `verify_literature_rows.py` | Checks transcribed published results by recomputing each row's printed average |
| `consistency_audit.py` | Traces every decimal number in the manuscript/rebuttal to the log, derived outputs, literature or arithmetic |
