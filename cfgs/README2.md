# Notes on the dataset configs (read before comparing with the paper)

The YAML files in this folder were inherited from the ABS code base, and some of
their keys do **not** control the UVA runs reported in the paper:

| Key | Status in this code | What actually controls it |
|---|---|---|
| `alpha` | passed to `load_precomputed_features()` but never used there | `main.py --clip_crop_r1` (default **0.6**) and `--clip_crop_r2` (default **0.9**) |
| `n_samples` | passed to `load_precomputed_features()` but never used there | `main.py --num_crops` (default **50** raw-image crops, plus 50 feature-space crops = 100 regions) |
| `patch_n` | feeds `random_indices` in the `n_run` loop of `main.py` / `methods/goal_adapter_eval.py`, which are computed but never used; all 100 regions are scored | – |
| `n_run` | repeats the same deterministic evaluation (the log notes the resulting `±0.00`) | – |

So `cfgs/cub.yaml` showing `alpha: 0.5`, `n_samples: 60`, `patch_n: 60` does **not**
mean the paper's ABS settings (α = 0.6, 50 + 50 regions, k = 20) are misreported:
the effective values are the command-line defaults above. The files are left
unchanged so that the original experiment configuration is preserved.

ImageNet-100: the 30-images-per-class training pool described in the paper was
applied with a local change to the data loader that is not in this repository
(see `EXPERIMENT_LOG.md`, around line 338). Evaluation reads
`<data_path>/validation`, i.e. the 13,000-image `validation` split of the
Hugging Face mirror `ilee0022/ImageNet100` (see `demo/materialize_imagenet100.py`).
