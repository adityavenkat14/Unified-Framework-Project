# Experiment Log

Tracks progress against the phased plan: (1) reproduce ABS baseline, (2) swap in GOAL backbone,
(3) add LaZSL fusion, (4) tune fusion weights, (5) evaluate across datasets.

## Phase 1 — ABS/WCA baseline reproduction

**Dataset:** OxfordPets (37 classes) | **Backbone:** vanilla CLIP ViT-B/16 (`--backbone openai`)

| Method | Accuracy | Notes |
|---|---|---|
| clip | 88.20 | plain "a photo of a {class}." — matches published zero-shot CLIP ViT-B/16 ballpark (~87-89%) |
| clip-e | 89.10 | class name + "a type of pet" |
| clip-d | 87.54 | LLM-generated single-attribute descriptions |
| waffle | 86.48 | randomized nonsense-word attributes by design; dip below plain clip is a known quirk, not a bug |
| cupl | 91.71 | LLM-generated multi-sentence descriptions, biggest lift of the baselines |
| ours (ABS/WCA) | **92.94** (±0.00) | beats every baseline incl. cupl (91.71) -- ABS's contribution is confirmed working |

**Verdict:** Phase 1 confirmed successful. `ours` beats all baselines as expected. Clear to move to Phase 2
(GOAL backbone swap).

Note on the ±0.00: `random_indices` is computed each of the 10 `n_run` iterations but never actually used --
`sampled_features` is always set to the full `precomputed_features` regardless, so every run is identical by
construction. This is a quirk in ABS's own upstream code, not something introduced here. Don't read 0.00 std
as "stable across random seeds" -- there's no real resampling happening.

## Phase 2 — GOAL backbone swap
Ready to start. Next: download a GOAL ViT-B/16 (DOCCI-tuned) checkpoint to
`{DRIVE_ROOT}/checkpoints/`, then run:
```
!python main.py --dataset_name oxford_pet --backbone goal --goal_ckpt {DRIVE_ROOT}/checkpoints/<file>.pt
```
Compare the resulting `clip`/`ours` numbers against this session's openai-backbone baseline
(clip 88.20, ours 92.94) to see whether GOAL's dense-caption finetuning helps or hurts classification.

## Phase 3 — LaZSL fusion score
Not started.

## Phase 4 — Fusion weight tuning
Not started.

## Phase 5 — Multi-dataset evaluation
Not started. Static class-name files (`features/<dataset>/<dataset>.json`) now in place for: dtd, imagenet,
imagenet-r, imagenetv2, imagenet-s, cub, imagenet-a, food101, place365, oxford_pet.

## Known environment gotchas (so we don't re-debug these)
- `requirements.txt` in the original ABS repo is a `conda create --file` export, not pip-installable —
  use the pip-compatible one instead.
- HF's `ViTModel` defaults to `sdpa` attention, which silently ignores `output_attentions=True` — DINO
  attention loading requires `attn_implementation="eager"` (already patched in `main.py`/`helper.py`).
- Default `num_workers=8` in `main.py` is too high for Colab's ~2 CPU cores — lowered to 2.
- Extracting many small files (e.g. OxfordPets' ~7,400 images) directly onto a Drive-mounted path is very
  slow/can hang — download archives to Drive for caching, but extract to local `/content` disk.
- `features/` is a symlink to Drive — git can't `git add` through it ("beyond a symbolic link"); files
  placed there don't need a commit/push, they're already persisted to Drive directly.
- Colab's file upload sidebar drops files in `/content/`, not your notebook's current working directory —
  check `os.getcwd()` vs where uploads land if a script reports files "not found".
