# Unified GOAL → ABS(WCA) → LaZSL pipeline

## What's actually merged, and what's a stub

| Piece | Status | Where |
|---|---|---|
| ABS (attention-guided region selection + WCA weighted cross-alignment) | **Working, unmodified** — this is the base repo | `main.py`, `helper.py`, `clip/` |
| GOAL backbone swap | **Wired up, untested** — converts GOAL's HF-CLIP checkpoint into this repo's OpenAI-CLIP key layout | `backbones.py` |
| LaZSL optimal-transport score | **Wired up, untested** — reuses ABS's patch embeddings, loops LaZSL's Sinkhorn solver per class | `fusion/lazsl_score.py` |
| Fusion `S = αS_global + βS_WCA + γS_LaZSL` | **Wired up, untested** | `fusion/fuse.py`, invoked from `main.py` with `--enable_fusion` |

"Untested" = it's syntax-checked and the tensor shapes are traced through by hand against ABS's/LaZSL's actual source, but none of it has run against real data or a GPU, because this sandbox has neither. **Budget your first Colab session for debugging shape mismatches**, not for getting numbers — that's normal for a merge like this, not a sign something's wrong with the plan.

The single highest-risk part is the GOAL→OpenAI-CLIP weight conversion in `backbones.py` — HuggingFace's `CLIPModel` and this repo's `clip/model.py` are the same architecture but different code, so key names had to be hand-mapped. If it errors, print `hf_sd.keys()` vs `model.state_dict().keys()` and diff them; the mapping logic is centralized in `convert_hf_to_openai()`.

## How to run it

```bash
# baseline reproduction (sanity check before touching anything new)
python main.py --dataset_name oxford_pet --backbone openai

# swap in a GOAL-finetuned backbone
python main.py --dataset_name oxford_pet --backbone goal --goal_ckpt /path/to/goal_vitb16_docci.pt

# add LaZSL + fusion on top (needs 'clip' and 'ours' both enabled in cfgs/<dataset>.yaml)
python main.py --dataset_name oxford_pet --enable_fusion --alpha 0.34 --beta 0.33 --gamma 0.33

# find good fusion weights on a validation split first, THEN hardcode them for your test run
python main.py --dataset_name oxford_pet --enable_fusion --fusion_search
```

## Colab Pro workflow — don't zip the repos into Drive

Zipping and re-uploading the code repos is the wrong pattern here, for two reasons: it's an extra manual step every time you fix a bug, and Drive I/O for many small files is slow. Use Drive for the things that are actually expensive to redo, and clone code fresh each session:

**Every session, clone straight from GitHub (cheap, ~seconds):**
```python
!git clone https://github.com/tmlr-group/WCA.git      # reference only
!git clone https://github.com/BIT-DA/ABS.git
!git clone https://github.com/shiming-chen/LaZSL.git   # reference only
!git clone https://github.com/PerceptualAI-Lab/GOAL.git
```
Then upload this `merged/` folder's *new* files (`backbones.py`, `fusion/`, the patched `main.py`) into the cloned `ABS` folder — or, better, push this merged folder to your own private GitHub repo once, and just `git clone` that one repo each session. That's the one thing worth doing once instead of every time.

**Put in Google Drive (mount it, point paths at it), because these are large and slow to regenerate:**
- The datasets themselves (`data_path` in `cfgs/<dataset>.yaml`)
- GOAL's fine-tuned checkpoints (the `.pt` files, ~1-2GB each)
- The `features/` cache — ABS's `helper.py` pickles precomputed patch embeddings per dataset/config so it doesn't recompute them every run; without Drive, that cache vanishes when the Colab runtime recycles and every run pays the DINO+CLIP forward-pass cost again

```python
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/unified_zsl'
# in cfgs/<dataset>.yaml: data_path -> f'{DRIVE_ROOT}/data/oxford_pet'
# helper.py's save_root for the features cache -> f'{DRIVE_ROOT}/features'
# --goal_ckpt -> f'{DRIVE_ROOT}/checkpoints/goal_vitb16_docci.pt'
```

That gets you: fast code iteration (fresh clone or your own repo, no zip juggling), and persistence for the one-time-download things (data, checkpoints, feature cache) that you don't want to lose every time the Colab runtime disconnects.

## Suggested first session checklist
1. Reproduce ABS's own baseline numbers on one small dataset (OxfordPets or CUB) with `--backbone openai` — confirms your environment + data path + feature cache are all correct before anything new is in the loop.
2. Download one GOAL checkpoint (ViT-B/16, DOCCI) to Drive, run `--backbone goal`, fix whatever the key-conversion assertion complains about.
3. Run `--enable_fusion` standalone (no GOAL) first, so if something breaks you know it's the LaZSL/fusion wiring and not the backbone swap.
4. Only then combine GOAL backbone + fusion together.
