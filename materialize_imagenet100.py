"""
materialize_imagenet100.py — One-time setup: downloads ilee0022/ImageNet100
(HF mirror of Kaggle's ambityga/imagenet100) and writes it to disk as
class-folders, matching the structure my_datasets/imagenet100.py expects.

Run once per Colab session (or once ever, if persisted to Drive).

Output structure:
    /content/data/imagenet100/
        classes.json          <- ordered list of 100 primary class names
        train/<class_name>/*.jpg
        validation/<class_name>/*.jpg
        test/<class_name>/*.jpg   (no labels reliably usable -- skipped by default, see NOTE below)

NOTE on splits: this pipeline's convention elsewhere (CUB, OxfordPets) is
"train split" for few-shot adapter training and "test split" for the
held-out zero-shot/eval baseline. ImageNet-100's own "test" split here has
no reliable labels for evaluation purposes in many HF ports of this
dataset -- use "train" for few-shot training and "validation" as the
held-out evaluation split (relabel that as `--split test` when calling
load_precomputed_features), unless you confirm the "test" split here is
properly labeled once you inspect it.
"""

import json
import os
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

OUT_ROOT = "/content/data/imagenet100"
SPLITS_TO_MATERIALIZE = ["train", "validation"]  # skip "test" by default, see NOTE above

os.makedirs(OUT_ROOT, exist_ok=True)

print("Downloading ilee0022/ImageNet100 (this pulls ~17GB, may take a while)...")
ds = load_dataset("ilee0022/ImageNet100")

# Build the label -> primary class name mapping once, from the train split
# (text field has comma-separated synonyms, e.g. "spiny lobster, langouste, ...")
label_to_name = {}
for row in ds["train"]:
    label = row["label"]
    if label not in label_to_name:
        primary_name = row["text"].split(",")[0].strip()
        label_to_name[label] = primary_name
    if len(label_to_name) == 100:
        break

class_names = [label_to_name[i] for i in range(100)]
with open(os.path.join(OUT_ROOT, "classes.json"), "w") as f:
    json.dump(class_names, f)
print(f"{len(class_names)} classes written to {OUT_ROOT}/classes.json")
print("First 5:", class_names[:5])

for split in SPLITS_TO_MATERIALIZE:
    print(f"\nMaterializing split: {split} ({len(ds[split])} images)")
    split_dir = os.path.join(OUT_ROOT, split)
    counters = defaultdict(int)
    for row in tqdm(ds[split]):
        label = row["label"]
        class_name = label_to_name[label]
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img = row["image"]  # PIL Image
        idx = counters[label]
        img.save(os.path.join(class_dir, f"{idx:05d}.jpg"))
        counters[label] += 1

print("\nDone. Directory structure:")
os.system(f"ls {OUT_ROOT}")
os.system(f"ls {OUT_ROOT}/train | head -5")
