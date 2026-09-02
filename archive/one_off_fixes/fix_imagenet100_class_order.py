"""
fix_imagenet100_class_order.py

Root cause: the materialized dataset's folders are named by WordNet synset
ID (n01498041, ...), so ImageFolder assigns label indices by alphabetical
synset order -- completely different from load_classes()'s human-readable
name order. This rebuilds features/imagenet100/imagenet100.json to match
ImageFolder's REAL order, using the standard public wnid->name mapping,
and explicitly reports any name that doesn't match cupl.json's actual
keys (rather than assuming it'll just work).
"""

import json
import os
import urllib.request

from torchvision.datasets import ImageFolder

DATA_ROOT = "/content/data/imagenet100/validation"
OUT_PATH = "features/imagenet100/imagenet100.json"
CUPL_PATH = "prompts/imagenet100/cupl.json"

# Standard, widely-mirrored ImageNet class index file: {"0": ["n01440764", "tench"], ...}
MAPPING_URL = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"

print("Fetching standard wnid -> class-name mapping ...")
with urllib.request.urlopen(MAPPING_URL) as resp:
    class_index = json.load(resp)

wnid_to_name = {}
for _, (wnid, name) in class_index.items():
    wnid_to_name[wnid] = name.replace("_", " ")

print(f"Loaded {len(wnid_to_name)} wnid->name mappings.")

# --- Get ImageFolder's REAL ordering (this is what image labels actually use) ---
imf = ImageFolder(root=DATA_ROOT)
wnid_order = imf.classes  # already sorted alphabetically, exactly as torchvision does internally
print(f"\nImageFolder found {len(wnid_order)} classes (real label order).")

# --- Map each wnid, in that exact order, to its class name ---
ordered_names = []
missing_wnid = []
for wnid in wnid_order:
    if wnid in wnid_to_name:
        ordered_names.append(wnid_to_name[wnid])
    else:
        ordered_names.append(None)
        missing_wnid.append(wnid)

if missing_wnid:
    print(f"\n[WARNING] {len(missing_wnid)} wnids not found in the standard mapping: {missing_wnid}")
    print("These will need manual resolution before proceeding.")

# --- Cross-check against cupl.json's actual keys ---
with open(CUPL_PATH) as f:
    cupl = json.load(f)
cupl_keys = set(cupl.keys())

mismatches = [(i, n) for i, n in enumerate(ordered_names) if n is not None and n not in cupl_keys]

print(f"\n{len(mismatches)} names not found as exact keys in cupl.json:")
for i, n in mismatches:
    print(f"  index {i}: '{n}'")

if mismatches or missing_wnid:
    print(
        "\nSTOPPING before writing -- resolve the mismatches above first "
        "(likely a synonym/capitalization difference between the standard "
        "mapping's name and cupl.json's key for that class). "
        "Print a few cupl.json keys near each mismatch to compare by hand:"
    )
    for i, n in mismatches[:5]:
        # show cupl keys that share the first few characters, as a hint
        candidates = [k for k in cupl_keys if k.lower()[:4] == (n or "").lower()[:4]]
        print(f"  '{n}' -> possible cupl.json matches: {candidates}")
else:
    with open(OUT_PATH, "w") as f:
        json.dump(ordered_names, f)
    print(f"\nAll {len(ordered_names)} names verified against cupl.json. Wrote {OUT_PATH}.")
    print("First 10:", ordered_names[:10])
