"""
apply_train_split_patch.py — Adds train-split loading to helper.py, needed
for GOAL adapter few-shot training.

What it does:
1. Backs up helper.py -> helper.py.bak_trainsplit
2. Modifies load_dataset() to accept a `split` argument ("train" or "test"),
   defaulting to "test" so every existing call site (which doesn't pass
   `split`) keeps its current behavior unchanged.
     - CUB: train=False (default/test) vs train=True (train)
     - OxfordIIITPet: split="test" (default) vs split="trainval" (train)
   Other datasets are left as-is (test-only) since they're not in scope for
   the adapter work yet -- extend later if needed.
3. Modifies load_precomputed_features() to accept a `split` argument,
   forwarding it to load_dataset(), and to build the cache filename with a
   "-train" suffix when split="train" -- so the train-split cache can NEVER
   collide with or silently reuse the existing test-split cache (the exact
   bug that bit the VR work earlier today).

Existing call sites in main.py (which don't pass `split`) are completely
unaffected -- this is purely additive.

Run from the repo root in Colab:
    !python apply_train_split_patch.py
"""

import shutil
import ast

HELPER_PATH = "helper.py"

shutil.copy(HELPER_PATH, HELPER_PATH + ".bak_trainsplit")
print(f"Backed up {HELPER_PATH} -> {HELPER_PATH}.bak_trainsplit")

with open(HELPER_PATH, "r") as f:
    content = f.read()

# --- 1. load_dataset(): add split param, default 'test' (matches current behavior) ---

old_signature = "def load_dataset(data_path, dataset_name, custom_loader):"
new_signature = 'def load_dataset(data_path, dataset_name, custom_loader, split="test"):'
if old_signature not in content:
    raise RuntimeError("Could not find load_dataset() signature to patch. Aborting -- helper.py may differ from expected.")
content = content.replace(old_signature, new_signature, 1)

old_cub = '''    elif dataset_name == MyDataset.CUB:
        dataset = CUBDataset(
            data_path,
            train=False,
            transform=None,
            loader=custom_loader,
        )'''
new_cub = '''    elif dataset_name == MyDataset.CUB:
        dataset = CUBDataset(
            data_path,
            train=(split == "train"),
            transform=None,
            loader=custom_loader,
        )'''
if old_cub not in content:
    raise RuntimeError("Could not find CUBDataset block in load_dataset(). Aborting -- check formatting/whitespace differs from expected.")
content = content.replace(old_cub, new_cub, 1)

old_pets = '''    elif dataset_name == MyDataset.OxfordIIITPet:
        dataset = OxfordIIITPet(
            data_path,
            transform=None,
            split="test",
            loader=custom_loader,
        )'''
new_pets = '''    elif dataset_name == MyDataset.OxfordIIITPet:
        dataset = OxfordIIITPet(
            data_path,
            transform=None,
            split=("trainval" if split == "train" else "test"),
            loader=custom_loader,
        )'''
if old_pets not in content:
    raise RuntimeError("Could not find OxfordIIITPet block in load_dataset(). Aborting -- check formatting/whitespace differs from expected.")
content = content.replace(old_pets, new_pets, 1)

print("Patched load_dataset() with split-aware CUB/OxfordIIITPet loading")

# --- 2. load_precomputed_features(): add split param, forward it, suffix cache filename ---

old_lpf_sig = '''def load_precomputed_features(
    model,
    args,
    processor,
    dataset_name: str,
    model_size: str,
    alpha: float,
    n_samples: int,
    batch_size: int,
    num_workers: int,
    data_path: str,
    custom_loader: callable,
    device: torch.device,
    layer1: int,
    layer2: int,
):'''
new_lpf_sig = '''def load_precomputed_features(
    model,
    args,
    processor,
    dataset_name: str,
    model_size: str,
    alpha: float,
    n_samples: int,
    batch_size: int,
    num_workers: int,
    data_path: str,
    custom_loader: callable,
    device: torch.device,
    layer1: int,
    layer2: int,
    split: str = "test",
):'''
if old_lpf_sig not in content:
    raise RuntimeError("Could not find load_precomputed_features() signature to patch. Aborting.")
content = content.replace(old_lpf_sig, new_lpf_sig, 1)

old_filename = '''    filename = os.path.join(save_root, f"{save_file}-crop_fea-1layer-{args.num_crops}-{args.top_k}-with-dino2.pkl")'''
new_filename = '''    filename = os.path.join(
        save_root,
        f"{save_file}-crop_fea-1layer-{args.num_crops}-{args.top_k}-with-dino2"
        f"{'-train' if split == 'train' else ''}.pkl",
    )'''
if old_filename not in content:
    raise RuntimeError("Could not find crop_fea filename construction line. Aborting.")
content = content.replace(old_filename, new_filename, 1)

old_load_dataset_call = '''        dataset = load_dataset(
            data_path=data_path,
            dataset_name=dataset_name,
            custom_loader=custom_loader,
        )'''
new_load_dataset_call = '''        dataset = load_dataset(
            data_path=data_path,
            dataset_name=dataset_name,
            custom_loader=custom_loader,
            split=split,
        )'''
if old_load_dataset_call not in content:
    raise RuntimeError("Could not find load_dataset() call inside load_precomputed_features(). Aborting.")
content = content.replace(old_load_dataset_call, new_load_dataset_call, 1)

print("Patched load_precomputed_features() with split forwarding + split-suffixed cache filename")

# --- 3. Syntax check before writing ---
try:
    ast.parse(content)
except SyntaxError as e:
    raise RuntimeError(
        f"Patch would leave a syntax error at line {e.lineno}: {e.msg}. "
        f"No changes written -- original helper.py is untouched (backup unaffected anyway)."
    )

with open(HELPER_PATH, "w") as f:
    f.write(content)

print("Syntax check passed. helper.py patched successfully.")
print("\nExisting call sites (main.py) are unaffected -- they don't pass `split`, so they default to 'test', identical to current behavior.")
print("To load a train split elsewhere: load_precomputed_features(..., split='train')")
