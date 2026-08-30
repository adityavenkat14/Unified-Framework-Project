"""
bootstrap_session.py — Run this FIRST in any new Colab session.

Restores everything persisted from previous sessions (repo via git,
datasets + feature caches via Drive tarballs) and prints a clear report of
what's present vs. still missing, so gaps are caught in seconds instead of
one crash at a time deep into a run.

Usage (paste into a Colab cell, not run as a script, since it needs
drive.mount()'s interactive auth flow):

    from google.colab import drive
    drive.mount('/content/drive')
    exec(open('bootstrap_session.py').read())

Or just run the numbered steps below manually -- either works.
"""

import os
import subprocess

REPO_URL = "https://github.com/adityavenkat14/Unified-Framework-Project.git"
REPO_DIR = "/content/Unified-Framework-Project"
DRIVE_ROOT = "/content/drive/MyDrive/unified-framework-cache"

CHECKPOINTS = [
    "adapters_oxford_pet_16shot.pt",
    "adapters_cub_16shot.pt",
]

CRITICAL_SMALL_FILES = [
    "features/oxford_pet/oxford_pet.json",
    "features/cub/cub.json",
    "prompts/oxford_pet/bifta-dr.json",
    "prompts/cub/bifta-dr.json",
]

CROP_CACHES = [
    "features/oxford_pet/oxford_pet-ViT-B-16-crop_fea-1layer-50-20-with-dino2.pkl",
    "features/oxford_pet/oxford_pet-ViT-B-16-crop_fea-1layer-50-20-with-dino2-train.pkl",
    "features/cub/cub-ViT-B-16-crop_fea-1layer-50-20-with-dino2.pkl",
    "features/cub/cub-ViT-B-16-crop_fea-1layer-50-20-with-dino2-train.pkl",
]


def run(cmd):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True)


print("=" * 60)
print("STEP 1: Clone/pull the repo")
print("=" * 60)
if not os.path.exists(REPO_DIR):
    run(f"git clone {REPO_URL} {REPO_DIR}")
else:
    run(f"cd {REPO_DIR} && git pull")
os.chdir(REPO_DIR)
print(f"cwd: {os.getcwd()}")

print("\n" + "=" * 60)
print("STEP 2: Restore data + features from Drive (if present)")
print("=" * 60)
if os.path.exists(f"{DRIVE_ROOT}/data.tar"):
    run(f"cp {DRIVE_ROOT}/data.tar /content/data.tar && tar -xf /content/data.tar -C /content")
    print("Restored /content/data from Drive.")
else:
    print("No data.tar in Drive -- datasets will need fresh downloads.")

if os.path.exists(f"{DRIVE_ROOT}/features.tar"):
    run(f"cp {DRIVE_ROOT}/features.tar /content/features.tar && tar -xf /content/features.tar -C {REPO_DIR}")
    print("Restored features/ from Drive.")
else:
    print("No features.tar in Drive -- crop caches will need rebuilding.")

print("\n" + "=" * 60)
print("STEP 3: Restore adapter checkpoints from Drive (if present)")
print("=" * 60)
for ckpt in CHECKPOINTS:
    drive_path = f"{DRIVE_ROOT}/{ckpt}"
    local_path = f"{REPO_DIR}/{ckpt}"
    if os.path.exists(local_path):
        print(f"[OK, already local] {ckpt}")
    elif os.path.exists(drive_path):
        run(f"cp {drive_path} {local_path}")
        print(f"[RESTORED FROM DRIVE] {ckpt}")
    else:
        print(f"[MISSING -- not in repo, not in Drive] {ckpt} -- will need retraining")

print("\n" + "=" * 60)
print("STATUS REPORT")
print("=" * 60)

print("\n-- Small generated files (should be in git) --")
for f in CRITICAL_SMALL_FILES:
    path = os.path.join(REPO_DIR, f)
    status = "OK" if os.path.exists(path) else "MISSING (regenerate: see project log)"
    print(f"  [{status}] {f}")

print("\n-- Crop feature caches (large, expensive to rebuild) --")
for f in CROP_CACHES:
    path = os.path.join(REPO_DIR, f)
    status = "OK" if os.path.exists(path) else "MISSING (rebuild: ~35-90 min depending on dataset/split)"
    print(f"  [{status}] {f}")

print("\n-- Adapter checkpoints --")
for ckpt in CHECKPOINTS:
    path = os.path.join(REPO_DIR, ckpt)
    status = "OK" if os.path.exists(path) else "MISSING (retrain: fast if crop cache above is OK, else rebuild cache first)"
    print(f"  [{status}] {ckpt}")

print("\n" + "=" * 60)
print("Bootstrap complete. Fix any MISSING items above before running")
print("anything expensive -- this avoids discovering gaps mid-run.")
print("=" * 60)
