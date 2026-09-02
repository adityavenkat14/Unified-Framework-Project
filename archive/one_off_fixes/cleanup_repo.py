"""
cleanup_repo.py — Archive one-time/superseded files into organized
subfolders. Uses `git mv` (preserves history) and only ever moves files
that are safe to relocate -- never touches core scripts that unified.py
or other scripts import/subprocess-call by relative path.

Run from the repo root:
    python cleanup_repo.py           # dry run, prints what WOULD move
    python cleanup_repo.py --apply   # actually moves files + stages them
"""

import argparse
import os
import subprocess

# (source path, destination path) -- only files confirmed safe to move:
# one-time patch scripts (already applied, kept as historical record /
# safety net for re-application after a bad reset), superseded scripts,
# and stray .bak_* backup files scattered around from today's patches.
MOVES = [
    # One-time patch scripts -- already applied to their target files;
    # kept in case a future reset needs them reapplied.
    ("apply_train_split_patch.py", "scripts/one_time_patches/apply_train_split_patch.py"),
    ("apply_vr_patch.py", "scripts/one_time_patches/apply_vr_patch.py"),
    ("fix_dino_device_mismatch.py", "scripts/one_time_patches/fix_dino_device_mismatch.py"),
    ("fix_vr_import_placement.py", "scripts/one_time_patches/fix_vr_import_placement.py"),
    ("apply_imagenet100_patch.py", "scripts/one_time_patches/apply_imagenet100_patch.py"),

    # Superseded / unused
    ("predict_image.py", "legacy/predict_image.py"),  # superseded by predict_image_real.py, then predict_unified.py
    ("imagenet100.py", "legacy/imagenet100.py"),  # unused -- ImageFolder (existing repo pattern) used instead

    # Stray backup files from today's in-place patches (.bak_* suffix) --
    # these accumulate every time a patch script runs; safe to archive.
    ("helper.py.bak_trainsplit", "legacy/backups/helper.py.bak_trainsplit"),
    ("clip/model.py.bak_vr", "legacy/backups/model.py.bak_vr"),
    ("clip/model.py.bak_vr_fix", "legacy/backups/model.py.bak_vr_fix"),
    ("goal_adapter_train.py.bak_devicefix", "legacy/backups/goal_adapter_train.py.bak_devicefix"),
    ("my_datasets/__init__.py.bak_imagenet100", "legacy/backups/__init__.py.bak_imagenet100"),
    ("helper.py.bak_imagenet100", "legacy/backups/helper.py.bak_imagenet100"),
]

# Diagnostic/exploratory scripts from the original 245-turn session --
# still historically valuable (produced the GOAL 0%-tail finding that
# motivated the adapter), but not part of the active pipeline. Archived
# separately, NOT deleted, since their content wasn't available to
# inspect/merge into goal_pipeline.py in this session.
DIAGNOSTIC_CANDIDATES = [
    "goal_supervision.py",
    "goal_class_diagnostic.py",
    "correlate_goal_diagnostic.py",
    "description_auditor.py",
    "abs_explain.py",
]


def run(cmd, apply):
    print(f"{'$ ' if apply else '[dry-run] would run: '}{' '.join(cmd)}")
    if apply:
        subprocess.run(cmd)


def main(apply: bool):
    print(f"{'APPLYING' if apply else 'DRY RUN (pass --apply to actually move files)'}\n")

    moved, skipped = 0, 0
    for src, dst in MOVES:
        if not os.path.exists(src):
            print(f"  [skip, not found] {src}")
            skipped += 1
            continue
        dst_dir = os.path.dirname(dst)
        if apply:
            os.makedirs(dst_dir, exist_ok=True)
        run(["git", "mv", src, dst], apply)
        moved += 1

    print(f"\n{moved} files moved, {skipped} skipped (not found -- already clean or never existed here).")

    print("\n--- Diagnostic scripts found (not moved automatically -- review first) ---")
    for f in DIAGNOSTIC_CANDIDATES:
        status = "FOUND" if os.path.exists(f) else "not present"
        print(f"  [{status}] {f}")
    print(
        "\nThese hold real historical value (the GOAL diagnostic here is what\n"
        "originally motivated building the adapter) but weren't merged into\n"
        "goal_pipeline.py in this session (their content was never pasted in).\n"
        "Recommend: `git mv <file> diagnostics/<file>` manually for any FOUND\n"
        "above, once you've confirmed nothing else in the repo still calls them."
    )

    if apply:
        print("\nStaged all moves. Review with `git status`, then commit:")
        print('  git commit -m "Archive one-time patches and superseded files"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually perform the moves (default: dry run)")
    args = parser.parse_args()
    main(args.apply)
