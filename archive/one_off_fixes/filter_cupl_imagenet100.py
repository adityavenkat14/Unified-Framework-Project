"""
Filter CuPL's full ImageNet-1K description JSON down to the 100 classes
used by ImageNet-100 (materialized via materialize_imagenet100.py).

Matches by the ImageFolder subfolder names (human-readable class names),
case-insensitively, against CuPL's keys. Writes prompts/imagenet100/cupl.json
in the same {class_name: [descriptions...]} shape as the other datasets'
cupl.json files.

Usage:
    python filter_cupl_imagenet100.py \
        --cupl_full cupl_imagenet_full.json \
        --data_root /content/data/imagenet100/train \
        --out prompts/imagenet100/cupl.json
"""
import argparse
import json
import os


def normalize(name: str) -> str:
    """Lowercase + collapse whitespace/underscores for robust matching."""
    return " ".join(name.replace("_", " ").lower().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cupl_full", default="cupl_imagenet_full.json")
    parser.add_argument("--data_root", default="/content/data/imagenet100/train")
    parser.add_argument("--out", default="prompts/imagenet100/cupl.json")
    args = parser.parse_args()

    with open(args.cupl_full) as f:
        cupl_full = json.load(f)

    # Build a normalized lookup: normalized_key -> (original_key, descriptions)
    norm_lookup = {}
    for k, v in cupl_full.items():
        norm_lookup[normalize(k)] = (k, v)

    # Our 100 class folder names, in whatever order os.listdir gives them
    # (doesn't matter for this dict-based json; ImageFolder sorts alphabetically
    # itself when it loads images, this file just needs the right keys present)
    class_folders = sorted(
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    )
    print(f"Found {len(class_folders)} class folders in {args.data_root}")

    filtered = {}
    unmatched = []

    for folder_name in class_folders:
        norm_name = normalize(folder_name)
        if norm_name in norm_lookup:
            _, descriptions = norm_lookup[norm_name]
            # Key the output by the EXACT folder name, since that's what
            # the rest of the pipeline will look up class descriptions by.
            filtered[folder_name] = descriptions
        else:
            unmatched.append(folder_name)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Matched: {len(filtered)}/{len(class_folders)}")
    print(f"Wrote {args.out}")

    if unmatched:
        print(f"\n⚠️  {len(unmatched)} class(es) had NO match in CuPL's file:")
        for name in unmatched:
            print(f"   - {name!r}")
        print(
            "\nThese need manual resolution — likely a naming mismatch "
            "(e.g. CuPL uses a different common name, or a compound/hyphenated "
            "form). Check imagenet_classnames_ref.txt or search the unmatched "
            "name manually against cupl_imagenet_full.json's keys."
        )
    else:
        print("\nAll 100 classes matched cleanly. No manual fixes needed.")


if __name__ == "__main__":
    main()
