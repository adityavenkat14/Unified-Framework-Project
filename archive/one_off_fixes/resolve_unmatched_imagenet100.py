"""
Resolve unmatched ImageNet-100 class folder names against CuPL's full
998-key ImageNet-1K json, by checking substring/token overlap instead of
exact string equality (CuPL likely keys some classes by their FULL
comma-separated synonym string, e.g. "sidewinder, horned rattlesnake",
while our folder name is just the first synonym, "sidewinder").

Usage:
    python resolve_unmatched_imagenet100.py --cupl_full cupl_imagenet_full.json
"""
import argparse
import json

UNMATCHED = [
    "bittern", "black and gold garden spider", "black widow", "common iguana",
    "common newt", "crane", "diamondback", "drake", "garden spider",
    "green lizard", "green snake", "hammerhead", "hognose snake",
    "horned viper", "king snake", "kite", "leatherback turtle", "loggerhead",
    "peacock", "prairie chicken", "red-backed sandpiper", "redshank",
    "sidewinder", "thunder snake", "water ouzel", "whiptail",
]


def normalize(s: str) -> str:
    return " ".join(s.replace("_", " ").replace("-", " ").lower().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cupl_full", default="cupl_imagenet_full.json")
    args = parser.parse_args()

    with open(args.cupl_full) as f:
        cupl_full = json.load(f)

    cupl_keys = list(cupl_full.keys())
    norm_keys = {k: normalize(k) for k in cupl_keys}

    print(f"{'Folder name':35s} -> Candidate CuPL key(s)")
    print("-" * 90)

    results = {}
    for folder_name in UNMATCHED:
        norm_folder = normalize(folder_name)
        candidates = []

        for orig_key, norm_key in norm_keys.items():
            # Split key on commas (CuPL/ImageNet synonym lists) and check
            # if our folder name matches any individual synonym fragment,
            # OR appears as a substring of the whole key, OR vice versa.
            fragments = [normalize(f) for f in orig_key.split(",")]
            if (
                norm_folder in fragments
                or norm_folder in norm_key
                or norm_key in norm_folder
                or any(norm_folder == frag for frag in fragments)
            ):
                candidates.append(orig_key)

        results[folder_name] = candidates
        cand_str = " | ".join(repr(c) for c in candidates) if candidates else "*** NO CANDIDATE ***"
        print(f"{folder_name:35s} -> {cand_str}")

    still_missing = [k for k, v in results.items() if not v]
    ambiguous = [k for k, v in results.items() if len(v) > 1]

    print(f"\nResolved candidates: {len(UNMATCHED) - len(still_missing)}/{len(UNMATCHED)}")
    if ambiguous:
        print(f"⚠️  {len(ambiguous)} have MULTIPLE candidates, need manual pick: {ambiguous}")
    if still_missing:
        print(f"⚠️  {len(still_missing)} still have NO candidate at all: {still_missing}")

    with open("unmatched_resolution.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote unmatched_resolution.json for review.")


if __name__ == "__main__":
    main()
