"""
Second-pass resolver for the classes that survived the first substring pass.
Uses individual word overlap (ignoring common filler words) instead of
requiring the whole phrase to match, since CuPL may use a differently
ordered or partially different synonym string.

Usage:
    python resolve_pass2.py --cupl_full cupl_imagenet_full.json
"""
import argparse
import json

# Words too generic to count as a meaningful match on their own
STOPWORDS = {"bird", "the", "a", "of", "and"}

REMAINING = [
    "common iguana", "drake", "hognose snake", "king snake",
    "leatherback turtle", "peacock", "prairie chicken",
    "red-backed sandpiper", "thunder snake", "water ouzel",
]


def normalize(s: str) -> str:
    return " ".join(s.replace("_", " ").replace("-", " ").lower().split())


def significant_words(phrase: str) -> set:
    return {w for w in normalize(phrase).split() if w not in STOPWORDS and len(w) > 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cupl_full", default="cupl_imagenet_full.json")
    args = parser.parse_args()

    with open(args.cupl_full) as f:
        cupl_full = json.load(f)

    cupl_keys = list(cupl_full.keys())

    print(f"{'Folder name':25s} -> Candidate CuPL key(s) (ranked by word overlap)")
    print("-" * 100)

    results = {}
    for folder_name in REMAINING:
        folder_words = significant_words(folder_name)
        scored = []
        for key in cupl_keys:
            # compare against each comma-separated synonym fragment separately
            for frag in key.split(","):
                frag_words = significant_words(frag)
                overlap = folder_words & frag_words
                if overlap:
                    scored.append((len(overlap), key, frag.strip()))
        scored.sort(reverse=True)
        top = scored[:5]
        results[folder_name] = [k for _, k, _ in top]

        if top:
            shown = " | ".join(f"{k!r} (matched on {ov} words via {frag!r})" for ov, k, frag in top[:3])
        else:
            shown = "*** STILL NO CANDIDATE ***"
        print(f"{folder_name:25s} -> {shown}")

    with open("unmatched_resolution_pass2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote unmatched_resolution_pass2.json for review.")
    print("Paste the printed table back — pick the correct key for each row before we finalize cupl.json.")


if __name__ == "__main__":
    main()
