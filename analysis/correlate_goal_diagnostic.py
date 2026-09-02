"""
Correlates goal_per_class_grounding.json (from goal_class_diagnostic.py) against
wca_per_class_acc.json (written by main.py's --enable_fusion pass) to test:
does GOAL's grounding quality for a class's descriptions predict how well
WCA classifies that class?

Usage:
    python correlate_goal_diagnostic.py --dataset_name oxford_pet
"""
import argparse
import json

import numpy as np


def main(args):
    with open(f"features/{args.dataset_name}/goal_per_class_grounding.json") as f:
        grounding = json.load(f)
    with open(f"features/{args.dataset_name}/wca_per_class_acc.json") as f:
        acc = json.load(f)

    common = sorted(set(grounding.keys()) & set(acc.keys()), key=int)
    if len(common) < 3:
        print(f"Only {len(common)} classes in common -- not enough to correlate. "
              f"Did both scripts run on the same dataset/backbone?")
        return

    top_scores = np.array([grounding[c]["top_score"] for c in common])
    mean_scores = np.array([grounding[c]["mean_score"] for c in common])
    accuracies = np.array([acc[c] for c in common])

    r_top = np.corrcoef(top_scores, accuracies)[0, 1]
    r_mean = np.corrcoef(mean_scores, accuracies)[0, 1]

    print(f"{len(common)} classes compared.")
    print(f"Correlation (GOAL top grounding score  vs. WCA per-class accuracy): r = {r_top:.3f}")
    print(f"Correlation (GOAL mean grounding score vs. WCA per-class accuracy): r = {r_mean:.3f}")
    print()
    print("Rough guide: |r| > 0.5 = real, usable signal. |r| 0.2-0.5 = weak/mixed."
          " |r| < 0.2 = essentially no relationship.")

    # show the extremes -- worth a manual look regardless of the correlation number
    order = np.argsort(mean_scores)
    print("\nLowest-grounding classes (GOAL found the weakest support for these descriptions):")
    for i in order[:5]:
        c = common[i]
        print(f"  {grounding[c]['class_name']}: grounding={mean_scores[i]:.3f}, wca_acc={accuracies[i]:.1f}")
    print("\nHighest-grounding classes:")
    for i in order[-5:]:
        c = common[i]
        print(f"  {grounding[c]['class_name']}: grounding={mean_scores[i]:.3f}, wca_acc={accuracies[i]:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
