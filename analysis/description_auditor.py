"""
Description-quality auditor -- roadmap step 6.

Independent of GOAL: uses vanilla CLIP (the backbone that actually works, per
Phase 1/2) to grade each description SOURCE (cupl, clip-d, waffle) by how well
its candidate descriptions ground in real cropped image regions, per class and
overall. Answers a question the project only ever inferred indirectly from
downstream accuracy: which description-generation method actually produces the
most visually specific descriptions, not just the most accurate final scores.

Reuses goal_supervision.py's crop-scoring machinery (score_descriptions_for_class
is model-agnostic -- works with any CLIP-family model, GOAL or vanilla).

Usage:
    python description_auditor.py --dataset_name oxford_pet \
        --data_path /content/data/oxford_pets

Output: features/<dataset>/description_audit.json (per-class, per-source scores)
        + printed summary table
"""
import argparse
import json
import random

import numpy as np
import torch
from PIL import Image

from backbones import load_backbone
from helper import load_dataset, load_classes
from goal_supervision import get_paths_and_labels, score_descriptions_for_class


SOURCES = ["cupl", "clip-d", "waffle"]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading vanilla CLIP ...")
    model, preprocess = load_backbone("openai", device=device)
    model.eval()

    print(f"Loading {args.dataset_name} dataset ...")
    dataset = load_dataset(args.data_path, args.dataset_name,
                            lambda p: Image.open(p).convert("RGB"))
    classes = load_classes(args.dataset_name)
    image_paths, labels = get_paths_and_labels(dataset)
    by_class_idx = {}
    for i, lbl in enumerate(labels):
        by_class_idx.setdefault(lbl, []).append(i)

    source_data = {}
    for source in SOURCES:
        try:
            with open(f"prompts/{args.dataset_name}/{source}.json") as f:
                source_data[source] = json.load(f)
        except FileNotFoundError:
            print(f"  {source}.json not found for this dataset, skipping")

    results = {source: {} for source in source_data}
    random.seed(args.seed)

    for class_idx, class_name in enumerate(classes):
        idxs = by_class_idx.get(class_idx, [])
        if not idxs:
            continue
        sample_idxs = random.sample(idxs, min(args.n_images_per_class, len(idxs)))
        paths = [image_paths[i] for i in sample_idxs]

        per_source_line = [f"{class_name}:"]
        for source, descriptions in source_data.items():
            if class_name not in descriptions:
                continue
            desc_list = descriptions[class_name]
            scores = score_descriptions_for_class(model, preprocess, paths, desc_list,
                                                   args.n_crops, device)
            results[source][class_name] = {
                "top_score": scores.max().item(),
                "mean_score": scores.mean().item(),
                "n_descriptions": len(desc_list),
            }
            per_source_line.append(f"{source}(mean={scores.mean().item():.3f})")
        print("  " + " ".join(per_source_line))

    # summary across all classes, per source
    print("\n=== Summary: overall grounding quality by description source ===")
    summary = {}
    for source, per_class in results.items():
        if not per_class:
            continue
        means = [v["mean_score"] for v in per_class.values()]
        tops = [v["top_score"] for v in per_class.values()]
        summary[source] = {
            "avg_mean_grounding": float(np.mean(means)),
            "avg_top_grounding": float(np.mean(tops)),
            "n_classes": len(per_class),
        }
        print(f"  {source}: avg_mean_grounding={np.mean(means):.3f}  "
              f"avg_top_grounding={np.mean(tops):.3f}  ({len(per_class)} classes)")

    ranked = sorted(summary.items(), key=lambda x: -x[1]["avg_mean_grounding"])
    print(f"\nRanking (best-grounded first): {' > '.join(s for s, _ in ranked)}")

    out_path = f"features/{args.dataset_name}/description_audit.json"
    with open(out_path, "w") as f:
        json.dump({"per_class": results, "summary": summary}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_images_per_class", type=int, default=5)
    parser.add_argument("--n_crops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args)
