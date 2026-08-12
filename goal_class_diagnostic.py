"""
GOAL as a diagnostic (not fusion, not curation) -- class-level version.

Question: do classes where GOAL finds strong, specific grounding for their
candidate descriptions tend to be the classes WCA actually classifies well,
and do weakly-grounded classes tend to be the ones WCA struggles with?

This reuses goal_supervision.py's crop-scoring machinery (same GOAL model,
same crop strategy) but reports the per-class grounding score instead of
using it to filter/curate descriptions. Correlate the output against
wca_per_class_acc.json (written by main.py's --enable_fusion pass) to test
the hypothesis.

Usage:
    python goal_class_diagnostic.py --dataset_name oxford_pet \
        --data_path /content/data/oxford_pets --goal_ckpt /path/to/goal_vitb16_docci.pt

Output: features/<dataset>/goal_per_class_grounding.json
        {"class_idx": {"class_name": ..., "top_score": ..., "mean_score": ...}, ...}

Then correlate:
    python correlate_goal_diagnostic.py --dataset_name oxford_pet
"""
import argparse
import json
import random

import torch

from backbones import load_backbone
from helper import load_dataset, load_classes
from goal_supervision import score_descriptions_for_class


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GOAL backbone from {args.goal_ckpt} ...")
    model, preprocess = load_backbone("goal", ckpt_path=args.goal_ckpt, device=device)
    model.eval()

    print(f"Loading {args.dataset_name} dataset ...")
    from PIL import Image
    plain_loader = lambda path: Image.open(path).convert("RGB")
    dataset = load_dataset(args.data_path, args.dataset_name, plain_loader)
    classes_wordified = load_classes(args.dataset_name)

    with open(f"prompts/{args.dataset_name}/{args.candidate_source}.json") as f:
        candidates = json.load(f)

    labels = dataset._labels
    by_class_idx = {}
    for i, lbl in enumerate(labels):
        by_class_idx.setdefault(lbl, []).append(i)
    image_paths_attr = dataset._images

    results = {}
    random.seed(args.seed)
    for class_idx, class_name in enumerate(classes_wordified):
        if class_name not in candidates:
            continue
        idxs = by_class_idx.get(class_idx, [])
        if not idxs:
            continue
        sample_idxs = random.sample(idxs, min(args.n_images_per_class, len(idxs)))
        paths = [image_paths_attr[i] for i in sample_idxs]

        desc_list = candidates[class_name]
        scores = score_descriptions_for_class(model, preprocess, paths, desc_list,
                                               args.n_crops, device)
        results[str(class_idx)] = {
            "class_name": class_name,
            "top_score": scores.max().item(),
            "mean_score": scores.mean().item(),
        }
        print(f"  {class_name}: top={results[str(class_idx)]['top_score']:.3f} "
              f"mean={results[str(class_idx)]['mean_score']:.3f}")

    out_path = f"features/{args.dataset_name}/goal_per_class_grounding.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--goal_ckpt", type=str, required=True)
    parser.add_argument("--candidate_source", type=str, default="cupl")
    parser.add_argument("--n_images_per_class", type=int, default=5)
    parser.add_argument("--n_crops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args)
