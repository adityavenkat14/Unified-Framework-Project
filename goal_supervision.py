"""
GOAL-as-supervision: use GOAL (never at inference, only here, offline) to grade
candidate LLM-generated descriptions by how strongly they ground in real image
regions of their class, and keep only the top-K per class.

Motivation: GOAL's local loss trained it specifically to align image REGIONS
with SENTENCES. That's a good fit for "does this candidate description
actually describe something visible in a real photo of this class", and a bad
fit for "encode this short label for zero-shot classification" (which Phase 2
showed GOAL is worse at than vanilla CLIP). This script uses GOAL for the
former only, and writes a plain description json that the normal vanilla-CLIP
pipeline (main.py --backbone openai) then consumes like any other method.

Usage:
    python goal_supervision.py --dataset_name oxford_pet \
        --goal_ckpt /path/to/GOAL_ViT_base16_DOCCI.pth \
        --candidate_source cupl --top_k 15

Output: prompts/<dataset_name>/goal-curated.json, same schema as cupl.json
        ({"class name": ["kept description", ...]}).
"""
import argparse
import json
import random

import torch
from PIL import Image
from tqdm import tqdm

from backbones import load_backbone
from helper import load_dataset, load_classes, wordify


def get_paths_and_labels(dataset):
    """Different dataset classes in this repo expose image paths/labels under different
    attribute names -- OxfordPets' custom class uses _images/_labels, CUB (built on
    torchvision's ImageFolder) uses samples/targets. Normalize both to plain lists."""
    if hasattr(dataset, "_images") and hasattr(dataset, "_labels"):
        return list(dataset._images), list(dataset._labels)
    if hasattr(dataset, "samples") and hasattr(dataset, "targets"):
        paths = [p for p, _ in dataset.samples]
        return paths, list(dataset.targets)
    raise AttributeError(
        f"Don't know how to get image paths/labels from {type(dataset).__name__} -- "
        f"add a case to get_paths_and_labels() for this dataset's attribute names."
    )


def make_crops(image: Image.Image, n_crops: int):
    """Full image + a few overlapping sub-crops -- this is what "region" means
    to GOAL's local loss (region = crop of the image, not a patch token)."""
    w, h = image.size
    crops = [image]  # whole image counts as one "region" too
    if n_crops <= 1:
        return crops
    # center crop + 4 corner-ish crops at ~70% scale, simple and robust
    scale = 0.7
    cw, ch = int(w * scale), int(h * scale)
    boxes = [
        (0, 0, cw, ch),
        (w - cw, 0, w, ch),
        (0, h - ch, cw, h),
        (w - cw, h - ch, w, h),
        ((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch),
    ]
    for box in boxes[: max(0, n_crops - 1)]:
        crops.append(image.crop(box))
    return crops


@torch.no_grad()
def score_descriptions_for_class(model, preprocess, image_paths, descriptions,
                                  n_crops, device):
    """score[d] = mean over sampled images of max over crops of cosine sim
    between crop embedding and description d's embedding."""
    import clip.clip as openai_clip  # local tokenizer, same as the rest of the pipeline

    text_tokens = openai_clip.tokenize(descriptions, truncate=True).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)  # [num_desc, D]

    per_image_scores = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  skip unreadable image {path}: {e}")
            continue
        crops = make_crops(img, n_crops)
        crop_tensors = torch.stack([preprocess(c) for c in crops]).to(device)
        crop_feats = model.encode_image(crop_tensors)
        crop_feats = crop_feats / crop_feats.norm(dim=-1, keepdim=True)  # [num_crops, D]

        sims = crop_feats @ text_feats.T  # [num_crops, num_desc]
        max_over_crops = sims.max(dim=0).values  # [num_desc] -- best-matching region per description
        per_image_scores.append(max_over_crops.cpu())

    if not per_image_scores:
        return torch.zeros(len(descriptions))
    return torch.stack(per_image_scores).mean(dim=0)  # [num_desc]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GOAL backbone from {args.goal_ckpt} ...")
    model, preprocess = load_backbone("goal", ckpt_path=args.goal_ckpt, device=device)
    model.eval()

    print(f"Loading {args.dataset_name} dataset (for per-class image sampling) ...")
    plain_loader = lambda path: Image.open(path).convert("RGB")
    dataset = load_dataset(args.data_path, args.dataset_name, plain_loader)
    classes_wordified = load_classes(args.dataset_name)  # display names, e.g. "wheaten terrier"

    with open(f"prompts/{args.dataset_name}/{args.candidate_source}.json") as f:
        candidates = json.load(f)  # {"class name": [sentences...]}

    # build class_name -> list of image indices, using the dataset's own label ints
    image_paths_attr, labels = get_paths_and_labels(dataset)
    by_class_idx = {}
    for i, lbl in enumerate(labels):
        by_class_idx.setdefault(lbl, []).append(i)

    curated = {}
    random.seed(args.seed)
    for class_idx, class_name in enumerate(classes_wordified):
        if class_name not in candidates:
            print(f"  no candidates for '{class_name}', skipping")
            continue
        idxs = by_class_idx.get(class_idx, [])
        if not idxs:
            print(f"  no images found for class '{class_name}' (idx {class_idx}), keeping all candidates unranked")
            curated[class_name] = candidates[class_name][: args.top_k]
            continue
        sample_idxs = random.sample(idxs, min(args.n_images_per_class, len(idxs)))
        paths = [image_paths_attr[i] for i in sample_idxs]

        desc_list = candidates[class_name]
        scores = score_descriptions_for_class(model, preprocess, paths, desc_list,
                                               args.n_crops, device)
        ranked = sorted(zip(desc_list, scores.tolist()), key=lambda x: -x[1])
        keep = [d for d, s in ranked[: args.top_k]]
        curated[class_name] = keep
        print(f"  {class_name}: kept {len(keep)}/{len(desc_list)} "
              f"(top score {ranked[0][1]:.3f}, cutoff score {ranked[len(keep)-1][1]:.3f})")

    out_path = f"prompts/{args.dataset_name}/goal-curated.json"
    with open(out_path, "w") as f:
        json.dump(curated, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                         help="same data_path you'd put in cfgs/<dataset>.yaml")
    parser.add_argument("--goal_ckpt", type=str, required=True)
    parser.add_argument("--candidate_source", type=str, default="cupl",
                         help="which existing prompts/<dataset>/<this>.json to grade")
    parser.add_argument("--top_k", type=int, default=15,
                         help="how many descriptions to keep per class")
    parser.add_argument("--n_images_per_class", type=int, default=5)
    parser.add_argument("--n_crops", type=int, default=4,
                         help="regions per image (whole image always included on top of this)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args)