"""
Run the ABS attention-cropping + simplified classification on ANY single image
-- not restricted to images already in the OxfordPets/CUB dataset folders.

The model still needs a fixed label set to predict among (that's inherent to
zero-shot classification against class descriptions, not a limitation of this
script) -- pick --dataset_name to match your image's domain: oxford_pet for
any cat/dog photo, cub for any bird photo, etc. Feeding a wildly out-of-domain
image (e.g. a car) will still produce a "best guess" among that class list,
same as any zero-shot classifier would.

Usage:
    python predict_image.py --image_path /content/my_photo.jpg --dataset_name oxford_pet

Output: outputs/predict/<image_name>_prediction.png -- the image with attention-
selected region boxes, the predicted class, and the top-3 candidate classes with
their scores. Same simplified cosine-similarity scoring as abs_explain.py -- NOT
the real WCA formula, for illustration only.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from backbones import load_backbone
from helper import load_classes
from abs_explain import load_dino, get_attention_regions, best_match_per_region


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    assert os.path.exists(args.image_path), f"Image not found: {args.image_path}"

    print("Loading vanilla CLIP ...")
    model_clip, preprocess = load_backbone("openai", device=device)
    model_clip.eval()
    print("Loading DINO for attention-guided region selection ...")
    model_dino, processor_dino = load_dino(device)

    classes = load_classes(args.dataset_name)
    with open(f"prompts/{args.dataset_name}/cupl.json") as f:
        all_descriptions = json.load(f)

    img = Image.open(args.image_path).convert("RGB")
    print(f"Running attention-guided region selection on {args.image_path} ...")
    boxes, crops = get_attention_regions(img, model_dino, processor_dino, device,
                                          num_crops=args.n_regions)

    print(f"Scoring against all {len(classes)} classes in '{args.dataset_name}' "
          f"(this can take a moment for datasets with many classes) ...")
    class_scores = {}
    for c_name in classes:
        if c_name not in all_descriptions:
            continue
        matches = best_match_per_region(model_clip, preprocess, crops,
                                         all_descriptions[c_name], device)
        class_scores[c_name] = float(np.mean([s for _, s in matches]))

    ranked = sorted(class_scores.items(), key=lambda x: -x[1])
    pred_label, pred_score = ranked[0]
    top3 = ranked[:3]

    # show each region's best match against the PREDICTED class's descriptions
    matches = best_match_per_region(model_clip, preprocess, crops,
                                     all_descriptions[pred_label], device)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img)
    colors = plt.cm.tab10(np.linspace(0, 1, len(boxes)))
    for idx, (box, (desc, score), color) in enumerate(zip(boxes, matches, colors)):
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              fill=False, edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x_min, max(y_min - 6, 0), f"R{idx+1}", color=color, fontsize=13,
                 fontweight="bold", bbox=dict(facecolor="black", alpha=0.5, pad=1))

    ax.set_title(f"Predicted: {pred_label}  (score={pred_score:.3f})",
                 fontsize=14, fontweight="bold")
    ax.axis("off")

    caption_lines = [f"R{idx+1}: \"{desc[:65]}{'...' if len(desc) > 65 else ''}\" (sim={score:.2f})"
                      for idx, (desc, score) in enumerate(matches)]
    top3_line = "Top-3: " + ", ".join(f"{name} ({score:.3f})" for name, score in top3)
    fig.text(0.02, -0.02, "\n".join(caption_lines) + f"\n\n{top3_line}",
              fontsize=9, family="monospace", va="top")

    img_name = os.path.splitext(os.path.basename(args.image_path))[0]
    out_path = f"{args.out_dir}/{img_name}_prediction.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)

    print(f"\nPredicted: {pred_label} (score={pred_score:.3f})")
    print(f"Top-3: {top3}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True,
                         help="Path to any image -- doesn't need to be in a dataset folder")
    parser.add_argument("--dataset_name", type=str, required=True,
                         help="Which class list to predict among, e.g. oxford_pet, cub")
    parser.add_argument("--n_regions", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="outputs/predict")
    args = parser.parse_args()
    main(args)
