"""
ABS explainability demo -- roadmap step 5.

Visualizes, for a handful of example images: which regions ABS's DINO-attention
step selected, and which candidate description each region best matches. This
is a presentation artifact, not a new accuracy result -- the region-description
matching here is plain cosine similarity for illustration, NOT WCA's actual
weighted log-softmax scoring formula (that lives in own_functional.py/own_nn.py
and main.py's tightly-coupled "ours" branch). Don't quote these per-crop scores
as the reported 92.94%/61.17% numbers -- those come from the real pipeline.

Usage:
    python abs_explain.py --dataset_name oxford_pet --data_path /content/data/oxford_pets \
        --n_examples 6

Output: outputs/abs_explain/<dataset>_example_N.png (one figure per example)
"""
import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from backbones import load_backbone
from helper import load_dataset, load_classes
from goal_supervision import get_paths_and_labels
from clip import clip as openai_clip
from transformers import AutoImageProcessor, ViTModel


PATCH_SIZE = 16
CROP_SIZE = 14


def load_dino(device):
    ckpt_id = "facebook/dino-vitb16"
    processor_dino = AutoImageProcessor.from_pretrained(ckpt_id)
    model_dino = ViTModel.from_pretrained(ckpt_id, attn_implementation="eager").eval().to(device)
    return model_dino, processor_dino


@torch.no_grad()
def get_attention_regions(img: Image.Image, model_dino, processor_dino, device,
                           top_k=20, num_crops=4, crop_r1=0.15, crop_r2=0.4):
    """Same logic as main.py's custom_loader, but returns pixel bounding boxes
    (which the original discards) instead of only cropped image tensors."""
    W, H = img.size
    attn_inputs = processor_dino(img, return_tensors="pt").to(device)
    attentions = model_dino(**attn_inputs, output_attentions=True).attentions
    n_head = attentions[11].shape[1]
    attention_map = attentions[11][0, :, 0, 1:].reshape(n_head, -1).float()
    att_map = attention_map.mean(dim=0)

    topk_values, topk_indices = torch.topk(att_map.flatten(), top_k)
    topk_probs = torch.softmax(topk_values / 0.03, dim=0)
    sampled_indices = torch.multinomial(topk_probs, num_crops, replacement=True)
    sampled_patch_indices = topk_indices[sampled_indices]

    boxes = []
    crops = []
    for sampled_index in sampled_patch_indices:
        i, j = sampled_index // CROP_SIZE, sampled_index % CROP_SIZE
        patch_x_min = int(j * PATCH_SIZE * (W / (CROP_SIZE * PATCH_SIZE)))
        patch_y_min = int(i * PATCH_SIZE * (H / (CROP_SIZE * PATCH_SIZE)))
        patch_x_max = min(patch_x_min + int(PATCH_SIZE * (W / (CROP_SIZE * PATCH_SIZE))), W)
        patch_y_max = min(patch_y_min + int(PATCH_SIZE * (H / (CROP_SIZE * PATCH_SIZE))), H)
        center_x, center_y = (patch_x_min + patch_x_max) // 2, (patch_y_min + patch_y_max) // 2
        crop_w = random.randint(int(W * crop_r1), int(W * crop_r2))
        crop_h = random.randint(int(H * crop_r1), int(H * crop_r2))
        x_min = max(center_x - crop_w // 2, 0)
        y_min = max(center_y - crop_h // 2, 0)
        x_max = min(center_x + crop_w // 2, W)
        y_max = min(center_y + crop_h // 2, H)
        boxes.append((x_min, y_min, x_max, y_max))
        crops.append(img.crop((x_min, y_min, x_max, y_max)))
    return boxes, crops


@torch.no_grad()
def best_match_per_region(model_clip, preprocess, crops, descriptions, device):
    """Simplified illustration scoring: plain cosine similarity, not WCA's real formula."""
    text_tokens = openai_clip.tokenize(descriptions, truncate=True).to(device)
    text_feats = model_clip.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    crop_tensors = torch.stack([preprocess(c) for c in crops]).to(device)
    crop_feats = model_clip.encode_image(crop_tensors)
    crop_feats = crop_feats / crop_feats.norm(dim=-1, keepdim=True)

    sims = crop_feats @ text_feats.T  # [num_crops, num_desc]
    best_idx = sims.argmax(dim=1)
    best_score = sims.max(dim=1).values
    return [(descriptions[i], s.item()) for i, s in zip(best_idx, best_score)]


def draw_regions(img: Image.Image, boxes, matches, true_label, pred_label, out_path):
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

    correct = (true_label == pred_label)
    title_color = "green" if correct else "red"
    ax.set_title(f"True: {true_label}  |  Predicted (simplified): {pred_label}",
                 color=title_color, fontsize=13, fontweight="bold")
    ax.axis("off")

    caption_lines = [f"R{idx+1}: \"{desc[:70]}{'...' if len(desc) > 70 else ''}\" (sim={score:.2f})"
                      for idx, (desc, score) in enumerate(matches)]
    fig.text(0.02, -0.02, "\n".join(caption_lines), fontsize=9, family="monospace", va="top")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading vanilla CLIP (the backbone that actually works, per Phase 1/2) ...")
    model_clip, preprocess = load_backbone("openai", device=device)
    model_clip.eval()
    print("Loading DINO for attention-guided region selection ...")
    model_dino, processor_dino = load_dino(device)

    dataset = load_dataset(args.data_path, args.dataset_name,
                            lambda p: Image.open(p).convert("RGB"))
    classes = load_classes(args.dataset_name)
    with open(f"prompts/{args.dataset_name}/cupl.json") as f:
        all_descriptions = json.load(f)

    image_paths, labels = get_paths_and_labels(dataset)

    random.seed(args.seed)
    example_idxs = random.sample(range(len(image_paths)), min(args.n_examples, len(image_paths)))

    for n, idx in enumerate(example_idxs):
        path, true_label_idx = image_paths[idx], labels[idx]
        true_label = classes[true_label_idx]
        img = Image.open(path).convert("RGB")

        boxes, crops = get_attention_regions(img, model_dino, processor_dino, device)

        # simplified prediction: for each candidate class, score = mean best-crop
        # similarity to that class's descriptions; NOT the real WCA formula
        class_scores = {}
        for c_idx, c_name in enumerate(classes):
            if c_name not in all_descriptions:
                continue
            matches = best_match_per_region(model_clip, preprocess, crops,
                                             all_descriptions[c_name], device)
            class_scores[c_name] = np.mean([s for _, s in matches])
        pred_label = max(class_scores, key=class_scores.get)

        # for the figure, show each region's best match against the TRUE class's
        # descriptions (most illustrative -- shows what ABS/WCA is "looking at")
        matches = best_match_per_region(model_clip, preprocess, crops,
                                         all_descriptions.get(true_label, ["(no descriptions)"]),
                                         device)

        out_path = f"{args.out_dir}/{args.dataset_name}_example_{n+1}.png"
        draw_regions(img, boxes, matches, true_label, pred_label, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_examples", type=int, default=6)
    parser.add_argument("--out_dir", type=str, default="outputs/abs_explain")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args)
