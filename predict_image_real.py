"""
The REAL unified pipeline on a single arbitrary image -- not the simplified
cosine-similarity demo in abs_explain.py/predict_image.py.

This reuses the actual, tested code paths rather than re-deriving formulas by
hand (safer -- avoids silently introducing a subtly-wrong "faithful copy"):
  - custom_loader's exact cropping logic (copied from main.py, parameterized)
  - helper.py's exact precompute inner loop, including model.encode_image_attention
    (the DINO-attention-guided internal pooling this pipeline actually uses --
    NOT just literal crops, which is what the earlier demo scripts approximated)
  - main.py's exact WCA weighted log-softmax formula
  - fusion/lazsl_score.py's compute_lazsl_scores (already generic/reusable)

Output: predicted class (real WCA score), top-3 candidates, and a LaZSL
agreement flag (high/low confidence) -- exactly the architecture validated in
EXPERIMENT_LOG.md, not an approximation of it.

Usage:
    python predict_image_real.py --image_path /content/my_photo.jpg --dataset_name oxford_pet
"""
import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import datasets
import matplotlib.pyplot as plt

from backbones import load_backbone
from helper import generate_weights, load_classes, load_ckpt
from fusion.lazsl_score import build_op_d, compute_lazsl_scores

PATCH_SIZE = 16
CROP_SIZE = 14


def run_custom_loader(path, model_dino, processor_dino, processor, args):
    """Exact copy of main.py's custom_loader body, parameterized instead of a closure.
    Produces the same stacked tensor: [dino-view, whole-image, crop_1, ..., crop_N]."""
    img = datasets.folder.default_loader(path)
    W, H = img.size
    augmented_imgs = [processor_dino(img, return_tensors="pt")["pixel_values"].squeeze(0)]
    augmented_imgs.extend(processor(img) for _ in range(1))
    attention_imgs_dino = processor_dino(img, return_tensors="pt")
    with torch.no_grad():
        image_attention_mh = model_dino(**attention_imgs_dino, output_attentions=True)
        image_attention_mh = image_attention_mh.attentions
    n_head = image_attention_mh[args.layer1].shape[1]
    attention_map = image_attention_mh[args.layer1][0, :, 0, 1:].reshape(n_head, -1).float()
    att_map = attention_map.mean(dim=0)

    top_k = args.top_k
    att_map_flat = att_map.flatten()
    topk_values, topk_indices = torch.topk(att_map_flat, top_k)
    topk_probs = torch.softmax(topk_values / 0.03, dim=0)

    num_samples = args.num_crops
    sampled_indices = torch.multinomial(topk_probs, num_samples, replacement=True)
    sampled_patch_indices = topk_indices[sampled_indices]

    boxes = []
    for sampled_index in sampled_patch_indices:
        i, j = sampled_index // CROP_SIZE, sampled_index % CROP_SIZE
        patch_x_min = int(j * PATCH_SIZE * (W / (CROP_SIZE * PATCH_SIZE)))
        patch_y_min = int(i * PATCH_SIZE * (H / (CROP_SIZE * PATCH_SIZE)))
        patch_x_max = min(patch_x_min + int(PATCH_SIZE * (W / (CROP_SIZE * PATCH_SIZE))), W)
        patch_y_max = min(patch_y_min + int(PATCH_SIZE * (H / (CROP_SIZE * PATCH_SIZE))), H)
        center_x, center_y = (patch_x_min + patch_x_max) // 2, (patch_y_min + patch_y_max) // 2
        crop_width = np.random.randint(int(W * args.clip_crop_r1), int(W * args.clip_crop_r2) + 1)
        crop_height = np.random.randint(int(H * args.clip_crop_r1), int(H * args.clip_crop_r2) + 1)
        x_min = max(center_x - crop_width // 2, 0)
        y_min = max(center_y - crop_height // 2, 0)
        x_max = min(center_x + crop_width // 2, W)
        y_max = min(center_y + crop_height // 2, H)
        cropped_image = img.crop((x_min, y_min, x_max, y_max))
        boxes.append((x_min, y_min, x_max, y_max))
        augmented_imgs.extend(processor(cropped_image) for _ in range(1))

    return torch.stack(augmented_imgs), boxes


@torch.no_grad()
def precompute_single_image(stacked_imgs, model, model_dino, args, device, layer1, layer2):
    """Exact copy of helper.py's precompute inner-loop body, applied to a batch of 1."""
    images = stacked_imgs.unsqueeze(0).to(device)  # [1, N, 3, 224, 224]
    b, ns = images.shape[:2]
    images_dino = images[:, 0, :, :, :]
    images_crop = images[:, 2:, :, :, :]
    images_whole = images[:, 1, :, :, :]
    images_crop = images_crop.flatten(0, 1)
    image_attention_mh = model_dino(pixel_values=images_dino, output_attentions=True)
    image_attention_mh = image_attention_mh.attentions
    n_head = image_attention_mh[layer1].shape[1]
    attention_map = image_attention_mh[layer1][:, :, 0, 1:].reshape(b, n_head, -1).float()
    image_features = model.encode_image_attention(images_whole, args, target_layer=[layer2],
                                                    attn_weights=attention_map)
    image_features = torch.nn.functional.normalize(image_features)
    image_features = image_features.view(b, args.num_crops + 1, -1)
    image_features_crop = model.encode_image(images_crop)
    image_features_crop = torch.nn.functional.normalize(image_features_crop)
    image_features_crop = image_features_crop.view(b, args.num_crops, -1)
    patch_features = torch.cat((image_features[:, 1:], image_features_crop), dim=1)
    image_features_global = image_features[:, :1]
    weight_image = (image_features_global * patch_features).sum(dim=-1, keepdim=True)
    patch_with_weights = torch.cat([patch_features, weight_image], -1)
    return patch_with_weights  # [1, 2*num_crops, D+1]


def wca_logits(patch_embeds, patch_weights, zeroshot_weights):
    """Exact copy of main.py's 'ours' WCA formula block, applied to a batch of 1."""
    total_size, crop_num, embed_dim = patch_embeds.shape
    num_classes, num_descriptions, embed_dim = zeroshot_weights.shape
    patch_embeds_flat = patch_embeds.reshape(-1, embed_dim)
    zeroshot_weights_flat = zeroshot_weights.reshape(-1, embed_dim)
    similarity_matrix_flat = torch.matmul(patch_embeds_flat, zeroshot_weights_flat.t())
    similarity_matrix_flat = similarity_matrix_flat / 0.03
    similarity_matrix = similarity_matrix_flat.reshape(total_size, crop_num, num_classes, num_descriptions)
    similarity_matrix = similarity_matrix.view(total_size, crop_num, -1)
    log_softmax_matrix = similarity_matrix.log_softmax(dim=-1)
    similarity_matrix_soft = log_softmax_matrix.exp()
    weighted_similarity_matrix = similarity_matrix_soft * similarity_matrix
    weighted_similarity_matrix = weighted_similarity_matrix.reshape(total_size, crop_num, num_classes, num_descriptions)
    logits_batch_crop_class = weighted_similarity_matrix.sum(dim=-1)
    logits = logits_batch_crop_class.sum(dim=1)
    return logits  # [1, num_classes]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert os.path.exists(args.image_path), f"Image not found: {args.image_path}"
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading vanilla CLIP (the validated backbone) ...")
    model, processor = load_backbone("openai", device=device)
    model.eval()
    model.requires_grad_(False)
    print("Loading DINO ...")
    model_dino, processor_dino = load_ckpt("facebook/dino-vitb16")
    model_dino.eval()
    model_dino.requires_grad_(False)
    model_dino.to(device)

    print(f"Running the real ABS pipeline (custom_loader) on {args.image_path} ...")
    stacked_imgs, boxes = run_custom_loader(args.image_path, model_dino, processor_dino, processor, args)

    print("Running the real precompute step (encode_image_attention + crop encoding) ...")
    patch_with_weights = precompute_single_image(stacked_imgs, model, model_dino, args, device,
                                                   args.layer1, args.layer2)
    patch_embeds = patch_with_weights[:, :, :-1]
    patch_weights = patch_with_weights[:, :, -1]

    print(f"Building real zeroshot weights for '{args.dataset_name}' (cupl descriptions, 'ours' scaling) ...")
    text_scale = torch.exp(torch.tensor(2.0)).to(device)  # matches cfgs/<dataset>.yaml's ours.text_scale
    zeroshot_weights = generate_weights("ours", model=model, dataset_name=args.dataset_name,
                                         tt_scale=text_scale, device=device)
    zeroshot_weights = zeroshot_weights.to(patch_embeds.dtype).permute(1, 0, 2)  # -> [num_classes, num_desc, D]

    print("Computing real WCA logits ...")
    logits = wca_logits(patch_embeds, patch_weights, zeroshot_weights)
    classes = load_classes(args.dataset_name)
    # main.py's real accuracy() computation uses argmax directly on raw logits --
    # it never applies softmax. Raw logits are a SUM over ~100 regions, so their
    # magnitude is large enough that a naive softmax saturates to exactly 1.0/0.0
    # in float32 once one class pulls ahead -- misleading display, not a real
    # problem with the computation. Rescale (divide by region count) before
    # softmax purely so relative confidence is actually visible in the printout.
    num_regions = patch_embeds.shape[1]
    display_probs = (logits[0] / num_regions).softmax(dim=-1)
    wca_ranked_raw = sorted(zip(classes, logits[0].tolist()), key=lambda x: -x[1])
    wca_ranked = sorted(zip(classes, display_probs.tolist()), key=lambda x: -x[1])
    wca_pred = wca_ranked_raw[0][0]  # argmax is identical either way -- rescaling is display-only

    print("Computing real LaZSL score for the agreement check ...")
    op_d = build_op_d()
    lazsl_scores = compute_lazsl_scores(patch_embeds, zeroshot_weights, op_d, batch_chunk=1)
    lazsl_pred = classes[lazsl_scores[0].argmax().item()]
    agree = (wca_pred == lazsl_pred)

    print(f"\n{'='*60}")
    print(f"WCA prediction (main engine):  {wca_pred}  (score={wca_ranked[0][1]:.4f})")
    print(f"LaZSL prediction (consistency check): {lazsl_pred}")
    print(f"Agreement: {'YES -- high confidence' if agree else 'NO -- LOW CONFIDENCE, flag this prediction'}")
    print(f"{'='*60}")
    print("Top-5 WCA candidates:")
    for name, score in wca_ranked[:5]:
        print(f"  {name}: {score:.4f}")

    # visualize
    img = Image.open(args.image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(boxes), 10)))
    for idx, box in enumerate(boxes[:10]):  # only draw first 10 for readability (50 crops is too many to show)
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              fill=False, edgecolor=colors[idx % 10], linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
    conf_color = "green" if agree else "red"
    conf_text = "HIGH CONFIDENCE (WCA/LaZSL agree)" if agree else "LOW CONFIDENCE (WCA/LaZSL disagree)"
    ax.set_title(f"Predicted: {wca_pred}\n{conf_text}", color=conf_color, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.text(0.02, -0.05,
              f"Note: showing first 10 of {len(boxes)} attention-selected regions actually used in scoring.\n"
              f"Top-5: " + ", ".join(f"{n} ({s:.3f})" for n, s in wca_ranked[:5]),
              fontsize=9, family="monospace", va="top")

    img_name = os.path.splitext(os.path.basename(args.image_path))[0]
    out_path = f"{args.out_dir}/{img_name}_real_prediction.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/predict_real")
    parser.add_argument("--num_crops", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--layer1", type=int, default=11)
    parser.add_argument("--layer2", type=int, default=11)
    parser.add_argument("--clip_crop_r1", type=float, default=0.6)
    parser.add_argument("--clip_crop_r2", type=float, default=0.9)
    args = parser.parse_args()
    main(args)
