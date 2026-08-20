"""
goal_adapter_train.py — Train the GOAL-style frozen-CLIP adapter pair.

Trains two small residual MLP adapters (image-side, text-side) on top of
frozen CLIP, using a contrastive loss between ABS-selected image regions
and their class's descriptions. CLIP itself is never updated.

WHAT THIS SCRIPT ASSUMES (deliberately decoupled from the main repo for now,
since main.py/helper.py in this sandbox may be stale relative to what's
actually deployed — see integration note at the bottom):

    1. Per-image region features are already extracted, shape [R, D],
       L2-normalized, R = num_crops (or however many ABS regions you use).
       This is exactly what the existing crop_fea pipeline
       (clip/model.py forward_with_attention) already produces.
    2. Per-class description embeddings are already extracted, shape [K, D],
       L2-normalized, one row per description (e.g. from bifta-dr.json,
       encoded the same way bifta_dr.py's encode_descriptions() does).
    3. A labels array mapping each image to its class index.

You provide these three via the FeatureBundle below (see
`load_feature_bundle()` — replace its body with your actual precomputed
feature loading once wired into the real pipeline; a stub is provided so
this script is runnable/testable in isolation with synthetic data first).

USAGE:
    python goal_adapter_train.py --dataset_name oxford_pet \
        --shots 16 --epochs 30 --lr 1e-3 --batch_size 64 \
        --temperature 0.07 --patience 5
"""

import argparse
import copy
import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from adapter import GoalAdapterPair
from backbones import load_backbone
from clip import clip as openai_clip
from helper import load_precomputed_features, load_classes, set_seed
from transformers import AutoImageProcessor, ViTModel


# --------------------------------------------------------------------------
# Feature bundle
# --------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    region_features: torch.Tensor      # [N_images, R, D], L2-normalized
    region_labels: torch.Tensor        # [N_images], class index per image
    description_features: torch.Tensor  # [num_classes, K, D], L2-normalized
    class_names: list                  # length num_classes


def _encode_class_descriptions(model, class_names, descriptions_path, device, fixed_k=30):
    """
    Encodes prompts/{dataset}/bifta-dr.json (or cupl.json) into a fixed-size
    [num_classes, fixed_k, D] tensor, L2-normalized. Classes with fewer than
    fixed_k descriptions (a few DR classes can land just under top_k after
    dedup, e.g. Egyptian Mau: 25) are padded by cycling their own
    descriptions -- never zero-padded, since a zero vector would corrupt
    the mean-pooled class prototype used in the contrastive loss.
    """
    with open(descriptions_path) as f:
        descriptions = json.load(f)

    all_embeds = []
    with torch.no_grad():
        for name in class_names:
            texts = descriptions[name]
            if len(texts) < fixed_k:
                # cycle to pad, e.g. [a,b,c] -> [a,b,c,a,b,...] up to fixed_k
                reps = (fixed_k // len(texts)) + 1
                texts = (texts * reps)[:fixed_k]
            else:
                texts = texts[:fixed_k]

            tokens = openai_clip.tokenize(texts, truncate=True).to(device)
            embeds = model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds)

    return torch.stack(all_embeds, dim=0)  # [num_classes, fixed_k, D]


def load_feature_bundle(
    dataset_name: str,
    split: str,
    backbone: str = "openai",
    descriptions_method: str = "bifta-dr",
    fixed_k: int = 30,
    seed: int = 1,
) -> FeatureBundle:
    """
    Real implementation, wired against helper.py's load_precomputed_features().

    - region_features: crop_fea cache (patches, weight column stripped),
      loaded via the same path main.py uses -- split-aware, so split="train"
      hits a SEPARATE cache file from split="test" (see
      apply_train_split_patch.py; never silently reuses/collides).
    - description_features: encodes prompts/{dataset}/{descriptions_method}.json
      per class with the same CLIP text encoder, padded/truncated to fixed_k
      so every class has an identical description count for stacking.
    - region_labels / class_names: from the same dataset loader + class json
      the rest of the pipeline uses.

    NOTE: requires helper.py to already have the train-split patch applied
    (apply_train_split_patch.py) if split="train" is requested.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    with open(f"cfgs/{dataset_name}.yaml") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    model_size = hparams["model_size"]
    alpha = hparams["alpha"]
    n_samples = hparams["n_samples"]
    batch_size = hparams["batch_size"]
    data_path = hparams["data_path"]

    print(f"Loading {model_size} (backbone={backbone})")
    model, processor = load_backbone(backbone, openai_name=model_size, ckpt_path=None, device=device)
    model.eval()
    model.requires_grad_(False)

    # --- args shim: load_precomputed_features()'s custom_loader (built in
    # main.py) needs args.num_crops/top_k/clip_crop_r1/clip_crop_r2, plus
    # layer1/layer2 for the DINO attention crop. Match main.py's CLI
    # defaults exactly so the crop cache this produces is directly
    # comparable to (and cache-key-compatible with) what main.py builds. ---
    class _ArgsShim:
        num_crops = 50
        top_k = 20
        clip_crop_r1 = 0.6
        clip_crop_r2 = 0.9
        layer1 = 11
        layer2 = 11
    args_shim = _ArgsShim()

    patch_size = 16
    crop_size = 14

    def load_ckpt(ckpt_id="facebook/dino-vitb16"):
        image_processor = AutoImageProcessor.from_pretrained(ckpt_id)
        m = ViTModel.from_pretrained(ckpt_id, attn_implementation="eager").eval()
        return m, image_processor

    model_dino, processor_dino = load_ckpt("facebook/dino-vitb16")
    model_dino.eval()
    model_dino.requires_grad_(False)
    model_dino.to(device)

    def custom_loader(path: str) -> torch.Tensor:
        # Identical to main.py's inline custom_loader -- kept in sync
        # manually since it's a closure in main.py, not an importable
        # function. If you change main.py's crop augmentation, mirror the
        # change here too.
        img = datasets.folder.default_loader(path)
        W, H = img.size
        augmented_imgs = [processor_dino(img, return_tensors="pt")["pixel_values"].squeeze(0)]
        augmented_imgs.extend(processor(img) for _ in range(1))

        attention_imgs_dino = processor_dino(img, return_tensors="pt")
        with torch.no_grad():
            image_attention_mh = model_dino(**attention_imgs_dino, output_attentions=True)
        image_attention_mh = image_attention_mh.attentions
        n_head = image_attention_mh[11].shape[1]
        attention_map = image_attention_mh[11][0, :, 0, 1:].reshape(n_head, -1).float()
        att_map = attention_map.mean(dim=0)

        att_map_flat = att_map.flatten()
        topk_values, topk_indices = torch.topk(att_map_flat, args_shim.top_k)
        topk_probs = torch.softmax(topk_values / 0.03, dim=0)
        sampled_indices = torch.multinomial(topk_probs, args_shim.num_crops, replacement=True)
        sampled_patch_indices = topk_indices[sampled_indices]

        for sampled_index in sampled_patch_indices:
            i, j = sampled_index // crop_size, sampled_index % crop_size
            patch_x_min = int(j * patch_size * (W / (crop_size * patch_size)))
            patch_y_min = int(i * patch_size * (H / (crop_size * patch_size)))
            patch_x_max = min(patch_x_min + int(patch_size * (W / (crop_size * patch_size))), W)
            patch_y_max = min(patch_y_min + int(patch_size * (H / (crop_size * patch_size))), H)
            center_x = (patch_x_min + patch_x_max) // 2
            center_y = (patch_y_min + patch_y_max) // 2
            crop_width = random.randint(int(W * args_shim.clip_crop_r1), int(W * args_shim.clip_crop_r2))
            crop_height = random.randint(int(H * args_shim.clip_crop_r1), int(H * args_shim.clip_crop_r2))
            x_min = max(center_x - crop_width // 2, 0)
            y_min = max(center_y - crop_height // 2, 0)
            x_max = min(center_x + crop_width // 2, W)
            y_max = min(center_y + crop_height // 2, H)
            cropped_image = img.crop((x_min, y_min, x_max, y_max))
            augmented_imgs.extend(processor(cropped_image) for _ in range(1))

        return torch.stack(augmented_imgs)

    print(f"Loading precomputed features (split={split})...")
    precomputed_features, target, _image_features = load_precomputed_features(
        model, args_shim, processor,
        dataset_name=dataset_name, model_size=model_size, alpha=alpha,
        n_samples=n_samples, batch_size=batch_size, num_workers=2,
        data_path=data_path, custom_loader=custom_loader, device=device,
        layer1=args_shim.layer1, layer2=args_shim.layer2, split=split,
    )

    # precomputed_features: [N, 2*num_crops, D+1] -- strip the trailing
    # attention-weight column (last dim), which isn't part of the feature
    # itself, just a per-crop scalar weight used elsewhere in the pipeline.
    region_features = precomputed_features[:, :, :-1].cpu()

    class_names = load_classes(dataset_name)
    descriptions_path = f"prompts/{dataset_name}/{descriptions_method}.json"
    description_features = _encode_class_descriptions(
        model, class_names, descriptions_path, device, fixed_k=fixed_k
    ).cpu()

    return FeatureBundle(
        region_features=region_features,
        region_labels=target.cpu(),
        description_features=description_features,
        class_names=class_names,
    )


# --------------------------------------------------------------------------
# Few-shot sampling
# --------------------------------------------------------------------------

def make_few_shot_split(bundle: FeatureBundle, shots: int, val_shots: int, seed: int = 0):
    """
    Deterministically samples `shots` images per class for training and a
    further `val_shots` per class (disjoint from train) for early-stopping
    validation. Everything else in `bundle` is left untouched, on the
    assumption that the caller passes only the pipeline's existing
    train-portion split here — the final held-out test split used for the
    zero-shot baseline (92.94/61.17 etc.) must stay completely separate and
    is never touched by this function.
    """
    rng = random.Random(seed)
    num_classes = bundle.description_features.shape[0]

    train_idx, val_idx = [], []
    for c in range(num_classes):
        class_idx = (bundle.region_labels == c).nonzero(as_tuple=True)[0].tolist()
        rng.shuffle(class_idx)
        if len(class_idx) < shots + val_shots:
            raise ValueError(
                f"Class {c} ({bundle.class_names[c]}) has only {len(class_idx)} "
                f"images, need at least {shots + val_shots} (shots + val_shots)."
            )
        train_idx.extend(class_idx[:shots])
        val_idx.extend(class_idx[shots:shots + val_shots])

    return torch.tensor(train_idx), torch.tensor(val_idx)


class RegionDescriptionDataset(Dataset):
    """Yields (region_features [R, D], class_label) for a subset of indices."""

    def __init__(self, bundle: FeatureBundle, indices: torch.Tensor):
        self.region_features = bundle.region_features[indices]
        self.labels = bundle.region_labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.region_features[i], self.labels[i]


# --------------------------------------------------------------------------
# Contrastive loss: ABS regions <-> class descriptions
# --------------------------------------------------------------------------

def region_description_contrastive_loss(
    adapted_regions: torch.Tensor,      # [B, R, D]
    labels: torch.Tensor,               # [B]
    adapted_descriptions: torch.Tensor,  # [num_classes, K, D]
    temperature: float,
):
    """
    For each region of each image in the batch: pull it toward its class's
    description embeddings, push it away from other classes' descriptions
    present in this batch (InfoNCE, region-to-class-description-set).

    Per-class descriptions are mean-pooled to a single prototype per class
    before the similarity computation — keeps this a standard InfoNCE over
    (region, class) pairs rather than needing a region-to-every-individual-
    description assignment, which would require picking a target
    description per region (not available/well-defined here).
    """
    B, R, D = adapted_regions.shape
    device = adapted_regions.device

    # Mean-pool descriptions per class -> [num_classes, D], re-normalize
    class_prototypes = adapted_descriptions.mean(dim=1)
    class_prototypes = class_prototypes / class_prototypes.norm(dim=-1, keepdim=True)

    # Flatten regions: [B*R, D]
    flat_regions = adapted_regions.reshape(B * R, D)
    flat_labels = labels.unsqueeze(1).expand(B, R).reshape(B * R).to(device)

    # Similarity to every class prototype present in this batch's label set
    # (using the full class_prototypes table, not just batch classes, since
    # num_classes is typically small enough that this is cheap and gives a
    # stronger negative signal than batch-only negatives).
    logits = (flat_regions @ class_prototypes.T) / temperature  # [B*R, num_classes]

    loss = F.cross_entropy(logits, flat_labels)
    return loss


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def evaluate_val_loss(adapters, val_loader, description_features, temperature, device):
    adapters.eval()
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        adapted_desc = adapters.forward_text(
            description_features.view(-1, description_features.shape[-1])
        ).view(description_features.shape)
        for region_feats, labels in val_loader:
            region_feats = region_feats.to(device)
            labels = labels.to(device)
            B, R, D = region_feats.shape
            adapted_regions = adapters.forward_image(region_feats.reshape(B * R, D)).reshape(B, R, D)
            loss = region_description_contrastive_loss(adapted_regions, labels, adapted_desc, temperature)
            total_loss += loss.item() * B
            total_count += B
    return total_loss / max(1, total_count)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = load_feature_bundle(
        args.dataset_name, split="train", backbone=args.backbone,
        descriptions_method=args.descriptions_method, fixed_k=args.fixed_k, seed=args.seed,
    )
    train_idx, val_idx = make_few_shot_split(bundle, args.shots, args.val_shots, seed=args.seed)

    train_ds = RegionDescriptionDataset(bundle, train_idx)
    val_ds = RegionDescriptionDataset(bundle, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    dim = bundle.region_features.shape[-1]
    adapters = GoalAdapterPair(dim=dim, hidden_ratio=args.hidden_ratio,
                                ratio=args.residual_ratio, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    description_features = bundle.description_features.to(device)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        adapters.train()
        running_loss, running_count = 0.0, 0
        for region_feats, labels in train_loader:
            region_feats = region_feats.to(device)
            labels = labels.to(device)
            B, R, D = region_feats.shape

            adapted_regions = adapters.forward_image(region_feats.reshape(B * R, D)).reshape(B, R, D)
            adapted_desc = adapters.forward_text(
                description_features.view(-1, description_features.shape[-1])
            ).view(description_features.shape)

            loss = region_description_contrastive_loss(adapted_regions, labels, adapted_desc, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B
            running_count += B

        train_loss = running_loss / max(1, running_count)
        val_loss = evaluate_val_loss(adapters, val_loader, description_features, args.temperature, device)

        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(adapters.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (no val improvement for {args.patience} epochs)")
                break

    if best_state is not None:
        adapters.load_state_dict(best_state)

    save_path = f"adapters_{args.dataset_name}_{args.shots}shot.pt"
    adapters.save(save_path)
    print(f"Saved best adapter pair (val_loss={best_val_loss:.4f}) to {save_path}")
    return adapters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="openai", choices=["openai", "goal"])
    parser.add_argument("--descriptions_method", type=str, default="bifta-dr",
                         help="Which prompts/{dataset}/{method}.json to use as class descriptions (e.g. bifta-dr, cupl)")
    parser.add_argument("--fixed_k", type=int, default=30, help="Descriptions per class after pad/truncate, for stacking")
    parser.add_argument("--shots", type=int, default=16, help="Training shots per class")
    parser.add_argument("--val_shots", type=int, default=4, help="Validation shots per class (disjoint from train)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hidden_ratio", type=float, default=0.25, help="Adapter hidden dim as a ratio of feature dim")
    parser.add_argument("--residual_ratio", type=float, default=0.2, help="Blend ratio: higher = trust the adapter more, lower = stay closer to frozen CLIP")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum val_loss improvement to reset patience")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(args)
