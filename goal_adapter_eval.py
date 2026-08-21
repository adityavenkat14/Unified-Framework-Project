"""
goal_adapter_eval.py — Evaluate the trained GOAL adapter pair against the
zero-shot 'ours' baseline, on the same held-out test split main.py already
reports (92.94 for oxford_pet, 61.17 for cub).

Faithfully replicates main.py's "ours" scoring block for method_name=="ours"
(the WCA cross-alignment formula), just swapping in adapted region/description
embeddings in place of raw frozen-CLIP ones. This intentionally reproduces
main.py's existing behavior exactly, including the fact that `image_scale`/
`w_i` are computed but never actually applied to logits_batch, and the
n_run loop's `random_indices` is computed but unused (both flagged in
main.py directly, and confirmed in this session's investigation of that
code) -- so this is a genuine apples-to-apples comparison against the real
reported baseline, not a "fixed" version of it.

USAGE:
    python goal_adapter_eval.py --dataset_name oxford_pet \
        --adapter_path adapters_oxford_pet_16shot.pt \
        --descriptions_method bifta-dr --fixed_k 30
"""

import argparse

import numpy as np
import torch
import yaml

from adapter import GoalAdapterPair
from backbones import load_backbone
from helper import accuracy, load_precomputed_features, load_classes, set_seed
from goal_adapter_train import _encode_class_descriptions  # reuse the exact same encoding logic used in training


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    with open(f"cfgs/{args.dataset_name}.yaml") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    model_size = hparams["model_size"]
    alpha = hparams["alpha"]
    n_samples = hparams["n_samples"]
    batch_size = hparams["batch_size"]
    data_path = hparams["data_path"]
    patch_num = hparams["patch_n"]
    n_run = hparams["n_run"]

    print(f"Loading {model_size} (backbone={args.backbone})")
    model, processor = load_backbone(args.backbone, openai_name=model_size, ckpt_path=None, device=device)
    model.eval()
    model.requires_grad_(False)

    class _ArgsShim:
        num_crops = 50
        top_k = 20
        clip_crop_r1 = 0.6
        clip_crop_r2 = 0.9
        layer1 = 11
        layer2 = 11
        patch_size = 14
    args_shim = _ArgsShim()

    # Test-split loading: identical custom_loader to main.py's (no split
    # kwarg needed here -- load_precomputed_features defaults to split="test",
    # so this loads/reuses the EXACT same crop cache main.py's 92.94 came from.)
    print("Loading precomputed TEST-split features (same cache main.py's baseline uses)...")

    def custom_loader(path: str) -> torch.Tensor:
        # Only needed if the test cache doesn't already exist -- in the
        # normal flow this cache is already built from earlier main.py runs,
        # so this function should not actually execute. Included for
        # completeness/consistency with goal_adapter_train.py.
        raise RuntimeError(
            "Test-split crop cache not found and custom_loader was invoked. "
            "This should already exist from your earlier main.py baseline runs "
            "-- if you're seeing this, something deleted "
            f"features/{args.dataset_name}/{args.dataset_name}-...crop_fea...pkl "
            "(no -train suffix). Re-run `python main.py --dataset_name "
            f"{args.dataset_name} --backbone {args.backbone}` first to rebuild it."
        )

    precomputed_features, target, image_features = load_precomputed_features(
        model, args_shim, processor,
        dataset_name=args.dataset_name, model_size=model_size, alpha=alpha,
        n_samples=n_samples, batch_size=batch_size, num_workers=0,
        data_path=data_path, custom_loader=custom_loader, device=device,
        layer1=args_shim.layer1, layer2=args_shim.layer2,  # split defaults to "test"
    )

    class_names = load_classes(args.dataset_name)
    descriptions_path = f"prompts/{args.dataset_name}/{args.descriptions_method}.json"
    description_features = _encode_class_descriptions(
        model, class_names, descriptions_path, device, fixed_k=args.fixed_k
    ).float()  # [num_classes, fixed_k, D]

    # --- Load trained adapters ---
    dim = description_features.shape[-1]
    adapters = GoalAdapterPair(dim=dim).to(device)
    adapters.load(args.adapter_path, map_location=device)
    adapters.eval()

    patch_embeds_raw = precomputed_features[:, :, :-1].float().to(device)  # [N, R, D]
    patch_weights = precomputed_features[:, :, -1].to(device)              # [N, R] (unused downstream, kept for parity with main.py)
    target = target.to(device)

    with torch.no_grad():
        # Adapt region features
        N, R, D = patch_embeds_raw.shape
        adapted_patch_embeds = adapters.forward_image(patch_embeds_raw.reshape(N * R, D)).reshape(N, R, D)

        # Adapt description features -> zeroshot_weights, matching main.py's
        # post-permute shape: [num_classes, num_descriptions, D]
        adapted_desc = adapters.forward_text(
            description_features.reshape(-1, D).to(device)
        ).reshape(description_features.shape)  # [num_classes, fixed_k, D]

        zeroshot_weights = adapted_desc  # already [num_classes, num_descriptions, D] -- no permute needed
        # (main.py permutes from [num_descriptions, num_classes, D] -> this shape;
        # our _encode_class_descriptions already produces this shape directly.)

        acc_list = []
        print(f"n_run: {n_run}")
        for i in range(n_run):
            patch_embeds = adapted_patch_embeds
            total_size, crop_num, embed_dim = patch_embeds.shape
            num_classes, num_descriptions, embed_dim = zeroshot_weights.shape

            batch_sz = 100
            num_batches = (total_size + batch_sz - 1) // batch_sz
            logits_total = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_sz
                end_idx = min((batch_idx + 1) * batch_sz, total_size)
                patch_embeds_batch = patch_embeds[start_idx:end_idx]
                patch_embeds_flat = patch_embeds_batch.reshape(-1, embed_dim)
                zeroshot_weights_flat = zeroshot_weights.reshape(-1, embed_dim)
                similarity_matrix_flat = torch.matmul(patch_embeds_flat, zeroshot_weights_flat.t())
                similarity_matrix_flat = similarity_matrix_flat / 0.03
                similarity_matrix = similarity_matrix_flat.reshape(
                    end_idx - start_idx, crop_num, num_classes, num_descriptions
                )
                similarity_matrix = similarity_matrix.view(end_idx - start_idx, crop_num, -1)
                log_softmax_matrix = similarity_matrix.log_softmax(dim=-1)
                similarity_matrix_soft = log_softmax_matrix.exp()
                weighted_similarity_matrix = similarity_matrix_soft * similarity_matrix
                weighted_similarity_matrix = weighted_similarity_matrix.reshape(
                    end_idx - start_idx, crop_num, num_classes, num_descriptions
                )
                logits_batch_crop_class = weighted_similarity_matrix.sum(dim=-1)
                logits_batch = logits_batch_crop_class.sum(dim=1)
                logits_total.append(logits_batch)

            logits = torch.cat(logits_total, dim=0)
            acc_list.append(accuracy(logits, target, patch_embeds.size(0), args.dataset_name))

        mean = np.mean(acc_list)
        std = np.std(acc_list)
        print(f"\nGOAL-adapter ours {args.dataset_name}: {mean:.2f}+-{std:.2f}")
        with open("results.txt", "a") as f:
            f.write(f"goal-adapter-ours {args.dataset_name}: {mean:.2f}+-{std:.2f}\n  ")

    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="openai", choices=["openai", "goal"])
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--descriptions_method", type=str, default="bifta-dr")
    parser.add_argument("--fixed_k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    evaluate(args)
