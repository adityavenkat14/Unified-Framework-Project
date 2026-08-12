import numpy as np
import torch
import yaml
import json
import argparse
from helper import (
    accuracy,
    generate_weights,
    load_precomputed_features,
    set_seed
)
from clip import clip
from torchvision.transforms import v2 as T
from torchvision import datasets
from torch.nn import functional as F
from PIL import Image
import cv2
from transformers import AutoImageProcessor, ViTModel
from tqdm import tqdm
import random
from backbones import load_backbone
from fusion.lazsl_score import build_op_d, compute_lazsl_scores
from fusion.fuse import fuse_scores, accuracy_from_scores, grid_search_weights

def main(args):  
    device: str = "cuda"
    seed: int = args.seed
    num_workers: int = 2
    
    def custom_loader(path: str) -> torch.Tensor:
        img = datasets.folder.default_loader(path)
        W, H = img.size
        img_cv2 = np.array(img)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
        image = img.resize((224, 224))
        augmented_imgs = [processor_dino(img, return_tensors="pt")["pixel_values"].squeeze(0)]
        augmented_imgs.extend(processor(img) for _ in range(1))
        attention_imgs_dino = processor_dino(img, return_tensors="pt")
        with torch.no_grad():
            image_attention_mh = model_dino(**attention_imgs_dino, output_attentions=True)
            image_attention_mh = image_attention_mh.attentions
        n_head = image_attention_mh[11].shape[1]
        attention_map = image_attention_mh[11][0, :, 0, 1:].reshape(n_head, -1).float()
        att_map = attention_map.mean(dim=0)

        top_k = args.top_k 
        att_map_flat = att_map.flatten() 
        topk_values, topk_indices = torch.topk(att_map_flat, top_k) 

        topk_probs = torch.softmax(topk_values / 0.03, dim=0) 

        num_samples = args.num_crops
        sampled_indices = torch.multinomial(topk_probs, num_samples, replacement=True) 
        sampled_patch_indices = topk_indices[sampled_indices] 
        crop_img_dino = []

        for sampled_index in sampled_patch_indices:
            i, j = sampled_index // crop_size, sampled_index % crop_size
            patch_x_min = int(j * patch_size * (W / (crop_size * patch_size))) 
            patch_y_min = int(i * patch_size * (H / (crop_size * patch_size))) 
            patch_x_max = min(patch_x_min + int(patch_size * (W / (crop_size * patch_size))), W)  
            patch_y_max = min(patch_y_min + int(patch_size * (H / (crop_size * patch_size))), H) 

            center_x = (patch_x_min + patch_x_max) // 2
            center_y = (patch_y_min + patch_y_max) // 2
            crop_width = random.randint(int(W * args.clip_crop_r1), int(W * args.clip_crop_r2))  
            crop_height = random.randint(int(H * args.clip_crop_r1), int(H * args.clip_crop_r2))  
            x_min = max(center_x - crop_width // 2, 0)
            y_min = max(center_y - crop_height // 2, 0)
            x_max = min(center_x + crop_width // 2, W)
            y_max = min(center_y + crop_height // 2, H)
            cropped_image = img.crop((x_min, y_min, x_max, y_max))
            crop_img_dino.append(cropped_image)
            augmented_imgs.extend(processor(cropped_image) for _ in range(1))

        return torch.stack(augmented_imgs)
    device = torch.device(device)
    print("Device:", device)
    print("num_workers:", num_workers)

    with open(file=f"cfgs/{args.dataset_name}.yaml") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    set_seed(seed)

    model_size = hparams["model_size"]
    alpha = hparams["alpha"]
    n_samples = hparams["n_samples"]
    batch_size = hparams["batch_size"]
    data_path = hparams["data_path"]

    # load model
    print(f"Loading {model_size} (backbone={args.backbone})")
    model, processor = load_backbone(
        args.backbone, openai_name=model_size, ckpt_path=args.goal_ckpt, device=device
    )
    model.eval()
    model.requires_grad_(False)
    patch_size = 16
    crop_size = 14

    def load_ckpt(ckpt_id="facebook/dino-vitb16"):
        image_processor = AutoImageProcessor.from_pretrained(ckpt_id)
        # sdpa (the new default attention backend) silently ignores
        # output_attentions=True -- force eager so attentions are actually returned.
        model = ViTModel.from_pretrained(ckpt_id, attn_implementation="eager").eval()
        return model, image_processor

    ckpt_id = "facebook/dino-vitb16"

    model_dino, processor_dino = load_ckpt(ckpt_id)
    model_dino.eval()
    model_dino.requires_grad_(False)


    precomputed_features,target,image_features,= load_precomputed_features(model,args,processor,dataset_name=args.dataset_name,model_size=model_size,alpha=alpha,n_samples=n_samples,batch_size=batch_size,num_workers=num_workers,data_path=data_path,custom_loader=custom_loader,device=device,layer1=args.layer1,layer2=args.layer2)

    max_size = precomputed_features.size(1)
    image_features = image_features.to(device)

    results = {}
    with torch.no_grad():
        methods = hparams["methods"]
        for method in methods:
            method = list(method.values())[0]
            method_name = method["name"]
            method_enabled = method["enabled"]

            text_scale = (
                torch.exp(torch.tensor(method["text_scale"])).to(device)
                if "text_scale" in method
                else None
            )
            image_scale = (
                torch.exp(torch.tensor(method["image_scale"])).to(device)
                if "image_scale" in method
                else None
            )

            if method_enabled:
                zeroshot_weights = generate_weights(
                    method_name,
                    model=model,
                    dataset_name=args.dataset_name,
                    tt_scale=text_scale,
                    device=device,
                )
            
                zeroshot_weights = zeroshot_weights.to(image_features.dtype)
            else:
                continue

            
            if method_name != "ours":
                logits = image_features.squeeze(1) @ zeroshot_weights
                baseline_acc = accuracy(
                    logits, target, image_features.size(0), args.dataset_name
                )
                print(f"{method_name}: {baseline_acc:.2f}\n")
                results[method_name] = round(baseline_acc, 2)
                if method_name == "clip":
                    # zeroshot_weights here is [num_classes, D] (whole image vs. plain
                    # class name) -> logits is already [B, num_classes] = S_global.
                    s_global_logits = logits

            if method_name == "ours":
                acc_list = []
                patch_num = hparams["patch_n"]
                # [num_classes, num_descriptions, D], pre-permute -- this is the shape
                # LaZSL's OP_d expects per-class description embeddings in.
                # zeroshot_classifier returns [num_descriptions, num_classes, D] (torch.stack(..., dim=1)).
                # The permute below turns it into [num_classes, num_descriptions, D] -- capture AFTER
                # permute, not before, since that's the shape compute_lazsl_scores expects (indexing by
                # class first). Capturing pre-permute was the bug that produced a [num_desc]-shaped score
                # tensor instead of a [num_classes]-shaped one.
                zeroshot_weights = zeroshot_weights.permute(1, 0, 2)
                zeroshot_weights_for_lazsl = zeroshot_weights.clone()
                print(f"n_run: {hparams['n_run']}")
                for i in range(hparams["n_run"]):
                    random_indices = torch.randint(0, max_size, (patch_num,))
                    sampled_features = precomputed_features
                    patch_embeds = sampled_features[:, :, :-1] 
                    patch_weights = sampled_features[:, :, -1]
                    del sampled_features
                    logits_sum = []
                    logits_total = []
                    batch_size = 100
                    total_size, crop_num, embed_dim = patch_embeds.shape
                    num_classes, num_descriptions, embed_dim = zeroshot_weights.shape
                    num_batches = (total_size + batch_size - 1) // batch_size  
                    logits_total = []
                    for batch_idx in tqdm(range(num_batches)):
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, total_size)
                        patch_weights_batch = patch_weights[start_idx:end_idx]
                        patch_embeds_batch = patch_embeds[start_idx:end_idx]  
                        patch_embeds_flat = patch_embeds_batch.reshape(-1, embed_dim) 
                        zeroshot_weights_flat = zeroshot_weights.reshape(-1, embed_dim) 
                        similarity_matrix_flat = torch.matmul(patch_embeds_flat, zeroshot_weights_flat.t()) 
                        similarity_matrix_flat = similarity_matrix_flat / 0.03
                        similarity_matrix = similarity_matrix_flat.reshape(end_idx - start_idx, crop_num, num_classes, num_descriptions)
                        similarity_matrix = similarity_matrix.view(end_idx - start_idx, crop_num, -1)
                        log_softmax_matrix = similarity_matrix.log_softmax(dim=-1)  
                        similarity_matrix_soft = log_softmax_matrix.exp()
                        weighted_similarity_matrix = similarity_matrix_soft * similarity_matrix  
                        weighted_similarity_matrix = weighted_similarity_matrix.reshape(end_idx - start_idx, crop_num, num_classes, num_descriptions)
                        logits_batch_crop_class = weighted_similarity_matrix.sum(dim=-1)  
                        w_i = (patch_weights_batch * image_scale).softmax(-1).unsqueeze(-1) 
                        logits_batch = (logits_batch_crop_class).sum(dim=1) 
                        logits_total.append(logits_batch)
        
                    logits = torch.cat(logits_total, dim=0)  
                    acc_list.append(
                        accuracy(logits, target, patch_embeds.size(0), args.dataset_name)
                    )

                mean = np.mean(acc_list)
                std = np.std(acc_list)
                with open('results.txt', 'a') as f:
                    f.write(f"{method_name} {args.dataset_name}: {mean:.2f}+-{std:.2f}\n  ")
                    for acc in acc_list:
                        f.write(f"{acc}\n")
                    f.write('----------------\n')

                # from the final run: S_WCA logits and the patch embeddings LaZSL reuses
                s_wca_logits = logits
                patch_embeds_for_lazsl = patch_embeds

        # ---- LaZSL score + fusion pass ----
        if args.enable_fusion:
            assert "s_global_logits" in locals(), \
                "enable 'clip' method in the cfg yaml so S_global is available for fusion"
            assert "s_wca_logits" in locals(), \
                "enable 'ours' method in the cfg yaml so S_WCA is available for fusion"

            print("\nComputing S_LaZSL via optimal transport (this loops per-class, slow)...")
            op_d = build_op_d(max_iter=args.lazsl_max_iter, gama=args.lazsl_gama,
                               theta=args.lazsl_theta, constrain_type=args.lazsl_constrain_type)
            s_lazsl_logits = compute_lazsl_scores(
                patch_embeds_for_lazsl, zeroshot_weights_for_lazsl, op_d,
                batch_chunk=args.lazsl_batch_chunk,
            )
            lazsl_acc = accuracy_from_scores(s_lazsl_logits, target)
            print(f"lazsl (standalone): {lazsl_acc:.2f}\n")

            # --- WCA/LaZSL agreement diagnostic ---
            # Weighted-sum fusion (below) asks "does adding S_LaZSL move accuracy."
            # This asks a different question: when WCA and LaZSL AGREE on a prediction,
            # is that prediction more trustworthy than when they disagree? If so, LaZSL
            # is useful as a confidence/consistency signal even though it doesn't help
            # as a raw fusion term (which is what we found on both OxfordPets and CUB).
            wca_pred = s_wca_logits.argmax(dim=1)

            # --- per-class accuracy, for the GOAL-grounding-quality diagnostic ---
            # Cheap to compute here (already have s_wca_logits/target aligned) --
            # separately, goal_class_diagnostic.py computes a per-class GOAL grounding
            # score. Correlating the two tests: do classes where GOAL finds strong,
            # specific grounding for their descriptions tend to be the classes WCA
            # actually gets right?
            num_classes_ = s_wca_logits.shape[1]
            per_class_acc = {}
            for c in range(num_classes_):
                mask = target == c
                if mask.sum() > 0:
                    per_class_acc[c] = (wca_pred[mask] == target[mask]).float().mean().item() * 100
            with open(f'features/{args.dataset_name}/wca_per_class_acc.json', 'w') as f:
                json.dump(per_class_acc, f)
            print(f"Saved per-class WCA accuracy for {len(per_class_acc)} classes.\n")

            lazsl_pred = s_lazsl_logits.argmax(dim=1)
            agree_mask = wca_pred == lazsl_pred
            n_agree = agree_mask.sum().item()
            pct_agree = 100.0 * n_agree / len(target)
            if n_agree > 0:
                acc_agree = (wca_pred[agree_mask] == target[agree_mask]).float().mean().item() * 100
            else:
                acc_agree = float('nan')
            disagree_mask = ~agree_mask
            n_disagree = disagree_mask.sum().item()
            if n_disagree > 0:
                acc_disagree_wca = (wca_pred[disagree_mask] == target[disagree_mask]).float().mean().item() * 100
            else:
                acc_disagree_wca = float('nan')
            print(f"WCA/LaZSL agreement: {pct_agree:.1f}% of images ({n_agree}/{len(target)})")
            print(f"  accuracy when they AGREE:       {acc_agree:.2f}  (WCA overall: {accuracy_from_scores(s_wca_logits, target):.2f})")
            print(f"  WCA accuracy when they DISAGREE: {acc_disagree_wca:.2f}\n")
            with open('results.txt', 'a') as f:
                f.write(f"[agreement diagnostic] {args.dataset_name}: agree={pct_agree:.1f}% "
                        f"acc_when_agree={acc_agree:.2f} wca_acc_when_disagree={acc_disagree_wca:.2f}\n")

            if args.fusion_search:
                # Methodology fix: the grid search must NOT see the data the final
                # accuracy is reported on, or "best" weights are partly just fitting
                # noise in this specific split (this is exactly what happened before --
                # a 1,331-combination search on the full test set found a +0.03 point
                # "improvement" that's indistinguishable from chance). Split into a val
                # half (search sees this) and a held-out test half (final number comes
                # from this only, using the weights chosen on val).
                g = torch.Generator().manual_seed(args.fusion_split_seed)
                n = target.shape[0]
                perm = torch.randperm(n, generator=g)
                val_idx, test_idx = perm[: n // 2], perm[n // 2 :]

                best = grid_search_weights(
                    s_global_logits[val_idx], s_wca_logits[val_idx], s_lazsl_logits[val_idx],
                    target[val_idx], steps=args.fusion_search_steps,
                )
                alpha, beta, gamma = best["alpha"], best["beta"], best["gamma"]
                print(f"Best fusion weights on VAL half (n={len(val_idx)}): {best}")

                fused_test = fuse_scores(s_global_logits[test_idx], s_wca_logits[test_idx],
                                          s_lazsl_logits[test_idx], alpha, beta, gamma)
                fused_test_acc = accuracy_from_scores(fused_test, target[test_idx])
                # also report plain WCA on the SAME held-out half, for a fair apples-to-apples
                # comparison -- otherwise fused_test_acc vs. the full-set "ours" number above
                # aren't measuring the same thing
                wca_test_acc = accuracy_from_scores(s_wca_logits[test_idx], target[test_idx])
                print(f"On HELD-OUT test half (n={len(test_idx)}, never seen by the search):")
                print(f"  wca alone:  {wca_test_acc:.2f}")
                print(f"  fusion (alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}): {fused_test_acc:.2f}")
                with open('results.txt', 'a') as f:
                    f.write(f"[val/test split, seed={args.fusion_split_seed}] "
                            f"wca_test: {wca_test_acc:.2f}  "
                            f"fusion_test(a={alpha:.2f},b={beta:.2f},g={gamma:.2f}): {fused_test_acc:.2f}\n")
                    f.write('----------------\n')
            else:
                alpha, beta, gamma = args.alpha, args.beta, args.gamma
                fused = fuse_scores(s_global_logits, s_wca_logits, s_lazsl_logits, alpha, beta, gamma)
                fused_acc = accuracy_from_scores(fused, target)
                print(f"fusion (alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}): {fused_acc:.2f}\n")
                with open('results.txt', 'a') as f:
                    f.write(f"lazsl {args.dataset_name}: {lazsl_acc:.2f}\n")
                    f.write(f"fusion(a={alpha:.2f},b={beta:.2f},g={gamma:.2f}) {args.dataset_name}: {fused_acc:.2f}\n")
                    f.write('----------------\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script using argparse")
    parser.add_argument('--dataset_name', type=str, default='oxford_pet', help='Name of datasets')
    parser.add_argument('--patch_size', type=int, default=14, help='Size of layer1')
    parser.add_argument('--num_crops', type=int, default=50, help='Number of crops')
    parser.add_argument('--top_k', type=int, default=20, help='Topk values')
    parser.add_argument('--layer1', type=int, default=11, help='Size of layer1')
    parser.add_argument('--layer2', type=int, default=11, help='Size of layer2')
    parser.add_argument('--clip_crop_r1', type=float, default=0.6, help='Clip crop ratio 1')
    parser.add_argument('--clip_crop_r2', type=float, default=0.9, help='Clip crop ratio 2')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    # --- backbone selection (step 1-2 of the unified plan: GOAL-finetuned CLIP) ---
    parser.add_argument('--backbone', type=str, default='openai', choices=['openai', 'goal'],
                         help="'openai' = vanilla pretrained CLIP, 'goal' = GOAL-finetuned checkpoint")
    parser.add_argument('--goal_ckpt', type=str, default=None,
                         help='Path to GOAL fine-tuned checkpoint (.pt), required if --backbone goal')

    # --- fusion (step 5 of the unified plan: S = a*S_global + b*S_WCA + g*S_LaZSL) ---
    parser.add_argument('--enable_fusion', action='store_true',
                         help='Also compute S_LaZSL and the fused score (requires clip + ours enabled in cfg yaml)')
    parser.add_argument('--alpha', type=float, default=0.34, help='weight on S_global')
    parser.add_argument('--beta', type=float, default=0.33, help='weight on S_WCA')
    parser.add_argument('--gamma', type=float, default=0.33, help='weight on S_LaZSL')
    parser.add_argument('--fusion_search', action='store_true',
                         help='grid-search alpha/beta/gamma on THIS split instead of using --alpha/beta/gamma '
                              '(only valid as a validation-split step, not for final reported test accuracy)')
    parser.add_argument('--fusion_search_steps', type=int, default=5,
                         help='grid resolution per weight, e.g. 5 -> {0, .25, .5, .75, 1}, 11 -> steps of 0.1')
    parser.add_argument('--fusion_split_seed', type=int, default=1,
                         help='seed for the val/test split used when --fusion_search is on')
    parser.add_argument('--lazsl_max_iter', type=int, default=100, help='Sinkhorn max iterations')
    parser.add_argument('--lazsl_gama', type=float, default=0.1, help='LaZSL OP_d gama hyperparam')
    parser.add_argument('--lazsl_theta', type=float, default=0.0, help='LaZSL OP_d theta hyperparam')
    parser.add_argument('--lazsl_constrain_type', type=str, default='const', choices=['const', 'patch', 'att'])
    parser.add_argument('--lazsl_batch_chunk', type=int, default=64,
                         help='images per Sinkhorn batch (lower if you hit OOM)')

    args = parser.parse_args()
    main(args)