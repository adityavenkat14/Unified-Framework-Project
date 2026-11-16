#!/usr/bin/env python
"""
low_compute_baselines.py -- two baselines for the ICLR 2027 reviews of UVA
that need NO image encoding: they read the global CLIP image feature that the
existing crop caches already store for every train and test image.

  tip_adapter     Training-free Tip-Adapter (Zhang et al., ECCV 2022), following
                  the official implementation (github.com/gaopengcuhk/Tip-Adapter,
                  main.py run_tip_adapter + utils.search_hp):
                    clip_logits  = 100 * f @ W_text
                    cache_logits = exp(-beta * (1 - f @ K^T)) @ onehot(L)
                    logits       = clip_logits + alpha * cache_logits
                  beta, alpha chosen by the official grid (search_scale [7, 3],
                  search_step [200, 20]) on the validation images.
  global_adapter  UVA's adapter pair, loss and training settings unchanged, but the
                  image side sees the single global CLIP embedding (R = 1)
                  instead of the 100 ABS regions. Answers BvN7 (B1): is the
                  gain specific to region-level alignment?

Fair-comparison choices (identical to UVA unless listed under deviations):
  * same train-split cache, same make_few_shot_split() (16 train + 4 val per
    class); with --seed 0 and the cache UVA was trained on, this reproduces
    UVA's exact draw
  * same test split and test cache as eval_reviewer.py
  * global_adapter: GoalAdapterPair(hidden 0.25, lambda 0.2, dropout 0.1),
    AdamW lr 1e-3 wd 1e-4, batch 64, <= 30 epochs, patience 5, min_delta 1e-4,
    InfoNCE tau 0.07 against mean-pooled adapted BiFTA-DR prototypes (K=30)

Deviations from the official Tip-Adapter (documented, unavoidable without images):
  * cache keys are ONE view per training image (the cached centre view), not the
    mean over 10 random-augmentation epochs
  * alpha/beta searched on UVA's 4 validation images per class, not the
    dataset's full official validation split
  * text classifier: "a photo of a {class}." (UVA's CLIP baseline), plus a
    variant with the dataset template the official repo uses for Pets
    ("a photo of a {}, a type of pet."); CUB has no official Tip-Adapter config
  * training-free Tip-Adapter only; Tip-Adapter-F (fine-tuned keys) not run

Prerequisites (files that must already exist; nothing is re-encoded):
  features/<ds>/<ds>-ViT-B-16-crop_fea-1layer-50-20-with-dino2.pkl        (test)
  features/<ds>/<ds>-ViT-B-16-crop_fea-1layer-50-20-with-dino2-train.pkl  (train)
CLIP ViT-B/16 is loaded only for its TEXT encoder.

Usage:
  python low_compute_baselines.py --datasets cub oxford_pet --seeds 0 1 2
Outputs: results/baselines_results.csv, results/baselines_meta.json,
         results/baselines_preds_<ds>_seed<k>.npz
"""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "methods"))
os.chdir(REPO)

from helper import generate_weights, load_classes, set_seed  # noqa: E402
from backbones import load_backbone  # noqa: E402
from adapter import GoalAdapterPair  # noqa: E402
from clip import clip as openai_clip  # noqa: E402
from goal_adapter_train import (  # noqa: E402
    FeatureBundle, RegionDescriptionDataset, _encode_class_descriptions,
    evaluate_val_loss, make_few_shot_split, region_description_contrastive_loss)

RESULTS_DIR = os.path.join(REPO, "results")
OFFICIAL_PET_TEMPLATE = "a photo of a {}, a type of pet."


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_path(ds, split):
    return os.path.join("features", ds, f"{ds}-ViT-B-16-crop_fea-1layer-50-20-with-dino2"
                                        f"{'-train' if split == 'train' else ''}.pkl")


def load_global(ds, split):
    p = cache_path(ds, split)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing -- this baseline only reuses existing caches")
    res = pickle.load(open(p, "rb"))
    g = res["images"].float()
    g = g.squeeze(1) if g.dim() == 3 else g
    g = g / g.norm(dim=-1, keepdim=True)
    return g, res["labels"].long().cpu(), p


@torch.no_grad()
def text_weights_template(model, class_names, template, device):
    toks = openai_clip.tokenize([template.format(c) for c in class_names], truncate=True).to(device)
    w = model.encode_text(toks).float()
    return (w / w.norm(dim=-1, keepdim=True)).t()  # [D, C]


def acc(logits, labels):
    return float((logits.argmax(1).cpu() == labels).float().mean() * 100.0)


# ---------------------------------------------------------------- Tip-Adapter
def tip_adapter(keys, key_labels, val_f, val_l, test_f, test_l, w_text, C,
                search_scale=(7, 3), search_step=(200, 20)):
    values = torch.nn.functional.one_hot(key_labels, C).float()
    beta_list = [i * (search_scale[0] - 0.1) / search_step[0] + 0.1 for i in range(search_step[0])]
    alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]

    def logits(f, beta, alpha):
        return 100.0 * f @ w_text + alpha * torch.exp(-beta * (1 - f @ keys.t())) @ values

    best = (-1.0, None, None)
    for beta in beta_list:  # official loop order; strict '>' keeps the first best
        for alpha in alpha_list:
            a = acc(logits(val_f, beta, alpha), val_l)
            if a > best[0]:
                best = (a, beta, alpha)
    val_acc, beta, alpha = best
    test_logits = logits(test_f, beta, alpha)
    return {"val_acc": val_acc, "beta": beta, "alpha": alpha,
            "test_acc": acc(test_logits, test_l), "zs_test_acc": acc(100.0 * test_f @ w_text, test_l),
            "pred": test_logits.argmax(1).cpu().numpy()}


# ---------------------------------------------------------- global adapter
def wca_logits(patches, weights, temp=0.03, batch=100):  # identical to eval_reviewer.py / main.py
    n, r, d = patches.shape
    c, k, _ = weights.shape
    w_flat = weights.reshape(-1, d)
    out = []
    for s in range(0, n, batch):
        e = min(s + batch, n)
        sim = torch.matmul(patches[s:e].reshape(-1, d), w_flat.t()) / temp
        sim = sim.reshape(e - s, r, c * k)
        soft = sim.log_softmax(dim=-1).exp()
        out.append((soft * sim).reshape(e - s, r, c, k).sum(dim=-1).sum(dim=1))
    return torch.cat(out, dim=0)


def train_global_adapter(bundle, train_idx, val_idx, seed, device, hp, log):
    """Same loop as goal_adapter_train.train(), with R = 1 global feature."""
    set_seed(seed)
    tr = DataLoader(RegionDescriptionDataset(bundle, train_idx), batch_size=hp["batch_size"],
                    shuffle=True, drop_last=False)
    va = DataLoader(RegionDescriptionDataset(bundle, val_idx), batch_size=hp["batch_size"], shuffle=False)
    D = bundle.region_features.shape[-1]
    ad = GoalAdapterPair(dim=D, hidden_ratio=hp["hidden_ratio"], ratio=hp["residual_ratio"],
                         dropout=hp["dropout"]).to(device)
    opt = torch.optim.AdamW(ad.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    desc = bundle.description_features.to(device)
    best, best_state, bad, best_epoch, epochs_run = float("inf"), None, 0, 0, 0
    for ep in range(hp["epochs"]):
        ad.train()
        run, cnt = 0.0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            B, R, _ = x.shape
            xr = ad.forward_image(x.reshape(B * R, D)).reshape(B, R, D)
            dd = ad.forward_text(desc.view(-1, D)).view(desc.shape)
            loss = region_description_contrastive_loss(xr, y, dd, hp["temperature"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * B
            cnt += B
        vl = evaluate_val_loss(ad, va, desc, hp["temperature"], device)
        epochs_run = ep + 1
        log.append({"epoch": ep + 1, "train_loss": run / max(1, cnt), "val_loss": vl})
        if vl < best - hp["min_delta"]:
            best, best_state, bad, best_epoch = vl, copy.deepcopy(ad.state_dict()), 0, ep + 1
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    ad.load_state_dict(best_state)
    ad.eval()
    return ad, {"best_val_loss": best, "best_epoch": best_epoch, "epochs_run": epochs_run}


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cub", "oxford_pet"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="seed 0 with UVA's train cache reproduces UVA's 16-shot draw")
    ap.add_argument("--shots", type=int, default=16)
    ap.add_argument("--val_shots", type=int, default=4)
    ap.add_argument("--fixed_k", type=int, default=30)
    args = ap.parse_args()
    hp = dict(hidden_ratio=0.25, residual_ratio=0.2, dropout=0.1, lr=1e-3, weight_decay=1e-4,
              batch_size=64, epochs=30, patience=5, min_delta=1e-4, temperature=0.07)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception as e:  # noqa: BLE001
        commit = f"unavailable ({e})"
    meta = {"command": " ".join([sys.executable] + sys.argv),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "git_commit": commit, "device": device, "adapter_hparams": hp,
            "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
            "datasets": {}}
    model, _ = load_backbone("openai", openai_name="ViT-B/16", ckpt_path=None, device=device)
    model.eval()
    model.requires_grad_(False)
    rows = []

    def add(ds, seed, method, metric, value, notes=""):
        rows.append({"dataset": ds, "seed": seed, "method": method, "metric": metric,
                     "value": round(float(value), 4), "notes": notes})

    for ds in args.datasets:
        info = meta["datasets"].setdefault(ds, {})
        try:
            tr_g, tr_l, tr_p = load_global(ds, "train")
            te_g, te_l, te_p = load_global(ds, "test")
        except FileNotFoundError as e:
            info["blocked"] = str(e)
            print(f"BLOCKED {ds}: {e}")
            continue
        info.update({"train_cache": tr_p, "train_cache_sha256": sha256(tr_p), "n_train_pool": len(tr_l),
                     "test_cache": te_p, "test_cache_sha256": sha256(te_p), "n_test": len(te_l)})
        classes = load_classes(ds)
        C = len(classes)
        with torch.no_grad():
            desc = _encode_class_descriptions(model, classes, f"prompts/{ds}/bifta-dr.json",
                                              device, fixed_k=args.fixed_k).float().cpu()
            w_clip = generate_weights("clip", model=model, dataset_name=ds, device=device).float()
            w_bifta = generate_weights("bifta-dr", model=model, dataset_name=ds, device=device).float()
            w_texts = {"classprompt": w_clip.to(device), "biftadr_mean": w_bifta.to(device)}
            if ds == "oxford_pet":
                w_texts["official_pet_template"] = text_weights_template(model, classes, OFFICIAL_PET_TEMPLATE, device)
        bundle = FeatureBundle(region_features=tr_g.unsqueeze(1), region_labels=tr_l,
                               description_features=desc, class_names=classes)
        te_dev, te_l_cpu = te_g.to(device), te_l

        # zero-shot references on the same global test features (no training)
        add(ds, "-", "zeroshot_global_classprompt", "test_acc", acc(te_dev @ w_texts["classprompt"], te_l_cpu))
        add(ds, "-", "zeroshot_global_wca_biftadr", "test_acc",
            acc(wca_logits(te_dev.unsqueeze(1), desc.to(device)), te_l_cpu))

        for seed in args.seeds:
            tr_idx, va_idx = make_few_shot_split(bundle, args.shots, args.val_shots, seed=seed)
            info.setdefault("draws", {})[str(seed)] = {
                "train_idx_sha256": hashlib.sha256(np.sort(tr_idx.numpy()).tobytes()).hexdigest()[:16],
                "n_train": len(tr_idx), "n_val": len(va_idx)}
            keys = tr_g[tr_idx].to(device)
            preds = {}
            for tname, w in w_texts.items():
                r = tip_adapter(keys, tr_l[tr_idx], tr_g[va_idx].to(device), tr_l[va_idx],
                                te_dev, te_l_cpu, w, C)
                m = f"tip_adapter_{tname}"
                add(ds, seed, m, "test_acc", r["test_acc"],
                    "training-free; single-view cache keys; alpha/beta on 4 val shots/class")
                add(ds, seed, m, "val_acc", r["val_acc"])
                add(ds, seed, m, "beta", r["beta"])
                add(ds, seed, m, "alpha", r["alpha"])
                preds[m] = r["pred"]
                print(f"{ds} seed {seed} {m:<38s} test {r['test_acc']:.2f} (beta {r['beta']:.2f}, alpha {r['alpha']:.2f})")

            log = []
            ad, st = train_global_adapter(bundle, tr_idx, va_idx, seed, device, hp, log)
            with torch.no_grad():
                g_ad = ad.forward_image(te_dev)
                d_ad = ad.forward_text(desc.to(device).reshape(-1, desc.shape[-1])).reshape(desc.shape)
                lg_wca = wca_logits(g_ad.unsqueeze(1), d_ad)
                proto = d_ad.mean(1)
                proto = proto / proto.norm(dim=-1, keepdim=True)
                lg_proto = g_ad @ proto.t()
            for m, lg, note in [("global_adapter_wca", lg_wca, "same scoring as UVA with R=1"),
                                ("global_adapter_prototype", lg_proto, "cosine to mean adapted description")]:
                add(ds, seed, m, "test_acc", acc(lg, te_l_cpu), note)
                preds[m] = lg.argmax(1).cpu().numpy()
                print(f"{ds} seed {seed} {m:<38s} test {acc(lg, te_l_cpu):.2f}")
            add(ds, seed, "global_adapter", "best_epoch", st["best_epoch"])
            add(ds, seed, "global_adapter", "epochs_run", st["epochs_run"])
            add(ds, seed, "global_adapter", "best_val_loss", st["best_val_loss"])
            info["draws"][str(seed)]["global_adapter_log"] = log
            np.savez_compressed(os.path.join(RESULTS_DIR, f"baselines_preds_{ds}_seed{seed}.npz"),
                                target=te_l_cpu.numpy(), **{f"pred__{k}": v for k, v in preds.items()})

    with open(os.path.join(RESULTS_DIR, "baselines_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "method", "metric", "value", "notes"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RESULTS_DIR, "baselines_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    # numpy self-check of every test accuracy against the saved predictions
    bad = 0
    for r in rows:
        if r["metric"] == "test_acc" and r["seed"] != "-":
            z = np.load(os.path.join(RESULTS_DIR, f"baselines_preds_{r['dataset']}_seed{r['seed']}.npz"))
            if abs((z[f"pred__{r['method']}"] == z["target"]).mean() * 100 - r["value"]) > 1e-3:
                bad += 1
    print(f"self-check mismatches: {bad}")
    blocked = [d for d, i in meta["datasets"].items() if "blocked" in i]
    sys.exit(0 if not bad and not blocked else 2)


if __name__ == "__main__":
    main()
