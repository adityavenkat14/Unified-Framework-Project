#!/usr/bin/env python
"""
eval_reviewer.py -- evaluation-only results for the ICLR 2027 reviews of UVA
(submission 37014). Nothing is trained: the saved adapters are loaded and
scored on the cached test-split ABS features.

Requests covered (eval-only):
  B1  (BvN7)  global-image control: the same scoring on the single global CLIP
              embedding instead of the 100 ABS regions, with and without the
              trained adapters.
  ABL (BvN7/TziH) missing ablations: ABS+WCA with BiFTA-DR descriptions and
              no adapters; image-only / text-only adapter switched on at test time.
  B4  (BvN7)  WCA/LaZSL agreement on CUB (and Pets), zero-shot and adapted.
  T2  (TziH)  accuracy by class-difficulty bin after adaptation, split-half design
              (difficulty ranked and accuracy measured on disjoint test images).

Flow per dataset:
  1. sanity gate -- reproduce the paper's CLIP, zero-shot ABS+WCA and UVA
     numbers from the same cache; stop if any is > --tol points off.
  2. all eval-only variants.
  3. write results/reviewer_results.csv, results/reviewer_tables.md,
     results/run_meta.json, results/preds_<dataset>.npz.
  4. numpy self-check: reread the saved predictions/logits from disk and
     recompute every accuracy and agreement number with numpy only.

Prerequisite: the test-split crop cache
  features/<ds>/<ds>-ViT-B-16-crop_fea-1layer-50-20-with-dino2.pkl
must exist (python main.py --dataset_name <ds> builds it).

Usage:
  python eval_reviewer.py --datasets cub oxford_pet
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "methods"))
os.chdir(REPO)

from helper import accuracy, generate_weights, load_classes, load_precomputed_features, set_seed  # noqa: E402
from backbones import load_backbone  # noqa: E402
from adapter import GoalAdapterPair  # noqa: E402
from goal_adapter_train import _encode_class_descriptions  # noqa: E402
from fusion.lazsl_score import build_op_d, compute_lazsl_scores  # noqa: E402

# Numbers printed in the paper (from EXPERIMENT_LOG.md). Used only for the
# sanity gate; never written into the results.
REPORTED = {
    "cub": {
        "clip_global_classprompt": [56.02],
        "clip_global_biftadr_mean": [57.39],
        "zs_abs_wca_cupl": [61.17],
        "uva_full": [80.60, 79.98],  # two draws of the 16-shot training set
    },
    "oxford_pet": {
        "clip_global_classprompt": [88.20],
        "zs_abs_wca_cupl": [92.94],
        "uva_full": [93.00, 93.54],
    },
}
REPORTED_AGREEMENT = {  # post-adapter, from the paper's agreement table
    "oxford_pet": {"rate": 97.98, "acc_agree": 94.24, "acc_disagree": 59.46},
}
GATED = ["clip_global_classprompt", "zs_abs_wca_cupl", "uva_full"]

CROPS_PER_IMAGE = 100  # 50 feature-space + 50 raw-image ABS crops (num_crops=50 each)
RESULTS_DIR = os.path.join(REPO, "results")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform(),
           "torch": torch.__version__, "numpy": np.__version__,
           "cuda_available": torch.cuda.is_available()}
    for mod in ["torchvision", "transformers", "yaml"]:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = "unavailable"
    if torch.cuda.is_available():
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def wca_logits(patches, weights, temp=0.03, batch=100):
    """Exact copy of main.py's 'ours' scoring (and goal_adapter_eval.py's).

    patches: [N, R, D]; weights: [C, K, D]; returns [N, C].
    """
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


def per_class_acc(pred, target, num_classes):
    accs = np.full(num_classes, np.nan)
    for c in range(num_classes):
        m = target == c
        if m.any():
            accs[c] = (pred[m] == c).mean() * 100.0
    return accs


def split_half_bins(preds, variants, target, num_classes, n_bins, n_splits, rng):
    """Split-half difficulty binning. variants[0] defines difficulty.

    For each split and class: a random half of the class's test images ranks
    difficulty (accuracy of variants[0]); the other half measures every variant.
    Returns acc_bins [n_splits, n_bins, len(variants)] (mean of per-class
    accuracies on the measurement halves) and the bin sizes.
    """
    cls_idx = [np.flatnonzero(target == c) for c in range(num_classes)]
    acc_bins = np.zeros((n_splits, n_bins, len(variants)))
    bin_sizes = None
    for s in range(n_splits):
        diff = np.full(num_classes, np.nan)
        meas = np.full((num_classes, len(variants)), np.nan)
        for c in range(num_classes):
            idx = rng.permutation(cls_idx[c])
            half = len(idx) // 2
            if half == 0:
                continue
            sel, ev = idx[:half], idx[half:]
            diff[c] = (preds[variants[0]][sel] == c).mean() * 100.0
            for j, v in enumerate(variants):
                meas[c, j] = (preds[v][ev] == c).mean() * 100.0
        valid = np.flatnonzero(~np.isnan(diff))
        order = valid[np.lexsort((rng.random(len(valid)), diff[valid]))]  # hardest first, random tie-break
        bins = np.array_split(order, n_bins)
        bin_sizes = [len(b) for b in bins]
        for k, bidx in enumerate(bins):
            acc_bins[s, k] = np.nanmean(meas[bidx], axis=0)
    return acc_bins, bin_sizes


def agreement_stats(pred_a, pred_b, target):
    agree = pred_a == pred_b
    correct = pred_a == target
    return {
        "rate": agree.mean() * 100.0,
        "acc_agree": correct[agree].mean() * 100.0 if agree.any() else float("nan"),
        "acc_disagree": correct[~agree].mean() * 100.0 if (~agree).any() else float("nan"),
        "n_agree": int(agree.sum()),
        "n_disagree": int((~agree).sum()),
    }


class _ArgsShim:  # same values as goal_adapter_eval.py / main.py defaults
    num_crops = 50
    top_k = 20
    clip_crop_r1 = 0.6
    clip_crop_r2 = 0.9
    layer1 = 11
    layer2 = 11
    patch_size = 14


def _no_rebuild_loader(path):
    raise RuntimeError(
        "Test-split crop cache missing. Build it first with "
        "`python main.py --dataset_name <ds>` (the notebook does this)."
    )


# --------------------------------------------------------------------------
# per-dataset evaluation
# --------------------------------------------------------------------------
def evaluate_dataset(ds, args, model, device, rows, meta, tables):
    print(f"\n{'=' * 70}\n{ds}\n{'=' * 70}")
    set_seed(args.seed)
    with open(f"cfgs/{ds}.yaml") as f:
        hp = yaml.load(f, Loader=yaml.FullLoader)
    model_size = hp["model_size"]

    cache = os.path.join(
        "features", ds,
        f"{(ds + '-' + model_size).replace('/', '-')}-crop_fea-1layer-50-20-with-dino2.pkl")
    if not os.path.exists(cache):
        raise FileNotFoundError(f"{cache} not found -- run main.py --dataset_name {ds} first")
    ckpt = args.adapter.get(ds, f"adapters_{ds}_16shot.pt")

    feats, target_t, global_t = load_precomputed_features(
        model, _ArgsShim(), None, dataset_name=ds, model_size=model_size, alpha=hp["alpha"],
        n_samples=hp["n_samples"], batch_size=hp["batch_size"], num_workers=0,
        data_path=hp["data_path"], custom_loader=_no_rebuild_loader, device=device,
        layer1=11, layer2=11)
    cache_dtype = feats.dtype
    target = target_t.cpu().numpy()
    n_img = int(target.shape[0])
    meta["datasets"][ds] = {
        "cache_file": cache, "cache_sha256": sha256(cache), "cache_dtype": str(cache_dtype),
        "checkpoint": ckpt, "checkpoint_sha256": sha256(ckpt), "n_test_images": n_img,
        "regions_per_image": int(feats.shape[1]),
    }
    print(f"cache: {cache}  ({n_img} images, {feats.shape[1]} regions, dtype {cache_dtype})")

    class_names = load_classes(ds)
    C = len(class_names)
    preds, saved_logits = {}, {}

    def record(variant, logits, request, notes="", keep_logits=False):
        p = logits.float().max(1)[1].cpu().numpy()  # same op as helper.accuracy
        acc = accuracy(logits, target_t, n_img, ds)
        preds[variant] = p
        if keep_logits:
            saved_logits[variant] = logits.float().cpu().numpy()
        rows.append({"dataset": ds, "request": request, "variant": variant, "metric": "top1_acc",
                     "value": round(acc, 4), "n": n_img, "notes": notes})
        print(f"  {variant:<34s} {acc:6.2f}")
        return acc

    # ---------------- raw features ----------------
    patches_raw16 = feats[:, :, :-1]                  # [N, R, D], cache dtype (main.py path)
    patches_raw = patches_raw16.float()
    glob_raw16 = global_t if global_t.dim() == 2 else global_t.squeeze(1)  # [N, D]
    glob_raw = glob_raw16.float()

    # ---------------- 1. sanity gate ----------------
    print("\n[1] sanity gate (paper numbers)")
    w_clip = generate_weights("clip", model=model, dataset_name=ds, device=device).to(glob_raw16.dtype)
    record("clip_global_classprompt", glob_raw16 @ w_clip, "sanity",
           "global CLIP embedding vs 'a photo of a [CLASS].' (main.py 'clip')")

    w_cupl = generate_weights("ours", model=model, dataset_name=ds,
                              tt_scale=torch.exp(torch.tensor(2.0)).to(device), device=device)
    w_cupl = w_cupl.to(cache_dtype).permute(1, 0, 2)  # [C, 50, D], as in main.py
    record("zs_abs_wca_cupl", wca_logits(patches_raw16, w_cupl), "sanity",
           "zero-shot ABS regions + WCA, first 50 CuPL descriptions (main.py 'ours')", keep_logits=True)

    desc_raw = _encode_class_descriptions(
        model, class_names, f"prompts/{ds}/bifta-dr.json", device, fixed_k=args.fixed_k).float()

    adapters = GoalAdapterPair(dim=desc_raw.shape[-1]).to(device)
    adapters.load(ckpt, map_location=device)
    adapters.eval()
    with torch.no_grad():
        N, R, D = patches_raw.shape
        patches_ad = adapters.forward_image(patches_raw.reshape(N * R, D)).reshape(N, R, D)
        desc_ad = adapters.forward_text(desc_raw.reshape(-1, D)).reshape(desc_raw.shape)
        glob_ad = adapters.forward_image(glob_raw)

    acc_uva = record("uva_full", wca_logits(patches_ad, desc_ad), "sanity",
                     "UVA: adapted ABS regions + adapted BiFTA-DR descriptions, WCA", keep_logits=True)

    gate = []
    for v in GATED:
        reported = REPORTED.get(ds, {}).get(v)
        if not reported:
            continue
        got = next(r["value"] for r in rows if r["dataset"] == ds and r["variant"] == v)
        diff = min(abs(got - x) for x in reported)
        ok = diff <= args.tol
        gate.append({"variant": v, "got": got, "reported": reported, "abs_diff": round(diff, 4), "pass": ok})
        print(f"  gate {v:<28s} got {got:6.2f}  reported {reported}  diff {diff:.2f}  {'PASS' if ok else 'FAIL'}")
    meta["datasets"][ds]["sanity_gate"] = gate
    if ds == "cub":
        closest = min(REPORTED["cub"]["uva_full"], key=lambda x: abs(x - acc_uva))
        meta["datasets"][ds]["checkpoint_matches_draw"] = closest
    if not all(g["pass"] for g in gate):
        meta["datasets"][ds]["stopped"] = "sanity gate failed"
        print(f"\nSTOP: sanity gate failed for {ds}; reviewer metrics not computed.")
        if not args.no_gate:
            return False

    # ---------------- 2. B1 global-image control + ablations ----------------
    print("\n[2] B1 global-image control and test-time ablations")
    w_bifta_mean = generate_weights("bifta-dr", model=model, dataset_name=ds, device=device).to(glob_raw16.dtype)
    record("clip_global_biftadr_mean", glob_raw16 @ w_bifta_mean, "B1",
           "global CLIP embedding vs mean BiFTA-DR embedding (main.py 'bifta-dr')")
    record("global_wca_biftadr_zeroshot", wca_logits(glob_raw.unsqueeze(1), desc_raw), "B1",
           "global embedding as the only region, raw BiFTA-DR (K=30), WCA; no adapters")
    record("global_wca_biftadr_adapted", wca_logits(glob_ad.unsqueeze(1), desc_ad), "B1",
           "global embedding + the region-trained adapters applied at test time; adapters NOT retrained on global features")
    record("abs_wca_biftadr_zeroshot", wca_logits(patches_raw, desc_raw), "ABL",
           "ABS regions + raw BiFTA-DR (K=30), WCA; no adapters", keep_logits=True)
    record("abs_wca_image_adapter_only", wca_logits(patches_ad, desc_raw), "ABL",
           "trained image adapter on, text adapter off at test time (not a separately trained model)")
    record("abs_wca_text_adapter_only", wca_logits(patches_raw, desc_ad), "ABL",
           "trained text adapter on, image adapter off at test time (not a separately trained model)")

    # ---------------- 3. B4 agreement ----------------
    print("\n[3] B4 WCA/LaZSL agreement")
    op_d = build_op_d()
    lz_zs = compute_lazsl_scores(patches_raw16, w_cupl, op_d, batch_chunk=args.lazsl_batch_chunk)
    lz_ad = compute_lazsl_scores(patches_ad, desc_ad, op_d, batch_chunk=args.lazsl_batch_chunk)
    preds["lazsl_zs_cupl"] = lz_zs.float().argmax(1).cpu().numpy()
    preds["lazsl_uva"] = lz_ad.float().argmax(1).cpu().numpy()
    for tag, wca_v, lz_v, note in [
        ("zeroshot", "zs_abs_wca_cupl", "lazsl_zs_cupl", "zero-shot ABS+WCA (CuPL) vs LaZSL on same regions/descriptions"),
        ("adapted", "uva_full", "lazsl_uva", "UVA vs LaZSL on adapted regions/descriptions"),
    ]:
        st = agreement_stats(preds[wca_v], preds[lz_v], target)
        lz_acc = (preds[lz_v] == target).mean() * 100.0
        for metric in ["rate", "acc_agree", "acc_disagree"]:
            rows.append({"dataset": ds, "request": "B4", "variant": f"agreement_{tag}", "metric": metric,
                         "value": round(st[metric], 4), "n": n_img,
                         "notes": f"{note}; n_agree={st['n_agree']} n_disagree={st['n_disagree']}"})
        rows.append({"dataset": ds, "request": "B4", "variant": f"agreement_{tag}", "metric": "gap",
                     "value": round(st["acc_agree"] - st["acc_disagree"], 4), "n": n_img, "notes": note})
        rows.append({"dataset": ds, "request": "B4", "variant": lz_v, "metric": "top1_acc",
                     "value": round(lz_acc, 4), "n": n_img, "notes": "LaZSL standalone"})
        print(f"  {tag:<9s} agree {st['rate']:.2f}%  acc|agree {st['acc_agree']:.2f}  "
              f"acc|disagree {st['acc_disagree']:.2f}  (LaZSL alone {lz_acc:.2f})")
    if ds in REPORTED_AGREEMENT:
        meta["datasets"][ds]["agreement_reported_for_comparison"] = REPORTED_AGREEMENT[ds]

    # ---------------- 4. T2 difficulty bins ----------------
    # Split-half design: difficulty is ranked on one random half of each class's
    # test images and accuracy is measured on the other half, so selection and
    # measurement never use the same observations (ranking and measuring on the
    # same images inflates gains in the hardest bins by regression to the mean).
    # Repeated over args.t2_splits random splits and averaged; every split is
    # unbiased on its own, averaging only reduces split-to-split noise.
    print(f"\n[4] T2 accuracy by class-difficulty bin (split-half, {args.t2_splits} random splits)")
    n_bins = 10 if C >= 100 else 5
    V = ["zs_abs_wca_cupl", "abs_wca_biftadr_zeroshot", "uva_full"]
    pc = {v: per_class_acc(preds[v], target, C) for v in V}
    with open(os.path.join(RESULTS_DIR, f"per_class_{ds}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_idx", "class_name", "n_test", *V])
        for c in range(C):
            w.writerow([c, class_names[c], int((target == c).sum()), *[round(pc[v][c], 4) for v in V]])
    acc_bins, bin_sizes = split_half_bins(preds, V, target, C, n_bins, args.t2_splits,
                                          np.random.default_rng(args.seed + 2027))
    mean = acc_bins.mean(0)
    gain_c = acc_bins[:, :, 2] - acc_bins[:, :, 0]
    gain_b = acc_bins[:, :, 2] - acc_bins[:, :, 1]
    note = (f"split-half: difficulty from zero-shot ABS+WCA (CuPL) on a random half of each class's test images, "
            f"accuracy on the other half; mean over {args.t2_splits} random splits; bin 1 = hardest; "
            f"p2.5/p97.5 = spread across splits, not a confidence interval")
    for k in range(n_bins):
        vals = [("zs_cupl_acc", mean[k, 0]), ("zs_biftadr_acc", mean[k, 1]), ("uva_acc", mean[k, 2]),
                ("gain_vs_zs_cupl", gain_c[:, k].mean()), ("gain_vs_zs_biftadr", gain_b[:, k].mean()),
                ("gain_vs_zs_cupl_p2.5", np.percentile(gain_c[:, k], 2.5)),
                ("gain_vs_zs_cupl_p97.5", np.percentile(gain_c[:, k], 97.5))]
        for metric, val in vals:
            rows.append({"dataset": ds, "request": "T2", "variant": f"bin{k + 1:02d}_of_{n_bins}",
                         "metric": metric, "value": round(float(val), 4), "n": bin_sizes[k], "notes": note})
        print(f"  bin {k + 1:2d}/{n_bins} ({bin_sizes[k]:3d} classes): zs {mean[k, 0]:6.2f}  "
              f"zs-bifta {mean[k, 1]:6.2f}  uva {mean[k, 2]:6.2f}  gain {gain_c[:, k].mean():+6.2f} "
              f"[{np.percentile(gain_c[:, k], 2.5):+.2f}, {np.percentile(gain_c[:, k], 97.5):+.2f}]")
    meta["datasets"][ds]["t2_design"] = {"type": "split-half", "splits": args.t2_splits,
                                         "rng_seed": args.seed + 2027, "bins": n_bins}

    np.savez_compressed(os.path.join(RESULTS_DIR, f"preds_{ds}.npz"), target=target,
                        **{f"pred__{k}": v for k, v in preds.items()},
                        **{f"logits__{k}": v for k, v in saved_logits.items()})
    return True


# --------------------------------------------------------------------------
# numpy self-check: separate code path, reads only files on disk
# --------------------------------------------------------------------------
def numpy_selfcheck(datasets_done):
    with open(os.path.join(RESULTS_DIR, "reviewer_results.csv")) as f:
        rows = list(csv.DictReader(f))
    checks = []
    for ds in datasets_done:
        z = np.load(os.path.join(RESULTS_DIR, f"preds_{ds}.npz"))
        t = z["target"]
        for r in rows:
            if r["dataset"] != ds:
                continue
            v, m, val = r["variant"], r["metric"], float(r["value"])
            if m == "top1_acc" and f"pred__{v}" in z:
                got = float((z[f"pred__{v}"] == t).mean() * 100.0)
                checks.append((ds, v, m, "preds", val, got))
                if f"logits__{v}" in z:  # fully independent: argmax of saved logits
                    got2 = float((np.argmax(z[f"logits__{v}"], axis=1) == t).mean() * 100.0)
                    checks.append((ds, v, m, "logits_argmax", val, got2))
            if v.startswith("agreement_") and m in ("rate", "acc_agree", "acc_disagree"):
                a = z["pred__zs_abs_wca_cupl"] if v.endswith("zeroshot") else z["pred__uva_full"]
                b = z["pred__lazsl_zs_cupl"] if v.endswith("zeroshot") else z["pred__lazsl_uva"]
                ag = a == b
                got = {"rate": ag.mean(), "acc_agree": (a == t)[ag].mean(),
                       "acc_disagree": (a == t)[~ag].mean()}[m] * 100.0
                checks.append((ds, v, m, "preds", val, float(got)))
    bad = [c for c in checks if abs(c[4] - c[5]) > 1e-3]
    with open(os.path.join(RESULTS_DIR, "selfcheck.txt"), "w") as f:
        f.write(f"{len(checks)} checks, {len(bad)} mismatches (tolerance 1e-3 points)\n")
        for c in checks:
            f.write(f"{c[0]:<11s} {c[1]:<32s} {c[2]:<13s} via {c[3]:<14s} csv={c[4]:.4f} numpy={c[5]:.4f}"
                    f"{'  MISMATCH' if abs(c[4] - c[5]) > 1e-3 else ''}\n")
    print(f"\nnumpy self-check: {len(checks)} checks, {len(bad)} mismatches")
    return not bad


# --------------------------------------------------------------------------
# markdown tables (built only from the rows of this run)
# --------------------------------------------------------------------------
def write_tables(rows, meta, datasets_done):
    def val(ds, variant, metric="top1_acc"):
        for r in rows:
            if r["dataset"] == ds and r["variant"] == variant and r["metric"] == metric:
                return r["value"]
        return None

    def f2(x):
        return "n/a" if x is None else f"{x:.2f}"

    name = {"cub": "CUB-200-2011", "oxford_pet": "Oxford-IIIT Pet"}
    L = [f"# Reviewer results (run {meta['timestamp_utc']})", "",
         "All numbers come from `eval_reviewer.py` in this run. Nothing was retrained.", ""]

    L += ["## Sanity check: reproduction of paper numbers", "",
          "| Dataset | Setting | This run | Paper | Diff | Pass |", "|---|---|---:|---|---:|:-:|"]
    for ds, info in meta["datasets"].items():
        for g in info.get("sanity_gate", []):
            L.append(f"| {name.get(ds, ds)} | `{g['variant']}` | {g['got']:.2f} | "
                     f"{' / '.join(f'{x:.2f}' for x in g['reported'])} | {g['abs_diff']:.2f} | "
                     f"{'yes' if g['pass'] else '**no**'} |")
    L.append("")
    failed = [ds for ds, info in meta["datasets"].items()
              if any(not g["pass"] for g in info.get("sanity_gate", []))]
    if failed and datasets_done:
        L += [f"> **WARNING:** the sanity gate failed for {', '.join(failed)} and the run was forced with "
              "`--no_gate`. Do not use those datasets' numbers in the rebuttal.", ""]

    if not datasets_done:
        return "\n".join(L + ["Sanity gate failed on every dataset; no reviewer metrics were computed."])

    L += ["## B1 (BvN7): global-image control", "",
          "Same frozen CLIP ViT-B/16 and the same scoring; only the image representation changes. "
          f"The global row uses 1 image embedding; ABS uses {CROPS_PER_IMAGE} regions per image. "
          "Adapters in the global rows were trained on ABS regions and applied to the global embedding at test time; "
          "they were not retrained on global features.", "",
          "| Image representation | Descriptions | Adapters | " + " | ".join(name.get(d, d) for d in datasets_done) + " |",
          "|---|---|---|" + "---:|" * len(datasets_done)]
    for label, desc, ad, v in [
        ("Global embedding", "class prompt", "no", "clip_global_classprompt"),
        ("Global embedding", "BiFTA-DR (mean)", "no", "clip_global_biftadr_mean"),
        ("Global embedding", "BiFTA-DR (WCA)", "no", "global_wca_biftadr_zeroshot"),
        ("Global embedding", "BiFTA-DR (WCA)", "yes (test time)", "global_wca_biftadr_adapted"),
        (f"ABS, {CROPS_PER_IMAGE} regions", "BiFTA-DR (WCA)", "no", "abs_wca_biftadr_zeroshot"),
        (f"ABS, {CROPS_PER_IMAGE} regions", "BiFTA-DR (WCA)", "yes (UVA)", "uva_full"),
    ]:
        L.append(f"| {label} | {desc} | {ad} | " + " | ".join(f2(val(d, v)) for d in datasets_done) + " |")
    L.append("")

    L += ["## Ablation: dividing the UVA gain (no retraining)", "",
          "Image-only / text-only rows switch one trained adapter off at test time; they are not separately trained models.", "",
          "| Setting | " + " | ".join(name.get(d, d) for d in datasets_done) + " |",
          "|---|" + "---:|" * len(datasets_done)]
    for label, v in [("Zero-shot ABS+WCA, CuPL (paper baseline)", "zs_abs_wca_cupl"),
                     ("Zero-shot ABS+WCA, BiFTA-DR", "abs_wca_biftadr_zeroshot"),
                     ("+ image adapter only", "abs_wca_image_adapter_only"),
                     ("+ text adapter only", "abs_wca_text_adapter_only"),
                     ("+ both adapters (UVA)", "uva_full")]:
        L.append(f"| {label} | " + " | ".join(f2(val(d, v)) for d in datasets_done) + " |")
    L.append("")

    L += ["## B4 (BvN7): WCA/LaZSL agreement", "",
          "| Dataset | Pipeline | Agree (%) | Acc. agree | Acc. disagree | Gap | LaZSL alone |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for d in datasets_done:
        for tag, pl, lz in [("zeroshot", "Zero-shot ABS+WCA (CuPL)", "lazsl_zs_cupl"), ("adapted", "UVA (adapted)", "lazsl_uva")]:
            v = f"agreement_{tag}"
            L.append(f"| {name.get(d, d)} | {pl} | {f2(val(d, v, 'rate'))} | {f2(val(d, v, 'acc_agree'))} | "
                     f"{f2(val(d, v, 'acc_disagree'))} | {f2(val(d, v, 'gap'))} | {f2(val(d, lz))} |")
    L.append("")

    for d in datasets_done:
        brows = sorted({r["variant"] for r in rows if r["dataset"] == d and r["request"] == "T2"})
        L += [f"## T2 (TziH): accuracy by class difficulty, {name.get(d, d)}", "",
              "Split-half design: for each class, a random half of its test images is used only to rank "
              "difficulty (zero-shot ABS+WCA with CuPL), and the other half only to measure accuracy. "
              f"Classes are split into {len(brows)} equal bins (bin 1 = hardest); values are means of per-class "
              f"accuracies on the measurement halves, averaged over {meta.get('t2_splits', '?')} random splits. "
              "The bracket is the 2.5-97.5 percentile of the gain across splits (split-to-split variation, "
              "not a confidence interval). Each half has about half the images per class, so per-bin values are noisy.", "",
              "| Bin | Classes | Zero-shot (CuPL) | Zero-shot (BiFTA-DR) | UVA | Gain vs CuPL [spread] | Gain vs BiFTA-DR |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for b in brows:
            n = next(r["n"] for r in rows if r["dataset"] == d and r["variant"] == b)
            L.append(f"| {b.split('_')[0][3:].lstrip('0')} | {n} | {f2(val(d, b, 'zs_cupl_acc'))} | "
                     f"{f2(val(d, b, 'zs_biftadr_acc'))} | {f2(val(d, b, 'uva_acc'))} | "
                     f"{val(d, b, 'gain_vs_zs_cupl'):+.2f} [{val(d, b, 'gain_vs_zs_cupl_p2.5'):+.2f}, "
                     f"{val(d, b, 'gain_vs_zs_cupl_p97.5'):+.2f}] | {val(d, b, 'gain_vs_zs_biftadr'):+.2f} |")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cub", "oxford_pet"])
    ap.add_argument("--adapter_cub", default="adapters_cub_16shot.pt")
    ap.add_argument("--adapter_oxford_pet", default="adapters_oxford_pet_16shot.pt")
    ap.add_argument("--fixed_k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tol", type=float, default=0.5, help="max allowed gap to paper numbers (points)")
    ap.add_argument("--no_gate", action="store_true", help="continue even if the sanity gate fails")
    ap.add_argument("--lazsl_batch_chunk", type=int, default=64)
    ap.add_argument("--t2_splits", type=int, default=200, help="random split-half repetitions for T2")
    args = ap.parse_args()
    args.adapter = {"cub": args.adapter_cub, "oxford_pet": args.adapter_oxford_pet}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = {"command": " ".join([sys.executable] + sys.argv),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "git_commit": git_commit(), "seed": args.seed, "device": device, "tol": args.tol,
            "fixed_k": args.fixed_k, "t2_splits": args.t2_splits, "versions": versions(), "datasets": {}}
    print(json.dumps({k: meta[k] for k in ["command", "git_commit", "device", "versions"]}, indent=1))

    model, _ = load_backbone("openai", openai_name="ViT-B/16", ckpt_path=None, device=device)
    model.eval()
    model.requires_grad_(False)

    rows, done = [], []
    with torch.no_grad():
        for ds in args.datasets:
            try:
                if evaluate_dataset(ds, args, model, device, rows, meta, None):
                    done.append(ds)
            except Exception as e:  # report, never hide
                meta["datasets"].setdefault(ds, {})["error"] = f"{type(e).__name__}: {e}"
                print(f"\nERROR on {ds}: {type(e).__name__}: {e}")

    with open(os.path.join(RESULTS_DIR, "reviewer_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "request", "variant", "metric", "value", "n", "notes"])
        w.writeheader()
        w.writerows(rows)
    ok = numpy_selfcheck(done) if done else False
    meta["selfcheck_passed"] = ok
    meta["datasets_completed"] = done
    with open(os.path.join(RESULTS_DIR, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(RESULTS_DIR, "reviewer_tables.md"), "w") as f:
        f.write(write_tables(rows, meta, done))
    print(f"\nWrote results/ (completed: {done or 'none'}; self-check {'PASSED' if ok else 'FAILED or not run'})")
    sys.exit(0 if (done and ok and len(done) == len(args.datasets)) else 2)


if __name__ == "__main__":
    main()
