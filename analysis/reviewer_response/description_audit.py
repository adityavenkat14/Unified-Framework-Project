"""
description_audit.py -- REAL analysis of the description files already in the
repository (prompts/<ds>/{cupl,clip-d,bifta-dr}.json). No model, no images.

Answers, by exact string matching:
  * pool size per class (CuPL + CLIP-D, exact duplicates removed, as bifta_dr.py does)
  * how many of the 30 BiFTA-DR descriptions come from CuPL vs CLIP-D
  * overlap between the BiFTA-DR set (used by UVA) and the first 50 CuPL
    descriptions (used by zero-shot ABS+WCA) -- quantifies how different the
    two description sets in the paper's 61.17 -> 80.29 comparison are
  * for classes that kept < 30, the exact number removed by the 0.99 cosine
    de-duplication (kept < 30 means every survivor was kept)

Usage: python description_audit.py <repo_root> <out_dir>
"""
import csv
import json
import os
import sys

repo, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
summary = []

for ds in ["cub", "oxford_pet", "imagenet100"]:
    p = lambda name: os.path.join(repo, "prompts", ds, f"{name}.json")
    cupl, clipd, dr = (json.load(open(p(n))) for n in ["cupl", "clip-d", "bifta-dr"])
    rows = []
    for cls, kept in dr.items():
        c, d = cupl.get(cls, []), clipd.get(cls, [])
        pool = list(dict.fromkeys(c + d))
        first50 = set(c[:50])
        cset, dset = set(c), set(d)
        from_cupl = sum(k in cset for k in kept)
        from_clipd_only = sum((k in dset) and (k not in cset) for k in kept)
        unknown = len(kept) - from_cupl - from_clipd_only
        rows.append({
            "class": cls, "cupl_n": len(c), "clipd_n": len(d), "pool_unique": len(pool),
            "kept": len(kept), "kept_from_cupl": from_cupl, "kept_from_clipd_only": from_clipd_only,
            "kept_from_neither": unknown,
            "kept_in_cupl_first50": sum(k in first50 for k in kept),
            "cupl_first50_n": len(first50),
            "dedup_removed_exact": (len(pool) - len(kept)) if len(kept) < 30 else "",
        })
    with open(os.path.join(out, f"descriptions_{ds}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    short = [r for r in rows if r["kept"] < 30]
    s = {
        "dataset": ds, "classes": n,
        "pool_unique_mean": round(mean("pool_unique"), 2),
        "kept_mean": round(mean("kept"), 2),
        "pct_kept_from_cupl": round(100 * sum(r["kept_from_cupl"] for r in rows) / sum(r["kept"] for r in rows), 2),
        "pct_kept_from_clipd_only": round(100 * sum(r["kept_from_clipd_only"] for r in rows) / sum(r["kept"] for r in rows), 2),
        "kept_from_neither_total": sum(r["kept_from_neither"] for r in rows),
        "pct_dr_also_in_cupl_first50": round(100 * sum(r["kept_in_cupl_first50"] for r in rows) / sum(r["kept"] for r in rows), 2),
        "mean_dr_descriptions_shared_with_cupl_first50": round(mean("kept_in_cupl_first50"), 2),
        "classes_kept_below_30": len(short),
        "dedup_removed_in_those_classes": sum(r["dedup_removed_exact"] for r in short),
        "min_kept": min(r["kept"] for r in rows),
    }
    summary.append(s)

with open(os.path.join(out, "descriptions_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
for s in summary:
    print(json.dumps(s))
