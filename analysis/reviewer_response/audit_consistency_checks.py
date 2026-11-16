"""
audit_consistency_checks.py -- arithmetic checks on the numbers recorded in
EXPERIMENT_LOG.md. Uses no model, data or prediction files (none exist in the
repo). It tests only whether the logged percentages can arise from whole
image counts on each test set, and derives the counts they imply.

Run: python audit_consistency_checks.py
"""

TEST_SIZE = {"cub": 5794, "oxford_pet": 3669}  # standard test splits used by the repo's loaders

LOGGED_ACC = {
    "cub": {"clip": 56.02, "cupl": 54.59, "ours (ABS+WCA)": 61.17, "bifta-dr": 57.39,
            "ours+VR": 61.22, "UVA run A": 80.60, "UVA run B": 79.98},
    "oxford_pet": {"clip": 88.20, "cupl": 91.71, "ours (ABS+WCA)": 92.94, "bifta-dr": 91.63,
                   "ours+VR": 92.80, "UVA run A": 93.00, "UVA run B": 93.54},
    "imagenet100": {"clip": 79.47, "clip-e": 78.92, "clip-d": 79.78, "cupl": 80.81,
                    "bifta-dr": 80.67, "ours (ABS+WCA)": 82.72, "UVA": 83.71},
}
AGREEMENT = {  # post-adapter, as logged: (agree %, acc|agree %, acc|disagree %)
    "oxford_pet": (97.98, 94.24, 59.46),
    "imagenet100": (95.43, 85.59, 44.44),
}


def achievable(p, n):
    k = round(p * n / 100)
    return any(round(j / n * 100, 2) == p for j in (k - 1, k, k + 1))


def reconstruct(n, rate, acc_a, acc_d):
    """All integer (agree, correct|agree, disagree, correct|disagree) matching the logged 2-dp values."""
    sols = []
    for a in range(round(rate * n / 100) - 2, round(rate * n / 100) + 3):
        if not 0 < a < n or round(a / n * 100, 2) != rate:
            continue
        d = n - a
        for ca in range(round(acc_a * a / 100) - 2, round(acc_a * a / 100) + 3):
            if not 0 <= ca <= a or round(ca / a * 100, 2) != acc_a:
                continue
            for cd in range(d + 1):
                if round(cd / d * 100, 2) == acc_d:
                    sols.append((a, ca, d, cd))
    return sols


def gate_metrics(a, ca, d, cd):
    """Disagreement used as an error detector: positive = flagged (disagree), condition = prediction wrong."""
    tp, fp = d - cd, cd          # flagged & wrong, flagged & correct
    fn, tn = a - ca, ca          # not flagged & wrong, not flagged & correct
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": 100 * prec, "recall": 100 * rec,
            "F1": 100 * 2 * prec * rec / (prec + rec),
            "specificity": 100 * tn / (tn + fp), "FPR": 100 * fp / (fp + tn),
            "detector_accuracy": 100 * (tp + tn) / (a + d),
            "classifier_accuracy": 100 * (ca + cd) / (a + d)}


if __name__ == "__main__":
    print("1. Are logged accuracies achievable on the test-set size?")
    for ds, n in TEST_SIZE.items():
        bad = [m for m, p in LOGGED_ACC[ds].items() if not achievable(p, n)]
        print(f"  {ds} (N={n}): {'all consistent' if not bad else 'NOT achievable: ' + str(bad)}")
    bad = [m for m, p in LOGGED_ACC["imagenet100"].items() if not achievable(p, 5000)]
    print(f"  imagenet100 (N=5000 assumed): NOT achievable: {bad}")
    cands = [n for n in range(500, 20001)
             if all(achievable(p, n) for p in LOGGED_ACC["imagenet100"].values())
             and any(round(100 * (s[1] + s[3]) / n, 2) == 83.71 for s in reconstruct(n, *AGREEMENT["imagenet100"]))]
    print(f"  imagenet100: smallest N consistent with all 7 accuracies + agreement: {cands[:5]} "
          f"({len(cands)} values of N in 500..20000)")

    print("\n2. Oxford-IIIT Pet agreement reconstruction (N=3669)")
    sols = reconstruct(3669, *AGREEMENT["oxford_pet"])
    print(f"  integer solutions: {len(sols)}")
    for s in sols:
        m = gate_metrics(*s)
        print(f"  agree={s[0]} correct|agree={s[1]} disagree={s[2]} correct|disagree={s[3]}")
        print("  " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
