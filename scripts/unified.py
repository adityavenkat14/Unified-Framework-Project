"""
unified.py — Single entry point for the whole framework.

Orchestrates every validated stage (bootstrap, description refinement,
zero-shot baseline scoring, adapter training, adapter evaluation, and
single-image prediction) as subcommands of one CLI.

Deliberately implemented as subprocess calls to the exact existing,
already-validated scripts (bifta_dr.py, main.py, goal_adapter_train.py,
goal_adapter_eval.py, predict_unified.py, bootstrap_session.py) rather
than reimplementing their logic here -- this guarantees the results are
identical to running each script by hand, with zero risk of silent drift
from what's actually been tested and logged.

USAGE:
    # First time in a new Colab session:
    python unified.py bootstrap

    # Individual stages:
    python unified.py describe --dataset oxford_pet
    python unified.py zeroshot --dataset oxford_pet
    python unified.py train --dataset oxford_pet --shots 16
    python unified.py eval --dataset oxford_pet --adapter_path adapters_oxford_pet_16shot.pt
    python unified.py predict --image_path photo.jpg --dataset oxford_pet --adapter_path adapters_oxford_pet_16shot.pt

    # Full pipeline for a dataset, start to finish (describe -> zeroshot -> train -> eval):
    python unified.py pipeline --dataset oxford_pet --shots 16
"""

import argparse
import subprocess
import sys


def run(cmd: list, description: str):
    print(f"\n{'='*70}\n{description}\n{'='*70}")
    print(f"$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[FAILED] {description} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)
    return result


def cmd_bootstrap(args):
    run(["python", "bootstrap_session.py"], "Bootstrapping session (restore repo/data/caches/checkpoints)")


def cmd_describe(args):
    run(
        ["python", "bifta_dr.py", "--dataset_name", args.dataset, "--epsilon", str(args.epsilon), "--top_k", str(args.top_k)],
        f"Generating BiFTA-DR descriptions for '{args.dataset}'",
    )


def cmd_zeroshot(args):
    enable_fusion = getattr(args, "enable_fusion", False)
    cmd = ["python", "main.py", "--dataset_name", args.dataset, "--backbone", "openai"]
    if enable_fusion:
        cmd.append("--enable_fusion")
    run(cmd, f"Running zero-shot baseline scoring for '{args.dataset}'" + (" (with LaZSL/fusion)" if enable_fusion else ""))


def cmd_train(args):
    run(
        [
            "python", "goal_adapter_train.py",
            "--dataset_name", args.dataset,
            "--shots", str(args.shots),
            "--val_shots", str(args.val_shots),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
        ],
        f"Training GOAL adapter for '{args.dataset}' ({args.shots}-shot)",
    )


def cmd_eval(args):
    adapter_path = args.adapter_path or f"adapters_{args.dataset}_{args.shots}shot.pt"
    run(
        [
            "python", "goal_adapter_eval.py",
            "--dataset_name", args.dataset,
            "--adapter_path", adapter_path,
            "--descriptions_method", args.descriptions_method,
            "--fixed_k", str(args.fixed_k),
        ],
        f"Evaluating trained adapter for '{args.dataset}' on held-out test split",
    )


def cmd_predict(args):
    run(
        [
            "python", "predict_unified.py",
            "--image_path", args.image_path,
            "--dataset_name", args.dataset,
            "--adapter_path", args.adapter_path,
            "--descriptions_method", args.descriptions_method,
            "--fixed_k", str(args.fixed_k),
        ],
        f"Running unified pipeline prediction on {args.image_path}",
    )


def cmd_pipeline(args):
    """Full pipeline for one dataset, start to finish: describe -> zeroshot sanity check -> train -> eval."""
    cmd_describe(args)
    cmd_zeroshot(args)
    cmd_train(args)
    args.adapter_path = f"adapters_{args.dataset}_{args.shots}shot.pt"
    cmd_eval(args)
    print(f"\n{'='*70}\nFull pipeline complete for '{args.dataset}'.\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified entry point for the whole framework.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bootstrap = sub.add_parser("bootstrap", help="Restore repo/data/caches/checkpoints from GitHub+Drive")
    p_bootstrap.set_defaults(func=cmd_bootstrap)

    p_describe = sub.add_parser("describe", help="Generate BiFTA-DR descriptions for a dataset")
    p_describe.add_argument("--dataset", required=True)
    p_describe.add_argument("--epsilon", type=float, default=0.99)
    p_describe.add_argument("--top_k", type=int, default=30)
    p_describe.set_defaults(func=cmd_describe)

    p_zeroshot = sub.add_parser("zeroshot", help="Run zero-shot baseline scoring for a dataset")
    p_zeroshot.add_argument("--dataset", required=True)
    p_zeroshot.add_argument("--enable_fusion", action="store_true",
                             help="Also compute LaZSL scores + WCA/LaZSL agreement diagnostic")
    p_zeroshot.set_defaults(func=cmd_zeroshot)

    p_train = sub.add_parser("train", help="Train the GOAL adapter for a dataset")
    p_train.add_argument("--dataset", required=True)
    p_train.add_argument("--shots", type=int, default=16)
    p_train.add_argument("--val_shots", type=int, default=4)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--patience", type=int, default=5)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate a trained adapter on the held-out test split")
    p_eval.add_argument("--dataset", required=True)
    p_eval.add_argument("--shots", type=int, default=16, help="Used to infer default adapter_path if not given")
    p_eval.add_argument("--adapter_path", default=None)
    p_eval.add_argument("--descriptions_method", default="bifta-dr")
    p_eval.add_argument("--fixed_k", type=int, default=30)
    p_eval.set_defaults(func=cmd_eval)

    p_predict = sub.add_parser("predict", help="Run the full unified pipeline on a single image")
    p_predict.add_argument("--image_path", required=True)
    p_predict.add_argument("--dataset", required=True)
    p_predict.add_argument("--adapter_path", required=True)
    p_predict.add_argument("--descriptions_method", default="bifta-dr")
    p_predict.add_argument("--fixed_k", type=int, default=30)
    p_predict.set_defaults(func=cmd_predict)

    p_pipeline = sub.add_parser("pipeline", help="Run the full pipeline for a dataset: describe -> zeroshot -> train -> eval")
    p_pipeline.add_argument("--dataset", required=True)
    p_pipeline.add_argument("--epsilon", type=float, default=0.99)
    p_pipeline.add_argument("--top_k", type=int, default=30)
    p_pipeline.add_argument("--shots", type=int, default=16)
    p_pipeline.add_argument("--val_shots", type=int, default=4)
    p_pipeline.add_argument("--epochs", type=int, default=30)
    p_pipeline.add_argument("--patience", type=int, default=5)
    p_pipeline.add_argument("--descriptions_method", default="bifta-dr")
    p_pipeline.add_argument("--fixed_k", type=int, default=30)
    p_pipeline.add_argument("--enable_fusion", action="store_true",
                             help="Also compute LaZSL scores + WCA/LaZSL agreement diagnostic during the zeroshot stage")
    p_pipeline.set_defaults(func=cmd_pipeline)

    args = parser.parse_args()
    args.func(args)
