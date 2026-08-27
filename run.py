"""
run.py - Single entrypoint for Smart Fridge Food Classifier

Examples:
  Train:
    python3 run.py train --extra-data data/raw --noise-data data/noise_examples

  Inference (assignment-format output only):
    python3 run.py infer ./datasets/FoodTest1

  Inference + precision/recall + confusion matrix plots:
    python3 run.py infer ./datasets/FoodTest1 --metrics --metrics-out metrics
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import train
import inference


def _add_train_subcommand(subparsers):
    p = subparsers.add_parser(
        "train",
        help="Train a model (wraps train.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="datasets/FoodTest1")
    p.add_argument("--extra-data", default=None)
    p.add_argument("--noise-data", default=None)
    p.add_argument("--output-dir", default="model")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--phase1-epochs", type=int, default=10)
    p.add_argument("--phase2-epochs", type=int, default=40)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.set_defaults(_cmd="train")


def _add_infer_subcommand(subparsers):
    p = subparsers.add_parser(
        "infer",
        help="Run inference (wraps inference.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dataset_folder", help="Flat dataset folder (FoodTest1-style)")
    p.add_argument("--model-path", default=inference.DEFAULT_MODEL_PATH)
    p.add_argument("--metrics", action="store_true")
    p.add_argument("--metrics-out", default="metrics")
    p.add_argument("--show-plots", action="store_true")
    p.set_defaults(_cmd="infer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Fridge Food Classifier — single entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_subcommand(subparsers)
    _add_infer_subcommand(subparsers)
    return parser.parse_args()


def main():
    args = parse_args()

    if args._cmd == "train":
        train_args = SimpleNamespace(
            data_dir=args.data_dir,
            extra_data=args.extra_data,
            noise_data=args.noise_data,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            val_split=args.val_split,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
        )
        train.run_training(train_args)
        return

    if args._cmd == "infer":
        class_total, class_correct, folder, y_true, y_pred = inference.run_inference(
            args.dataset_folder,
            model_path=args.model_path,
        )
        inference.print_results(class_total, class_correct, folder)
        if args.metrics:
            paths = inference.run_metrics_and_plots(
                y_true=y_true,
                y_pred=y_pred,
                dataset_folder=folder,
                out_dir=args.metrics_out,
                show_plots=args.show_plots,
            )
            print("\nSaved metrics:")
            print(f"  precision/recall CSV: {paths['csv']}")
            print(f"  confusion matrix (counts): {paths['cm_counts']}")
            print(f"  confusion matrix (normalized): {paths['cm_norm']}")
        return

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()

