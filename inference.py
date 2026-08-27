"""
inference.py - Inference script for Smart Fridge Food Classifier

COMP3330/6380 Assessment 1, Semester 1 2026

Loads images from a flat dataset folder (same naming convention as FoodTest1),
runs inference using a trained EfficientNet-B0 model, and prints per-class and
overall accuracy in the required format.

Usage:
    python inference.py <dataset_folder>
    python inference.py ./FoodTest1
    python inference.py ./FoodTest1 --model-path model/best_model.pth

Expected output format:
    Dataset: ./FoodTest1
    Class              Samples  Correct   Accuracy
    0_Asparagus              3        2     66.67%
    ...
    39_Cheese               11        9     81.82%
    -----------------------------------------------
    Mean Class Acc:  72.24%
    Overall Acc:     78.57%
"""

import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Optional metrics/plots (enabled via CLI flags)
try:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
except Exception:  # pragma: no cover
    confusion_matrix = None
    precision_recall_fscore_support = None

try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None

# ─────────────────────────── Constants ────────────────────────────────────────

CLASS_NAMES = [
    "Asparagus", "Carrotts", "Oysters", "Pork", "Salmon",
    "Zuccini", "Strawberries", "Sausages", "Garlic", "Ginger",
    "Cauliflower", "Capsicum", "Pumpkin", "Rockmelon", "Watermelon",
    "Avocado", "Tomato", "Pineapple", "Pears", "Apples",
    "Peach", "Trout", "Snapper", "Barra", "Prawns",
    "TropicalFish", "Steak", "Chicken", "Lamb", "Mushrooms",
    "RedOnion", "Tortellini", "Blueberries", "Lettuce", "Milk",
    "Eggs", "Juice", "Kiwi", "Butter", "Cheese",
]

NUM_CLASSES    = len(CLASS_NAMES)
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# Default model path relative to this script's location
_SCRIPT_DIR        = Path(__file__).parent
DEFAULT_MODEL_PATH = str(_SCRIPT_DIR / "model" / "best_model.pth")


# ─────────────────────────── Model ────────────────────────────────────────────

def _build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Rebuild the same EfficientNet-B0 architecture used during training."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load trained weights and put model in eval mode."""
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        print(
            "Train a model first with:  python train.py",
            file=sys.stderr,
        )
        sys.exit(1)

    model = _build_model()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────── Transforms ───────────────────────────────────────

def get_inference_transforms() -> transforms.Compose:
    """Deterministic transforms — identical to the validation transforms in train.py."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ─────────────────────────── Data loading ─────────────────────────────────────

def _parse_label(filename: str) -> int:
    """
    Extract integer class label from a filename.
    '00_Asparagus_001.JPG' → 0
    '39_Cheese_007.JPG'    → 39
    """
    stem = Path(filename).stem      # '00_Asparagus_001'
    prefix = stem.split("_")[0]     # '00'
    return int(prefix)


def load_dataset(folder: str) -> list:
    """
    Return sorted list of (image_path, true_label) from a flat dataset folder.
    Files whose names do not start with a valid two-digit class prefix are skipped.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"ERROR: Dataset folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    samples = []
    for f in sorted(folder_path.iterdir()):
        if f.suffix.lower() not in IMG_EXTENSIONS:
            continue
        try:
            label = _parse_label(f.name)
            if 0 <= label < NUM_CLASSES:
                samples.append((f, label))
        except (ValueError, IndexError):
            continue

    return samples


# ─────────────────────────── Inference ────────────────────────────────────────

@torch.no_grad()
def run_inference(
    dataset_folder: str,
    model_path: str = DEFAULT_MODEL_PATH,
    device: torch.device = None,
) -> tuple:
    """
    Run inference on all images in dataset_folder.

    Returns:
        class_total   : dict {class_idx: int}  — number of samples per class
        class_correct : dict {class_idx: int}  — number of correct predictions
        dataset_folder: str                    — echoed back for output formatting
        y_true        : list[int]              — true labels (for metrics)
        y_pred        : list[int]              — predicted labels (for metrics)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model     = load_model(model_path, device)
    transform = get_inference_transforms()
    samples   = load_dataset(dataset_folder)

    if not samples:
        print(
            f"ERROR: No valid images found in '{dataset_folder}'.\n"
            "Images must be named like '00_Asparagus_001.JPG'.",
            file=sys.stderr,
        )
        sys.exit(1)

    class_total   = defaultdict(int)
    class_correct = defaultdict(int)
    y_true, y_pred = [], []

    for img_path, true_label in samples:
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"WARNING: Could not open {img_path}: {e}", file=sys.stderr)
            continue

        tensor = transform(image).unsqueeze(0).to(device)
        output = model(tensor)
        pred   = output.argmax(dim=1).item()

        class_total[true_label]   += 1
        if pred == true_label:
            class_correct[true_label] += 1
        y_true.append(true_label)
        y_pred.append(pred)

    return class_total, class_correct, dataset_folder, y_true, y_pred


def _require_metrics_deps():
    missing = []
    if confusion_matrix is None or precision_recall_fscore_support is None:
        missing.append("scikit-learn")
    if matplotlib is None or plt is None:
        missing.append("matplotlib")
    if missing:
        raise RuntimeError(
            "Metrics/plots requested but dependencies are missing: "
            + ", ".join(missing)
            + ". Install with: pip install -r requirements.txt"
        )


def _save_precision_recall_table(
    y_true: list,
    y_pred: list,
    out_dir: Path,
) -> Path:
    """
    Saves a CSV with per-class precision/recall/F1/support.
    Precision is the key false-positive detector: low precision => many FP for that class.
    """
    _require_metrics_deps()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(range(NUM_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    csv_path = out_dir / "precision_recall_f1.csv"
    lines = ["class_idx,class_name,precision,recall,f1,support"]
    for i in labels:
        lines.append(
            f"{i},{CLASS_NAMES[i]},{precision[i]:.6f},{recall[i]:.6f},{f1[i]:.6f},{int(support[i])}"
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def _save_confusion_matrix_plot(
    y_true: list,
    y_pred: list,
    out_dir: Path,
    normalize: bool,
    title: str,
) -> Path:
    _require_metrics_deps()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0)
        data = cm_norm
        fmt = ".2f"
        suffix = "normalized"
    else:
        data = cm
        fmt = "d"
        suffix = "counts"

    # Ensure headless save works reliably.
    matplotlib.use("Agg", force=True)

    fig_w = max(12, int(NUM_CLASSES * 0.35))
    fig_h = max(10, int(NUM_CLASSES * 0.32))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    im = ax.imshow(data, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Ticks: use "idx_Name" to match assignment convention
    tick_labels = [f"{i}_{CLASS_NAMES[i]}" for i in labels]
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)

    # Annotate cells only when matrix is not too dense on screen.
    annotate = NUM_CLASSES <= 40  # current project = 40 classes
    if annotate:
        thresh = (data.max() / 2.0) if data.size else 0.0
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = data[i, j]
                text = format(val, fmt) if not normalize else f"{val:{fmt}}"
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    fontsize=5,
                    color="white" if val > thresh else "black",
                )

    fig.tight_layout()
    out_path = out_dir / f"confusion_matrix_{suffix}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_metrics_and_plots(
    y_true: list,
    y_pred: list,
    dataset_folder: str,
    out_dir: str,
    show_plots: bool = False,
) -> dict:
    """
    Computes precision/recall/F1 and saves:
      - precision_recall_f1.csv
      - confusion_matrix_counts.png
      - confusion_matrix_normalized.png
    """
    out_dir_p = Path(out_dir)

    csv_path = _save_precision_recall_table(y_true, y_pred, out_dir_p)
    cm_counts = _save_confusion_matrix_plot(
        y_true, y_pred, out_dir_p, normalize=False,
        title=f"Confusion Matrix (counts) — {dataset_folder}",
    )
    cm_norm = _save_confusion_matrix_plot(
        y_true, y_pred, out_dir_p, normalize=True,
        title=f"Confusion Matrix (row-normalized) — {dataset_folder}",
    )

    if show_plots:
        # Switch to interactive backend if available; fall back gracefully.
        try:
            matplotlib.use("TkAgg", force=True)
            for p in [cm_counts, cm_norm]:
                img = plt.imread(p)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis("off")
                plt.title(p.name)
            plt.show()
        except Exception:
            pass

    return {"csv": str(csv_path), "cm_counts": str(cm_counts), "cm_norm": str(cm_norm)}


# ─────────────────────────── Output ───────────────────────────────────────────

def print_results(
    class_total: dict,
    class_correct: dict,
    dataset_folder: str,
) -> None:
    """Print accuracy results in the format required by the assignment spec."""

    # Build column-header row
    col_class    = "Class"
    col_samples  = "Samples"
    col_correct  = "Correct"
    col_accuracy = "Accuracy"

    # Width of the class column — wide enough for the longest label
    class_col_w = max(
        max(len(f"{i}_{CLASS_NAMES[i]}") for i in range(NUM_CLASSES)),
        len(col_class),
    )

    sep_width = class_col_w + 2 + 7 + 2 + 7 + 2 + 9

    print(f"\nDataset: {dataset_folder}")
    print(
        f"{col_class:<{class_col_w}}  "
        f"{col_samples:>7}  {col_correct:>7}  {col_accuracy:>9}"
    )
    print("-" * sep_width)

    grand_total   = 0
    grand_correct = 0
    class_accs    = []  # per-class accuracy (only for classes with ≥1 sample)

    for i in range(NUM_CLASSES):
        label_str = f"{i}_{CLASS_NAMES[i]}"
        n_samples = class_total.get(i, 0)
        n_correct = class_correct.get(i, 0)

        if n_samples > 0:
            acc = n_correct / n_samples * 100.0
            class_accs.append(acc)
            acc_str = f"{acc:.2f}%"
        else:
            acc_str = "N/A"

        print(
            f"{label_str:<{class_col_w}}  "
            f"{n_samples:>7}  {n_correct:>7}  {acc_str:>9}"
        )

        grand_total   += n_samples
        grand_correct += n_correct

    print("-" * sep_width)

    mean_class_acc = sum(class_accs) / len(class_accs) if class_accs else 0.0
    overall_acc    = (grand_correct / grand_total * 100.0) if grand_total > 0 else 0.0

    print(f"Mean Class Acc: {mean_class_acc:.2f}%")
    print(f"Overall Acc:    {overall_acc:.2f}%")


# ─────────────────────────── CLI ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Fridge Food Classifier — Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_folder",
        help="Path to dataset folder (flat, same structure as FoodTest1)",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model weights (.pth file)",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Also compute precision/recall and save confusion matrix plots",
    )
    parser.add_argument(
        "--metrics-out",
        default="metrics",
        help="Output directory for metrics CSV/plots (only used with --metrics)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Attempt to open confusion matrix plots in a window (optional)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    class_total, class_correct, folder, y_true, y_pred = run_inference(
        args.dataset_folder,
        model_path=args.model_path,
    )
    print_results(class_total, class_correct, folder)

    if args.metrics:
        paths = run_metrics_and_plots(
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
