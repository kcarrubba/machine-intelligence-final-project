"""
train.py - Training script for Smart Fridge Food Classifier

COMP3330/6380 Assessment 1, Semester 1 2026

Trains an EfficientNet-B0 model on 40 food item classes using transfer learning
and data augmentation. Supports loading images from:
  - FoodTest1/  (flat folder, filenames like 00_Asparagus_001.JPG)
  - data/raw/   (structured folder, subdirs like 00_Asparagus/)
  - Noise examples from Canvas (same naming as FoodTest1)

Usage:
    # Basic (FoodTest1 only):
    python train.py

    # With extra downloaded data:
    python train.py --extra-data data/raw

    # With noise examples included:
    python train.py --extra-data data/raw --noise-data data/noise_examples

    # Custom options:
    python train.py --batch-size 32 --phase1-epochs 15 --phase2-epochs 40
"""

import os
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

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

NUM_CLASSES = len(CLASS_NAMES)
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────── Dataset ──────────────────────────────────────────

class FoodDataset(Dataset):
    """
    Dataset for 40-class food image classification.

    Supports two image layouts:
      1. Flat directory: all images in one folder named like NN_ClassName_XXX.jpg
         (e.g., FoodTest1/, noise example folders)
      2. Structured directory: one subdirectory per class named NN_ClassName/
         (e.g., data/raw/ produced by download_data.py)

    In both cases the integer class label is read from the two-digit numeric
    prefix at the start of the filename (or subdirectory name).
    """

    def __init__(self, samples: list, transform=None):
        """
        Args:
            samples:   list of (image_path_str, label_int) tuples
            transform: torchvision transform to apply to each image
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    # ── Label parsing helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_label(name: str) -> int:
        """
        Parse class label from a filename or directory name.
        '00_Asparagus_001.JPG' → 0
        '00_Asparagus'         → 0
        """
        prefix = Path(name).stem.split("_")[0]
        return int(prefix)

    # ── Factory class methods ──────────────────────────────────────────────────

    @classmethod
    def from_flat_dir(cls, directory, transform=None):
        """
        Load samples from a flat folder where files are named NN_ClassName_XXX.jpg.
        Files that don't match the NN_ prefix pattern are silently skipped.
        """
        directory = Path(directory)
        samples = []
        for f in sorted(directory.iterdir()):
            if f.suffix.lower() not in IMG_EXTENSIONS:
                continue
            try:
                label = cls._parse_label(f.name)
                if 0 <= label < NUM_CLASSES:
                    samples.append((str(f), label))
            except (ValueError, IndexError):
                continue
        return cls(samples, transform)

    @classmethod
    def from_structured_dir(cls, directory, transform=None):
        """
        Load samples from a structured folder where each subdirectory is named
        NN_ClassName/ and contains the corresponding class images.
        """
        directory = Path(directory)
        samples = []
        for subdir in sorted(directory.iterdir()):
            if not subdir.is_dir():
                continue
            try:
                label = cls._parse_label(subdir.name)
                if not (0 <= label < NUM_CLASSES):
                    continue
            except (ValueError, IndexError):
                continue
            for f in sorted(subdir.iterdir()):
                if f.suffix.lower() in IMG_EXTENSIONS:
                    samples.append((str(f), label))
        return cls(samples, transform)

    @classmethod
    def combined(cls, flat_dirs=None, structured_dirs=None, transform=None):
        """
        Combine samples from any number of flat and/or structured directories.
        Each directory is checked for existence before loading.
        """
        all_samples = []
        for d in (flat_dirs or []):
            if Path(d).exists():
                tmp = cls.from_flat_dir(d)
                all_samples.extend(tmp.samples)
                print(f"  Loaded {len(tmp.samples):>4} images from flat dir:       {d}")
            else:
                print(f"  SKIPPED (not found): {d}")
        for d in (structured_dirs or []):
            if Path(d).exists():
                tmp = cls.from_structured_dir(d)
                all_samples.extend(tmp.samples)
                print(f"  Loaded {len(tmp.samples):>4} images from structured dir:  {d}")
            else:
                print(f"  SKIPPED (not found): {d}")
        return cls(all_samples, transform)


# ─────────────────────────── Transforms ───────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """
    Augmentation pipeline for training.

    Includes noise-robustness techniques:
    - GaussianBlur: simulates camera blur and low-frequency noise (Noise1/2/3)
    - ColorJitter:  simulates lighting variation and colour shifts
    - RandomErasing: simulates partial occlusion
    - RandomResizedCrop: simulates different zoom levels and framing
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(
            brightness=0.35,
            contrast=0.35,
            saturation=0.35,
            hue=0.08,
        ),
        # Apply blur with 40% probability to build noise robustness
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))],
            p=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # Randomly erase small patches (simulates occlusion / noise artefacts)
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def get_val_transforms() -> transforms.Compose:
    """Deterministic transforms for validation — no augmentation."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ─────────────────────────── Model ────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = True) -> nn.Module:
    """
    EfficientNet-B0 with ImageNet weights, custom 40-class head.

    Phase 1 (freeze_backbone=True):  only the new classifier is trainable.
    Phase 2 (freeze_backbone=False): entire network is fine-tuned end-to-end.
    """
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final linear layer (1280 → num_classes)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─────────────────────────── Utilities ────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_distribution(samples: list) -> dict:
    dist = defaultdict(int)
    for _, label in samples:
        dist[label] += 1
    return dist


def stratified_split(samples: list, val_fraction: float, seed: int = 42):
    """
    Split samples into train/val while keeping per-class ratios approximately
    equal. For classes with very few samples (< 2) all go to train.
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)

    train_samples, val_samples = [], []
    for label in sorted(by_class):
        items = by_class[label][:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction)) if len(items) >= 4 else 0
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


# ─────────────────────────── Training Loops ───────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler=None
) -> tuple:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, leave=False, desc="  train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────── Main Training ────────────────────────────────────

def run_training(args):
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pth"

    # ── Load all data ──────────────────────────────────────────────────────────
    print("\nLoading data...")
    flat_dirs = [args.data_dir]
    if args.noise_data:
        flat_dirs.append(args.noise_data)

    structured_dirs = []
    if args.extra_data:
        structured_dirs.append(args.extra_data)

    full_dataset = FoodDataset.combined(
        flat_dirs=flat_dirs,
        structured_dirs=structured_dirs,
    )

    if len(full_dataset) == 0:
        raise RuntimeError(
            f"No images found. Check that --data-dir ({args.data_dir}) exists "
            "and contains images with the NN_ClassName_XXX naming convention."
        )

    print(f"\nTotal samples: {len(full_dataset)}")
    dist = class_distribution(full_dataset.samples)
    classes_covered = len([c for c in dist if dist[c] > 0])
    print(f"Classes covered: {classes_covered}/{NUM_CLASSES}")
    if classes_covered < NUM_CLASSES:
        missing = [f"{i}_{CLASS_NAMES[i]}" for i in range(NUM_CLASSES)
                   if dist.get(i, 0) == 0]
        print(f"WARNING — {NUM_CLASSES - classes_covered} classes have 0 samples:")
        print("  " + ", ".join(missing))
        print("  Run download_data.py to get additional training images.\n")

    # ── Train / val split ─────────────────────────────────────────────────────
    train_samples, val_samples = stratified_split(
        full_dataset.samples, val_fraction=args.val_split
    )

    train_set = FoodDataset(train_samples, get_train_transforms())
    val_set   = FoodDataset(val_samples,   get_val_transforms())
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    # num_workers=0 is safe across all platforms; increase if training is slow
    n_workers = min(4, os.cpu_count() or 1) if device.type != "mps" else 0
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=(device.type == "cuda"),
    )

    # ── Model, loss, AMP ──────────────────────────────────────────────────────
    model = build_model(freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_acc = 0.0

    # ── Phase 1: train classifier head only ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 1: Training classifier head ({args.phase1_epochs} epochs)")
    print(f"{'='*60}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.phase1_epochs, eta_min=1e-5
    )

    for epoch in range(1, args.phase1_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        marker = " *" if val_acc > best_val_acc else ""
        print(
            f"  Epoch {epoch:3d}/{args.phase1_epochs}"
            f" | Train Loss {train_loss:.4f}  Acc {train_acc:.4f}"
            f" | Val Loss {val_loss:.4f}  Acc {val_acc:.4f}"
            f"{marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    # ── Phase 2: fine-tune entire network ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 2: Fine-tuning full network ({args.phase2_epochs} epochs)")
    print(f"{'='*60}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": args.backbone_lr},
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.phase2_epochs, eta_min=1e-6
    )

    for epoch in range(1, args.phase2_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        marker = " *" if val_acc > best_val_acc else ""
        print(
            f"  Epoch {epoch:3d}/{args.phase2_epochs}"
            f" | Train Loss {train_loss:.4f}  Acc {train_acc:.4f}"
            f" | Val Loss {val_loss:.4f}  Acc {val_acc:.4f}"
            f"{marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to:      {best_model_path}")


# ─────────────────────────── CLI ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Smart Fridge Food Classifier (EfficientNet-B0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="datasets/FoodTest1",
        help="Path to FoodTest1 flat image folder",
    )
    parser.add_argument(
        "--extra-data", default=None,
        help="Path to additional structured data folder (data/raw/)",
    )
    parser.add_argument(
        "--noise-data", default=None,
        help="Path to noise example flat folder from Canvas (data/noise_examples/)",
    )
    parser.add_argument(
        "--output-dir", default="model",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--batch-size",      type=int,   default=16)
    parser.add_argument("--phase1-epochs",   type=int,   default=10,
                        help="Epochs to train the classifier head (frozen backbone)")
    parser.add_argument("--phase2-epochs",   type=int,   default=40,
                        help="Epochs to fine-tune the full network")
    parser.add_argument("--val-split",       type=float, default=0.2,
                        help="Fraction of data to reserve for validation")
    parser.add_argument("--backbone-lr",     type=float, default=1e-4,
                        help="Learning rate for backbone during phase 2")
    parser.add_argument("--head-lr",         type=float, default=5e-4,
                        help="Learning rate for classifier head during phase 2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
