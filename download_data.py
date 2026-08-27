"""
download_data.py - Prepare data in PyTorch ImageFolder format

This script prepares datasets into a folder-per-class structure compatible with
`torchvision.datasets.ImageFolder` / `DatasetFolder`.

Usage:
    # Convert FoodTest1 (flat) to ImageFolder-style structure:
    python download_data.py --input datasets/FoodTest1 --output data/raw

    # Only print a summary of an existing structured folder:
    python download_data.py --summary --output data/raw

Requirements (install before running):
    pip install torchvision   (already in requirements.txt)

Output structure:
    data/raw/
        00_Asparagus/
            image_000.jpg
            image_001.jpg
            ...
        01_Carrotts/
        ...
        39_Cheese/
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Optional

# ─────────────────────────── Class Definitions ────────────────────────────────

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

# Food-101 class names that map to our classes (Food-101 → our class index)
FOOD101_MAPPING = {
    "oysters":          2,   # Oysters
    "grilled_salmon":   4,   # Salmon (partial match)
    "pork_chop":        3,   # Pork (partial match)
    "steak":            26,  # Steak
    "prime_rib":        26,  # Steak (additional)
    "filet_mignon":     26,  # Steak (additional)
    "chicken_wings":    27,  # Chicken (partial)
    "strawberry_shortcake": 6,  # Strawberries (partial)
    "deviled_eggs":     35,  # Eggs (partial)
    "eggs_benedict":    35,  # Eggs (partial)
    "guacamole":        15,  # Avocado (partial)
}


# ─────────────────────────── ImageFolder Prep ─────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_images_in_dir(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)

def _parse_label_from_filename(filename: str) -> Optional[int]:
    """
    Parse the two-digit class prefix from filenames like:
      00_Asparagus_001.JPG -> 0
    Returns None if it doesn't match.
    """
    stem = Path(filename).stem
    prefix = stem.split("_")[0]
    if not prefix.isdigit():
        return None
    label = int(prefix)
    if 0 <= label < len(CLASS_NAMES):
        return label
    return None


def prepare_imagefolder(input_dir: Path, output_dir: Path, copy_files: bool = True) -> int:
    """
    Convert a flat folder (FoodTest1-style) into ImageFolder structure:
      output_dir/00_Asparagus/*.jpg
      output_dir/01_Carrotts/*.jpg
      ...
    Returns the number of images copied/moved.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    n_done = 0
    n_skipped = 0
    for f in sorted(input_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue

        label = _parse_label_from_filename(f.name)
        if label is None:
            n_skipped += 1
            continue

        folder_name = f"{label:02d}_{CLASS_NAMES[label]}"
        dest_dir = output_dir / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / f.name
        if dest_path.exists():
            # Keep existing file; don't overwrite.
            continue

        if copy_files:
            shutil.copy2(f, dest_path)
        else:
            shutil.move(str(f), str(dest_path))
        n_done += 1

    print(f"\nPrepared ImageFolder data at: {output_dir}")
    print(f"  Images processed: {n_done}")
    if n_skipped > 0:
        print(f"  Skipped (didn't match NN_ prefix): {n_skipped}")
    return n_done


# ─────────────────────────── Food-101 Extraction ──────────────────────────────

def extract_food101(output_dir: Path, food101_root: str = "data/food101"):
    """
    Download Food-101 via torchvision and copy matching classes to output_dir.
    Note: Food-101 is ~5 GB and has limited overlap with our 40 classes.
    """
    try:
        import torchvision.datasets as tvd
    except ImportError:
        print("ERROR: torchvision not installed.")
        return

    food101_path = Path(food101_root)
    food101_path.mkdir(parents=True, exist_ok=True)

    print("Downloading Food-101 dataset (this may take a while, ~5 GB)...")
    dataset = tvd.Food101(root=str(food101_path), split="train", download=True)

    # Build a mapping from Food-101 class name to image paths
    food101_classes = dataset.classes  # list of class name strings
    food101_class_to_idx = {c: i for i, c in enumerate(food101_classes)}

    # Group sample paths by Food-101 class name
    from collections import defaultdict
    food101_samples = defaultdict(list)
    for img_path, label_idx in zip(dataset._image_files, dataset._labels):
        food101_samples[food101_classes[label_idx]].append(img_path)

    copied_total = 0
    for food101_class, our_class_idx in FOOD101_MAPPING.items():
        if food101_class not in food101_samples:
            print(f"  WARNING: Food-101 class '{food101_class}' not found, skipping.")
            continue

        our_class_name = CLASS_NAMES[our_class_idx]
        dest_dir = output_dir / f"{our_class_idx:02d}_{our_class_name}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        images = food101_samples[food101_class]
        print(f"  Copying {len(images)} images: {food101_class} → {our_class_name}")
        for src in images:
            dst = dest_dir / Path(src).name
            if not dst.exists():
                shutil.copy2(src, dst)
                copied_total += 1

    print(f"\nFood-101 extraction complete. Copied {copied_total} images to {output_dir}")


# ─────────────────────────── Summary ─────────────────────────────────────────

def print_summary(output_dir: Path):
    """Print a count of downloaded images per class."""
    print(f"\n{'='*55}")
    print(f"  Data Summary: {output_dir}")
    print(f"{'='*55}")
    total = 0
    missing = []
    for i, name in enumerate(CLASS_NAMES):
        folder = output_dir / f"{i:02d}_{name}"
        if folder.exists():
            imgs = [f for f in folder.iterdir()
                    if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
            count = len(imgs)
        else:
            count = 0
        status = "OK" if count >= 20 else ("LOW" if count > 0 else "MISSING")
        print(f"  {i:02d}_{name:<18} {count:>4} images  [{status}]")
        total += count
        if count == 0:
            missing.append(f"{i:02d}_{name}")
    print(f"{'='*55}")
    print(f"  Total: {total} images across {len(CLASS_NAMES)} classes")
    if missing:
        print(f"\n  Classes with 0 images ({len(missing)}):")
        for m in missing:
            print(f"    - {m}")
        print("\n  Tip: Run download_data.py to prepare structured folders from a flat dataset.")


# ─────────────────────────── CLI ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare data in PyTorch ImageFolder format"
    )
    parser.add_argument(
        "--input", default="datasets/FoodTest1",
        help="Input directory (default: datasets/FoodTest1)"
    )
    parser.add_argument(
        "--output", default="data/raw",
        help="Output directory (ImageFolder structure) (default: data/raw)"
    )
    parser.add_argument(
        "--move", action="store_true",
        help="Move files instead of copying (default: copy)"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Only print a summary of existing downloaded data, then exit"
    )
    parser.add_argument(
        "--include-food101", action="store_true",
        help="Also download Food-101 and extract overlapping classes (optional, large download)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summary:
        print_summary(output_dir)
        return

    prepare_imagefolder(Path(args.input), output_dir, copy_files=(not args.move))

    if args.include_food101:
        extract_food101(output_dir)

    print_summary(output_dir)


if __name__ == "__main__":
    main()
