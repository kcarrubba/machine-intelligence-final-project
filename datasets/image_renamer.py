"""
IMAGE RENAMER SCRIPT – HOW TO USE

WHAT THIS SCRIPT DOES:
----------------------
This script renames image files in a single food-class folder so that they match
the naming convention of the original dataset.

Expected naming format:
    itemNumber_itemName_photoNumber.EXT

Example:
    00_Asparagus_001.JPG
    00_Asparagus_002.JPEG

The script:
- Detects files that already match the correct format (these are treated as ORIGINALS)
- Leaves those original files unchanged
- Finds the highest existing photo number among originals
- Renames all other images in the folder starting from (highest + 1)
- Preserves each file’s original extension (e.g., .JPG vs .JPEG)
- Does NOT fill in missing numbers (e.g., 001, 003, 007 → next is 008)
- Performs safety checks to prevent overwriting files


WHERE TO PLACE THIS FILE:
-------------------------
Place this script (image_renamer.py) in the SAME directory that contains
all your food folders.

Example structure:

FOODTEST1_EXPANDED/
│
├── 0asparagus/
├── 1carrot/
├── 2oyster/
├── ...
├── 39cheese/
│
└── image_renamer.py   ← place the script here


HOW TO RUN THE SCRIPT:
----------------------

1. Open a terminal in the folder where this script is located.

2. Run the script and pass in ONE folder at a time:

   Mac/Linux:
       python3 image_renamer.py "0asparagus"

   Windows:
       python image_renamer.py "0asparagus"

   (Use quotes if the folder name has spaces.)

3. The script will:
   - Show you a preview of all renaming changes
   - Ask for confirmation before making any changes

4. Type:
       y
   to proceed, or anything else to cancel.


IMPORTANT NOTES:
----------------
- Run this script ONE folder at a time.
- Each folder should contain ONLY ONE food class.
- There must be at least one correctly named file in the folder
  (from the original dataset) so the script knows the naming format.
- All files in the folder are assumed to be images.
- The script will STOP if it detects any potential overwrite or conflict.

OPTIONAL TIP:
--------------
You can drag and drop a folder path into the terminal instead of typing it manually.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Matches names like:
# 00_Asparagus_001.JPG
# 01_Carrot_007.JPEG
PATTERN = re.compile(r"^(\d{2})_(.+)_(\d{3})$")

# Treat every file in the folder as a photo file, as requested.
# Hidden files are skipped automatically.
def get_files(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")],
        key=lambda p: p.name.lower()
    )


def parse_original_file(file_path: Path) -> Tuple[str, str, int] | None:
    """
    Returns (item_number, item_name, photo_number) if the file stem matches
    the original dataset naming convention. Otherwise returns None.
    """
    match = PATTERN.match(file_path.stem)
    if not match:
        return None

    item_number, item_name, photo_number = match.groups()
    return item_number, item_name, int(photo_number)


def build_rename_plan(folder: Path) -> List[Tuple[Path, Path]]:
    files = get_files(folder)

    originals: List[Tuple[Path, str, str, int]] = []
    non_originals: List[Path] = []

    for file_path in files:
        parsed = parse_original_file(file_path)
        if parsed is None:
            non_originals.append(file_path)
        else:
            item_number, item_name, photo_number = parsed
            originals.append((file_path, item_number, item_name, photo_number))

    if not originals:
        raise ValueError(
            "No correctly named original files were found in this folder. "
            "At least one file must already match the format like 00_Asparagus_001.JPG."
        )

    # Make sure all originals agree on the same item number + item name.
    prefixes = {(item_number, item_name) for _, item_number, item_name, _ in originals}
    if len(prefixes) != 1:
        raise ValueError(
            "Multiple different original naming prefixes were found in this folder. "
            "Each folder should contain only one food class."
        )

    item_number, item_name = next(iter(prefixes))
    highest_existing_number = max(photo_number for _, _, _, photo_number in originals)

    # Build target names for all non-original files.
    plan: List[Tuple[Path, Path]] = []
    next_number = highest_existing_number + 1

    for file_path in non_originals:
        new_name = f"{item_number}_{item_name}_{next_number:03d}{file_path.suffix}"
        new_path = folder / new_name
        plan.append((file_path, new_path))
        next_number += 1

    # Safety check: do not allow any overwrite/conflict.
    existing_names = {p.name for p in files}
    new_names = [new_path.name for _, new_path in plan]

    if len(new_names) != len(set(new_names)):
        raise ValueError("The script generated duplicate target filenames. No changes were made.")

    for old_path, new_path in plan:
        # If target exists and is not the same file, stop.
        if new_path.exists() and new_path.resolve() != old_path.resolve():
            raise ValueError(
                f"Target filename already exists: {new_path.name}. "
                "No changes were made."
            )

    # Extra safety: make sure no new target collides with any untouched original.
    original_names = {file_path.name for file_path, _, _, _ in originals}
    for _, new_path in plan:
        if new_path.name in original_names:
            raise ValueError(
                f"Generated filename would collide with an original file: {new_path.name}. "
                "No changes were made."
            )

    return plan


def apply_rename_plan(plan: List[Tuple[Path, Path]]) -> None:
    """
    Two-step rename with temporary names to avoid any edge-case collisions.
    """
    temp_paths: List[Tuple[Path, Path]] = []

    # Step 1: rename all files to temporary names
    for index, (old_path, _) in enumerate(plan, start=1):
        temp_name = f"__temp_rename_{index:04d}__{old_path.name}"
        temp_path = old_path.parent / temp_name

        if temp_path.exists():
            raise ValueError(
                f"Temporary filename already exists: {temp_path.name}. "
                "Please remove it and try again."
            )

        old_path.rename(temp_path)
        temp_paths.append((temp_path, old_path))

    # Step 2: rename temporary files to final names
    for (temp_path, _), (_, final_path) in zip(temp_paths, plan):
        temp_path.rename(final_path)


def main() -> None:
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1]).expanduser().resolve()
    else:
        folder_input = input("Drag the folder here or type the folder path: ").strip().strip('"')
        folder = Path(folder_input).expanduser().resolve()

    if not folder.exists() or not folder.is_dir():
        print("Error: that path is not a valid folder.")
        sys.exit(1)

    try:
        plan = build_rename_plan(folder)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if not plan:
        print("Nothing to rename. All files already match the correct naming pattern.")
        return

    print("\nRename plan:")
    for old_path, new_path in plan:
        print(f"{old_path.name}  ->  {new_path.name}")

    confirm = input("\nProceed with renaming? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled. No files were changed.")
        return

    try:
        apply_rename_plan(plan)
    except Exception as exc:
        print(f"Error during renaming: {exc}")
        print("Some files may have been temporarily renamed. Please check the folder.")
        sys.exit(1)

    print(f"\nDone. Renamed {len(plan)} file(s).")


if __name__ == "__main__":
    main()