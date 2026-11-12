#!/usr/bin/env python3
"""
Dataset Preparation — Driver Drowsiness Dataset (DDD)

Splits the raw DDD folders into an ImageFolder-compatible train/val tree:

    processed_dataset/
    ├── train/
    │   ├── 0_alert/
    │   └── 1_drowsy/
    └── val/
        ├── 0_alert/
        └── 1_drowsy/

Usage:
    python ddd_preprocessing.py --source <path_to_raw_DDD> [--output <path>] [--val-ratio 0.2]

If --source is not supplied, the script looks for a folder called
"Driver Drowsiness Dataset (DDD)" next to this file.
"""

import os
import sys
import shutil
import random
import argparse
from tqdm import tqdm


# --- Default configuration ---
DEFAULT_VAL_SPLIT_RATIO = 0.2
RANDOM_SEED = 42

# Source class name -> target class name
# Alphabetical target names fix label indices deterministically:
#   0_alert -> 0,  1_drowsy -> 1
SOURCE_CLASSES = {
    "Non Drowsy": "0_alert",
    "Drowsy": "1_drowsy",
}

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")
SETS = ["train", "val"]


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_source = os.path.join(script_dir, "Driver Drowsiness Dataset (DDD)")
    default_output = os.path.join(script_dir, "processed_dataset")

    parser = argparse.ArgumentParser(
        description="Split the raw DDD dataset into train/val folders."
    )
    parser.add_argument(
        "--source",
        default=default_source,
        help="Path to the raw DDD dataset directory (contains 'Drowsy/' and 'Non Drowsy/' folders).",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="Path where the processed dataset will be written.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_SPLIT_RATIO,
        help=f"Fraction of data to use for validation (default: {DEFAULT_VAL_SPLIT_RATIO}).",
    )
    return parser.parse_args()


def create_directories(output_dir):
    """Create the train/val directory tree."""
    print(f"Creating directory structure at: {output_dir}")
    for split in SETS:
        for target_class in SOURCE_CLASSES.values():
            path = os.path.join(output_dir, split, target_class)
            os.makedirs(path, exist_ok=True)
    print("Directory structure created successfully.\n")


def split_and_copy(source_dir, output_dir, val_ratio):
    """Shuffle, split, and copy images into train/val folders."""
    random.seed(RANDOM_SEED)

    total_copied = 0
    total_skipped = 0

    for source_class, target_class in SOURCE_CLASSES.items():
        source_folder = os.path.join(source_dir, source_class)

        if not os.path.isdir(source_folder):
            print(f"ERROR: Source folder not found: {source_folder}")
            print("       Make sure --source points to the directory that contains "
                  "'Drowsy/' and 'Non Drowsy/' sub-folders.")
            sys.exit(1)

        # Collect only image files
        all_files = sorted(
            f
            for f in os.listdir(source_folder)
            if os.path.isfile(os.path.join(source_folder, f))
            and f.lower().endswith(VALID_EXTENSIONS)
        )

        if not all_files:
            print(f"WARNING: No images found in {source_folder}")
            continue

        random.shuffle(all_files)

        split_idx = int(len(all_files) * val_ratio)
        splits = {
            "val": all_files[:split_idx],
            "train": all_files[split_idx:],
        }

        print(f"Class: {source_class}")
        print(f"  Total images : {len(all_files)}")
        print(f"  Training     : {len(splits['train'])}")
        print(f"  Validation   : {len(splits['val'])}")

        for split_name, file_list in splits.items():
            dest_folder = os.path.join(output_dir, split_name, target_class)
            for fname in tqdm(file_list, desc=f"  {split_name:>5s}"):
                try:
                    shutil.copy2(
                        os.path.join(source_folder, fname),
                        os.path.join(dest_folder, fname),
                    )
                    total_copied += 1
                except Exception as e:
                    print(f"  Error copying {fname}: {e}")
                    total_skipped += 1

        print()

    return total_copied, total_skipped


def main():
    args = parse_args()

    if not os.path.isdir(args.source):
        print(f"ERROR: Source directory does not exist: {args.source}")
        print("\nUsage:")
        print("  python ddd_preprocessing.py --source /path/to/Driver\\ Drowsiness\\ Dataset\\ \\(DDD\\)")
        print("\nYou can download the dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd")
        sys.exit(1)

    create_directories(args.output)

    print("Splitting and copying files...\n")
    copied, skipped = split_and_copy(args.source, args.output, args.val_ratio)

    print("--- Pre-processing Complete ---")
    print(f"Total files copied  : {copied}")
    print(f"Total files skipped : {skipped}")
    print(f"Processed dataset   : {args.output}")
    print(f"\nNext step — train the model:")
    print(f"  python pyt2.py --data {args.output}")


if __name__ == "__main__":
    main()
