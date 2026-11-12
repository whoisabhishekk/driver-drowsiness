#!/usr/bin/env python3
"""
Download the Driver Drowsiness Dataset (DDD) from Kaggle.

Prerequisites:
    pip install kaggle
    Place your kaggle.json API token in ~/.kaggle/kaggle.json
    (Get it from https://www.kaggle.com/settings → "Create New Token")

Usage:
    python download_dataset.py                    # downloads to ./dataset/
    python download_dataset.py --output ./data    # downloads to ./data/

After downloading, run the preprocessing script:
    python ddd_preprocessing.py --source ./dataset/Driver\\ Drowsiness\\ Dataset\\ \\(DDD\\)
"""

import os
import sys
import zipfile
import argparse


KAGGLE_DATASET = "ismailnasri20/driver-drowsiness-dataset-ddd"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "dataset")

    parser = argparse.ArgumentParser(description="Download the DDD dataset from Kaggle.")
    parser.add_argument("--output", default=default_output,
                        help="Directory to download and extract the dataset into.")
    return parser.parse_args()


def check_kaggle_cli():
    """Check if the kaggle CLI is installed and configured."""
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        return False


def download_with_kaggle(output_dir):
    """Download and extract using the Kaggle API."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {KAGGLE_DATASET}")
    print(f"Destination: {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)

    api.dataset_download_files(
        KAGGLE_DATASET,
        path=output_dir,
        unzip=True,
    )

    print(f"\n✓ Dataset downloaded and extracted to: {output_dir}")

    # List contents
    for item in sorted(os.listdir(output_dir)):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            count = len(os.listdir(item_path))
            print(f"  📁 {item}/ ({count} items)")
        else:
            size_mb = os.path.getsize(item_path) / (1024 * 1024)
            print(f"  📄 {item} ({size_mb:.1f} MB)")


def print_manual_instructions(output_dir):
    """Print instructions for manual download."""
    print("=" * 60)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("The Kaggle CLI is not installed or not configured.")
    print()
    print("Option 1 — Install the Kaggle CLI:")
    print("  pip install kaggle")
    print("  # Place your API token at ~/.kaggle/kaggle.json")
    print("  # Get it from: https://www.kaggle.com/settings")
    print("  # Then re-run this script.")
    print()
    print("Option 2 — Download manually:")
    print(f"  1. Go to: {KAGGLE_URL}")
    print("  2. Click 'Download' (you may need to sign in)")
    print(f"  3. Extract the ZIP into: {output_dir}")
    print()
    print("After downloading, the folder structure should be:")
    print(f"  {output_dir}/")
    print("  ├── Drowsy/          (~22,348 images)")
    print("  └── Non Drowsy/      (~19,445 images)")
    print()
    print("Then run preprocessing:")
    print(f"  python ddd_preprocessing.py --source {output_dir}")
    print()


def main():
    args = parse_args()

    if check_kaggle_cli():
        try:
            download_with_kaggle(args.output)
            print(f"\nNext step — preprocess the dataset:")
            # Try to find the DDD subfolder
            ddd_dir = args.output
            for item in os.listdir(args.output):
                if "DDD" in item or "drowsiness" in item.lower():
                    ddd_dir = os.path.join(args.output, item)
                    break
            print(f"  python ddd_preprocessing.py --source \"{ddd_dir}\"")
        except Exception as e:
            print(f"\nERROR: Kaggle download failed: {e}")
            print_manual_instructions(args.output)
            sys.exit(1)
    else:
        print_manual_instructions(args.output)
        sys.exit(1)


if __name__ == "__main__":
    main()
