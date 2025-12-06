#!/usr/bin/env python3
"""
Download the UCI Drug Review dataset.

This script downloads the drugsComTrain_raw.csv and drugsComTest_raw.csv files.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"

# Kaggle dataset URL (requires kaggle authentication)
KAGGLE_DATASET = "jessicali9530/kuc-hackathon-winter-2018"

# Alternative: Direct download URLs (if available)
# Note: The dataset is from UCI/Kaggle, you may need to download manually


def download_with_kaggle():
    """Download using kaggle API."""
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(DATA_DIR),
            unzip=True
        )
        print(f"Dataset downloaded to {DATA_DIR}")
        return True
    except ImportError:
        print("kaggle package not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return False


def check_data_exists():
    """Check if data files already exist."""
    train_path = DATA_DIR / "drugsComTrain_raw.csv"
    test_path = DATA_DIR / "drugsComTest_raw.csv"
    return train_path.exists() and test_path.exists()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if check_data_exists():
        print("Data files already exist!")
        print(f"  - {DATA_DIR / 'drugsComTrain_raw.csv'}")
        print(f"  - {DATA_DIR / 'drugsComTest_raw.csv'}")
        return

    print("Attempting to download UCI Drug Review dataset...")
    print()

    # Try kaggle download
    if download_with_kaggle():
        return

    # Manual download instructions
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("Please download the dataset manually:")
    print()
    print("Option 1: Kaggle (requires account)")
    print("  1. Go to: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018")
    print("  2. Download the dataset")
    print("  3. Extract to ./data/")
    print()
    print("Option 2: UCI Machine Learning Repository")
    print("  1. Go to: https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29")
    print("  2. Download drugsComTrain_raw.zip and drugsComTest_raw.zip")
    print("  3. Extract to ./data/")
    print()
    print(f"Required files in {DATA_DIR}:")
    print("  - drugsComTrain_raw.csv")
    print("  - drugsComTest_raw.csv")
    print()


if __name__ == "__main__":
    main()
