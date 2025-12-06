#!/usr/bin/env python3
"""
Download pretrained models for the experiments.

Downloads:
1. bert-base-cased from HuggingFace
2. emilyalsentzer/Bio_ClinicalBERT from HuggingFace
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


MODEL_DIR = Path(__file__).parent.parent / "pretrained_models"

MODELS = {
    "bert-base-cased": "bert-base-cased",
    "Bio_ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT"
}


def download_model(model_name: str, huggingface_id: str, output_dir: Path):
    """Download a model from HuggingFace."""
    from transformers import AutoTokenizer, AutoModel

    save_path = output_dir / model_name

    if save_path.exists():
        print(f"  {model_name}: Already exists, skipping...")
        return True

    print(f"  {model_name}: Downloading from {huggingface_id}...")

    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(huggingface_id)
        tokenizer.save_pretrained(str(save_path))

        # Download model
        model = AutoModel.from_pretrained(huggingface_id)
        model.save_pretrained(str(save_path))

        print(f"  {model_name}: Saved to {save_path}")
        return True

    except Exception as e:
        print(f"  {model_name}: Failed - {e}")
        return False


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("Downloading Pretrained Models")
    print("=" * 60)
    print(f"\nOutput directory: {MODEL_DIR}\n")

    success_count = 0
    for model_name, huggingface_id in MODELS.items():
        if download_model(model_name, huggingface_id, MODEL_DIR):
            success_count += 1

    print()
    print("=" * 60)
    print(f"Downloaded {success_count}/{len(MODELS)} models successfully")
    print("=" * 60)

    if success_count < len(MODELS):
        print("\nSome models failed to download. Please check your internet connection")
        print("and try again, or download manually from HuggingFace.")
        sys.exit(1)


if __name__ == "__main__":
    main()
