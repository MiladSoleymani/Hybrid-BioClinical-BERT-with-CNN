# Hybrid Bio+Clinical BERT with CNN

Sentiment analysis of drug reviews using a hybrid approach combining Bio+Clinical BERT embeddings with CNN feature extraction.

## Models Compared

| # | Model | Description |
|---|-------|-------------|
| 1 | BERT Baseline | Frozen BERT + MLP (Feature Extraction) |
| 2 | CNN-Word2Vec | CNN with trainable embeddings (kernel sizes 1-5) |
| 3 | BERT Fine-Tuned | Fine-tuning last 4 layers of BERT |
| 4 | Bio+Clinical BERT | Fine-tuned domain-specific BERT |
| 5 | **Hybrid (Proposed)** | Bio+Clinical BERT + CNN |

## Project Structure

```
.
├── train.py                 # Main training script (all code in one file)
├── requirements.txt         # Python dependencies
├── scripts/
│   ├── download_data.py    # Download dataset
│   └── download_models.py  # Download pretrained models
├── data/                   # Dataset files (drugsComTrain_raw.csv, drugsComTest_raw.csv)
├── pretrained_models/      # Downloaded BERT models
│   ├── bert-base-cased/
│   └── Bio_ClinicalBERT/
├── checkpoints/            # Model checkpoints
└── experiment_results/     # Output results & plots
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Pretrained Models

```bash
python scripts/download_models.py
```

This downloads:
- `bert-base-cased` from HuggingFace
- `emilyalsentzer/Bio_ClinicalBERT` from HuggingFace

### 3. Download Data

```bash
python scripts/download_data.py
```

Or manually download from:
- [Kaggle: UCI Drug Review Dataset](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)

Place `drugsComTrain_raw.csv` and `drugsComTest_raw.csv` in `./data/`

### 4. Run Training

```bash
python train.py
```

## Configuration

Edit the constants at the top of `train.py` to modify settings:

```python
DATA_DIR = "./data"
MODEL_CACHE_DIR = "./pretrained_models"
RESULTS_DIR = "./experiment_results"
CHECKPOINT_DIR = "./checkpoints"

MAX_LEN = 128
BATCH_SIZE = 16  # Reduce to 8 if OOM
N_FOLDS = 3
EPOCHS = 4
```

## Results

After training, results are saved to `./experiment_results/`:
- `final_results_plot.png` - ROC curves and confusion matrices
- `performance_summary.csv` - Accuracy, Precision, Recall, F1 for all models

## Dataset

UCI Drug Review Dataset with 3-class sentiment:
- **Negative**: Rating 1-4
- **Neutral**: Rating 5-8
- **Positive**: Rating 9-10

Data split:
- 80% for K-Fold cross-validation (training)
- 20% hold-out test set

## Hardware Requirements

- GPU recommended (NVIDIA with CUDA support)
- Minimum 8GB GPU memory (reduce `BATCH_SIZE` to 8 if limited)
- CPU training is supported but significantly slower
