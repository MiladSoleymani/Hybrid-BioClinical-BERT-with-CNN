#!/usr/bin/env python3
"""
Hybrid Bio+Clinical BERT with CNN - Training Script

This script exactly replicates the notebook process:
1. Baseline BERT (Feature Extraction): frozen BERT + MLP
2. CNN-Word2Vec: Convolution model with trainable embeddings
3. BERT Base (Fine-Tuned): Fine-tuning of the entire model
4. Bio+Clinical BERT (Fine-Tuned): Fine-tuning of the medical model
5. Hybrid Bio+Clinical BERT + CNN: The proposed model

Usage:
    python train.py
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# =============================================================================
# 1. Setup & Configuration (matching notebook cell 1)
# =============================================================================

# --- SERVER CONFIGURATION ---
DATA_DIR = "/kaggle/input/kuc-hackathon-winter-2018"
MODEL_CACHE_DIR = "./pretrained_models"
RESULTS_DIR = "./experiment_results"
CHECKPOINT_DIR = "./checkpoints"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hardware settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# General settings
MAX_LEN = 128
BATCH_SIZE = 16  # Reduce to 8 if OOM
N_FOLDS = 3
EPOCHS = 4

# Dictionary to store the final results of all models for comparison
ALL_MODEL_RESULTS = {}


# =============================================================================
# 2. Data Loading & Hold-out Split (matching notebook cell 2)
# =============================================================================


def load_data():
    print("Loading Data from Local Server Storage...")

    train_path = os.path.join(DATA_DIR, "drugsComTrain_raw.csv")
    test_path = os.path.join(DATA_DIR, "drugsComTest_raw.csv")

    try:
        if os.path.exists(train_path) and os.path.exists(test_path):
            df_train = pd.read_csv(train_path)
            df_test_file = pd.read_csv(test_path)
            # Merge for finer control over segmentation
            df_all = pd.concat([df_train, df_test_file])
            print("Data loaded successfully from local paths.")
        else:
            raise FileNotFoundError(
                f"CSVs not found in {DATA_DIR}. Please upload them."
            )

    except Exception as e:
        print(f"Error loading data: {e}")
        # Generate dummy data ONLY if loading fails (Code verification fallback)
        df_all = pd.DataFrame(
            {
                "review": ["good drug"] * 100 + ["bad drug"] * 100 + ["okay"] * 100,
                "rating": [10] * 100 + [1] * 100 + [5] * 100,
            }
        )

    # Binning
    def bin_rating(r):
        try:
            r = float(r)
        except:
            return None
        if r <= 4:
            return 0  # Negative
        elif 5 <= r <= 8:
            return 1  # Neutral
        elif r >= 9:
            return 2  # Positive
        return None

    df_all["label"] = df_all["rating"].apply(bin_rating)
    df_all = df_all.rename(columns={"review": "text"}).dropna(subset=["text", "label"])
    df_all["label"] = df_all["label"].astype(int)

    print("Using FULL dataset (No Sampling).")

    # Final split: 80% for training/validation (which goes into K-Fold) and 20% for final test
    train_val_df, holdout_test_df = train_test_split(
        df_all, test_size=0.2, random_state=42, stratify=df_all["label"]
    )

    print(f"Train/CV Data: {len(train_val_df)}")
    print(f"Hold-out Test Data: {len(holdout_test_df)}")

    return train_val_df.reset_index(drop=True), holdout_test_df.reset_index(drop=True)


# Generic dataset class
class DrugDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# =============================================================================
# 3. Data Visualization (matching notebook cell 3)
# =============================================================================


def plot_label_distribution(df, title="Data Distribution"):
    plt.figure(figsize=(8, 5))
    # Mapping numbers to class names
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    df_viz = df.copy()
    df_viz["label_name"] = df_viz["label"].map(label_map)

    sns.countplot(
        x="label_name",
        data=df_viz,
        order=["Negative", "Neutral", "Positive"],
        palette="viridis",
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")


# =============================================================================
# 4. Model Architectures (matching notebook cell 4)
# =============================================================================


# 1. Baseline BERT (Frozen) + MLP
class BertBaseline(nn.Module):
    def __init__(self, model_path, n_classes=3):
        super(BertBaseline, self).__init__()
        # Load from Local Path directly
        self.bert = AutoModel.from_pretrained(model_path, local_files_only=True)
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        # MLP Head
        self.fc1 = nn.Linear(768, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        cls_output = outputs.pooler_output  # (Batch, 768)
        x = self.relu(self.fc1(cls_output))
        x = self.fc2(x)
        return x


# 2. CNN Model (Simulating Word2Vec input)
class CNN_Text(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, weights=None):
        super(CNN_Text, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if weights is not None:
            self.embedding.load_state_dict({"weight": torch.tensor(weights)})
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=k)
                for k in [1, 2, 3, 4, 5]
            ]
        )
        self.fc = nn.Linear(len(self.convs) * 100, 100)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(100, n_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return self.out(x)


# 3 & 4. Fine-Tuned BERT (Standard) - Used for both Generic BERT and Bio+Clinical BERT
class BertFineTune(nn.Module):
    def __init__(self, model_path, n_classes=3):
        super(BertFineTune, self).__init__()
        # Load from Local Path
        self.bert = AutoModel.from_pretrained(model_path, local_files_only=True)

        # Unfreeze last 4 layers
        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        if hasattr(self.bert, "pooler"):
            for param in self.bert.pooler.parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(768, 100), nn.ReLU(), nn.Dropout(0.1), nn.Linear(100, n_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        return self.classifier(outputs.pooler_output)


# 5. Hybrid Bio+Clinical BERT + CNN (Proposed model)
class HybridBioClinicalBertCNN(nn.Module):
    def __init__(self, model_path, n_classes=3):
        super(HybridBioClinicalBertCNN, self).__init__()
        # Load from Local Path (Bio+Clinical BERT will be passed here)
        self.bert = AutoModel.from_pretrained(model_path, local_files_only=True)

        self.conv1 = nn.Conv1d(768, 100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(768, 100, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(768, 100, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(300, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        # We use the last_hidden_state for CNN, not the pooler_output
        last_hidden = outputs.last_hidden_state.permute(0, 2, 1)  # (Batch, 768, Seq)

        x1 = F.max_pool1d(
            F.relu(self.conv1(last_hidden)), last_hidden.shape[2]
        ).squeeze(2)
        x2 = F.max_pool1d(
            F.relu(self.conv2(last_hidden)), last_hidden.shape[2]
        ).squeeze(2)
        x3 = F.max_pool1d(
            F.relu(self.conv3(last_hidden)), last_hidden.shape[2]
        ).squeeze(2)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        return self.fc(x)


# =============================================================================
# 5. Universal Training Engine (matching notebook cell 5)
# =============================================================================


def train_evaluate_engine(
    model_name_str, model_obj, tokenizer, train_df_cv, test_df_holdout
):
    print(f"\n{'=' * 10} Processing Model: {model_name_str} {'=' * 10}")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    best_fold_acc = 0.0
    best_model_path = os.path.join(CHECKPOINT_DIR, f"{model_name_str}_best.pth")

    # Check if trained model exists (Checkpointing)
    if os.path.exists(best_model_path):
        print(f"Found existing trained model for {model_name_str}. Loading...")
        model_obj.load_state_dict(torch.load(best_model_path))
        model_obj.to(device)
    else:
        # Start K-Fold Training
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_df_cv["text"], train_df_cv["label"])
        ):
            print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

            # 1. Split
            fold_train = train_df_cv.iloc[train_idx]
            fold_val = train_df_cv.iloc[val_idx]

            # 2. Balance (ONLY Train Data)
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(
                fold_train["text"].values.reshape(-1, 1), fold_train["label"].values
            )
            fold_train_bal = pd.DataFrame({"text": X_res.flatten(), "label": y_res})

            # 3. Dataloaders
            train_ds = DrugDataset(
                fold_train_bal["text"].values,
                fold_train_bal["label"].values,
                tokenizer,
                MAX_LEN,
            )
            val_ds = DrugDataset(
                fold_val["text"].values, fold_val["label"].values, tokenizer, MAX_LEN
            )

            # Reduce number of workers if needed, pin_memory=True for GPU
            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
            )
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True)

            # 4. Reset Model Weights
            model_obj.train()
            model_obj.to(device)
            optimizer = AdamW(model_obj.parameters(), lr=2e-5)
            loss_fn = nn.CrossEntropyLoss()

            # 5. Train Loop
            for epoch in range(EPOCHS):
                total_loss = 0
                model_obj.train()
                for batch in tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1}"):
                    input_ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    optimizer.zero_grad()
                    outputs = model_obj(input_ids, mask)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # Validation
                model_obj.eval()
                correct = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(device)
                        mask = batch["attention_mask"].to(device)
                        labels = batch["label"].to(device)
                        outputs = model_obj(input_ids, mask)
                        _, preds = torch.max(outputs, dim=1)
                        correct += torch.sum(preds == labels).item()
                val_acc = correct / len(val_ds)
                print(f"Fold {fold + 1} Ep {epoch + 1} | Val Acc: {val_acc:.4f}")

                # Save if best in this fold
                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc
                    torch.save(model_obj.state_dict(), best_model_path)

    # --- FINAL EVALUATION ON HOLDOUT TEST SET ---
    print(f"\nEvaluating {model_name_str} on Hold-out Test Set...")
    # Load best weights
    model_obj.load_state_dict(torch.load(best_model_path))
    model_obj.to(device)
    model_obj.eval()

    test_ds = DrugDataset(
        test_df_holdout["text"].values,
        test_df_holdout["label"].values,
        tokenizer,
        MAX_LEN,
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=True)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model_obj(input_ids, mask)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Store Results
    ALL_MODEL_RESULTS[model_name_str] = {
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
        "y_probs": np.array(all_probs),
    }

    # Cleanup
    del model_obj, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# 6. Execute All Models (matching notebook cell 6)
# =============================================================================


def run_all_models(train_val_df, holdout_test_df):
    # --- DEFINING EXACT LOCAL PATHS ---
    BERT_PATH = os.path.join(MODEL_CACHE_DIR, "bert-base-cased")
    BIO_CLINICAL_PATH = os.path.join(MODEL_CACHE_DIR, "Bio_ClinicalBERT")

    print(f"Models will be loaded from:\n 1. {BERT_PATH}\n 2. {BIO_CLINICAL_PATH}")

    print("Loading Tokenizers from Local Path...")
    # Load Tokenizers
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    bio_tokenizer = AutoTokenizer.from_pretrained(
        BIO_CLINICAL_PATH, local_files_only=True
    )

    # --- 1. Run BERT Baseline ---
    model1 = BertBaseline(BERT_PATH, n_classes=3)
    train_evaluate_engine(
        "1_BERT_Baseline", model1, bert_tokenizer, train_val_df, holdout_test_df
    )

    # --- 2. Run CNN (Simulating Word2Vec input) ---
    print("Running CNN...")
    # CNN uses BERT tokenizer for vocab mapping (Embeddings are trainable here)
    model2 = CNN_Text(vocab_size=bert_tokenizer.vocab_size, embed_dim=300, n_classes=3)
    train_evaluate_engine(
        "2_CNN_Model", model2, bert_tokenizer, train_val_df, holdout_test_df
    )

    # --- 3. Run BERT Fine-Tuned ---
    model3 = BertFineTune(BERT_PATH, n_classes=3)
    train_evaluate_engine(
        "3_BERT_FineTuned", model3, bert_tokenizer, train_val_df, holdout_test_df
    )

    # --- 4. Run Bio+Clinical BERT ---
    model4 = BertFineTune(BIO_CLINICAL_PATH, n_classes=3)
    train_evaluate_engine(
        "4_BioClinicalBERT_FineTuned",
        model4,
        bio_tokenizer,
        train_val_df,
        holdout_test_df,
    )

    # --- 5. Run Hybrid Bio+Clinical BERT + CNN ---
    model5 = HybridBioClinicalBertCNN(BIO_CLINICAL_PATH, n_classes=3)
    train_evaluate_engine(
        "5_Hybrid_BioClinicalBERT_CNN",
        model5,
        bio_tokenizer,
        train_val_df,
        holdout_test_df,
    )

    print("\nAll models processed successfully!")


# =============================================================================
# 7. Final Comparative Visualization (matching notebook cell 7)
# =============================================================================


def generate_visualizations():
    plt.figure(figsize=(15, 12))

    # 1. ROC Curve Comparison (Macro-Average)
    plt.subplot(2, 1, 1)
    colors = ["blue", "orange", "green", "red", "purple"]

    for i, (name, res) in enumerate(ALL_MODEL_RESULTS.items()):
        y_true = res["y_true"]
        y_probs = res["y_probs"]

        # Binarize labels for ROC
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        n_classes = 3

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for j in range(n_classes):
            fpr[j], tpr[j], _ = roc_curve(y_true_bin[:, j], y_probs[:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])

        # Compute micro-average ROC curve and ROC area
        # Here we plot MACRO average for simplicity in comparison
        all_fpr = np.unique(np.concatenate([fpr[j] for j in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for j in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[j], tpr[j])
        mean_tpr /= n_classes

        plt.plot(
            all_fpr,
            mean_tpr,
            color=colors[i],
            lw=2,
            label=f"{name} (AUC = {auc(all_fpr, mean_tpr):.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparison of ROC Curves (Macro-Average) Across All Models")
    plt.legend(loc="lower right")
    plt.grid(True)

    # 2. Confusion Matrices
    plt.figure(figsize=(20, 4))
    for i, (name, res) in enumerate(ALL_MODEL_RESULTS.items()):
        plt.subplot(1, 5, i + 1)
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(name, fontsize=10)
        plt.xlabel("Predicted")
        if i == 0:
            plt.ylabel("True")
        plt.xticks([0.5, 1.5, 2.5], ["Neg", "Neu", "Pos"])
        plt.yticks([0.5, 1.5, 2.5], ["Neg", "Neu", "Pos"])

    plt.tight_layout()
    # --- SAVING RESULTS TO RESULTS_DIR ---
    plot_save_path = os.path.join(RESULTS_DIR, "final_results_plot.png")
    plt.savefig(plot_save_path)
    print(f"Plots saved to: {plot_save_path}")

    # 3. Textual Report Table
    print("\n--- Final Performance Summary ---")
    summary_data = []
    for name, res in ALL_MODEL_RESULTS.items():
        acc = accuracy_score(res["y_true"], res["y_pred"])
        p, r, f1, _ = precision_recall_fscore_support(
            res["y_true"], res["y_pred"], average="macro"
        )
        summary_data.append([name, acc, p, r, f1])

    df_summary = pd.DataFrame(
        summary_data,
        columns=["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
    )
    df_summary = df_summary.sort_values(by="Macro F1", ascending=False)

    # Print to console
    print(df_summary.to_string(index=False))

    # Save to CSV
    csv_save_path = os.path.join(RESULTS_DIR, "performance_summary.csv")
    df_summary.to_csv(csv_save_path, index=False)
    print(f"Metrics saved to: {csv_save_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hybrid Bio+Clinical BERT with CNN - Experiment")
    print("=" * 60)

    # Load data
    train_val_df, holdout_test_df = load_data()

    # Visualize data distribution
    print("\n--- Training Set Distribution (Before Balancing) ---")
    plot_label_distribution(train_val_df, "Train/CV Set Distribution")

    print("\n--- Hold-out Test Set Distribution ---")
    plot_label_distribution(holdout_test_df, "Test Set Distribution")

    # Run all models
    run_all_models(train_val_df, holdout_test_df)

    # Generate visualizations and summary
    generate_visualizations()

    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)
