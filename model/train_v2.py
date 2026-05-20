"""
Train LookkingMatcher V2 on real-data training set (training_data_v2.csv).
Adds:
  - loss + accuracy curves PNG
  - per-class precision/recall/F1
  - confusion matrix PNG
  - HONEST evaluation on holdout (data/holdout_test.csv)
  - metrics.json with all numbers

Run from project root:
    python3 model/train_v2.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, f1_score
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.model import LABEL_MAP, LABEL_NAMES, LookkingMatcher, get_device, get_tokenizer

DATA_PATH    = Path(__file__).parent.parent / "data" / "training_data_v2.csv"
HOLDOUT_PATH = Path(__file__).parent.parent / "data" / "holdout_test.csv"
MODEL_OUT    = Path(__file__).parent / "lookking_model.pt"
METRICS_OUT  = Path(__file__).parent / "metrics_v2.json"
CM_OUT       = Path(__file__).parent / "confusion_matrix_v2.png"
CURVES_OUT   = Path(__file__).parent / "training_curves.png"

BATCH_SIZE = 16
EPOCHS = 8
MAX_LEN = 128


class MatchDataset(Dataset):
    def __init__(self, queries, candidates, labels, tokenizer):
        self.pairs = [
            f"[QUERY]: {q} [SEP] [CANDIDATE]: {c}"
            for q, c in zip(queries, candidates)
        ]
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.pairs[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids, mask)
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return preds, trues


def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ticks = range(len(LABEL_NAMES))
    ax.set_xticks(ticks)
    ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(LABEL_NAMES)
    thresh = cm.max() / 2
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[+] Confusion matrix → {path}")


def plot_curves(train_losses, val_accs, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, marker="o", linewidth=2, color="#1f77b4")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.set_title("Training loss over epochs", fontweight="bold")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_accs, marker="s", linewidth=2, color="#2ca02c")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation accuracy")
    ax2.set_title("Validation accuracy over epochs", fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[+] Training curves → {path}")


def main():
    print("=" * 60)
    print("Lookking DL Matcher — Training V2 (real data)")
    print("=" * 60)

    print("\n[1] Loading training data...")
    df = pd.read_csv(DATA_PATH)
    df["label_id"] = df["label"].map(LABEL_MAP)
    print(f"    {len(df)} pairs, label counts: {df['label'].value_counts().to_dict()}")

    X_train, X_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_id"]
    )
    print(f"    Train: {len(X_train)}  |  Test: {len(X_test)}")

    tokenizer = get_tokenizer()
    device    = get_device()
    print(f"    Device: {device}")

    train_ds = MatchDataset(X_train["query"].tolist(), X_train["candidate"].tolist(),
                            X_train["label_id"].tolist(), tokenizer)
    test_ds  = MatchDataset(X_test["query"].tolist(),  X_test["candidate"].tolist(),
                            X_test["label_id"].tolist(),  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model = LookkingMatcher(num_classes=3).to(device)

    bert_params = [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n or "drop" in n]
    optimizer = torch.optim.AdamW(
        [{"params": bert_params, "lr": 2e-5}, {"params": head_params, "lr": 1e-3}],
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\n[2] Training {EPOCHS} epochs...")
    train_losses = []
    val_accs = []
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        preds, trues = evaluate(model, test_loader, device)
        acc = accuracy_score(trues, preds)

        train_losses.append(avg_loss)
        val_accs.append(acc)

        flag = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_OUT)
            flag = " ← saved"
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={acc:.4f}{flag}")

    # Final eval on best model
    print("\n[3] Final test-set evaluation (best checkpoint)...")
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device, weights_only=True))
    preds, trues = evaluate(model, test_loader, device)

    test_acc = accuracy_score(trues, preds)
    test_f1_macro = f1_score(trues, preds, average="macro")
    test_cm  = confusion_matrix(trues, preds)
    test_rep = classification_report(trues, preds, target_names=LABEL_NAMES,
                                     output_dict=True, zero_division=0)

    print(f"    Test accuracy: {test_acc*100:.2f}%")
    print(f"    Test macro F1: {test_f1_macro:.4f}")
    print("\n    Classification report:")
    print(classification_report(trues, preds, target_names=LABEL_NAMES, zero_division=0))

    # Holdout evaluation (the HONEST test)
    print("\n[4] HOLDOUT evaluation (hand-written unseen queries)...")
    holdout_acc = None
    holdout_f1 = None
    holdout_rep = None
    if HOLDOUT_PATH.exists():
        hdf = pd.read_csv(HOLDOUT_PATH)
        hdf["label_id"] = hdf["label"].map(LABEL_MAP)
        hd_ds = MatchDataset(hdf["query"].tolist(), hdf["candidate"].tolist(),
                             hdf["label_id"].tolist(), tokenizer)
        hd_loader = DataLoader(hd_ds, batch_size=BATCH_SIZE, shuffle=False)
        hpreds, htrues = evaluate(model, hd_loader, device)
        holdout_acc = accuracy_score(htrues, hpreds)
        holdout_f1  = f1_score(htrues, hpreds, average="macro")
        holdout_cm  = confusion_matrix(htrues, hpreds)
        holdout_rep = classification_report(htrues, hpreds, target_names=LABEL_NAMES,
                                            output_dict=True, zero_division=0)
        print(f"    Holdout size: {len(hdf)}")
        print(f"    Holdout accuracy: {holdout_acc*100:.2f}%")
        print(f"    Holdout macro F1: {holdout_f1:.4f}")
        print("\n    Holdout classification report:")
        print(classification_report(htrues, hpreds, target_names=LABEL_NAMES, zero_division=0))

        plot_confusion_matrix(
            holdout_cm,
            Path(__file__).parent / "confusion_matrix_holdout.png",
            "Holdout Confusion Matrix (real unseen queries)",
        )
    else:
        print(f"    ! No holdout file at {HOLDOUT_PATH}")

    # Plots
    print("\n[5] Saving plots...")
    plot_confusion_matrix(test_cm, CM_OUT, "Test Set Confusion Matrix (V2 / real data)")
    plot_curves(train_losses, val_accs, CURVES_OUT)

    # Metrics JSON
    metrics = {
        "training_data_size":   int(len(X_train)),
        "test_data_size":       int(len(X_test)),
        "holdout_data_size":    int(len(pd.read_csv(HOLDOUT_PATH))) if HOLDOUT_PATH.exists() else 0,
        "epochs":               EPOCHS,
        "best_val_accuracy":    round(best_acc, 4),
        "final_test_accuracy":  round(test_acc, 4),
        "final_test_macro_f1":  round(test_f1_macro, 4),
        "holdout_accuracy":     round(holdout_acc, 4) if holdout_acc is not None else None,
        "holdout_macro_f1":     round(holdout_f1, 4) if holdout_f1 is not None else None,
        "label_names":          LABEL_NAMES,
        "test_confusion_matrix":    test_cm.tolist(),
        "test_classification_report": test_rep,
        "holdout_classification_report": holdout_rep,
        "training_losses":      [round(x, 4) for x in train_losses],
        "validation_accs":      [round(x, 4) for x in val_accs],
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Metrics saved → {METRICS_OUT}")

    print("\n" + "=" * 60)
    print(f"DONE. Best test acc: {best_acc*100:.2f}%   "
          f"Holdout acc: {(holdout_acc or 0)*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
