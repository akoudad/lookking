"""
Train the LookkingMatcher model.
Run from project root:
    python3 model/train.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.model import LookkingMatcher, get_tokenizer, get_device, LABEL_MAP, LABEL_NAMES

DATA_PATH = Path(__file__).parent.parent / "data" / "training_data.csv"
MODEL_OUT = Path(__file__).parent / "lookking_model.pt"
METRICS_OUT = Path(__file__).parent / "metrics.json"
CM_OUT = Path(__file__).parent / "confusion_matrix.png"

BATCH_SIZE = 16
EPOCHS = 12
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
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids, mask)
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return preds, trues


def plot_confusion_matrix(cm, path):
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
    ax.set_title("Lookking Matcher — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"[+] Confusion matrix saved → {path}")


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["label_id"] = df["label"].map(LABEL_MAP)

    X_train, X_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_id"]
    )
    print(f"    Train: {len(X_train)}  |  Test: {len(X_test)}")

    tokenizer = get_tokenizer()
    device = get_device()
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

    best_acc = 0.0
    print("\nTraining...")
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

        preds, trues = evaluate(model, test_loader, device)
        acc = accuracy_score(trues, preds)
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={acc:.4f}", end="")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_OUT)
            print("  ← saved", end="")
        print()

    # Final eval on best model
    print("\n=== Final Evaluation ===")
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device, weights_only=True))
    preds, trues = evaluate(model, test_loader, device)

    acc = accuracy_score(trues, preds)
    cm  = confusion_matrix(trues, preds)
    rep = classification_report(trues, preds, target_names=LABEL_NAMES)

    print(f"Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print("\nClassification Report:")
    print(rep)

    plot_confusion_matrix(cm, CM_OUT)

    metrics = {
        "accuracy": round(acc, 4),
        "best_val_accuracy": round(best_acc, 4),
        "confusion_matrix": cm.tolist(),
        "label_names": LABEL_NAMES,
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Metrics saved → {METRICS_OUT}")
    print(f"\nTraining complete! Best accuracy: {best_acc*100:.1f}%")


if __name__ == "__main__":
    main()
