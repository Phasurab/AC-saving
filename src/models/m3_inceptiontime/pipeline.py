"""
M3 InceptionTime — Pipeline (Detection + Forecast)
====================================================
InceptionTime 1D CNN for time series classification.
Uses parallel convolution filters at multiple scales.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score,
    precision_recall_curve, auc, log_loss, brier_score_loss, roc_auc_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import os
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BATCH_SIZE = 512
EPOCHS = 50
LR = 1e-3
POS_WEIGHT = 5.0
PATIENCE = 10
N_INCEPTION_BLOCKS = 3


# ─── Dataset ──────────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # InceptionTime expects (B, C, T) — channels first
        self.X = torch.FloatTensor(X).permute(0, 2, 1)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── InceptionTime Model ─────────────────────────────────────────────────────
class InceptionBlock(nn.Module):
    """Single Inception block with parallel convolutions at different scales."""

    def __init__(self, in_channels, n_filters=32, bottleneck_size=32):
        super().__init__()
        # Bottleneck
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)

        # Parallel convolutions at different kernel sizes
        self.conv_10 = nn.Conv1d(bottleneck_size, n_filters, kernel_size=10,
                                  padding=4, bias=False)
        self.conv_20 = nn.Conv1d(bottleneck_size, n_filters, kernel_size=7,
                                  padding=3, bias=False)
        self.conv_40 = nn.Conv1d(bottleneck_size, n_filters, kernel_size=3,
                                  padding=1, bias=False)

        # MaxPool path
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        # BatchNorm + activation
        self.bn = nn.BatchNorm1d(n_filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Bottleneck
        bottleneck = self.bottleneck(x)

        # Parallel paths
        conv1 = self.conv_10(bottleneck)
        conv2 = self.conv_20(bottleneck)
        conv3 = self.conv_40(bottleneck)

        # MaxPool path
        pool = self.maxpool(x)
        pool = self.conv_pool(pool)

        # Ensure all have same temporal dimension (trim if needed)
        min_len = min(conv1.size(2), conv2.size(2), conv3.size(2), pool.size(2))
        conv1 = conv1[:, :, :min_len]
        conv2 = conv2[:, :, :min_len]
        conv3 = conv3[:, :, :min_len]
        pool = pool[:, :, :min_len]

        # Concatenate
        out = torch.cat([conv1, conv2, conv3, pool], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class InceptionTime(nn.Module):
    """InceptionTime: ensemble of Inception blocks for time series classification."""

    def __init__(self, n_channels, n_classes=2, n_filters=32,
                 n_blocks=3, bottleneck_size=32):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        in_ch = n_channels
        for i in range(n_blocks):
            self.blocks.append(InceptionBlock(in_ch, n_filters, bottleneck_size))
            # Residual shortcut every 3 blocks
            if i % 3 == 2:
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(n_channels if i == 2 else n_filters * 4,
                              n_filters * 4, kernel_size=1, bias=False),
                    nn.BatchNorm1d(n_filters * 4),
                ))
            else:
                self.shortcuts.append(None)
            in_ch = n_filters * 4

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(n_filters * 4, n_classes)

    def forward(self, x):
        residual = x
        for i, (block, shortcut) in enumerate(zip(self.blocks, self.shortcuts)):
            x = block(x)
            if shortcut is not None:
                # Adjust residual to match dims
                if residual.size(2) != x.size(2):
                    residual = residual[:, :, :x.size(2)]
                res = shortcut(residual)
                x = x + res
                residual = x

        x = self.gap(x)  # (B, C, 1)
        x = x.squeeze(-1)  # (B, C)
        return self.head(x)


def train_model(model, train_loader, val_loader, epochs, lr, device, task_name):
    """Train InceptionTime."""
    print(f"\n{'─'*50}")
    print(f"Training InceptionTime — {task_name}")

    weight = torch.FloatTensor([1.0, POS_WEIGHT]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}")

    train_time = time.time() - start_time
    model.load_state_dict(best_state)
    print(f"  Training time: {train_time:.1f}s")
    return model, train_time


def evaluate_model(model, test_loader, device, results_dir, task_name, prefix):
    """Evaluate model."""
    print(f"\n{'─'*50}")
    print(f"Evaluating {task_name}")

    model.eval()
    all_probs = []
    all_labels = []

    start = time.time()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
    total_inference = time.time() - start

    y_prob = np.array(all_probs)
    y_test = np.array(all_labels)
    y_pred = (y_prob >= 0.5).astype(int)
    inference_ms = total_inference / len(y_test) * 1000

    report = classification_report(y_test, y_pred,
                                   target_names=["Unoccupied", "Occupied"],
                                   output_dict=True)
    print(classification_report(y_test, y_pred,
                                target_names=["Unoccupied", "Occupied"]))

    metrics = {
        "recall_occupied": report["Occupied"]["recall"],
        "precision_unoccupied": report["Unoccupied"]["precision"],
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "inference_time_ms_per_sample": inference_ms,
    }

    if "forecast" in prefix:
        metrics["brier_score"] = brier_score_loss(y_test, y_prob)
        metrics["log_loss"] = log_loss(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        metrics["pr_auc"] = auc(rec, prec)

    print(f"  Key Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    cmap = "Blues" if "detect" in prefix else "Oranges"
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=["Unoccupied", "Occupied"],
                yticklabels=["Unoccupied", "Occupied"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"M3 InceptionTime — {task_name}")
    plt.tight_layout()
    plt.savefig(results_dir / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    return metrics


def main():
    print("=" * 60)
    print("M3 InceptionTime — Pipeline (Detection + Forecast)")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open(RESULTS_DIR / "feature_meta.json") as f:
        meta = json.load(f)

    for task, task_label in [("detect", "Detection"), ("forecast", "Forecast (+30min)")]:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_label}")

        X_train = np.load(RESULTS_DIR / f"train_{task}_X.npy")
        y_train = np.load(RESULTS_DIR / f"train_{task}_y.npy")
        X_val = np.load(RESULTS_DIR / f"val_{task}_X.npy")
        y_val = np.load(RESULTS_DIR / f"val_{task}_y.npy")
        X_test = np.load(RESULTS_DIR / f"test_{task}_X.npy")
        y_test = np.load(RESULTS_DIR / f"test_{task}_y.npy")

        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        n_channels = X_train.shape[2]
        train_ds = TimeSeriesDataset(X_train, y_train)
        val_ds = TimeSeriesDataset(X_val, y_val)
        test_ds = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = InceptionTime(
            n_channels=n_channels, n_classes=2,
            n_filters=32, n_blocks=N_INCEPTION_BLOCKS,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}")

        model, train_time = train_model(
            model, train_loader, val_loader, EPOCHS, LR, DEVICE, task_label
        )

        metrics = evaluate_model(
            model, test_loader, DEVICE, RESULTS_DIR, task_label, task
        )
        metrics["train_time_s"] = train_time
        metrics["n_params"] = n_params

        model_path = RESULTS_DIR / f"model_{task}.pt"
        torch.save(model.state_dict(), model_path)
        model_size = os.path.getsize(model_path) / 1024 / 1024

        task_metrics = {
            "model": "M3_inceptiontime",
            "task": task,
            f"{task}_metrics": metrics,
            f"{task}_model_size_MB": model_size,
            "channels": meta[f"{task}_channels"],
        }
        with open(RESULTS_DIR / f"metrics_{task}.json", "w") as f:
            json.dump(task_metrics, f, indent=2, default=str)

    # Combine
    detect_m = json.load(open(RESULTS_DIR / "metrics_detect.json"))
    forecast_m = json.load(open(RESULTS_DIR / "metrics_forecast.json"))
    combined = {
        "model": "M3_inceptiontime",
        "detection": detect_m.get("detect_metrics", {}),
        "detection_model_size_MB": detect_m.get("detect_model_size_MB", 0),
        "forecast": forecast_m.get("forecast_metrics", {}),
        "forecast_model_size_MB": forecast_m.get("forecast_model_size_MB", 0),
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("M3 InceptionTime COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
