"""
M2 PatchTST — Pipeline (Detection + Forecast)
================================================
Lightweight PatchTST transformer for time series classification.
Uses MPS (Apple Silicon) if available.
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

# Model hyperparameters
PATCH_LEN = 4        # patch length
STRIDE = 2           # patch stride
D_MODEL = 64         # model dimension
N_HEADS = 4          # attention heads
N_LAYERS = 2         # transformer layers
D_FF = 128           # feedforward dimension
DROPOUT = 0.2
BATCH_SIZE = 512
EPOCHS = 50
LR = 1e-3
POS_WEIGHT = 5.0     # Asymmetric cost
PATIENCE = 10        # Early stopping patience


# ─── Dataset ──────────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)  # (N, T, C)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── PatchTST Model ──────────────────────────────────────────────────────────
class PatchTST(nn.Module):
    """Simplified PatchTST for time series classification."""

    def __init__(self, n_channels, seq_len, patch_len, stride,
                 d_model, n_heads, n_layers, d_ff, dropout, n_classes=2):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Per-channel patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches * n_channels, d_model) * 0.02)
        self.channel_embed = nn.Embedding(n_channels, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def create_patches(self, x):
        """x: (B, T, C) -> patches: (B, n_patches * C, patch_len)"""
        B, T, C = x.shape
        patches = []
        channel_ids = []
        for c in range(C):
            channel_data = x[:, :, c]  # (B, T)
            for i in range(self.n_patches):
                start = i * self.stride
                patch = channel_data[:, start:start + self.patch_len]  # (B, patch_len)
                patches.append(patch)
                channel_ids.append(c)
        # Stack: (B, n_patches * C, patch_len)
        patches = torch.stack(patches, dim=1)
        channel_ids = torch.tensor(channel_ids, device=x.device)
        return patches, channel_ids

    def forward(self, x):
        """x: (B, T, C) -> logits: (B, n_classes)"""
        patches, channel_ids = self.create_patches(x)  # (B, P, patch_len)
        B, P, _ = patches.shape

        # Embed patches
        z = self.patch_embed(patches)  # (B, P, d_model)
        z = z + self.pos_embed[:, :P, :]
        z = z + self.channel_embed(channel_ids).unsqueeze(0)  # (1, P, d_model)

        # Transformer
        z = self.transformer(z)  # (B, P, d_model)

        # Global average pooling
        z = self.norm(z.mean(dim=1))  # (B, d_model)

        return self.head(z)  # (B, n_classes)

    def get_attention_weights(self, x):
        """Extract attention weights for interpretability."""
        patches, channel_ids = self.create_patches(x)
        B, P, _ = patches.shape
        z = self.patch_embed(patches) + self.pos_embed[:, :P, :] + \
            self.channel_embed(channel_ids).unsqueeze(0)

        # Get attention from first layer
        attn_layer = self.transformer.layers[0].self_attn
        attn_output, attn_weights = attn_layer(z, z, z, need_weights=True)
        return attn_weights, channel_ids


def train_model(model, train_loader, val_loader, epochs, lr, device, task_name):
    """Train the PatchTST model."""
    print(f"\n{'─'*50}")
    print(f"Training PatchTST — {task_name}")
    print(f"  Device: {device}")

    # Weighted loss for asymmetric cost
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
        # Train
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

        # Validate
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
    print(f"  Training time: {train_time:.1f}s, Best val loss: {best_val_loss:.4f}")
    return model, train_time


def evaluate_model(model, test_loader, device, results_dir, task_name, prefix):
    """Evaluate model and generate metrics + plots."""
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

    if "forecast" in prefix.lower() or "forecast" in task_name.lower():
        metrics["brier_score"] = brier_score_loss(y_test, y_prob)
        metrics["log_loss"] = log_loss(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        metrics["pr_auc"] = auc(rec, prec)

    print(f"  Key Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    cmap = "Blues" if "detect" in prefix else "Oranges"
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=["Unoccupied", "Occupied"],
                yticklabels=["Unoccupied", "Occupied"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"M2 PatchTST — {task_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    return metrics


def main():
    print("=" * 60)
    print("M2 PatchTST — Pipeline (Detection + Forecast)")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open(RESULTS_DIR / "feature_meta.json") as f:
        meta = json.load(f)

    n_channels = meta["n_detect_channels"]
    seq_len = meta["window_size"]

    for task, task_label in [("detect", "Detection"), ("forecast", "Forecast (+30min)")]:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_label}")
        print(f"{'='*60}")

        # Load data
        X_train = np.load(RESULTS_DIR / f"train_{task}_X.npy")
        y_train = np.load(RESULTS_DIR / f"train_{task}_y.npy")
        X_val = np.load(RESULTS_DIR / f"val_{task}_X.npy")
        y_val = np.load(RESULTS_DIR / f"val_{task}_y.npy")
        X_test = np.load(RESULTS_DIR / f"test_{task}_X.npy")
        y_test = np.load(RESULTS_DIR / f"test_{task}_y.npy")

        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # Create dataloaders
        train_ds = TimeSeriesDataset(X_train, y_train)
        val_ds = TimeSeriesDataset(X_val, y_val)
        test_ds = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=False)

        n_ch = meta[f"n_{task}_channels"] if f"n_{task}_channels" in meta else n_channels

        # Build model
        model = PatchTST(
            n_channels=n_ch, seq_len=seq_len,
            patch_len=PATCH_LEN, stride=STRIDE,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            d_ff=D_FF, dropout=DROPOUT, n_classes=2,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}")

        # Train
        model, train_time = train_model(
            model, train_loader, val_loader, EPOCHS, LR, DEVICE, task_label
        )

        # Evaluate
        metrics = evaluate_model(
            model, test_loader, DEVICE, RESULTS_DIR, task_label, task
        )
        metrics["train_time_s"] = train_time
        metrics["n_params"] = n_params

        # Save model
        model_path = RESULTS_DIR / f"model_{task}.pt"
        torch.save(model.state_dict(), model_path)
        model_size = os.path.getsize(model_path) / 1024 / 1024

        # Save metrics
        all_metrics = {
            "model": "M2_patchtst",
            "task": task,
            f"{task}_metrics": metrics,
            f"{task}_model_size_MB": model_size,
            "channels": meta[f"{task}_channels"],
            "n_channels": n_ch,
            "window_size": seq_len,
        }
        with open(RESULTS_DIR / f"metrics_{task}.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)

    # Combine metrics
    detect_m = json.load(open(RESULTS_DIR / "metrics_detect.json"))
    forecast_m = json.load(open(RESULTS_DIR / "metrics_forecast.json"))
    combined = {
        "model": "M2_patchtst",
        "detection": detect_m.get("detect_metrics", {}),
        "detection_model_size_MB": detect_m.get("detect_model_size_MB", 0),
        "forecast": forecast_m.get("forecast_metrics", {}),
        "forecast_model_size_MB": forecast_m.get("forecast_model_size_MB", 0),
        "channels": meta["detect_channels"],
        "window_size": seq_len,
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("M2 PatchTST COMPLETE")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
