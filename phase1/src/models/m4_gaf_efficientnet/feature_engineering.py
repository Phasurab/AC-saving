"""
M4 GAF + EfficientNet — Feature Engineering
=============================================
Converts sensor time series into Gramian Angular Summation Field (GASF)
images for classification with EfficientNet-B0.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "prepared_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

WINDOW_SIZE = 12   # 60 minutes → 12×12 GAF image

# 3 channels encoded as RGB
CHANNEL_COMBOS = {
    "A": ["CO2", "temp", "RH"],
    "B": ["CO2", "Motion", "RH"],
    "C": ["CO2", "temp", "Motion"],
}
SELECTED_COMBO = "A"  # Start with environmental sensors

SUBSAMPLE_ROOMS = 100  # Increased from 30 for fairer comparison
SEED = 42


def compute_gasf(series):
    """Compute Gramian Angular Summation Field from a 1D time series.
    Input: 1D array of length T
    Output: T×T GASF matrix
    """
    # Normalize to [-1, 1]
    _min = np.nanmin(series)
    _max = np.nanmax(series)
    if _max - _min < 1e-8:
        scaled = np.zeros_like(series)
    else:
        scaled = 2 * (series - _min) / (_max - _min) - 1
    scaled = np.clip(scaled, -1, 1)

    # Convert to polar coordinates
    phi = np.arccos(scaled)

    # GASF: cos(phi_i + phi_j)
    gasf = np.cos(np.outer(phi, np.ones_like(phi)) + np.outer(np.ones_like(phi), phi))
    return gasf.astype(np.float32)


def create_gaf_dataset(df, channels, target_col, window_size):
    """Create GAF images from sliding windows of sensor data.
    Each window becomes a 3-channel (RGB) GAF image."""
    images = []
    labels = []

    rooms = df["room_area"].unique()
    for room in rooms:
        room_data = df[df["room_area"] == room].sort_values("timestamp")
        values = {ch: room_data[ch].values.astype(np.float32) for ch in channels}
        targets = room_data[target_col].values

        # Fill NaN
        for ch in channels:
            values[ch] = np.nan_to_num(values[ch], nan=0.0)

        for i in range(len(room_data) - window_size):
            target_val = targets[i + window_size - 1]
            if np.isnan(target_val):
                continue

            # Create 3-channel GAF image
            gasf_channels = []
            for ch in channels:
                window = values[ch][i:i + window_size]
                gasf = compute_gasf(window)
                gasf_channels.append(gasf)

            # Stack as (3, H, W) image
            img = np.stack(gasf_channels, axis=0)
            images.append(img)
            labels.append(int(target_val))

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    print("=" * 60)
    print("M4 GAF + EfficientNet — Feature Engineering")
    print(f"  Channel combo: {SELECTED_COMBO} = {CHANNEL_COMBOS[SELECTED_COMBO]}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    channels = CHANNEL_COMBOS[SELECTED_COMBO]

    for split_name in ["train", "val", "test"]:
        print(f"\nProcessing {split_name}...")
        df = pd.read_parquet(DATA_DIR / f"{split_name}.parquet")

        rooms = df["room_area"].unique()
        if split_name == "train" and len(rooms) > SUBSAMPLE_ROOMS:
            np.random.seed(SEED)
            selected_rooms = np.random.choice(rooms, SUBSAMPLE_ROOMS, replace=False)
            df = df[df["room_area"].isin(selected_rooms)]
            print(f"  Subsampled to {SUBSAMPLE_ROOMS} rooms ({len(df):,} rows)")

        # Detection GAF
        print(f"  Creating detection GAF images...")
        X_det, y_det = create_gaf_dataset(df, channels, "target", WINDOW_SIZE)
        print(f"    Detection: X={X_det.shape}, y={y_det.shape}, balance={y_det.mean():.3f}")
        np.save(OUTPUT_DIR / f"{split_name}_detect_X.npy", X_det)
        np.save(OUTPUT_DIR / f"{split_name}_detect_y.npy", y_det)

        # Forecast GAF
        print(f"  Creating forecast GAF images...")
        X_fcast, y_fcast = create_gaf_dataset(df, channels, "target_forecast", WINDOW_SIZE)
        print(f"    Forecast: X={X_fcast.shape}, y={y_fcast.shape}")
        np.save(OUTPUT_DIR / f"{split_name}_forecast_X.npy", X_fcast)
        np.save(OUTPUT_DIR / f"{split_name}_forecast_y.npy", y_fcast)

    meta = {
        "window_size": WINDOW_SIZE,
        "gaf_image_size": WINDOW_SIZE,
        "channel_combo": SELECTED_COMBO,
        "channels": channels,
        "n_channels": len(channels),
        "subsample_rooms_train": SUBSAMPLE_ROOMS,
    }
    with open(OUTPUT_DIR / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ M4 GAF data saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
