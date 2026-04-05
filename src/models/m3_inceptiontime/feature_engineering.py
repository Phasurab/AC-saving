"""
M3 InceptionTime — Feature Engineering
========================================
Creates sliding window sequences for InceptionTime 1D CNN.
Uses the same windowing approach as M2 but may select different channels
based on permutation importance analysis.
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

WINDOW_SIZE = 12  # 60 minutes

# InceptionTime can handle more channels efficiently (1D convolutions are fast)
ALL_CHANNELS = [
    "CO2", "temp", "RH", "Motion",
    "hour_sin", "hour_cos", "dayofweek",
    "is_night", "CO2_above_baseline", "motion_binary",
]

FORECAST_CHANNELS = [
    "CO2", "temp", "RH", "Motion",
    "hour_sin", "hour_cos", "dayofweek",
    "is_night", "CO2_above_baseline", "motion_binary",
]

SUBSAMPLE_ROOMS = 150
SEED = 42


def create_sliding_windows(df, channels, target_col, window_size):
    """Create sliding window sequences per room."""
    X_windows = []
    y_windows = []

    rooms = df["room_area"].unique()
    for room in rooms:
        room_data = df[df["room_area"] == room].sort_values("timestamp")
        values = room_data[channels].values.astype(np.float32)
        targets = room_data[target_col].values
        values = np.nan_to_num(values, nan=0.0)

        for i in range(len(values) - window_size):
            target_val = targets[i + window_size - 1]
            if np.isnan(target_val):
                continue
            X_windows.append(values[i:i + window_size])
            y_windows.append(int(target_val))

    return np.array(X_windows, dtype=np.float32), np.array(y_windows, dtype=np.int64)


def main():
    print("=" * 60)
    print("M3 InceptionTime — Feature Engineering (Sliding Windows)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        print(f"\nProcessing {split_name}...")
        df = pd.read_parquet(DATA_DIR / f"{split_name}.parquet")

        rooms = df["room_area"].unique()
        if split_name == "train" and len(rooms) > SUBSAMPLE_ROOMS:
            np.random.seed(SEED)
            selected_rooms = np.random.choice(rooms, SUBSAMPLE_ROOMS, replace=False)
            df = df[df["room_area"].isin(selected_rooms)]
            print(f"  Subsampled to {SUBSAMPLE_ROOMS} rooms ({len(df):,} rows)")

        # Detection
        print(f"  Creating detection windows...")
        X_det, y_det = create_sliding_windows(df, ALL_CHANNELS, "target", WINDOW_SIZE)
        print(f"    Detection: X={X_det.shape}, y={y_det.shape}")
        np.save(OUTPUT_DIR / f"{split_name}_detect_X.npy", X_det)
        np.save(OUTPUT_DIR / f"{split_name}_detect_y.npy", y_det)

        # Forecast
        print(f"  Creating forecast windows...")
        X_fcast, y_fcast = create_sliding_windows(
            df, FORECAST_CHANNELS, "target_forecast", WINDOW_SIZE
        )
        print(f"    Forecast: X={X_fcast.shape}, y={y_fcast.shape}")
        np.save(OUTPUT_DIR / f"{split_name}_forecast_X.npy", X_fcast)
        np.save(OUTPUT_DIR / f"{split_name}_forecast_y.npy", y_fcast)

    meta = {
        "window_size": WINDOW_SIZE,
        "detect_channels": ALL_CHANNELS,
        "forecast_channels": FORECAST_CHANNELS,
        "n_detect_channels": len(ALL_CHANNELS),
        "n_forecast_channels": len(FORECAST_CHANNELS),
        "subsample_rooms_train": SUBSAMPLE_ROOMS,
    }
    with open(OUTPUT_DIR / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ M3 feature data saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
