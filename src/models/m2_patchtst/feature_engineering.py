"""
M2 PatchTST — Feature Engineering
===================================
Creates sliding window sequences for PatchTST transformer model.
Selects channels via preliminary attention weight analysis.
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

WINDOW_SIZE = 12  # 60 minutes of history (12 × 5-min steps)

# Candidate channels for the transformer
# These are the raw + key engineered features the transformer will learn from
ALL_CHANNELS = [
    "CO2", "temp", "RH", "Motion",
    "hour_sin", "hour_cos", "dayofweek",
    "is_night", "CO2_above_baseline", "motion_binary",
]

# For forecast: exclude features derived from Presence
FORECAST_CHANNELS = [
    "CO2", "temp", "RH", "Motion",
    "hour_sin", "hour_cos", "dayofweek",
    "is_night", "CO2_above_baseline", "motion_binary",
]

SUBSAMPLE_ROOMS = 150  # Increased from 50 for fairer comparison
SEED = 42


def create_sliding_windows(df: pd.DataFrame, channels: list,
                           target_col: str, window_size: int):
    """Create sliding window sequences per room.
    Returns X: (n_windows, window_size, n_channels), y: (n_windows,)"""
    X_windows = []
    y_windows = []

    rooms = df["room_area"].unique()
    for room in rooms:
        room_data = df[df["room_area"] == room].sort_values("timestamp")
        values = room_data[channels].values.astype(np.float32)
        targets = room_data[target_col].values

        # Fill NaN in features with 0 for the transformer
        values = np.nan_to_num(values, nan=0.0)

        for i in range(len(values) - window_size):
            target_val = targets[i + window_size - 1]  # detection: current step
            if target_col == "target_forecast":
                target_val = targets[i + window_size - 1]  # forecast target already shifted
            if np.isnan(target_val):
                continue
            X_windows.append(values[i:i + window_size])
            y_windows.append(int(target_val))

    X = np.array(X_windows, dtype=np.float32)
    y = np.array(y_windows, dtype=np.int64)
    return X, y


def main():
    print("=" * 60)
    print("M2 PatchTST — Feature Engineering (Sliding Windows)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        print(f"\nProcessing {split_name}...")
        df = pd.read_parquet(DATA_DIR / f"{split_name}.parquet")

        # Subsample rooms for train to keep memory manageable
        rooms = df["room_area"].unique()
        if split_name == "train" and len(rooms) > SUBSAMPLE_ROOMS:
            np.random.seed(SEED)
            selected_rooms = np.random.choice(rooms, SUBSAMPLE_ROOMS, replace=False)
            df = df[df["room_area"].isin(selected_rooms)]
            print(f"  Subsampled to {SUBSAMPLE_ROOMS} rooms ({len(df):,} rows)")

        # Detection windows
        print(f"  Creating detection windows (window={WINDOW_SIZE})...")
        X_det, y_det = create_sliding_windows(
            df, ALL_CHANNELS, "target", WINDOW_SIZE
        )
        print(f"    Detection: X={X_det.shape}, y={y_det.shape}, "
              f"balance={y_det.mean():.3f}")

        np.save(OUTPUT_DIR / f"{split_name}_detect_X.npy", X_det)
        np.save(OUTPUT_DIR / f"{split_name}_detect_y.npy", y_det)

        # Forecast windows
        print(f"  Creating forecast windows...")
        X_fcast, y_fcast = create_sliding_windows(
            df, FORECAST_CHANNELS, "target_forecast", WINDOW_SIZE
        )
        print(f"    Forecast: X={X_fcast.shape}, y={y_fcast.shape}, "
              f"balance={y_fcast.mean():.3f}")

        np.save(OUTPUT_DIR / f"{split_name}_forecast_X.npy", X_fcast)
        np.save(OUTPUT_DIR / f"{split_name}_forecast_y.npy", y_fcast)

    # Save metadata
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

    print(f"\n✓ M2 feature data saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
