"""
M0 Baseline — Feature Engineering
==================================
Minimal feature preparation: raw sensors only + lagged targets for forecast.
No engineered features — establishes the accuracy floor.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "prepared_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# M0 uses ONLY raw sensor readings — no engineered features
DETECT_FEATURES = ["CO2", "temp", "RH", "Motion"]

# For forecast: add lagged target values so the model has some memory
FORECAST_LAG_STEPS = [6, 12, 24, 48]  # 30min, 1hr, 2hr, 4hr


def prepare_detection_data(df: pd.DataFrame):
    """Extract raw sensor features + target for detection."""
    X = df[DETECT_FEATURES].copy()
    y = df["target"].values
    return X, y


def prepare_forecast_data(df: pd.DataFrame):
    """Extract raw sensor features + lagged target for forecast.
    Remove Presence-derived columns to prevent leakage."""
    X = df[DETECT_FEATURES].copy()

    # Add lagged target: "was the room occupied N steps ago?"
    for lag in FORECAST_LAG_STEPS:
        col_name = f"target_lag_{lag}"
        X[col_name] = df.groupby("room_area", observed=True)["target"].shift(lag)

    # Target is 30-min-ahead occupancy
    y = df["target_forecast"].values

    # Drop rows where forecast target is NaN (end of each room's timeline)
    valid = ~np.isnan(y)
    # Also drop rows where any lag is NaN (start of each room's timeline)
    valid &= X.notna().all(axis=1).values

    return X[valid], y[valid]


def main():
    print("=" * 60)
    print("M0 BASELINE — Feature Engineering")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        print(f"\nProcessing {split_name}...")
        df = pd.read_parquet(DATA_DIR / f"{split_name}.parquet")

        # Detection
        X_det, y_det = prepare_detection_data(df)
        print(f"  Detection: X={X_det.shape}, y={len(y_det)}, "
              f"NaN rows in X: {X_det.isna().any(axis=1).sum():,}")

        # Forecast
        X_fcast, y_fcast = prepare_forecast_data(df)
        print(f"  Forecast:  X={X_fcast.shape}, y={len(y_fcast)}")

        # Save
        X_det.to_parquet(OUTPUT_DIR / f"{split_name}_detect_X.parquet", index=False)
        pd.DataFrame({"target": y_det}).to_parquet(
            OUTPUT_DIR / f"{split_name}_detect_y.parquet", index=False
        )
        X_fcast.to_parquet(OUTPUT_DIR / f"{split_name}_forecast_X.parquet", index=False)
        pd.DataFrame({"target_forecast": y_fcast}).to_parquet(
            OUTPUT_DIR / f"{split_name}_forecast_y.parquet", index=False
        )

    print(f"\n✓ M0 feature data saved to {OUTPUT_DIR}/")
    print(f"  Detection features: {DETECT_FEATURES}")
    print(f"  Forecast features: {DETECT_FEATURES} + lagged targets {FORECAST_LAG_STEPS}")


if __name__ == "__main__":
    main()
