"""
data_preparation.py — Shared Data Preparation for All Model Pipelines
=====================================================================
Loads the enriched parquet, performs leakage audit, applies temporal split,
and exports base datasets for all model folders to consume.

Usage:
    python data_preparation.py

Outputs:
    prepared_data/
        train.parquet
        val.parquet
        test.parquet
        feature_audit.csv
        split_summary.json
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_PATH = "eda_feat_enriched.parquet"
OUTPUT_DIR = Path("prepared_data")
TRAIN_CUTOFF = "2026-01-23"   # 70% of 77 days
VAL_CUTOFF = "2026-02-04"     # 85% of 77 days

# Presence codes 3 (CO2 disconnected) and 4 (Motion disconnected) are invalid labels
VALID_PRESENCE_VALUES = [0.0, 1.0]

# ─── Feature Audit ───────────────────────────────────────────────────────────

# Features safe for BOTH detection and forecast (strictly backward-looking)
SAFE_FEATURES = [
    # Raw sensors (except Presence — guarded separately)
    "CO2", "temp", "RH", "Motion",
    # Instantaneous derived
    "motion_binary", "CO2_above_baseline", "temp_above_baseline",
    # Lags (backward-looking by definition)
    "RH_lag1", "CO2_lag1", "temp_lag1", "CO2_lag3",
    # Diffs (backward-looking)
    "RH_diff1", "CO2_diff1", "temp_diff1", "CO2_diff3",
    # Rolling stats (backward-looking window)
    "CO2_roll_mean_6", "temp_roll_mean_6", "RH_roll_mean_6",
    "CO2_roll_std_6", "CO2_roll_mean_12", "RH_roll_std_6",
    "motion_roll_max_6", "motion_cumsum_12",
    # Motion streak & decay (backward)
    "motion_streak", "CO2_decay_rate",
    # Cross-sensor interactions (computed at time t from current values)
    "CO2_x_motion", "CO2_rising_while_motion", "CO2_falling_no_motion",
    "temp_RH_product", "motion_but_low_CO2", "high_CO2_no_motion",
    "sensor_agreement",
    # Suite features
    "suite_any_motion", "suite_max_CO2", "suite_zone_mismatch",
    # Cyclic time (deterministic — known at any time)
    "hour", "dayofweek", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "minute_of_day_sin", "minute_of_day_cos",
    # Hotel behavior (deterministic)
    "is_night", "is_checkin_window", "is_checkout_window",
    "hours_since_checkin_time", "hours_until_checkout_time",
    "checkout_CO2_drop", "checkin_CO2_rise",
    # External context (known/deterministic)
    "is_weekend", "is_public_holiday", "is_bridge_holiday", "is_long_weekend",
    "is_pattaya_major_event", "is_sriracha_local_event",
    "tourism_season_numeric", "tourism_macro_demand_proxy_score",
    "days_to_next_event", "days_since_prev_event",
    # Weather (Phase 2 — daily granularity, known)
    "total_energy_kWh", "outdoor_drybulb", "outdoor_RH", "outdoor_wetbulb",
    "is_hot_day", "cooling_load_proxy", "temp_delta_outdoor", "RH_delta_outdoor",
    # Room metadata
    "room_baseline_CO2", "room_baseline_temp", "room_baseline_RH",
    "room_occupancy_rate", "is_suite", "room_floor",
    # ASHRAE (derived from backward-looking logic)
    "ashrae_state",
]

# Features that need recomputation for forecast (they peek forward in current form)
RECOMPUTE_FOR_FORECAST = [
    "steps_since_motion",      # May use future data in cumulative count
    "steps_since_presence",    # Same issue
    "occupancy_duration",      # Counts full session length (needs future endpoint)
]

# Features to REMOVE for forecast (derived from target or use forward-fill of target)
REMOVE_FOR_FORECAST = [
    "Presence",                # This IS the target
    "presence_roll_max_6",     # Directly uses Presence values
    "occ_binary",              # Derived from Presence
    "occupancy",               # Derived from Presence (categorical)
    "ashrae_state_label",      # String label of ashrae_state
    "tourism_season_proxy",    # String — not numeric
    "tourism_macro_demand_proxy_label",  # String — not numeric
    "long_weekend_length",     # 89.6% NaN — too sparse
]

# Metadata columns (not features, but needed for splitting/grouping)
META_COLUMNS = ["timestamp", "room_area", "room_number", "room_zone"]


def load_and_validate(path: str) -> pd.DataFrame:
    """Load enriched parquet and validate basic integrity."""
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Unique rooms: {df['room_area'].nunique()}")
    return df


def filter_valid_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where Presence has a valid ground-truth label (0 or 1).
    Conservative approach: exclude NaN, code 3, and code 4."""
    before = len(df)
    mask = df["Presence"].isin(VALID_PRESENCE_VALUES)
    df_valid = df[mask].copy()
    after = len(df_valid)
    print(f"\nLabel filtering (conservative):")
    print(f"  Before: {before:,} rows")
    print(f"  After:  {after:,} rows ({after/before*100:.1f}%)")
    print(f"  Removed: {before - after:,} rows (Presence = NaN, 3, or 4)")
    print(f"  Class balance: Occupied={df_valid['Presence'].sum()/len(df_valid)*100:.1f}%, "
          f"Unoccupied={(1 - df_valid['Presence'].sum()/len(df_valid))*100:.1f}%")
    return df_valid


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create clean binary target column."""
    df["target"] = df["Presence"].astype(int)
    return df


def temporal_split(df: pd.DataFrame, train_cutoff: str, val_cutoff: str):
    """Split data temporally (per-room, same date cutoffs).
    This prevents data leakage for forecast models."""
    train_mask = df["timestamp"].dt.date <= pd.Timestamp(train_cutoff).date()
    val_mask = (df["timestamp"].dt.date > pd.Timestamp(train_cutoff).date()) & \
               (df["timestamp"].dt.date <= pd.Timestamp(val_cutoff).date())
    test_mask = df["timestamp"].dt.date > pd.Timestamp(val_cutoff).date()

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    print(f"\nTemporal split:")
    print(f"  Train: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%) "
          f"[{train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}]")
    print(f"  Val:   {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%) "
          f"[{val_df['timestamp'].min().date()} to {val_df['timestamp'].max().date()}]")
    print(f"  Test:  {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%) "
          f"[{test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}]")

    # Verify no leakage: max train timestamp < min val timestamp
    assert train_df["timestamp"].max() < val_df["timestamp"].min(), "Train/Val temporal leakage!"
    assert val_df["timestamp"].max() < test_df["timestamp"].min(), "Val/Test temporal leakage!"
    print("  ✓ No temporal leakage detected")

    return train_df, val_df, test_df


def recompute_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute features that may have used future data.
    Makes them strictly backward-looking for safe forecast usage."""
    print("\nRecomputing causal features...")

    # steps_since_motion: count steps since last motion=1 (backward only)
    # Already computed per-room with groupby + cumsum, which is backward-looking
    # But we verify by recomputing here
    for feat in RECOMPUTE_FOR_FORECAST:
        if feat == "steps_since_motion":
            def _steps_since(group):
                motion = group["motion_binary"].values
                result = np.zeros(len(motion), dtype=np.float32)
                counter = 0
                for i in range(len(motion)):
                    if motion[i] == 1:
                        counter = 0
                    else:
                        counter += 1
                    result[i] = counter
                return pd.Series(result, index=group.index)
            df["steps_since_motion_causal"] = df.groupby("room_area", observed=True).apply(
                _steps_since
            ).reset_index(level=0, drop=True)
            print(f"  ✓ {feat} → steps_since_motion_causal (recomputed, backward-only)")

        elif feat == "steps_since_presence":
            def _steps_since_pres(group):
                pres = group["Presence"].values
                result = np.zeros(len(pres), dtype=np.float32)
                counter = 0
                for i in range(len(pres)):
                    if pres[i] == 1.0:
                        counter = 0
                    else:
                        counter += 1
                    result[i] = counter
                return pd.Series(result, index=group.index)
            df["steps_since_presence_causal"] = df.groupby("room_area", observed=True).apply(
                _steps_since_pres
            ).reset_index(level=0, drop=True)
            print(f"  ✓ {feat} → steps_since_presence_causal (recomputed, backward-only)")

        elif feat == "occupancy_duration":
            def _occ_dur(group):
                pres = group["Presence"].values
                result = np.zeros(len(pres), dtype=np.float32)
                counter = 0
                for i in range(len(pres)):
                    if pres[i] == 1.0:
                        counter += 1
                    else:
                        counter = 0
                    result[i] = counter
                return pd.Series(result, index=group.index)
            df["occupancy_duration_causal"] = df.groupby("room_area", observed=True).apply(
                _occ_dur
            ).reset_index(level=0, drop=True)
            print(f"  ✓ {feat} → occupancy_duration_causal (recomputed, backward-only)")

    return df


def create_forecast_target(df: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    """Create forecast target: occupancy state at t+horizon (30 min = 6 steps).
    Shifted within each room to prevent cross-room leakage."""
    print(f"\nCreating forecast target (horizon = {horizon} steps = {horizon*5} min)...")
    df["target_forecast"] = df.groupby("room_area", observed=True)["target"].shift(-horizon)
    valid_forecast = df["target_forecast"].notna().sum()
    print(f"  Valid forecast labels: {valid_forecast:,} ({valid_forecast/len(df)*100:.1f}%)")
    print(f"  NaN (last {horizon} steps per room): {df['target_forecast'].isna().sum():,}")
    return df


def generate_feature_audit(df: pd.DataFrame, output_path: Path):
    """Generate a CSV audit report of all features and their safety status."""
    audit = []
    for col in df.columns:
        if col in META_COLUMNS or col in ["target", "target_forecast"]:
            status = "META"
        elif col in REMOVE_FOR_FORECAST:
            status = "REMOVE_FORECAST"
        elif col in RECOMPUTE_FOR_FORECAST:
            status = "RECOMPUTED"
        elif col in SAFE_FEATURES:
            status = "SAFE"
        elif col.endswith("_causal"):
            status = "SAFE_CAUSAL"
        else:
            status = "UNKNOWN"

        audit.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "nan_pct": df[col].isna().mean() * 100,
            "leakage_status": status,
        })

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(output_path, index=False)
    print(f"\nFeature audit saved to {output_path}")
    print(f"  SAFE: {(audit_df['leakage_status']=='SAFE').sum()}")
    print(f"  SAFE_CAUSAL: {(audit_df['leakage_status']=='SAFE_CAUSAL').sum()}")
    print(f"  RECOMPUTED: {(audit_df['leakage_status']=='RECOMPUTED').sum()}")
    print(f"  REMOVE_FORECAST: {(audit_df['leakage_status']=='REMOVE_FORECAST').sum()}")
    print(f"  META: {(audit_df['leakage_status']=='META').sum()}")


def save_splits(train_df, val_df, test_df, output_dir: Path):
    """Save train/val/test splits as parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    summary = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "total_rows": len(train_df) + len(val_df) + len(test_df),
        "train_date_range": [str(train_df["timestamp"].min()), str(train_df["timestamp"].max())],
        "val_date_range": [str(val_df["timestamp"].min()), str(val_df["timestamp"].max())],
        "test_date_range": [str(test_df["timestamp"].min()), str(test_df["timestamp"].max())],
        "train_class_balance": {
            "occupied": float(train_df["target"].mean()),
            "unoccupied": float(1 - train_df["target"].mean()),
        },
        "n_rooms": int(train_df["room_area"].nunique()),
        "n_features": len([c for c in train_df.columns if c not in META_COLUMNS + ["target", "target_forecast"]]),
        "forecast_horizon_steps": 6,
        "forecast_horizon_minutes": 30,
    }

    with open(output_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSplits saved to {output_dir}/")
    print(f"  train.parquet: {len(train_df):,} rows")
    print(f"  val.parquet:   {len(val_df):,} rows")
    print(f"  test.parquet:  {len(test_df):,} rows")


def main():
    print("=" * 70)
    print("DATA PREPARATION — Shared Pipeline for All Models")
    print("=" * 70)

    # Step 1: Load
    df = load_and_validate(INPUT_PATH)

    # Step 2: Filter valid labels only (conservative)
    df = filter_valid_labels(df)

    # Step 3: Create binary target
    df = create_binary_target(df)

    # Step 3.5: Drop rows where core sensors are NaN (fair comparison)
    core_sensors = ["CO2", "temp", "RH", "Motion"]
    before_cc = len(df)
    df = df.dropna(subset=core_sensors)
    after_cc = len(df)
    print(f"\nComplete cases filter (core sensors: {core_sensors}):")
    print(f"  Before: {before_cc:,} rows")
    print(f"  After:  {after_cc:,} rows ({after_cc/before_cc*100:.1f}%)")
    print(f"  Dropped: {before_cc - after_cc:,} rows with NaN in core sensors")

    # Step 4: Recompute causal features for forecast safety
    df = recompute_causal_features(df)

    # Step 5: Create forecast target (30 min ahead)
    df = create_forecast_target(df, horizon=6)

    # Step 6: Temporal split
    train_df, val_df, test_df = temporal_split(df, TRAIN_CUTOFF, VAL_CUTOFF)

    # Step 7: Generate feature audit
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_feature_audit(df, OUTPUT_DIR / "feature_audit.csv")

    # Step 8: Save
    save_splits(train_df, val_df, test_df, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
