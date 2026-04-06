"""
M1 LightGBM — Feature Engineering with RF Feature Selection
=============================================================
Uses Random Forest to discover the most important features from
all 93 engineered features, then selects the top subset for
LightGBM detection and forecast models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "prepared_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
AUDIT_PATH = DATA_DIR / "feature_audit.csv"

TOP_K_DETECT = 25    # top features for detection
TOP_K_FORECAST = 20  # top causal features for forecast
SAMPLE_SIZE = 100_000  # subsample for RF (speed)
SEED = 42

# Columns that are metadata / target / PROXY LEAKERS — not features
EXCLUDE_COLS = [
    "timestamp", "room_area", "room_number", "room_zone",
    "target", "target_forecast",
    "Presence", "occ_binary", "occupancy",
    "presence_roll_max_6",
    "ashrae_state_label", "tourism_season_proxy",
    "tourism_macro_demand_proxy_label", "long_weekend_length",
    # PROXY LEAKERS: derived directly from Presence target
    "occupancy_duration",            # counts how long Presence=1
    "occupancy_duration_causal",     # same, recomputed
    "steps_since_presence",          # steps since Presence=1
    "steps_since_presence_causal",   # same, recomputed
    "sensor_agreement",              # uses Presence as input
]

# Additional columns to exclude for FORECAST (leakage risk)
FORECAST_EXCLUDE = [
    "steps_since_motion", "steps_since_motion_causal",
]


def get_feature_columns(df: pd.DataFrame, for_forecast: bool = False):
    """Get valid numeric feature columns, excluding metadata and targets."""
    exclude = set(EXCLUDE_COLS)
    if for_forecast:
        exclude.update(FORECAST_EXCLUDE)

    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype in ["object", "category", "str"]:
            continue
        cols.append(c)
    return cols


def run_rf_importance(X_train, y_train, feature_names, task_name: str):
    """Train a Random Forest and extract feature importance."""
    print(f"\n  Training Random Forest for {task_name}...")
    print(f"    Subsampled: {len(X_train):,} rows × {X_train.shape[1]} features")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=SEED,
        class_weight={0: 1, 1: 5},  # Asymmetric cost
    )
    rf.fit(X_train, y_train)

    # Gini importance
    gini_imp = pd.DataFrame({
        "feature": feature_names,
        "gini_importance": rf.feature_importances_,
    }).sort_values("gini_importance", ascending=False)

    print(f"    RF accuracy (train subsample): {rf.score(X_train, y_train):.4f}")
    print(f"    Top 5 features: {list(gini_imp['feature'].head(5))}")

    return gini_imp, rf


def select_top_features(importance_df: pd.DataFrame, top_k: int):
    """Select top-K features by importance."""
    selected = importance_df.head(top_k)["feature"].tolist()
    return selected


def main():
    print("=" * 60)
    print("M1 LightGBM — Feature Engineering (RF Selection)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    # ── Detection Feature Discovery ────────────────────────────
    print("\n" + "─" * 50)
    print("DETECTION — Feature Discovery")
    detect_cols = get_feature_columns(train_df, for_forecast=False)
    print(f"  Candidate features: {len(detect_cols)}")

    # Subsample for speed
    sample_idx = train_df.sample(n=min(SAMPLE_SIZE, len(train_df)),
                                 random_state=SEED).index
    X_sample = train_df.loc[sample_idx, detect_cols].fillna(-999)
    y_sample = train_df.loc[sample_idx, "target"].values

    gini_detect, rf_detect = run_rf_importance(
        X_sample, y_sample, detect_cols, "Detection"
    )
    selected_detect = select_top_features(gini_detect, TOP_K_DETECT)
    print(f"\n  Selected {TOP_K_DETECT} detection features:")
    for i, f in enumerate(selected_detect):
        imp = gini_detect[gini_detect["feature"] == f]["gini_importance"].values[0]
        print(f"    {i+1:2d}. {f:40s} {imp:.4f}")

    # ── Forecast Feature Discovery ─────────────────────────────
    print("\n" + "─" * 50)
    print("FORECAST — Feature Discovery")
    forecast_cols = get_feature_columns(train_df, for_forecast=True)
    # Add causal versions
    for c in ["steps_since_motion_causal", "steps_since_presence_causal",
              "occupancy_duration_causal"]:
        if c in train_df.columns and c not in forecast_cols:
            forecast_cols.append(c)
    print(f"  Candidate features: {len(forecast_cols)}")

    # Forecast target
    forecast_valid = train_df["target_forecast"].notna()
    sample_idx_f = train_df[forecast_valid].sample(
        n=min(SAMPLE_SIZE, forecast_valid.sum()), random_state=SEED
    ).index
    X_sample_f = train_df.loc[sample_idx_f, forecast_cols].fillna(-999)
    y_sample_f = train_df.loc[sample_idx_f, "target_forecast"].values.astype(int)

    gini_forecast, rf_forecast = run_rf_importance(
        X_sample_f, y_sample_f, forecast_cols, "Forecast"
    )
    selected_forecast = select_top_features(gini_forecast, TOP_K_FORECAST)
    print(f"\n  Selected {TOP_K_FORECAST} forecast features:")
    for i, f in enumerate(selected_forecast):
        imp = gini_forecast[gini_forecast["feature"] == f]["gini_importance"].values[0]
        print(f"    {i+1:2d}. {f:40s} {imp:.4f}")

    # ── Save Feature Importance Plots ──────────────────────────
    # Detection importance
    top30_d = gini_detect.head(30).sort_values("gini_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#E74C3C" if f in selected_detect else "#BDC3C7"
              for f in top30_d["feature"]]
    ax.barh(top30_d["feature"], top30_d["gini_importance"], color=colors)
    ax.set_xlabel("Gini Importance")
    ax.set_title(f"M1 — RF Feature Importance (Detection)\nRed = Selected Top {TOP_K_DETECT}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rf_importance_detect.png", dpi=150)
    plt.close()

    # Forecast importance
    top30_f = gini_forecast.head(30).sort_values("gini_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#3498DB" if f in selected_forecast else "#BDC3C7"
              for f in top30_f["feature"]]
    ax.barh(top30_f["feature"], top30_f["gini_importance"], color=colors)
    ax.set_xlabel("Gini Importance")
    ax.set_title(f"M1 — RF Feature Importance (Forecast +30min)\nBlue = Selected Top {TOP_K_FORECAST}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rf_importance_forecast.png", dpi=150)
    plt.close()

    # ── Prepare and Save Feature Data ──────────────────────────
    print("\n" + "─" * 50)
    print("Exporting prepared data...")

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        # Detection
        X_det = df[selected_detect].copy()
        y_det = df["target"].values
        X_det.to_parquet(OUTPUT_DIR / f"{split_name}_detect_X.parquet", index=False)
        pd.DataFrame({"target": y_det}).to_parquet(
            OUTPUT_DIR / f"{split_name}_detect_y.parquet", index=False
        )

        # Forecast
        forecast_mask = df["target_forecast"].notna()
        X_fcast = df.loc[forecast_mask, selected_forecast].copy()
        y_fcast = df.loc[forecast_mask, "target_forecast"].values
        X_fcast.to_parquet(OUTPUT_DIR / f"{split_name}_forecast_X.parquet", index=False)
        pd.DataFrame({"target_forecast": y_fcast}).to_parquet(
            OUTPUT_DIR / f"{split_name}_forecast_y.parquet", index=False
        )

        print(f"  {split_name}: detect={X_det.shape}, forecast={X_fcast.shape}")

    # Save feature selection metadata
    selection_meta = {
        "detect_features": selected_detect,
        "forecast_features": selected_forecast,
        "n_detect": len(selected_detect),
        "n_forecast": len(selected_forecast),
        "importance_method": "Random Forest Gini Importance",
        "rf_n_estimators": 200,
        "rf_subsample_size": SAMPLE_SIZE,
    }
    with open(OUTPUT_DIR / "feature_selection.json", "w") as f:
        json.dump(selection_meta, f, indent=2)

    gini_detect.to_csv(OUTPUT_DIR / "importance_detect.csv", index=False)
    gini_forecast.to_csv(OUTPUT_DIR / "importance_forecast.csv", index=False)

    print(f"\n✓ M1 feature engineering complete. Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
