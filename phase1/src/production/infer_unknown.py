"""
Infer Unknown Presence — Detection Only
=========================================
For every row where Presence ∈ {3, 4, NaN}:
  1. Identify room type via gateway
  2. Load the room-type-specific M1 model + threshold
  3. Compute features (fill NaN → -999; zero motion for missing-sensor)
  4. Run detection inference → P(occupied)
  5. Apply optimal threshold → binary 0/1
  6. Validate via proxy checks (CO2, temporal consistency)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from room_gateway import classify_all_rooms, get_segment_rooms, zero_motion_features

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "inference_output"


def load_models_and_thresholds():
    """Load per-segment models and their optimal thresholds."""
    config = {}
    for seg in ["regular", "suite", "missing_sensor"]:
        seg_dir = RESULTS_DIR / seg
        model = lgb.Booster(model_file=str(seg_dir / "model_detect.txt"))
        with open(seg_dir / "threshold.json") as f:
            meta = json.load(f)

        config[seg] = {
            "model": model,
            "threshold": meta["optimal_threshold"],
            "features": meta.get("features", model.feature_name()),
        }
        print(f"  {seg}: threshold={meta['optimal_threshold']:.2f}")

    return config


def infer_segment(df_segment, model, features, threshold, zero_motion=False):
    """Run detection inference on a dataframe segment.

    Returns the segment with added prediction columns.
    """
    df = df_segment.copy()

    if zero_motion:
        df = zero_motion_features(df)

    # Keep only features that exist in the data
    available = [f for f in features if f in df.columns]
    X = df[available].fillna(-999)

    # Predict
    y_prob = model.predict(X)
    y_pred = (y_prob >= threshold).astype(int)

    # Confidence: distance from threshold, normalized to [0, 1]
    distance = np.abs(y_prob - threshold)
    max_dist = max(threshold, 1 - threshold)
    confidence = np.clip(distance / max_dist, 0, 1)

    df["occupancy_probability"] = y_prob
    df["predicted_occupancy"] = y_pred
    df["confidence"] = confidence
    df["confidence_level"] = np.where(confidence >= 0.5, "high", "low")

    return df


def validate_predictions(df_inferred, output_dir):
    """Run proxy validation checks on the inferred predictions."""
    print(f"\n{'='*60}")
    print("VALIDATION — Proxy Checks")
    print(f"{'='*60}")

    results = {}

    # 1. Prediction distribution
    occ_rate = df_inferred["predicted_occupancy"].mean()
    prob_mean = df_inferred["occupancy_probability"].mean()
    prob_std = df_inferred["occupancy_probability"].std()
    print(f"\n  Prediction Distribution:")
    print(f"    Predicted occupancy rate: {occ_rate:.3f}")
    print(f"    Probability mean: {prob_mean:.3f} ± {prob_std:.3f}")
    print(f"    High confidence: {(df_inferred['confidence_level']=='high').mean()*100:.1f}%")
    results["occ_rate"] = occ_rate
    results["prob_mean"] = prob_mean
    results["high_confidence_pct"] = float((df_inferred["confidence_level"] == "high").mean())

    # 2. Temporal consistency (flip-flop rate)
    flip_rates = []
    for room in df_inferred["room_area"].unique():
        room_data = df_inferred[df_inferred["room_area"] == room].sort_values("timestamp")
        preds = room_data["predicted_occupancy"].values
        if len(preds) < 2:
            continue
        flips = (preds[1:] != preds[:-1]).sum()
        flip_rate = flips / (len(preds) - 1)
        flip_rates.append(flip_rate)

    avg_flip = np.mean(flip_rates) if flip_rates else 0
    print(f"\n  Temporal Consistency:")
    print(f"    Average flip-flop rate: {avg_flip:.3f} ({avg_flip*100:.1f}% of timesteps)")
    print(f"    Interpretation: {'⚠ High noise' if avg_flip > 0.20 else '✓ Stable'}")
    results["avg_flip_rate"] = avg_flip

    # 3. CO2 agreement
    if "CO2" in df_inferred.columns and "room_baseline_CO2" in df_inferred.columns:
        pred_occ = df_inferred["predicted_occupancy"] == 1
        co2_valid = df_inferred["CO2"].notna() & df_inferred["room_baseline_CO2"].notna()
        mask = pred_occ & co2_valid
        if mask.sum() > 0:
            co2_above = (df_inferred.loc[mask, "CO2"] >
                         df_inferred.loc[mask, "room_baseline_CO2"]).mean()
            print(f"\n  CO2 Agreement:")
            print(f"    When predicted occupied, CO2 above baseline: {co2_above*100:.1f}%")
            print(f"    Interpretation: {'✓ Good' if co2_above > 0.60 else '⚠ Low agreement'}")
            results["co2_agreement"] = co2_above

    # 4. Day/night pattern
    if "hour" in df_inferred.columns or "is_night" in df_inferred.columns:
        if "is_night" in df_inferred.columns:
            night = df_inferred["is_night"] == 1
        else:
            night = df_inferred["hour"].isin(range(0, 6))
        night_occ = df_inferred.loc[night, "predicted_occupancy"].mean()
        day_occ = df_inferred.loc[~night, "predicted_occupancy"].mean()
        print(f"\n  Day/Night Pattern:")
        print(f"    Night occupancy rate: {night_occ:.3f}")
        print(f"    Day occupancy rate:   {day_occ:.3f}")
        print(f"    Interpretation: {'✓ Expected' if night_occ > day_occ else '⚠ Unusual'}")
        results["night_occ_rate"] = night_occ
        results["day_occ_rate"] = day_occ

    # 5. Per-segment stats
    print(f"\n  Per-Segment Prediction Rates:")
    for seg in df_inferred["prediction_source"].unique():
        seg_data = df_inferred[df_inferred["prediction_source"] == seg]
        rate = seg_data["predicted_occupancy"].mean()
        n = len(seg_data)
        print(f"    {seg}: {rate:.3f} occ rate ({n:,} rows)")

    # Save
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Histogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Probability distribution
    axes[0].hist(df_inferred["occupancy_probability"], bins=50, color="#3498DB", alpha=0.7, edgecolor="white")
    axes[0].set_xlabel("P(Occupied)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Prediction Probability Distribution")

    # Per-segment occupancy rate
    seg_rates = df_inferred.groupby("prediction_source")["predicted_occupancy"].mean()
    axes[1].barh(seg_rates.index, seg_rates.values, color=["#E74C3C", "#3498DB", "#2ECC71"])
    axes[1].set_xlabel("Predicted Occupancy Rate")
    axes[1].set_title("Occ Rate by Room Type")

    # Confidence distribution
    axes[2].hist(df_inferred["confidence"], bins=50, color="#2ECC71", alpha=0.7, edgecolor="white")
    axes[2].set_xlabel("Confidence")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Prediction Confidence Distribution")

    plt.tight_layout()
    plt.savefig(output_dir / "inference_validation.png", dpi=150)
    plt.close()

    return results


def main():
    print("=" * 70)
    print("PRODUCTION INFERENCE — Predict Unknown Presence (Detection Only)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load full dataset (including all Presence codes)
    print("\nLoading full dataset...")
    df = pd.read_parquet(BASE_DIR / "eda_feat_enriched.parquet")
    print(f"  Total: {len(df):,} rows, {df['room_area'].nunique()} rooms")

    # Recompute causal features that are generated by data_preparation.py
    # but don't exist in the raw enriched parquet
    print("\nRecomputing causal features...")

    def _steps_since_motion(group):
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

    if "steps_since_motion_causal" not in df.columns:
        df["steps_since_motion_causal"] = df.groupby(
            "room_area", observed=True
        ).apply(_steps_since_motion).reset_index(level=0, drop=True)
        print("  ✓ steps_since_motion_causal computed")

    # Separate known vs unknown
    known_mask = df["Presence"].isin([0.0, 1.0])
    unknown_mask = df["Presence"].isin([3.0, 4.0]) | df["Presence"].isna()

    df_known = df[known_mask].copy()
    df_unknown = df[unknown_mask].copy()
    print(f"  Known (0,1):     {len(df_known):,} ({len(df_known)/len(df)*100:.1f}%)")
    print(f"  Unknown (3,4,NaN): {len(df_unknown):,} ({len(df_unknown)/len(df)*100:.1f}%)")

    if len(df_unknown) == 0:
        print("No unknown rows to infer. Exiting.")
        return

    # Load models & thresholds
    print("\nLoading per-segment models...")
    config = load_models_and_thresholds()

    # Classify rooms
    room_map = classify_all_rooms(df)

    # Run inference per segment
    inferred_parts = []
    for seg_name in ["regular", "suite", "missing_sensor"]:
        seg_rooms = get_segment_rooms(room_map, seg_name)
        seg_data = df_unknown[df_unknown["room_area"].isin(seg_rooms)]

        if len(seg_data) == 0:
            print(f"\n  {seg_name}: 0 unknown rows — skipping")
            continue

        print(f"\n  Inferring {seg_name}: {len(seg_data):,} rows, "
              f"{seg_data['room_area'].nunique()} rooms")

        is_missing = seg_name == "missing_sensor"
        result = infer_segment(
            seg_data,
            config[seg_name]["model"],
            config[seg_name]["features"],
            config[seg_name]["threshold"],
            zero_motion=is_missing,
        )
        result["prediction_source"] = f"model_{seg_name}"
        inferred_parts.append(result)

    # Combine
    df_inferred = pd.concat(inferred_parts, ignore_index=True)
    print(f"\nTotal inferred: {len(df_inferred):,} rows")

    # Also add ground-truth rows for reference
    df_known["occupancy_probability"] = df_known["Presence"].astype(float)
    df_known["predicted_occupancy"] = df_known["Presence"].astype(int)
    df_known["confidence"] = 1.0
    df_known["confidence_level"] = "ground_truth"
    df_known["prediction_source"] = "ground_truth"

    # Combined full dataset
    df_full = pd.concat([df_known, df_inferred], ignore_index=True)
    df_full = df_full.sort_values(["room_area", "timestamp"]).reset_index(drop=True)
    print(f"Full dataset with predictions: {len(df_full):,} rows")

    # Validate
    validation = validate_predictions(df_inferred, OUTPUT_DIR)

    # Save outputs
    print(f"\nSaving outputs...")

    # Save just the inferred rows (compact)
    inferred_cols = ["timestamp", "room_area", "Presence",
                     "occupancy_probability", "predicted_occupancy",
                     "confidence", "confidence_level", "prediction_source"]
    available_cols = [c for c in inferred_cols if c in df_inferred.columns]
    df_inferred[available_cols].to_parquet(OUTPUT_DIR / "inferred_predictions.parquet", index=False)
    print(f"  Inferred predictions: {OUTPUT_DIR / 'inferred_predictions.parquet'}")

    # Save full combined dataset
    full_cols = available_cols + ["CO2", "temp", "RH", "Motion",
                                  "is_suite", "is_night", "hour"]
    full_available = [c for c in full_cols if c in df_full.columns]
    df_full[full_available].to_parquet(OUTPUT_DIR / "full_predictions.parquet", index=False)
    print(f"  Full predictions:     {OUTPUT_DIR / 'full_predictions.parquet'}")

    # Summary
    print(f"\n{'='*70}")
    print("INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"  Rows inferred:    {len(df_inferred):,}")
    print(f"  Predicted occ:    {df_inferred['predicted_occupancy'].mean():.3f}")
    print(f"  High confidence:  {(df_inferred['confidence_level']=='high').mean()*100:.1f}%")
    print(f"  Flip-flop rate:   {validation.get('avg_flip_rate', 0):.3f}")
    print(f"  CO2 agreement:    {validation.get('co2_agreement', 0)*100:.1f}%")
    print(f"  Output saved to:  {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
