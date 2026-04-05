"""
Segment-Level Model Evaluation
===============================
Evaluates each model's performance on 3 room segments:
1. Regular rooms (not suite, not severe sensor failure)
2. Suite rooms (is_suite=1, have living_room + bedroom)
3. Missing sensor rooms (>50% Presence code 3/4)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import json
from pathlib import Path
from sklearn.metrics import (
    classification_report, f1_score, cohen_kappa_score,
    roc_auc_score, brier_score_loss, precision_recall_curve, auc,
    fbeta_score
)

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "prepared_data"

# The 4 severe missing sensor rooms
MISSING_SENSOR_ROOMS = [
    "room_1002_bedroom",  # 86.4% code 3/4
    "room_1005_bedroom",  # 67.8% code 3/4
    "room_1602_bedroom",  # 55.9% code 3/4
    "room_1032_bedroom",  # 50.3% code 3/4
]


def identify_segments(test_df):
    """Split test data into 3 segments."""
    # Suite rooms
    suite_rooms = test_df[test_df["is_suite"] == 1]["room_area"].unique()
    # Missing sensor rooms
    missing_rooms = [r for r in MISSING_SENSOR_ROOMS if r in test_df["room_area"].unique()]
    # Regular rooms
    regular_mask = (~test_df["room_area"].isin(suite_rooms)) & \
                   (~test_df["room_area"].isin(missing_rooms))

    segments = {
        "Regular": test_df[regular_mask],
        "Suite": test_df[test_df["room_area"].isin(suite_rooms)],
        "Missing Sensor": test_df[test_df["room_area"].isin(missing_rooms)],
    }

    for name, seg in segments.items():
        n_rooms = seg["room_area"].nunique()
        print(f"  {name}: {len(seg):,} rows, {n_rooms} rooms, "
              f"occ_rate={seg['target'].mean():.3f}")

    return segments


def eval_lgb(model_path, X, y, task):
    """Evaluate LightGBM model on a segment."""
    model = lgb.Booster(model_file=str(model_path))
    y_prob = model.predict(X)
    y_pred = (y_prob >= 0.5).astype(int)

    valid = ~np.isnan(y)
    y_v = y[valid].astype(int)
    y_prob_v = y_prob[valid]
    y_pred_v = y_pred[valid]

    if len(np.unique(y_v)) < 2:
        return {"note": f"Only class {y_v[0]} present", "n_samples": len(y_v)}

    # Calculate F-beta scores
    # F0.5 for Unoccupied (class 0) -> weights precision over recall
    f05_unocc = fbeta_score(y_v, y_pred_v, beta=0.5, pos_label=0)
    # F2 for Occupied (class 1) -> weights recall over precision
    f2_occ = fbeta_score(y_v, y_pred_v, beta=2.0, pos_label=1)

    metrics = {
        "macro_f1": f1_score(y_v, y_pred_v, average="macro"),
        "recall_occ": (y_pred_v[y_v == 1] == 1).mean() if (y_v == 1).sum() > 0 else None,
        "kappa": cohen_kappa_score(y_v, y_pred_v),
        "roc_auc": roc_auc_score(y_v, y_prob_v),
        "f05_unocc": f05_unocc,
        "f2_occ": f2_occ,
        "n_samples": len(y_v),
    }
    if task == "forecast":
        metrics["brier"] = brier_score_loss(y_v, y_prob_v)
        prec, rec, _ = precision_recall_curve(y_v, y_prob_v)
        metrics["pr_auc"] = auc(rec, prec)

    return metrics


def eval_pytorch(model_path, X_npy, y_npy, room_mask, ModelClass, model_kwargs, task, device="cpu"):
    """Evaluate PyTorch model on a segment using room mask."""
    # We can't directly map window indices to rooms easily, so we return overall metrics
    # This is a limitation — would need room-level tracking in feature engineering
    return None


def main():
    print("=" * 70)
    print("SEGMENT-LEVEL MODEL EVALUATION")
    print("=" * 70)

    # Load test data
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    print(f"\nTest set: {len(test_df):,} rows, {test_df['room_area'].nunique()} rooms")

    segments = identify_segments(test_df)

    # ── M0 Baseline ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("M0 BASELINE — Per Segment")
    print("=" * 70)

    m0_feats_d = ["CO2", "temp", "RH", "Motion"]
    m0_feats_f = ["CO2", "temp", "RH", "Motion",
                  "target_lag_6", "target_lag_12", "target_lag_24", "target_lag_48"]

    m0_results = {}
    for seg_name, seg_df in segments.items():
        if len(seg_df) == 0:
            print(f"  {seg_name}: No data in test set")
            continue
        print(f"\n  --- {seg_name} ---")

        # Detection
        X_d = seg_df[m0_feats_d].values
        y_d = seg_df["target"].values
        det = eval_lgb(BASE_DIR / "m0_baseline/results/model_detect.txt", X_d, y_d, "detect")
        rec_str = f"{det.get('recall_occ', 0):.4f}" if det.get('recall_occ') is not None else "N/A"
        print(f"    Detection: F1={det.get('macro_f1', 0):.4f}, "
              f"Recall(Occ)={rec_str}, "
              f"F0.5(Unocc)={det.get('f05_unocc', 0):.4f}, "
              f"F2(Occ)={det.get('f2_occ', 0):.4f}")

        # Forecast
        fmask = seg_df["target_forecast"].notna()
        if fmask.sum() > 0:
            # Need to compute lagged targets
            seg_sorted = seg_df.copy()
            for lag in [6, 12, 24, 48]:
                col = f"target_lag_{lag}"
                if col not in seg_sorted.columns:
                    seg_sorted[col] = seg_sorted.groupby("room_area")["target"].shift(lag)
            X_f = seg_sorted.loc[fmask, m0_feats_f].values
            y_f = seg_sorted.loc[fmask, "target_forecast"].values
            fcast = eval_lgb(BASE_DIR / "m0_baseline/results/model_forecast.txt", X_f, y_f, "forecast")
            print(f"    Forecast:  F1={fcast.get('macro_f1', 'N/A'):.4f}, "
                  f"Brier={fcast.get('brier', 'N/A'):.4f}, "
                  f"PR-AUC={fcast.get('pr_auc', 'N/A'):.4f}")

        m0_results[seg_name] = {"detect": det, "forecast": fcast if fmask.sum() > 0 else {}}

    # ── M1 LightGBM+RF ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("M1 LightGBM+RF — Per Segment")
    print("=" * 70)

    with open(BASE_DIR / "m1_lightgbm/results/feature_selection.json") as f:
        m1_meta = json.load(f)

    m1_results = {}
    for seg_name, seg_df in segments.items():
        if len(seg_df) == 0:
            continue
        print(f"\n  --- {seg_name} ---")

        # Detection
        X_d = seg_df[m1_meta["detect_features"]].fillna(-999).values
        y_d = seg_df["target"].values
        det = eval_lgb(BASE_DIR / "m1_lightgbm/results/model_detect.txt", X_d, y_d, "detect")
        rec_str = f"{det.get('recall_occ', 0):.4f}" if det.get('recall_occ') is not None else "N/A"
        print(f"    Detection: F1={det.get('macro_f1', 0):.4f}, "
              f"Recall(Occ)={rec_str}, "
              f"F0.5(Unocc)={det.get('f05_unocc', 0):.4f}, "
              f"F2(Occ)={det.get('f2_occ', 0):.4f}")

        # Forecast
        fmask = seg_df["target_forecast"].notna()
        if fmask.sum() > 0:
            X_f = seg_df.loc[fmask, m1_meta["forecast_features"]].fillna(-999).values
            y_f = seg_df.loc[fmask, "target_forecast"].values
            fcast = eval_lgb(BASE_DIR / "m1_lightgbm/results/model_forecast.txt", X_f, y_f, "forecast")
            print(f"    Forecast:  F1={fcast.get('macro_f1', 'N/A'):.4f}, "
                  f"Brier={fcast.get('brier', 'N/A'):.4f}, "
                  f"PR-AUC={fcast.get('pr_auc', 'N/A'):.4f}")

        m1_results[seg_name] = {"detect": det, "forecast": fcast if fmask.sum() > 0 else {}}

    # ── Summary Table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SEGMENT SUMMARY — Detection Macro F1")
    print("=" * 70)
    print(f"{'Segment':<20} {'M0':>10} {'M1':>10}")
    print("-" * 42)
    for seg in ["Regular", "Suite", "Missing Sensor"]:
        m0_f1 = m0_results.get(seg, {}).get("detect", {}).get("macro_f1", "N/A")
        m1_f1 = m1_results.get(seg, {}).get("detect", {}).get("macro_f1", "N/A")
        m0_str = f"{m0_f1:.4f}" if isinstance(m0_f1, float) else str(m0_f1)
        m1_str = f"{m1_f1:.4f}" if isinstance(m1_f1, float) else str(m1_f1)
        print(f"{seg:<20} {m0_str:>10} {m1_str:>10}")

    print(f"\n{'Segment':<20} {'M0 Brier':>10} {'M1 Brier':>10}")
    print("-" * 42)
    for seg in ["Regular", "Suite", "Missing Sensor"]:
        m0_b = m0_results.get(seg, {}).get("forecast", {}).get("brier", "N/A")
        m1_b = m1_results.get(seg, {}).get("forecast", {}).get("brier", "N/A")
        m0_str = f"{m0_b:.4f}" if isinstance(m0_b, float) else str(m0_b)
        m1_str = f"{m1_b:.4f}" if isinstance(m1_b, float) else str(m1_b)
        print(f"{seg:<20} {m0_str:>10} {m1_str:>10}")

    # Save results
    results = {"M0": m0_results, "M1": m1_results}
    with open(BASE_DIR / "comparison_results/segment_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to comparison_results/segment_metrics.json")


if __name__ == "__main__":
    main()
