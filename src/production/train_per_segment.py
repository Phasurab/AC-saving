"""
Train Per-Segment — M1 LightGBM with threshold optimization
=============================================================
Trains separate M1 LightGBM models for regular and suite rooms,
then sweeps classification thresholds to optimize segment-specific
business metrics:
  - Regular  → F0.5 (Unoccupied) — confident energy savings
  - Suite    → F2 (Occupied) — never miss a VIP guest
  - Missing  → uses regular model with conservative Brier-optimal threshold
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import time
import os
import sys
from pathlib import Path
from sklearn.metrics import (
    fbeta_score, f1_score, cohen_kappa_score, roc_auc_score,
    brier_score_loss, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from room_gateway import classify_all_rooms, get_segment_rooms, zero_motion_features, MOTION_FEATURES

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "prepared_data"
M1_RESULTS = BASE_DIR / "m1_lightgbm" / "results"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 42

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": 5.0,
    "verbose": -1,
    "seed": SEED,
    "n_jobs": -1,
    "min_child_samples": 100,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

# Threshold sweep grid
THRESHOLDS = np.arange(0.05, 0.96, 0.05)


def load_feature_meta():
    """Load the feature list from M1's feature selection."""
    with open(M1_RESULTS / "feature_selection.json") as f:
        meta = json.load(f)
    return meta["detect_features"]


def filter_segment_data(df: pd.DataFrame, room_list: list) -> pd.DataFrame:
    """Filter dataframe to only include rooms in the given list."""
    return df[df["room_area"].isin(room_list)].copy()


def sweep_threshold(y_true, y_prob, metric_fn, metric_name, higher_is_better=True):
    """Sweep thresholds and find the optimal one for the given metric."""
    results = []
    for t in THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        score = metric_fn(y_true, y_pred)
        results.append({"threshold": round(t, 2), metric_name: score})

    results_df = pd.DataFrame(results)
    if higher_is_better:
        best_idx = results_df[metric_name].idxmax()
    else:
        best_idx = results_df[metric_name].idxmin()

    best = results_df.loc[best_idx]
    return float(best["threshold"]), float(best[metric_name]), results_df


def train_and_optimize(segment_name, train_df, val_df, test_df,
                       features, metric_fn, metric_name,
                       higher_is_better, results_subdir):
    """Train M1 LightGBM on a segment and find the optimal threshold."""
    print(f"\n{'='*60}")
    print(f"  SEGMENT: {segment_name.upper()}")
    print(f"  Metric: {metric_name}")
    print(f"={'='*60}")

    results_subdir.mkdir(parents=True, exist_ok=True)

    # Filter features that exist in the data
    available_feats = [f for f in features if f in train_df.columns]
    missing = [f for f in features if f not in train_df.columns]
    if missing:
        print(f"  ⚠ Missing features (skipped): {missing}")
    print(f"  Using {len(available_feats)} features")

    X_train = train_df[available_feats].fillna(-999)
    y_train = train_df["target"].values
    X_val = val_df[available_feats].fillna(-999)
    y_val = val_df["target"].values
    X_test = test_df[available_feats].fillna(-999)
    y_test = test_df["target"].values

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Train class balance: {y_train.mean():.3f} occupied")

    # ── Train ──────────────────────────────────────────────
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(200),
    ]

    start = time.time()
    model = lgb.train(
        LGB_PARAMS, train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_data],
        callbacks=callbacks,
    )
    train_time = time.time() - start
    print(f"  Train time: {train_time:.1f}s, best iter: {model.best_iteration}")

    # ── Threshold Sweep on Validation ───────────────────────
    y_val_prob = model.predict(X_val)
    best_threshold, best_score, sweep_df = sweep_threshold(
        y_val, y_val_prob, metric_fn, metric_name, higher_is_better
    )
    print(f"\n  Threshold sweep ({metric_name}):")
    print(f"    Best threshold: {best_threshold}")
    print(f"    Best {metric_name}: {best_score:.4f}")
    print(f"    Default (0.5) {metric_name}: "
          f"{metric_fn(y_val, (y_val_prob >= 0.5).astype(int)):.4f}")

    # ── Evaluate on Test with Optimal Threshold ─────────────
    y_test_prob = model.predict(X_test)
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    test_metric = metric_fn(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    test_kappa = cohen_kappa_score(y_test, y_test_pred)
    test_roc = roc_auc_score(y_test, y_test_prob)
    test_brier = brier_score_loss(y_test, y_test_prob)

    # Also compute both F-beta scores for comparison
    test_f05_unocc = fbeta_score(y_test, y_test_pred, beta=0.5, pos_label=0)
    test_f2_occ = fbeta_score(y_test, y_test_pred, beta=2.0, pos_label=1)
    test_recall_occ = (y_test_pred[y_test == 1] == 1).mean()

    print(f"\n  Test Results (threshold={best_threshold}):")
    print(classification_report(y_test, y_test_pred,
                                target_names=["Unoccupied", "Occupied"]))
    print(f"    {metric_name}: {test_metric:.4f}")
    print(f"    F0.5(Unocc): {test_f05_unocc:.4f}")
    print(f"    F2(Occ):     {test_f2_occ:.4f}")
    print(f"    Recall(Occ): {test_recall_occ:.4f}")
    print(f"    Macro F1:    {test_f1:.4f}")
    print(f"    Cohen κ:     {test_kappa:.4f}")
    print(f"    ROC-AUC:     {test_roc:.4f}")
    print(f"    Brier:       {test_brier:.4f}")

    # ── Save ───────────────────────────────────────────────
    model.save_model(str(results_subdir / "model_detect.txt"))

    threshold_meta = {
        "segment": segment_name,
        "optimization_metric": metric_name,
        "optimal_threshold": best_threshold,
        f"val_{metric_name}": best_score,
        f"test_{metric_name}": test_metric,
        "test_f05_unocc": test_f05_unocc,
        "test_f2_occ": test_f2_occ,
        "test_recall_occ": test_recall_occ,
        "test_macro_f1": test_f1,
        "test_kappa": test_kappa,
        "test_roc_auc": test_roc,
        "test_brier": test_brier,
        "train_time_s": train_time,
        "best_iteration": model.best_iteration,
        "features": available_feats,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }

    with open(results_subdir / "threshold.json", "w") as f:
        json.dump(threshold_meta, f, indent=2, default=str)

    # Threshold sweep plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sweep_df["threshold"], sweep_df[metric_name], "o-", color="#E74C3C", lw=2)
    ax.axvline(best_threshold, color="#3498DB", ls="--", lw=1.5,
               label=f"Optimal: {best_threshold} ({best_score:.4f})")
    ax.axvline(0.5, color="#95A5A6", ls=":", lw=1, label="Default: 0.5")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{segment_name.upper()} — Threshold vs {metric_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_subdir / "threshold_sweep.png", dpi=150)
    plt.close()

    print(f"\n  ✓ Saved model + threshold to {results_subdir}/")
    return model, best_threshold, threshold_meta


def main():
    print("=" * 70)
    print("PRODUCTION PIPELINE — Train Per-Segment M1 + Threshold Optimization")
    print("=" * 70)

    # Load data
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df   = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df  = pd.read_parquet(DATA_DIR / "test.parquet")
    features = load_feature_meta()

    print(f"\nLoaded splits: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    print(f"M1 features ({len(features)}): {features}")

    # Classify rooms
    room_map = classify_all_rooms(train_df)
    regular_rooms = get_segment_rooms(room_map, "regular")
    suite_rooms = get_segment_rooms(room_map, "suite")
    missing_rooms = get_segment_rooms(room_map, "missing_sensor")

    print(f"\nRoom counts: regular={len(regular_rooms)}, suite={len(suite_rooms)}, "
          f"missing={len(missing_rooms)}")

    all_results = {}

    # ── 1. REGULAR ROOMS ────────────────────────────────────────
    train_reg = filter_segment_data(train_df, regular_rooms)
    val_reg   = filter_segment_data(val_df, regular_rooms)
    test_reg  = filter_segment_data(test_df, regular_rooms)

    def f05_unocc(y_true, y_pred):
        return fbeta_score(y_true, y_pred, beta=0.5, pos_label=0)

    model_reg, thresh_reg, meta_reg = train_and_optimize(
        "regular", train_reg, val_reg, test_reg,
        features, f05_unocc, "f05_unocc",
        higher_is_better=True,
        results_subdir=RESULTS_DIR / "regular",
    )
    all_results["regular"] = meta_reg

    # ── 2. SUITE ROOMS ──────────────────────────────────────────
    train_suite = filter_segment_data(train_df, suite_rooms)
    val_suite   = filter_segment_data(val_df, suite_rooms)
    test_suite  = filter_segment_data(test_df, suite_rooms)

    def f2_occ(y_true, y_pred):
        return fbeta_score(y_true, y_pred, beta=2.0, pos_label=1)

    model_suite, thresh_suite, meta_suite = train_and_optimize(
        "suite", train_suite, val_suite, test_suite,
        features, f2_occ, "f2_occ",
        higher_is_better=True,
        results_subdir=RESULTS_DIR / "suite",
    )
    all_results["suite"] = meta_suite

    # ── 3. MISSING SENSOR ROOMS ─────────────────────────────────
    # Use regular model but with motion features zeroed, optimize for Brier
    print(f"\n{'='*60}")
    print(f"  SEGMENT: MISSING_SENSOR")
    print(f"  Strategy: Regular model + zero motion + Brier-optimal threshold")
    print(f"={'='*60}")

    results_missing = RESULTS_DIR / "missing_sensor"
    results_missing.mkdir(parents=True, exist_ok=True)

    # No separate training — we reuse the regular model
    # But we need to find the best threshold when motion is zeroed
    # Test on regular val set with motion zeroed (simulating degraded input)
    val_degraded = zero_motion_features(val_reg)
    available_feats = [f for f in features if f in val_degraded.columns]
    X_val_deg = val_degraded[available_feats].fillna(-999)
    y_val_deg = val_degraded["target"].values

    y_val_prob_deg = model_reg.predict(X_val_deg)

    # Sweep threshold for best Brier score
    brier_results = []
    for t in THRESHOLDS:
        y_pred_t = (y_val_prob_deg >= t).astype(int)
        brier = brier_score_loss(y_val_deg, y_val_prob_deg)  # Brier doesn't depend on threshold
        f05 = fbeta_score(y_val_deg, y_pred_t, beta=0.5, pos_label=0)
        f2 = fbeta_score(y_val_deg, y_pred_t, beta=2.0, pos_label=1)
        recall_occ = (y_pred_t[y_val_deg == 1] == 1).mean()
        brier_results.append({
            "threshold": round(t, 2), "brier": brier,
            "f05_unocc": f05, "f2_occ": f2, "recall_occ": recall_occ,
        })

    brier_df = pd.DataFrame(brier_results)
    # For missing sensor rooms, prioritize safety: use a threshold that keeps
    # Recall(Occ) > 0.95 while maximizing F0.5(Unocc)
    safe_df = brier_df[brier_df["recall_occ"] >= 0.95]
    if len(safe_df) > 0:
        best_idx = safe_df["f05_unocc"].idxmax()
        best_missing = safe_df.loc[best_idx]
    else:
        # If no threshold gives >95% recall with degraded data, use very conservative
        best_missing = brier_df.loc[brier_df["f2_occ"].idxmax()]

    thresh_missing = float(best_missing["threshold"])
    print(f"\n  Missing sensor threshold: {thresh_missing}")
    print(f"    Recall(Occ): {best_missing['recall_occ']:.4f}")
    print(f"    F0.5(Unocc): {best_missing['f05_unocc']:.4f}")
    print(f"    Brier: {best_missing['brier']:.4f}")

    # Save — copy the regular model as the missing sensor model
    import shutil
    shutil.copy(
        str(RESULTS_DIR / "regular" / "model_detect.txt"),
        str(results_missing / "model_detect.txt"),
    )

    missing_meta = {
        "segment": "missing_sensor",
        "strategy": "regular_model_with_motion_zeroed",
        "optimization_metric": "f05_unocc (with recall_occ >= 0.95 constraint)",
        "optimal_threshold": thresh_missing,
        "val_recall_occ": float(best_missing["recall_occ"]),
        "val_f05_unocc": float(best_missing["f05_unocc"]),
        "val_brier": float(best_missing["brier"]),
        "zeroed_features": MOTION_FEATURES,
        "note": "Reuses regular model. Motion features set to -999 at inference.",
    }

    with open(results_missing / "threshold.json", "w") as f:
        json.dump(missing_meta, f, indent=2, default=str)

    all_results["missing_sensor"] = missing_meta
    print(f"  ✓ Saved threshold to {results_missing}/")

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY — Per-Segment Thresholds")
    print(f"{'='*70}")
    print(f"  {'Segment':<20} {'Threshold':>10} {'Primary Metric':>20} {'Score':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Regular':<20} {thresh_reg:>10.2f} {'F0.5(Unocc)':>20} "
          f"{meta_reg['test_f05_unocc']:>10.4f}")
    print(f"  {'Suite':<20} {thresh_suite:>10.2f} {'F2(Occ)':>20} "
          f"{meta_suite['test_f2_occ']:>10.4f}")
    print(f"  {'Missing Sensor':<20} {thresh_missing:>10.2f} {'F0.5 (safe)':>20} "
          f"{best_missing['f05_unocc']:>10.4f}")

    # Save combined summary
    with open(RESULTS_DIR / "segment_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✓ All models and thresholds saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
