"""
Model Comparison — Side-by-Side Summary
=========================================
Compares all 5 model pipelines across detection and forecast tasks.
"""

import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "comparison_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "M0_baseline": BASE_DIR / "m0_baseline" / "results" / "metrics.json",
    "M1_lightgbm": BASE_DIR / "m1_lightgbm" / "results" / "metrics.json",
    "M2_patchtst": BASE_DIR / "m2_patchtst" / "results" / "metrics.json",
    "M3_inception": BASE_DIR / "m3_inceptiontime" / "results" / "metrics.json",
    "M4_gaf_cnn": BASE_DIR / "m4_gaf_efficientnet" / "results" / "metrics.json",
}


def load_all_metrics():
    """Load metrics from all model result files."""
    all_data = {}
    for name, path in MODELS.items():
        if path.exists():
            with open(path) as f:
                all_data[name] = json.load(f)
        else:
            print(f"  ⚠ {name} metrics not found at {path}")
    return all_data


def create_detection_table(all_data):
    """Create detection comparison table."""
    rows = []
    for name, data in all_data.items():
        det = data.get("detection", {})
        if not det:
            continue
        rows.append({
            "Model": name,
            "Recall (Occ)": det.get("recall_occupied", "N/A"),
            "Prec (Unocc)": det.get("precision_unoccupied", "N/A"),
            "Macro F1": det.get("macro_f1", "N/A"),
            "Cohen κ": det.get("cohen_kappa", "N/A"),
            "ROC-AUC": det.get("roc_auc", "N/A"),
            "Inference (ms)": det.get("inference_time_ms_per_sample", "N/A"),
            "Train (s)": det.get("train_time_s", "N/A"),
        })
    return pd.DataFrame(rows)


def create_forecast_table(all_data):
    """Create forecast comparison table."""
    rows = []
    for name, data in all_data.items():
        fcast = data.get("forecast", {})
        if not fcast:
            continue
        rows.append({
            "Model": name,
            "Brier": fcast.get("brier_score", "N/A"),
            "Log Loss": fcast.get("log_loss", "N/A"),
            "PR-AUC": fcast.get("pr_auc", "N/A"),
            "Recall (Occ)": fcast.get("recall_occupied", "N/A"),
            "Macro F1": fcast.get("macro_f1", "N/A"),
            "Cohen κ": fcast.get("cohen_kappa", "N/A"),
            "Inference (ms)": fcast.get("inference_time_ms_per_sample", "N/A"),
        })
    return pd.DataFrame(rows)


def create_size_table(all_data):
    """Create model size comparison."""
    rows = []
    for name, data in all_data.items():
        rows.append({
            "Model": name,
            "Detect Size (MB)": data.get("detection_model_size_MB", "N/A"),
            "Forecast Size (MB)": data.get("forecast_model_size_MB", "N/A"),
            "# Detect Features": data.get("n_features_detect", "N/A"),
            "# Forecast Features": data.get("n_features_forecast", "N/A"),
        })
    return pd.DataFrame(rows)


def plot_comparison(det_df, fcast_df, output_dir):
    """Create comparison bar charts."""
    # Detection comparison
    if len(det_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        metrics = [("Recall (Occ)", "≥0.98"), ("Macro F1", "≥0.90"), ("Cohen κ", "≥0.80")]
        colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"]

        for ax, (metric, target) in zip(axes, metrics):
            vals = pd.to_numeric(det_df[metric], errors="coerce")
            bars = ax.bar(det_df["Model"], vals, color=colors[:len(det_df)])
            ax.set_title(f"Detection: {metric} (target {target})")
            ax.set_ylim(0, 1.1)
            ax.tick_params(axis="x", rotation=45)
            # Add target line
            target_val = float(target.replace("≥", ""))
            ax.axhline(y=target_val, color="gray", linestyle="--", alpha=0.7)
            # Add value labels
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / "detection_comparison.png", dpi=150)
        plt.close()

    # Forecast comparison
    if len(fcast_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        f_metrics = [("Brier", "≤0.15"), ("PR-AUC", "≥0.85"), ("Macro F1", "")]
        colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"]

        for ax, (metric, target) in zip(axes, f_metrics):
            vals = pd.to_numeric(fcast_df[metric], errors="coerce")
            bars = ax.bar(fcast_df["Model"], vals, color=colors[:len(fcast_df)])
            ax.set_title(f"Forecast: {metric}" + (f" (target {target})" if target else ""))
            if metric == "Brier":
                ax.set_ylim(0, 0.5)
            else:
                ax.set_ylim(0, 1.1)
            ax.tick_params(axis="x", rotation=45)
            if target:
                if "≤" in target:
                    target_val = float(target.replace("≤", ""))
                else:
                    target_val = float(target.replace("≥", ""))
                ax.axhline(y=target_val, color="gray", linestyle="--", alpha=0.7)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / "forecast_comparison.png", dpi=150)
        plt.close()


def main():
    print("=" * 60)
    print("MODEL COMPARISON — All 5 Pipelines")
    print("=" * 60)

    all_data = load_all_metrics()

    # Detection table
    det_df = create_detection_table(all_data)
    print("\n" + "=" * 60)
    print("DETECTION COMPARISON")
    print("=" * 60)
    print(det_df.to_string(index=False))

    # Forecast table
    fcast_df = create_forecast_table(all_data)
    print("\n" + "=" * 60)
    print("FORECAST COMPARISON (+30 min)")
    print("=" * 60)
    print(fcast_df.to_string(index=False))

    # Size table
    size_df = create_size_table(all_data)
    print("\n" + "=" * 60)
    print("MODEL SIZE COMPARISON")
    print("=" * 60)
    print(size_df.to_string(index=False))

    # Save tables
    det_df.to_csv(OUTPUT_DIR / "detection_comparison.csv", index=False)
    fcast_df.to_csv(OUTPUT_DIR / "forecast_comparison.csv", index=False)
    size_df.to_csv(OUTPUT_DIR / "size_comparison.csv", index=False)

    # Plots
    plot_comparison(det_df, fcast_df, OUTPUT_DIR)

    # Final recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    # Find best detection model by Macro F1
    det_df["Macro F1 num"] = pd.to_numeric(det_df["Macro F1"], errors="coerce")
    best_det = det_df.loc[det_df["Macro F1 num"].idxmax(), "Model"]
    best_det_f1 = det_df["Macro F1 num"].max()

    # Find best forecast model by Brier Score (lower is better)
    fcast_df["Brier num"] = pd.to_numeric(fcast_df["Brier"], errors="coerce")
    best_fcast = fcast_df.loc[fcast_df["Brier num"].idxmin(), "Model"]
    best_fcast_brier = fcast_df["Brier num"].min()

    print(f"\n  Best Detection:  {best_det} (Macro F1 = {best_det_f1:.4f})")
    print(f"  Best Forecast:   {best_fcast} (Brier = {best_fcast_brier:.4f})")
    print(f"\n  Recommended production pipeline: {best_det} for detection,")
    print(f"  {best_fcast} for forecast → 5-state hotel control logic")

    print(f"\n  All results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
