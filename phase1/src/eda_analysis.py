"""
Enhanced EDA & Feature Analysis
================================
Produces visualizations and feature importance from the enriched dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
import os

warnings.filterwarnings("ignore")

# ── Config ──
ENRICHED_PATH = "/Users/phasurab/Desktop/Alto_test/eda_feat_enriched.parquet"
PLOT_DIR = "/Users/phasurab/Desktop/Alto_test/eda_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("ggplot")

plt.rcParams.update({
    "figure.figsize": (14, 6), "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12, "figure.dpi": 120,
})
COLORS = {"occupied": "#e74c3c", "unoccupied": "#3498db", "uncertain": "#95a5a6"}


# ── Load ──
print("Loading enriched dataset...")
eda = pd.read_parquet(ENRICHED_PATH)
print(f"  Shape: {eda.shape}")

# Filter to labeled data only for comparisons
mask_labeled = eda["occupancy"].isin(["occupied", "unoccupied"])
compare = eda.loc[mask_labeled].copy()
print(f"  Labeled rows for comparison: {len(compare):,}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  1. NEW FEATURE DISTRIBUTIONS: Occupied vs Unoccupied      ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[1] Distribution plots for new features...")

new_features = [
    ("CO2_above_baseline", "CO2 Above Room Baseline (ppm)"),
    ("temp_above_baseline", "Temp Above Room Baseline (°C)"),
    ("CO2_diff3", "CO2 15-min Trend (Δ3)"),
    ("motion_cumsum_12", "Motion Intensity (60-min sum)"),
    ("motion_streak", "Motion Streak (consecutive steps)"),
    ("CO2_decay_rate", "CO2 Decay Rate (ppm/step)"),
    ("RH_roll_std_6", "RH Variability (30-min std)"),
    ("occupancy_duration", "Occupancy State Duration (steps)"),
    ("CO2_x_motion", "CO2 × Motion Interaction"),
    ("temp_delta_outdoor", "Indoor-Outdoor Temp Δ (°C)"),
    ("temp_RH_product", "Temp × RH Product"),
    ("sensor_agreement", "Sensor Agreement (Motion ↔ Presence)"),
]

plot_sample = compare.sample(n=min(500_000, len(compare)), random_state=42)

fig, axes = plt.subplots(4, 3, figsize=(20, 20))
for idx, (feat, title) in enumerate(new_features):
    ax = axes.flat[idx]
    for label, color in [("occupied", COLORS["occupied"]), ("unoccupied", COLORS["unoccupied"])]:
        vals = plot_sample.loc[plot_sample["occupancy"] == label, feat].dropna()
        if len(vals) > 0:
            q01, q99 = vals.quantile(0.01), vals.quantile(0.99)
            vals_clip = vals[(vals >= q01) & (vals <= q99)]
            if len(vals_clip.unique()) > 1:
                sns.kdeplot(vals_clip, ax=ax, label=label, color=color, fill=True, alpha=0.3)
            else:
                ax.bar([label], [len(vals_clip)], color=color, alpha=0.5)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)

fig.suptitle("Occupied vs Unoccupied — New Feature Distributions", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_new_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_new_feature_distributions.png")

del plot_sample
gc.collect()


# ╔══════════════════════════════════════════════════════════════╗
# ║  2. CROSS-SENSOR INTERACTION PATTERNS                      ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[2] Cross-sensor interaction analysis...")

interaction_features = [
    "CO2_rising_while_motion", "CO2_falling_no_motion",
    "motion_but_low_CO2", "high_CO2_no_motion", "sensor_agreement",
]

cross_tab = compare.groupby("occupancy", observed=True)[interaction_features].mean().round(3)
print("\n  Cross-Sensor Interaction Rates (mean):")
print(cross_tab.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
cross_tab.T.plot.barh(ax=ax, color=[COLORS["occupied"], COLORS["unoccupied"]], edgecolor="white")
ax.set_title("Cross-Sensor Interaction Feature Rates by Occupancy")
ax.set_xlabel("Rate (proportion of timestamps)")
ax.legend(title="Occupancy")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_cross_sensor_interactions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_cross_sensor_interactions.png")


# ╔══════════════════════════════════════════════════════════════╗
# ║  3. HOTEL CHECK-IN/CHECKOUT PATTERN                        ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[3] Check-in/check-out hourly patterns...")

hourly_occ = compare.groupby("hour")["occ_binary"].mean()
hourly_motion = compare.groupby("hour")["motion_binary"].mean()
hourly_co2_above = compare.groupby("hour")["CO2_above_baseline"].mean()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.bar(hourly_occ.index, hourly_occ.values, color="#e74c3c", alpha=0.7, edgecolor="white")
ax.axvline(14, color="green", ls="--", lw=2, label="Check-in (14:00)")
ax.axvline(12, color="blue", ls="--", lw=2, label="Check-out (12:00)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Occupancy Rate")
ax.set_title("Occupancy Rate by Hour")
ax.legend(fontsize=9)

ax = axes[1]
ax.bar(hourly_motion.index, hourly_motion.values, color="#27ae60", alpha=0.7, edgecolor="white")
ax.axvline(14, color="green", ls="--", lw=2, label="Check-in")
ax.axvline(12, color="blue", ls="--", lw=2, label="Check-out")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Motion Rate")
ax.set_title("Motion Detection Rate by Hour")
ax.legend(fontsize=9)

ax = axes[2]
ax.plot(hourly_co2_above.index, hourly_co2_above.values, "o-", color="#8e44ad", lw=2)
ax.axvline(14, color="green", ls="--", lw=2, label="Check-in")
ax.axvline(12, color="blue", ls="--", lw=2, label="Check-out")
ax.axhline(0, color="gray", ls=":", alpha=0.5)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("CO2 Above Baseline (ppm)")
ax.set_title("CO2 Above Room Baseline by Hour")
ax.legend(fontsize=9)

fig.suptitle("Hotel Behavior — Check-in 14:00 / Check-out 12:00", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_checkin_checkout_patterns.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_checkin_checkout_patterns.png")


# ╔══════════════════════════════════════════════════════════════╗
# ║  4. FEATURE IMPORTANCE (Random Forest)                     ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[4] Feature importance analysis (Random Forest)...")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Select numeric features for importance
feature_cols = [
    "CO2", "RH", "temp", "CO2_above_baseline", "temp_above_baseline",
    "CO2_diff1", "CO2_diff3", "temp_diff1", "RH_diff1",
    "CO2_roll_mean_6", "CO2_roll_std_6", "CO2_roll_mean_12",
    "RH_roll_std_6", "motion_binary", "motion_cumsum_12", "motion_streak",
    "motion_roll_max_6", "steps_since_motion", "steps_since_presence",
    "CO2_decay_rate", "occupancy_duration",
    "CO2_x_motion", "CO2_rising_while_motion", "CO2_falling_no_motion",
    "motion_but_low_CO2", "high_CO2_no_motion", "sensor_agreement",
    "temp_RH_product", "temp_delta_outdoor", "RH_delta_outdoor",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "minute_of_day_sin", "minute_of_day_cos",
    "is_night", "is_checkin_window", "is_checkout_window",
    "hours_since_checkin_time", "hours_until_checkout_time",
    "is_weekend", "is_public_holiday",
    "tourism_macro_demand_proxy_score",
    "outdoor_drybulb", "outdoor_RH", "cooling_load_proxy",
    "room_occupancy_rate", "is_suite", "room_floor",
    "is_hot_day",
]

# Subsample for speed
N_RF = min(200_000, len(compare))
rf_sample = compare.sample(n=N_RF, random_state=42)
X_rf = rf_sample[feature_cols].astype("float32")
y_rf = (rf_sample["occ_binary"] == 1).astype(int)

# Drop rows with any NaN
valid_mask = X_rf.notna().all(axis=1) & y_rf.notna()
X_rf = X_rf[valid_mask]
y_rf = y_rf[valid_mask]
print(f"  Training RF on {len(X_rf):,} samples, {len(feature_cols)} features...")

rf = RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_leaf=50,
    random_state=42, n_jobs=-1, class_weight="balanced",
)
rf.fit(X_rf, y_rf)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n  Top-20 Features by Importance:")
for i, (feat, imp) in enumerate(importances.head(20).items()):
    marker = "★" if imp > 0.05 else "·"
    print(f"    {i+1:2d}. {marker} {feat:40s} {imp:.4f}")

# Classification report
y_pred = rf.predict(X_rf)
print("\n  In-sample Classification Report:")
print(classification_report(y_rf, y_pred, target_names=["unoccupied", "occupied"]))

# Plot importance
fig, ax = plt.subplots(figsize=(10, 12))
top25 = importances.head(25)
colors = ["#e74c3c" if v > 0.05 else "#3498db" if v > 0.02 else "#95a5a6" for v in top25.values]
top25.plot.barh(ax=ax, color=colors, edgecolor="white")
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Top-25 Feature Importance — Random Forest Classifier", fontsize=14)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_feature_importance.png")

del rf, X_rf, y_rf, rf_sample
gc.collect()


# ╔══════════════════════════════════════════════════════════════╗
# ║  5. CORRELATION HEATMAP                                    ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[5] Correlation heatmap of key features...")

corr_cols = importances.head(20).index.tolist()
corr_sample = compare.sample(n=min(200_000, len(compare)), random_state=42)
corr_matrix = corr_sample[corr_cols].astype("float32").corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, square=True, linewidths=0.5,
            cbar_kws={"label": "Pearson Correlation"})
ax.set_title("Correlation Heatmap — Top-20 Features", fontsize=14)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 05_correlation_heatmap.png")

del corr_sample
gc.collect()


# ╔══════════════════════════════════════════════════════════════╗
# ║  6. ASHRAE STATE MACHINE — Example Room Visualization      ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[6] ASHRAE state machine visualization...")

# Find a well-behaved room with occupancy transitions
coverage = (
    eda.groupby("room_area", observed=True).agg(
        n_rows=("timestamp", "count"),
        co2_cov=("CO2", lambda x: x.notna().mean()),
        motion_cov=("Motion", lambda x: x.notna().mean()),
        presence_cov=("Presence", lambda x: x.notna().mean()),
        occ_ratio=("occ_binary", "mean"),
    )
    .query("co2_cov > 0.85 and motion_cov > 0.85 and presence_cov > 0.85")
    .query("occ_ratio > 0.1 and occ_ratio < 0.9")
    .sort_values("n_rows", ascending=False)
)

if len(coverage) > 0:
    example_room = coverage.index[0]
    print(f"  Example room: {example_room}")

    rdf = eda[eda["room_area"] == example_room].copy()
    t_start = rdf["timestamp"].min()
    rdf = rdf[rdf["timestamp"] <= t_start + pd.Timedelta(days=4)]

    state_colors = {0: "#3498db", 1: "#f39c12", 2: "#e74c3c"}
    state_labels_map = {0: "UNOCCUPIED", 1: "STANDBY", 2: "OCCUPIED"}

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

    # CO2 with ASHRAE state background
    ax0 = axes[0]
    ax0.plot(rdf["timestamp"], rdf["CO2"], color="#2c3e50", lw=1, label="CO2 (ppm)")
    ax0.axhline(rdf["room_baseline_CO2"].iloc[0], color="gray", ls=":", alpha=0.7, label="Room Baseline")
    for state_val, state_color in state_colors.items():
        mask = rdf["ashrae_state"] == state_val
        if mask.sum() > 0:
            ax0.fill_between(rdf["timestamp"], rdf["CO2"].min(), rdf["CO2"].max(),
                           where=mask, alpha=0.1, color=state_color, label=state_labels_map[state_val])
    ax0.set_ylabel("CO2 (ppm)")
    ax0.legend(loc="upper right", fontsize=8, ncol=2)
    ax0.set_title(f"{example_room} — ASHRAE State Machine (4-day view)", fontsize=13)

    # CO2 above baseline
    ax1 = axes[1]
    ax1.fill_between(rdf["timestamp"], 0, rdf["CO2_above_baseline"], alpha=0.5,
                     where=rdf["CO2_above_baseline"] > 0, color="#e74c3c", label="Above baseline")
    ax1.fill_between(rdf["timestamp"], 0, rdf["CO2_above_baseline"], alpha=0.5,
                     where=rdf["CO2_above_baseline"] <= 0, color="#3498db", label="Below baseline")
    ax1.axhline(150, color="red", ls="--", lw=1, alpha=0.7, label="ASHRAE occupied threshold (+150)")
    ax1.axhline(50, color="orange", ls="--", lw=1, alpha=0.7, label="ASHRAE standby threshold (+50)")
    ax1.set_ylabel("CO2 Δ Baseline (ppm)")
    ax1.legend(fontsize=8)

    # Motion
    ax2 = axes[2]
    ax2.bar(rdf["timestamp"], rdf["Motion"].fillna(0), width=0.003, color="#27ae60", alpha=0.7)
    ax2.set_ylabel("Motion")
    ax2.set_ylim(-0.1, 1.5)

    # ASHRAE State
    ax3 = axes[3]
    for state_val, state_color in state_colors.items():
        mask = rdf["ashrae_state"] == state_val
        ax3.fill_between(rdf["timestamp"], 0, 1, where=mask, alpha=0.7,
                        color=state_color, label=state_labels_map[state_val])
    ax3.set_ylabel("ASHRAE State")
    ax3.set_xlabel("Time")
    ax3.set_yticks([])
    ax3.legend(loc="upper right", fontsize=9, ncol=3)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/06_ashrae_state_machine.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 06_ashrae_state_machine.png")
else:
    print("  WARNING: No rooms with sufficient coverage found for visualization.")


# ╔══════════════════════════════════════════════════════════════╗
# ║  7. SUITE vs REGULAR ROOM COMPARISON                       ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[7] Suite vs Regular room comparison...")

suite_compare = compare.copy()
suite_compare["room_type"] = np.where(suite_compare["is_suite"] == 1, "Suite", "Regular")

suite_stats = suite_compare.groupby("room_type").agg(
    mean_CO2=("CO2", "mean"),
    mean_temp=("temp", "mean"),
    occ_rate=("occ_binary", "mean"),
    mean_motion=("motion_binary", "mean"),
    mean_CO2_above_base=("CO2_above_baseline", "mean"),
).round(3)
print("\n  Suite vs Regular Room Stats:")
print(suite_stats.to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = [("CO2_above_baseline", "CO2 Above Baseline (ppm)"),
           ("motion_cumsum_12", "Motion Intensity (60-min)"),
           ("occupancy_duration", "Occupancy Duration (steps)")]

plot_suite_sample = suite_compare.sample(n=min(300_000, len(suite_compare)), random_state=42)
for idx, (feat, title) in enumerate(metrics):
    ax = axes[idx]
    for rt, color in [("Regular", "#3498db"), ("Suite", "#e74c3c")]:
        vals = plot_suite_sample.loc[plot_suite_sample["room_type"] == rt, feat].dropna()
        if len(vals) > 0:
            q01, q99 = vals.quantile(0.01), vals.quantile(0.99)
            vals_clip = vals[(vals >= q01) & (vals <= q99)]
            if len(vals_clip.unique()) > 1:
                sns.kdeplot(vals_clip, ax=ax, label=rt, color=color, fill=True, alpha=0.3)
    ax.set_title(title)
    ax.legend()

fig.suptitle("Suite Rooms vs Regular Rooms — Feature Comparison", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_suite_vs_regular.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 07_suite_vs_regular.png")


# ╔══════════════════════════════════════════════════════════════╗
# ║  8. HOLIDAY / WEATHER IMPACT                               ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[8] Holiday & weather impact analysis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Occupancy on holidays vs normal days
ax = axes[0]
holiday_occ = compare.groupby("is_public_holiday")["occ_binary"].mean()
ax.bar(["Normal Day", "Public Holiday"], holiday_occ.values,
       color=["#3498db", "#e74c3c"], edgecolor="white", alpha=0.8)
ax.set_ylabel("Occupancy Rate")
ax.set_title("Occupancy: Normal vs Public Holiday")
for i, v in enumerate(holiday_occ.values):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11)

# (b) Occupancy vs outdoor temperature
ax = axes[1]
temp_bins = pd.cut(compare["outdoor_drybulb"], bins=8)
temp_occ = compare.groupby(temp_bins, observed=True)["occ_binary"].mean()
temp_occ.plot.bar(ax=ax, color="#8e44ad", edgecolor="white", alpha=0.8)
ax.set_xlabel("Outdoor Temperature (°C)")
ax.set_ylabel("Occupancy Rate")
ax.set_title("Occupancy vs Outdoor Temperature")
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

# (c) Weekend vs weekday occupancy by hour
ax = axes[2]
for wk, color, label in [(0, "#3498db", "Weekday"), (1, "#e74c3c", "Weekend")]:
    hourly = compare[compare["is_weekend"] == wk].groupby("hour")["occ_binary"].mean()
    ax.plot(hourly.index, hourly.values, "o-", color=color, label=label, lw=2)
ax.axvline(14, color="green", ls="--", alpha=0.5, label="Check-in")
ax.axvline(12, color="orange", ls="--", alpha=0.5, label="Check-out")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Occupancy Rate")
ax.set_title("Weekday vs Weekend Occupancy by Hour")
ax.legend(fontsize=9)

fig.suptitle("External Factor Impact on Occupancy", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/08_holiday_weather_impact.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 08_holiday_weather_impact.png")

print("\n" + "=" * 60)
print("ALL EDA PLOTS SAVED!")
print(f"Output directory: {PLOT_DIR}")
print("=" * 60)
