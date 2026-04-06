"""
Feature Engineering — Advanced Features for Occupancy Detection & AC Control
=============================================================================
Hotel: Oakwood Hotel & Residence Sri Racha
Sensor Period: 2025-12-01 → 2026-02-15
Check-in: 14:00 / Check-out: 12:00

Tasks:
  1. Instantaneous features
  2. Temporal / behavioral features
  3. Cross-sensor interaction features
  4. Room-personalized baseline features (regular + suite)
  5. Cyclic time encoding
  6. Hotel check-in / check-out behavior features
  7. Holiday / event / tourism features
  8. Weather / comfort-context features
"""

import pandas as pd
import numpy as np
import gc
import warnings
import time

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
CLEANED_PATH   = "/Users/phasurab/Desktop/Alto_test/final_cleaned_phase1.csv"
HOLIDAY_PATH   = "/Users/phasurab/Desktop/Alto_test/th_holiday_event_macro_features_2023-05-31_to_2026-03-10.csv"
WEATHER_PATH   = "/Users/phasurab/Desktop/Alto_test/phase_2_dataset.csv"
OUTPUT_PATH    = "/Users/phasurab/Desktop/Alto_test/eda_feat_enriched.parquet"

print("=" * 60)
print("FEATURE ENGINEERING PIPELINE")
print("=" * 60)

# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 0 — Load cleaned data & pivot to wide                ║
# ╚══════════════════════════════════════════════════════════════╝
t0 = time.time()
print("\n[0/8] Loading cleaned long-format data...")

long_df = pd.read_csv(
    CLEANED_PATH,
    usecols=["timestamp", "room_area", "resolved_sensor_type",
             "quality_flag", "imputed_value", "outlier_flag", "n_candidates"],
    dtype={"imputed_value": "float32"},
    parse_dates=["timestamp"],
    low_memory=True,
)
for col in ["room_area", "resolved_sensor_type", "quality_flag"]:
    long_df[col] = long_df[col].astype("category")
print(f"  Loaded {long_df.shape[0]:,} rows in {time.time()-t0:.1f}s")

# Pivot to wide
print("  Pivoting long → wide...")
t1 = time.time()
valid = long_df[long_df["resolved_sensor_type"].notna()].copy()
wide = valid.pivot_table(
    index=["timestamp", "room_area"],
    columns="resolved_sensor_type",
    values="imputed_value",
    aggfunc="first",
    observed=True,
).reset_index()
wide.columns.name = None
del long_df, valid
gc.collect()
print(f"  Wide shape: {wide.shape}  ({time.time()-t1:.1f}s)")

# Sort for per-room temporal ops
eda = wide.sort_values(["room_area", "timestamp"]).reset_index(drop=True)
del wide
gc.collect()


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 0b — Occupancy labels                                ║
# ╚══════════════════════════════════════════════════════════════╝
print("  Creating occupancy labels...")
eda["occupancy"] = np.select(
    [eda["Presence"] == 1, eda["Presence"] == 0, eda["Presence"].isin([3, 4])],
    ["occupied", "unoccupied", "sensor_disconnected"],
    default="",
)
eda.loc[eda["occupancy"] == "", "occupancy"] = np.nan
eda["occupancy"] = eda["occupancy"].astype("category")

eda["occ_binary"] = np.where(
    eda["Presence"] == 1, 1.0,
    np.where(eda["Presence"] == 0, 0.0, np.nan)
).astype("float32")


# ╔══════════════════════════════════════════════════════════════╗
# ║  TASK 4 — Room-Personalized Baselines (compute early)      ║
# ║  (needed by Task 1)                                        ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[4/8] Computing room-personalized baselines...")
t4 = time.time()

# Identify suite rooms
all_rooms = eda["room_area"].unique()
suite_numbers = sorted(set(
    str(r).split("_")[1] for r in all_rooms if "living_room" in str(r)
))
eda["room_number"] = eda["room_area"].astype(str).str.extract(r"room_(\d+)_")[0]
eda["room_floor"]  = eda["room_number"].str[:-2].astype("int16")
eda["is_suite"]    = eda["room_number"].isin(suite_numbers).astype("int8")
eda["room_zone"]   = np.where(
    eda["room_area"].astype(str).str.contains("living_room"), "living_room", "bedroom"
)

# Per-room baselines from verified unoccupied periods
unoccupied_mask = eda["Presence"] == 0
room_baselines = (
    eda.loc[unoccupied_mask]
    .groupby("room_area", observed=True)
    .agg(
        room_baseline_CO2=("CO2", "median"),
        room_baseline_temp=("temp", "median"),
        room_baseline_RH=("RH", "median"),
    )
    .astype("float32")
)
eda = eda.merge(room_baselines, on="room_area", how="left")

# Per-room occupancy rate over the whole period
room_occ_rate = (
    eda.groupby("room_area", observed=True)["occ_binary"]
    .mean()
    .rename("room_occupancy_rate")
    .astype("float32")
)
eda = eda.merge(room_occ_rate, on="room_area", how="left")

print(f"  Suites found: {len(suite_numbers)}")
print(f"  Rooms with baselines: {room_baselines.shape[0]}")
print(f"  Done ({time.time()-t4:.1f}s)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  TASK 1 — Instantaneous Features                           ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[1/8] Computing instantaneous features...")
t1 = time.time()

eda["motion_binary"] = (eda["Motion"].fillna(0) > 0).astype("int8")

# ASHRAE-inspired CO2 differential above room's own baseline
eda["CO2_above_baseline"] = (eda["CO2"] - eda["room_baseline_CO2"]).astype("float32")
eda["temp_above_baseline"] = (eda["temp"] - eda["room_baseline_temp"]).astype("float32")

print(f"  Done ({time.time()-t1:.1f}s)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  TASK 2 — Temporal / Behavioral Features                   ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[2/8] Computing temporal/behavioral features...")
t2 = time.time()
g = eda.groupby("room_area", observed=True)

# ── Lags ──
for col in ["RH", "CO2", "temp"]:
    eda[f"{col}_lag1"] = g[col].shift(1).astype("float32")
eda["CO2_lag3"] = g["CO2"].shift(3).astype("float32")

# ── Diffs ──
for col in ["RH", "CO2", "temp"]:
    eda[f"{col}_diff1"] = g[col].diff(1).astype("float32")
eda["CO2_diff3"] = (eda["CO2"] - eda["CO2_lag3"]).astype("float32")

# ── Rolling stats ──
W6, W12 = 6, 12  # 30-min and 60-min at 5-min intervals
for col in ["CO2", "temp", "RH"]:
    eda[f"{col}_roll_mean_6"] = g[col].transform(
        lambda x: x.rolling(W6, min_periods=1).mean()
    ).astype("float32")

eda["CO2_roll_std_6"] = g["CO2"].transform(
    lambda x: x.rolling(W6, min_periods=1).std()
).astype("float32")

eda["CO2_roll_mean_12"] = g["CO2"].transform(
    lambda x: x.rolling(W12, min_periods=1).mean()
).astype("float32")

eda["RH_roll_std_6"] = g["RH"].transform(
    lambda x: x.rolling(W6, min_periods=1).std()
).astype("float32")

# ── Motion rolling features ──
eda["motion_roll_max_6"] = g["Motion"].transform(
    lambda x: x.rolling(W6, min_periods=1).max()
).astype("float32")

eda["motion_cumsum_12"] = g["motion_binary"].transform(
    lambda x: x.rolling(W12, min_periods=1).sum()
).astype("float32")

# ── Presence rolling max ──
eda["presence_roll_max_6"] = g["occ_binary"].transform(
    lambda x: x.rolling(W6, min_periods=1).max()
).astype("float32")

# ── Steps since last event ──
def _steps_since(s, condition_fn):
    active = condition_fn(s)
    grps = active.cumsum()
    return grps.groupby(grps).cumcount()

eda["steps_since_motion"] = g["Motion"].transform(
    lambda s: _steps_since(s, lambda x: x.fillna(0).gt(0))
).astype("float32")

eda["steps_since_presence"] = g["Presence"].transform(
    lambda s: _steps_since(s, lambda x: x.fillna(0).eq(1))
).astype("float32")

# ── Motion streak (consecutive steps with motion > 0) ──
def _streak(s):
    mask = s.fillna(0).gt(0)
    streaks = mask.groupby((~mask).cumsum()).cumsum()
    return streaks

eda["motion_streak"] = g["Motion"].transform(_streak).astype("float32")

# ── CO2 decay rate (slope when CO2 is dropping & no motion) ──
co2_falling_no_motion = (eda["CO2_diff1"] < 0) & (eda["motion_binary"] == 0)
eda["CO2_decay_rate"] = np.where(co2_falling_no_motion, eda["CO2_diff1"], 0.0).astype("float32")

# ── Occupancy duration (cumulative steps in current occ state) ──
def _occ_duration(s):
    changes = s.ne(s.shift()).cumsum()
    return changes.groupby(changes).cumcount() + 1

eda["occupancy_duration"] = g["occ_binary"].transform(_occ_duration).astype("float32")

print(f"  Done ({time.time()-t2:.1f}s)")
gc.collect()


# ╔══════════════════════════════════════════════════════════════╗
# ║  TASK 3 — Cross-Sensor Interaction Features                ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[3/8] Computing cross-sensor interaction features...")
t3 = time.time()

# CO2 × motion interaction
eda["CO2_x_motion"] = (
    eda["CO2_above_baseline"].fillna(0) * eda["motion_binary"]
).astype("float32")

# CO2 rising while motion active
eda["CO2_rising_while_motion"] = (
    (eda["CO2_diff1"].fillna(0) > 0) & (eda["motion_binary"] == 1)
).astype("int8")

# CO2 falling with no motion (post-checkout signal)
eda["CO2_falling_no_motion"] = (
    (eda["CO2_diff1"].fillna(0) < 0) & (eda["motion_binary"] == 0)
).astype("int8")

# Temp × RH product (comfort/enthalpy proxy)
eda["temp_RH_product"] = (eda["temp"] * eda["RH"]).astype("float32")

# Housekeeping / brief visit detector: motion but low CO2
eda["motion_but_low_CO2"] = (
    (eda["motion_binary"] == 1) & (eda["CO2"].fillna(9999) < 500)
).astype("int8")

# Sleeping guest detector: high CO2 but no motion
eda["high_CO2_no_motion"] = (
    (eda["CO2"].fillna(0) > 600) & (eda["motion_binary"] == 0)
).astype("int8")

# Sensor agreement: motion and presence agree
eda["sensor_agreement"] = (
    (eda["motion_binary"] == 1) == (eda["Presence"] == 1)
).astype("int8")

print(f"  Done ({time.time()-t3:.1f}s)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  TASK 4b — Suite-Specific Cross-Zone Features              ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[4b/8] Computing suite cross-zone features...")
t4b = time.time()

# For suites, create cross-zone features
suite_mask = eda["is_suite"] == 1
if suite_mask.sum() > 0:
    suite_data = eda.loc[suite_mask, ["timestamp", "room_number", "room_zone",
                                       "motion_binary", "CO2", "occ_binary"]].copy()

    # Pivot by zone within each suite
    bedroom = suite_data[suite_data["room_zone"] == "bedroom"].set_index(["timestamp", "room_number"])
    living  = suite_data[suite_data["room_zone"] == "living_room"].set_index(["timestamp", "room_number"])

    cross = bedroom[["motion_binary", "CO2"]].join(
        living[["motion_binary", "CO2"]],
        lsuffix="_bed", rsuffix="_liv",
        how="outer"
    ).reset_index()

    cross["suite_any_motion"] = (
        (cross["motion_binary_bed"].fillna(0) > 0) |
        (cross["motion_binary_liv"].fillna(0) > 0)
    ).astype("int8")

    cross["suite_max_CO2"] = np.fmax(
        cross["CO2_bed"].fillna(0), cross["CO2_liv"].fillna(0)
    ).astype("float32")

    cross["suite_zone_mismatch"] = (
        cross["motion_binary_bed"].fillna(0) != cross["motion_binary_liv"].fillna(0)
    ).astype("int8")

    # Merge back to main df
    cross_merge = cross[["timestamp", "room_number",
                          "suite_any_motion", "suite_max_CO2", "suite_zone_mismatch"]]
    eda = eda.merge(cross_merge, on=["timestamp", "room_number"], how="left")

    del suite_data, bedroom, living, cross, cross_merge
    gc.collect()
else:
    eda["suite_any_motion"] = np.nan
    eda["suite_max_CO2"] = np.nan
    eda["suite_zone_mismatch"] = np.nan

print(f"  Done ({time.time()-t4b:.1f}s)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  TASK 5 — Cyclic Time Encoding                             ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[5/8] Computing cyclic time features...")
t5 = time.time()

eda["hour"] = eda["timestamp"].dt.hour.astype("int8")
eda["dayofweek"] = eda["timestamp"].dt.dayofweek.astype("int8")
eda["month"] = eda["timestamp"].dt.month.astype("int8")
minute_of_day = eda["hour"] * 60 + eda["timestamp"].dt.minute

# Cyclic encoding
eda["hour_sin"] = np.sin(2 * np.pi * eda["hour"] / 24).astype("float32")
eda["hour_cos"] = np.cos(2 * np.pi * eda["hour"] / 24).astype("float32")
eda["dow_sin"]  = np.sin(2 * np.pi * eda["dayofweek"] / 7).astype("float32")
eda["dow_cos"]  = np.cos(2 * np.pi * eda["dayofweek"] / 7).astype("float32")
eda["month_sin"] = np.sin(2 * np.pi * eda["month"] / 12).astype("float32")
eda["month_cos"] = np.cos(2 * np.pi * eda["month"] / 12).astype("float32")
eda["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440).astype("float32")
eda["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440).astype("float32")

eda["is_night"] = ((eda["hour"] >= 22) | (eda["hour"] < 7)).astype("int8")

del minute_of_day
print(f"  Done ({time.time()-t5:.1f}s)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  TASK 6 — Hotel Check-in/Check-out Behavior Features       ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[6/8] Computing check-in/check-out behavior features...")
t6 = time.time()

eda["is_checkin_window"]  = ((eda["hour"] >= 14) & (eda["hour"] < 18)).astype("int8")
eda["is_checkout_window"] = ((eda["hour"] >= 10) & (eda["hour"] < 12)).astype("int8")

eda["hours_since_checkin_time"]  = ((eda["hour"] - 14) % 24).astype("int8")
eda["hours_until_checkout_time"] = ((12 - eda["hour"]) % 24).astype("int8")

# Checkout CO2 drop: average CO2 slope during 10:00-13:00 when motion stops
checkout_window = (eda["hour"] >= 10) & (eda["hour"] <= 13)
eda["checkout_CO2_drop"] = np.where(
    checkout_window & (eda["motion_binary"] == 0),
    eda["CO2_diff1"].fillna(0),
    0.0
).astype("float32")

# Checkin CO2 rise: average CO2 slope during 14:00-18:00 when motion starts
checkin_window = (eda["hour"] >= 14) & (eda["hour"] <= 18)
eda["checkin_CO2_rise"] = np.where(
    checkin_window & (eda["motion_binary"] == 1),
    eda["CO2_diff1"].fillna(0),
    0.0
).astype("float32")

print(f"  Done ({time.time()-t6:.1f}s)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  TASK 7 — Holiday / Event / Tourism Features               ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[7/8] Merging holiday/event/tourism features...")
t7 = time.time()

holidays = pd.read_csv(HOLIDAY_PATH, parse_dates=["date"])

# Select relevant columns
holiday_cols = [
    "date", "is_weekend", "is_public_holiday", "is_bridge_holiday",
    "is_long_weekend", "long_weekend_length",
    "is_pattaya_major_event", "is_sriracha_local_event",
    "tourism_season_proxy", "tourism_macro_demand_proxy_score",
    "tourism_macro_demand_proxy_label",
    "days_to_next_local_or_pattaya_event",
    "days_since_prev_local_or_pattaya_event",
]
holidays = holidays[holiday_cols].copy()
holidays.rename(columns={
    "days_to_next_local_or_pattaya_event": "days_to_next_event",
    "days_since_prev_local_or_pattaya_event": "days_since_prev_event",
}, inplace=True)

# Create date key for merge
eda["date"] = eda["timestamp"].dt.normalize()
holidays["date"] = pd.to_datetime(holidays["date"]).dt.normalize()

n_before = len(eda)
eda = eda.merge(holidays, on="date", how="left")
n_after = len(eda)
assert n_before == n_after, f"Row count changed during holiday merge! {n_before} → {n_after}"

matched_dates = eda["is_weekend"].notna().sum()
print(f"  Holiday merge: {matched_dates:,} / {len(eda):,} rows matched ({matched_dates/len(eda)*100:.1f}%)")

# Encode tourism_season_proxy as numeric
season_map = {"green_low_season": 0, "shoulder_season": 1, "peak_cool_season": 2}
eda["tourism_season_numeric"] = (
    eda["tourism_season_proxy"].map(season_map).astype("float32")
)

# Convert bool columns to int8
for col in ["is_weekend", "is_public_holiday", "is_bridge_holiday",
            "is_long_weekend", "is_pattaya_major_event", "is_sriracha_local_event"]:
    eda[col] = eda[col].astype("float32").astype("Int8")

del holidays
gc.collect()
print(f"  Done ({time.time()-t7:.1f}s)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  TASK 8 — Weather / Comfort-Context Features               ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[8/8] Merging weather/comfort features...")
t8 = time.time()

weather = pd.read_csv(WEATHER_PATH, parse_dates=["datetime"])
weather.rename(columns={
    "datetime": "date",
    "On-site Dry-Bulb Temperature (°C)": "outdoor_drybulb",
    "On-site Relative Humidity (%)": "outdoor_RH",
    "On-site Wet-Bulb Temperature (°C)": "outdoor_wetbulb",
    "Total Consumption (kWh)": "total_energy_kWh",
}, inplace=True)
weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()

# Derived weather features
weather["is_hot_day"] = (weather["outdoor_drybulb"] > 33).astype("int8")
weather["cooling_load_proxy"] = (
    weather["outdoor_drybulb"] * weather["outdoor_RH"] / 100
).astype("float32")

n_before = len(eda)
eda = eda.merge(weather, on="date", how="left")
n_after = len(eda)
assert n_before == n_after, f"Row count changed during weather merge! {n_before} → {n_after}"

# Indoor-outdoor differentials
eda["temp_delta_outdoor"] = (eda["temp"] - eda["outdoor_drybulb"]).astype("float32")
eda["RH_delta_outdoor"]   = (eda["RH"]   - eda["outdoor_RH"]).astype("float32")

matched_weather = eda["outdoor_drybulb"].notna().sum()
print(f"  Weather merge: {matched_weather:,} / {len(eda):,} rows matched ({matched_weather/len(eda)*100:.1f}%)")

del weather
gc.collect()
print(f"  Done ({time.time()-t8:.1f}s)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  ASHRAE 62.1 — State Machine Labels                        ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[ASHRAE] Computing state machine labels...")
ta = time.time()

# 3-state: UNOCCUPIED=0, OCCUPIED_STANDBY=1, OCCUPIED=2
co2_above = eda["CO2_above_baseline"].fillna(0)
co2_declining = eda["CO2_diff1"].fillna(0) < 0
motion_active = eda["motion_binary"] == 1
steps_no_motion = eda["steps_since_motion"].fillna(999)

eda["ashrae_state"] = np.select(
    [
        # OCCUPIED: motion recently OR CO2 significantly above baseline
        motion_active | (co2_above > 150),
        # OCCUPIED_STANDBY: no motion for 6 steps (30 min) but CO2 still above baseline
        (steps_no_motion >= 6) & (steps_no_motion < 12) & (co2_above > 50),
        # UNOCCUPIED: no motion for 12 steps (60 min) AND CO2 near baseline
        (steps_no_motion >= 12) & (co2_above <= 50),
    ],
    [2, 1, 0],
    default=0  # default to unoccupied
).astype("int8")

eda["ashrae_state_label"] = np.select(
    [eda["ashrae_state"] == 0, eda["ashrae_state"] == 1, eda["ashrae_state"] == 2],
    ["UNOCCUPIED", "OCCUPIED_STANDBY", "OCCUPIED"],
    default="UNOCCUPIED",
)

print(f"  State distribution:")
for state, cnt in eda["ashrae_state_label"].value_counts().items():
    print(f"    {state:25s} {cnt:>12,}  ({cnt/len(eda)*100:.1f}%)")
print(f"  Done ({time.time()-ta:.1f}s)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CLEANUP & SAVE                                            ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n[SAVE] Saving enriched dataset...")
ts = time.time()

# Drop intermediate columns
drop_cols = ["date"]
eda.drop(columns=[c for c in drop_cols if c in eda.columns], inplace=True)

# Summary
print(f"\n{'='*60}")
print(f"ENRICHED DATASET SUMMARY")
print(f"{'='*60}")
print(f"  Shape: {eda.shape}")
print(f"  Columns: {len(eda.columns)}")
print(f"  Memory: {eda.memory_usage(deep=True).sum() / 1e9:.2f} GB")
print(f"\n  Column list:")
for i, col in enumerate(sorted(eda.columns)):
    dtype = str(eda[col].dtype)
    null_pct = eda[col].isna().mean() * 100
    print(f"    {i+1:3d}. {col:40s} {dtype:12s} {null_pct:5.1f}% null")

# Save as parquet for efficiency
eda.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
print(f"\n  Saved to: {OUTPUT_PATH}")
print(f"  File size: {__import__('os').path.getsize(OUTPUT_PATH) / 1e9:.2f} GB")
print(f"  Save time: {time.time()-ts:.1f}s")
print(f"\n  Total pipeline time: {time.time()-t0:.1f}s")
print("=" * 60)
print("DONE!")
