"""
Phase 2 M&V Pipeline — End-to-End Baseline Model & Savings Analysis
====================================================================
IPMVP Option C | ASHRAE Guideline 14 | Credibility-First Framework

This script reproduces the entire Phase 2 analysis:
  1. Data loading & merging (on-site + Si Racha weather)
  2. Feature engineering (CDD, seasonality, weekday)
  3. Baseline model training (Ridge, GAM, XGBoost, LightGBM)
  4. Holdout validation & ASHRAE compliance check
  5. Placebo test (credibility verification)
  6. Savings estimation (counterfactual prediction)
  7. Cross-model comparison & visualization

Usage:
  python notebooks/01_baseline_and_savings.py

Outputs saved to: outputs/charts/ and outputs/data/
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_CHARTS = os.path.join(PROJECT_ROOT, 'outputs', 'charts')
OUTPUT_DATA = os.path.join(PROJECT_ROOT, 'outputs', 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(OUTPUT_CHARTS, exist_ok=True)
os.makedirs(OUTPUT_DATA, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Add src to path for module imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 2 M&V PIPELINE — Baseline Model & Savings Analysis")
print("=" * 70)

print("\n[1/7] Loading data...")

# Load on-site energy + weather
energy_path = os.path.join(DATA_DIR, 'phase_2_dataset.csv')
weather_path = os.path.join(DATA_DIR, 'sriracha_weather_enriched.csv')

df_energy = pd.read_csv(energy_path, parse_dates=['datetime'])
df_energy = df_energy.rename(columns={
    'datetime': 'date',
    'Total Consumption (kWh)': 'energy_kwh',
    'On-site Dry-Bulb Temperature (°C)': 'temp_dry_bulb_onsite',
    'On-site Relative Humidity (%)': 'rh_onsite',
    'On-site Wet-Bulb Temperature (°C)': 'temp_wet_bulb_onsite',
})
df_energy['date'] = pd.to_datetime(df_energy['date']).dt.normalize()

df_weather = pd.read_csv(weather_path, parse_dates=['date'])
df_weather['date'] = pd.to_datetime(df_weather['date']).dt.normalize()

# Merge
df = df_energy.merge(df_weather, on='date', how='left')
df = df.sort_values('date').reset_index(drop=True)

print(f"   Total rows: {len(df)}")
print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"   Energy mean: {df['energy_kwh'].mean():,.0f} kWh/day")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

print("\n[2/7] Engineering features...")

# Temporal features
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear
df['year'] = df['date'].dt.year

# Seasonal cycles (circular encoding)
df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

# Cooling Degree Days (base 24°C — Thai thermostat setpoint)
if 'temperature_2m_mean' in df.columns:
    df['cdd_24c'] = np.maximum(0, df['temperature_2m_mean'] - 24)
else:
    df['cdd_24c'] = np.maximum(0, df['temp_dry_bulb_onsite'] - 24)

# Define period labels
df['period'] = 'gap'
df.loc[df['date'] < '2024-03-01', 'period'] = 'baseline'
df.loc[df['date'] >= '2025-01-01', 'period'] = 'reporting'

baseline = df[df['period'] == 'baseline'].copy()
reporting = df[df['period'] == 'reporting'].copy()

print(f"   Baseline period: {len(baseline)} days ({baseline['date'].min().date()} → {baseline['date'].max().date()})")
print(f"   Reporting period: {len(reporting)} days ({reporting['date'].min().date()} → {reporting['date'].max().date()})")
print(f"   Gap (excluded): {len(df[df['period'] == 'gap'])} days")

# Feature sets
RIDGE_FEATURES = ['temp_wet_bulb_onsite', 'cdd_24c', 'is_weekend', 'sin_doy', 'cos_doy']
TARGET = 'energy_kwh'

print(f"   Ridge features (5): {RIDGE_FEATURES}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: EDA VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════

print("\n[3/7] Generating EDA visualizations...")

# 3.1 Time series overview
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle('Daily Energy Consumption — Baseline vs Reporting Period', fontsize=14, fontweight='bold')

colors = {'baseline': '#3498db', 'gap': '#bdc3c7', 'reporting': '#e67e22'}
for period_name, group in df.groupby('period'):
    axes[0].plot(group['date'], group['energy_kwh'], '.', 
                 color=colors.get(period_name, 'gray'), alpha=0.6, markersize=3, label=period_name.title())
axes[0].axvline(pd.Timestamp('2024-03-01'), color='red', linestyle='--', alpha=0.5, label='Gap start')
axes[0].axvline(pd.Timestamp('2025-01-01'), color='green', linestyle='--', alpha=0.5, label='AI deployment')
axes[0].set_ylabel('Energy (kWh/day)')
axes[0].legend(loc='upper right', fontsize=8)
axes[0].set_title('Energy Consumption')

axes[1].plot(df['date'], df['temp_wet_bulb_onsite'], color='#27ae60', alpha=0.5, linewidth=0.8, label='On-site Wet-Bulb')
if 'dew_point_2m_mean' in df.columns:
    axes[1].plot(df['date'], df['dew_point_2m_mean'], color='#8e44ad', alpha=0.4, linewidth=0.8, label='Si Racha Dew Point')
axes[1].set_ylabel('Temperature (°C)')
axes[1].set_xlabel('Date')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].set_title('Weather Context')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_CHARTS, '01_eda_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 01_eda_timeseries.png")

# 3.2 Weather-Energy scatter
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Weather–Energy Relationships (Baseline Period)', fontsize=13, fontweight='bold')

for ax, feat, name in zip(axes, 
    ['temp_wet_bulb_onsite', 'temp_dry_bulb_onsite', 'rh_onsite'],
    ['Wet-Bulb Temp (°C)', 'Dry-Bulb Temp (°C)', 'Relative Humidity (%)']):
    ax.scatter(baseline[feat], baseline[TARGET], alpha=0.5, s=20, color='#2980b9')
    z = np.polyfit(baseline[feat].dropna(), baseline.loc[baseline[feat].notna(), TARGET], 1)
    p = np.poly1d(z)
    x_range = np.linspace(baseline[feat].min(), baseline[feat].max(), 100)
    ax.plot(x_range, p(x_range), 'r--', alpha=0.7, linewidth=2)
    r = baseline[feat].corr(baseline[TARGET])
    ax.set_xlabel(name)
    ax.set_ylabel('Energy (kWh)')
    ax.set_title(f'r = {r:.3f}')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_CHARTS, '02_eda_weather_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 02_eda_weather_scatter.png")

# 3.3 Weekday boxplot
fig, ax = plt.subplots(figsize=(10, 5))
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
baseline_box = baseline.copy()
baseline_box['day_name'] = baseline_box['day_of_week'].map(dict(enumerate(day_names)))
sns.boxplot(data=baseline_box, x='day_name', y=TARGET, order=day_names, ax=ax, palette='RdYlBu_r')
ax.set_title('Energy Consumption by Day of Week (Baseline)', fontweight='bold')
ax.set_xlabel('Day')
ax.set_ylabel('Energy (kWh)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_CHARTS, '03_eda_weekday_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 03_eda_weekday_boxplot.png")

# 3.4 Correlation heatmap
corr_features = ['energy_kwh', 'temp_wet_bulb_onsite', 'temp_dry_bulb_onsite', 
                  'rh_onsite', 'cdd_24c', 'sin_doy', 'cos_doy']
corr_cols = [c for c in corr_features if c in baseline.columns]
fig, ax = plt.subplots(figsize=(8, 7))
corr = baseline[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, square=True)
ax.set_title('Feature Correlation Matrix (Baseline)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_CHARTS, '04_eda_correlation_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 04_eda_correlation_matrix.png")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════

print("\n[4/7] Training baseline models...")

# Prepare data
X_base = baseline[RIDGE_FEATURES].values
y_base = baseline[TARGET].values

# Train/holdout split (80/20 chronological)
n_train = int(len(baseline) * 0.8)
X_train, X_hold = X_base[:n_train], X_base[n_train:]
y_train, y_hold = y_base[:n_train], y_base[n_train:]

# ── Ridge (Primary) ──────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_hold_sc = scaler.transform(X_hold)

ridge = Ridge(alpha=100)
ridge.fit(X_train_sc, y_train)

# Save model artifacts
joblib.dump(ridge, os.path.join(MODELS_DIR, 'ridge_v5.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'ridge_scaler_v5.pkl'))

coef_dict = {name: float(c) for name, c in zip(RIDGE_FEATURES, ridge.coef_)}
coef_dict['intercept'] = float(ridge.intercept_)
with open(os.path.join(MODELS_DIR, 'ridge_v5_coefficients.json'), 'w') as f:
    json.dump(coef_dict, f, indent=2)

print(f"   ✓ Ridge (α=100) trained on {n_train} days")
print(f"     Coefficients: {coef_dict}")

# ── GAM (Backup) ────────────────────────────────────────────────────────
try:
    from pygam import LinearGAM, s, l

    gam = LinearGAM(
        s(0, n_splines=8) +   # wet-bulb (smooth)
        s(1, n_splines=6) +   # cdd_24c (smooth)
        l(2) +                 # is_weekend (linear)
        l(3) +                 # sin_doy (linear)
        l(4)                   # cos_doy (linear)
    )
    gam.fit(X_train, y_train)
    joblib.dump(gam, os.path.join(MODELS_DIR, 'gam_v5.pkl'))
    print(f"   ✓ GAM trained")
    HAS_GAM = True
except ImportError:
    print("   ⚠ pygam not installed — skipping GAM")
    HAS_GAM = False

# ── XGBoost (Robustness) ────────────────────────────────────────────────
try:
    import xgboost as xgb

    XGB_FEATURES = RIDGE_FEATURES + ['day_of_week']
    if 'shortwave_radiation_sum' in baseline.columns:
        XGB_FEATURES.append('shortwave_radiation_sum')
    
    X_train_xgb = baseline[XGB_FEATURES].values[:n_train]
    X_hold_xgb = baseline[XGB_FEATURES].values[n_train:]
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    xgb_model.fit(X_train_xgb, y_train)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_v5.pkl'))
    print(f"   ✓ XGBoost trained ({len(XGB_FEATURES)} features)")
    HAS_XGB = True
except ImportError:
    print("   ⚠ xgboost not installed — skipping XGBoost")
    HAS_XGB = False

# ── LightGBM (Benchmark) ────────────────────────────────────────────────
try:
    import lightgbm as lgb

    LGBM_FEATURES = ['temp_dry_bulb_onsite', 'rh_onsite', 'temp_wet_bulb_onsite',
                      'is_weekend', 'day_of_week', 'sin_doy', 'cos_doy']
    # Add optional features if available
    for feat in ['is_public_holiday_th', 'shortwave_radiation_sum', 
                 'diurnal_temperature_range', 'cdd_24c', 'month']:
        if feat in baseline.columns:
            LGBM_FEATURES.append(feat)
    LGBM_FEATURES = [f for f in LGBM_FEATURES if f in baseline.columns]
    
    X_train_lgb = baseline[LGBM_FEATURES].values[:n_train]
    X_hold_lgb = baseline[LGBM_FEATURES].values[n_train:]
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
    )
    lgb_model.fit(X_train_lgb, y_train)
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, 'lgbm_v5.pkl'))
    print(f"   ✓ LightGBM trained ({len(LGBM_FEATURES)} features)")
    HAS_LGBM = True
except ImportError:
    print("   ⚠ lightgbm not installed — skipping LightGBM")
    HAS_LGBM = False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: HOLDOUT VALIDATION & ASHRAE COMPLIANCE
# ═════════════════════════════════════════════════════════════════════════════

print("\n[5/7] Holdout validation & ASHRAE compliance check...")

def compute_mv_metrics(y_true, y_pred, model_name="Model"):
    """Compute IPMVP/ASHRAE calibration metrics."""
    n = len(y_true)
    residuals = y_true - y_pred
    mean_actual = np.mean(y_true)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    cv_rmse = (rmse / mean_actual) * 100  # %
    nmbe = (np.sum(residuals) / (n * mean_actual)) * 100  # %
    
    # ASHRAE Guideline 14 daily thresholds
    ashrae_pass = (abs(cv_rmse) <= 25.0) and (abs(nmbe) <= 10.0)
    
    return {
        'model': model_name,
        'R2': round(r2, 4),
        'RMSE': round(rmse, 1),
        'CV_RMSE': round(cv_rmse, 2),
        'NMBE': round(nmbe, 2),
        'ASHRAE_pass': ashrae_pass,
    }

# Ridge holdout
y_pred_ridge_hold = ridge.predict(X_hold_sc)
results = [compute_mv_metrics(y_hold, y_pred_ridge_hold, "Ridge")]

# GAM holdout
if HAS_GAM:
    y_pred_gam_hold = gam.predict(X_hold)
    results.append(compute_mv_metrics(y_hold, y_pred_gam_hold, "GAM"))

# XGBoost holdout
if HAS_XGB:
    y_pred_xgb_hold = xgb_model.predict(X_hold_xgb)
    results.append(compute_mv_metrics(y_hold, y_pred_xgb_hold, "XGBoost"))

# LightGBM holdout
if HAS_LGBM:
    y_pred_lgb_hold = lgb_model.predict(X_hold_lgb)
    results.append(compute_mv_metrics(y_hold, y_pred_lgb_hold, "LightGBM"))

results_df = pd.DataFrame(results)
print("\n   HOLDOUT VALIDATION RESULTS:")
print("   " + "─" * 62)
print(results_df.to_string(index=False))
print("   " + "─" * 62)
ashrae_status = "✅ ALL PASS" if all(r['ASHRAE_pass'] for r in results) else "⚠ SOME FAIL"
print(f"   ASHRAE Guideline 14 (daily): {ashrae_status}")

# Actual vs Predicted chart
fig, ax = plt.subplots(figsize=(10, 6))
hold_dates = baseline['date'].values[n_train:]
ax.plot(hold_dates, y_hold, 'k-', linewidth=1.5, label='Actual', alpha=0.8)
ax.plot(hold_dates, y_pred_ridge_hold, '--', color='#e74c3c', linewidth=1.5, label='Ridge Predicted')
if HAS_GAM:
    ax.plot(hold_dates, y_pred_gam_hold, '--', color='#3498db', linewidth=1.2, label='GAM Predicted', alpha=0.7)
ax.set_title('Holdout Validation: Actual vs Predicted Baseline', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Energy (kWh)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_CHARTS, '05_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 05_actual_vs_predicted.png")

# Residual diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Ridge Residual Diagnostics (Holdout)', fontsize=13, fontweight='bold')

residuals_hold = y_hold - y_pred_ridge_hold
axes[0, 0].scatter(y_pred_ridge_hold, residuals_hold, alpha=0.5, s=20, color='#2980b9')
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Residual')
axes[0, 0].set_title('Residuals vs Predicted')

axes[0, 1].hist(residuals_hold, bins=20, color='#3498db', edgecolor='white', alpha=0.8)
axes[0, 1].axvline(0, color='red', linestyle='--')
axes[0, 1].set_xlabel('Residual (kWh)')
axes[0, 1].set_title('Residual Distribution')

axes[1, 0].plot(hold_dates, residuals_hold, '-o', markersize=3, color='#2980b9', alpha=0.6)
axes[1, 0].axhline(0, color='red', linestyle='--')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Residual')
axes[1, 0].set_title('Residuals Over Time')

from scipy import stats
stats.probplot(residuals_hold, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_CHARTS, '06_residual_diagnostics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 06_residual_diagnostics.png")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: PLACEBO TEST
# ═════════════════════════════════════════════════════════════════════════════

print("\n[6/7] Running placebo test...")

# Train on Jun-Nov 2023, predict Dec 2023 - Feb 2024
placebo_train = baseline[(baseline['date'] >= '2023-06-01') & (baseline['date'] < '2023-12-01')]
placebo_test = baseline[(baseline['date'] >= '2023-12-01') & (baseline['date'] <= '2024-02-28')]

X_plac_train = placebo_train[RIDGE_FEATURES].values
y_plac_train = placebo_train[TARGET].values
X_plac_test = placebo_test[RIDGE_FEATURES].values
y_plac_test = placebo_test[TARGET].values

scaler_plac = StandardScaler()
X_plac_train_sc = scaler_plac.fit_transform(X_plac_train)
X_plac_test_sc = scaler_plac.transform(X_plac_test)

ridge_plac = Ridge(alpha=100)
ridge_plac.fit(X_plac_train_sc, y_plac_train)
y_pred_plac = ridge_plac.predict(X_plac_test_sc)

pseudo_savings = np.sum(y_pred_plac - y_plac_test)
pseudo_savings_pct = (pseudo_savings / np.sum(y_pred_plac)) * 100

print(f"   Placebo period: {placebo_test['date'].min().date()} → {placebo_test['date'].max().date()}")
print(f"   Expected pseudo-savings: ~0%")
print(f"   Ridge pseudo-savings: {pseudo_savings_pct:+.2f}%")
print(f"   {'⚠ Seasonal bias detected' if abs(pseudo_savings_pct) > 3 else '✅ Within tolerance'}")

# Placebo chart
fig, ax = plt.subplots(figsize=(10, 5))
models_plac = {'Ridge': pseudo_savings_pct}

# Run placebo for other models if available
if HAS_GAM:
    gam_plac = LinearGAM(s(0, n_splines=8) + s(1, n_splines=6) + l(2) + l(3) + l(4))
    gam_plac.fit(X_plac_train, y_plac_train)
    y_gam_plac = gam_plac.predict(X_plac_test)
    gam_plac_pct = (np.sum(y_gam_plac - y_plac_test) / np.sum(y_gam_plac)) * 100
    models_plac['GAM'] = gam_plac_pct

if HAS_XGB:
    X_plac_train_xgb = placebo_train[XGB_FEATURES].values
    X_plac_test_xgb = placebo_test[XGB_FEATURES].values
    xgb_plac = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_plac.fit(X_plac_train_xgb, y_plac_train)
    y_xgb_plac = xgb_plac.predict(X_plac_test_xgb)
    xgb_plac_pct = (np.sum(y_xgb_plac - y_plac_test) / np.sum(y_xgb_plac)) * 100
    models_plac['XGBoost'] = xgb_plac_pct

if HAS_LGBM:
    X_plac_train_lgb = placebo_train[LGBM_FEATURES].values
    X_plac_test_lgb = placebo_test[LGBM_FEATURES].values
    lgb_plac = lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    lgb_plac.fit(X_plac_train_lgb, y_plac_train)
    y_lgb_plac = lgb_plac.predict(X_plac_test_lgb)
    lgb_plac_pct = (np.sum(y_lgb_plac - y_plac_test) / np.sum(y_lgb_plac)) * 100
    models_plac['LightGBM'] = lgb_plac_pct

colors_plac = ['#e74c3c' if abs(v) > 3 else '#27ae60' for v in models_plac.values()]
bars = ax.bar(models_plac.keys(), models_plac.values(), color=colors_plac, edgecolor='white', linewidth=1.5)
ax.axhline(0, color='black', linewidth=1)
ax.axhspan(-3, 3, alpha=0.1, color='green', label='±3% tolerance')
ax.set_ylabel('Pseudo-Savings (%)')
ax.set_title('Placebo Test: Expected ≈ 0% (No AI in Test Period)', fontweight='bold')
for bar, val in zip(bars, models_plac.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:+.1f}%', 
            ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_CHARTS, '07_placebo_test.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 07_placebo_test.png")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: SAVINGS ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

print("\n[7/7] Estimating savings on reporting period...")

# Re-train Ridge on FULL baseline (not just train split)
scaler_full = StandardScaler()
X_base_full_sc = scaler_full.fit_transform(X_base)
ridge_full = Ridge(alpha=100)
ridge_full.fit(X_base_full_sc, y_base)

# Predict counterfactual baseline for reporting period
X_rep = reporting[RIDGE_FEATURES].values
X_rep_sc = scaler_full.transform(X_rep)
y_pred_ridge_rep = ridge_full.predict(X_rep_sc)
y_actual_rep = reporting[TARGET].values

# Calculate savings
total_predicted = np.sum(y_pred_ridge_rep)
total_actual = np.sum(y_actual_rep)
total_savings = total_predicted - total_actual
savings_pct = (total_savings / total_predicted) * 100
cost_savings = total_savings * 4.0  # THB

savings_all = {'Ridge': {'predicted': total_predicted, 'actual': total_actual,
                          'savings_kwh': total_savings, 'savings_pct': savings_pct}}

print(f"\n   ┌──────────────────────────────────────────────┐")
print(f"   │  RIDGE PRIMARY SAVINGS ESTIMATE               │")
print(f"   ├──────────────────────────────────────────────┤")
print(f"   │  Predicted baseline:  {total_predicted:>12,.0f} kWh       │")
print(f"   │  Actual consumption:  {total_actual:>12,.0f} kWh       │")
print(f"   │  Total savings:       {total_savings:>12,.0f} kWh       │")
print(f"   │  Savings percentage:  {savings_pct:>11.2f}%           │")
print(f"   │  Cost savings (4 THB/kWh): {cost_savings:>10,.0f} THB   │")
print(f"   └──────────────────────────────────────────────┘")

# Other models
if HAS_GAM:
    gam_full = LinearGAM(s(0, n_splines=8) + s(1, n_splines=6) + l(2) + l(3) + l(4))
    gam_full.fit(X_base, y_base)
    y_gam_rep = gam_full.predict(X_rep)
    gam_savings = np.sum(y_gam_rep) - total_actual
    gam_pct = (gam_savings / np.sum(y_gam_rep)) * 100
    savings_all['GAM'] = {'savings_kwh': gam_savings, 'savings_pct': gam_pct}
    print(f"   GAM savings: {gam_savings:,.0f} kWh ({gam_pct:.2f}%)")

if HAS_XGB:
    X_rep_xgb = reporting[XGB_FEATURES].values
    xgb_full = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_full.fit(baseline[XGB_FEATURES].values, y_base)
    y_xgb_rep = xgb_full.predict(X_rep_xgb)
    xgb_savings = np.sum(y_xgb_rep) - total_actual
    xgb_pct = (xgb_savings / np.sum(y_xgb_rep)) * 100
    savings_all['XGBoost'] = {'savings_kwh': xgb_savings, 'savings_pct': xgb_pct}
    print(f"   XGBoost savings: {xgb_savings:,.0f} kWh ({xgb_pct:.2f}%)")

if HAS_LGBM:
    X_rep_lgb = reporting[LGBM_FEATURES].values
    lgb_full = lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    lgb_full.fit(baseline[LGBM_FEATURES].values, y_base)
    y_lgb_rep = lgb_full.predict(X_rep_lgb)
    lgb_savings = np.sum(y_lgb_rep) - total_actual
    lgb_pct = (lgb_savings / np.sum(y_lgb_rep)) * 100
    savings_all['LightGBM'] = {'savings_kwh': lgb_savings, 'savings_pct': lgb_pct}
    print(f"   LightGBM savings: {lgb_savings:,.0f} kWh ({lgb_pct:.2f}%)")

# Save daily savings CSV
daily_savings = pd.DataFrame({
    'date': reporting['date'].values,
    'actual_kwh': y_actual_rep,
    'ridge_predicted_kwh': y_pred_ridge_rep,
    'ridge_savings_kwh': y_pred_ridge_rep - y_actual_rep,
})
daily_savings.to_csv(os.path.join(OUTPUT_DATA, 'daily_savings_ridge_v5.csv'), index=False)
print(f"\n   ✓ daily_savings_ridge_v5.csv saved ({len(daily_savings)} rows)")

# Monthly savings chart
daily_savings['month'] = pd.to_datetime(daily_savings['date']).dt.to_period('M')
monthly = daily_savings.groupby('month').agg(
    baseline=('ridge_predicted_kwh', 'sum'),
    actual=('actual_kwh', 'sum')
).reset_index()
monthly['savings_pct'] = ((monthly['baseline'] - monthly['actual']) / monthly['baseline']) * 100

fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(monthly))
width = 0.35
ax.bar([i - width/2 for i in x], monthly['baseline'] / 1000, width, label='Adjusted Baseline', color='#3498db', alpha=0.8)
ax.bar([i + width/2 for i in x], monthly['actual'] / 1000, width, label='Actual', color='#e74c3c', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([str(m) for m in monthly['month']], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Energy (MWh)')
ax.set_title('Monthly Adjusted Baseline vs Actual (Ridge Primary)', fontweight='bold')
ax.legend()

# Add savings % labels
for i, pct in enumerate(monthly['savings_pct']):
    color = '#27ae60' if pct > 0 else '#e74c3c'
    ax.text(i, max(monthly['baseline'].iloc[i], monthly['actual'].iloc[i]) / 1000 + 5,
            f'{pct:+.1f}%', ha='center', fontsize=7, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_CHARTS, '08_monthly_savings.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 08_monthly_savings.png")

# Cross-model comparison chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Cross-Model Savings Comparison', fontsize=14, fontweight='bold')

model_names = list(savings_all.keys())
model_pcts = [savings_all[m]['savings_pct'] for m in model_names]
model_kwh = [savings_all[m]['savings_kwh'] for m in model_names]
colors_bar = ['#e74c3c' if m == 'Ridge' else '#3498db' for m in model_names]

axes[0].barh(model_names, model_pcts, color=colors_bar, edgecolor='white')
axes[0].set_xlabel('Savings (%)')
axes[0].set_title('Savings Percentage')
for i, v in enumerate(model_pcts):
    axes[0].text(v + 0.05, i, f'{v:.2f}%', va='center', fontweight='bold')

axes[1].barh(model_names, [k/1000 for k in model_kwh], color=colors_bar, edgecolor='white')
axes[1].set_xlabel('Savings (MWh)')
axes[1].set_title('Total Savings (kWh)')
for i, v in enumerate(model_kwh):
    axes[1].text(v/1000 + 2, i, f'{v:,.0f}', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_CHARTS, '09_cross_model_savings.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 09_cross_model_savings.png")

# Cumulative savings chart
fig, ax = plt.subplots(figsize=(12, 6))
cum_ridge = np.cumsum(y_pred_ridge_rep - y_actual_rep) / 1000
ax.plot(reporting['date'].values, cum_ridge, linewidth=2, color='#e74c3c', label='Ridge (Primary)')

if HAS_GAM:
    cum_gam = np.cumsum(y_gam_rep - y_actual_rep) / 1000
    ax.plot(reporting['date'].values, cum_gam, linewidth=1.5, color='#3498db', label='GAM', alpha=0.8)
if HAS_XGB:
    cum_xgb = np.cumsum(y_xgb_rep - y_actual_rep) / 1000
    ax.plot(reporting['date'].values, cum_xgb, linewidth=1.5, color='#27ae60', label='XGBoost', alpha=0.8)
if HAS_LGBM:
    cum_lgb = np.cumsum(y_lgb_rep - y_actual_rep) / 1000
    ax.plot(reporting['date'].values, cum_lgb, linewidth=1.5, color='#8e44ad', label='LightGBM', alpha=0.8)

ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Savings (MWh)')
ax.set_title('Cumulative Energy Savings — All Regression Models', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_CHARTS, '10_cumulative_savings.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 10_cumulative_savings.png")

# Save metadata
metadata = {
    'version': 'v5',
    'timestamp': datetime.now().isoformat(),
    'baseline_period': f"{baseline['date'].min().date()} to {baseline['date'].max().date()}",
    'reporting_period': f"{reporting['date'].min().date()} to {reporting['date'].max().date()}",
    'ridge_features': RIDGE_FEATURES,
    'ridge_alpha': 100,
    'ashrae_compliance': {r['model']: r['ASHRAE_pass'] for r in results},
    'holdout_metrics': results,
    'savings_summary': {m: {'savings_kwh': round(d['savings_kwh'], 0), 
                             'savings_pct': round(d['savings_pct'], 2)}
                        for m, d in savings_all.items()},
}
with open(os.path.join(MODELS_DIR, 'training_metadata_v5.json'), 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"\n  Outputs:")
print(f"    Charts: {OUTPUT_CHARTS}/")
print(f"    Data:   {OUTPUT_DATA}/daily_savings_ridge_v5.csv")
print(f"    Models: {MODELS_DIR}/")
print(f"\n  Primary Savings Estimate: {savings_pct:.2f}%")
print(f"  Cross-model range: {min(model_pcts):.2f}% – {max(model_pcts):.2f}%")
print(f"  Models: {', '.join(model_names)}")
