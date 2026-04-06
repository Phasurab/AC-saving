"""
Data preparation module for Phase 2 M&V analysis (v4).
Uses Si Racha enriched weather data + on-site measurements.
"""
import pandas as pd
import numpy as np
import json
import os
from typing import Tuple, Dict, List, Optional


TARGET = 'energy_kwh'


def load_main_dataset(path: str) -> pd.DataFrame:
    """Load and parse the main Phase 2 energy dataset."""
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.rename(columns={
        'datetime': 'date',
        'Total Consumption (kWh)': 'energy_kwh',
        'On-site Dry-Bulb Temperature (°C)': 'temp_dry_bulb_onsite',
        'On-site Relative Humidity (%)': 'rh_onsite',
        'On-site Wet-Bulb Temperature (°C)': 'temp_wet_bulb_onsite',
    })
    df['date'] = df['date'].dt.normalize()
    return df


def load_sriracha_weather(path: str) -> pd.DataFrame:
    """Load the Si Racha enriched weather dataset."""
    df = pd.read_csv(path, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    
    # Select useful columns (avoid bringing everything in)
    keep_cols = [
        'date',
        # Temperatures
        'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
        'apparent_temperature_mean',
        # Humidity
        'relative_humidity_2m_mean', 'relative_humidity_2m_max', 'relative_humidity_2m_min',
        'dew_point_2m_mean',
        # Precipitation
        'precipitation_sum', 'rain_sum',
        # Cloud & solar
        'cloud_cover_mean',
        'shortwave_radiation_sum',
        # Wind
        'wind_speed_10m_mean',
        # Derived indices
        'humidex_mean', 'heat_index_mean',
        'diurnal_temperature_range',
        # CDD at various bases
        'cdd_18c', 'cdd_20c', 'cdd_22c', 'cdd_24c', 'cdd_26c',
        # Weather condition
        'weather_code', 'weather_condition',
    ]
    available = [c for c in keep_cols if c in df.columns]
    return df[available].copy()


def merge_datasets(main_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge main dataset with Si Racha weather on date."""
    merged = main_df.merge(weather_df, on='date', how='left')
    return merged


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and seasonal derived features."""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['year'] = df['date'].dt.year
    df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    return df


def add_thai_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Add Thai public holiday flag. Uses a hardcoded list for known dates."""
    # Major Thai public holidays (approximate, covers 2023-2026)
    holidays = [
        # 2023
        '2023-07-28', '2023-07-29', '2023-08-01', '2023-08-02',
        '2023-08-12', '2023-08-14',
        '2023-10-13', '2023-10-23',
        '2023-12-05', '2023-12-10', '2023-12-11', '2023-12-31',
        # 2024
        '2024-01-01', '2024-02-24', '2024-02-26',
        '2024-04-06', '2024-04-13', '2024-04-14', '2024-04-15', '2024-04-16',
        '2024-05-01', '2024-05-04', '2024-05-06', '2024-05-22',
        '2024-06-03', '2024-07-20', '2024-07-21', '2024-07-22', '2024-07-29',
        '2024-08-12', '2024-10-13', '2024-10-14', '2024-10-23',
        '2024-12-05', '2024-12-10', '2024-12-31',
        # 2025
        '2025-01-01', '2025-02-12', '2025-02-14',
        '2025-04-06', '2025-04-07', '2025-04-13', '2025-04-14', '2025-04-15',
        '2025-05-01', '2025-05-04', '2025-05-05', '2025-05-11', '2025-05-12',
        '2025-06-03', '2025-07-10', '2025-07-11',
        '2025-07-28', '2025-08-12',
        '2025-10-13', '2025-10-23',
        '2025-12-05', '2025-12-10', '2025-12-31',
        # 2026
        '2026-01-01', '2026-01-02', '2026-03-01', '2026-03-02',
    ]
    holiday_dates = pd.to_datetime(holidays).normalize()
    df = df.copy()
    df['is_public_holiday_th'] = df['date'].isin(holiday_dates).astype(int)
    return df


def split_periods(
    df: pd.DataFrame,
    baseline_start: str = '2023-06-01',
    baseline_end: str = '2024-02-28',
    reporting_start: str = '2025-01-01',
    exclude_outlier_date: Optional[str] = '2023-05-31',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into baseline and reporting periods."""
    baseline = df[(df['date'] >= baseline_start) & (df['date'] <= baseline_end)].copy()
    reporting = df[df['date'] >= reporting_start].copy()
    
    if exclude_outlier_date:
        baseline = baseline[baseline['date'] != pd.Timestamp(exclude_outlier_date)]
    
    return baseline, reporting


# =============================================================================
# Feature sets per model
# =============================================================================

def get_ridge_features() -> List[str]:
    """Ridge baseline: 7 features (on-site + Si Racha external)."""
    return [
        'temp_wet_bulb_onsite',   # On-site: primary HVAC load driver
        'cdd_24c',                # Si Racha: standard cooling degree-days
        'dew_point_2m_mean',      # Si Racha: latent humidity driver
        'precipitation_sum',      # Si Racha: rainy-day cooling proxy
        'is_weekend',             # Calendar
        'sin_doy',                # Seasonal
        'cos_doy',                # Seasonal
    ]


def get_gam_features() -> List[str]:
    """GAM challenger: same 7 features, some with smooth terms."""
    return get_ridge_features()  # Same features, different treatment


def get_gam_smooth_features() -> List[str]:
    """Features that get nonlinear spline terms in the GAM."""
    return ['temp_wet_bulb_onsite', 'cdd_24c', 'dew_point_2m_mean']


def get_gam_linear_features() -> List[str]:
    """Features that stay linear in the GAM."""
    return ['precipitation_sum', 'is_weekend', 'sin_doy', 'cos_doy']


def get_xgb_features() -> List[str]:
    """XGBoost challenger: 9 features (deliberately limited)."""
    return [
        'temp_wet_bulb_onsite',
        'cdd_24c',
        'dew_point_2m_mean',
        'precipitation_sum',
        'shortwave_radiation_sum',
        'is_weekend',
        'day_of_week',
        'sin_doy',
        'cos_doy',
    ]


def prepare_model_inputs(df: pd.DataFrame, feature_list: List[str]):
    """Extract X, y from a dataframe."""
    X = df[feature_list].copy()
    y = df[TARGET].copy()
    return X, y


def check_data_quality(df: pd.DataFrame, period_name: str = "dataset") -> Dict:
    """Run data quality checks."""
    report = {
        'period': period_name,
        'n_rows': len(df),
        'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}",
        'expected_days': (df['date'].max() - df['date'].min()).days + 1,
        'missing_days': (df['date'].max() - df['date'].min()).days + 1 - len(df),
        'null_counts': {k: int(v) for k, v in df.isnull().sum().items() if v > 0},
        'energy_stats': {
            'mean': round(float(df['energy_kwh'].mean()), 1),
            'std': round(float(df['energy_kwh'].std()), 1),
            'min': round(float(df['energy_kwh'].min()), 1),
            'max': round(float(df['energy_kwh'].max()), 1),
        },
    }
    return report


def full_preparation_pipeline(
    main_path: str,
    weather_path: str,
    output_dir: str = 'outputs/data',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full data preparation pipeline (v4)."""
    # Load
    main_df = load_main_dataset(main_path)
    weather_df = load_sriracha_weather(weather_path)
    
    # Merge
    merged = merge_datasets(main_df, weather_df)
    
    # Add features
    merged = add_calendar_features(merged)
    merged = add_thai_holidays(merged)
    
    # Split
    baseline, reporting = split_periods(merged)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    merged.to_csv(os.path.join(output_dir, 'features_merged_v4.csv'), index=False)
    baseline.to_csv(os.path.join(output_dir, 'baseline_clean_v4.csv'), index=False)
    reporting.to_csv(os.path.join(output_dir, 'reporting_clean_v4.csv'), index=False)
    
    # Quality reports
    for period_df, name in [(baseline, 'baseline'), (reporting, 'reporting')]:
        report = check_data_quality(period_df, name)
        with open(os.path.join(output_dir, f'{name}_quality_report_v4.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    # Feature availability check
    all_feats = list(set(get_ridge_features() + get_xgb_features()))
    print(f"✅ Data prepared (v4 — Si Racha weather):")
    print(f"   Full merged:  {len(merged)} rows")
    print(f"   Baseline:     {len(baseline)} rows ({baseline['date'].min().date()} → {baseline['date'].max().date()})")
    print(f"   Reporting:    {len(reporting)} rows ({reporting['date'].min().date()} → {reporting['date'].max().date()})")
    print(f"   Ridge features (7):   {get_ridge_features()}")
    print(f"   GAM features (7):     {get_gam_features()}")
    print(f"   XGBoost features (9): {get_xgb_features()}")
    
    # Check nulls in features
    for f in all_feats:
        nulls_b = baseline[f].isnull().sum() if f in baseline.columns else 'MISSING'
        nulls_r = reporting[f].isnull().sum() if f in reporting.columns else 'MISSING'
        status = '✅' if nulls_b == 0 and nulls_r == 0 else '⚠️'
        print(f"   {status} {f}: baseline nulls={nulls_b}, reporting nulls={nulls_r}")
    
    print(f"   Saved to: {output_dir}/")
    return merged, baseline, reporting
