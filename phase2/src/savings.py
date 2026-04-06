"""
Savings calculation module for Phase 2 M&V analysis.
Handles counterfactual prediction, savings computation, and aggregation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def calculate_daily_savings(
    predicted_baseline: np.ndarray,
    actual: np.ndarray,
    dates: pd.Series,
) -> pd.DataFrame:
    """
    Calculate daily adjusted savings.
    
    Savings = Predicted baseline (counterfactual) - Actual consumption
    Positive savings = energy saved by AI intervention
    """
    df = pd.DataFrame({
        'date': dates.values,
        'predicted_baseline_kwh': predicted_baseline,
        'actual_kwh': actual,
        'savings_kwh': predicted_baseline - actual,
    })
    
    df['savings_pct'] = (df['savings_kwh'] / df['predicted_baseline_kwh']) * 100
    df['cumulative_savings_kwh'] = df['savings_kwh'].cumsum()
    
    return df


def aggregate_monthly_savings(daily_savings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily savings to monthly level."""
    daily_savings = daily_savings.copy()
    daily_savings['year_month'] = pd.to_datetime(daily_savings['date']).dt.to_period('M')
    
    monthly = daily_savings.groupby('year_month').agg(
        n_days=('date', 'count'),
        predicted_baseline_kwh=('predicted_baseline_kwh', 'sum'),
        actual_kwh=('actual_kwh', 'sum'),
        savings_kwh=('savings_kwh', 'sum'),
        mean_daily_savings_kwh=('savings_kwh', 'mean'),
    ).reset_index()
    
    monthly['savings_pct'] = (monthly['savings_kwh'] / monthly['predicted_baseline_kwh']) * 100
    monthly['cumulative_savings_kwh'] = monthly['savings_kwh'].cumsum()
    
    return monthly


def calculate_cost_savings(
    savings_kwh: float,
    tariff_rate: float = 4.0,  # THB/kWh default
) -> float:
    """Convert energy savings to cost savings."""
    return savings_kwh * tariff_rate


def savings_summary(
    daily_savings: pd.DataFrame,
    tariff_rate: float = 4.0,
) -> Dict:
    """Generate a comprehensive savings summary."""
    total_savings = daily_savings['savings_kwh'].sum()
    total_predicted = daily_savings['predicted_baseline_kwh'].sum()
    total_actual = daily_savings['actual_kwh'].sum()
    n_days = len(daily_savings)
    
    summary = {
        'reporting_period': {
            'start': str(daily_savings['date'].min()),
            'end': str(daily_savings['date'].max()),
            'n_days': n_days,
        },
        'energy': {
            'total_predicted_baseline_kwh': round(total_predicted, 2),
            'total_actual_kwh': round(total_actual, 2),
            'total_savings_kwh': round(total_savings, 2),
            'avg_daily_savings_kwh': round(total_savings / n_days, 2),
            'savings_percentage': round((total_savings / total_predicted) * 100, 2),
        },
        'cost': {
            'tariff_rate_thb_per_kwh': tariff_rate,
            'total_cost_savings_thb': round(total_savings * tariff_rate, 2),
            'avg_daily_cost_savings_thb': round((total_savings / n_days) * tariff_rate, 2),
            'monthly_avg_cost_savings_thb': round((total_savings / n_days * 30) * tariff_rate, 2),
        },
        'positive_savings_days': int((daily_savings['savings_kwh'] > 0).sum()),
        'negative_savings_days': int((daily_savings['savings_kwh'] < 0).sum()),
        'positive_savings_ratio': round(
            (daily_savings['savings_kwh'] > 0).mean() * 100, 1
        ),
    }
    
    return summary


def compare_model_savings(
    daily_savings_lgbm: pd.DataFrame,
    daily_savings_ridge: pd.DataFrame,
    tariff_rate: float = 4.0,
) -> pd.DataFrame:
    """Compare savings from LightGBM vs Ridge models."""
    lgbm_summary = savings_summary(daily_savings_lgbm, tariff_rate)
    ridge_summary = savings_summary(daily_savings_ridge, tariff_rate)
    
    comparison = pd.DataFrame({
        'Metric': [
            'Total Savings (kWh)',
            'Savings (%)',
            'Avg Daily Savings (kWh)',
            'Total Cost Savings (THB)',
            'Positive Savings Days (%)',
        ],
        'LightGBM (Model B)': [
            lgbm_summary['energy']['total_savings_kwh'],
            lgbm_summary['energy']['savings_percentage'],
            lgbm_summary['energy']['avg_daily_savings_kwh'],
            lgbm_summary['cost']['total_cost_savings_thb'],
            lgbm_summary['positive_savings_ratio'],
        ],
        'Ridge (Model A)': [
            ridge_summary['energy']['total_savings_kwh'],
            ridge_summary['energy']['savings_percentage'],
            ridge_summary['energy']['avg_daily_savings_kwh'],
            ridge_summary['cost']['total_cost_savings_thb'],
            ridge_summary['positive_savings_ratio'],
        ],
    })
    
    comparison['Difference'] = comparison['LightGBM (Model B)'] - comparison['Ridge (Model A)']
    
    return comparison
