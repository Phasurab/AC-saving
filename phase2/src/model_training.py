"""
Model training module for Phase 2 M&V analysis (v4).
Supports Ridge (primary), GAM (challenger), XGBoost (challenger).
"""
import numpy as np
import pandas as pd
import json
import os
import joblib
from typing import Tuple, Dict, List, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from pygam import LinearGAM, s, l, f
from datetime import datetime


# =============================================================================
# Time-Series Cross-Validation
# =============================================================================

def expanding_window_cv(
    df: pd.DataFrame,
    date_col: str = 'date',
    min_train_months: int = 2,
    val_month_size: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window time-series CV splits."""
    df = df.copy()
    df['_ym'] = df[date_col].dt.to_period('M')
    unique_months = sorted(df['_ym'].unique())
    
    folds = []
    for i in range(min_train_months, len(unique_months)):
        train_months = unique_months[:i]
        val_months = [unique_months[i]]
        
        train_mask = df['_ym'].isin(train_months)
        val_mask = df['_ym'].isin(val_months)
        
        train_idx = df[train_mask].index.values
        val_idx = df[val_mask].index.values
        
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
    
    return folds


def holdout_split(df, holdout_frac=0.2, date_col='date'):
    """Time-ordered 80/20 split."""
    df_s = df.sort_values(date_col).reset_index(drop=True)
    split = int(len(df_s) * (1 - holdout_frac))
    return df_s.iloc[:split].copy(), df_s.iloc[split:].copy()


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute ASHRAE-informed M&V metrics."""
    residuals = y_true - y_pred
    y_mean = np.mean(y_true)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mbe = np.mean(residuals)
    
    return {
        'R2': round(float(r2_score(y_true, y_pred)), 4),
        'RMSE': round(float(rmse), 2),
        'MAE': round(float(mean_absolute_error(y_true, y_pred)), 2),
        'MBE': round(float(mbe), 2),
        'CV_RMSE': round(float((rmse / y_mean) * 100), 2),
        'NMBE': round(float((mbe / y_mean) * 100), 2),
        'n_samples': len(y_true),
        'y_mean': round(float(y_mean), 2),
    }


def check_ashrae_compliance(metrics: Dict) -> Dict[str, bool]:
    """Check daily-adapted ASHRAE Guideline 14 thresholds."""
    return {
        'CV_RMSE_pass': abs(metrics['CV_RMSE']) <= 25.0,
        'NMBE_pass': abs(metrics['NMBE']) <= 10.0,
        'R2_above_050': metrics['R2'] >= 0.50,
        'R2_above_075': metrics['R2'] >= 0.75,
        'overall_pass': abs(metrics['CV_RMSE']) <= 25.0 and abs(metrics['NMBE']) <= 10.0,
    }


# =============================================================================
# MODEL A: Ridge Regression
# =============================================================================

def train_ridge(X_train, y_train, alpha=1.0):
    """Train Ridge with standardization."""
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)
    model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
    model.fit(X_sc, y_train)
    return model, scaler


def predict_ridge(model, scaler, X):
    return model.predict(scaler.transform(X))


def tune_ridge_alpha(df, feature_cols, target_col, alphas=None, date_col='date'):
    """Tune alpha via expanding-window CV."""
    if alphas is None:
        alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    
    folds = expanding_window_cv(df, date_col=date_col)
    results = []
    
    for alpha in alphas:
        fold_r2, fold_cvrmse, fold_nmbe = [], [], []
        for train_idx, val_idx in folds:
            X_tr = df.iloc[train_idx][feature_cols]
            y_tr = df.iloc[train_idx][target_col]
            X_val = df.iloc[val_idx][feature_cols]
            y_val = df.iloc[val_idx][target_col]
            
            m, sc = train_ridge(X_tr, y_tr, alpha=alpha)
            y_pred = predict_ridge(m, sc, X_val)
            met = compute_metrics(y_val.values, y_pred)
            fold_r2.append(met['R2'])
            fold_cvrmse.append(met['CV_RMSE'])
            fold_nmbe.append(met['NMBE'])
        
        results.append({
            'alpha': alpha,
            'mean_R2': np.mean(fold_r2), 'std_R2': np.std(fold_r2),
            'mean_CV_RMSE': np.mean(fold_cvrmse), 'std_CV_RMSE': np.std(fold_cvrmse),
            'mean_NMBE': np.mean(fold_nmbe), 'n_folds': len(folds),
        })
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['mean_CV_RMSE'].abs().idxmin()
    best_alpha = results_df.loc[best_idx, 'alpha']
    return best_alpha, results_df


def ridge_cv_eval(df, feature_cols, target_col, alpha, date_col='date'):
    """Run expanding-window CV at given alpha, return per-fold and aggregate."""
    folds = expanding_window_cv(df, date_col=date_col)
    fold_metrics = []
    for i, (tr_idx, val_idx) in enumerate(folds):
        m, sc = train_ridge(df.iloc[tr_idx][feature_cols], df.iloc[tr_idx][target_col], alpha)
        y_pred = predict_ridge(m, sc, df.iloc[val_idx][feature_cols])
        met = compute_metrics(df.iloc[val_idx][target_col].values, y_pred)
        met['fold'] = i + 1
        fold_metrics.append(met)
    
    agg = {
        'mean_R2': np.mean([m['R2'] for m in fold_metrics]),
        'std_R2': np.std([m['R2'] for m in fold_metrics]),
        'mean_RMSE': np.mean([m['RMSE'] for m in fold_metrics]),
        'mean_CV_RMSE': np.mean([m['CV_RMSE'] for m in fold_metrics]),
        'mean_NMBE': np.mean([m['NMBE'] for m in fold_metrics]),
        'n_folds': len(folds),
    }
    return fold_metrics, agg


# =============================================================================
# MODEL B: GAM
# =============================================================================

def build_gam_terms(feature_cols, smooth_features, n_splines=10):
    """Build pygam term specification."""
    terms = None
    for i, col in enumerate(feature_cols):
        if col in smooth_features:
            term = s(i, n_splines=n_splines)
        else:
            term = l(i)
        terms = term if terms is None else terms + term
    return terms


def train_gam(X_train, y_train, feature_cols, smooth_features, n_splines=10, lam=0.6):
    """Train a GAM with smooth + linear terms."""
    terms = build_gam_terms(feature_cols, smooth_features, n_splines)
    gam = LinearGAM(terms)
    gam.gridsearch(X_train.values, y_train.values, lam=np.logspace(-3, 3, 11))
    return gam


def predict_gam(gam, X):
    return gam.predict(X.values)


def gam_cv_eval(df, feature_cols, target_col, smooth_features, n_splines=10, date_col='date'):
    """Run expanding-window CV for GAM."""
    folds = expanding_window_cv(df, date_col=date_col)
    fold_metrics = []
    
    for i, (tr_idx, val_idx) in enumerate(folds):
        X_tr = df.iloc[tr_idx][feature_cols]
        y_tr = df.iloc[tr_idx][target_col]
        X_val = df.iloc[val_idx][feature_cols]
        y_val = df.iloc[val_idx][target_col]
        
        try:
            gam = train_gam(X_tr, y_tr, feature_cols, smooth_features, n_splines)
            y_pred = predict_gam(gam, X_val)
            met = compute_metrics(y_val.values, y_pred)
        except Exception as e:
            met = {'R2': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'MBE': np.nan,
                   'CV_RMSE': np.nan, 'NMBE': np.nan, 'n_samples': len(val_idx),
                   'y_mean': np.nan, 'error': str(e)}
        
        met['fold'] = i + 1
        fold_metrics.append(met)
    
    valid_folds = [m for m in fold_metrics if not np.isnan(m.get('R2', np.nan))]
    agg = {
        'mean_R2': np.mean([m['R2'] for m in valid_folds]) if valid_folds else np.nan,
        'std_R2': np.std([m['R2'] for m in valid_folds]) if valid_folds else np.nan,
        'mean_RMSE': np.mean([m['RMSE'] for m in valid_folds]) if valid_folds else np.nan,
        'mean_CV_RMSE': np.mean([m['CV_RMSE'] for m in valid_folds]) if valid_folds else np.nan,
        'mean_NMBE': np.mean([m['NMBE'] for m in valid_folds]) if valid_folds else np.nan,
        'n_folds': len(valid_folds),
    }
    return fold_metrics, agg


# =============================================================================
# MODEL C: XGBoost
# =============================================================================

DEFAULT_XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbosity': 0,
}


def train_xgb(X_train, y_train, X_val=None, y_val=None, params=None, early_stopping=30):
    """Train XGBoost regressor."""
    if params is None:
        params = DEFAULT_XGB_PARAMS.copy()
    
    model = xgb.XGBRegressor(**params)
    
    fit_kw = {}
    if X_val is not None and y_val is not None:
        fit_kw['eval_set'] = [(X_val, y_val)]
        fit_kw['verbose'] = False
    
    model.fit(X_train, y_train, **fit_kw)
    return model


def xgb_cv_eval(df, feature_cols, target_col, params=None, date_col='date', early_stopping=30):
    """Run expanding-window CV for XGBoost."""
    folds = expanding_window_cv(df, date_col=date_col)
    fold_metrics = []
    
    for i, (tr_idx, val_idx) in enumerate(folds):
        X_tr = df.iloc[tr_idx][feature_cols]
        y_tr = df.iloc[tr_idx][target_col]
        X_val = df.iloc[val_idx][feature_cols]
        y_val = df.iloc[val_idx][target_col]
        
        model = train_xgb(X_tr, y_tr, X_val, y_val, params, early_stopping)
        y_pred = model.predict(X_val)
        met = compute_metrics(y_val.values, y_pred)
        met['fold'] = i + 1
        met['best_iteration'] = model.best_iteration if hasattr(model, 'best_iteration') else None
        fold_metrics.append(met)
    
    agg = {
        'mean_R2': np.mean([m['R2'] for m in fold_metrics]),
        'std_R2': np.std([m['R2'] for m in fold_metrics]),
        'mean_RMSE': np.mean([m['RMSE'] for m in fold_metrics]),
        'mean_CV_RMSE': np.mean([m['CV_RMSE'] for m in fold_metrics]),
        'mean_NMBE': np.mean([m['NMBE'] for m in fold_metrics]),
        'n_folds': len(folds),
    }
    return fold_metrics, agg


# =============================================================================
# Model Saving
# =============================================================================

def save_ridge_model(model, scaler, feature_names, metrics, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, 'ridge_baseline.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'ridge_scaler.pkl'))
    
    coefs = {
        'intercept': float(model.intercept_),
        'coefficients': {n: float(c) for n, c in zip(feature_names, model.coef_)},
        'alpha': float(model.alpha),
        'validation_metrics': metrics,
    }
    with open(os.path.join(output_dir, 'ridge_coefficients.json'), 'w') as f:
        json.dump(coefs, f, indent=2)
    
    print(f"✅ Ridge model saved to {output_dir}/")
    for n, c in coefs['coefficients'].items():
        print(f"   {n}: {c:+.4f}")
    print(f"   intercept: {coefs['intercept']:.4f}")


def save_gam_model(gam, feature_names, metrics, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(gam, os.path.join(output_dir, 'gam_model.pkl'))
    
    summary = {'features': feature_names, 'validation_metrics': metrics}
    with open(os.path.join(output_dir, 'gam_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"✅ GAM model saved to {output_dir}/")


def save_xgb_model(model, feature_names, metrics, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    model.save_model(os.path.join(output_dir, 'xgb_model.json'))
    joblib.dump(model, os.path.join(output_dir, 'xgb_model.pkl'))
    
    print(f"✅ XGBoost model saved to {output_dir}/")


def save_training_metadata(all_info: Dict, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    all_info['training_date'] = datetime.now().isoformat()
    with open(os.path.join(output_dir, 'training_metadata_v4.json'), 'w') as f:
        json.dump(all_info, f, indent=2, default=str)
    print(f"✅ Training metadata saved to {output_dir}/training_metadata_v4.json")


def load_ridge_model(model_dir='models'):
    model = joblib.load(os.path.join(model_dir, 'ridge_baseline.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'ridge_scaler.pkl'))
    return model, scaler


def load_gam_model(model_dir='models'):
    return joblib.load(os.path.join(model_dir, 'gam_model.pkl'))


def load_xgb_model(model_dir='models'):
    return joblib.load(os.path.join(model_dir, 'xgb_model.pkl'))
