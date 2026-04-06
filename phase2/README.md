# ⚡ AI-Driven HVAC Energy Optimization — M&V Analysis

> **Phase 2 — Measurement & Verification**  
> Weather-adjusted baseline regression and savings estimation under IPMVP Option C.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://python.org)
[![Ridge Regression](https://img.shields.io/badge/Primary_Model-Ridge-success.svg)]()
[![ASHRAE 14](https://img.shields.io/badge/ASHRAE_14-Compliant-brightgreen.svg)]()
[![IPMVP](https://img.shields.io/badge/IPMVP-Option_C-blue.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Models Evaluated](#models-evaluated)
- [Foundation Model Experiments](#foundation-model-experiments)
- [Documentation](#documentation)

---

## 🎯 Overview

This project implements a **credibility-first M&V framework** for quantifying energy savings from an AI-driven HVAC optimization pilot at a commercial facility in Si Racha, Thailand.

**Core approach:** Build weather-adjusted regression baselines that predict *what energy consumption would have been without AI intervention*, then compare with actual metered data to estimate savings.

> **Guiding principle:** *"Optimize for credible, stable savings estimation — not maximum estimated savings."*

### Key Achievements

- **2.73% primary savings** (Ridge) — 246,939 kWh over 15 months
- **4 independent models** converge to 1.56%–2.73% range (spread = 1.17 pp)
- **ASHRAE Guideline 14** compliant: CV(RMSE) ≤ 25%, NMBE ± 10% on all regression models
- **6 foundation model experiments** tested (Chronos-2, TimesFM) — 3 dropped, 1 promoted to appendix
- **~988,000 THB** estimated cost savings (~$27,400 USD)

---

## 📊 Key Results

### Primary Savings Estimate

| Metric | Value |
|---|---|
| **Primary model** | Ridge Regression (5 features, α = 100) |
| **Total energy saved** | **246,939 kWh** |
| **Savings percentage** | **2.73%** |
| **Cross-model mean** | 2.25% |
| **Cross-model range** | 1.56% – 2.73% |
| **Estimated cost savings** | ~988,000 THB (~$27,400 USD) |

### Model Comparison

| Model | Role | Savings % | |NMBE| | CV(RMSE) | ASHRAE |
|---|---|---|---|---|---|
| **Ridge** | **Primary** | **2.73** | **0.67%** | 14.98% | ✅ |
| GAM | Backup | 2.26 | 1.73% | 14.06% | ✅ |
| LightGBM | Benchmark | 2.43 | 3.58% | 16.02% | ✅ |
| XGBoost | Robustness | 1.56 | 1.41% | 15.47% | ✅ |
| Ridge+TimesFM | Appendix | 1.85 | 1.88% | 6.74% | ✅ |

> **Ridge selected as primary** because it has the lowest bias (NMBE = −0.67%), fully interpretable coefficients, and is trivially auditable by third-party energy auditors.

### Cumulative Savings

![Cumulative Savings](docs/assets/05_cumulative_savings.png)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                               │
│  On-site BMS (4 columns) + Si Racha Weather (32 columns)     │
│  693 daily observations (May 2023 – Mar 2026)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               FEATURE ENGINEERING                            │
│  Wet-bulb temp │ CDD₂₄ │ is_weekend │ sin/cos_doy           │
│  Collinearity removal │ 8 configs tested │ NMBE-first        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 MODEL TRAINING                               │
│  Ridge (Primary) │ GAM (Backup) │ XGBoost │ LightGBM        │
│  Expanding-window CV │ ASHRAE Guideline 14 compliance        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              VALIDATION LAYER                                │
│  Holdout test │ Placebo test │ Coefficient sign check         │
│  Cross-model consensus │ Foundation model benchmarking       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            SAVINGS ESTIMATION                                │
│  Counterfactual prediction on reporting period               │
│  Savings = Predicted baseline − Actual metered               │
│  Primary: 2.73% │ Range: 1.56%–2.73% │ Cost: ~988k THB      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
├── README.md                           # This file
├── .gitignore                          # Excludes large data/models
├── requirements.txt                    # Python dependencies
│
├── notebooks/
│   └── 01_baseline_and_savings.py      # ★ End-to-end pipeline script
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py             # Data loading, merging, feature engineering
│   ├── model_training.py               # Ridge, GAM, XGBoost training + CV
│   └── savings.py                      # Counterfactual savings calculation
│
├── docs/
│   ├── Executive_Summary.md            # 1-page executive summary
│   ├── Technical_Appendix.md           # Full technical appendix
│   ├── FM_Drop_Analysis_Report.md      # Foundation model assessment report
│   └── assets/                         # Key report visualizations
│       ├── 01_eda_timeseries.png
│       ├── 02_eda_weather_scatter.png
│       ├── 03_eda_weekday_boxplot.png
│       ├── 04_cross_model_savings.png
│       ├── 05_cumulative_savings.png
│       └── 06_all_models_comparison.png
│
├── data/
│   └── external/                       # Small external feature files
│       └── (weather CSVs via .gitignore)
│
├── models/                             # Trained model artifacts (excluded by .gitignore)
│   ├── ridge_v5.pkl
│   ├── ridge_scaler_v5.pkl
│   ├── ridge_v5_coefficients.json      # ★ Included: human-readable coefficients
│   ├── training_metadata_v5.json       # ★ Included: training metadata
│   └── foundation_experiments_v5.json  # ★ Included: FM experiment results
│
└── outputs/
    ├── charts/                         # Generated visualizations
    └── data/                           # Daily/monthly savings CSVs
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- ~4 GB RAM

### Installation

```bash
git clone https://github.com/<your-username>/hvac-mv-analysis.git
cd hvac-mv-analysis
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Place data files in data/ directory:
#   - phase_2_dataset.csv (on-site energy + weather)
#   - sriracha_weather_enriched.csv (external weather)

# Run the end-to-end pipeline
python notebooks/01_baseline_and_savings.py
```

**Output:**
- `outputs/charts/` — 10 diagnostic & reporting visualizations
- `outputs/data/daily_savings_ridge_v5.csv` — daily savings time series
- `models/` — trained model weights + metadata JSON

> **Note:** Raw data files and model weights are excluded via `.gitignore`. Contact the project owner for data access.

---

## 🤖 Models Evaluated

### Regression Models (IPMVP-Compliant)

| # | Model | Approach | Features | Why Selected / Role |
|---|-------|----------|----------|---------------------|
| 1 | **Ridge** | L2-regularized linear regression (α=100) | 5 | 🥇 Primary — lowest bias (NMBE = −0.67%) |
| 2 | **GAM** | Generalized Additive Model with splines | 5 (smooth) | 🥈 Backup — best fit (R² = 0.355) |
| 3 | **LightGBM** | Gradient-boosted trees | 12 | Nonlinear benchmark |
| 4 | **XGBoost** | Extreme gradient boosting | 7 | Best placebo test (−0.28%) |

### Ridge Coefficients (Primary Model)

| Feature | Coefficient | Physical Interpretation |
|---|---|---|
| `temp_wet_bulb_onsite` | +1,495 kWh/σ | ↑ Heat + humidity → ↑ AC work |
| `cdd_24c` | +357 kWh/σ | ↑ Cooling demand → ↑ HVAC load |
| `is_weekend` | −95 kWh | Slightly lower weekend occupancy |
| `sin_doy` | −183 kWh | Annual cycle (peaks Jun–Aug) |
| `cos_doy` | −900 kWh | Cool-season ramp-down (Dec–Feb) |

All coefficients have physically defensible signs. ✅

---

## 🔬 Foundation Model Experiments

Six experiments tested whether foundation time-series models (Chronos-2, TimesFM) could replace or improve regression baselines:

| Exp | Model | Savings | Decision | Reason |
|---|---|---|---|---|
| — | Chronos-T5 naive | 6.72% | ❌ Drop | No weather adjustment; inflated ×3 |
| — | TimesFM naive | 9.62% | ❌ Drop | No weather adjustment; inflated ×4 |
| A1 | Chronos-2 + covariates | −45.77% | ❌ Drop | 10-month gap broke attention mechanism |
| A2 | Chronos-2 naive (120M) | −141.49% | ❌ Drop | Catastrophic divergence over gap |
| B1 | Ridge + Chronos-2 resid | −143.48% | ❌ Drop | Chronos destroyed Ridge baseline |
| B2 | Ridge + TimesFM resid | **1.85%** | 📎 Appendix | ✅ Validates regression range |

**Conclusion:** Foundation models are not a replacement for weather-adjusted regression in M&V with discontinuous baselines. TimesFM works only as a residual correction tool when disciplined by regression.

See [FM Drop Analysis Report](docs/FM_Drop_Analysis_Report.md) for full details.

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [Executive Summary](docs/Executive_Summary.md) | 1-page stakeholder-ready summary |
| [Technical Appendix](docs/Technical_Appendix.md) | Full technical details, feature selection, residual analysis |
| [FM Analysis Report](docs/FM_Drop_Analysis_Report.md) | 25-page foundation model assessment with EDA, all experiments, and conclusions |

---

## 📚 References

| # | Reference |
|---|-----------|
| 1 | IPMVP (2022) — Option C Whole-Facility Regression |
| 2 | ASHRAE Guideline 14-2014 — Daily CV(RMSE) ≤ 25%, NMBE ± 10% |
| 3 | Das et al. (2025) — Chronos-2: Learning the Language of Time Series |
| 4 | Das et al. (2024) — TimesFM: A Decoder-Only Foundation Model |

---

*Built with ❤️ for credible energy savings verification*
