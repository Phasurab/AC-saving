# 🏨 Hotel Occupancy Detection & HVAC Control System

> **Phase 1 — "The Eyes & The Brain"**  
> AI-driven room occupancy detection with intelligent HVAC control logic for The Seaview Grand hotel.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-success.svg)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Models Evaluated](#models-evaluated)
- [HVAC Control Logic](#hvac-control-logic)
- [Data](#data)
- [Documentation](#documentation)

---

## 🎯 Overview

This project builds a **room occupancy detection system** using multi-sensor data (CO2, Temperature, Humidity, Motion) from 400+ hotel rooms. The AI predictions drive a graduated **5-state HVAC control strategy** that balances energy savings with guest comfort — per the GM's guiding constraint:

> *"If a single VIP guest complains that their room was warm, this whole project is dead."*

### Key Achievements

- **98.4% Recall** for occupied rooms — virtually no sleeping guests missed
- **0.840 Macro F1** on the production model (M1 LightGBM)
- **3 room-type-specific thresholds** optimized for different risk profiles
- **30-minute ahead forecast** enabling proactive PRE-COOL before guest returns
- **Proxy inference** for 56.4% of data with corrupted presence labels

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SENSOR LAYER                          │
│  CO2 (ppm) │ Temp (°C) │ RH (%) │ Motion (events/5min)  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               DATA PREPARATION                           │
│  Outlier detection (Rolling MAD) → Imputation → Splits   │
│  Feature Engineering (93 features incl. holiday/events)   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│             ROOM-TYPE ROUTING                            │
│  Regular (396) │ VIP Suite (82) │ Missing Sensor (4)     │
│  threshold=0.55│ threshold=0.40 │ threshold=0.45         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│          M1 LightGBM DETECTION + FORECAST                │
│  Detection: 25 RF-selected features, P(Occupied)         │
│  Forecast:  20 features, +30 min ahead probability       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         5-STATE HVAC CONTROL LOGIC                       │
│  🟢 OCCUPIED → 🟠 STANDBY (+1°C) → 🟤 UNOCCUPIED (+3°C)│
│  → 🔴 DEEP SAVINGS (28°C) │ 🟡 PRE-COOL (forecast)     │
│  + Safety overrides: Sleep, VIP, Sensor failure          │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Key Results

### Detection Benchmark (5 Models)

| Model | Architecture | Macro F1 | Recall (Occ) | Cohen κ | ROC-AUC |
|-------|-------------|:--------:|:------------:|:-------:|:-------:|
| **M0** | Threshold Rules | 0.755 | 0.975 | 0.522 | 0.921 |
| **M1** ✅ | LightGBM + RF | **0.840** | **0.984** | 0.683 | **0.974** |
| **M2** | PatchTST | 0.469 | 0.995 | 0.060 | 0.907 |
| **M3** | InceptionTime | **0.845** | 0.972 | **0.691** | 0.962 |
| **M4** | GAF-EfficientNet | 0.449 | 0.999 | 0.034 | 0.737 |

> ✅ **M1 LightGBM selected** for production: best forecast (Brier=0.105), 200× faster inference, no GPU required, fully interpretable.

### Forecast Benchmark (+30 min ahead)

| Model | Brier ↓ | PR-AUC ↑ | Macro F1 | Cohen κ |
|-------|:-------:|:--------:|:--------:|:-------:|
| **M1** ✅ | **0.105** | **0.984** | **0.748** | **0.507** |
| M3 | 0.136 | 0.964 | 0.637 | 0.309 |
| M0 | 0.146 | 0.943 | 0.538 | 0.159 |

### Business-Aligned Thresholds

| Room Type | Metric Optimized | Threshold | Score |
|-----------|------------------|:---------:|:-----:|
| Regular (396 rooms) | F0.5 (Unocc) — precision of "empty" calls | **0.55** | 0.854 |
| VIP Suite (82 rooms) | F2 (Occ) — recall of "occupied" calls | **0.40** | 0.965 |
| Missing Sensor (4 rooms) | F0.5 + Recall ≥ 0.95 | **0.45** | 0.740 |

---

## 📁 Project Structure

```
├── README.md                          # This file
├── .gitignore                         # Excludes large data/models
├── requirements.txt                   # Python dependencies
│
├── docs/
│   ├── phase1_report.md               # Full Phase 1 report (markdown)
│   ├── data_dictionary.md             # Sensor data dictionary
│   ├── control_logic.md               # HVAC control logic documentation
│   └── assets/                        # Key report visualizations
│       ├── detection_comparison.png
│       ├── forecast_comparison.png
│       ├── control_logic_regular.png
│       ├── control_logic_suite.png
│       ├── control_logic_missing_sensor.png
│       └── architecture_overview.png
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb         # Raw data cleaning pipeline
│   └── 02_eda_and_modeling.ipynb      # EDA + feature engineering + models
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py            # Raw data → cleaned splits
│   ├── feature_engineering.py         # 93 engineered features
│   ├── eda_analysis.py                # EDA visualization code
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── m0_baseline/               # Simple threshold rules
│   │   ├── m1_lightgbm/               # LightGBM + RF feature selection ✅
│   │   ├── m2_patchtst/               # PatchTST transformer
│   │   ├── m3_inceptiontime/          # InceptionTime 1D-CNN
│   │   └── m4_gaf_efficientnet/       # GAF image + EfficientNet
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── model_comparison.py        # Cross-model benchmark
│   │   └── segment_evaluation.py      # Per-segment F-beta evaluation
│   │
│   └── production/
│       ├── __init__.py
│       ├── room_gateway.py            # Room-type → model routing
│       ├── train_per_segment.py       # Per-segment production training
│       └── infer_unknown.py           # Proxy inference for corrupted labels
│
├── data/
│   └── external/
│       └── th_holiday_event_macro_features.csv   # Thai holidays + events
│
└── control_logic/
    ├── control_logic_regular.mermaid
    ├── control_logic_suite.mermaid
    └── control_logic_missing_sensor.mermaid
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- ~16 GB RAM (for full dataset processing)

### Installation

```bash
git clone https://github.com/<your-username>/hotel-occupancy-detection.git
cd hotel-occupancy-detection
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# 1. Data preparation (requires raw CSV in data/)
python src/data_preparation.py

# 2. Feature engineering
python src/feature_engineering.py

# 3. Train & evaluate a model (e.g., M1 LightGBM)
python src/models/m1_lightgbm/pipeline.py

# 4. Compare all models
python src/evaluation/model_comparison.py

# 5. Production inference on unknown labels
python src/production/infer_unknown.py
```

> **Note:** Raw data files (`phase1_dataset.csv`, ~5.5 GB) and trained model weights are excluded from this repo via `.gitignore`. Contact the project owner for data access.

---

## 🤖 Models Evaluated

| # | Model | Approach | Strengths | Weaknesses |
|---|-------|----------|-----------|------------|
| M0 | **Baseline** | Hand-coded if/else rules | Fast, interpretable, no training | Low F1, no probability calibration |
| M1 | **LightGBM** | Gradient boosting + RF feature selection | Best forecast, fast inference, interpretable | Requires feature engineering |
| M2 | **PatchTST** | Transformer on sensor patches | Captures long-range temporal dependencies | High model bias, slow training |
| M3 | **InceptionTime** | Multi-scale 1D-CNN | Best detection F1 (0.845), strong κ | GPU required, 200× slower inference |
| M4 | **GAF-EfficientNet** | Gramian angular field images + CNN | Novel approach, image-based | Poor calibration, needs GPU |

---

## 🌡 HVAC Control Logic

A graduated **5-state thermal drift strategy** aligned with ASHRAE Standards 55 & 36:

| State | Trigger | AC Setpoint | Recovery |
|-------|---------|-------------|:--------:|
| 🟢 **OCCUPIED** | Guest detected | Guest's choice | — |
| 🟡 **PRE-COOL** | Forecast: return in 30 min | 100% capacity | 0 min |
| 🟠 **STANDBY** | Empty < 60 min | +1°C | < 2 min |
| 🟤 **UNOCCUPIED** | Empty 1–12 hrs | +3°C | ~8 min |
| 🔴 **DEEP SAVINGS** | Empty > 12 hrs | 28°C cap | ~15 min |

### Safety Overrides ("The Khun Somchai Rules")

| Edge Case | Rule |
|-----------|------|
| 🛏 **Sleeping guest** | Night + CO2 ≥ baseline → **Force OCCUPIED** |
| ⚠ **Sensor failure** | Route to degraded model. Confidence < 60% → **Force OCCUPIED** |
| 🏆 **VIP Suite** | Capped at STANDBY. Never enters UNOCCUPIED or DEEP SAVINGS |
| 🔌 **Total comm failure** | All sensors offline → **Immediate Force OCCUPIED** |

---

## 📂 Data

### Source Dataset
- **53.3M** raw sensor rows from 482 rooms over 77 days
- 5-minute sampling interval
- Sensors: CO2 (ppm), Temperature (°C), Humidity (%), Motion (events)
- 56.4% of presence labels corrupted (sensor disconnections)

### External Features
- Thai holiday & event calendar (1,016 days, 25 columns)
- Tourism season proxies, local Pattaya/Sri Racha events
- Cyclic time encoding (hour/day-of-week sin/cos)

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [Phase 1 Report](docs/phase1_report.md) | Full technical report with all EDA, model results, and control logic |
| [Data Dictionary](docs/data_dictionary.md) | Sensor types, value ranges, and data structure |
| [Control Logic](docs/control_logic.md) | HVAC control strategy with mermaid flowcharts |

---

## 📝 License

This project was developed as part of the AltoTech AI Engineer Assessment. Data is anonymized and derived from real hotel deployments.

---

*Built with ❤️ for energy-efficient hotel operations*
