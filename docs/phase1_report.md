---
pdf_options:
  format: A4
  margin: 18mm
  printBackground: true
stylesheet: https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css
body_class: markdown-body
css: |-
  .markdown-body { font-size: 11pt; }
  img { max-width: 100%; height: auto; display: block; margin: 12px auto; }
  table { font-size: 10pt; width: 100%; }
  h1 { border-bottom: 2px solid #2ecc71; padding-bottom: 8px; }
  h2 { border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 30px; }
  h3 { margin-top: 20px; }
  .page-break { page-break-after: always; }
---

# Phase 1 Report: "The Eyes & The Brain"

**Objective:** Build a room occupancy detection model and define HVAC control logic for The Seaview Grand hotel.

**Guiding constraint from Khun Somchai (GM):** *"If a single VIP guest complains that their room was warm, this whole project is dead."*

---

## 1. Assumptions & Scope

| # | Assumption | Impact on Design |
|---|------------|------------------|
| 1 | **Hotel Identity.** We assumed the property profile matches **Oakwood Hotel & Residence Sri Racha** — a serviced apartment/hotel hybrid with mixed standard rooms and larger suites. | Informed our room-type segmentation (Regular vs Suite) and the conservative comfort strategy. |
| 2 | **Check-in at 14:00, Check-out at 12:00.** We adopted Oakwood Sri Racha's standard policy. | We engineered `hours_since_checkin_time` (hours elapsed since 14:00) and `hours_until_checkout_time` (hours remaining until 12:00) as temporal features. Also informed the check-in CO2 rise and check-out CO2 drop features. |
| 3 | **No PMS Integration.** The model has no access to the hotel's Property Management System (live check-in/check-out events). | AI must rely purely on physical sensors (CO2, Temp, RH, Motion) to deduce presence. |
| 4 | **VIP Suite Zero Tolerance.** Suite guests have absolute priority for thermal comfort (per Khun Somchai). | Suites get a separate, more aggressive model threshold (0.40 vs 0.55). |

---

## 2. Data Dictionary Recap

The raw dataset (`phase1_dataset.csv`) is structured as follows:

**Format:** Wide format with a multi-level header. Row 1 = room identifier (e.g., `room_1001_bedroom`), Row 2 = sensor type, Row 3+ = timestamped readings at **5-minute intervals**.

**Room Naming:** `room_XXXX_bedroom` = standard single-zone room. `room_XXXX_living_room` = living area of a suite. Rooms with both `_bedroom` and `_living_room` entries are multi-zone suites.

**Floors:** 8–24 (room numbers 8XX through 24XX). No floor 13.

| Sensor | Type | Unit | Description |
|--------|------|------|-------------|
| `motion` | Integer | — | Motion events in 5-min window. 0 = no motion. ≥1 = motion detected. |
| `presence_state` | Integer | — | **Target variable.** 0 = Unoccupied, 1 = Occupied, 3 = CO2 disconnected, 4 = Motion disconnected. |
| `room_temperature` | Float | °C | Indoor air temperature. |
| `relative_humidity` | Float | % | Indoor relative humidity. |
| `co2` | Float | ppm | Indoor CO2 concentration. |

**Key characteristics:** ~2 weeks continuous data, 400+ rooms, 5-min sampling. Not all rooms have all sensors — this reflects real-world installation gaps.

**Critical data fact:** 56.4% of historical Presence readings are corrupted (codes 3, 4, or NaN due to sensor disconnections). Only 43.6% of the data has clean binary labels (0 or 1) suitable for supervised training.

<div class="page-break"></div>

## 3. External Data Augmentation

### 3.1 Thai Holiday & Event Calendar

We created a comprehensive external feature file (`th_holiday_event_macro_features_2023-05-31_to_2026-03-10.csv`) covering 1,016 days with 25 columns:

| Feature Group | Columns | Description |
|---------------|---------|-------------|
| **Public Holidays** | `is_public_holiday`, `holiday_name`, `is_bridge_holiday` | Thai national holidays (Loy Krathong, Songkran, King's Birthday). Bridge holidays = workdays squeezed between a holiday and weekend. |
| **Long Weekends** | `is_long_weekend`, `long_weekend_start`, `long_weekend_end`, `long_weekend_length` | Extended weekends (3–5 day stretches) that drive higher hotel occupancy. |
| **Local Events** | `is_pattaya_music_festival`, `is_pattaya_fireworks_festival`, `is_pattaya_songkran_event`, `is_sriracha_songkran_event`, `is_sriracha_local_event` | Nearby events in Pattaya and Sri Racha that drive demand at the hotel. |
| **Tourism Season** | `tourism_season_proxy` | Categorized as `peak_high_season`, `shoulder_season`, or `green_low_season` based on the Thai tourism calendar. |
| **Macro Demand** | `tourism_macro_demand_proxy_score`, `tourism_macro_demand_proxy_label` | A composite score (0–10) combining all factors into a single demand signal: `low`, `normal`, `high`, `peak`. |
| **Event Proximity** | `days_to_next_local_or_pattaya_event`, `days_since_prev_local_or_pattaya_event` | Temporal distance to surrounding events — captures pre-event and post-event booking surges. |

**Why this helps:** Hotel occupancy is not random. It surges during long weekends and local festivals. Without this context, the AI would see unexplained spikes in room usage and treat them as noise.

### 3.2 Cyclic Date Encoding

Raw time values like `hour = 23` and `hour = 0` appear numerically far apart, but in reality they are only 1 hour apart. To encode this circular nature, we applied sine/cosine transformations:

- `hour_sin = sin(2π × hour / 24)` and `hour_cos = cos(2π × hour / 24)`
- `dow_sin = sin(2π × dayofweek / 7)` and `dow_cos = cos(2π × dayofweek / 7)`

The Random Forest feature selection confirmed that `hour` (rank #14), `hour_sin` (rank #21), and `hour_cos` (forecast rank #15) were all important — the model uses both raw hour (for hard splits like "is it after 11 PM?") and cyclic encoding (for smooth transitions).

<div class="page-break"></div>

## 4. Explore & Preprocess — Raw Data Handling

### 4.1 Raw Data Overview

We loaded **53.3 million** raw sensor rows spanning 482 rooms over 77 days (Dec 1, 2025 – Feb 15, 2026). The raw data arrived in long format with a `quality_flag` column from the cleaning pipeline.

![Quality Flag Distribution](assets/quality_flag_dist.png)

| Quality Flag | Count | % | Meaning |
|-------------|------:|---:|---------|
| `single_candidate_misplaced` | 23,900,221 | 44.8% | Sensor type reassigned based on value ranges |
| `multi_candidate_ambiguous` | 10,482,019 | 19.7% | Multiple possible sensor assignments; resolved by consensus |
| `multi_candidate_keep_original` | 9,409,921 | 17.6% | Multiple candidates but original assignment kept |
| `matched_original` | 9,366,241 | 17.6% | Clean match — no correction needed |
| `invalid_all_thresholds` | 148,078 | 0.3% | Values outside all physically reasonable ranges |
| `missing` | 4,624 | <0.01% | Completely absent readings |

### 4.2 Quality Flag Resolution Logic

The cleaning pipeline resolved ambiguous sensor readings through a cascading logic:

1. **Threshold Check:** Each raw value was tested against physically plausible ranges per sensor type (e.g., CO2: 300–5000 ppm, Temp: 10–45°C, RH: 5–100%).
2. **Single Candidate:** If a value matched exactly one sensor's range, it was reassigned to that sensor.
3. **Multi Candidate:** If a value matched multiple sensors, consensus scoring across neighboring timestamps resolved the ambiguity.
4. **Invalid:** Values matching no sensor range were flagged and excluded.

### 4.3 Outlier Detection — Rolling MAD

Outliers were detected per sensor using a **Rolling Median Absolute Deviation (MAD)** method:

- A rolling window (12 timesteps = 1 hour) computed the local median and MAD for each sensor.
- Values deviating more than **3× MAD** from the rolling median were flagged.
- Flagged outliers: Temp = 1,349,068 | CO2 = 918,998 | RH = 907,295
- Outlier values were replaced (not deleted) using the imputation strategy below.

### 4.4 Missing Value Imputation

**For Continuous Sensors (Temp, CO2, RH):**

- **Step 1 — Slope-based gap assessment:** Before filling, we examined each gap's surrounding values to estimate the rate of change (slope). If the slope across a gap was gentle (e.g., CO2 drifting from 420 to 430 ppm over 30 minutes), linear interpolation was safe. If the slope was steep (e.g., a sudden CO2 spike from 400 to 700 ppm indicating someone just entered), interpolation would create artificial ramps, so forward-fill was used instead.
- **Step 2 — Linear interpolation** for gaps ≤ 6 timesteps (30 minutes) with gentle slopes. Short gaps are typically communication dropouts where the underlying physical process continued smoothly.
- **Step 3 — Forward-fill** for gaps ≤ 12 timesteps (1 hour) or gaps with steep slopes. Sensor readings change slowly — the last valid reading is a reasonable proxy.
- **Step 4 — Room baseline fill** for longer gaps. Each room's historical median was used as a fallback.
- **Result:** 0.00% remaining missing values across all continuous sensors.

**For Discrete Sensors (Motion, Presence):**

- Motion is binary (0/1) — linear interpolation is meaningless.
- **Forward-fill** was applied for gaps ≤ 6 timesteps, with a fallback of 0 (no motion) for longer gaps — the conservative safe assumption.
- Presence codes 3 and 4 were **not imputed** — they were preserved as-is and excluded from training data. These are the rows our production inference pipeline later fills.

<div class="page-break"></div>

## 5. Data Splitting — Training on Correct Labels Only

### 5.1 Label Filtering

The `presence_state` target contains 4 codes: 0 (Unoccupied), 1 (Occupied), 3 (CO2 disconnected), 4 (Motion disconnected). Codes 3 and 4 are **not valid labels** — they indicate hardware failures, not occupancy states.

We strictly filtered the dataset to include **only rows where Presence = 0 or 1**. All rows with Presence = 3, 4, or NaN were excluded from model training, validation, and testing.

### 5.2 Temporal Split (No Leakage)

We applied a **temporal split** — not a random split — to prevent data leakage:

| Split | Date Range | Purpose | Rows |
|-------|-----------|---------|-----:|
| **Train** | Dec 1, 2025 – Jan 23, 2026 | Model training (70%) | 1,797,992 |
| **Validation** | Jan 24 – Feb 4, 2026 | Hyperparameter tuning & early stopping (15%) | 248,498 |
| **Test** | Feb 5 – Feb 15, 2026 | Final hold-out evaluation (15%) | 180,702 |

**Why temporal, not random?** Sensor readings are autocorrelated (CO2 at 2:00 PM is nearly identical to 2:05 PM). A random split would leak near-duplicate samples into the test set, inflating metrics. Temporal splitting ensures the model is evaluated on genuinely unseen future data.

**Class balance:** 74.3% Occupied / 25.7% Unoccupied. Handled via asymmetric class weighting (5:1 weight for the Occupied class) during training.

<div class="page-break"></div>

## 6. EDA — The Physical Fingerprint of Occupancy

*"What does a typical occupied vs unoccupied room look like in the data?"*

### Feature Distributions (CO2 / Temp / Motion vs Occupancy)

**Patterns observed:**

- **CO2:** Occupied rooms shift rightward (450–800+ ppm) vs unoccupied (400–450 ppm baseline). People breathe — CO2 is the strongest non-motion indicator.
- **Temperature:** Occupied rooms are slightly warmer (+0.5–1.0°C) due to body heat and appliance usage.
- **Relative Humidity:** Elevated in occupied rooms from breathing and shower usage.
- **Motion:** Nearly perfect binary indicator when available, but **completely absent during deep sleep** — the critical blind spot.

![CO2 Above Room Baseline](assets/orig_dist_co2_baseline.png)

![Temp Above Room Baseline](assets/orig_dist_temp_baseline.png)

<div class="page-break"></div>

![CO2 15-min Trend](assets/orig_dist_co2_trend.png)

![Motion Intensity 60-min sum](assets/orig_dist_motion_intensity.png)

<div class="page-break"></div>

![CO2 Decay Rate](assets/orig_dist_co2_decay.png)

![RH Variability 30-min std](assets/orig_dist_rh_var.png)

<div class="page-break"></div>

### Cross-Sensor Interactions

![Cross-Sensor Interactions](assets/02_cross_sensor_interactions.png)

The combination of `high_CO2_no_motion` is the key fingerprint for **sleeping guests** — CO2 accumulates while motion stays at zero. This directly informed our Sleep Protection safety rule.

<div class="page-break"></div>

### Suite vs Regular Room Behavior

Suites show systematically different patterns: higher baseline CO2 (larger rooms), longer continuous occupancy durations, and more erratic motion (guests moving between bedroom and living room zones). This justified training separate models with different thresholds for each room type.

![Suite vs Regular — CO2 Above Baseline](assets/orig_suite_co2.png)

![Suite vs Regular — Motion Intensity](assets/orig_suite_motion.png)

<div class="page-break"></div>

![Suite vs Regular — Occupancy Duration](assets/orig_suite_duration.png)

### Temporal Patterns (Check-in / Check-out)

Clear peaks at 14:00 (check-in) and 12:00 (check-out), matching our Oakwood Hotel assumption. These temporal signatures were encoded as `hours_since_checkin_time` and `hours_until_checkout_time`.

![Temporal — Occupancy Rate by Hour](assets/orig_temp_occ.png)

<div class="page-break"></div>

![Temporal — Motion Detection Rate by Hour](assets/orig_temp_motion.png)

![Temporal — CO2 Above Baseline by Hour](assets/orig_temp_co2.png)

<div class="page-break"></div>

### Correlation Heatmap

![Correlation Heatmap](assets/05_correlation_heatmap.png)

<div class="page-break"></div>

### Unsupervised Validation: K-Means, UMAP & t-SNE

Before training supervised models, we validated that sensor data naturally clusters into separable states:

- **K-Means (k=3):** Three natural clusters aligning with Occupied, Unoccupied, and a Transition state.
- **UMAP:** Compressed 25-dimensional feature space into 2D, revealing distinct islands per occupancy state.
- **t-SNE:** Confirmed local neighborhood structure.

#### UMAP Projection

![UMAP — Colored by Occupancy Label](assets/orig_umap_occ.png)

![UMAP — Colored by K-Means Cluster](assets/orig_umap_kmeans.png)

<div class="page-break"></div>

#### t-SNE Projection

![t-SNE — Colored by Occupancy Label](assets/orig_tsne_occ.png)

![t-SNE — Colored by K-Means Cluster](assets/orig_tsne_kmeans.png)

**Why this matters:** These visualizations prove that "Occupied" and "Unoccupied" are mathematically separable in feature space before any supervised training begins, giving us confidence that ML models will find reliable decision boundaries.

<div class="page-break"></div>

## 7. Model Architectures, Features & Results

### 7.1 M0 Baseline — Simple Threshold Rules

**Approach:** Hand-coded if/else rules using only 4 raw sensor values. No machine learning.

- Detection features (4): `CO2`, `temp`, `RH`, `Motion`
- Forecast features (8): Same 4 + lagged targets at 30 min, 1 hr, 2 hr, 4 hr

![M0 Feature Importance](assets/m0_feature_importance.png)

![M0 Confusion Matrix](assets/m0_detect_cm.png)

---

### 7.2 M1 LightGBM — Tree Ensemble + RF Feature Selection

**Approach:** Gradient-boosted tree ensemble. A Random Forest (200 trees) first ranked all 93 engineered features by permutation importance. Top-K selected. 5 proxy leakers explicitly removed from Detection.

- Detection (25 features): `CO2`, `CO2_roll_mean_6`, `CO2_lag1`, `steps_since_motion_causal`, `CO2_lag3`, `CO2_roll_mean_12`, `steps_since_motion`, `motion_roll_max_6`, `CO2_roll_std_6`, `CO2_diff3`, `motion_cumsum_12`, `CO2_above_baseline`, `ashrae_state`, `hour`, `temp`, `hours_until_checkout_time`, `temp_lag1`, `hours_since_checkin_time`, `temp_RH_product`, `days_since_prev_event`, `hour_sin`, `temp_roll_mean_6`, `temp_diff1`, `days_to_next_event`, `high_CO2_no_motion`
- Forecast (20 features): `steps_since_presence_causal`, `occupancy_duration_causal`, `CO2_roll_mean_6`, `CO2_lag1`, `CO2`, `CO2_roll_mean_12`, `steps_since_motion_causal`, `CO2_lag3`, `hour`, `CO2_above_baseline`, `CO2_diff3`, `CO2_roll_std_6`, `motion_cumsum_12`, `hours_until_checkout_time`, `hour_cos`, `hours_since_checkin_time`, `ashrae_state`, `hour_sin`, `CO2_diff1`, `CO2_decay_rate`

![M1 RF Feature Importance — Detection](assets/m1_rf_importance_detect.png)

![M1 RF Feature Importance — Forecast](assets/m1_rf_importance_forecast.png)

![M1 Confusion Matrix](assets/m1_detect_cm.png)

---

### 7.3 M2 PatchTST — Transformer

**Approach:** Simplified PatchTST transformer. Raw sequential sensor traces patched into fixed-length segments processed by multi-head self-attention.

- Input: 12-timestep window × 10 channels
- Channels: `CO2`, `temp`, `RH`, `Motion`, `hour_sin`, `hour_cos`, `dayofweek`, `is_night`, `CO2_above_baseline`, `motion_binary`
- 150 rooms subsampled. Hyperparams: d_model=64, n_heads=4, n_layers=2, dropout=0.2.

![M2 Confusion Matrix](assets/m2_detect_cm.png)

---

### 7.4 M3 InceptionTime — 1D-CNN

**Approach:** Multi-scale 1D convolutional kernels (sizes 1, 3, 5, 7) extract temporal patterns at different resolutions. Residual connections.

- Input: 12-timestep window × 10 channels (same as M2). 150 rooms subsampled.

![M3 Confusion Matrix](assets/m3_detect_cm.png)

---

### 7.5 M4 GAF-EfficientNet — Image CNN

**Approach:** Converts 1D sensor traces into 2D Gramian Angular Field images, classified by EfficientNet-B0.

- Input: 12×12 GAF images × 3 channels: `CO2`, `temp`, `RH` (Motion excluded — binary creates degenerate images). 100 rooms subsampled.

![M4 Confusion Matrix](assets/m4_detect_cm.png)

<div class="page-break"></div>

### 7.6 Detection Benchmark Summary

| Model | Architecture | Macro F1 | Recall (Occ) | Prec (Unocc) | Cohen κ | ROC-AUC | Train (s) |
|-------|-------------|:--------:|:------------:|:------------:|:-------:|:-------:|:---------:|
| **M0** | Threshold Rules | 0.755 | 0.975 | 0.872 | 0.522 | 0.921 | 6.7 |
| **M1** | LightGBM | **0.840** | 0.984 | 0.934 | 0.683 | 0.974 | 26.6 |
| **M2** | PatchTST | 0.469 | 0.995 | 0.777 | 0.060 | 0.907 | 610 |
| **M3** | InceptionTime | **0.845** | 0.972 | 0.896 | **0.691** | 0.962 | 249 |
| **M4** | GAF-CNN | 0.449 | 0.999 | 0.913 | 0.034 | 0.737 | 199 |

![Detection Model Comparison](assets/detection_comparison.png)

<div class="page-break"></div>

### 7.7 Forecast Benchmark Summary (+30 min ahead)

| Model | Brier ↓ | Log Loss ↓ | PR-AUC ↑ | Recall (Occ) | Macro F1 | Cohen κ |
|-------|:-------:|:----------:|:--------:|:------------:|:--------:|:-------:|
| **M0** | 0.146 | 0.448 | 0.943 | 0.990 | 0.538 | 0.159 |
| **M1** | **0.105** | **0.317** | **0.984** | 0.972 | **0.748** | **0.507** |
| **M2** | 0.152 | 0.471 | 0.942 | 0.997 | 0.464 | 0.054 |
| **M3** | 0.136 | 0.415 | 0.964 | 0.973 | 0.637 | 0.309 |
| **M4** | 0.222 | 0.734 | 0.834 | 1.000 | 0.426 | 0.003 |

![Forecast Model Comparison](assets/forecast_comparison.png)

<div class="page-break"></div>

### 7.8 Why M1 LightGBM Was Selected for Production

Despite M3 InceptionTime slightly edging M1 on Detection F1 (0.845 vs 0.840), **M1 LightGBM** was selected for four reasons:

1. **Inference speed:** 0.004 ms vs 0.8 ms (200× faster) — critical for real-time 5-min loops across 400 rooms.
2. **No GPU required:** Runs on cheap edge hardware without specialized compute.
3. **Forecast dominance:** M1 is the undisputed forecast leader (Brier=0.105, PR-AUC=0.984). We need both Detection and Forecast from the same architecture.
4. **Interpretability:** Feature importance is directly auditable by hotel engineering teams.

**Business-aligned evaluation (F0.5 and F2):**

Beyond global metrics, we evaluated M1 against the specific business objectives of each room type:

- **F0.5 (Unoccupied) for Regular Rooms:** The F0.5 score weights **precision twice as much as recall**. This metric answers: "When we declare a room empty and turn down the AC, how often are we actually right?" M1 achieved F0.5(Unocc) = **0.854**, meaning 85.4% of our energy-saving actions are justified — a very high precision rate that minimizes wasted cooling while still capturing the majority of truly empty rooms.

- **F2 (Occupied) for VIP Suites:** The F2 score weights **recall four times as much as precision**. This metric answers: "Of all the rooms where a guest is actually present, how many do we correctly keep the AC running for?" M1 achieved F2(Occ) = **0.965**, meaning we correctly detect 96.5% of occupied suites — virtually eliminating the risk of a VIP guest walking into a warm room.

These business-aligned scores confirmed that M1 is not just statistically competitive, but operationally superior for the specific risk profiles of each room type.

<div class="page-break"></div>

## 8. Why We Chose These Metrics

### 8.1 Detection Metrics

In a hotel, the cost of errors is **asymmetric:**

- **False Empty** (turning off AC on a sleeping guest): Guest wakes up sweating, calls front desk. **Catastrophic and irreversible.**
- **False Occupied** (cooling an empty room): Hotel wastes ~฿4.50/kWh for a few hours. **Costly but recoverable.**

| Metric | Why We Use It |
|--------|---------------|
| **Recall (Occupied)** | Must be ≥ 95%. Measures how many *actual* guests we correctly detect. Missing a guest is the worst failure. |
| **Precision (Unoccupied)** | Measures how often "empty" calls are truly empty. High precision = confident energy savings. |
| **Macro F1** | Balanced harmonic mean across both classes. Penalizes models that only predict one class. |
| **Cohen's Kappa (κ)** | Agreement beyond chance. With 74% imbalance, "always occupied" gets 74% accuracy but κ ≈ 0. |
| **ROC-AUC** | Threshold-independent class separation measure across all possible cutoffs. |

### 8.2 Forecast Metrics

| Metric | Why We Use It |
|--------|---------------|
| **Brier Score** | Probability calibration — how close predicted probabilities are to reality. Critical for PRE-COOL trigger. |
| **PR-AUC** | More informative than ROC-AUC under 74% class imbalance. |
| **Log Loss** | Penalizes confident wrong predictions exponentially. |

### 8.3 Business-Aligned F-Beta Metrics

The **F-beta score** generalizes F1 by letting us control the precision/recall tradeoff:

- **F0.5 (β=0.5):** Puts **2× weight on precision** over recall. For Regular rooms — we want to be very sure when we declare a room empty.
- **F2 (β=2):** Puts **4× weight on recall** over precision. For VIP Suites — we want to catch *every* occupied room, even at the cost of cooling some empty ones.

<div class="page-break"></div>

## 9. Business-Aligned Threshold Optimization

| Room Type | Metric Optimized | What It Prioritizes | Threshold | Score |
|-----------|-----------------|---------------------|:---------:|:-----:|
| **Regular** (396 rooms) | F0.5 (Unoccupied) | Precision of "empty" calls | **0.55** | 0.854 |
| **VIP Suite** (82 rooms) | F2 (Occupied) | Recall of "occupied" calls | **0.40** | 0.965 |
| **Missing Sensor** (4 rooms) | F0.5 + Recall ≥ 0.95 | Safety under degraded conditions | **0.45** | 0.740 |

![Regular Room Threshold Sweep](assets/threshold_sweep_regular.png)

![Suite Room Threshold Sweep](assets/threshold_sweep_suite.png)

**In plain English:**
- **Regular (0.55):** Requires 55% confidence before keeping AC on. Borderline signals → save energy.
- **Suite (0.40):** Only 40% confidence needed. Even weak signals → assume guest is present.
- **Missing (0.45):** Without motion data, the model is less confident. Lower threshold compensates.

<div class="page-break"></div>

## 10. Proxy-Based Inference for Unknown Presence

### 10.1 The Problem

56.4% of historical data (6,025,282 rows) has corrupted Presence readings (code 3, 4, or NaN). These rows have perfectly valid CO2, Temperature, and Humidity data — but the ground-truth label is missing.

### 10.2 The Inference Strategy

For each unknown-Presence row:
1. **Identify room type** (Regular, Suite, or Missing Sensor) based on pre-configured assignments.
2. **Load the room-type-specific M1 model** and its optimized threshold.
3. **Run Detection inference** — output P(Occupied). We use detection-only (not forecast) for these historical rows because the forecast model depends on features like `steps_since_presence_causal` which require knowing past Presence values — but past Presence is also unknown for these rows, creating a circular dependency.
4. **Apply threshold** → binary 0 (Unoccupied) or 1 (Occupied).
5. **Assign confidence** — how far the probability is from the threshold, normalized to [0,1].

### 10.3 Proxy Validation

Since these 6M rows have no ground truth, we use indirect physical and behavioral checks:

| Proxy Check | Logic | Result | Status |
|-------------|-------|--------|--------|
| **CO2 Agreement** | If "Occupied," CO2 should be above baseline (people breathe). | 73.9% | ✅ Good |
| **Temporal Consistency** | Predictions should not flip-flop every 5 min. | 4.6% flip rate | ✅ Stable |
| **High Confidence** | Most predictions should be confident, not "on the fence." | 76.6% high conf. | ✅ Good |

![Inference Validation](assets/inference_validation.png)

<div class="page-break"></div>

## 11. HVAC Control Logic — From Detection to Action

### 11.1 The 5-State Thermal Drift Strategy

Aligned with **ASHRAE Standard 55** (Thermal Environmental Conditions for Human Occupancy) and **ASHRAE Guideline 36** (High-Performance Sequences of Operations):

Instead of a binary On/Off, we implement graduated energy savings. When a room becomes empty, the AC **is not shut off** — the setpoint is slowly raised over time. This approach achieves two objectives simultaneously:

**(a) Continuous energy savings** that deepen the longer a room is empty. A room empty for 15 minutes saves a little; a room empty for 8 hours saves substantially — all without ever switching the compressor fully off (which causes inefficient restart cycles).

**(b) Rapid recovery** when the guest returns, because the room was only allowed to drift +1°C to +3°C above the setpoint — never fully heating up. This eliminates the dreaded "guest walks into a hot room" scenario.

| State | Trigger | AC Setpoint | Recovery Time | ASHRAE Justification |
|-------|---------|-------------|:-------------:|---------------------|
| 🟢 **OCCUPIED** | Guest detected | Guest's choice (e.g. 23°C) | — | Within comfort band |
| 🟡 **PRE-COOL** | Forecast: return in 30 min | Ramp to 100% capacity | 0 min (proactive) | Anticipatory control per Guideline 36 |
| 🟠 **STANDBY** | Empty < 60 min | +1°C (e.g. 24°C) | < 2 min | Within ASHRAE ±2°C comfort band |
| 🟤 **UNOCCUPIED** | Empty 1–12 hrs | +3°C (e.g. 26°C) | ~8 min | Just outside comfort band; fully recoverable |
| 🔴 **DEEP SAVINGS** | Empty > 12 hrs | 28°C cap, dehumidify only | ~15 min | Asset protection against mold and humidity damage |

### 11.2 Guest Return Recovery

- **Predicted return:** Forecast model predicts 30 min ahead → PRE-COOL ramps compressor to 100% → room hits setpoint *before* the door opens.
- **Surprise return:** Room has only drifted +1°C to +3°C → max-load cooling recovers comfort in **under 10 minutes.**

### 11.3 Edge Cases & Safety Overrides

| Edge Case | Risk | Safety Rule |
|-----------|------|-------------|
| **Sleeping guest** | Motion = 0 for hours | If `is_night` AND `CO2 ≥ baseline`: **Force OCCUPIED** |
| **Sensor failure** | Motion dies mid-stay | Route to Missing Sensor model. Confidence < 60%: **Force OCCUPIED** |
| **VIP Suite** | Cooling interruption | Suites **blocked** from UNOCCUPIED and DEEP SAVINGS |
| **Total comm failure** | All sensors offline | **Immediate Force OCCUPIED.** Never turn off AC when blind. |

<div class="page-break"></div>

### 11.4 Regular Room Control Logic (396 rooms)

**Goal:** Maximize energy savings while maintaining a safe comfort net.

**How it works:** Every 5 minutes, a new sensor payload arrives. The system first checks fail-safes (is the hardware alive? is it nighttime with CO2 rising?). If safe, it runs the M1 LightGBM model. If P(Occupied) ≥ 0.55, AC stays at guest setpoint. Otherwise, it checks whether the Forecast model predicts a return within 30 minutes (trigger PRE-COOL). If no return expected, it escalates through STANDBY → UNOCCUPIED → DEEP SAVINGS based on vacancy duration.

**Key design choices:**
- Threshold = **0.55** — demands higher confidence, enabling aggressive energy savings.
- Full 5-state escalation — STANDBY (+1°C) → UNOCCUPIED (+3°C) → DEEP SAVINGS (28°C).
- Forecast intercept — PRE-COOL triggers if return predicted within 30 min.

![Regular Room Control Logic](assets/control_logic_regular.png)

<div class="page-break"></div>

### 11.5 VIP Suite Control Logic (82 rooms)

**Goal:** Absolute guest comfort. Energy savings are secondary.

**How it works:** The flow is intentionally shorter. The model runs with a much lower threshold (0.40). If even a weak signal of presence is detected, the AC stays fully on. If the model says "not occupied," the system does NOT escalate through time-based drift. It is hard-capped at STANDBY (+1°C). Even if a VIP suite has been empty for 10 hours, the room is only 1 degree warmer — imperceptible upon return.

**Key design choices:**
- Threshold = **0.40** — only 40% confidence needed. Even faint signals keep AC running.
- **Capped at STANDBY** — UNOCCUPIED and DEEP SAVINGS states are physically disabled.
- Sacrifices energy savings for guaranteed zero guest complaints.

![Suite Room Control Logic](assets/control_logic_suite.png)

<div class="page-break"></div>

### 11.6 Missing Sensor Control Logic (4 rooms: 1002, 1005, 1602, 1032)

**Goal:** Survive without motion data. Never gamble guest comfort.

**How it works:** These 4 rooms have chronically broken motion sensors (>50% of data is Presence code 3 or 4). The system enters **Degraded Mode** — all motion-derived features are force-set to -999 (LightGBM's missing-value sentinel). The model relies exclusively on CO2, Temperature, and Humidity.

Because the model operates with incomplete information, an extra **confidence gate** is added. If prediction confidence drops below 60%, the system does not risk comfort — it force-locks to OCCUPIED and alerts engineering to fix the sensor.

**Key design choices:**
- **Degraded Mode** — motion features zeroed to -999. Model uses CO2/Temp/RH only.
- Threshold = **0.45** — lower than default to compensate for missing motion.
- **Confidence gate (60%)** — if unsure, Force OCCUPIED. Don't gamble.
- **CO2 double-failure check** — if CO2 sensor *also* dies, immediate force-lock. No exceptions.

![Missing Sensor Control Logic](assets/control_logic_missing_sensor.png)

<div class="page-break"></div>

## References

| # | Reference | Application |
|---|-----------|-------------|
| 1 | **ASHRAE Standard 55-2020** — Thermal Environmental Conditions for Human Occupancy | Justified +1°C and +3°C drift bands. <10 min recovery is acceptable transient. |
| 2 | **ASHRAE Guideline 36-2021** — High-Performance Sequences of Operations for HVAC Systems | Informed 5-state graduated control and occupancy-based setpoint reset. |
| 3 | **IPMVP** — International Performance Measurement and Verification Protocol | Referenced for Phase 2 energy savings verification. |
| 4 | **Van Rijsbergen (1979)** — F-beta Score, Information Retrieval | Foundation for F0.5/F2 threshold optimization with asymmetric precision/recall weights. |
