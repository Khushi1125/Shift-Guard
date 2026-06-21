# Shift-Guard — Data Schema & Dictionary

This document explains every dataset and every column used in the project: the raw
signals, the captured Empatica E4 data, and — most importantly — the
`features_30s.csv` table that the model trains on.

---

## 1. The signals (what the sensors measure)

| Signal | Full name | Sampling rate | Unit | What it physically measures | Stress behaviour |
|--------|-----------|---------------|------|-----------------------------|------------------|
| **ACC** | 3-axis accelerometer | 32 Hz | 1/64 g | Wrist movement (x, y, z) | Restless/fidgety motion can rise |
| **BVP** | Blood Volume Pulse (PPG) | 64 Hz | arbitrary (AC-coupled) | Optical pulse waveform; each peak = 1 heartbeat | Amplitude/rhythm change |
| **EDA** | Electrodermal Activity | 4 Hz | µS (microsiemens) | Skin conductance from sweat-gland activity | **Rises strongly** under stress |
| **TEMP** | Skin temperature | 4 Hz | °C | Surface temperature of the wrist | **Falls** under stress (vasoconstriction) |
| **HR** | Heart Rate (derived) | 1 Hz | BPM | Beats per minute, derived from BVP peaks | **Rises** under stress |

> **EDA vs the "EDA notebook":** confusingly, "EDA" is overloaded. In this project
> it means **Electrodermal Activity** (the sweat/skin-conductance sensor). The file
> `notebooks/EDA.ipynb` instead uses "EDA" in the data-science sense of
> *Exploratory Data Analysis*. They are unrelated.

---

## 2. Label scheme

The WESAD experiment had 5 conditions; we collapse them to a binary target.

| Raw WESAD label | Condition | Our label | Meaning |
|-----------------|-----------|-----------|---------|
| 1 | Baseline (sitting calmly) | **0** | calm |
| 2 | TSST (stressful speech + mental math) | **1** | stressed |
| 0 | transition | dropped | — |
| 3 | amusement | dropped | — |
| 4 | meditation | dropped | — |

---

## 3. Raw WESAD pickle — `data/WESAD/S<id>/S<id>.pkl`

A nested Python dict (pickled under Python 2 → load with `encoding='latin1'`):

```
raw
├── 'signal'
│   ├── 'chest'   → dict of RespiBAN signals (ACC, ECG, EMG, EDA, Temp, Resp) @ 700 Hz   [unused]
│   └── 'wrist'   → dict of Empatica E4 signals:
│         ├── 'ACC'  → ndarray (N, 3)   @ 32 Hz
│         ├── 'BVP'  → ndarray (N, 1)   @ 64 Hz
│         ├── 'EDA'  → ndarray (N, 1)   @  4 Hz
│         └── 'TEMP' → ndarray (N, 1)   @  4 Hz
├── 'label'       → ndarray (M,)        @ 700 Hz   (values 0–4; see label scheme)
└── 'subject'     → e.g. 'S2'
```

Note: there is **no `HR`** in the pickle — it is derived from BVP (see §6).
Labels are at 700 Hz, so they are sub-sampled to each signal's rate before use.

---

## 4. Empatica E4 export — `data/WESAD/S<id>/S<id>_E4_Data.zip`

Each zip holds the E4's own CSV exports. Continuous CSVs have a 2-row header
(row 0 = session start unix timestamp, row 1 = sample rate), then the samples.

| File | Contents |
|------|----------|
| `ACC.csv`, `BVP.csv`, `EDA.csv`, `TEMP.csv` | Same signals as the pickle, standalone |
| `HR.csv` | E4's on-device heart rate (1 Hz) — the one missing from the pickle |
| `IBI.csv` | Inter-beat intervals: `[seconds_since_start, interval_duration_s]` (irregular) |
| `tags.csv` | Event button-press timestamps (often empty) |
| `info.txt` | Human-readable format description |

---

## 5. Captured E4 data — `outputs/e4_captured.pkl`

Produced by `src/e4_loader.py`. A dict `{subject_id: {...}}`; each subject has:

| Key | Type | Fields |
|-----|------|--------|
| `'ACC'`,`'BVP'`,`'EDA'`,`'TEMP'`,`'HR'` | dict | `start_time`, `rate`, `data`, `n_samples` |
| `'IBI'` | dict | `start_time`, `time_s`, `ibi_s`, `instant_bpm`, `n_beats` |
| `'tags'` | ndarray | event timestamps |
| `'info'` | str | raw `info.txt` |

> ⚠️ The E4 timestamps are a **different timeline** from the WESAD pickle (the
> pickle is trimmed and chest-synchronized). Do not index-align them directly.

---

## 6. Modeling table — `outputs/features_30s.csv`  ← the main dataset

One **row per 30-second window** (50% overlap, so windows step every 15 s). Only
windows that are 100% one condition are kept. Built by `src/extract_features.py`.

**Shape:** ~1,780 rows × 21 columns · **classes:** calm 1,143 / stressed 637 (ratio 0.56)

### Identifier & target columns

| Column | Type | Meaning |
|--------|------|---------|
| `subject` | int | Subject ID (2–17). Used as the **group** for Leave-One-Subject-Out CV. Not a feature. |
| `window_start_s` | int | Start time of the window in seconds from session start. Not a feature. |
| `label` | int | **Target.** 0 = calm, 1 = stressed. |

### Feature columns (18 total)

All features summarize one signal over the 30-second window. "slope" = the linear
trend (sign = rising/falling, magnitude = steepness) fit across the window.

**Accelerometer (movement)** — magnitude is orientation-invariant; HF isolates fast motion.

| Column | Meaning | Unit |
|--------|---------|------|
| `acc_mag_mean` | Average movement magnitude √(x²+y²+z²) | 1/64 g |
| `acc_mag_std` | Variability of movement magnitude | 1/64 g |
| `acc_hf_mean` | Mean high-frequency motion (magnitude minus 1 s rolling mean) = fidgeting | 1/64 g |

**BVP (raw pulse waveform)**

| Column | Meaning | Unit |
|--------|---------|------|
| `bvp_mean` | Average pulse-wave value over the window | arbitrary |
| `bvp_std` | Pulse-wave amplitude/variability (tends to rise under stress) | arbitrary |

**HR (derived heart rate)** — strongest *generalizing* signal.

| Column | Meaning | Unit |
|--------|---------|------|
| `hr_mean` | Average heart rate in the window | BPM |
| `hr_std` | HR variability within the window (a crude HRV proxy) | BPM |
| `hr_slope` | HR trend — positive = HR climbing (stress onset) | BPM/sample |
| `hr_min` | Lowest HR in the window | BPM |
| `hr_max` | Highest HR in the window | BPM |

**TEMP (skin temperature)** — slow signal; trend matters more than level.

| Column | Meaning | Unit |
|--------|---------|------|
| `temp_mean` | Average skin temperature | °C |
| `temp_slope` | Temperature trend — negative = cooling (stress) | °C/sample |
| `temp_delta` | Net change = last value − first value of the window | °C |

**EDA (electrodermal activity / skin conductance)** — top discriminator overall.

| Column | Meaning | Unit |
|--------|---------|------|
| `eda_mean` | **Average skin conductance over the window** — the baseline sweat-response level | µS |
| `eda_std` | Variability of skin conductance (phasic activity / spikes) | µS |
| `eda_slope` | EDA trend — positive = conductance rising (stress build-up) | µS/sample |
| `eda_min` | Lowest conductance in the window | µS |
| `eda_max` | Highest conductance in the window (the strongest single feature) | µS |

> **So what is `eda_mean`?** It is the *mean skin-conductance level* (in
> microsiemens) across the 30-second window. Higher sweat-gland activity → higher
> EDA. It is a strong raw stress marker, but note: after **per-subject
> normalization** in the model, the absolute mean carries little cross-person
> signal (everyone's mean centers near 0), so `eda_max` and the HR features end up
> more predictive on unseen subjects. See the feature-importance section in
> `baseline.ipynb`.

---

## 7. Trained model — `outputs/rf_baseline.joblib`

A dict saved with `joblib`:

| Key | Meaning |
|-----|---------|
| `model` | Trained `RandomForestClassifier` (fit on all subjects) |
| `feature_cols` | The 18 feature names, in the order the model expects |
| `normalization` | Note: apply per-subject z-score at inference (per-user calibration) |
| `window_s`, `overlap` | Windowing config used to build features (30 s, 0.5) |
| `loso_metrics` | Leave-One-Subject-Out metrics dict (accuracy, precision, recall, f1, roc_auc) |

**Inference reminder:** windows fed to the model must be normalized the same way —
collect a short calm baseline per user, store that user's mean/std, then z-score
live windows against it before calling `model.predict`.

## 8. ONNX model — `outputs/baseline_model.onnx`

The same Random Forest exported to ONNX for cross-platform / edge inference (C++,
mobile, browser, embedded) via ONNX Runtime.

| Input | Shape | Type | Notes |
|-------|-------|------|-------|
| `float_input` | `[N, 18]` | float32 | 18 features, **per-subject z-scored** (same order as `feature_cols`) |

| Output | Shape | Type | Meaning |
|--------|-------|------|---------|
| `label` | `[N]` | int64 | Predicted class (0 = calm, 1 = stressed) |
| `probabilities` | `[N, 2]` | float32 | `[P(calm), P(stressed)]` — use column 1 as the stress score |

> The export repairs a known skl2onnx 1.20.0 bug (binary RandomForest leaf weights
> were all assigned to class 0, producing negative probabilities). After the fix,
> ONNX matches sklearn to ~2e-7 on probabilities and 100% on labels. See the
> export cell in `notebooks/baseline.ipynb`.
