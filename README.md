# Shift-Guard — Early Burnout Detection for First Responders & Caregivers

Shift-Guard is a wrist-worn early-stress / burnout detector. It reads four
physiological signals from a wristband, and classifies short windows of data as
**calm** or **stressed** in real time. This repo contains the data analysis,
feature engineering, and a Random Forest baseline model, trained on the public
[WESAD](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/) dataset (which
was recorded on the same Empatica E4 sensors as the hardware prototype).

## Repository layout

```
Shift-Guard/
├── data/
│   └── WESAD/                  # raw dataset (one folder per subject: S2, S3, …)
├── notebooks/
│   ├── EDA.ipynb              # exploratory analysis + feature export (Sections 1–11)
│   └── baseline.ipynb        # Random Forest model + full evaluation suite
├── src/
│   ├── e4_loader.py          # reads the Empatica E4 zip archives (HR, IBI, …)
│   └── extract_features.py   # builds the 30 s windowed feature table
├── outputs/                  # generated artifacts (created by the code)
│   ├── features_30s.csv      # modeling-ready feature table
│   ├── e4_captured.pkl       # all E4 signals captured from the zips (large)
│   ├── e4_capture_summary.csv
│   ├── rf_baseline.joblib    # trained baseline model + metadata
│   ├── rf_baseline.pkl       # same artifact as a plain pickle
│   └── baseline_model.onnx   # ONNX export for cross-platform inference
├── docs/
│   └── SCHEMA.md             # data dictionary: every signal and feature explained
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Install dependencies (a virtualenv is recommended)
pip install -r requirements.txt jupyter

# 2. (Optional) regenerate the feature table from the raw data
python src/extract_features.py        # writes outputs/features_30s.csv

# 3. (Optional) capture the E4 HR/IBI signals from the zip archives
python src/e4_loader.py               # writes outputs/e4_captured.pkl

# 4. Open the notebooks
jupyter notebook notebooks/EDA.ipynb
jupyter notebook notebooks/baseline.ipynb
```

The notebooks anchor their working directory to the project root automatically, so
they can be launched from anywhere.

## Pipeline overview

1. **EDA.ipynb** — validates data quality, aligns labels, derives HR from BVP,
   quantifies how strongly each signal separates calm vs stressed, and exports
   `outputs/features_30s.csv`.
2. **extract_features.py** — slices each subject's session into 30-second windows
   (50% overlap), keeps only pure calm/stressed windows, and computes per-signal
   features for ACC, BVP, HR (derived), TEMP and EDA.
3. **baseline.ipynb** — trains a Random Forest with **Leave-One-Subject-Out**
   cross-validation (the leakage-safe metric for a wearable) and reports accuracy,
   precision/recall/F1, ROC-AUC, confusion matrix, feature importances, and an
   overfitting analysis.

## Baseline result (Leave-One-Subject-Out)

| Metric | Score |
|--------|-------|
| Accuracy | 0.905 |
| F1 (stressed) | 0.866 |
| ROC-AUC | 0.970 |

vs. a majority-class baseline of 0.64. See `docs/SCHEMA.md` for the full data
dictionary and `notebooks/baseline.ipynb` for the complete evaluation.
