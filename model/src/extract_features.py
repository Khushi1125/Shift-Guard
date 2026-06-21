"""Feature extraction for the Shift-Guard burnout model.

Turns the raw WESAD wrist signals into a modeling-ready feature table — one row
per fixed-length window — following the recommendations from EDA.ipynb:

    - Window length 30 s, 50% overlap (15 s step)
    - Keep only PURE-condition windows (no calm/stressed transitions mid-window)
    - Binary label: Baseline(1) -> 0 (calm), TSST(2) -> 1 (stressed)
    - Per-signal features for ACC, BVP, HR (derived), TEMP and EDA

The EDA signal (electrodermal activity / skin conductance) is included because it
is one of the strongest physiological stress markers and is present in every
pickle's wrist dict, even though the EDA notebook's earlier sections did not load it.

Output:
    outputs/features_30s.csv — columns: subject, window_start_s, label, <features...>

NOTE ON NORMALIZATION: features are exported RAW (not per-subject normalized).
The `subject` column lets the modeling notebook apply per-subject z-scoring and a
grouped / leave-one-subject-out split, which is the leakage-safe approach.

Usage:
    python src/extract_features.py              # writes outputs/features_30s.csv
    from extract_features import build_feature_table
    df = build_feature_table(window_s=30, step_s=15)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

# Resolve paths relative to the project root (the folder that contains src/),
# so this works whether run from the repo root, from src/, or imported elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WESAD_ROOT = str(PROJECT_ROOT / "data" / "WESAD")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_FEATURES_CSV = str(OUTPUTS_DIR / "features_30s.csv")

SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

LABEL_RATE = 700  # Hz — raw WESAD label timeline
RATES = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4, "HR": 1}

# WESAD label 1 (Baseline) -> calm(0); label 2 (TSST) -> stressed(1). Others dropped.
KEEP_LABELS = {1: 0, 2: 1}


def derive_hr_from_bvp(bvp_array, bvp_rate: int = 64, smoothing_window_s: int = 10):
    """Derive a 1 Hz heart-rate (BPM) signal from a raw BVP waveform via peaks."""
    bvp_1d = np.asarray(bvp_array, dtype=float).flatten()
    duration_s = len(bvp_1d) // bvp_rate

    peaks, _ = scipy_signal.find_peaks(
        bvp_1d,
        distance=int(bvp_rate * 0.4),
        prominence=0.1 * np.std(bvp_1d),
    )
    if len(peaks) < 2:
        return np.full(duration_s, np.nan)

    peak_times_s = peaks / bvp_rate
    bpm_instant = 60.0 / np.diff(peak_times_s)
    mids = (peak_times_s[:-1] + peak_times_s[1:]) / 2
    half = smoothing_window_s / 2

    hr = np.full(duration_s, np.nan)
    for t in range(duration_s):
        m = (mids >= t - half) & (mids <= t + half)
        if m.any():
            vb = bpm_instant[m]
            vb = vb[(vb >= 30) & (vb <= 220)]
            if len(vb):
                hr[t] = vb.mean()
    return pd.Series(hr).ffill().bfill().values


def _slope(y: np.ndarray) -> float:
    """Linear-trend slope of a 1-D array (per sample). 0 if too short/flat."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def _load_subject(sid: int) -> dict:
    """Load one subject's raw wrist signals + label array from the pickle."""
    path = os.path.join(WESAD_ROOT, f"S{sid}", f"S{sid}.pkl")
    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    wrist = raw["signal"]["wrist"]
    return {
        "labels": raw["label"].flatten(),               # 700 Hz
        "ACC": wrist["ACC"].astype(float),              # (N, 3) @ 32 Hz
        "BVP": wrist["BVP"].astype(float).flatten(),    # 64 Hz
        "EDA": wrist["EDA"].astype(float).flatten(),    # 4 Hz
        "TEMP": wrist["TEMP"].astype(float).flatten(),  # 4 Hz
    }


def _window_features(sig: dict, start_s: int, end_s: int) -> dict:
    """Extract all per-signal features for the window [start_s, end_s) seconds."""
    feats: dict = {}

    # ── ACC: orientation-invariant magnitude + high-frequency (fidget) intensity
    a0, a1 = start_s * RATES["ACC"], end_s * RATES["ACC"]
    mag = sig["_acc_mag"][a0:a1]
    hf = sig["_acc_hf"][a0:a1]
    feats["acc_mag_mean"] = float(np.mean(mag))
    feats["acc_mag_std"] = float(np.std(mag))
    feats["acc_hf_mean"] = float(np.mean(hf))

    # ── BVP: raw pulse-wave amplitude statistics
    b0, b1 = start_s * RATES["BVP"], end_s * RATES["BVP"]
    bvp = sig["BVP"][b0:b1]
    feats["bvp_mean"] = float(np.mean(bvp))
    feats["bvp_std"] = float(np.std(bvp))

    # ── HR (derived): level, variability (HRV proxy) and within-window trend
    h0, h1 = start_s * RATES["HR"], end_s * RATES["HR"]
    hr = sig["HR"][h0:h1]
    feats["hr_mean"] = float(np.nanmean(hr))
    feats["hr_std"] = float(np.nanstd(hr))
    feats["hr_slope"] = _slope(hr)
    feats["hr_min"] = float(np.nanmin(hr))
    feats["hr_max"] = float(np.nanmax(hr))

    # ── TEMP: level, drift (slope) and net change across the window
    e0, e1 = start_s * RATES["TEMP"], end_s * RATES["TEMP"]
    temp = sig["TEMP"][e0:e1]
    feats["temp_mean"] = float(np.mean(temp))
    feats["temp_slope"] = _slope(temp)
    feats["temp_delta"] = float(temp[-1] - temp[0]) if len(temp) else 0.0

    # ── EDA: skin-conductance level, variability and rising trend
    eda = sig["EDA"][e0:e1]
    feats["eda_mean"] = float(np.mean(eda))
    feats["eda_std"] = float(np.std(eda))
    feats["eda_slope"] = _slope(eda)
    feats["eda_min"] = float(np.min(eda))
    feats["eda_max"] = float(np.max(eda))

    return feats


def build_feature_table(window_s: int = 30, step_s: int = 15,
                        save_path: str | None = DEFAULT_FEATURES_CSV) -> pd.DataFrame:
    """Build the windowed feature table across all subjects and (optionally) save it."""
    rows = []

    for sid in SUBJECTS:
        path = os.path.join(WESAD_ROOT, f"S{sid}", f"S{sid}.pkl")
        if not os.path.exists(path):
            print(f"[WARN] S{sid}: pickle not found — skipping")
            continue

        sig = _load_subject(sid)
        sig["HR"] = derive_hr_from_bvp(sig["BVP"], bvp_rate=RATES["BVP"])

        # Precompute ACC magnitude + high-frequency residual once per subject.
        mag = np.linalg.norm(sig["ACC"], axis=1)
        mag_lf = pd.Series(mag).rolling(RATES["ACC"], center=True, min_periods=1).mean().values
        sig["_acc_mag"] = mag
        sig["_acc_hf"] = np.abs(mag - mag_lf)

        # Usable session length = shortest signal duration in whole seconds.
        total_s = min(
            len(sig["ACC"]) // RATES["ACC"],
            len(sig["BVP"]) // RATES["BVP"],
            len(sig["EDA"]) // RATES["EDA"],
            len(sig["TEMP"]) // RATES["TEMP"],
            len(sig["HR"]),
            len(sig["labels"]) // LABEL_RATE,
        )

        n_windows = 0
        start_s = 0
        while start_s + window_s <= total_s:
            end_s = start_s + window_s

            # Window label from the ground-truth 700 Hz label array.
            seg = sig["labels"][start_s * LABEL_RATE:end_s * LABEL_RATE]
            uniq = np.unique(seg)
            # Keep only pure calm/stressed windows (no mid-window transition).
            if len(uniq) == 1 and int(uniq[0]) in KEEP_LABELS:
                feats = _window_features(sig, start_s, end_s)
                feats["subject"] = sid
                feats["window_start_s"] = start_s
                feats["label"] = KEEP_LABELS[int(uniq[0])]
                rows.append(feats)
                n_windows += 1

            start_s += step_s

        print(f"[OK]   S{sid:>2} | {n_windows} pure windows from {total_s}s session")

    df = pd.DataFrame(rows)

    # Put identifier/label columns first for readability.
    lead = ["subject", "window_start_s", "label"]
    df = df[lead + [c for c in df.columns if c not in lead]]

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\nSaved {len(df)} feature rows × {df.shape[1]} cols → {save_path}")

    return df


def summarize(df: pd.DataFrame) -> None:
    """Print class balance and per-subject window counts."""
    n_calm = int((df["label"] == 0).sum())
    n_stress = int((df["label"] == 1).sum())
    print("\n=== FEATURE TABLE SUMMARY ===")
    print(f"Total windows : {len(df)}")
    print(f"Calm (0)      : {n_calm}")
    print(f"Stressed (1)  : {n_stress}")
    print(f"Ratio (s/c)   : {n_stress / n_calm:.3f}" if n_calm else "Ratio: N/A")
    print(f"Feature cols  : {df.shape[1] - 3} (excluding subject/window_start_s/label)")
    print("\nPer-subject window counts:")
    print(df.groupby("subject")["label"].agg(["count", "mean"]).to_string())


if __name__ == "__main__":
    table = build_feature_table(window_s=30, step_s=15)
    summarize(table)
