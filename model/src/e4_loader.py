"""Empatica E4 zip loader for the WESAD dataset (Shift-Guard project).

The WESAD synchronized pickle (`SX.pkl`) only stores the RAW wrist signals
(ACC, BVP, EDA, TEMP). It deliberately omits the E4's on-device derived signals
HR and IBI. Those live in the separate per-subject export archive
`SX_E4_Data.zip`, which this module reads directly (without unzipping to disk).

E4 CSV format (documented in each archive's info.txt):
    - Continuous signals (ACC, BVP, EDA, TEMP, HR):
        row 0 = session start time (unix UTC timestamp), repeated per column
        row 1 = sample rate in Hz, repeated per column
        row 2+ = the signal samples (ACC has 3 columns x/y/z, others 1 column)
    - IBI.csv (inter-beat intervals, irregularly sampled — no fixed rate):
        row 0 = session start timestamp (+ a literal " IBI" header on col 2)
        row 1+ = [seconds since session start, interval duration in seconds]
    - tags.csv = button-press event marks (may be empty)
    - info.txt = human-readable format description

Usage:
    from e4_loader import load_e4_subject, load_all_e4, summarize

    one = load_e4_subject(2)          # dict of all signals for subject S2
    everything = load_all_e4()        # {sid: subject_dict} for all 15 subjects
    summarize(everything)             # print a capture summary table
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Subjects in the public WESAD release (S1 and S12 excluded by the original authors).
SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# Resolve paths relative to the project root (the folder that contains src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WESAD_ROOT = str(PROJECT_ROOT / "data" / "WESAD")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Continuous E4 signals that share the "start / rate / samples" CSV layout.
CONTINUOUS_SIGNALS = ["ACC", "BVP", "EDA", "TEMP", "HR"]


def _zip_path(sid: int, root: str = WESAD_ROOT) -> str:
    """Return the path to subject `sid`'s E4 export archive."""
    return os.path.join(root, f"S{sid}", f"S{sid}_E4_Data.zip")


def _parse_continuous_csv(raw_bytes: bytes) -> dict:
    """Parse a continuous E4 CSV (ACC/BVP/EDA/TEMP/HR).

    Returns a dict with:
        start_time : float — unix UTC timestamp of the first sample
        rate       : float — sampling rate in Hz
        data       : np.ndarray — shape (N,) for 1-col signals, (N, 3) for ACC
        n_samples  : int
    """
    text = raw_bytes.decode("utf-8").strip()
    lines = text.split("\n")

    # Row 0 = start timestamp (same value repeated per column); take the first.
    start_time = float(lines[0].split(",")[0])
    # Row 1 = sample rate (same value repeated per column); take the first.
    rate = float(lines[1].split(",")[0])

    # Remaining rows are the samples. ACC has 3 comma-separated columns.
    sample_rows = [ln for ln in lines[2:] if ln.strip()]
    data = np.array(
        [[float(v) for v in ln.split(",")] for ln in sample_rows],
        dtype=float,
    )
    # Collapse single-column signals from (N, 1) to a flat (N,) array.
    if data.ndim == 2 and data.shape[1] == 1:
        data = data.ravel()

    return {
        "start_time": start_time,
        "rate": rate,
        "data": data,
        "n_samples": int(data.shape[0]),
    }


def _parse_ibi_csv(raw_bytes: bytes) -> dict:
    """Parse IBI.csv (inter-beat intervals — irregular sampling).

    Returns a dict with:
        start_time     : float — session start unix timestamp
        time_s         : np.ndarray — seconds since session start for each beat
        ibi_s          : np.ndarray — interval duration in seconds (gap to prev beat)
        instant_bpm    : np.ndarray — instantaneous heart rate = 60 / ibi_s
        n_beats        : int
    """
    text = raw_bytes.decode("utf-8").strip()
    lines = text.split("\n")

    start_time = float(lines[0].split(",")[0])

    times, ibis = [], []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        times.append(float(parts[0]))
        ibis.append(float(parts[1]))

    time_s = np.array(times, dtype=float)
    ibi_s = np.array(ibis, dtype=float)
    # Guard against zero-duration intervals before dividing.
    with np.errstate(divide="ignore", invalid="ignore"):
        instant_bpm = np.where(ibi_s > 0, 60.0 / ibi_s, np.nan)

    return {
        "start_time": start_time,
        "time_s": time_s,
        "ibi_s": ibi_s,
        "instant_bpm": instant_bpm,
        "n_beats": int(time_s.shape[0]),
    }


def _parse_tags_csv(raw_bytes: bytes) -> np.ndarray:
    """Parse tags.csv (event button-press timestamps). May be empty."""
    text = raw_bytes.decode("utf-8").strip()
    if not text:
        return np.array([], dtype=float)
    return np.array(
        [float(ln.split(",")[0]) for ln in text.split("\n") if ln.strip()],
        dtype=float,
    )


def load_e4_subject(sid: int, root: str = WESAD_ROOT) -> dict:
    """Open subject `sid`'s E4 zip and capture every signal it contains.

    Reads directly from the archive in memory — nothing is written to disk.

    Returns a dict keyed by signal name:
        'ACC','BVP','EDA','TEMP','HR' → continuous-signal dicts
        'IBI'                          → inter-beat-interval dict
        'tags'                         → np.ndarray of event timestamps
        'info'                         → raw info.txt text
    """
    path = _zip_path(sid, root)
    out: dict = {"subject": sid}

    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())

        for sig in CONTINUOUS_SIGNALS:
            fname = f"{sig}.csv"
            if fname in names:
                out[sig] = _parse_continuous_csv(z.read(fname))
            else:
                out[sig] = None

        out["IBI"] = _parse_ibi_csv(z.read("IBI.csv")) if "IBI.csv" in names else None
        out["tags"] = _parse_tags_csv(z.read("tags.csv")) if "tags.csv" in names else np.array([])
        out["info"] = z.read("info.txt").decode("utf-8") if "info.txt" in names else ""

    return out


def load_all_e4(subjects=SUBJECTS, root: str = WESAD_ROOT) -> dict:
    """Load and capture E4 data for every subject. Returns {sid: subject_dict}."""
    captured = {}
    for sid in subjects:
        path = _zip_path(sid, root)
        if not os.path.exists(path):
            print(f"[WARN] S{sid}: zip not found at {path} — skipping")
            continue
        captured[sid] = load_e4_subject(sid, root)
        print(f"[OK]   S{sid:>2} captured from {os.path.basename(path)}")
    return captured


def summarize(captured: dict) -> pd.DataFrame:
    """Build and print a per-subject summary of what was captured."""
    rows = []
    for sid, d in captured.items():
        hr = d.get("HR")
        ibi = d.get("IBI")
        bvp = d.get("BVP")
        rows.append(
            {
                "subject": f"S{sid}",
                "HR_samples": hr["n_samples"] if hr else 0,
                "HR_rate": hr["rate"] if hr else None,
                "HR_minutes": round(hr["n_samples"] / hr["rate"] / 60, 1) if hr else 0,
                "IBI_beats": ibi["n_beats"] if ibi else 0,
                "BVP_samples": bvp["n_samples"] if bvp else 0,
                "mean_HR_bpm": round(float(np.nanmean(hr["data"])), 1) if hr else None,
                "n_tags": int(len(d.get("tags", []))),
            }
        )
    df = pd.DataFrame(rows)
    print("\n=== E4 CAPTURE SUMMARY ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    data = load_all_e4()
    summary = summarize(data)

    # Persist the captured data so the modeling notebook can load it instantly
    # instead of re-reading 15 zip archives every run.
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = OUTPUTS_DIR / "e4_captured.pkl"
    pd.to_pickle(data, out_path)
    print(f"\nSaved captured E4 data for {len(data)} subjects → {out_path}")
    summary_path = OUTPUTS_DIR / "e4_capture_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary table → {summary_path}")
