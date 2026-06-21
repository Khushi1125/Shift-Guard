"""
final_scoring.py — Locked public interface for the Shift-Guard ML pipeline.

This module is the single import point for the backend/frontend teammate.

Four pure scoring functions (stateless):
    predict_sensor_score(temp_mean, temp_slope, bpm_mean, bpm_std) -> float
    run_voice_checkin(wav_path)                                     -> dict
    compute_final_risk_score(sensor_proba, combined_voice)          -> dict
    check_and_get_intervention(final_score)                         -> dict

Two stateful update functions (recommended for live use):
    on_sensor_update(temp_mean, temp_slope, bpm_mean, bpm_std)      -> dict
    on_voice_update(wav_path)                                        -> dict

One callback registration:
    register_score_callback(fn)   — wire final_score to your WebSocket push

How the two signals combine
---------------------------
Sensor score updates every ~10 s from streaming hardware data.
Voice score updates only when the user clicks "Check In" on the dashboard.
These two clocks are completely independent.

The stateful update functions always combine the MOST RECENT value of each
signal, regardless of when each was last updated:

    # at server startup:
    register_score_callback(lambda result: ws.send_json(result))

    # every ~10 s, called by your sensor polling loop:
    on_sensor_update(temp, temp_slope, bpm, bpm_std)
    # → uses latest sensor + last known voice (even if minutes old)
    # → immediately pushes updated final_score to dashboard via callback

    # only when user clicks mic button:
    on_voice_update("/tmp/clip.wav")
    # → uses latest voice + last known sensor
    # → immediately pushes updated final_score to dashboard via callback

See INTEGRATION.md at the repo root for the full contract.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
from typing import Any

import numpy as np

# ── make src/ importable when run as a script ─────────────────────────────
_SRC = pathlib.Path(__file__).parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from semantic_analysis import get_semantic_score, transcribe_audio
from tone_analysis import get_tone_score, load_stress_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Paths (resolved relative to the repo root so the module works regardless of
# the current working directory when the backend imports it)
# ─────────────────────────────────────────────────────────────────────────────
_ONNX_PATH         = _ROOT / "outputs" / "baseline_model.onnx"
_PLACEHOLDERS_PATH = _ROOT / "outputs" / "acc_placeholders.json"

# ─────────────────────────────────────────────────────────────────────────────
# Scoring weights  (do not change without re-evaluating on validation data)
# ─────────────────────────────────────────────────────────────────────────────
_SEMANTIC_W = 0.5   # within the voice channel: semantic vs tone
_TONE_W     = 0.5

_SENSOR_W   = 0.7   # sensor vs voice in the final blend
_VOICE_W    = 0.3   # sensor F1=0.86 (LOSO/WESAD) vs voice ~75% on 8 clips

# Intervention threshold — scores at or above this are flagged as high-risk.
# Set to 0.60 rather than a harder cutoff: with only 4 live sensor channels
# and 14 imputed from training means, the sensor model's dynamic range is
# compressed toward the centre.  0.60 is calibrated to separate the stressed
# and calm test cases while keeping false-positive rate low.
_INTERVENTION_THRESHOLD = 0.60


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons — loaded once, reused on every call
# (loading each takes 3-10 s; reloading per call would ruin live-demo latency)
# ─────────────────────────────────────────────────────────────────────────────
_sensor_session: Any = None       # onnxruntime.InferenceSession
_stress_model:   Any = None       # (feature_extractor, Wav2Vec2 model) pair
_imputation:     dict | None = None  # per-feature means from training data


def _get_sensor_session():
    """Return the ONNX inference session, creating it on the first call."""
    global _sensor_session
    if _sensor_session is None:
        import onnxruntime as ort
        _sensor_session = ort.InferenceSession(str(_ONNX_PATH))
        logger.debug("ONNX sensor session loaded from %s", _ONNX_PATH)
    return _sensor_session


def _get_stress_model():
    """Return the tone stress model tuple, downloading weights once if needed."""
    global _stress_model
    if _stress_model is None:
        _stress_model = load_stress_model()
        logger.debug("Tone stress model loaded")
    return _stress_model


def _get_imputation() -> dict:
    """Return the per-feature mean values used to fill in hardware-gap channels."""
    global _imputation
    if _imputation is None:
        if not _PLACEHOLDERS_PATH.exists():
            raise FileNotFoundError(
                f"Imputation file not found: {_PLACEHOLDERS_PATH}\n"
                "Run final_scoring.py directly once to generate it: "
                "python src/final_scoring.py --generate-placeholders"
            )
        with open(_PLACEHOLDERS_PATH) as f:
            data = json.load(f)
        _imputation = data["feature_means"]
        logger.debug("Imputation values loaded from %s", _PLACEHOLDERS_PATH)
    return _imputation


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION 1 — Sensor score
# ─────────────────────────────────────────────────────────────────────────────

def predict_sensor_score(
    temp_mean:  float,
    temp_slope: float,
    bpm_mean:   float,
    bpm_std:    float,
) -> float:
    """Estimate stress probability from wearable sensor readings.

    You do NOT need to pass accelerometer or EDA data — those channels are
    unavailable in the live demo and are automatically filled with safe
    placeholder values computed from the training dataset.

    Parameters
    ----------
    temp_mean : float
        Average skin temperature over the last ~10 seconds, in degrees Celsius.
        Normal range: 33–37 °C.  Higher than usual = potential stress signal.
    temp_slope : float
        How fast skin temperature is rising or falling (°C per second).
        Negative = cooling (often a stress response).  Typical range: −0.01 to +0.01.
    bpm_mean : float
        Average heart rate over the last ~10 seconds, in beats per minute.
        Normal resting: 60–80 BPM.  Stress typically pushes this above 90.
    bpm_std : float
        How much heart rate varied over the window (standard deviation, BPM).
        Higher variability can indicate either exercise or stress.

    Returns
    -------
    float
        Stress probability in [0.0, 1.0].
        0.0 – 0.4  = calm / low risk
        0.4 – 0.65 = borderline
        0.65 – 1.0 = likely stressed

        Returns 0.5 (neutral / uncertain) if inference fails for any reason.
    """
    try:
        imp = _get_imputation()
        sess = _get_sensor_session()

        # The underlying ONNX model was trained on 18 features.  Build the full
        # vector by inserting live values at their correct positions and filling
        # everything else with training-data means (mean imputation).
        feature_order = [
            'acc_mag_mean', 'acc_mag_std', 'acc_hf_mean',
            'bvp_mean',     'bvp_std',
            'hr_mean',      'hr_std',  'hr_slope', 'hr_min', 'hr_max',
            'temp_mean',    'temp_slope', 'temp_delta',
            'eda_mean',     'eda_std', 'eda_slope', 'eda_min', 'eda_max',
        ]
        # Start from training means, then override with live readings.
        live_values = {
            'hr_mean':    bpm_mean,
            'hr_std':     bpm_std,
            'temp_mean':  temp_mean,
            'temp_slope': temp_slope,
        }
        vector = [live_values.get(feat, imp[feat]) for feat in feature_order]

        x = np.array(vector, dtype=np.float32).reshape(1, -1)
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: x})

        # outputs[1] shape: (1, 2) — [P(calm), P(stressed)]
        stress_proba = float(outputs[1][0][1])
        return max(0.0, min(1.0, stress_proba))

    except Exception as exc:
        print(
            f"[final_scoring] ERROR in predict_sensor_score: {type(exc).__name__}: {exc}"
            f"\n  Returning 0.5 (neutral) to avoid crashing the demo."
        )
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION 2 — Voice check-in
# ─────────────────────────────────────────────────────────────────────────────

def run_voice_checkin(wav_path: str) -> dict:
    """Analyse a short voice recording for stress signals.

    This is the slow path — it calls the Deepgram transcription API (network)
    and runs a local Wav2Vec2 model.  Only call it when the user explicitly
    clicks "Check In" on the dashboard, NOT continuously.

    Audio input comes from the browser (laptop microphone via MediaRecorder API).
    The backend should save the uploaded audio blob to a .wav file and pass that
    path here.

    Parameters
    ----------
    wav_path : str
        Path to the .wav file saved by the backend from the browser upload.
        16 kHz mono is preferred; other formats are handled automatically.

    Returns
    -------
    dict with keys:
        transcript     (str)   — What the user said.  Empty string if Deepgram
                                 fails or if DEEPGRAM_API_KEY is not set.
        semantic_score (float) — VADER sentiment of the transcript.
                                 Range: −1.0 (very negative) to +1.0 (positive).
                                 More negative → more likely stressed.
        tone_stress    (float) — How stressed the speaker SOUNDS (voice quality),
                                 regardless of the words.  Range: 0.0–1.0.
        combined_voice (float) — Blended voice stress score, range 0.0–1.0.
                                 Pass this directly to compute_final_risk_score().

    Notes
    -----
    All failures are caught and return safe neutral defaults so the demo
    never hard-crashes on a bad audio file or API outage.
    """
    try:
        stress_model = _get_stress_model()

        transcript     = transcribe_audio(wav_path)
        semantic_score = get_semantic_score(transcript)
        tone_stress    = get_tone_score(wav_path, stress_model)

        # Rescale semantic from [−1, +1] to [0, 1] where 1 = more stressed,
        # then blend equally with the acoustic tone score.
        semantic_stress = (-semantic_score + 1.0) / 2.0
        combined_voice  = _SEMANTIC_W * semantic_stress + _TONE_W * tone_stress

        return {
            "transcript":     transcript,
            "semantic_score": round(semantic_score,  4),
            "tone_stress":    round(tone_stress,     4),
            "combined_voice": round(combined_voice,  4),
        }

    except Exception as exc:
        print(
            f"[final_scoring] ERROR in run_voice_checkin: {type(exc).__name__}: {exc}"
            f"\n  Returning neutral defaults to avoid crashing the demo."
        )
        return {
            "transcript":     "",
            "semantic_score": 0.0,
            "tone_stress":    0.5,
            "combined_voice": 0.5,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION 3 — Final blended risk score
# ─────────────────────────────────────────────────────────────────────────────

def compute_final_risk_score(sensor_proba: float, combined_voice: float) -> dict:
    """Combine sensor and voice scores into the final stress risk score.

    Both inputs should be in [0, 1] where 1 = most stressed.  The sensor
    model carries more weight (0.7) because it was validated more rigorously
    (F1 = 0.86 via Leave-One-Subject-Out cross-validation on 15 subjects)
    than the voice layer (~75% on 8 manual test clips).

    Parameters
    ----------
    sensor_proba : float
        Output of predict_sensor_score().  Range: 0.0–1.0.
    combined_voice : float
        The "combined_voice" value from run_voice_checkin()'s return dict.
        Range: 0.0–1.0.

    Returns
    -------
    dict with keys:
        sensor_proba   (float) — the sensor input, echoed back for logging
        combined_voice (float) — the voice input, echoed back for logging
        final_score    (float) — weighted blend, range 0.0–1.0.
                                 0.7 × sensor_proba + 0.3 × combined_voice
    """
    try:
        raw = _SENSOR_W * float(sensor_proba) + _VOICE_W * float(combined_voice)
        final_score = max(0.0, min(1.0, raw))
        return {
            "sensor_proba":   round(float(sensor_proba),   4),
            "combined_voice": round(float(combined_voice), 4),
            "final_score":    round(final_score,           4),
        }
    except Exception as exc:
        print(
            f"[final_scoring] ERROR in compute_final_risk_score: {type(exc).__name__}: {exc}"
            f"\n  Returning neutral defaults."
        )
        return {"sensor_proba": 0.5, "combined_voice": 0.5, "final_score": 0.5}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION 4 — Intervention decision
# ─────────────────────────────────────────────────────────────────────────────

_INTERVENTION_MESSAGES = [
    # Ordered most → least severe; first matching threshold wins.
    # (threshold, key, text)
    # The "key" links each message to its pre-generated TTS clip in
    # model/outputs/intervention_<key>.wav (see tts.INTERVENTION_TEXTS — keep the
    # text in both files in sync).
    (0.80, "high",     "Your stress level is quite high right now. Consider stepping away for 5 minutes — a short walk or a few deep breaths can help reset your nervous system."),
    (0.60, "elevated", "You're showing signs of elevated stress. Try a brief grounding exercise: name 5 things you can see, 4 you can hear, 3 you can touch."),
]


def _get_intervention_audio(key: str, text: str) -> str | None:
    """Return a playable .wav path for an intervention, or None if unavailable.

    Uses the pre-generated clip in model/outputs/ when present.  If it's missing
    (e.g. tts.py was never run), it attempts a one-time live synthesis and
    caches the result.  Any failure (no key, no network) returns None so the
    caller falls back to text-only — TTS is a nice-to-have, never a blocker.
    """
    try:
        from tts import intervention_audio_path, synthesize_speech

        dest = intervention_audio_path(key)
        if dest.exists():
            return str(dest)

        # Lazy fallback: synthesize once and cache.  Returns "" on failure.
        path = synthesize_speech(text, str(dest))
        return path or None

    except Exception as exc:
        logger.warning(
            "Could not obtain intervention audio for %r: %s: %s",
            key, type(exc).__name__, exc,
        )
        return None


def check_and_get_intervention(final_score: float) -> dict:
    """Decide whether to show the user a stress-relief intervention.

    Call this after compute_final_risk_score() whenever you want to know
    if the dashboard should display an alert.  It is cheap (no ML inference)
    and can be called any time final_score updates.

    Parameters
    ----------
    final_score : float
        The "final_score" value from compute_final_risk_score()'s return dict.
        Range: 0.0–1.0.

    Returns
    -------
    dict with keys:
        triggered   (bool)      — True if the score is high enough to show an alert.
        text        (str|None)  — The intervention message to display, or None if
                                  no intervention is needed.
        audio_path  (str|None)  — Path to a spoken .wav of the message (Deepgram
                                  TTS), ready to play in the browser.  None when
                                  no intervention fires or audio can't be made
                                  (in which case fall back to showing `text`).
    """
    try:
        score = float(final_score)
        for threshold, key, message in _INTERVENTION_MESSAGES:
            if score >= threshold:
                audio_path = _get_intervention_audio(key, message)
                return {"triggered": True, "text": message, "audio_path": audio_path}
        return {"triggered": False, "text": None, "audio_path": None}

    except Exception as exc:
        print(
            f"[final_scoring] ERROR in check_and_get_intervention: "
            f"{type(exc).__name__}: {exc}\n  Returning no-intervention default."
        )
        return {"triggered": False, "text": None, "audio_path": None}


# ─────────────────────────────────────────────────────────────────────────────
# LIVE STATE — in-memory most-recent values
#
# Both values default to 0.5 (neutral/uncertain) so the system produces a
# safe, non-alarming baseline score before any real data arrives.
#
# Voice value does NOT expire or decay between mic clicks — it stays at its
# last known reading until the user checks in again.  This is intentional:
# the alternative (resetting to 0.5 after N seconds) would make the system
# ignore real stress signals just because the user hasn't clicked recently.
# ─────────────────────────────────────────────────────────────────────────────

_state: dict = {
    "latest_sensor_proba":  0.5,
    "latest_voice_result":  {"combined_voice": 0.5},
}

# Optional callback registered by the backend to push final_score to the
# dashboard via WebSocket or any other transport.  Set via
# register_score_callback(); None = no push (safe default).
_score_update_callback = None


def register_score_callback(fn) -> None:
    """Register a function to call every time final_score is recomputed.

    The backend should call this ONCE at server startup, passing whatever
    function sends data to the frontend (WebSocket, SSE, etc.).

    The callback receives the full result dict — every intermediate value
    plus final_score and the intervention alert — so the dashboard can
    display anything it needs without making a second request.

    Parameters
    ----------
    fn : callable
        Function that accepts one argument: the result dict produced by
        _recompute_and_push().  Example keys:
            transcript, semantic_score, tone_stress, combined_voice,
            sensor_proba, final_score, alert: {triggered, text, audio_path}

    Example
    -------
    ::

        # FastAPI + WebSocket example:
        import final_scoring as fs

        @app.on_event("startup")
        async def startup():
            fs.register_score_callback(
                lambda result: asyncio.create_task(ws.send_json(result))
            )
    """
    global _score_update_callback
    _score_update_callback = fn


def _recompute_and_push() -> dict:
    """Recompute final score from current state; push to dashboard if callback set.

    Internal — called by on_sensor_update and on_voice_update.
    Always reads from _state so both update paths share the same logic.
    """
    sensor_proba   = _state["latest_sensor_proba"]
    combined_voice = _state["latest_voice_result"]["combined_voice"]

    risk  = compute_final_risk_score(sensor_proba, combined_voice)
    alert = check_and_get_intervention(risk["final_score"])

    full_result = {
        **_state["latest_voice_result"],   # transcript, semantic_score, tone_stress, combined_voice
        **risk,                             # sensor_proba, combined_voice (same), final_score
        "alert": alert,
    }

    if _score_update_callback is not None:
        try:
            _score_update_callback(full_result)
        except Exception as exc:
            # Never let a bad callback crash the scoring loop.
            print(f"[final_scoring] WARNING: score callback raised {type(exc).__name__}: {exc}")

    return full_result


# ─────────────────────────────────────────────────────────────────────────────
# STATEFUL UPDATE FUNCTIONS — recommended entry points for live use
# ─────────────────────────────────────────────────────────────────────────────

def on_sensor_update(
    temp_mean:  float,
    temp_slope: float,
    bpm_mean:   float,
    bpm_std:    float,
) -> dict:
    """Process a new sensor window and immediately update the final risk score.

    Call this every time a ~10-second sensor window closes (i.e. when your
    hardware polling loop has a fresh batch of HR + TEMP readings).

    This function:
      1. Runs predict_sensor_score() with the new readings.
      2. Stores the result as the latest known sensor value.
      3. Recomputes final_score using the new sensor value combined with
         WHATEVER voice value was most recently recorded — even if that
         voice reading is minutes old.  The voice value is never reset.
      4. Calls the registered score callback (if any) so the dashboard
         receives the updated score immediately, without waiting for the
         next voice check-in.

    Parameters
    ----------
    temp_mean  : float  — average skin temperature, °C
    temp_slope : float  — temperature trend, °C/s
    bpm_mean   : float  — average heart rate, BPM
    bpm_std    : float  — heart rate variability, BPM

    Returns
    -------
    dict
        Same full result dict that _recompute_and_push() returns:
        transcript, semantic_score, tone_stress, combined_voice,
        sensor_proba, final_score, alert.
    """
    _state["latest_sensor_proba"] = predict_sensor_score(
        temp_mean, temp_slope, bpm_mean, bpm_std
    )
    return _recompute_and_push()


def on_voice_update(wav_path: str) -> dict:
    """Process a mic check-in and immediately update the final risk score.

    Call this ONLY when the user clicks the "Check In" button on the
    dashboard — NOT on a polling loop.  This function is slow (Deepgram API
    + local ML inference, ~2–5 s total).

    This function:
      1. Runs run_voice_checkin() on the uploaded audio file.
      2. Stores the result as the latest known voice value.
      3. Recomputes final_score using the new voice value combined with
         WHATEVER sensor value was most recently recorded.
      4. Calls the registered score callback so the dashboard updates
         immediately rather than waiting for the next sensor window.

    Parameters
    ----------
    wav_path : str
        Path to the .wav file the backend saved from the browser upload.
        Must be a valid PCM .wav file (convert from WebM/Opus first if
        the browser's MediaRecorder API produced a compressed format).

    Returns
    -------
    dict
        Same full result dict: transcript, semantic_score, tone_stress,
        combined_voice, sensor_proba, final_score, alert.
    """
    _state["latest_voice_result"] = run_voice_checkin(wav_path)
    return _recompute_and_push()


# ─────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_placeholders():
    """Compute and save acc_placeholders.json from the training CSV."""
    import pandas as pd

    csv_path = _ROOT / "outputs" / "features_30s.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run extract_features.py first.")
        return

    df = pd.read_csv(csv_path)
    feature_order = [
        'acc_mag_mean', 'acc_mag_std', 'acc_hf_mean',
        'bvp_mean',     'bvp_std',
        'hr_mean',      'hr_std',  'hr_slope', 'hr_min', 'hr_max',
        'temp_mean',    'temp_slope', 'temp_delta',
        'eda_mean',     'eda_std', 'eda_slope', 'eda_min', 'eda_max',
    ]
    means = {col: float(df[col].mean()) for col in feature_order}
    payload = {"feature_means": means, "model_feature_order": feature_order}
    _PLACEHOLDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_PLACEHOLDERS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved imputation values to {_PLACEHOLDERS_PATH}")


if __name__ == "__main__":
    import sys as _sys

    if "--generate-placeholders" in _sys.argv:
        _generate_placeholders()
        _sys.exit(0)

    # Quick smoke-test: score one sensor reading and print the result.
    logging.basicConfig(level=logging.WARNING)
    print("predict_sensor_score(temp_mean=36.8, temp_slope=-0.001, bpm_mean=92, bpm_std=8)")
    score = predict_sensor_score(36.8, -0.001, 92.0, 8.0)
    print(f"  sensor_proba = {score:.4f}")
    result = compute_final_risk_score(score, 0.5)
    print(f"  final_score  = {result['final_score']:.4f}")
    alert = check_and_get_intervention(result["final_score"])
    print(f"  triggered    = {alert['triggered']}")
    if alert["triggered"]:
        print(f"  text         = {alert['text']}")
