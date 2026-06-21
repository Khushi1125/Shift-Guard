"""
tone_analysis.py — Acoustic/vocal tone analysis for Shift-Guard.

What this module does
---------------------
Analyzes HOW someone speaks — pitch, rhythm, vocal quality — by running a
pretrained speech-emotion-recognition model directly on the raw audio waveform.
It returns a stress probability (0.0–1.0) for any .wav recording.

How it differs from semantic_analysis.py
-----------------------------------------
semantic_analysis.py  → *what* was said  (transcription + VADER sentiment)
tone_analysis.py      → *how* it was said (acoustic waveform → stress prob)

These two signals are complementary.  Semantic analysis fails on stress that
is expressed through urgency or exhaustion without negative vocabulary; tone
analysis catches those cases because it works on the audio signal itself.

Model
-----
HuggingFace: superb/wav2vec2-base-superb-er
    • Fine-tuned on IEMOCAP (emotion recognition benchmark)
    • Input: raw 16 kHz mono audio waveform
    • Output: per-class probabilities for neutral / happy / sad / angry
    • Stress proxy: P(angry) — anger is the acoustic nearest-neighbour of
      work stress in IEMOCAP; no fairseq or custom embedding step required.

Quickstart for teammates
------------------------
    from tone_analysis import load_stress_model, get_tone_score

    model = load_stress_model()               # call ONCE — downloads weights
    score = get_tone_score("clip.wav", model) # fast, call per file/chunk
    # score → float in [0.0, 1.0]; 0 = calm, 1 = stressed
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

logger = logging.getLogger(__name__)

MODEL_ID = "superb/wav2vec2-base-superb-er"
TARGET_SAMPLE_RATE = 16_000   # model expects 16 kHz input


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio_mono_16k(wav_path: str) -> np.ndarray:
    """Read a .wav file, mix to mono, resample to 16 kHz if needed.

    Returns a 1-D float32 NumPy array (samples,).
    """
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    # Mix channels to mono: (samples, channels) → (samples,)
    audio = audio.mean(axis=1)

    if sr != TARGET_SAMPLE_RATE:
        # Polyphase resample via scipy — no external codec required.
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(TARGET_SAMPLE_RATE, sr)
        audio = resample_poly(audio, TARGET_SAMPLE_RATE // g, sr // g).astype(np.float32)

    return audio


def _stress_probability(model: Any, logits: torch.Tensor) -> float:
    """Convert model logits to a stress score in [0.0, 1.0].

    Composite stress signal (tuned for IEMOCAP-trained models):

    * **P(angry)**  carries full weight — acoustic signature of acute stress.
    * **P(sad)**    carries 0.65 weight — exhausted/deflated speech overlaps
      heavily with occupational stress but also with genuine sadness, hence
      the discount.
    * **P(fearful)** carries full weight — anxiety-type stress.
    * **P(happy)** is subtracted (×1.0) — strong happiness contradicts stress.
    * **P(neutral)** is subtracted (×0.3) — mild discount only; calm monotone
      speech is neutral but not necessarily stress-free.

    The result is clipped to [0.0, 1.0].  If no recognisable label is found,
    falls back to ``1 – P(calm) – P(neutral)`` and finally to 0.5.

    Parameters
    ----------
    model : AutoModelForAudioClassification
        The loaded classifier (needed for its label map).
    logits : torch.Tensor
        Raw logits tensor of shape (1, num_classes).

    Returns
    -------
    float
        Stress probability in [0.0, 1.0].
    """
    probs = torch.softmax(logits, dim=-1)[0]  # (num_classes,)

    id2label = model.config.id2label  # {0: 'neu', 1: 'hap', ...}

    # Contribution weights — positive = stress signal, negative = calm signal.
    WEIGHTS = {
        # stress
        "ang": 1.00, "angry": 1.00,
        "fea": 1.00, "fear": 1.00, "fearful": 1.00,
        "stress": 1.00, "stressed": 1.00,
        "sad": 0.65,   # partial — also present in non-stressed sad speech
        "dis": 0.50, "disgust": 0.50,
        # calm (subtract)
        "hap": -1.00, "happy": -1.00,
        "neu": -0.30, "neutral": -0.30,
        "cal": -0.80, "calm": -0.80,
    }

    score = 0.0
    matched = False

    for idx, label in id2label.items():
        lbl = label.lower()
        w   = WEIGHTS.get(lbl)
        if w is not None:
            score += w * probs[idx].item()
            matched = True

    if matched:
        # Re-centre: raw score is in roughly [-1, 1]; push to [0, 1].
        return float(max(0.0, min(1.0, (score + 0.5))))

    # Fallback: 1 – P(happy) – 0.5*P(neutral)
    logger.debug("No known labels matched; using 1-P(hap)-0.5*P(neu) fallback.")
    hap = sum(probs[i].item() for i, l in id2label.items() if l.lower() in {"hap", "happy"})
    neu = sum(probs[i].item() for i, l in id2label.items() if l.lower() in {"neu", "neutral"})
    if hap > 0.0 or neu > 0.0:
        return max(0.0, min(1.0, 1.0 - hap - 0.5 * neu))

    logger.warning("Could not derive stress score from labels: %s", id2label)
    return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_stress_model() -> Tuple[Any, Any]:
    """Download and initialise the speech-emotion classifier (call once).

    Downloads the feature extractor config and model weights from HuggingFace
    on the first call; subsequent calls use the local cache (fast).

    Model: ``superb/wav2vec2-base-superb-er``
      • Fine-tuned Wav2Vec2-Base on IEMOCAP emotion recognition
      • ~94 M parameters, ~380 MB on disk
      • No fairseq dependency — works on Python 3.11+

    Returns
    -------
    tuple[AutoFeatureExtractor, AutoModelForAudioClassification]
        A ``(feature_extractor, model)`` pair.  Pass this tuple unchanged into
        :func:`get_tone_score` — do **not** unpack it yourself.

    Example
    -------
    ::

        stress_model = load_stress_model()     # once at startup
        for wav in audio_files:
            score = get_tone_score(wav, stress_model)
    """
    logger.debug("Loading feature extractor from %s", MODEL_ID)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

    logger.debug("Loading classification model from %s", MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()

    logger.debug(
        "Stress model ready — labels: %s", dict(model.config.id2label)
    )
    return feature_extractor, model


def get_tone_score(wav_path: str, stress_model: Tuple[Any, Any]) -> float:
    """Score the acoustic stress level of a .wav audio recording.

    Parameters
    ----------
    wav_path : str
        Path to a .wav audio file.  Any sample rate is accepted; audio is
        resampled to 16 kHz internally.
    stress_model : tuple
        The ``(feature_extractor, model)`` pair returned by
        :func:`load_stress_model`.  Pass the whole tuple — do not unpack.

    Returns
    -------
    float
        Probability of stress in [0.0, 1.0].

        Interpretation guide:
          0.0 – 0.35   calm / relaxed
          0.35 – 0.55  borderline / uncertain
          0.55 – 1.0   stressed / agitated

        Returns 0.5 (uncertain) on any failure (bad file, model error) so
        the backend never raises an exception during a live demo.

    Example
    -------
    ::

        model = load_stress_model()
        score = get_tone_score("recording.wav", model)
        print(f"Stress: {score:.2f}")
    """
    try:
        feature_extractor, model = stress_model

        # Load audio as mono float32 array at 16 kHz.
        audio = _load_audio_mono_16k(wav_path)

        # Tokenise: pads/truncates to a fixed-length float tensor expected by
        # Wav2Vec2's feature extractor.
        inputs = feature_extractor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        return _stress_probability(model, outputs.logits)

    except Exception as exc:
        logger.error(
            "get_tone_score failed for %r — %s: %s. Returning 0.5 (uncertain).",
            wav_path, type(exc).__name__, exc,
        )
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Quick local test (never runs when imported by the backend)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "tests")

    wav_files = sorted(
        f for f in os.listdir(TEST_DIR) if f.endswith(".wav")
    ) if os.path.isdir(TEST_DIR) else []

    if not wav_files:
        print("No .wav files found in tests/ — nothing to score.")
        sys.exit(0)

    print("Loading stress model (downloads on first run, then cached)…")
    stress_model = load_stress_model()
    _, m = stress_model
    print(f"Model ready.  Labels: {dict(m.config.id2label)}\n")

    col_w = max(len(f) for f in wav_files) + 2
    print(f"  {'File':<{col_w}}  tone_score   interpretation")
    print("  " + "-" * (col_w + 30))
    for fname in wav_files:
        path = os.path.join(TEST_DIR, fname)
        score = get_tone_score(path, stress_model)
        label = (
            "calm"       if score < 0.35 else
            "borderline" if score < 0.55 else
            "stressed"
        )
        print(f"  {fname:<{col_w}}  {score:.4f}       {label}")
