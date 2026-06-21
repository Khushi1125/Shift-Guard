"""
tts.py — Text-to-speech for Shift-Guard interventions.

What this module does
---------------------
Turns an intervention *message* (plain text) into a spoken *audio file* (.wav)
using Deepgram's text-to-speech ("speak") API.  This is the voice side of the
loop: when the final risk score crosses a threshold, the dashboard can play a
calm spoken prompt instead of only showing text.

Two functions
-------------
  1. synthesize_speech(text, out_path)  → str
       Synthesizes one message to a .wav file and returns the saved path
       (or "" on failure — never raises, so a live demo can't crash on it).

  2. generate_intervention_audio()      → dict
       Pre-generates the fixed intervention messages once and caches them in
       model/outputs/.  Run this ahead of a demo so playback is instant and
       needs no network at trigger time.

Quickstart for teammates
------------------------
    from tts import synthesize_speech
    path = synthesize_speech("You seem stressed, take a short break.",
                             "outputs/alert.wav")

Environment setup
-----------------
Reuses the same Deepgram key as semantic_analysis.py.  Copy .env.example → .env
and set:

    DEEPGRAM_API_KEY=your_key_here

SDK note
--------
Targets deepgram-sdk 7.x:  `dg.speak.v1.audio.generate(...)` returns an
iterator of audio byte chunks.  The Deepgram SDK changes its surface often —
if you upgrade the SDK, re-check this call signature.
"""

from __future__ import annotations

import logging
import os
import pathlib

from dotenv import load_dotenv
from deepgram import DeepgramClient

load_dotenv()

logger = logging.getLogger(__name__)

# Aura voice used for all interventions.  "asteria" is a calm, neutral female
# voice — a deliberate choice for stress relief (a harsh voice would defeat the
# purpose).  Swap for any other "aura-*-en" model if you prefer.
_DEFAULT_VOICE = "aura-asteria-en"

# Where pre-generated intervention clips are cached.  Kept next to the other
# model artifacts so the backend can serve them from a known location.
_OUTPUTS_DIR = pathlib.Path(__file__).parent.parent / "outputs"

# The fixed intervention messages, keyed by a stable name.  These MUST stay in
# sync with _INTERVENTION_MESSAGES in final_scoring.py — the key is what links a
# triggered threshold to its cached .wav file.
INTERVENTION_TEXTS = {
    "high": (
        "Your stress level is quite high right now. Consider stepping away for "
        "5 minutes — a short walk or a few deep breaths can help reset your "
        "nervous system."
    ),
    "elevated": (
        "You're showing signs of elevated stress. Try a brief grounding "
        "exercise: name 5 things you can see, 4 you can hear, 3 you can touch."
    ),
}


def synthesize_speech(
    text: str,
    out_path: str,
    model: str = _DEFAULT_VOICE,
) -> str:
    """Synthesize spoken audio from text and save it as a .wav file.

    Parameters
    ----------
    text : str
        The message to speak.  Must be non-empty.
    out_path : str
        Destination file path for the generated audio (a .wav file).
        Parent directories are created automatically if missing.
    model : str, optional
        Deepgram Aura voice to use (default "aura-asteria-en", a calm voice).

    Returns
    -------
    str
        The path the audio was written to on success, or "" on any failure
        (missing key, empty text, network error, write error).  This function
        never raises — it logs the problem and returns "" so callers in a live
        demo can degrade gracefully (e.g. fall back to showing text only).
    """
    if not text:
        logger.error("synthesize_speech called with empty text; nothing to do.")
        return ""

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error(
            "DEEPGRAM_API_KEY is not set. "
            "Create a .env file using .env.example as a template."
        )
        return ""

    try:
        out = pathlib.Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        client = DeepgramClient(api_key=api_key)

        # deepgram-sdk 7.x: speak.v1.audio.generate streams the audio back as an
        # iterator of byte chunks.  container="wav" + encoding="linear16" gives
        # a standard 16-bit PCM WAV that any browser <audio> tag can play.
        chunks = client.speak.v1.audio.generate(
            text=text,
            model=model,
            container="wav",
            encoding="linear16",
        )

        with open(out, "wb") as f:
            for chunk in chunks:
                if chunk:
                    f.write(chunk)

        logger.debug("Synthesized %d chars → %s", len(text), out)
        return str(out)

    except Exception as exc:
        logger.error(
            "Text-to-speech failed — %s: %s. "
            "Check your API key and network connection.",
            type(exc).__name__,
            exc,
        )
        return ""


def intervention_audio_path(key: str) -> pathlib.Path:
    """Return the cache path for a given intervention key (no I/O)."""
    return _OUTPUTS_DIR / f"intervention_{key}.wav"


def generate_intervention_audio(force: bool = False) -> dict:
    """Pre-generate the fixed intervention clips into model/outputs/.

    Run this once before a demo so that, at trigger time, the system only has to
    return an already-saved file path (instant, offline) instead of calling the
    TTS API live.

    Parameters
    ----------
    force : bool, optional
        If True, regenerate even when the cached file already exists.
        If False (default), skip any clip that's already on disk.

    Returns
    -------
    dict
        Maps each intervention key ("high", "elevated") to the saved .wav path,
        or to "" for any clip that failed to generate.
    """
    results: dict[str, str] = {}
    for key, text in INTERVENTION_TEXTS.items():
        dest = intervention_audio_path(key)
        if dest.exists() and not force:
            logger.debug("Intervention audio already cached: %s", dest)
            results[key] = str(dest)
            continue
        results[key] = synthesize_speech(text, str(dest))
    return results


if __name__ == "__main__":
    # Pre-generate both intervention clips when run directly:
    #   python model/src/tts.py
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("--- Generating intervention audio ---")
    out = generate_intervention_audio(force=True)
    for key, path in out.items():
        status = path if path else "FAILED"
        print(f"  {key:10s} → {status}")
