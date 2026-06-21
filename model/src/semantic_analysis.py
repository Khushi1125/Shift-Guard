"""
semantic_analysis.py — Semantic content analysis for Shift-Guard.

What this module does
---------------------
Analyzes the SEMANTIC content of a voice recording: *what* someone said, not
*how* they said it.  It provides two composable functions:

  1. transcribe_audio(audio_file_path)  → str
       Sends a .wav file to the Deepgram API and returns the transcript text.

  2. get_semantic_score(transcript)     → float  (range: -1.0 to +1.0)
       Runs VADER sentiment analysis on the transcript and returns a compound
       score (negative = stressed / negative language, positive = calm).

Quickstart for teammates
------------------------
    from semantic_analysis import transcribe_audio, get_semantic_score

    transcript = transcribe_audio("path/to/recording.wav")
    score      = get_semantic_score(transcript)

Environment setup
-----------------
Copy .env.example → .env and fill in your Deepgram API key:

    DEEPGRAM_API_KEY=your_key_here

The functions read this key automatically via python-dotenv; you do not need to
set the environment variable manually.

SDK note
--------
This file targets deepgram-sdk 7.x  (`dg.listen.v1.media.transcribe_file`).
The Deepgram SDK changes its API surface frequently — if you upgrade the SDK,
re-check the call signature in this file.
"""

import logging
import os

from dotenv import load_dotenv
from deepgram import DeepgramClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

# Module-level logger.  Callers (or the backend) control the log level and
# handler; this module never touches the root logger or adds its own handlers.
# To see debug output during local development, set the level in your script:
#   logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe a .wav audio file using Deepgram's nova-2 speech-to-text model.

    Parameters
    ----------
    audio_file_path : str
        Absolute or relative path to a .wav audio file.

    Returns
    -------
    str
        The transcribed text.  Returns an empty string ("") if transcription
        fails for any reason (missing file, network error, invalid API key,
        unexpected API response).  An error message is printed to stderr
        describing what went wrong.

    Notes
    -----
    - Requires the environment variable ``DEEPGRAM_API_KEY`` to be set (see
      .env.example).
    - Uses Deepgram model "nova-2" with smart_format=True (adds punctuation,
      capitalisation, and numbers formatting automatically).
    - This function is synchronous and blocks until the API call completes.
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error(
            "DEEPGRAM_API_KEY is not set. "
            "Create a .env file using .env.example as a template."
        )
        return ""

    logger.debug("transcribe_audio called: path=%r", audio_file_path)

    try:
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()
        logger.debug("Audio file read: %d bytes", len(audio_bytes))
    except FileNotFoundError:
        logger.error(
            "Audio file not found: %r. Check the path and try again.",
            audio_file_path,
        )
        return ""
    except OSError as exc:
        logger.error("Could not read audio file %r: %s", audio_file_path, exc)
        return ""

    try:
        client = DeepgramClient(api_key=api_key)

        # deepgram-sdk 7.x: dg.listen.v1.media.transcribe_file(request=<bytes>, **options)
        response = client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model="nova-2",
            smart_format=True,
        )

        transcript = response.results.channels[0].alternatives[0].transcript
        logger.debug("Transcription succeeded: %d chars", len(transcript))
        return transcript

    except IndexError:
        logger.error(
            "Deepgram returned a response but it contained no transcription "
            "channels or alternatives. The audio file may be silent or in an "
            "unsupported format."
        )
        return ""
    except Exception as exc:
        logger.error(
            "Transcription failed — %s: %s. "
            "Check your API key, network connection, and that the audio file is a valid .wav.",
            type(exc).__name__,
            exc,
        )
        return ""


def get_semantic_score(transcript: str) -> float:
    """Score the emotional valence of a transcript using VADER sentiment analysis.

    Parameters
    ----------
    transcript : str
        Plain text to analyse.  Typically the output of ``transcribe_audio()``.
        Pass an empty string to get a neutral score (0.0) without errors.

    Returns
    -------
    float
        VADER compound sentiment score in the range [-1.0, +1.0].

        Interpretation guide:
          -1.0 to -0.05  — negative / stressed-sounding language
           -0.05 to 0.05 — neutral
           0.05 to +1.0  — positive / calm-sounding language

        Returns 0.0 (neutral) when ``transcript`` is an empty string.

    Notes
    -----
    VADER is optimised for short social-media-style text; it also works
    reasonably well on conversational speech transcripts.  It does not require
    a network connection or API key.
    """
    if not transcript:
        logger.debug("get_semantic_score received empty transcript; returning 0.0")
        return 0.0

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(transcript)
    compound = scores["compound"]
    logger.debug(
        "VADER scores: pos=%.3f neu=%.3f neg=%.3f compound=%.4f",
        scores["pos"], scores["neu"], scores["neg"], compound,
    )
    return compound


if __name__ == "__main__":
    # Quick local test — only runs when you execute this file directly, never
    # when the backend imports it.  Replace the path with any .wav you have.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    TEST_FILE = "test_audio/calm_test.wav"

    print(f"--- Semantic Analysis Test ---")
    print(f"Audio file : {TEST_FILE}")

    transcript = transcribe_audio(TEST_FILE)
    print(f"Transcript : {transcript!r}")

    score = get_semantic_score(transcript)
    print(f"Sentiment score (compound): {score:.4f}  "
          f"({'negative/stressed' if score < -0.05 else 'positive/calm' if score > 0.05 else 'neutral'})")
