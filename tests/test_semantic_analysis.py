"""
Unit tests for src/semantic_analysis.py.

Two test groups:
  - Mocked tests (23 tests): run offline, no API key needed, always run in CI.
  - Real audio tests (TestRealAudio): hit the real Deepgram API using the .wav
    files in tests/. These are skipped automatically when DEEPGRAM_API_KEY is
    not set. Run them locally after filling in your .env file:

        pytest tests/test_semantic_analysis.py -v -k real_audio

Run all mocked tests:
    pytest tests/test_semantic_analysis.py -v
"""

import logging
import os
import sys
import types
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Make sure src/ is on the path regardless of where pytest is invoked from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import semantic_analysis
from semantic_analysis import get_semantic_score, transcribe_audio


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_deepgram_response(transcript_text: str):
    """Build a minimal fake Deepgram ListenV1Response."""
    alt = MagicMock()
    alt.transcript = transcript_text
    channel = MagicMock()
    channel.alternatives = [alt]
    results = MagicMock()
    results.channels = [channel]
    response = MagicMock()
    response.results = results
    return response


# ──────────────────────────────────────────────────────────────────────────────
# get_semantic_score
# ──────────────────────────────────────────────────────────────────────────────

class TestGetSemanticScore:
    def test_empty_string_returns_zero(self):
        assert get_semantic_score("") == 0.0

    def test_whitespace_only_returns_zero(self):
        # Falsy strings should all short-circuit to 0.0.
        assert get_semantic_score("   ") == 0.0

    def test_negative_text_scores_below_threshold(self):
        score = get_semantic_score(
            "I am completely overwhelmed, exhausted, and cannot cope anymore."
        )
        assert score < -0.05, f"Expected negative score, got {score}"

    def test_positive_text_scores_above_threshold(self):
        score = get_semantic_score(
            "Everything is going great, I feel calm, happy, and energised today."
        )
        assert score > 0.05, f"Expected positive score, got {score}"

    def test_neutral_text_near_zero(self):
        score = get_semantic_score("The meeting starts at three o'clock.")
        assert -0.5 < score < 0.5, f"Neutral text gave extreme score {score}"

    def test_return_type_is_float(self):
        assert isinstance(get_semantic_score("hello"), float)

    def test_score_within_valid_range(self):
        for text in [
            "I hate everything!",
            "I love this so much!!!",
            "The sky is blue.",
            "",
        ]:
            s = get_semantic_score(text)
            assert -1.0 <= s <= 1.0, f"Score {s} out of [-1, 1] for {text!r}"

    def test_deterministic(self):
        text = "Feeling stressed and under pressure."
        assert get_semantic_score(text) == get_semantic_score(text)

    def test_logs_debug_on_empty(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="semantic_analysis"):
            get_semantic_score("")
        assert any("empty" in r.message.lower() for r in caplog.records)

    def test_logs_vader_scores_on_nonempty(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="semantic_analysis"):
            get_semantic_score("I feel great today.")
        assert any("vader" in r.message.lower() for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────────────
# transcribe_audio — error paths (no API call made)
# ──────────────────────────────────────────────────────────────────────────────

class TestTranscribeAudioErrors:
    def test_missing_api_key_returns_empty(self, caplog):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the env var is absent even if a .env file exists.
            os.environ.pop("DEEPGRAM_API_KEY", None)
            with caplog.at_level(logging.ERROR, logger="semantic_analysis"):
                result = transcribe_audio("any_file.wav")
        assert result == ""
        assert any("DEEPGRAM_API_KEY" in r.message for r in caplog.records)

    def test_file_not_found_returns_empty(self, caplog):
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake-key"}):
            with caplog.at_level(logging.ERROR, logger="semantic_analysis"):
                result = transcribe_audio("/nonexistent/path/file.wav")
        assert result == ""
        assert any("not found" in r.message.lower() for r in caplog.records)

    def test_os_error_returns_empty(self, caplog):
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake-key"}):
            with patch("builtins.open", side_effect=OSError("permission denied")):
                with caplog.at_level(logging.ERROR, logger="semantic_analysis"):
                    result = transcribe_audio("some_file.wav")
        assert result == ""
        assert any("could not read" in r.message.lower() for r in caplog.records)

    def test_empty_response_channels_returns_empty(self, caplog):
        """Deepgram returns a response but channels list is empty (silent audio)."""
        bad_response = MagicMock()
        bad_response.results.channels = []   # IndexError when we access [0]

        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake-key"}):
            with patch("builtins.open", mock_open(read_data=b"fake-audio")):
                with patch("semantic_analysis.DeepgramClient") as MockClient:
                    MockClient.return_value.listen.v1.media.transcribe_file.return_value = bad_response
                    with caplog.at_level(logging.ERROR, logger="semantic_analysis"):
                        result = transcribe_audio("audio.wav")
        assert result == ""
        assert any("no transcription" in r.message.lower() or "channel" in r.message.lower()
                   for r in caplog.records)

    def test_api_exception_returns_empty(self, caplog):
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake-key"}):
            with patch("builtins.open", mock_open(read_data=b"fake-audio")):
                with patch("semantic_analysis.DeepgramClient") as MockClient:
                    MockClient.return_value.listen.v1.media.transcribe_file.side_effect = (
                        RuntimeError("network timeout")
                    )
                    with caplog.at_level(logging.ERROR, logger="semantic_analysis"):
                        result = transcribe_audio("audio.wav")
        assert result == ""
        assert any("transcription failed" in r.message.lower() for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────────────
# transcribe_audio — happy path (mocked Deepgram)
# ──────────────────────────────────────────────────────────────────────────────

class TestTranscribeAudioSuccess:
    def test_returns_transcript_string(self):
        expected = "Hello, this is a test transcript."
        response = _make_deepgram_response(expected)

        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake-key"}):
            with patch("builtins.open", mock_open(read_data=b"fake-wav-bytes")):
                with patch("semantic_analysis.DeepgramClient") as MockClient:
                    MockClient.return_value.listen.v1.media.transcribe_file.return_value = response
                    result = transcribe_audio("test.wav")

        assert result == expected

    def test_calls_nova2_smart_format(self):
        """Deepgram must be called with model='nova-2' and smart_format=True."""
        response = _make_deepgram_response("ok")

        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake-key"}):
            with patch("builtins.open", mock_open(read_data=b"bytes")):
                with patch("semantic_analysis.DeepgramClient") as MockClient:
                    mock_transcribe = MockClient.return_value.listen.v1.media.transcribe_file
                    mock_transcribe.return_value = response
                    transcribe_audio("test.wav")
                    _, kwargs = mock_transcribe.call_args
                    assert kwargs.get("model") == "nova-2"
                    assert kwargs.get("smart_format") is True

    def test_audio_bytes_passed_as_request(self):
        """The raw file bytes must be forwarded as the `request` kwarg."""
        audio_content = b"\x52\x49\x46\x46fake-wav"
        response = _make_deepgram_response("test")

        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake-key"}):
            with patch("builtins.open", mock_open(read_data=audio_content)):
                with patch("semantic_analysis.DeepgramClient") as MockClient:
                    mock_transcribe = MockClient.return_value.listen.v1.media.transcribe_file
                    mock_transcribe.return_value = response
                    transcribe_audio("test.wav")
                    _, kwargs = mock_transcribe.call_args
                    assert kwargs.get("request") == audio_content

    def test_empty_transcript_from_api_returns_empty_string(self):
        """If Deepgram returns an empty transcript we pass it through as-is."""
        response = _make_deepgram_response("")

        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake-key"}):
            with patch("builtins.open", mock_open(read_data=b"bytes")):
                with patch("semantic_analysis.DeepgramClient") as MockClient:
                    MockClient.return_value.listen.v1.media.transcribe_file.return_value = response
                    result = transcribe_audio("silent.wav")
        assert result == ""

    def test_logs_debug_on_success(self, caplog):
        response = _make_deepgram_response("some words here")

        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake-key"}):
            with patch("builtins.open", mock_open(read_data=b"bytes")):
                with patch("semantic_analysis.DeepgramClient") as MockClient:
                    MockClient.return_value.listen.v1.media.transcribe_file.return_value = response
                    with caplog.at_level(logging.DEBUG, logger="semantic_analysis"):
                        transcribe_audio("test.wav")
        assert any("succeeded" in r.message.lower() for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────────────
# Integration: transcribe_audio → get_semantic_score pipeline
# ──────────────────────────────────────────────────────────────────────────────

class TestPipeline:
    def test_pipeline_stressed_text(self):
        transcript = "I can't handle this anymore. Everything is falling apart."
        response = _make_deepgram_response(transcript)

        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake-key"}):
            with patch("builtins.open", mock_open(read_data=b"bytes")):
                with patch("semantic_analysis.DeepgramClient") as MockClient:
                    MockClient.return_value.listen.v1.media.transcribe_file.return_value = response
                    result = transcribe_audio("stressed.wav")

        score = get_semantic_score(result)
        assert score < -0.05, f"Stressed transcript scored {score}, expected negative"

    def test_pipeline_calm_text(self):
        transcript = "I feel wonderful and well-rested. Today is a great day."
        response = _make_deepgram_response(transcript)

        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake-key"}):
            with patch("builtins.open", mock_open(read_data=b"bytes")):
                with patch("semantic_analysis.DeepgramClient") as MockClient:
                    MockClient.return_value.listen.v1.media.transcribe_file.return_value = response
                    result = transcribe_audio("calm.wav")

        score = get_semantic_score(result)
        assert score > 0.05, f"Calm transcript scored {score}, expected positive"

    def test_pipeline_on_api_failure_score_is_neutral(self):
        """If transcription fails, the pipeline must return 0.0 (not crash)."""
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake-key"}):
            with patch("builtins.open", mock_open(read_data=b"bytes")):
                with patch("semantic_analysis.DeepgramClient") as MockClient:
                    MockClient.return_value.listen.v1.media.transcribe_file.side_effect = (
                        ConnectionError("unreachable")
                    )
                    result = transcribe_audio("test.wav")

        score = get_semantic_score(result)
        assert score == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Real audio tests — require DEEPGRAM_API_KEY in environment / .env
# Skipped automatically when the key is absent so CI never breaks.
# Run locally:  pytest tests/test_semantic_analysis.py -v -k real_audio
# ──────────────────────────────────────────────────────────────────────────────

TESTS_DIR = os.path.dirname(__file__)

# Map each .wav file to its expected sentiment direction.
# (filename, expected_label, xfail_reason_or_None)
# xfail_reason is set for clips where VADER is known to struggle:
#   stress expressed as urgency/panic/exhaustion rather than overtly negative words.
#   These clips require tone/prosody analysis (Part 2) to score correctly.
REAL_AUDIO_CASES = [
    ("Chill.wav",      "calm",     None),
    ("Chill_1.wav",    "calm",     None),
    ("Chill_2.wav",    "calm",     None),
    ("chill_3.wav",    "calm",     None),
    ("Stressed.wav",   "stressed",
        "Stress expressed through urgency/panic ('What am I supposed to do? Someone help') "
        "— no overtly negative words; VADER reads this as positive. Needs tone analysis."),
    ("Stressed_1.wav", "stressed",
        "Stress expressed as exhausted relief ('need to go home, sleep') "
        "— VADER reads escape-language as positive. Needs tone analysis."),
    ("Stress_2.wav",   "stressed", None),
    ("stressed_3.wav", "stressed", None),
]

needs_api_key = pytest.mark.skipif(
    not os.getenv("DEEPGRAM_API_KEY"),
    reason="DEEPGRAM_API_KEY not set — skipping real audio tests",
)


@needs_api_key
class TestRealAudio:
    """End-to-end tests: real .wav files → Deepgram transcription → VADER score.

    Each test hits the live Deepgram API, so it requires a valid key in .env
    and a network connection.  Results are printed so you can read the actual
    transcripts and scores during development.
    """

    @pytest.mark.parametrize("filename,expected_label,xfail_reason", REAL_AUDIO_CASES)
    def test_real_audio_file(self, filename, expected_label, xfail_reason):
        if xfail_reason:
            pytest.xfail(xfail_reason)
        wav_path = os.path.join(TESTS_DIR, filename)

        if not os.path.exists(wav_path):
            pytest.skip(f"{filename} not found in tests/ — convert the m4a first")

        transcript = transcribe_audio(wav_path)
        score = get_semantic_score(transcript)

        print(f"\n  [{expected_label.upper():>8}]  {filename}")
        print(f"   transcript : {transcript!r}")
        print(f"   score      : {score:+.4f}")

        assert isinstance(transcript, str), "transcribe_audio must return a string"

        # A real audio file must produce a non-empty transcript.
        # If it's empty the API call failed (bad key, network, bad file).
        assert transcript != "", (
            f"{filename}: transcription returned an empty string — "
            "check that DEEPGRAM_API_KEY is valid and the file is a good .wav."
        )

        assert -1.0 <= score <= 1.0, "score must be in [-1, 1]"

        # Direction check: calm → positive score, stressed → negative score.
        # We allow a ±0.05 dead-band so genuinely ambiguous clips don't hard-fail.
        if expected_label == "calm":
            assert score >= -0.05, (
                f"{filename}: expected calm (score ≥ -0.05) but got {score:+.4f}.\n"
                f"Transcript: {transcript!r}"
            )
        elif expected_label == "stressed":
            assert score <= 0.05, (
                f"{filename}: expected stressed (score ≤ 0.05) but got {score:+.4f}.\n"
                f"Transcript: {transcript!r}"
            )
