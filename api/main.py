from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import asyncio
import wave
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from collections import deque
from urllib.parse import quote_plus
import os
import random
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_FILE = os.path.join(BASE_DIR, "frontend", "dashboard.html")

# Load environment variables from .env (DEEPGRAM_API_KEY, etc.)
from dotenv import load_dotenv
load_dotenv()

# ========== VOICE / DEEPGRAM SETUP ==========
# The dashboard records the mic (WebM/Opus), uploads it here, we convert it to
# WAV with pydub (-> ffmpeg) and send it to Deepgram for transcription.
from deepgram import DeepgramClient, AsyncDeepgramClient
from pydub import AudioSegment

# Where converted WAV files land. Created on startup; git-ignored.
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Deepgram client — None if the key is missing, so the app still boots and the
# /transcribe endpoint can return a clean "add your key" error instead of crashing.
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if DEEPGRAM_API_KEY:
    deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
    async_deepgram = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)  # for the live /ws/transcribe stream
    print("[OK] Deepgram client initialized")
else:
    deepgram = None
    async_deepgram = None
    print("[WARN] DEEPGRAM_API_KEY not set — /transcribe will return an error until you add it to .env")

# ========== VOICE → MODEL (semantic scoring) ==========
# Score the voice check-in with the model team's semantic analysis (VADER
# sentiment) so the risk reflects WHAT the worker said. Tone analysis (Wav2Vec2)
# is intentionally skipped — it needs torch+transformers, which don't fit on this
# disk. We import semantic_analysis.py unchanged (friend's file) and only borrow
# get_semantic_score. Guarded so the app still boots if the module is unavailable.
import sys
sys.path.insert(0, os.path.join(BASE_DIR, "model", "src"))
try:
    from semantic_analysis import get_semantic_score
    print("[OK] semantic_analysis (VADER sentiment) loaded")
except Exception as e:
    get_semantic_score = None
    print(f"[WARN] semantic_analysis unavailable ({e}); voice will not move the risk")

# ========== INTERVENTION AUDIO ==========
# Pre-generated spoken stress-relief clips (model/src/tts.py). We reuse tts.py
# (lightweight — no torch/soundfile) for the clip paths + message text rather than
# importing final_scoring, which fails on this env (missing soundfile via tone_analysis).
try:
    from tts import intervention_audio_path, INTERVENTION_TEXTS
    print("[OK] tts intervention audio loaded")
except Exception as e:
    intervention_audio_path, INTERVENTION_TEXTS = None, {}
    print(f"[WARN] tts unavailable ({e}); intervention audio disabled")

# Risk blend weights — mirror final_scoring.py / INTEGRATION.md (0.7 sensor, 0.3 voice).
SENSOR_WEIGHT = 0.7
VOICE_WEIGHT = 0.3

import re

# Explicit distress phrases that HARD-TRIGGER max stress regardless of overall
# sentiment — a worker saying these is asking for help, so we don't let VADER
# average them away behind neutral/positive chatter.
DISTRESS_KEYWORDS = (
    "stressed", "stress", "overwhelmed", "can't cope", "cant cope",
    "burnt out", "burned out", "burnout", "exhausted", "i need help",
    "breaking down", "at my limit", "can't do this", "cant do this",
    "panic", "panicking", "anxious", "anxiety",
)


def score_voice_stress(transcript: str) -> Optional[float]:
    """Turn a transcript into a 0-1 voice stress score (higher = more stressed).

    1. If it contains an explicit distress phrase -> hard-trigger 1.0, which engages
       the blend_risk voice override (-> HIGH). The worker's words can't be averaged
       away by surrounding chatter.
    2. Otherwise score each SENTENCE with VADER and take the most-stressed one, so
       one stressed sentence isn't diluted by neutral/positive talk around it.

    Returns None if semantic scoring is unavailable or there's no text.
    """
    if get_semantic_score is None or not transcript or not transcript.strip():
        return None

    low = transcript.lower()
    hit = next((kw for kw in DISTRESS_KEYWORDS if kw in low), None)
    if hit:
        print(f"[VOICE→MODEL] distress keyword '{hit}' -> voice_stress=1.0 (HIGH)")
        return 1.0

    # Per-sentence max — the worst moment drives the score.
    sentences = [s for s in re.split(r"[.!?]+", transcript) if s.strip()] or [transcript]
    stresses = []
    for s in sentences:
        try:
            stresses.append((1.0 - get_semantic_score(s)) / 2.0)
        except Exception:
            pass
    return max(stresses) if stresses else None

# Voice override: a very stressed check-in floors the overall risk into the HIGH
# band even when the body sensors read calm — so an explicit "I'm at my max stress
# level" can trip the alert on words alone. Trade-off: voice can trigger HIGH by
# itself. Tune VOICE_OVERRIDE_THRESHOLD up to make it harder to set off.
VOICE_OVERRIDE_THRESHOLD = 0.8   # voice_stress at/above this engages the override
VOICE_OVERRIDE_FLOOR     = 0.85  # risk floored here (>= 0.65 HIGH cutoff = clearly HIGH)


def blend_risk(sensor: float, voice: float) -> float:
    """Blend the sensor and voice signals into one 0-1 risk score.

    Normally 0.7*sensor + 0.3*voice. But if the voice check-in is very stressed
    (voice >= VOICE_OVERRIDE_THRESHOLD), floor the result into the HIGH band so the
    worker's own words can raise the alert regardless of what the vitals say."""
    risk = SENSOR_WEIGHT * sensor + VOICE_WEIGHT * voice
    if voice >= VOICE_OVERRIDE_THRESHOLD:
        risk = max(risk, VOICE_OVERRIDE_FLOOR)
    return risk


# Intervention tiers — mirror final_scoring._INTERVENTION_MESSAGES (most → least severe).
# A risk in [0.60, 0.80) plays the "elevated" clip; >= 0.80 plays the "high" clip.
INTERVENTION_TIERS = [(0.80, "high"), (0.60, "elevated")]


class Intervention(BaseModel):
    level: str                      # "none" | "elevated" | "high"
    text: Optional[str] = None      # spoken message (also shown on screen)
    audio_url: Optional[str] = None # GET endpoint for the .wav, or None


def intervention_for(risk: float) -> Intervention:
    """Map a 0-1 risk score to the intervention tier whose clip should play."""
    for threshold, key in INTERVENTION_TIERS:
        if risk >= threshold:
            return Intervention(
                level=key,
                text=INTERVENTION_TEXTS.get(key),
                audio_url=f"/intervention-audio/{key}",
            )
    return Intervention(level="none")


app = FastAPI(title="ShiftGuard API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SLIDING WINDOW ────────────────────────────────────────────────────────────
WINDOW_SIZE = 20
risk_window = deque(maxlen=WINDOW_SIZE)

# ── PYDANTIC MODELS ───────────────────────────────────────────────────────────

class Contributors(BaseModel):
    heart_rate: int       # bpm (ESP32 sensor)
    temperature: float    # °C (ESP32 sensor)
    voice_fatigue: int    # 0-100, from the latest voice recording's acoustic fatigue


class Song(BaseModel):
    name: str
    artist: str
    url: str


class DashboardResponse(BaseModel):
    risk: float
    risk_label: str
    heart_rate: int
    temperature: float
    contributors: Contributors
    recommendation: str
    song: Song
    intervention: Intervention


class TranscriptResponse(BaseModel):
    transcript: str
    speech_rate: int
    acoustic_fatigue: float
    timestamp: str


class HistoryPoint(BaseModel):
    timestamp: str
    risk: float
    heart_rate: int
    temperature: float


class HistoryResponse(BaseModel):
    history: List[HistoryPoint]
    window_size: int


class ArduinoReading(BaseModel):
    """Live data posted by bridge.py"""
    bpm: float
    temp_c: float
    risk_score: float
    timestamp: str


latest_arduino_reading: Optional[ArduinoReading] = None

# Store the most recent real transcription from /transcribe.
# None until the first recording — /latest-transcript falls back to mock until then.
latest_transcript: Optional[TranscriptResponse] = None

# Latest voice stress from the last check-in, 0.0-1.0 (higher = more stressed).
# None until the first recording; the risk blend uses 0.5 (neutral) until then.
latest_voice_stress: Optional[float] = None

# ── HELPERS ───────────────────────────────────────────────────────────────────

def risk_label(score: float) -> str:
    if score >= 0.65:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"


def voice_fatigue_score() -> int:
    """Latest voice stress (0.0-1.0 semantic sentiment) scaled to 0-100. This is
    the same signal that moves the risk. Returns 0 until the first check-in."""
    if latest_voice_stress is not None:
        return round(latest_voice_stress * 100)
    return 0


def build_contributors(heart_rate: int, temperature: float) -> Contributors:
    """The three raw signals shown on the Contributors card: live heart rate,
    temperature, and the latest voice fatigue score."""
    return Contributors(
        heart_rate=heart_rate,
        temperature=temperature,
        voice_fatigue=voice_fatigue_score(),
    )


CALM_SONGS = [
    ("Weightless", "Marconi Union"),
    ("Saturn", "Sleeping at Last"),
    ("Holocene", "Bon Iver"),
    ("An Ending (Ascent)", "Brian Eno"),
]
FOCUS_SONGS = [
    ("Here Comes the Sun", "The Beatles"),
    ("Good as Hell", "Lizzo"),
    ("Sunflower", "Post Malone"),
    ("Walking on Sunshine", "Katrina & The Waves"),
]


def pick_song(songs: List) -> Song:
    name, artist = random.choice(songs)
    query = quote_plus(f"{name} {artist}")
    return Song(name=name, artist=artist, url=f"https://open.spotify.com/search/{query}")


LOOKBACK = 5
ALERT_THRESHOLD = 0.6


def decide_recommendation(window: deque):
    if not window:
        return "Waiting for sensor data...", pick_song(CALM_SONGS)

    recent_scores = [r["risk"] for r in list(window)[-LOOKBACK:]]
    avg_recent = sum(recent_scores) / len(recent_scores)

    if avg_recent >= ALERT_THRESHOLD:
        return (
            "Sustained fatigue detected. Take a 10-minute break, hydrate, "
            "and check in with your supervisor.",
            pick_song(CALM_SONGS),
        )
    if window[-1]["risk"] >= 0.5:
        return (
            "Brief fatigue spike. Ease off for a moment and reset.",
            pick_song(CALM_SONGS),
        )
    return (
        "You're clear. Keep the rhythm going and stay hydrated.",
        pick_song(FOCUS_SONGS),
    )


def record_reading(risk: float, heart_rate: int, temperature: float) -> None:
    risk_window.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "risk": risk,
        "heart_rate": heart_rate,
        "temperature": temperature
    })


def compute_acoustic_fatigue(words: List[dict], duration_sec: float, speech_rate: int) -> float:
    """
    Turn Deepgram's word-level output into a 0.0–1.0 acoustic fatigue score
    (higher = more fatigued-sounding voice). Shown on the dashboard's Fatigue metric.

    You are given:
      words        - list of {"word": str, "start": float, "end": float, "confidence": float},
                     one entry per spoken word, timings in seconds.
      duration_sec - active speaking span in seconds (first word start -> last word
                     end; leading/trailing silence already removed).
      speech_rate  - words per minute (already computed for you).

    Signals combined (tweak thresholds/weights to taste):
      - Slow speech: low WPM reads as tired. Fresh ~130-160 WPM; under ~90 looks fatigued.
      - Long pauses: large gaps between words[i]["end"] and words[i+1]["start"].
      - Low confidence: slurred/mumbled words lower Deepgram's per-word confidence.

    Returns a float clamped to [0.0, 1.0], rounded to 2 decimals.
    """
    # Nothing to score on an empty/silent clip.
    if not words or duration_sec <= 0:
        return 0.0

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    # 1. Slow speech: 150 WPM (fresh) -> 0.0, 90 WPM or slower (tired) -> 1.0.
    slow_score = clamp01((150 - speech_rate) / (150 - 90))

    # 2. Pause ratio: share of the clip spent silent between words.
    #    ~0% silence -> 0.0, >=40% silence -> 1.0. (needs >=2 words for a gap)
    total_gap = sum(
        max(0.0, words[i + 1]["start"] - words[i]["end"])
        for i in range(len(words) - 1)
    )
    pause_score = clamp01((total_gap / duration_sec) / 0.40)

    # 3. Mumbling: avg per-word confidence. 0.95 (crisp) -> 0.0, 0.60 (slurred) -> 1.0.
    avg_conf = sum(w["confidence"] for w in words) / len(words)
    mumble_score = clamp01((0.95 - avg_conf) / (0.95 - 0.60))

    # Weighted blend — WPM is the most reliable signal, confidence the noisiest.
    fatigue = 0.5 * slow_score + 0.3 * pause_score + 0.2 * mumble_score
    return round(clamp01(fatigue), 2)


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    return FileResponse(DASHBOARD_FILE)


@app.post("/")
async def receive_arduino_data(reading: ArduinoReading):
    """Receive live sensor data from bridge.py"""
    global latest_arduino_reading
    latest_arduino_reading = reading
    record_reading(reading.risk_score, int(reading.bpm), reading.temp_c)
    print(f"[ARDUINO] BPM={reading.bpm}, Temp={reading.temp_c}°C, Risk={reading.risk_score:.2f}")
    return {"status": "ok", "received_at": datetime.now().isoformat()}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "shiftguard-api"
    }


@app.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard():
    # Sensor signal: bridge's risk_score if a live reading exists, else neutral 0.5.
    if latest_arduino_reading is not None:
        sensor      = latest_arduino_reading.risk_score
        heart_rate  = int(latest_arduino_reading.bpm)
        temperature = round(latest_arduino_reading.temp_c, 1)
    else:
        sensor, heart_rate, temperature = 0.5, 0, 0.0

    # Voice signal from the last check-in (neutral 0.5 until the user records).
    voice = latest_voice_stress if latest_voice_stress is not None else 0.5

    # Blend: 0.7 sensor + 0.3 voice, with the high-voice override (see blend_risk).
    # The voice check-in now visibly moves the risk; the 2s poll reflects it within ≤2s.
    risk = blend_risk(sensor, voice)

    recommendation, song = decide_recommendation(risk_window)
    print(f"[DASHBOARD] sensor={sensor:.2f} voice={voice:.2f} -> risk={risk:.2f} ({risk_label(risk)})")

    return DashboardResponse(
        risk=round(risk, 4),
        risk_label=risk_label(risk),
        heart_rate=heart_rate,
        temperature=temperature,
        contributors=build_contributors(heart_rate, temperature),
        recommendation=recommendation,
        song=song,
        intervention=intervention_for(risk),
    )


@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(file: UploadFile = File(...)):
    """
    Receive a mic recording (WebM/Opus) from the dashboard, convert it to WAV,
    transcribe it with Deepgram, and return the transcript + voice metrics.

    The result is also cached so GET /latest-transcript serves the real one.
    """
    global latest_transcript, latest_voice_stress

    if deepgram is None:
        raise HTTPException(
            status_code=503,
            detail="Deepgram not configured. Add DEEPGRAM_API_KEY to .env and restart the server."
        )

    webm_bytes = await file.read()
    if not webm_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    # Save the incoming WebM to a temp file so ffmpeg/pydub can read it.
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(webm_bytes)
        webm_path = tmp.name

    try:
        # 1. Convert WebM/Opus -> WAV (pydub shells out to ffmpeg). Keep the WAV
        #    as the artifact.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(RECORDINGS_DIR, f"voice_{timestamp}.wav")
        AudioSegment.from_file(webm_path).export(wav_path, format="wav")

        # 2. Send the WAV bytes to Deepgram.
        with open(wav_path, "rb") as f:
            audio_data = f.read()

        response = deepgram.listen.v1.media.transcribe_file(
            request=audio_data,
            model="nova-3",
            smart_format=True,
            punctuate=True,
            filler_words=True,   # surfaces "um"/"uh" — a fatigue signal
        )

        # 3. Extract the transcript + word-level timings.
        alt = response.results.channels[0].alternatives[0]
        transcript = alt.transcript or ""
        words = [
            {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
            for w in (alt.words or [])
        ]

        # Measure over the SPEAKING span (first word start -> last word end), not the
        # whole clip. Otherwise dead air before/after you talk (e.g. reaching for the
        # Stop button) drags WPM down and inflates fatigue. Fall back to clip duration
        # if there's 0-1 words.
        clip_duration = float(response.metadata.duration or 0.0)
        speaking_span = (words[-1]["end"] - words[0]["start"]) if len(words) >= 2 else clip_duration

        # 4. Derive metrics.
        speech_rate = round(len(words) / (speaking_span / 60)) if speaking_span > 0 else 0
        acoustic_fatigue = compute_acoustic_fatigue(words, speaking_span, speech_rate)

        result = TranscriptResponse(
            transcript=transcript or "(no speech detected)",
            speech_rate=speech_rate,
            acoustic_fatigue=acoustic_fatigue,
            timestamp=datetime.now().isoformat(),
        )
        latest_transcript = result

        # Score the check-in so it moves the risk: keyword hard-trigger + per-sentence
        # max (so an explicit "I'm so stressed" isn't averaged away by chatter).
        vs = score_voice_stress(transcript)
        if vs is not None:
            latest_voice_stress = vs
            print(f"[VOICE→MODEL] voice_stress={vs:.2f}")

        print(f"[DEEPGRAM] '{transcript}' | {len(words)} words, "
              f"clip={clip_duration:.1f}s speaking={speaking_span:.1f}s, "
              f"{speech_rate} WPM, fatigue={acoustic_fatigue}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        # Drop the temp WebM (the WAV in recordings/ is the kept artifact).
        if os.path.exists(webm_path):
            os.remove(webm_path)


@app.get("/history", response_model=HistoryResponse)
async def get_history():
    history = [
        HistoryPoint(
            timestamp=r["timestamp"],
            risk=r["risk"],
            heart_rate=r["heart_rate"],
            temperature=r["temperature"]
        )
        for r in risk_window
    ]
    return HistoryResponse(history=history, window_size=WINDOW_SIZE)


@app.get("/latest-transcript", response_model=TranscriptResponse)
async def get_latest_transcript():
    """Return the latest real transcription from /transcribe if we have one.

    Before the first recording there is no real data, so return a STABLE empty
    sentinel (transcript="") rather than random mock text. The dashboard polls
    this every 2s; returning random data here is what made the card appear to
    "update" on its own. An empty transcript tells the frontend to show its
    idle/waiting state instead."""
    if latest_transcript is not None:
        return latest_transcript

    return TranscriptResponse(
        transcript="",
        speech_rate=0,
        acoustic_fatigue=0.0,
        timestamp=datetime.now().isoformat(),
    )


# ── LIVE STREAMING VOICE (Step 2) ─────────────────────────────────────────────

def _pcm_to_wav(pcm: bytes, path: str, sample_rate: int = 16000) -> None:
    """Wrap raw 16-bit mono PCM as a .wav artifact in recordings/."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)            # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _blend_risk() -> float:
    """Current blended risk (same formula as /dashboard) for the live WS push."""
    sensor = latest_arduino_reading.risk_score if latest_arduino_reading is not None else 0.5
    voice = latest_voice_stress if latest_voice_stress is not None else 0.5
    return round(blend_risk(sensor, voice), 4)


@app.get("/pcm-worklet.js")
async def serve_worklet():
    """Serve the AudioWorklet that turns mic audio into 16 kHz PCM frames."""
    return FileResponse(
        os.path.join(BASE_DIR, "frontend", "pcm-worklet.js"),
        media_type="application/javascript",
    )


@app.get("/intervention-audio/{level}")
async def intervention_audio(level: str):
    """Serve the pre-generated stress-relief clip for an intervention tier.

    The dashboard plays this when the risk crosses into the elevated/high band.
    """
    if intervention_audio_path is None or level not in ("high", "elevated"):
        raise HTTPException(status_code=404, detail="unknown intervention level")
    path = intervention_audio_path(level)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="intervention audio not generated; run: python model/src/tts.py",
        )
    return FileResponse(str(path), media_type="audio/wav")


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    """
    Live streaming voice. The browser streams 16 kHz linear16 PCM frames in; we
    RELAY them to Deepgram's live API (Deepgram does the STT) and stream the
    interim/final transcripts back for live captions. On stop we score the full
    transcript (semantic) → move the risk, and push the new risk back instantly.
    """
    global latest_transcript, latest_voice_stress
    await ws.accept()

    if async_deepgram is None:
        await ws.send_json({"type": "error", "message": "Deepgram not configured (set DEEPGRAM_API_KEY)."})
        await ws.close()
        return

    pcm_buffer = bytearray()
    final_parts: List[str] = []

    try:
        async with async_deepgram.listen.v1.connect(
            model="nova-3",
            encoding="linear16",
            sample_rate=16000,
            interim_results=True,
            punctuate=True,
            smart_format=True,
            utterance_end_ms="1000",
            vad_events=True,
        ) as dg:

            async def pump_audio():
                """Browser → Deepgram. Stops on a 'stop' text message or disconnect."""
                try:
                    while True:
                        msg = await ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            break
                        if msg.get("bytes") is not None:
                            pcm_buffer.extend(msg["bytes"])
                            await dg.send_media(msg["bytes"])
                        elif msg.get("text") == "stop":
                            break
                except WebSocketDisconnect:
                    pass
                finally:
                    try:
                        await dg.send_close_stream()   # tell Deepgram we're done → it finalizes
                    except Exception:
                        pass

            async def pump_transcripts():
                """Deepgram → browser: live captions + LIVE risk on each final phrase."""
                global latest_voice_stress
                async for event in dg:
                    if getattr(event, "type", None) != "Results":
                        continue
                    channel = getattr(event, "channel", None)
                    alts = getattr(channel, "alternatives", None) if channel else None
                    text = (alts[0].transcript if alts else "") or ""
                    if not text:
                        continue
                    is_final = bool(getattr(event, "is_final", False))
                    try:
                        await ws.send_json({"type": "transcript", "text": text, "is_final": is_final})
                    except Exception:
                        break
                    # On a finalized phrase, re-score everything said so far and push
                    # the updated risk LIVE — no need to wait for the user to stop.
                    if is_final:
                        final_parts.append(text)
                        vs = score_voice_stress(" ".join(final_parts))
                        if vs is not None:
                            latest_voice_stress = vs
                            risk = _blend_risk()
                            try:
                                await ws.send_json({"type": "risk", "risk": risk, "risk_label": risk_label(risk)})
                            except Exception:
                                break

            await asyncio.gather(pump_audio(), pump_transcripts())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] /ws/transcribe error: {type(e).__name__}: {e}")
    finally:
        full = " ".join(final_parts).strip()
        if full:
            # Save the captured audio as a WAV artifact.
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                _pcm_to_wav(bytes(pcm_buffer), os.path.join(RECORDINGS_DIR, f"stream_{ts}.wav"))
            except Exception as e:
                print(f"[WS] WAV save failed: {e}")
            # Score → move the risk (keyword hard-trigger + per-sentence max).
            vs = score_voice_stress(full)
            if vs is not None:
                latest_voice_stress = vs
                print(f"[WS VOICE→MODEL] '{full[:60]}' voice_stress={vs:.2f}")
            latest_transcript = TranscriptResponse(
                transcript=full, speech_rate=0, acoustic_fatigue=0.0,
                timestamp=datetime.now().isoformat(),
            )
            # Push the freshly blended risk back instantly (no 2s poll wait).
            try:
                risk = _blend_risk()
                await ws.send_json({"type": "risk", "risk": risk, "risk_label": risk_label(risk), "transcript": full})
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
