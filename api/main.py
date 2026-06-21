from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
from deepgram import DeepgramClient
from pydub import AudioSegment

# Where converted WAV files land. Created on startup; git-ignored.
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Deepgram client — None if the key is missing, so the app still boots and the
# /transcribe endpoint can return a clean "add your key" error instead of crashing.
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if DEEPGRAM_API_KEY:
    deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
    print("[OK] Deepgram client initialized")
else:
    deepgram = None
    print("[WARN] DEEPGRAM_API_KEY not set — /transcribe will return an error until you add it to .env")

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

# ── HELPERS ───────────────────────────────────────────────────────────────────

def risk_label(score: float) -> str:
    if score >= 0.65:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"


def voice_fatigue_score() -> int:
    """Latest acoustic fatigue (0.0-1.0 from compute_acoustic_fatigue) scaled to
    0-100. Returns 0 until the first voice recording comes through /transcribe."""
    if latest_transcript is not None:
        return round(latest_transcript.acoustic_fatigue * 100)
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
    # No live sensor data yet — still surface the latest voice fatigue if recorded.
    if latest_arduino_reading is None:
        return DashboardResponse(
            risk=0.0,
            risk_label="LOW",
            heart_rate=0,
            temperature=0.0,
            contributors=build_contributors(0, 0.0),
            recommendation="Waiting for sensor data...",
            song=pick_song(CALM_SONGS)
        )

    risk        = latest_arduino_reading.risk_score
    heart_rate  = int(latest_arduino_reading.bpm)
    temperature = round(latest_arduino_reading.temp_c, 1)
    recommendation, song = decide_recommendation(risk_window)

    print(f"[DASHBOARD] Risk={risk:.2f} ({risk_label(risk)}), BPM={heart_rate}, Temp={temperature}")

    return DashboardResponse(
        risk=round(risk, 4),
        risk_label=risk_label(risk),
        heart_rate=heart_rate,
        temperature=temperature,
        contributors=build_contributors(heart_rate, temperature),
        recommendation=recommendation,
        song=song
    )


@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(file: UploadFile = File(...)):
    """
    Receive a mic recording (WebM/Opus) from the dashboard, convert it to WAV,
    transcribe it with Deepgram, and return the transcript + voice metrics.

    The result is also cached so GET /latest-transcript serves the real one.
    """
    global latest_transcript

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
    """Return the latest real transcription from /transcribe if we have one,
    otherwise fall back to mock data so the dashboard still shows something
    before the first recording."""
    if latest_transcript is not None:
        return latest_transcript

    transcripts = [
        "I'm doing okay, just pushing through.",
        "Yeah, I'm fine. Just a bit tired.",
        "Everything's under control. No issues here.",
        "I could use a break soon, but I'm managing.",
        "Feeling pretty good actually. Ready to keep going.",
        "It's been a long shift, but I'm hanging in there."
    ]
    return TranscriptResponse(
        transcript=random.choice(transcripts),
        speech_rate=random.randint(80, 120),
        acoustic_fatigue=round(random.uniform(0.1, 0.9), 2),
        timestamp=datetime.now().isoformat()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
