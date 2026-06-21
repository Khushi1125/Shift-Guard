from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
from urllib.parse import quote_plus
import os
import sys
import random
import tempfile

# Add project root to Python path so we can import model, pipeline, etc.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# ========== VOICE / DEEPGRAM SETUP ==========
# The dashboard records the mic (WebM/Opus), uploads it here, we convert it to
# WAV with pydub (-> ffmpeg) and send it to Deepgram for transcription.
from deepgram import DeepgramClient
from pydub import AudioSegment

# Where converted WAV files land. Created on startup; git-ignored.
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recordings")
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

# ========== REAL MODEL INTEGRATION ==========
from model.predict import predict_stress
from pipeline.schemas import ModelFeatures
import pandas as pd
import joblib
import numpy as np

# BASE_DIR already set above for sys.path
DASHBOARD_FILE = os.path.join(BASE_DIR, "frontend", "dashboard.html")

# Load pre-computed WESAD features for replay
FEATURES_CSV = os.path.join(BASE_DIR, "model", "outputs", "features_30s.csv")
try:
    features_df = pd.read_csv(FEATURES_CSV)
    print(f"[OK] Loaded {len(features_df)} WESAD windows for replay")
except FileNotFoundError:
    print(f"[WARN] Features CSV not found at {FEATURES_CSV}")
    features_df = None

# Load model feature importances for contributor calculation
MODEL_PATH = os.path.join(BASE_DIR, "model", "outputs", "rf_baseline.joblib")
try:
    model_data = joblib.load(MODEL_PATH)
    feature_importances = dict(zip(model_data['feature_cols'], model_data['model'].feature_importances_))
    print(f"[OK] Loaded model feature importances")
except Exception as e:
    print(f"[WARN] Could not load feature importances: {e}")
    feature_importances = None


class WESADReplay:
    """Cycles through WESAD windows for continuous demo"""
    def __init__(self, subject_id: int = 2):
        if features_df is None:
            raise ValueError("WESAD features not loaded")

        df = features_df[features_df['subject'] == subject_id]
        self.features = df.to_dict('records')
        self.idx = 0
        print(f"[OK] Replay initialized with {len(self.features)} windows from S{subject_id}")

    def get_next(self) -> ModelFeatures:
        """Get next 30s window, cycle at end"""
        if not self.features:
            raise ValueError("No WESAD features loaded")

        row = self.features[self.idx]
        self.idx = (self.idx + 1) % len(self.features)

        # Extract 18 features in correct order
        return ModelFeatures(
            acc_mag_mean=row['acc_mag_mean'],
            acc_mag_std=row['acc_mag_std'],
            acc_hf_mean=row['acc_hf_mean'],
            bvp_mean=row['bvp_mean'],
            bvp_std=row['bvp_std'],
            hr_mean=row['hr_mean'],
            hr_std=row['hr_std'],
            hr_slope=row['hr_slope'],
            hr_min=row['hr_min'],
            hr_max=row['hr_max'],
            temp_mean=row['temp_mean'],
            temp_slope=row['temp_slope'],
            temp_delta=row['temp_delta'],
            eda_mean=row['eda_mean'],
            eda_std=row['eda_std'],
            eda_slope=row['eda_slope'],
            eda_min=row['eda_min'],
            eda_max=row['eda_max'],
        )


# Initialize replay (will use subject 2 by default)
try:
    replay = WESADReplay(subject_id=2)
except Exception as e:
    print(f"[WARN] Could not initialize WESAD replay: {e}")
    replay = None

app = FastAPI(title="Burnout Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Sliding window of recent readings.
# Each reading is binary: 1 = AT RISK, 0 = OK (this is what the data/model
# side sends). The window keeps only the most recent WINDOW_SIZE readings,
# so /history "slides" forward as new readings arrive.
# ---------------------------------------------------------------------------
WINDOW_SIZE = 20
risk_window = deque(maxlen=WINDOW_SIZE)  # holds dicts: {"timestamp": str, "risk": int}


# Pydantic Models
class Contributors(BaseModel):
    heart_rate: int       # bpm (ESP32 sensor)
    temperature: float    # °C (ESP32 sensor)
    voice_fatigue: int    # 0-100, from the latest voice recording's acoustic fatigue


class Song(BaseModel):
    name: str
    artist: str
    url: str             # Spotify search link (never 404s, unlike a hardcoded track id)


class DashboardResponse(BaseModel):
    risk: int            # 0 or 1
    risk_label: str      # "YES" or "NO"
    heart_rate: int      # bpm (ESP32 sensor)
    temperature: float   # °C (ESP32 sensor)
    contributors: Contributors
    recommendation: str
    song: Song


class PredictResponse(BaseModel):
    risk: int            # 0 or 1  <-- the model team returns this
    contributors: Contributors
    timestamp: str


class TranscriptResponse(BaseModel):
    transcript: str
    speech_rate: int
    acoustic_fatigue: float
    timestamp: str


class HistoryPoint(BaseModel):
    timestamp: str
    risk: int            # 0 or 1
    heart_rate: int      # bpm
    temperature: float   # °C


class HistoryResponse(BaseModel):
    history: List[HistoryPoint]
    window_size: int


class ArduinoReading(BaseModel):
    """Live data from Arduino via bridge.py"""
    bpm: float
    temp_c: float
    risk_score: float
    timestamp: str


# Store latest live Arduino reading from bridge.py
latest_arduino_reading: Optional[ArduinoReading] = None

# Store the most recent real transcription from /transcribe.
# None until the first recording — /latest-transcript falls back to mock until then.
latest_transcript: Optional[TranscriptResponse] = None


# Contributor Calculation using Feature Importances
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


# Mock Data Generators
def generate_risk() -> int:
    """Mock binary risk reading (1 = at risk, 0 = ok). Model team replaces this."""
    return random.choice([0, 0, 0, 1, 1])  # weighted toward OK


def generate_vitals(risk: int):
    """Mock wearable vitals. Elevated when at risk so the data tells a coherent story.
    Returns (heart_rate_bpm, temperature_celsius)."""
    if risk == 1:
        return random.randint(95, 120), round(random.uniform(37.2, 38.0), 1)
    return random.randint(60, 85), round(random.uniform(36.4, 37.0), 1)


# Curated songs. CALM = wind down when fatigued; FOCUS = keep the rhythm when clear.
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
    """Turn a (name, artist) pick into a clickable Spotify search link."""
    name, artist = random.choice(songs)
    query = quote_plus(f"{name} {artist}")
    return Song(name=name, artist=artist, url=f"https://open.spotify.com/search/{query}")


# Debounce thresholds: of the last LOOKBACK readings, this many 1s == sustained risk.
LOOKBACK = 5
ALERT_THRESHOLD = 3


def decide_recommendation(window: deque):
    """
    Decide what to tell the worker AND what to play, from the sliding window.

    Debounced: a single stray 1 is noise, so we only raise the burnout alert
    when ALERT_THRESHOLD of the last LOOKBACK readings are 1. The song follows
    the same debounced state (calm music only when risk is actually sustained).

    Returns (recommendation_text, Song).
    """
    if not window:
        return "Waiting for sensor data...", pick_song(CALM_SONGS)

    recent = [r["risk"] for r in window][-LOOKBACK:]
    sustained = sum(recent) >= ALERT_THRESHOLD

    if sustained:
        return (
            "Sustained fatigue detected. Take a 10-minute break, hydrate, "
            "and check in with your supervisor.",
            pick_song(CALM_SONGS),
        )
    if window[-1]["risk"] == 1:
        # A blip, not a trend — gentle nudge, no alarm.
        return (
            "Brief fatigue spike. Ease off for a moment and reset.",
            pick_song(CALM_SONGS),
        )
    return (
        "You're clear. Keep the rhythm going and stay hydrated.",
        pick_song(FOCUS_SONGS),
    )


def generate_transcript() -> str:
    """Generate realistic mock transcripts"""
    transcripts = [
        "I'm doing okay, just pushing through.",
        "Yeah, I'm fine. Just a bit tired.",
        "Everything's under control. No issues here.",
        "I could use a break soon, but I'm managing.",
        "Feeling pretty good actually. Ready to keep going.",
        "It's been a long shift, but I'm hanging in there."
    ]
    return random.choice(transcripts)


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

    Signals you can combine (your call — this shapes what the dashboard reports):
      - Slow speech: low WPM reads as tired. A fresh speaker is ~130-160 WPM;
        under ~90 WPM looks fatigued.
      - Long pauses: large gaps between words[i]["end"] and words[i+1]["start"]
        = hesitation / dragging.
      - Low confidence: slurred/mumbled words lower Deepgram's per-word confidence.

    Return a float clamped to [0.0, 1.0], rounded to 2 decimals.

    Default implementation: a weighted blend of three sub-scores (each 0-1).
    Tweak the thresholds and weights below to taste.
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


def record_reading(risk: int, heart_rate: int, temperature: float) -> None:
    """Append a new reading (risk + vitals) to the sliding window."""
    risk_window.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "risk": risk,
        "heart_rate": heart_rate,
        "temperature": temperature
    })


# API Endpoints
@app.get("/")
async def serve_dashboard():
    """Serve the dashboard UI at the root, so localhost:8000 shows the app."""
    return FileResponse(DASHBOARD_FILE)


@app.post("/")
async def receive_arduino_data(reading: ArduinoReading):
    """Receive live sensor data from bridge.py (Arduino → Serial → bridge → here)"""
    global latest_arduino_reading
    latest_arduino_reading = reading

    # Convert risk_score to binary (0 or 1) for consistency with WESAD flow
    risk = 1 if reading.risk_score >= 0.5 else 0

    # Record in sliding window
    record_reading(risk, int(reading.bpm), reading.temp_c)

    print(f"[ARDUINO] BPM={reading.bpm}, Temp={reading.temp_c}°C, Risk={reading.risk_score:.2f} → {risk}")

    return {"status": "ok", "received_at": datetime.now().isoformat()}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "burnout-detection-api"
    }


@app.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard():
    """Get current dashboard state with REAL model prediction.

    Priority: Live Arduino data > WESAD replay > Mock data
    """
    # PRIORITY 1: Use live Arduino data if available
    if latest_arduino_reading is not None:
        risk = 1 if latest_arduino_reading.risk_score >= 0.5 else 0
        heart_rate = int(latest_arduino_reading.bpm)
        temperature = round(latest_arduino_reading.temp_c, 1)

        print(f"[LIVE] Arduino data: BPM={heart_rate}, Temp={temperature}, Risk={risk}")

    # PRIORITY 2: Use WESAD replay if available and no live data
    elif replay is not None:
        try:
            # Get next WESAD window features
            features = replay.get_next()

            # Predict with real model
            result = predict_stress(features)
            risk = result['prediction']  # 0 or 1

            # Derive vitals from features (use actual HR and temp from sensors)
            heart_rate = int(features.hr_mean)
            temperature = round(features.temp_mean, 1)

            print(f"[MODEL] Predicted risk={risk} (prob={result['probability']:.2f}), HR={heart_rate}, Temp={temperature}")

        except Exception as e:
            print(f"[ERROR] Model prediction failed: {e}, falling back to mock")
            risk = generate_risk()
            heart_rate, temperature = generate_vitals(risk)

    # PRIORITY 3: Fallback to mock if nothing else available
    else:
        risk = generate_risk()
        heart_rate, temperature = generate_vitals(risk)

    # Contributors card = the three raw signals (HR, temp, latest voice fatigue).
    contributors = build_contributors(heart_rate, temperature)

    # Get recommendation based on sliding window
    recommendation, song = decide_recommendation(risk_window)

    print(f"[DEBUG] Dashboard reading: risk={risk}, window_len={len(risk_window)}")

    return DashboardResponse(
        risk=risk,
        risk_label="YES" if risk == 1 else "NO",
        heart_rate=heart_rate,
        temperature=temperature,
        contributors=contributors,
        recommendation=recommendation,
        song=song
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(features: ModelFeatures):
    """REAL prediction endpoint - accepts 18 features, returns model prediction.

    This is the contract for the model team: send 18 sensor features,
    get back binary risk (0/1) with contributors breakdown.
    """
    try:
        # Predict with real model
        result = predict_stress(features)
        risk = result['prediction']  # 0 or 1

        # Contributors = raw signals: HR + temp from the features, latest voice fatigue.
        contributors = build_contributors(int(features.hr_mean), round(features.temp_mean, 1))

        print(f"[PREDICT] Input features → risk={risk} (prob={result['probability']:.2f})")

        return PredictResponse(
            risk=risk,
            contributors=contributors,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        # Fallback to mock if prediction fails
        return PredictResponse(
            risk=generate_risk(),
            contributors=build_contributors(int(features.hr_mean), round(features.temp_mean, 1)),
            timestamp=datetime.now().isoformat()
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
        #    as the artifact the user asked for.
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

        # 4. Derive metrics (speech_rate here, fatigue is yours to define).
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


@app.get("/latest-transcript", response_model=TranscriptResponse)
async def get_latest_transcript():
    """Return the latest real transcription from /transcribe if we have one,
    otherwise fall back to mock data so the dashboard still shows something
    before the first recording."""
    if latest_transcript is not None:
        return latest_transcript

    transcript = generate_transcript()
    return TranscriptResponse(
        transcript=transcript,
        speech_rate=random.randint(80, 120),  # words per minute
        acoustic_fatigue=round(random.uniform(0.1, 0.9), 2),
        timestamp=datetime.now().isoformat()
    )


@app.get("/history", response_model=HistoryResponse)
async def get_history():
    """Return the sliding window of recent binary readings."""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
