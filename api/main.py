from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime
from collections import deque
from urllib.parse import quote_plus
import os
import random

# ========== REAL MODEL INTEGRATION ==========
from model.predict import predict_stress
from pipeline.schemas import ModelFeatures
import pandas as pd
import joblib
import numpy as np

# Project root (one level up from this api/ folder), used to locate the frontend.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    voice_fatigue: int
    movement_drift: int
    hrv: int
    shift_duration: int


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


# Contributor Calculation using Feature Importances
def calculate_contributors(features: ModelFeatures) -> Contributors:
    """
    Calculate contributors based on actual model feature importances.

    Groups features by sensor type and weights by importance:
    - EDA (63%): Stress markers from skin conductance
    - Heart Rate (21%): Cardiovascular response
    - Temperature (11%): Thermal regulation
    - Movement (4%): Physical activity patterns

    Returns normalized scores (0-100) showing relative contribution to risk.
    """
    if feature_importances is None:
        # Fallback to mock if importances not loaded
        return generate_contributors()

    # Convert features to dict for easier access
    feat_dict = features.model_dump()

    # Calculate weighted scores for each sensor group
    # Formula: sum(feature_value * normalized_feature_importance * scaling_factor)

    # EDA group (63% importance): eda_mean, eda_max, eda_std
    # WESAD S2 range: mean 0.09-1.28 (avg ~0.48), max 0.1-1.5 (avg ~0.53), std 0.001-0.13 (avg ~0.02)
    eda_importance = sum([
        feature_importances.get('eda_mean', 0),
        feature_importances.get('eda_max', 0),
        feature_importances.get('eda_std', 0)
    ])
    eda_weighted = (
        feat_dict['eda_mean'] * feature_importances.get('eda_mean', 0) +
        feat_dict['eda_max'] * feature_importances.get('eda_max', 0) +
        feat_dict['eda_std'] * feature_importances.get('eda_std', 0) * 20  # std is much smaller
    )
    eda_score = (eda_weighted / eda_importance) * 100  # Scale: 0.5 mean → ~50/100, 1.0 → ~100/100

    # Heart Rate group (21% importance): hr_mean, hr_min, hr_max, hr_std
    # WESAD range: mean 50-140 (avg ~84), std 0-23 (avg ~6), min 43-133, max 54-145
    hr_importance = sum([
        feature_importances.get('hr_mean', 0),
        feature_importances.get('hr_min', 0),
        feature_importances.get('hr_max', 0),
        feature_importances.get('hr_std', 0)
    ])
    hr_weighted = (
        (feat_dict['hr_mean'] - 70) * feature_importances.get('hr_mean', 0) +  # Baseline 70 bpm
        (feat_dict['hr_max'] - feat_dict['hr_min']) * feature_importances.get('hr_min', 0) * 0.1 +
        feat_dict['hr_std'] * feature_importances.get('hr_std', 0)
    )
    hr_score = (hr_weighted / hr_importance) * 2.5  # Scale: 84 mean → ~35/100, 110 → ~75/100

    # Temperature group (11% importance): temp_mean, temp_slope, temp_delta
    # WESAD range: mean 29-36°C (avg ~33), delta -0.4 to +0.5
    temp_importance = sum([
        feature_importances.get('temp_mean', 0),
        feature_importances.get('temp_slope', 0),
        feature_importances.get('temp_delta', 0)
    ])
    temp_weighted = (
        (feat_dict['temp_mean'] - 33.0) * feature_importances.get('temp_mean', 0) * 5 +  # Baseline 33°C
        abs(feat_dict['temp_slope']) * feature_importances.get('temp_slope', 0) * 100 +
        abs(feat_dict['temp_delta']) * feature_importances.get('temp_delta', 0) * 50
    )
    temp_score = (temp_weighted / temp_importance) * 10  # Scale appropriately

    # Movement group (4% importance): acc_mag_mean, acc_mag_std, acc_hf_mean
    # WESAD range: mag_mean 62-67 (gravity units, avg ~63.7), std 0-22 (avg ~2.6)
    acc_importance = sum([
        feature_importances.get('acc_mag_mean', 0),
        feature_importances.get('acc_mag_std', 0),
        feature_importances.get('acc_hf_mean', 0)
    ])
    movement_weighted = (
        abs(feat_dict['acc_mag_mean'] - 63.7) * feature_importances.get('acc_mag_mean', 0) * 5 +  # Deviation from mean
        feat_dict['acc_mag_std'] * feature_importances.get('acc_mag_std', 0) +
        feat_dict['acc_hf_mean'] * feature_importances.get('acc_hf_mean', 0) * 10
    )
    movement_score = (movement_weighted / acc_importance) * 15  # Scale: 2.6 std → ~40/100

    # Clamp all scores to 0-100 range
    return Contributors(
        voice_fatigue=0,  # Not implemented yet (placeholder for future Deepgram integration)
        movement_drift=max(0, min(100, int(movement_score))),
        hrv=max(0, min(100, int(hr_score))),
        shift_duration=max(0, min(100, int(eda_score)))  # Using EDA as "stress accumulation" proxy
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


def generate_contributors() -> Contributors:
    """Generate signal levels that drove the decision."""
    return Contributors(
        voice_fatigue=random.randint(5, 40),
        movement_drift=random.randint(5, 30),
        hrv=random.randint(5, 20),
        shift_duration=random.randint(5, 25)
    )


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
    """Get current dashboard state with REAL model prediction."""
    # Use real model if replay is available, otherwise fallback to mock
    if replay is not None:
        try:
            # Get next WESAD window features
            features = replay.get_next()

            # Predict with real model
            result = predict_stress(features)
            risk = result['prediction']  # 0 or 1

            # Derive vitals from features (use actual HR and temp from sensors)
            heart_rate = int(features.hr_mean)
            temperature = round(features.temp_mean, 1)

            # Calculate contributors using feature importances
            contributors = calculate_contributors(features)

            print(f"[MODEL] Predicted risk={risk} (prob={result['probability']:.2f}), HR={heart_rate}, Temp={temperature}")

        except Exception as e:
            print(f"[ERROR] Model prediction failed: {e}, falling back to mock")
            risk = generate_risk()
            heart_rate, temperature = generate_vitals(risk)
            contributors = generate_contributors()
    else:
        # Fallback to mock if replay not initialized
        risk = generate_risk()
        heart_rate, temperature = generate_vitals(risk)
        contributors = generate_contributors()

    # Record reading and get recommendation
    record_reading(risk, heart_rate, temperature)
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

        # Calculate contributors using feature importances
        contributors = calculate_contributors(features)

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
            contributors=generate_contributors(),
            timestamp=datetime.now().isoformat()
        )


@app.get("/latest-transcript", response_model=TranscriptResponse)
async def get_latest_transcript():
    """Get latest transcript from Deepgram (mock data)"""
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
