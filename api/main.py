from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from collections import deque
from urllib.parse import quote_plus
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_FILE = os.path.join(BASE_DIR, "frontend", "dashboard.html")

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
    voice_fatigue: int
    movement_drift: int
    hrv: int
    shift_duration: int


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

# ── HELPERS ───────────────────────────────────────────────────────────────────

def risk_label(score: float) -> str:
    if score >= 0.65:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"


def generate_contributors() -> Contributors:
    return Contributors(
        voice_fatigue=random.randint(5, 40),
        movement_drift=random.randint(5, 30),
        hrv=random.randint(5, 20),
        shift_duration=random.randint(5, 25)
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
    if latest_arduino_reading is None:
        return DashboardResponse(
            risk=0.0,
            risk_label="LOW",
            heart_rate=0,
            temperature=0.0,
            contributors=generate_contributors(),
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
        contributors=generate_contributors(),
        recommendation=recommendation,
        song=song
    )


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