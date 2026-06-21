from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime, timedelta
import random

app = FastAPI(title="Burnout Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state to sync current risk with history
current_state = {
    "risk_score": 50,
    "risk_level": "MEDIUM",
    "contributors": None,
    "recommendation": "",
    "history": []
}


# Pydantic Models
class Contributors(BaseModel):
    voice_fatigue: int
    movement_drift: int
    hrv: int
    shift_duration: int


class DashboardResponse(BaseModel):
    risk_score: int
    risk_level: str
    contributors: Contributors
    recommendation: str


class PredictResponse(BaseModel):
    risk_score: int
    risk_level: str
    contributors: Contributors
    timestamp: str


class TranscriptResponse(BaseModel):
    transcript: str
    speech_rate: int
    acoustic_fatigue: float
    timestamp: str


class HistoryPoint(BaseModel):
    timestamp: str
    risk_score: int


class HistoryResponse(BaseModel):
    history: List[HistoryPoint]


# Mock Data Generators
def generate_risk_level(score: int) -> str:
    """Convert risk score to risk level"""
    if score < 40:
        return "LOW"
    elif score < 70:
        return "MEDIUM"
    else:
        return "HIGH"


def generate_contributors() -> Contributors:
    """Generate random contributor values that sum reasonably"""
    return Contributors(
        voice_fatigue=random.randint(5, 40),
        movement_drift=random.randint(5, 30),
        hrv=random.randint(5, 20),
        shift_duration=random.randint(5, 25)
    )


def generate_recommendation(risk_level: str) -> str:
    """Generate recommendation based on risk level"""
    recommendations = {
        "LOW": "Keep up the good work! Stay hydrated.",
        "MEDIUM": "Consider taking a short break. Stay aware of fatigue signs.",
        "HIGH": "Take a 10-minute break immediately. Hydrate and check in with supervisor."
    }
    return recommendations.get(risk_level, "Monitor your condition.")


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


# API Endpoints
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
    """Get current dashboard state with all metrics"""
    risk_score = random.randint(10, 95)
    risk_level = generate_risk_level(risk_score)
    contributors = generate_contributors()
    recommendation = generate_recommendation(risk_level)

    # Update global state so history can sync
    current_state["risk_score"] = risk_score
    current_state["risk_level"] = risk_level
    current_state["contributors"] = contributors
    current_state["recommendation"] = recommendation

    print(f"[DEBUG] Dashboard updated state: risk_score={risk_score}")

    return DashboardResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        contributors=contributors,
        recommendation=recommendation
    )


@app.post("/predict", response_model=PredictResponse)
async def predict():
    """Predict burnout risk (mock endpoint for model team contract)"""
    risk_score = random.randint(10, 95)
    risk_level = generate_risk_level(risk_score)
    contributors = generate_contributors()

    return PredictResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        contributors=contributors,
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
    """Get risk score history over time"""
    now = datetime.now()
    history = []

    print(f"[DEBUG] History reading state: risk_score={current_state['risk_score']}")

    # Generate 8 hours of data points (every 2 hours)
    for i in range(5):
        time_point = now - timedelta(hours=8 - i * 2)

        # Last point should match current risk score
        if i == 4:
            score = current_state["risk_score"]
            print(f"[DEBUG] Using state for last point: {score}")
        else:
            # Simulate increasing risk throughout the day
            base_score = 15 + (i * 15)
            score = base_score + random.randint(-5, 10)

        history.append(HistoryPoint(
            timestamp=time_point.strftime("%I:%M %p"),
            risk_score=score
        ))

    return HistoryResponse(history=history)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
