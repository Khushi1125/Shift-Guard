# Integration Guide: Schemas → FastAPI

## Overview

This guide shows how the new Phase 1 schemas (`SensorReading`, `VoiceFeatures`, `ModelFeatures`) integrate with the existing FastAPI endpoints in `main.py`.

---

## Current Architecture

### Existing (main.py)
**API Response Models** - What the dashboard consumes:
- `DashboardResponse` - Full dashboard state
- `PredictResponse` - Risk prediction results
- `TranscriptResponse` - Latest voice transcript
- `HistoryResponse` - Historical risk scores

### New (schemas.py)
**Data Pipeline Models** - Internal data contracts:
- `SensorReading` - Raw sensor data
- `VoiceFeatures` - Voice analysis results
- `ModelFeatures` - Model input contract

---

## Integration Points

### 1. POST /predict Endpoint

**Current (Mock):**
```python
@app.post("/predict", response_model=PredictResponse)
async def predict():
    risk_score = random.randint(10, 95)  # Mock
    risk_level = generate_risk_level(risk_score)
    contributors = generate_contributors()

    return PredictResponse(...)
```

**Future (Real Pipeline):**
```python
from schemas import ModelFeatures

@app.post("/predict", response_model=PredictResponse)
async def predict(features: ModelFeatures):
    """
    Receives ModelFeatures from data pipeline.
    Returns PredictResponse for dashboard.
    """
    # Model prediction (Shrima's code)
    result = model.predict(features)

    # Convert to API response format
    return PredictResponse(
        risk_score=result.risk_score,
        risk_level=result.risk_level,
        contributors=Contributors(
            voice_fatigue=result.voice_contribution,
            movement_drift=result.movement_contribution,
            hrv=result.hrv_contribution,
            shift_duration=result.shift_contribution
        ),
        timestamp=datetime.now().isoformat()
    )
```

---

### 2. GET /latest-transcript Endpoint

**Current (Mock):**
```python
@app.get("/latest-transcript", response_model=TranscriptResponse)
async def get_latest_transcript():
    transcript = generate_transcript()
    return TranscriptResponse(
        transcript=transcript,
        speech_rate=random.randint(80, 120),
        acoustic_fatigue=round(random.uniform(0.1, 0.9), 2),
        timestamp=datetime.now().isoformat()
    )
```

**Future (Real Data):**
```python
from schemas import VoiceFeatures

@app.get("/latest-transcript", response_model=TranscriptResponse)
async def get_latest_transcript():
    """
    Fetch latest VoiceFeatures from Deepgram processing.
    Convert to TranscriptResponse for dashboard.
    """
    # Get from voice processing pipeline
    voice_data = get_latest_voice_features()  # Returns VoiceFeatures

    return TranscriptResponse(
        transcript=voice_data.transcript,
        speech_rate=voice_data.speech_rate,
        acoustic_fatigue=voice_data.acoustic_fatigue,
        timestamp=datetime.fromtimestamp(voice_data.timestamp).isoformat()
    )
```

---

### 3. Data Pipeline Flow (New)

**Add internal pipeline endpoint:**
```python
from schemas import SensorReading, VoiceFeatures, ModelFeatures
from typing import List

@app.post("/internal/sensor-readings")
async def ingest_sensor_readings(readings: List[SensorReading]):
    """
    Internal endpoint for WESAD replay or ESP32 live data.
    Stores readings in processing buffer.
    """
    # Add to windowing buffer
    sensor_buffer.extend(readings)
    return {"status": "ingested", "count": len(readings)}


@app.post("/internal/voice-features")
async def ingest_voice_features(features: VoiceFeatures):
    """
    Internal endpoint for Deepgram voice processing results.
    Stores latest voice features.
    """
    global latest_voice_features
    latest_voice_features = features
    return {"status": "stored"}


@app.get("/internal/model-features", response_model=ModelFeatures)
async def get_model_features():
    """
    Internal endpoint that combines sensor + voice features.
    Returns ModelFeatures ready for model.predict().
    """
    # Extract from 30-second sensor window
    sensor_features = extract_features(sensor_buffer[-30:])

    # Combine with voice features
    model_input = ModelFeatures(
        movement_score=sensor_features["movement_score"],
        hrv_score=sensor_features["hrv_score"],
        speech_rate=latest_voice_features.speech_rate,
        acoustic_fatigue=latest_voice_features.acoustic_fatigue,
        shift_duration_hours=get_shift_duration()
    )

    return model_input
```

---

## Complete Data Flow

```
WESAD/ESP32 Data
    ↓
POST /internal/sensor-readings (SensorReading)
    ↓
Windowing Buffer (30s)
    ↓
Feature Extraction
    ↓ + Deepgram
    ↓   ↓
    ↓   POST /internal/voice-features (VoiceFeatures)
    ↓   ↓
GET /internal/model-features (ModelFeatures)
    ↓
Model.predict()
    ↓
POST /predict (PredictResponse)
    ↓
GET /dashboard (DashboardResponse)
    ↓
Dashboard UI
```

---

## Model Team Handoff

### What Shrima's Model Receives
```python
from schemas import ModelFeatures

def predict(features: ModelFeatures):
    """
    Input: ModelFeatures object with 5 fields
    Output: Risk prediction with score, level, and contributors
    """
    # Access validated fields
    movement = features.movement_score      # 0.0-1.0
    hrv = features.hrv_score                # 0.0-1.0
    speech = features.speech_rate           # WPM
    acoustic = features.acoustic_fatigue    # 0.0-1.0
    duration = features.shift_duration_hours  # hours

    # Model logic here
    risk_score = ...
    risk_level = "LOW" | "MEDIUM" | "HIGH"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "contributors": {
            "voice_fatigue": ...,
            "movement_drift": ...,
            "hrv": ...,
            "shift_duration": ...
        }
    }
```

### Contract Guarantee
- **Type Safety:** All fields validated by Pydantic
- **Bounds Checking:** Scores are 0.0-1.0, duration ≥ 0
- **No Missing Data:** All fields required (no None/null)
- **Consistent Format:** Same structure from WESAD or ESP32

---

## Testing Integration

### Test Schema Compatibility
```python
from schemas import ModelFeatures
from main import PredictResponse, Contributors

# Create model input
model_input = ModelFeatures(
    movement_score=0.31,
    hrv_score=0.58,
    speech_rate=92,
    acoustic_fatigue=0.74,
    shift_duration_hours=10.5
)

# Simulate model prediction
risk_score = 72
risk_level = "HIGH"

# Convert to API response
response = PredictResponse(
    risk_score=risk_score,
    risk_level=risk_level,
    contributors=Contributors(
        voice_fatigue=30,
        movement_drift=20,
        hrv=12,
        shift_duration=10
    ),
    timestamp="2024-06-20T15:30:00"
)

print(response.model_dump_json(indent=2))
```

---

## Next Steps

1. **Replace Mock Data** - Swap `random.randint()` with real pipeline data
2. **Add Internal Endpoints** - Create `/internal/*` routes for data ingestion
3. **Implement Windowing** - Buffer 30 seconds of `SensorReading` objects
4. **Connect Model** - Call Shrima's `predict()` with `ModelFeatures`
5. **Test End-to-End** - WESAD replay → Model → Dashboard

---

**Status:** Ready for Phase 2 (Data Loaders) and Phase 4 (Model Integration)
