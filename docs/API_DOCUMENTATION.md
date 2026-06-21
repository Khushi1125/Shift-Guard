# Burnout Detection API Documentation

## Phase 1: Backend Contract & Mock Data ✅

Base URL: `http://localhost:8000`

---

## Endpoints

### 1. Health Check
**GET** `/health`

Health check endpoint to verify the API is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-06-20T15:01:29.538196",
  "service": "burnout-detection-api"
}
```

---

### 2. Dashboard (Main Endpoint)
**GET** `/dashboard`

Returns the complete current dashboard state with all metrics.

**Response:**
```json
{
  "risk_score": 72,
  "risk_level": "HIGH",
  "contributors": {
    "voice_fatigue": 35,
    "movement_drift": 20,
    "hrv": 10,
    "shift_duration": 12
  },
  "recommendation": "Take a 10-minute break immediately. Hydrate and check in with supervisor."
}
```

**Risk Levels:**
- `LOW`: 0-39
- `MEDIUM`: 40-69
- `HIGH`: 70-95

---

### 3. Predict
**POST** `/predict`

Main prediction endpoint. This is the contract endpoint for the model team.

**Response:**
```json
{
  "risk_score": 46,
  "risk_level": "MEDIUM",
  "contributors": {
    "voice_fatigue": 38,
    "movement_drift": 30,
    "hrv": 19,
    "shift_duration": 22
  },
  "timestamp": "2026-06-20T15:01:53.608643"
}
```

---

### 4. Latest Transcript
**GET** `/latest-transcript`

Returns the latest transcript from Deepgram (currently mock data).

**Response:**
```json
{
  "transcript": "Everything's under control. No issues here.",
  "speech_rate": 84,
  "acoustic_fatigue": 0.85,
  "timestamp": "2026-06-20T15:02:09.359068"
}
```

**Fields:**
- `transcript`: Text transcription of the latest audio
- `speech_rate`: Words per minute (80-120 range)
- `acoustic_fatigue`: Fatigue score from 0.0-1.0
- `timestamp`: ISO format timestamp

---

### 5. History
**GET** `/history`

Returns risk score history over time (8-hour shift with 2-hour intervals).

**Response:**
```json
{
  "history": [
    {"timestamp": "07:02 AM", "risk_score": 24},
    {"timestamp": "09:02 AM", "risk_score": 32},
    {"timestamp": "11:02 AM", "risk_score": 52},
    {"timestamp": "01:02 PM", "risk_score": 64},
    {"timestamp": "03:02 PM", "risk_score": 79}
  ]
}
```

---

## Testing

### Quick Test Commands

```bash
# Health check
curl http://localhost:8000/health

# Dashboard
curl http://localhost:8000/dashboard

# Predict
curl -X POST http://localhost:8000/predict

# Latest transcript
curl http://localhost:8000/latest-transcript

# History
curl http://localhost:8000/history
```

### Interactive Dashboard

Open `test_dashboard.html` in your browser to see a visual dashboard that refreshes with new mock data.

---

## Running the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python main.py

# Or with uvicorn directly
uvicorn main:app --reload
```

The server will start on `http://localhost:8000`

---

## For Model Team Integration (Phase 3)

When integrating the real model, replace the mock data generators in `main.py` with your model outputs:

1. **Input to model** (from sensors/Deepgram):
   - Speech rate
   - Acoustic fatigue
   - Movement score
   - HRV score

2. **Expected output from model**:
   - Risk score (0-95)
   - Risk level (LOW/MEDIUM/HIGH)
   - Contributors breakdown
   - Recommendation text

The `/predict` endpoint is your integration point.

---

## Notes

- All endpoints return JSON
- Mock data uses `random` module to generate realistic variations
- Risk scores increase throughout the day in `/history` to simulate fatigue
- CORS is enabled for all origins (for frontend testing)
- Each request generates new random data to simulate live updates
