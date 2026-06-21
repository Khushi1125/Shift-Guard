Phase 1: Backend Contract & Mock Data

Goal

Get the dashboard working before the model is finished.

Gap #2: Mock Data Generator

You need a way to demo before Deepgram, ESP32, and the model are finished.

Add to Phase 1:

Mock Data Script
risk_score = random.randint(10, 95)

Generate:

transcript
fatigue score
risk level
contributors

This lets your dashboard work immediately.

Tasks
Create FastAPI endpoints
Define JSON schema
Return fake burnout scores

At the end of this phase, clicking refresh should update the dashboard.

Define exact response schema that the model team must return
Create /predict
Create /health
Create /latest-transcript (mock Deepgram data)
Create /history (risk score timeline)


Gap 3: API Endpoint for Current Dashboard State

Right now you have:

/health
/predict
/history
/latest-transcript

I would add:

/dashboard

Example response:

{
  "risk_score": 72,
  "risk_level": "HIGH",
  "contributors": {
    "voice_fatigue": 35,
    "movement_drift": 20,
    "hrv": 10,
    "shift_duration": 12
  },
  "recommendation": "Take a 10-minute break"
}


Phase 2: Build Dashboard UI (2-3 hours)
Goal

Gap #3: System Status Card

Your project brief repeatedly says:

show sponsor integrations live

I'd add this to Phase 2.

System Status

🟢 Deepgram
🟢 Model
🟢 Dashboard
🟡 ESP32
🟢 QNX

When judges arrive, they instantly see:

hardware
Deepgram
QNX

all connected.

So add:

8AM   15
10AM  22
12PM  31
2PM   48
4PM   72

as a line chart.

Make something judges understand in 5 seconds.

Components
Current Status Card
Risk Score: 72

HIGH RISK


8AM  15
10AM 22
12PM 31
2PM  48
4PM  72


Contributors Panel
Contributors

Voice Fatigue      +35
Movement Drift     +20
HRV                +10
Shift Duration     +12


Recommendation Card
Suggested Action

Take 10-minute break
Hydrate
Check in with supervisor
Deliverable

A complete dashboard that works even with fake data.

Phase 3: Connect Real Model Outputs (2-4 hours)
Goal

Replace fake data with Shrima's model.

Updated Phase 3: Model Integration

I'd make this more specific:

Inputs expected from model
{
  "speech_rate": 92,
  "acoustic_fatigue": 0.74,
  "movement_score": 0.31,
  "hrv_score": 0.58
}
Outputs expected
{
  "risk_score": 82,
  "risk_level": "HIGH",
  "contributors": {...}
}

That gives Shrima a very clear contract.


Phase 4: Real-Time Demo Mode (Final Polish)
Goal

Make judges say "wow."

Add
Live Updates

Updated Phase 4: Real-Time Demo Mode


What I Would Put in Phase 4

Add one final dashboard section:

Why Is The Score High?
Risk Score: 82

Primary Contributors:

Voice Fatigue      +35
Long Shift         +22
Low Activity       +15
HRV                +10

Judges love explainability.

This phase should contain everything judges physically see.

Live Transcript Panel
Latest Check-In

"I'm doing okay, just pushing through."

Speech Rate: 92 WPM
Acoustic Fatigue: 0.74

Alert Banner
⚠ High Fatigue Detected

Suggested Action:
Take a break
Hydrate

Voice Nudge Display

Your brief mentions Deepgram TTS.

Add:

Last Intervention

"You're showing signs of fatigue.
Consider taking a short break."

This makes the Deepgram integration visible.



Poll every 2-5 seconds.

setInterval(fetchRiskScore, 3000)
Status Colors
Green   0-40
Yellow  40-70
Red     70+
Deepgram Transcript Panel
Latest Check-In

"I'm doing okay, just pushing through."

Speech Rate: 92 WPM
Fatigue Score: 0.74
Alert Banner
⚠ High Fatigue Detected

Recommended:
Take a break
Hydrate
Deliverable
Voice/Sensor
     ↓
Model
     ↓
FastAPI
     ↓
Dashboard

updating live.

What I'd personally prioritize first

If you only get one thing done before lunch:

✅ FastAPI endpoint

✅ Dashboard UI

✅ Mock data flowing