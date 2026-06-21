# Arize Monitoring Setup Guide

## What We Added

Arize logging is now integrated into the Shift-Guard prediction pipeline. Every time `predict_stress()` is called, it logs:
- **All 18 input features** (accelerometer, BVP, heart rate, temperature, EDA)
- **Prediction outputs** (risk level, confidence scores, probabilities)
- **Model metadata** (model name, version)
- **Timestamp** for each prediction

## Setup Instructions

### 1. Sign up for Arize (FREE)
1. Go to https://app.arize.com/
2. Create an account (hackathon tier is free)
3. Create a new project called "shift-guard"

### 2. Get Your API Keys
1. In Arize dashboard, go to **Settings > API Keys**
2. Copy your:
   - **API Key** (starts with `arize_`)
   - **Space ID** (looks like a UUID)

### 3. Configure Environment Variables

✅ **Already done!** Your `.env` file is already set up with:
- ARIZE_API_KEY
- ARIZE_SPACE_ID

The code automatically loads these on startup.

### 4. Install Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Test It!

Just run the API - everything auto-loads:

```bash
python api/main.py
```

You should see:
```
[OK] Arize monitoring initialized  ✅
[OK] Loaded 1780 WESAD windows for replay
[OK] Loaded model feature importances
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Then test predictions:
```bash
# In another terminal
curl http://localhost:8000/dashboard
```

Every prediction will log to Arize automatically!

## What to Show Judges

When demoing to Arize track judges:

1. **Show the logs in terminal** - They'll see `[ARIZE] Logged prediction` messages
2. **Open Arize dashboard** - Navigate to your project and show:
   - Live predictions streaming in
   - Feature distributions (18 sensor inputs)
   - Model performance metrics
   - Prediction timeline
3. **Explain the value**:
   - "We're monitoring every prediction in real-time"
   - "All 18 sensor features are tracked for drift detection"
   - "Risk scores are logged with full input/output traceability"

## What Gets Logged

Each prediction creates a span with:
- `span_id`: Timestamp-based unique ID
- `timestamp`: When the prediction was made
- `input.*`: All 18 sensor features
- `output.prediction`: Binary risk (0 or 1)
- `output.risk_level`: "LOW" or "HIGH"
- `output.confidence`: Model confidence (0.0-1.0)
- `output.calm_probability`: Probability of calm state
- `output.stressed_probability`: Probability of stressed state
- `attributes.llm.model_name`: "shift-guard-rf-baseline"
- `attributes.model_version`: "1.0"

## Troubleshooting

**"Arize API keys not found"**
- Make sure `.env` file exists in project root
- Check that variable names are `ARIZE_API_KEY` and `ARIZE_SPACE_ID` (not SPACE_KEY)
- Load environment variables: `source .env` (if running manually)

**Predictions work but no logs in Arize**
- Check HTTP status: Look for `[ARIZE] Log failed: HTTP XXX` messages
- Verify API keys are correct
- Check internet connection (Arize needs to send data to cloud)

**"ARIZE ERROR: ..."**
- Predictions will still work! Arize logging is non-blocking
- Check the error message for details
- Common issue: Space ID format (should be UUID format)

## Performance Impact

- Logging is **asynchronous** - doesn't block predictions
- Adds ~10-20ms overhead per prediction
- Safe to use in production/demo

## For the Hackathon

**What judges want to see:**
1. Live predictions flowing into Arize dashboard
2. Feature tracking (all 18 sensor inputs visible)
3. Model monitoring setup (drift detection potential)
4. Traceability - every prediction has full context

**Demo script:**
> "We're using Arize to monitor every prediction our model makes. You can see here (show dashboard) all 18 sensor features being tracked in real-time. This gives us full traceability - for any risk alert, we can see exactly what sensor readings triggered it and drill down into the model's decision."
