# Shift-Guard: Worker Fatigue Detection System

AI-powered fatigue detection for healthcare workers using sensor data and voice analysis.

---

## Project Structure

```
Shift-Guard/
├── pipeline/              # Data pipeline (Abby's code)
│   ├── schemas.py         # Pydantic models (SensorReading, VoiceFeatures, ModelFeatures)
│   ├── loaders.py         # Phase 2: WESAD/ESP32 data loaders (TODO)
│   ├── features.py        # Phase 3: Feature extraction (TODO)
│   └── windowing.py       # Phase 3: 30-second windowing (TODO)
│
├── model/                 # ML model & EDA (Friend's code)
│   └── (Drop your ML code here)
│
├── api/                   # FastAPI backend
│   └── main.py            # REST API endpoints
│
├── frontend/              # Dashboard UI
│   └── dashboard.html     # Real-time monitoring dashboard
│
├── tests/                 # Test suite
│   ├── test_schemas.py    # Schema validation tests
│   └── test_phase1.py     # Comprehensive Phase 1 tests (12 tests)
│
├── scripts/               # Utilities & demos
│   └── example_pipeline.py  # Pipeline demonstration
│
├── docs/                  # Documentation
│   ├── PHASE1_COMPLETE.md    # Phase 1 summary
│   ├── INTEGRATION_GUIDE.md  # How to integrate schemas with API
│   ├── TEST_RESULTS.md       # Test results
│   └── README_nextmove.md    # Phase 2-4 roadmap
│
├── requirements.txt
└── venv/
```

---

## Quick Start

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Tests
```bash
# Quick schema validation
python -m tests.test_schemas

# Comprehensive tests (12 tests)
python -m tests.test_phase1

# Pipeline demo
python -m scripts.example_pipeline
```

### Run API Server
```bash
# Start FastAPI backend
python -m api.main
# Visit: http://localhost:8000/docs
```

### Open Dashboard
```bash
# Open in browser
open frontend/dashboard.html
# Or visit: http://localhost:8000 (when serving)
```

---

## Current Status

### ✅ Phase 1 Complete (Schemas)
- [x] `SensorReading` schema (accelerometer + HR + temp)
- [x] `VoiceFeatures` schema (transcript + speech rate + fatigue)
- [x] `ModelFeatures` schema (model input contract)
- [x] All tests passing (12/12)

### 🔄 Phase 2 In Progress (Data Loaders)
- [ ] WESAD loader (`pipeline/loaders.py`)
- [ ] ESP32 loader
- [ ] Label mapping (WESAD → LOW/MEDIUM/HIGH)

### 📋 Phase 3 Planned (Features)
- [ ] 30-second windowing
- [ ] Feature extraction
- [ ] Sensor + voice merge

### 📋 Phase 4 Planned (Integration)
- [ ] ML model integration
- [ ] FastAPI endpoints
- [ ] End-to-end testing

---

## Team Ownership

| Folder | Owner | Purpose |
|--------|-------|---------|
| `pipeline/` | Abby | Data pipeline (Phase 1-3) |
| `model/` | Friend | ML model & EDA |
| `api/` | Shared | FastAPI backend |
| `frontend/` | Shared | Dashboard UI |

---

## How to Integrate

### For Pipeline Team (Abby)
```python
# Import schemas in your pipeline code
from pipeline.schemas import SensorReading, VoiceFeatures, ModelFeatures

# Create sensor readings
reading = SensorReading(
    timestamp=1719000000,
    accel_x=0.12,
    accel_y=-0.91,
    accel_z=0.08,
    heart_rate=82,
    temperature=98.2
)
```

### For ML Team (Friend)
```python
# Import schemas in your model code
from pipeline.schemas import ModelFeatures

# Your model receives this contract
def predict_risk(features: ModelFeatures):
    # Access validated fields
    movement = features.movement_score
    hrv = features.hrv_score
    speech = features.speech_rate
    # ... your model logic
    return risk_prediction
```

### For API Team (Integration)
```python
# In api/main.py
from pipeline.schemas import ModelFeatures
from model.predict import predict_risk  # Friend's model

@app.post("/predict")
async def predict(features: ModelFeatures):
    result = predict_risk(features)
    return result
```

---

## Data Flow

```
WESAD/ESP32 Sensors
    ↓
pipeline/loaders.py → SensorReading
    ↓
pipeline/windowing.py (30-second window)
    ↓
pipeline/features.py → Extract features
    ↓
Deepgram Voice → VoiceFeatures
    ↓
Merge → ModelFeatures
    ↓
model/ → predict_risk()
    ↓
api/main.py → POST /predict
    ↓
frontend/dashboard.html
```

---

## Documentation

- **Phase 1 Summary**: `docs/PHASE1_COMPLETE.md`
- **Integration Guide**: `docs/INTEGRATION_GUIDE.md`
- **Test Results**: `docs/TEST_RESULTS.md`
- **Roadmap**: `docs/README_nextmove.md`

---

## Testing

All Phase 1 tests passing:
- ✓ SensorReading validation
- ✓ VoiceFeatures validation
- ✓ ModelFeatures validation
- ✓ Pipeline integration
- ✓ Edge cases & error handling

**Total: 12/12 tests passing**

---

## Next Steps

1. **ML Team**: Drop EDA code in `model/` folder
2. **Pipeline Team**: Build WESAD loader (Phase 2)
3. **Integration**: Connect pipeline → model → API

---

**Hackathon Project** | **Team**: Abby + Friend | **Last Updated**: June 20, 2026
