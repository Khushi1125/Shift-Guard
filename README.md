# Shift-Guard: Worker Fatigue Detection System

AI-powered fatigue detection for healthcare workers using sensor data and voice analysis.

---

## Project Structure

```
Shift-Guard/
├── pipeline/              # Data pipeline (Abby)
│   ├── schemas.py         # Pydantic models (SensorReading, VoiceFeatures, ModelFeatures)
│   ├── loaders.py         # WESAD/ESP32 data loaders
│   ├── features.py        # Feature extraction
│   └── windowing.py       # 30-second windowing
│
├── model/                 # ML model & EDA (Khushi)
│   ├── notebooks/
│   │   ├── EDA.ipynb              # Exploratory analysis + feature export
│   │   └── baseline.ipynb         # Random Forest model + evaluation suite
│   ├── src/
│   │   ├── e4_loader.py           # Reads Empatica E4 zip archives
│   │   ├── extract_features.py    # Builds 30 s windowed feature table
│   │   ├── semantic_analysis.py   # Deepgram STT + VADER sentiment
│   │   ├── tone_analysis.py       # Wav2Vec2 acoustic stress classifier
│   │   ├── final_scoring.py       # Locked 4-function ML interface
│   │   └── eval_semantic.py       # VADER evaluation script
│   ├── outputs/
│   │   ├── features_30s.csv       # Modeling-ready feature table
│   │   ├── baseline_model.onnx    # ONNX export for cross-platform inference
│   │   ├── rf_baseline.joblib     # Trained Random Forest + metadata
│   │   ├── acc_placeholders.json  # Imputation values for missing sensor channels
│   │   └── eval_semantic.json     # VADER evaluation results
│   └── predict.py                 # Prediction entry point
│
├── api/                   # FastAPI backend
│   └── main.py            # REST API endpoints
│
├── frontend/              # Dashboard UI
│   └── dashboard.html     # Real-time monitoring dashboard
│
├── arduino/               # Hardware / Arduino
│   └── main.ino
│
├── tests/                 # Test suite
│   ├── test_schemas.py
│   ├── test_phase1.py
│   ├── test_full_integration.py   # ML pipeline integration test
│   └── test_semantic_analysis.py
│
├── docs/                  # Documentation
├── INTEGRATION.md         # ML function interface contract for backend
├── requirements.txt
└── venv/
```

---

## Quick Start

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### ML Pipeline (Khushi)
```bash
# Run full integration test (sensor + voice + final score)
HF_HOME=.hf_cache python tests/test_full_integration.py

# Smoke-test the scoring interface
python model/src/final_scoring.py
```

### API Server
```bash
python -m api.main
# Visit: http://localhost:8000/docs
```

### Dashboard
```bash
open frontend/dashboard.html
```

---

## ML Interface Contract

The backend imports four stable functions from `model/src/final_scoring.py`.
See **[INTEGRATION.md](INTEGRATION.md)** for the full contract.

```python
from model.src.final_scoring import (
    predict_sensor_score,       # temp + HR → stress probability
    run_voice_checkin,          # wav path → transcript + scores
    compute_final_risk_score,   # sensor + voice → final_score dict
    check_and_get_intervention, # final_score → intervention alert
)

# Stateful helpers (recommended for live use):
from model.src.final_scoring import on_sensor_update, on_voice_update
```

---

## Current Status

### ✅ Phase 1 Complete — Schemas & ML pipeline
- [x] `SensorReading`, `VoiceFeatures`, `ModelFeatures` schemas
- [x] Random Forest baseline (F1 = 0.86, LOSO cross-validation on WESAD)
- [x] ONNX export for cross-platform inference
- [x] Semantic analysis (Deepgram + VADER)
- [x] Tone analysis (Wav2Vec2 emotion model)
- [x] Final scoring: stateful independent-clock combination logic
- [x] All ML integration tests passing

### 🔄 Phase 2 In Progress
- [ ] WESAD loader (`pipeline/loaders.py`)
- [ ] ESP32 loader
- [ ] FastAPI endpoints connected to ML functions

## Ethical Considerations

Shift-Guard was designed for high-stress workplaces where fatigue detection can improve safety, but it can also create real risks if it is treated like a surveillance tool. We addressed those risks with a local-first, opt-in architecture and clear limits on what the system records, stores, and decides.

### Privacy & Security
- We avoided ambient always-on recording. Voice capture only happens through an explicit check-in flow, so the microphone is not continuously collecting audio in the background.
- Raw voice data is treated as transient input. It is transcribed, converted into features or scores, and not kept as a persistent audio archive.
- Sensor data is reduced into fixed windows and model features rather than being exposed as open-ended personal telemetry.
- Sensitive configuration is handled through environment variables and local settings rather than hard-coded secrets.

### Social Impact
- The project is positioned as a support and safety aid, not as a disciplinary monitoring system.
- Model outputs are framed as risk signals and recommendations, which keeps a human in the loop for any operational response.
- Per-subject normalization and calibration reduce the chance that one person's baseline is incorrectly treated as another person's abnormal state.
- The design acknowledges the risk of false positives, stigmatization, and over-trust in AI, so the output is intended to inform review rather than make autonomous decisions.

### Environmental Impact
- The system uses lightweight classical ML for the core fatigue model instead of requiring a large foundation model for every prediction.
- We export the model to ONNX for efficient cross-platform inference, which helps keep deployment and runtime costs lower.
- Where possible, the workflow reuses cached or local artifacts instead of reprocessing or retraining from scratch on every run.

### Accountability
- The pipeline is modular and testable, so each stage can be validated independently.
- The model contract is explicit, which makes it easier to audit the inputs, outputs, and assumptions behind a prediction.
- The output is a recommendation layer, not an automated enforcement mechanism, so supervisors or clinicians retain responsibility for final action.

---

## Team Ownership

| Folder | Owner | Purpose |
|--------|-------|---------|
| `pipeline/` | Abby | Data pipeline & schemas |
| `model/` | Khushi | ML model, EDA, scoring |
| `api/` | Shared | FastAPI backend |
| `frontend/` | Shared | Dashboard UI |
| `arduino/` | Hardware team | Sensor firmware |

---

**Hackathon Project** | **Last Updated**: June 21, 2026
