# Sensor Data Pipeline Plan

## Architecture Overview

The natural data flow:

```
Raw Data
    ↓
Schema
    ↓
Loader
    ↓
Windowing
    ↓
Features
    ↓
Model
    ↓
FastAPI
    ↓
Dashboard
```

---

## Phase 1: Schema + Data Models

### Goal
Define all data contracts upfront so every component knows what to expect.

### Tasks

#### 1. Define `SensorReading` (Pydantic model)

Standard payload for all sensors:

```python
{
  "timestamp": 1719000000,
  "accel_x": 0.12,
  "accel_y": -0.91,
  "accel_z": 0.08,
  "heart_rate": 82,
  "temperature": 98.2
}
```

**Deliverable:** Both WESAD and ESP32 can produce identical records.

---

#### 2. Define `VoiceFeatures` (Pydantic model)

Voice data from Deepgram:

```python
{
  "transcript": "I'm doing okay, just pushing through.",
  "speech_rate": 92,
  "acoustic_fatigue": 0.74,
  "timestamp": 1719000000
}
```

**Why:** Dashboard expects `speech_rate` and `acoustic_fatigue`, but sensor schema only has accel/HR/temp.

---

#### 3. Define `ModelFeatures` (Final Model Input)

The contract between you and Shrima:

```python
{
  "movement_score": 0.31,
  "hrv_score": 0.58,
  "speech_rate": 92,
  "acoustic_fatigue": 0.74,
  "shift_duration_hours": 10.5
}
```

This is computed from:
- `SensorFeatures` (extracted from sensor readings)
- `VoiceFeatures` (from Deepgram)

**Deliverable:** Clear contract for model team.

---

### Phase 1 Complete ✓
- [ ] WESAD loads
- [ ] `SensorReading` validates
- [ ] Sample records print
- [ ] `VoiceFeatures` schema defined
- [ ] `ModelFeatures` schema defined

---

## Phase 2: Data Loaders

### Goal
Load data from multiple sources without changing downstream code.

### Tasks

#### 1. Build Loaders

```python
load_wesad(subject_id) -> List[SensorReading]
load_esp32() -> SensorReading
```

Both return the same `SensorReading` objects.

Example:

```python
reading = load_next_reading()
# Works for both WESAD and ESP32
```

---

#### 2. WESAD Label Mapping

WESAD labels are:
- Baseline
- Stress
- Amusement
- Meditation

But your dashboard needs:
- LOW
- MEDIUM
- HIGH

**Add translation layer:**

```python
WESAD_TO_RISK = {
    "baseline": "LOW",
    "stress": "HIGH",
    "amusement": "LOW",
    "meditation": "LOW"
}
```

**Why:** Prevents messy dashboard logic.

---

#### 3. WESAD to Schema Converter

```python
def wesad_to_schema(raw_wesad_record):
    return SensorReading(
        timestamp=...,
        accel_x=...,
        heart_rate=...,
        ...
    )
```

#### 4. ESP32 to Schema Converter

```python
def esp32_to_schema(raw_esp32_payload):
    return SensorReading(...)
```

**Deliverable:** Model never knows where data came from.

---

### Phase 2 Complete ✓
- [ ] `load_wesad()` works
- [ ] `load_esp32()` works
- [ ] Both return `List[SensorReading]`
- [ ] Label mapping tested
- [ ] Schema converters validated

---

## Phase 3: Windowing + Feature Extraction

### Goal
Convert raw sensor streams into model-ready features.

### Tasks

#### 1. Windowing Layer (MOST IMPORTANT)

ML models don't operate on a single reading. They need windows.

**Change:**

```python
# ❌ Old
extract_features(reading)

# ✅ New
extract_features(window)
```

**Example:**

```python
window = last_30_seconds_of_data  # 500 readings

features = extract_features(window)
```

**Input:**
- 500 sensor readings (30 seconds @ ~16Hz sampling)

**Output:**
- Fixed-size analysis window

**Deliverable:** Feature extraction always receives fixed-size windows.

---

#### 2. Feature Extraction Function

```python
def extract_features(window: List[SensorReading]) -> dict:
    """
    Compute statistical features from a time window.
    """
    return {
        "heart_rate_mean": 82,
        "heart_rate_std": 6,
        "motion_mean": 0.32,
        "motion_variance": 0.11,
        "temperature_mean": 98.4
    }
```

Compute:
- Motion magnitude
- Movement variance
- Heart rate mean/std
- Temperature mean
- HRV score

---

#### 3. Merge Sensor + Voice Features

```python
sensor_features = extract_features(window)
voice_features = get_latest_voice_features()

model_features = {
    **sensor_features,
    **voice_features,
    "shift_duration_hours": get_shift_duration()
}
```

**Output:**

```python
{
  "movement_score": 0.31,
  "hrv_score": 0.58,
  "speech_rate": 92,
  "acoustic_fatigue": 0.74,
  "shift_duration_hours": 10.5
}
```

**Deliverable:** Model receives the same feature format regardless of source.

---

### Phase 3 Complete ✓
- [ ] Windowing layer implemented
- [ ] Window size configurable (default: 30 seconds)
- [ ] Feature extraction function works
- [ ] Sensor + voice features merged
- [ ] Output matches `ModelFeatures` schema

---

## Phase 4: Replay/Live Integration + FastAPI

### Goal
Allow switching between WESAD replay and live ESP32 with one config value.

### Tasks

#### 1. Config Toggle

```python
DATA_SOURCE = "wesad"  # or "esp32"
REPLAY_SPEED = 1.0     # 1.0 = real-time, 5.0 = 5x speed
```

**Why replay speed matters:**

Judges won't wait 10 hours to see fatigue drift. Speed up replay for demos.

Examples:
- `1.0` = real-time
- `5.0` = 5x speed
- `10.0` = 10x speed

---

#### 2. Unified Pipeline

```python
if DATA_SOURCE == "wesad":
    reading = load_wesad()
elif DATA_SOURCE == "esp32":
    reading = load_esp32()

window = create_window(reading)
features = extract_features(window)
prediction = model.predict(features)
```

No model changes required.

---

#### 3. FastAPI Integration Endpoint

**Document the full handoff:**

```
model.predict(features)
      ↓
POST /predict
      ↓
dashboard updates
```

**FastAPI Endpoint:**

```python
@app.post("/predict")
async def predict(features: ModelFeatures):
    """
    Receives model features, returns risk prediction.
    """
    result = model.predict(features)
    return {
        "risk_score": result.score,
        "risk_level": result.level,
        "contributors": result.contributors
    }
```

**Deliverable:** Clear handoff between model team and API team.

---

### Phase 4 Complete ✓
- [ ] Toggle between WESAD/ESP32 working
- [ ] Replay speed control implemented
- [ ] Model receives data from both sources
- [ ] FastAPI receives prediction
- [ ] Dashboard displays score
- [ ] End-to-end test passes

---

## Full Pipeline Diagram

```
WESAD Replay          ESP32 Live
    ↓                     ↓
  Loader              Loader
    ↓                     ↓
    └─────→ SensorReading ←─────┘
                ↓
            Windowing (30s)
                ↓
         Feature Extraction
                ↓
       ModelFeatures (contract)
                ↓
           model.predict()
                ↓
          POST /predict
                ↓
            FastAPI
                ↓
           Dashboard

      (with VoiceFeatures merged in)
```

---

## Acceptance Criteria Summary

### Phase 1: Schema + Data Models ✓
- WESAD loads
- `SensorReading` validates
- Sample records print
- `VoiceFeatures` schema defined
- `ModelFeatures` schema defined

### Phase 2: Loaders ✓
- `load_wesad()` works
- `load_esp32()` works
- Both return `List[SensorReading]`
- Label mapping tested
- Schema converters validated

### Phase 3: Windowing + Features ✓
- Windowing layer implemented
- Window size configurable
- Feature extraction works
- Sensor + voice features merged
- Output matches `ModelFeatures` schema

### Phase 4: Integration ✓
- Toggle WESAD/ESP32 working
- Replay speed control implemented
- Model receives data
- FastAPI receives prediction
- Dashboard displays score
- End-to-end test passes

---

## What's Already Done ✅

- ✅ FastAPI backend skeleton
- ✅ Dashboard scaffold (light mode, responsive, charts)
- ✅ Mock data endpoints (`/dashboard`, `/predict`, `/health`, `/history`, `/latest-transcript`)
- ✅ Pydantic response models
- ✅ CORS configured

---

## What's Next

**Priority 1:** Build Phase 1-3 (sensor pipeline)
**Priority 2:** Connect Shrima's model (Phase 4)
**Priority 3:** Add System Status card to dashboard
**Priority 4:** Real-time updates (auto-refresh every 3s)
