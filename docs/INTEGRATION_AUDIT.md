# Burnout Detection System - Integration Audit Report

**Generated:** 2026-06-20
**Project:** Burnout Detection Dashboard + Sensor Pipeline
**Scope:** FastAPI Backend ↔ Model ↔ Sensor Pipeline ↔ Dashboard

---

## 1. Who Calls Who Across Boundaries

### Dashboard → Backend API

| Caller (Dashboard) | Calls | Callee (FastAPI) | Contract | Status |
|-------------------|-------|------------------|----------|---------|
| `fetchDashboard()` | GET | `/dashboard` | `DashboardResponse` | ✅ Works (mock) |
| `fetchHistory()` | GET | `/history` | `HistoryResponse` | ✅ Works (mock) |
| `fetchTranscript()` | GET | `/latest-transcript` | `TranscriptResponse` | ✅ Works (mock) |
| Refresh button | POST | `/predict` | `PredictResponse` | ⚠️ Mock only |

### Backend API → Model

| Caller (FastAPI) | Calls | Callee (Model) | Contract | Status |
|-----------------|-------|----------------|----------|---------|
| `/predict` endpoint | - | `model.predict()` | `ModelFeatures` → `PredictResponse` | ❌ Not connected |
| Feature extractor | - | Model inference | `ModelFeatures` input | ❌ No model loaded |

### Sensor Pipeline → Backend API

| Caller (Sensors) | Calls | Callee (FastAPI) | Contract | Status |
|-----------------|-------|------------------|----------|---------|
| WESAD loader | - | *(none - should POST to endpoint)* | `SensorReading` | ❌ No endpoint |
| ESP32 device | - | *(none - should POST to endpoint)* | `SensorReading` | ❌ No endpoint |
| Deepgram | - | *(none - should POST to endpoint)* | `VoiceFeatures` | ❌ No endpoint |

### Backend API → External Services

| Caller (FastAPI) | Calls | Callee (External) | Contract | Status |
|-----------------|-------|-------------------|----------|---------|
| Voice pipeline | POST | Deepgram API `/transcribe` | Audio → Transcript | ❌ Not implemented |
| *(missing)* | - | Deepgram TTS | Text → Audio nudges | ❌ Not implemented |

---

## 2. Supported Workflows (End-to-End Status)

### Workflow 1: Mock Dashboard Display ✅ WORKS

**Flow:**
```
User clicks Refresh
    ↓
Dashboard calls GET /dashboard
    ↓
FastAPI generates random mock data
    ↓
Returns DashboardResponse
    ↓
Dashboard renders risk score, contributors, chart
```

**Status:** ✅ **Fully functional**

**Evidence:**
- `/dashboard` returns valid JSON ✅
- Dashboard displays all components ✅
- Chart syncs with current risk score ✅
- Manual refresh works ✅

**Gaps:**
- No auto-refresh (manual button only)
- System Status card missing
- All data is random (not from real sensors/model)

---

### Workflow 2: WESAD Replay → Model → Dashboard ❌ BROKEN

**Flow:**
```
load_wesad(subject_id)
    ↓
wesad_to_schema() → SensorReading
    ↓
create_window(readings) → 30-second window
    ↓
extract_features(window) → ModelFeatures
    ↓
model.predict(features) → PredictResponse
    ↓
POST /predict (internal) → FastAPI
    ↓
Dashboard polls GET /dashboard
    ↓
Display updated risk score
```

**Status:** ❌ **Completely broken**

**Blockers:**
1. **WESAD loader doesn't exist** (Phase 2, Task 1)
2. **No windowing logic** (Phase 3, Task 1)
3. **No feature extraction** (Phase 3, Task 2)
4. **No model loaded** (Phase 4, Task 2)
5. **No `/sensor-data` POST endpoint** for sensors to push data

**What Works:**
- Nothing in this flow works yet

**What's Defined:**
- `SensorReading` schema exists in README ✅
- `ModelFeatures` schema exists in README ✅

---

### Workflow 3: Live ESP32 → Model → Dashboard ❌ BROKEN

**Flow:**
```
ESP32 reads sensors (accel, HR, temp)
    ↓
HTTP POST to /sensor-data
    ↓
esp32_to_schema() → SensorReading
    ↓
Add to sliding window buffer
    ↓
extract_features(window) → ModelFeatures
    ↓
model.predict(features) → PredictResponse
    ↓
Update /dashboard state
    ↓
Dashboard auto-refreshes (SSE or polling)
    ↓
Display live risk score
```

**Status:** ❌ **Completely broken**

**Blockers:**
1. **No `/sensor-data` POST endpoint**
2. **No ESP32 communication protocol defined**
3. **No windowing buffer**
4. **No feature extraction**
5. **No model integration**
6. **No auto-refresh in dashboard**

**What Works:**
- Nothing

---

### Workflow 4: Voice Analysis → Dashboard ❌ BROKEN

**Flow:**
```
Microphone audio (or replay file)
    ↓
Audio chunk (30s WAV)
    ↓
POST to Deepgram /v1/audio/transcriptions
    ↓
Receive transcript + timing
    ↓
Compute acoustic features (pitch, energy, pauses)
    ↓
VoiceFeatures (transcript, speech_rate, acoustic_fatigue)
    ↓
Merge with sensor features → ModelFeatures
    ↓
model.predict() → Risk score
    ↓
Dashboard displays transcript + fatigue
```

**Status:** ❌ **Completely broken**

**Blockers:**
1. **No audio capture pipeline**
2. **No Deepgram API integration**
3. **No acoustic feature extraction** (Deepgram only gives transcript, not fatigue)
4. **No `/voice-data` POST endpoint**
5. **No sensor-voice merge logic**

**Critical Gap:**
- `acoustic_fatigue` in `VoiceFeatures` schema has no implementation plan
- Deepgram provides transcripts, not audio features
- You need a separate audio processing pipeline (librosa, parselmouth, or similar)

---

## 3. Contract Breakage Table

### Field-Level Mismatches

| Field | Schema Expects | Current Implementation | Impact | Owner |
|-------|---------------|------------------------|--------|-------|
| `ModelFeatures.movement_score` | float (0-1) from windowed accel data | ❌ Not computed | HIGH - Model can't predict | Sensor team |
| `ModelFeatures.hrv_score` | float (0-1) from HR variability | ❌ Not computed | HIGH - Model can't predict | Sensor team |
| `ModelFeatures.speech_rate` | int (WPM) from Deepgram | ❌ Random mock data | HIGH - Voice input broken | Voice team |
| `ModelFeatures.acoustic_fatigue` | float (0-1) from audio features | ❌ Random mock data | HIGH - No real fatigue detection | Voice team |
| `ModelFeatures.shift_duration_hours` | float from session tracker | ❌ Not tracked | MEDIUM - Missing time context | Backend team |
| `DashboardResponse.contributors` | Breakdown from model | ✅ Mock data works | LOW - Demo works, not real | Model team |
| `SensorReading` (actual data) | Streaming from ESP32 or WESAD | ❌ No data source | CRITICAL - No input pipeline | Sensor team |
| `VoiceFeatures` (actual data) | From Deepgram + audio analysis | ❌ No pipeline | CRITICAL - Voice missing | Voice team |

### Endpoint Contract Breaks

| Endpoint | Expected Behavior | Current Behavior | Impact |
|----------|------------------|------------------|--------|
| `POST /predict` | Receives `ModelFeatures`, returns model prediction | Returns random mock data | HIGH - No real predictions |
| `POST /sensor-data` | Receives `SensorReading`, adds to window buffer | ❌ **Endpoint doesn't exist** | CRITICAL - Can't ingest sensors |
| `POST /voice-data` | Receives `VoiceFeatures`, updates state | ❌ **Endpoint doesn't exist** | CRITICAL - Can't ingest voice |
| `GET /dashboard` | Returns current model prediction | Returns random data | MEDIUM - Demo works but fake |
| `GET /history` | Returns stored risk timeline | Generates fake progression | LOW - Demo works |

---

## 4. Source Data Problems

### Mock Data Issues

**Problem 1: No Real Sensor Data**
- **Schema defined:** ✅ `SensorReading` in README
- **Data source:** ❌ None
- **WESAD loader:** ❌ Not implemented
- **ESP32 connection:** ❌ Not implemented
- **Impact:** Cannot test feature extraction or model integration

**Problem 2: No Real Voice Data**
- **Schema defined:** ✅ `VoiceFeatures` in README
- **Deepgram integration:** ❌ Not implemented
- **Audio pipeline:** ❌ Not implemented
- **Acoustic feature extraction:** ❌ Not implemented
- **Impact:** `acoustic_fatigue` is random, no real voice analysis

**Problem 3: No Historical Data Persistence**
- **Current:** `/history` generates fake 8-hour progression on every request
- **Problem:** No actual storage, chart is inconsistent across refreshes
- **Impact:** Can't track real fatigue trends over time

**Problem 4: No Session/Shift Tracking**
- **Schema expects:** `shift_duration_hours` in `ModelFeatures`
- **Current:** No session start time, no duration tracker
- **Impact:** Model missing critical time-based context

**Problem 5: Contributors Breakdown Not Real**
- **Dashboard displays:** `Voice Fatigue +35`, `Movement Drift +20`, etc.
- **Source:** Random mock data in `generate_contributors()`
- **Problem:** Not computed from actual model explainability (SHAP, LIME, etc.)
- **Impact:** Misleading "why" explanation to users

### Missing Files & Dependencies

| Required | Status | Path | Impact |
|----------|--------|------|--------|
| WESAD dataset | ❌ Missing | `/data/WESAD/S2.pkl` etc. | Can't test replay mode |
| Model checkpoint | ❌ Missing | `/models/burnout_model.pkl` | Can't make predictions |
| Deepgram API key | ❌ Not configured | `.env` | Can't transcribe audio |
| Audio sample files | ❌ Missing | `/data/audio/` | Can't test voice pipeline |
| ESP32 firmware | ❌ Missing | - | Can't test live sensors |

### Schema Drift Risks

| Schema | Defined In | Used By | Sync Status |
|--------|-----------|---------|-------------|
| `SensorReading` | README.md Phase 1 | No code yet | ⚠️ No validation |
| `VoiceFeatures` | README.md Phase 1 | No code yet | ⚠️ No validation |
| `ModelFeatures` | README.md Phase 1 | No code yet | ⚠️ No validation |
| `DashboardResponse` | main.py:37 | dashboard.html | ✅ In sync |
| `PredictResponse` | main.py:44 | No consumer | ⚠️ Unused |

**Risk:** Schemas in README may diverge from actual implementation since they're not enforced by Pydantic models in code yet.

---

## 5. Prioritized Fix List

### Priority 1: CRITICAL - Blocks All Real Functionality

| # | Fix | Module | Estimated Effort | Blocks |
|---|-----|--------|------------------|--------|
| 1.1 | **Create `/sensor-data` POST endpoint** to receive `SensorReading` | FastAPI Backend | 30 min | Workflows 2, 3 |
| 1.2 | **Create `/voice-data` POST endpoint** to receive `VoiceFeatures` | FastAPI Backend | 20 min | Workflow 4 |
| 1.3 | **Implement WESAD loader** (`load_wesad()` with `.pkl` parsing) | Sensor Pipeline | 2 hours | Workflow 2 |
| 1.4 | **Add Pydantic models** for all schemas (SensorReading, VoiceFeatures, ModelFeatures) | Backend | 30 min | All data validation |

**Impact:** Without these, no real data can flow into the system.

---

### Priority 2: HIGH - Blocks Model Integration

| # | Fix | Module | Estimated Effort | Blocks |
|---|-----|--------|------------------|--------|
| 2.1 | **Implement windowing logic** (30s sliding window, circular buffer) | Sensor Pipeline | 2 hours | Feature extraction |
| 2.2 | **Implement `extract_features(window)`** (motion variance, HR mean/std, etc.) | Sensor Pipeline | 3 hours | Model input |
| 2.3 | **Load Shrima's model** at FastAPI startup (model.pkl, inference function) | Backend + Model Team | 1 hour | Real predictions |
| 2.4 | **Connect `/predict` to real model** (replace mock data with model.predict()) | Backend | 30 min | Real predictions |
| 2.5 | **Implement session tracker** (shift_start_time → shift_duration_hours) | Backend | 1 hour | Model context |

**Impact:** Without these, model can't make real predictions.

---

### Priority 3: MEDIUM - Blocks Voice Analysis

| # | Fix | Module | Estimated Effort | Blocks |
|---|-----|--------|------------------|--------|
| 3.1 | **Deepgram API integration** (POST audio, get transcript) | Voice Pipeline | 2 hours | Transcription |
| 3.2 | **Audio feature extraction** (librosa for pitch, energy, pauses → `acoustic_fatigue`) | Voice Pipeline | 4 hours | Fatigue detection |
| 3.3 | **Audio capture pipeline** (mic input or file replay) | Voice Pipeline | 2 hours | Live audio |
| 3.4 | **Sensor-voice merge logic** (timestamp alignment, handle stale voice data) | Backend | 1 hour | Unified features |

**Impact:** Without these, voice-based fatigue detection doesn't work.

---

### Priority 4: MEDIUM - Improves Demo Quality

| # | Fix | Module | Estimated Effort | Blocks |
|---|-----|--------|------------------|--------|
| 4.1 | **Add System Status card to dashboard** (Deepgram, QNX, ESP32 status) | Dashboard | 1 hour | Sponsor visibility |
| 4.2 | **Implement auto-refresh** (SSE or polling every 3s) | Dashboard + Backend | 1.5 hours | Real-time feel |
| 4.3 | **Add Alert Banner** (⚠️ High Fatigue Detected) | Dashboard | 30 min | UX polish |
| 4.4 | **Persist history** (in-memory or SQLite) | Backend | 2 hours | Consistent charts |
| 4.5 | **Replay speed control** (REPLAY_SPEED = 5.0 for fast demos) | Sensor Pipeline | 1 hour | Demo efficiency |

**Impact:** Makes demos more impressive for judges.

---

### Priority 5: LOW - Nice to Have

| # | Fix | Module | Estimated Effort | Blocks |
|---|-----|--------|------------------|--------|
| 5.1 | **ESP32 live connection** (HTTP POST from hardware) | ESP32 Firmware + Backend | 3 hours | Live hardware demo |
| 5.2 | **Voice Nudge Display** (show Deepgram TTS intervention text) | Dashboard | 30 min | TTS visibility |
| 5.3 | **Contributors explainability** (SHAP values from model) | Model Team | 4 hours | Real attribution |
| 5.4 | **Error recovery** (fallback to last known state if sensors drop) | Backend | 2 hours | Robustness |

---

## Summary Statistics

### Integration Health Score: 25/100

**Breakdown:**
- ✅ **Mock Dashboard Works:** 25 points
- ❌ **No Real Data Sources:** -50 points (WESAD, ESP32, Deepgram all missing)
- ❌ **No Model Integration:** -15 points
- ❌ **No Voice Pipeline:** -10 points

### Workflow Coverage

| Workflow | Status | Completion |
|----------|--------|------------|
| Mock Dashboard | ✅ Works | 100% |
| WESAD Replay | ❌ Broken | 0% |
| Live ESP32 | ❌ Broken | 0% |
| Voice Analysis | ❌ Broken | 0% |

### Recommended Sprint Plan

**Week 1: Get One Real Workflow Working**
- Priority 1 fixes (endpoints, WESAD loader, schemas)
- Priority 2.1-2.2 (windowing + feature extraction)
- Goal: WESAD → Features → Mock Model → Dashboard

**Week 2: Model Integration**
- Priority 2.3-2.5 (load model, connect /predict, session tracker)
- Goal: WESAD → Real Model → Dashboard

**Week 3: Voice + Polish**
- Priority 3 fixes (Deepgram, audio features)
- Priority 4.1-4.2 (System Status, auto-refresh)
- Goal: Full demo ready

**If Time:**
- Priority 5 (ESP32, explainability)

---

## Attribution Table

| Module | Owner | Critical Fixes | Total Effort |
|--------|-------|----------------|--------------|
| **Sensor Pipeline** | Shrima (?) | 1.3, 2.1, 2.2, 4.5 | ~8 hours |
| **Backend API** | Abby | 1.1, 1.2, 1.4, 2.4, 2.5, 3.4, 4.4 | ~7 hours |
| **Model Team** | Shrima | 2.3, 5.3 | ~5 hours |
| **Voice Pipeline** | TBD | 3.1, 3.2, 3.3 | ~8 hours |
| **Dashboard** | Abby | 4.1, 4.2, 4.3, 5.2 | ~3.5 hours |
| **ESP32 Firmware** | Hardware Team | 5.1 | ~3 hours |

**Total Estimated Effort:** ~34.5 hours across all modules

---

## Next Steps

1. **Validate this report** with the team
2. **Assign owners** to each Priority 1 fix
3. **Set up WESAD dataset** (download, verify `.pkl` files load)
4. **Create Pydantic models** for all schemas (30 min - quick win)
5. **Implement `/sensor-data` endpoint** (30 min - unblocks testing)
6. **Schedule model handoff meeting** with Shrima (get model.pkl + input/output contract)
