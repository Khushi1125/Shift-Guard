# Arduino Bridge Integration Audit: New Code vs. Existing API

**Date:** June 20, 2026
**Scope:** Integration between newly merged bridge.py + Arduino sensor code and existing API/dashboard system
**Boundary:** Arduino → bridge.py → model.py → API → dashboard.html

---

## Section 1: Boundary Map — What Bridge Expects from API

| Component | File | What it does |
|-----------|------|--------------|
| Arduino Serial Output | arduino/main.ino:69-73 | Prints "BPM: {bpm}  Temp: {temp_c} C" to Serial every 1s |
| Bridge Serial Reader | model/src/bridge.py:95-101 | Expects CSV format "{bpm},{temp_c}" from Arduino, splits on comma |
| Bridge Model Call | model/src/bridge.py:11, 106 | Calls `from model import predict` → `predict(bpm, temp_c)` expecting 2-param function |
| Bridge Dashboard POST | model/src/bridge.py:22, 115-120 | POSTs to `http://localhost:8000/` with JSON `{bpm, temp_c, risk_score, timestamp}` |
| Bridge Arize POST | model/src/bridge.py:51-64 | POSTs to Arize API with 2-feature data `{bpm, temp_c}` and risk label |
| Model Loader | model/src/model.py:7 | Loads `rf_baseline.pkl` from same directory using joblib |
| API Model Loader | model/predict.py:13, 24 | Loads `rf_baseline.joblib` from outputs/ directory, expects dict format |
| API Dashboard Endpoint | api/main.py:376 | GET /dashboard uses 18-feature ModelFeatures, cycles through WESAD replay |
| API Predict Endpoint | api/main.py:426 | POST /predict expects ModelFeatures (18 features), returns PredictResponse |

**Deployment Wiring Status:** Bridge.py is NOT wired into any deployment artifacts. No docker-compose, systemd service, or startup script exists to run it. The system "exists in code" but won't run in production without manual intervention.

---

## Section 2: Three Workflow Status

### Workflow 1: Arduino Sensor → Dashboard Display

**Expected flow:** Arduino sensors → Serial → bridge.py → model.py → POST /api → dashboard updates

**Arduino side (arduino/main.ino):**
✗ Prints human-readable format `"BPM: 72  Temp: 36.50 C"` (line 69-73)

**Bridge side (model/src/bridge.py):**
✗ Expects CSV format `"72,36.50"` with comma split (line 101)

**API side (api/main.py):**
✗ No POST endpoint at `/` to receive bridge data
✗ Existing GET /dashboard expects 18-feature WESAD data, not 2-feature Arduino data

**Gap:** Serial format mismatch breaks parsing at bridge.py:101. Even if fixed, API has no endpoint to receive the POST. Dashboard polls GET /dashboard which uses WESAD replay, not live Arduino data.

---

### Workflow 2: Real-Time Model Prediction from Arduino

**Expected flow:** BPM + temp_c → model.predict() → risk score → display on Arduino LCD

**Bridge model call (model/src/bridge.py:106):**
✓ Calls `predict(bpm, temp_c)` with 2 parameters
✓ Expects float return value

**New model.py (model/src/model.py:11-18):**
✓ Implements `predict(bpm: float, temp_c: float) -> float`
✓ Loads rf_baseline.pkl and returns float

**Existing model/predict.py:**
✗ Implements `predict_stress(features: ModelFeatures) -> dict`
✗ Expects 18 features (acc, bvp, hr stats, temp stats, eda stats)
✗ Returns dict `{prediction, risk_level, probability}`

**Gap:** Two incompatible model interfaces exist. Bridge.py imports `from model import predict` which resolves to model/predict.py (package __init__), but model/src/model.py is not in the import path. The 2-feature model exists but is never called.

---

### Workflow 3: Arize Logging + CSV Recording

**Expected flow:** Each reading → log to Arize API → append to local CSV

**Bridge Arize logging (model/src/bridge.py:49-66):**
✓ POSTs to Arize with 2-feature data
✓ Has API key in arize.env
✓ Includes error handling (timeout=2s)

**Bridge CSV logging (model/src/bridge.py:70-79):**
✓ Creates CSV with header if missing
✓ Appends {timestamp, bpm, temp_c, risk_score}

**Gap:** End-to-end ✓ — no integration issues found. This workflow is self-contained within bridge.py and doesn't depend on the API. Will work if bridge.py can successfully parse Arduino serial data and call model.predict().

---

## Section 3: Contract Mismatches — Field Bridge Expects vs. What API Provides

| Field Bridge Expects | What API Actually Provides | Effect |
|----------------------|----------------------------|--------|
| POST endpoint at `/` (bridge.py:115) | GET `/` serves dashboard.html (api/main.py:360-363) | Bridge POST fails with 405 Method Not Allowed; readings never reach API |
| `predict(bpm, temp_c)` callable (bridge.py:11, 106) | `predict_stress(features: ModelFeatures)` at model/predict.py:31 | Import resolves to wrong module; TypeError on call with 2 args vs 1 ModelFeatures arg |
| CSV serial format `"72,36.50"` (bridge.py:101) | Arduino prints `"BPM: 72  Temp: 36.50 C"` (main.ino:69-73) | Bridge split(",") fails; ValueError on line 101 |
| `rf_baseline.pkl` in model/src/ (model.py:7) | `rf_baseline.joblib` in model/outputs/ (predict.py:13) | Two different model files; bridge.py model never loads because import fails first |
| Float risk score (bridge.py:106) | Dict `{prediction, risk_level, probability}` (predict.py:69-81) | If bridge.py somehow calls predict_stress, float(dict) raises TypeError |
| Dependencies: pyserial, python-dotenv, requests | requirements.txt has fastapi, uvicorn, pydantic only | ModuleNotFoundError on `import serial` when bridge.py runs |
| Dashboard expects `{bpm, temp_c, risk_score, timestamp}` JSON (bridge.py:115-120) | API record_reading() expects (risk: int, heart_rate: int, temperature: float) (main.py:349-356) | If endpoint existed, parameter names mismatch; API wouldn't know how to extract risk from risk_score |

---

## Section 4: Source Data / Mock Data Problems

### Missing Python Packages

**What's missing:** `pyserial`, `python-dotenv`, `requests`
**Where code expects it:** bridge.py:4-6, 12
**Evidence:** Attempted `import serial` raised `ModuleNotFoundError: No module named 'serial'`
**Available elsewhere:** No — requirements.txt only has ML + API deps (numpy, pandas, fastapi)

### Model File Duplication

**Problem:** Two model files with same name but different formats exist
**Files:**
- `model/src/rf_baseline.pkl` (2703420 bytes, joblib format)
- `model/outputs/rf_baseline.pkl` (2703420 bytes, joblib format)
- `model/outputs/rf_baseline.joblib` (2727129 bytes, joblib dict with metadata)

**Where code expects it:**
- bridge.py → model/src/model.py:7 expects pkl in src/
- api/main.py → model/predict.py:13 expects joblib in outputs/

**Schema drift:** The .joblib file stores a dict `{model, feature_cols, window_s, overlap}` while .pkl stores raw model object. Existing API extracts model via `model_data['model']` (predict.py:24); new model.py loads raw model with `joblib.load()`.

### Serial Format Divergence

**Problem:** Arduino output doesn't match bridge.py parser expectations
**What Arduino writes:** `"BPM: 72  Temp: 36.50 C\n"` (main.ino:69-73)
**What bridge.py reads:** Expects `"72,36.50"` CSV (bridge.py:101: `bpm_str, temp_str = raw.split(",")`)
**Effect:** Every reading raises `ValueError` because no comma exists in the string
**Available elsewhere:** Data IS available, just in wrong format — regex parse would work but isn't implemented

---

## Section 5: Priority Fix List

### 1. Create POST /live-reading endpoint in API to receive Arduino data
**Where:** api/main.py (add new endpoint around line 425, before /predict)
**Why Priority 1:** Without this, bridge.py cannot send data to the dashboard. This is the missing link that breaks the entire Arduino → dashboard flow. User sees stale WESAD replay data instead of live sensor readings.

### 2. Fix Arduino serial output format to match bridge.py parser
**Where:** arduino/main.ino:69-73 (change print format from "BPM: X Temp: Y C" to "X,Y")
**Why Priority 1:** Bridge.py crashes on every reading with ValueError. No data flows through the pipeline. Alternative: fix bridge.py parser to handle current format, but Arduino is easier to change (1 line vs regex).

### 3. Add bridge.py dependencies to requirements.txt
**Where:** requirements.txt (append pyserial==3.5, python-dotenv==1.0.0, requests==2.31.0)
**Why Priority 1:** Bridge.py won't run at all without these. Import failure is immediate on startup.

### 4. Resolve model import path conflict
**Where:** model/src/model.py needs to be importable as `from model import predict`
**Options:**
  a) Add model/src/ to sys.path in bridge.py
  b) Move bridge.py to project root so relative imports work
  c) Add `from model.src.model import predict` in bridge.py
**Why Priority 2:** Current import resolves to wrong module. Even with endpoint + serial fixes, prediction will fail with TypeError.

### 5. Update API record_reading() to accept bridge.py JSON schema
**Where:** api/main.py:349-356 (modify to extract from `{bpm, temp_c, risk_score, timestamp}`)
**Why Priority 2:** POST endpoint from fix #1 needs to map bridge.py's field names to API's internal representation. Currently expects different param names (risk vs risk_score, heart_rate vs bpm).

### 6. Standardize model file format or clarify which file each module uses
**Where:** Either unify on .joblib dict format, or document that bridge uses .pkl and API uses .joblib
**Why Priority 3:** Both systems work independently with their respective files, but duplication causes confusion. If both load same file, one will break (dict vs raw model).

### 7. Create startup script or systemd service for bridge.py
**Where:** Add scripts/start_bridge.sh or docker-compose service entry
**Why Priority 3:** Bridge.py must run as a background process to continuously read Arduino serial. Currently requires manual `python model/src/bridge.py` in a terminal.

### 8. Wire Arduino port detection or make configurable
**Where:** bridge.py:84 hardcodes `/dev/cu.usbserial-0001`, but find_arduino_port() exists (unused)
**Why Priority 4:** Works on current machine but breaks on different Arduino or USB port. Minor convenience issue.

### 9. Add health check endpoint for bridge.py connectivity status
**Where:** api/main.py (new GET /bridge-status endpoint)
**Why Priority 4:** Dashboard System Status panel shows "ESP32" as warning (main.py:602) but has no actual check. Would improve observability.

### 10. Update dashboard.html to show live/replay mode indicator
**Where:** frontend/dashboard.html (add badge showing data source)
**Why Priority 5:** User can't tell if they're seeing live Arduino data or WESAD replay. Pure UX improvement, doesn't affect functionality.

---

**End of Integration Audit**
