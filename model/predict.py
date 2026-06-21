"""
Model prediction interface for Shift-Guard.

Wraps the trained Random Forest model and provides simple prediction API.
"""

import joblib
import numpy as np
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from pipeline.schemas import ModelFeatures

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Arize imports
try:
    from arize import ArizeClient
    ARIZE_AVAILABLE = True
except ImportError:
    ARIZE_AVAILABLE = False
    print("[WARN] Arize SDK not installed - monitoring disabled")

# Path to trained model
MODEL_PATH = Path(__file__).parent / "outputs" / "rf_baseline.joblib"

# Load model once at module import
_model = None

# Initialize Arize client
arize_client = None
ARIZE_SPACE_ID = None
ARIZE_PROJECT_NAME = "shift-guard"

if ARIZE_AVAILABLE:
    ARIZE_API_KEY = os.getenv("ARIZE_API_KEY", "")
    ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID", "")  # Note: Changed from SPACE_KEY to SPACE_ID

    if ARIZE_API_KEY and ARIZE_SPACE_ID:
        try:
            arize_client = ArizeClient(api_key=ARIZE_API_KEY)
            print("[OK] Arize monitoring initialized")
        except Exception as e:
            print(f"[WARN] Arize initialization failed: {e}")
    else:
        print("[WARN] Arize API keys not found - monitoring disabled (set ARIZE_API_KEY and ARIZE_SPACE_ID)")


def load_model():
    """Load the trained Random Forest model."""
    global _model
    if _model is None:
        model_data = joblib.load(MODEL_PATH)
        _model = model_data['model']  # Extract model from dict
        print(f"[OK] Loaded model from {MODEL_PATH}")
        print(f"     Features: {len(model_data['feature_cols'])}")
        print(f"     Window: {model_data['window_s']}s with {model_data['overlap']*100:.0f}% overlap")
    return _model


def predict_stress(features: ModelFeatures) -> dict:
    """
    Predict stress level from 18-feature input.

    Args:
        features: ModelFeatures object with 18 sensor features

    Returns:
        dict with:
            - prediction: int (0=calm, 1=stressed)
            - risk_level: str ("LOW" or "HIGH")
            - probability: float (confidence 0.0-1.0)
    """
    model = load_model()

    # Convert Pydantic model to feature array in correct order
    feature_array = np.array([[
        features.acc_mag_mean,
        features.acc_mag_std,
        features.acc_hf_mean,
        features.bvp_mean,
        features.bvp_std,
        features.hr_mean,
        features.hr_std,
        features.hr_slope,
        features.hr_min,
        features.hr_max,
        features.temp_mean,
        features.temp_slope,
        features.temp_delta,
        features.eda_mean,
        features.eda_std,
        features.eda_slope,
        features.eda_min,
        features.eda_max,
    ]])

    # Predict
    prediction = int(model.predict(feature_array)[0])
    probabilities = model.predict_proba(feature_array)[0]
    confidence = float(probabilities[prediction])

    # Map to risk levels
    risk_level = "LOW" if prediction == 0 else "HIGH"

    # Log to Arize
    if arize_client is not None and ARIZE_SPACE_ID is not None:
        try:
            # Create prediction record as DataFrame
            now = datetime.now()
            log_data = pd.DataFrame({
                "span_id": [now.isoformat()],
                "start_time": [now],  # Required by Arize spans API
                "end_time": [now],    # Required by Arize spans API
                # Input features (18 sensor inputs)
                "input.acc_mag_mean": [features.acc_mag_mean],
                "input.acc_mag_std": [features.acc_mag_std],
                "input.acc_hf_mean": [features.acc_hf_mean],
                "input.bvp_mean": [features.bvp_mean],
                "input.bvp_std": [features.bvp_std],
                "input.hr_mean": [features.hr_mean],
                "input.hr_std": [features.hr_std],
                "input.hr_slope": [features.hr_slope],
                "input.hr_min": [features.hr_min],
                "input.hr_max": [features.hr_max],
                "input.temp_mean": [features.temp_mean],
                "input.temp_slope": [features.temp_slope],
                "input.temp_delta": [features.temp_delta],
                "input.eda_mean": [features.eda_mean],
                "input.eda_std": [features.eda_std],
                "input.eda_slope": [features.eda_slope],
                "input.eda_min": [features.eda_min],
                "input.eda_max": [features.eda_max],
                # Output/prediction
                "output.prediction": [prediction],
                "output.risk_level": [risk_level],
                "output.confidence": [confidence],
                "output.calm_probability": [float(probabilities[0])],
                "output.stressed_probability": [float(probabilities[1])],
                # Model metadata
                "attributes.llm.model_name": ["shift-guard-rf-baseline"],
                "attributes.model_version": ["1.0"],
            })

            # Log to Arize
            response = arize_client.spans.log(
                space_id=ARIZE_SPACE_ID,
                project_name=ARIZE_PROJECT_NAME,
                dataframe=log_data,
                validate=False  # Skip validation for faster logging
            )

            if response.status_code == 200:
                print(f"[ARIZE] Logged prediction: {risk_level} (confidence={confidence:.2f})")
            else:
                print(f"[ARIZE] Log failed: HTTP {response.status_code}")

        except Exception as e:
            # Don't fail predictions if Arize logging fails
            print(f"[ARIZE ERROR] {e}")

    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "probability": confidence,
        "calm_probability": float(probabilities[0]),
        "stressed_probability": float(probabilities[1]),
    }


def predict_batch(features_list: list[ModelFeatures]) -> list[dict]:
    """Predict stress for multiple feature sets."""
    return [predict_stress(f) for f in features_list]


if __name__ == "__main__":
    # Test the model with example features
    from pipeline.schemas import ModelFeatures

    test_features = ModelFeatures(
        acc_mag_mean=1.02,
        acc_mag_std=0.15,
        acc_hf_mean=0.08,
        bvp_mean=-45.2,
        bvp_std=120.5,
        hr_mean=82.3,
        hr_std=6.8,
        hr_slope=0.02,
        hr_min=72.0,
        hr_max=95.0,
        temp_mean=32.1,
        temp_slope=0.001,
        temp_delta=0.03,
        eda_mean=2.45,
        eda_std=0.82,
        eda_slope=0.015,
        eda_min=1.8,
        eda_max=4.2,
    )

    result = predict_stress(test_features)
    print("\nTest Prediction:")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['probability']:.1%}")
    print(f"  Calm prob: {result['calm_probability']:.1%}")
    print(f"  Stressed prob: {result['stressed_probability']:.1%}")
