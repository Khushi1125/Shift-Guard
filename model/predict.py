"""
Model prediction interface for Shift-Guard.

Wraps the trained Random Forest model and provides simple prediction API.
"""

import joblib
import numpy as np
from pathlib import Path
from pipeline.schemas import ModelFeatures

# Path to trained model
MODEL_PATH = Path(__file__).parent / "outputs" / "rf_baseline.joblib"

# Load model once at module import
_model = None


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
