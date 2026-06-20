import joblib
import numpy as np
import os
 
# Load your trained Random Forest model
# Make sure model.pkl is in the same folder as this script
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_baseline.pkl")
 
rf_model = joblib.load(MODEL_PATH)
 
def predict(bpm: float, temp_c: float) -> float:
    """
    Takes heart rate (BPM) and skin temperature (Celsius),
    returns a risk score from your Random Forest model.
    """
    features = np.array([[bpm, temp_c]])
    prediction = rf_model.predict(features)[0]
    return float(prediction)