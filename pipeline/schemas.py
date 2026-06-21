"""
Data schemas for Shift-Guard sensor pipeline.

This module defines the core data contracts for the entire pipeline:
- SensorReading: Standardized sensor data from WESAD and ESP32
- VoiceFeatures: Voice analysis data from Deepgram
- ModelFeatures: Final model input contract
"""

from pydantic import BaseModel, Field
from typing import Optional


class SensorReading(BaseModel):
    """
    Standard payload for all sensor sources (WESAD + ESP32).

    Ensures both data sources produce identical records for downstream processing.
    """
    timestamp: int = Field(..., description="Unix timestamp in seconds")
    accel_x: float = Field(..., description="Acceleration X-axis (g)")
    accel_y: float = Field(..., description="Acceleration Y-axis (g)")
    accel_z: float = Field(..., description="Acceleration Z-axis (g)")
    heart_rate: int = Field(..., description="Heart rate in BPM")
    temperature: float = Field(..., description="Body temperature in Fahrenheit")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": 1719000000,
                "accel_x": 0.12,
                "accel_y": -0.91,
                "accel_z": 0.08,
                "heart_rate": 82,
                "temperature": 98.2
            }
        }


class VoiceFeatures(BaseModel):
    """
    Voice analysis data from Deepgram.

    Provides acoustic fatigue indicators and speech pattern analysis
    that complement sensor data for fatigue detection.
    """
    transcript: str = Field(..., description="Transcribed speech text")
    speech_rate: int = Field(..., description="Words per minute")
    acoustic_fatigue: float = Field(..., ge=0.0, le=1.0, description="Fatigue score (0-1)")
    timestamp: int = Field(..., description="Unix timestamp in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "transcript": "I'm doing okay, just pushing through.",
                "speech_rate": 92,
                "acoustic_fatigue": 0.74,
                "timestamp": 1719000000
            }
        }


class ModelFeatures(BaseModel):
    """
    Final model input contract - 18 features extracted from WESAD sensors.

    Matches the feature extraction from extract_features.py (30s windows).
    This is the exact schema the trained Random Forest model expects.
    """
    # Accelerometer features (movement + fidgeting)
    acc_mag_mean: float = Field(..., description="Mean of ACC magnitude (orientation-invariant)")
    acc_mag_std: float = Field(..., description="Std dev of ACC magnitude")
    acc_hf_mean: float = Field(..., description="Mean high-frequency ACC intensity (fidgeting)")

    # Blood volume pulse features
    bvp_mean: float = Field(..., description="Mean BVP amplitude")
    bvp_std: float = Field(..., description="Std dev of BVP amplitude")

    # Heart rate features (derived from BVP)
    hr_mean: float = Field(..., description="Mean heart rate (BPM)")
    hr_std: float = Field(..., description="Std dev of heart rate (HRV proxy)")
    hr_slope: float = Field(..., description="Linear trend in heart rate")
    hr_min: float = Field(..., description="Minimum heart rate in window")
    hr_max: float = Field(..., description="Maximum heart rate in window")

    # Temperature features
    temp_mean: float = Field(..., description="Mean body temperature")
    temp_slope: float = Field(..., description="Temperature drift (trend)")
    temp_delta: float = Field(..., description="Net temperature change across window")

    # Electrodermal activity features (skin conductance - stress marker)
    eda_mean: float = Field(..., description="Mean skin conductance level")
    eda_std: float = Field(..., description="Std dev of skin conductance")
    eda_slope: float = Field(..., description="Skin conductance rising trend")
    eda_min: float = Field(..., description="Minimum skin conductance")
    eda_max: float = Field(..., description="Maximum skin conductance")

    class Config:
        json_schema_extra = {
            "example": {
                "acc_mag_mean": 1.02,
                "acc_mag_std": 0.15,
                "acc_hf_mean": 0.08,
                "bvp_mean": -45.2,
                "bvp_std": 120.5,
                "hr_mean": 82.3,
                "hr_std": 6.8,
                "hr_slope": 0.02,
                "hr_min": 72.0,
                "hr_max": 95.0,
                "temp_mean": 32.1,
                "temp_slope": 0.001,
                "temp_delta": 0.03,
                "eda_mean": 2.45,
                "eda_std": 0.82,
                "eda_slope": 0.015,
                "eda_min": 1.8,
                "eda_max": 4.2
            }
        }
