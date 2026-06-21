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
    Final model input contract - computed from SensorFeatures + VoiceFeatures.

    This is the handoff point between data pipeline and ML model.
    All features must be present for model inference.
    """
    movement_score: float = Field(..., ge=0.0, le=1.0, description="Derived from accelerometer data")
    hrv_score: float = Field(..., ge=0.0, le=1.0, description="Heart rate variability score")
    speech_rate: int = Field(..., description="Words per minute from voice analysis")
    acoustic_fatigue: float = Field(..., ge=0.0, le=1.0, description="Fatigue score from voice")
    shift_duration_hours: float = Field(..., ge=0.0, description="Hours elapsed in current shift")

    class Config:
        json_schema_extra = {
            "example": {
                "movement_score": 0.31,
                "hrv_score": 0.58,
                "speech_rate": 92,
                "acoustic_fatigue": 0.74,
                "shift_duration_hours": 10.5
            }
        }
