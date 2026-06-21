"""
Example demonstrating how the three schemas work together in the data pipeline.

Shows the flow: SensorReading → Feature Extraction → ModelFeatures (with VoiceFeatures)
"""

from pipeline.schemas import SensorReading, VoiceFeatures, ModelFeatures
from typing import List
import math


def simulate_sensor_stream() -> List[SensorReading]:
    """Simulate a 30-second window of sensor readings (placeholder for WESAD/ESP32 loader)"""
    readings = []
    base_timestamp = 1719000000

    # Simulate 30 readings (1 per second)
    for i in range(30):
        readings.append(SensorReading(
            timestamp=base_timestamp + i,
            accel_x=0.1 + (i * 0.01),  # Gradually increasing motion
            accel_y=-0.9 - (i * 0.005),
            accel_z=0.08,
            heart_rate=75 + i,  # Increasing heart rate
            temperature=98.2 + (i * 0.01)
        ))

    return readings


def extract_sensor_features(window: List[SensorReading]) -> dict:
    """
    Extract features from a window of sensor readings.

    This is a simplified version - real implementation would compute:
    - Motion magnitude from accelerometer
    - HRV from heart rate variability
    - Temperature trends
    """
    # Calculate movement score from accelerometer data
    motion_magnitudes = []
    for reading in window:
        magnitude = math.sqrt(
            reading.accel_x**2 + reading.accel_y**2 + reading.accel_z**2
        )
        motion_magnitudes.append(magnitude)

    movement_score = sum(motion_magnitudes) / len(motion_magnitudes)
    movement_score = min(movement_score, 1.0)  # Normalize to 0-1

    # Calculate HRV score (simplified - real HRV needs proper calculation)
    heart_rates = [r.heart_rate for r in window]
    hr_variance = sum((hr - sum(heart_rates)/len(heart_rates))**2 for hr in heart_rates)
    hr_std = math.sqrt(hr_variance / len(heart_rates))
    hrv_score = min(hr_std / 100.0, 1.0)  # Normalize to 0-1

    return {
        "movement_score": round(movement_score, 2),
        "hrv_score": round(hrv_score, 2)
    }


def get_latest_voice_features() -> VoiceFeatures:
    """Simulate fetching latest voice features from Deepgram (placeholder)"""
    return VoiceFeatures(
        transcript="I'm doing okay, just pushing through.",
        speech_rate=92,
        acoustic_fatigue=0.74,
        timestamp=1719000030
    )


def compute_shift_duration() -> float:
    """Calculate hours elapsed in current shift (placeholder)"""
    return 10.5


def main():
    """Demonstrate the full pipeline flow"""
    print("=" * 70)
    print("SHIFT-GUARD DATA PIPELINE EXAMPLE")
    print("=" * 70)

    # Step 1: Get sensor readings (from WESAD or ESP32)
    print("\n[1] Loading sensor readings (30-second window)...")
    sensor_window = simulate_sensor_stream()
    print(f"✓ Loaded {len(sensor_window)} sensor readings")
    print(f"   First reading: HR={sensor_window[0].heart_rate} BPM, "
          f"Accel=({sensor_window[0].accel_x:.2f}, "
          f"{sensor_window[0].accel_y:.2f}, {sensor_window[0].accel_z:.2f})")

    # Step 2: Extract features from sensor window
    print("\n[2] Extracting sensor features from window...")
    sensor_features = extract_sensor_features(sensor_window)
    print(f"✓ Extracted features: {sensor_features}")

    # Step 3: Get voice features
    print("\n[3] Fetching voice features from Deepgram...")
    voice_features = get_latest_voice_features()
    print(f"✓ Voice features: speech_rate={voice_features.speech_rate} WPM, "
          f"acoustic_fatigue={voice_features.acoustic_fatigue}")

    # Step 4: Compute shift duration
    print("\n[4] Computing shift duration...")
    shift_duration = compute_shift_duration()
    print(f"✓ Shift duration: {shift_duration} hours")

    # Step 5: Combine into ModelFeatures (the contract for ML model)
    print("\n[5] Creating ModelFeatures for model input...")
    model_input = ModelFeatures(
        movement_score=sensor_features["movement_score"],
        hrv_score=sensor_features["hrv_score"],
        speech_rate=voice_features.speech_rate,
        acoustic_fatigue=voice_features.acoustic_fatigue,
        shift_duration_hours=shift_duration
    )
    print(f"✓ Model input created:")
    print(f"   {model_input.model_dump()}")

    # Step 6: Model prediction (placeholder)
    print("\n[6] Sending to model for prediction...")
    print("✓ Ready for model.predict(model_input)")
    print("\n" + "=" * 70)
    print("Pipeline demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
