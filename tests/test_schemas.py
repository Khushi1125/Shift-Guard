"""
Schema validation script for Phase 1.

Tests all three Pydantic models with sample data to ensure:
1. SensorReading validates correctly
2. VoiceFeatures validates correctly
3. ModelFeatures validates correctly
"""

from pipeline.schemas import SensorReading, VoiceFeatures, ModelFeatures
import json


def test_sensor_reading():
    """Test SensorReading schema with sample data"""
    print("=" * 60)
    print("Testing SensorReading Schema")
    print("=" * 60)

    # Sample sensor data
    sensor_data = {
        "timestamp": 1719000000,
        "accel_x": 0.12,
        "accel_y": -0.91,
        "accel_z": 0.08,
        "heart_rate": 82,
        "temperature": 98.2
    }

    try:
        reading = SensorReading(**sensor_data)
        print("✓ SensorReading validated successfully!")
        print(f"\nParsed data:")
        print(json.dumps(reading.model_dump(), indent=2))
        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def test_voice_features():
    """Test VoiceFeatures schema with sample data"""
    print("\n" + "=" * 60)
    print("Testing VoiceFeatures Schema")
    print("=" * 60)

    # Sample voice data
    voice_data = {
        "transcript": "I'm doing okay, just pushing through.",
        "speech_rate": 92,
        "acoustic_fatigue": 0.74,
        "timestamp": 1719000000
    }

    try:
        features = VoiceFeatures(**voice_data)
        print("✓ VoiceFeatures validated successfully!")
        print(f"\nParsed data:")
        print(json.dumps(features.model_dump(), indent=2))
        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def test_model_features():
    """Test ModelFeatures schema with sample data"""
    print("\n" + "=" * 60)
    print("Testing ModelFeatures Schema")
    print("=" * 60)

    # Sample model input features
    model_data = {
        "movement_score": 0.31,
        "hrv_score": 0.58,
        "speech_rate": 92,
        "acoustic_fatigue": 0.74,
        "shift_duration_hours": 10.5
    }

    try:
        features = ModelFeatures(**model_data)
        print("✓ ModelFeatures validated successfully!")
        print(f"\nParsed data:")
        print(json.dumps(features.model_dump(), indent=2))
        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def test_validation_rules():
    """Test schema validation rules (constraints)"""
    print("\n" + "=" * 60)
    print("Testing Validation Rules")
    print("=" * 60)

    # Test acoustic_fatigue bounds (should be 0-1)
    print("\n1. Testing acoustic_fatigue bounds...")
    try:
        VoiceFeatures(
            transcript="Test",
            speech_rate=100,
            acoustic_fatigue=1.5,  # Invalid - exceeds 1.0
            timestamp=1719000000
        )
        print("✗ Failed to catch out-of-bounds value")
    except Exception as e:
        print(f"✓ Correctly rejected invalid value: {type(e).__name__}")

    # Test required fields
    print("\n2. Testing required fields...")
    try:
        SensorReading(timestamp=1719000000)  # Missing required fields
        print("✗ Failed to catch missing required fields")
    except Exception as e:
        print(f"✓ Correctly rejected missing fields: {type(e).__name__}")


def main():
    """Run all validation tests"""
    print("\n🔍 Phase 1 Schema Validation Tests\n")

    results = []
    results.append(("SensorReading", test_sensor_reading()))
    results.append(("VoiceFeatures", test_voice_features()))
    results.append(("ModelFeatures", test_model_features()))
    test_validation_rules()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n✅ All schemas validated successfully!")
        print("\nPhase 1 Checklist Progress:")
        print("✓ SensorReading schema defined")
        print("✓ VoiceFeatures schema defined")
        print("✓ ModelFeatures schema defined")
        print("✓ Sample records validated")
    else:
        print("\n❌ Some validations failed. Please review errors above.")


if __name__ == "__main__":
    main()
