"""
Comprehensive Phase 1 Test Suite

Tests all Phase 1 deliverables with real-world scenarios:
1. SensorReading validation
2. VoiceFeatures validation
3. ModelFeatures validation
4. Edge cases and error handling
5. Data pipeline flow
"""

import sys
from pipeline.schemas import SensorReading, VoiceFeatures, ModelFeatures
from typing import List
import json


class TestResult:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def record(self, name: str, passed: bool, message: str = ""):
        self.tests.append((name, passed, message))
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_summary(self):
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        for name, passed, message in self.tests:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {name}")
            if message and not passed:
                print(f"  → {message}")

        print(f"\nTotal: {self.passed + self.failed} tests")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")

        if self.failed == 0:
            print("\n✅ ALL TESTS PASSED!")
            return True
        else:
            print(f"\n❌ {self.failed} TEST(S) FAILED")
            return False


def test_sensor_reading_valid_data(results: TestResult):
    """Test 1: Valid SensorReading data"""
    print("\n[TEST 1] Valid SensorReading")
    try:
        reading = SensorReading(
            timestamp=1719000000,
            accel_x=0.12,
            accel_y=-0.91,
            accel_z=0.08,
            heart_rate=82,
            temperature=98.2
        )
        print(f"  Created: HR={reading.heart_rate} BPM, Temp={reading.temperature}°F")
        results.record("SensorReading - Valid Data", True)
    except Exception as e:
        print(f"  Error: {e}")
        results.record("SensorReading - Valid Data", False, str(e))


def test_sensor_reading_batch(results: TestResult):
    """Test 2: Create batch of SensorReading objects (simulate WESAD)"""
    print("\n[TEST 2] Batch SensorReading Creation (30 readings)")
    try:
        readings = []
        base_timestamp = 1719000000

        for i in range(30):
            readings.append(SensorReading(
                timestamp=base_timestamp + i,
                accel_x=0.1 + (i * 0.01),
                accel_y=-0.9,
                accel_z=0.08,
                heart_rate=75 + i,
                temperature=98.0 + (i * 0.01)
            ))

        print(f"  Created {len(readings)} sensor readings")
        print(f"  First: HR={readings[0].heart_rate}, Last: HR={readings[-1].heart_rate}")
        results.record("SensorReading - Batch Creation", True)
    except Exception as e:
        print(f"  Error: {e}")
        results.record("SensorReading - Batch Creation", False, str(e))


def test_sensor_reading_json_serialization(results: TestResult):
    """Test 3: SensorReading JSON serialization"""
    print("\n[TEST 3] SensorReading JSON Serialization")
    try:
        reading = SensorReading(
            timestamp=1719000000,
            accel_x=0.12,
            accel_y=-0.91,
            accel_z=0.08,
            heart_rate=82,
            temperature=98.2
        )

        # Serialize to JSON
        json_str = reading.model_dump_json()
        print(f"  JSON: {json_str[:60]}...")

        # Deserialize back
        data = json.loads(json_str)
        reading2 = SensorReading(**data)

        assert reading.heart_rate == reading2.heart_rate
        print("  ✓ Serialization round-trip successful")
        results.record("SensorReading - JSON Serialization", True)
    except Exception as e:
        print(f"  Error: {e}")
        results.record("SensorReading - JSON Serialization", False, str(e))


def test_voice_features_valid(results: TestResult):
    """Test 4: Valid VoiceFeatures"""
    print("\n[TEST 4] Valid VoiceFeatures")
    try:
        voice = VoiceFeatures(
            transcript="I'm doing okay, just pushing through.",
            speech_rate=92,
            acoustic_fatigue=0.74,
            timestamp=1719000000
        )
        print(f"  Speech rate: {voice.speech_rate} WPM")
        print(f"  Fatigue: {voice.acoustic_fatigue}")
        print(f"  Transcript: \"{voice.transcript[:40]}...\"")
        results.record("VoiceFeatures - Valid Data", True)
    except Exception as e:
        print(f"  Error: {e}")
        results.record("VoiceFeatures - Valid Data", False, str(e))


def test_voice_features_edge_cases(results: TestResult):
    """Test 5: VoiceFeatures boundary values"""
    print("\n[TEST 5] VoiceFeatures Boundary Values")

    # Test minimum fatigue
    try:
        voice_low = VoiceFeatures(
            transcript="Feeling great!",
            speech_rate=120,
            acoustic_fatigue=0.0,  # Minimum
            timestamp=1719000000
        )
        print(f"  ✓ Min fatigue (0.0) accepted")

        # Test maximum fatigue
        voice_high = VoiceFeatures(
            transcript="So exhausted...",
            speech_rate=60,
            acoustic_fatigue=1.0,  # Maximum
            timestamp=1719000000
        )
        print(f"  ✓ Max fatigue (1.0) accepted")
        results.record("VoiceFeatures - Boundary Values", True)
    except Exception as e:
        print(f"  Error: {e}")
        results.record("VoiceFeatures - Boundary Values", False, str(e))


def test_voice_features_invalid(results: TestResult):
    """Test 6: VoiceFeatures should reject out-of-bounds values"""
    print("\n[TEST 6] VoiceFeatures Invalid Values (should fail)")

    try:
        # This should FAIL - acoustic_fatigue > 1.0
        voice = VoiceFeatures(
            transcript="Test",
            speech_rate=100,
            acoustic_fatigue=1.5,  # INVALID
            timestamp=1719000000
        )
        print("  ✗ Failed to reject invalid value!")
        results.record("VoiceFeatures - Reject Invalid", False, "Did not reject fatigue > 1.0")
    except Exception as e:
        print(f"  ✓ Correctly rejected: {type(e).__name__}")
        results.record("VoiceFeatures - Reject Invalid", True)


def test_model_features_valid(results: TestResult):
    """Test 7: Valid ModelFeatures"""
    print("\n[TEST 7] Valid ModelFeatures")
    try:
        features = ModelFeatures(
            movement_score=0.31,
            hrv_score=0.58,
            speech_rate=92,
            acoustic_fatigue=0.74,
            shift_duration_hours=10.5
        )
        print(f"  Movement: {features.movement_score}")
        print(f"  HRV: {features.hrv_score}")
        print(f"  Speech: {features.speech_rate} WPM")
        print(f"  Shift: {features.shift_duration_hours} hours")
        results.record("ModelFeatures - Valid Data", True)
    except Exception as e:
        print(f"  Error: {e}")
        results.record("ModelFeatures - Valid Data", False, str(e))


def test_model_features_extreme_shift(results: TestResult):
    """Test 8: ModelFeatures with extreme shift duration"""
    print("\n[TEST 8] ModelFeatures Extreme Shift Duration")
    try:
        # Test 16-hour shift (realistic extreme)
        features = ModelFeatures(
            movement_score=0.85,
            hrv_score=0.25,  # Low HRV = high stress
            speech_rate=70,   # Slower speech
            acoustic_fatigue=0.95,  # Very fatigued
            shift_duration_hours=16.0
        )
        print(f"  ✓ Accepted 16-hour shift")
        print(f"  High fatigue indicators: fatigue={features.acoustic_fatigue}, HRV={features.hrv_score}")
        results.record("ModelFeatures - Extreme Shift", True)
    except Exception as e:
        print(f"  Error: {e}")
        results.record("ModelFeatures - Extreme Shift", False, str(e))


def test_required_fields(results: TestResult):
    """Test 9: Missing required fields should fail"""
    print("\n[TEST 9] Required Fields Validation (should fail)")

    try:
        # Missing heart_rate and temperature
        reading = SensorReading(
            timestamp=1719000000,
            accel_x=0.12,
            accel_y=-0.91,
            accel_z=0.08
        )
        print("  ✗ Failed to catch missing fields!")
        results.record("Required Fields - Validation", False, "Did not reject missing fields")
    except Exception as e:
        print(f"  ✓ Correctly rejected: {type(e).__name__}")
        results.record("Required Fields - Validation", True)


def test_pipeline_integration(results: TestResult):
    """Test 10: Full pipeline integration"""
    print("\n[TEST 10] Full Pipeline Integration")
    try:
        # Step 1: Create sensor window
        sensor_window = []
        for i in range(30):
            sensor_window.append(SensorReading(
                timestamp=1719000000 + i,
                accel_x=0.1,
                accel_y=-0.9,
                accel_z=0.08,
                heart_rate=80 + i,
                temperature=98.2
            ))
        print(f"  ✓ Created {len(sensor_window)} sensor readings")

        # Step 2: Create voice features
        voice = VoiceFeatures(
            transcript="I'm doing okay.",
            speech_rate=92,
            acoustic_fatigue=0.74,
            timestamp=1719000030
        )
        print(f"  ✓ Created voice features")

        # Step 3: Create model input
        model_input = ModelFeatures(
            movement_score=0.45,
            hrv_score=0.60,
            speech_rate=voice.speech_rate,
            acoustic_fatigue=voice.acoustic_fatigue,
            shift_duration_hours=10.5
        )
        print(f"  ✓ Created model features")
        print(f"  Pipeline: {len(sensor_window)} readings → Voice → Model input")

        results.record("Pipeline - Full Integration", True)
    except Exception as e:
        print(f"  Error: {e}")
        results.record("Pipeline - Full Integration", False, str(e))


def test_type_coercion(results: TestResult):
    """Test 11: Type coercion and conversion"""
    print("\n[TEST 11] Type Coercion")
    try:
        # Pydantic should coerce compatible types
        reading = SensorReading(
            timestamp=1719000000,
            accel_x="0.12",  # String → float
            accel_y=-0.91,
            accel_z=0.08,
            heart_rate="82",  # String → int
            temperature=98.2
        )
        assert isinstance(reading.accel_x, float)
        assert isinstance(reading.heart_rate, int)
        print(f"  ✓ Type coercion working: accel_x={type(reading.accel_x).__name__}, heart_rate={type(reading.heart_rate).__name__}")
        results.record("Type Coercion", True)
    except Exception as e:
        print(f"  Error: {e}")
        results.record("Type Coercion", False, str(e))


def test_realistic_scenarios(results: TestResult):
    """Test 12: Realistic fatigue progression scenario"""
    print("\n[TEST 12] Realistic Fatigue Progression Scenario")
    try:
        scenarios = [
            # Start of shift - low fatigue
            ModelFeatures(
                movement_score=0.20,
                hrv_score=0.85,
                speech_rate=110,
                acoustic_fatigue=0.15,
                shift_duration_hours=1.0
            ),
            # Mid shift - moderate fatigue
            ModelFeatures(
                movement_score=0.45,
                hrv_score=0.55,
                speech_rate=95,
                acoustic_fatigue=0.50,
                shift_duration_hours=6.0
            ),
            # End of shift - high fatigue
            ModelFeatures(
                movement_score=0.75,
                hrv_score=0.30,
                speech_rate=75,
                acoustic_fatigue=0.85,
                shift_duration_hours=12.0
            ),
        ]

        print(f"  ✓ Hour 1:  Fatigue={scenarios[0].acoustic_fatigue}, HRV={scenarios[0].hrv_score}")
        print(f"  ✓ Hour 6:  Fatigue={scenarios[1].acoustic_fatigue}, HRV={scenarios[1].hrv_score}")
        print(f"  ✓ Hour 12: Fatigue={scenarios[2].acoustic_fatigue}, HRV={scenarios[2].hrv_score}")
        print(f"  Progression shows increasing fatigue over shift")

        results.record("Realistic Scenarios - Fatigue Progression", True)
    except Exception as e:
        print(f"  Error: {e}")
        results.record("Realistic Scenarios - Fatigue Progression", False, str(e))


def main():
    """Run all Phase 1 tests"""
    print("=" * 70)
    print("PHASE 1 COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print("\nTesting all schemas with real-world scenarios...\n")

    results = TestResult()

    # Run all tests
    test_sensor_reading_valid_data(results)
    test_sensor_reading_batch(results)
    test_sensor_reading_json_serialization(results)
    test_voice_features_valid(results)
    test_voice_features_edge_cases(results)
    test_voice_features_invalid(results)
    test_model_features_valid(results)
    test_model_features_extreme_shift(results)
    test_required_fields(results)
    test_pipeline_integration(results)
    test_type_coercion(results)
    test_realistic_scenarios(results)

    # Print summary
    all_passed = results.print_summary()

    if all_passed:
        print("\n" + "=" * 70)
        print("✅ PHASE 1 VALIDATION COMPLETE")
        print("=" * 70)
        print("\nAll schemas are working correctly!")
        print("Ready to proceed to Phase 2 (Data Loaders)")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
