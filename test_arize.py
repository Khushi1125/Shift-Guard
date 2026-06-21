#!/usr/bin/env python3
"""
Quick test script to verify Arize logging is working.
Run this to make 5 predictions and check if they're logged to Arize.
"""

from model.predict import predict_stress
from pipeline.schemas import ModelFeatures
import time

print("=" * 60)
print("ARIZE INTEGRATION TEST")
print("=" * 60)
print()

# Create 5 test predictions with different stress levels
test_cases = [
    {
        "name": "High Stress Example",
        "features": ModelFeatures(
            acc_mag_mean=1.5, acc_mag_std=0.3, acc_hf_mean=0.12,
            bvp_mean=-50.0, bvp_std=150.0,
            hr_mean=110.0, hr_std=12.0, hr_slope=0.05, hr_min=95.0, hr_max=125.0,
            temp_mean=37.5, temp_slope=0.003, temp_delta=0.08,
            eda_mean=3.5, eda_std=1.2, eda_slope=0.025, eda_min=2.0, eda_max=5.5,
        )
    },
    {
        "name": "Low Stress Example",
        "features": ModelFeatures(
            acc_mag_mean=1.0, acc_mag_std=0.1, acc_hf_mean=0.05,
            bvp_mean=-40.0, bvp_std=80.0,
            hr_mean=70.0, hr_std=4.0, hr_slope=0.01, hr_min=65.0, hr_max=75.0,
            temp_mean=32.0, temp_slope=0.001, temp_delta=0.02,
            eda_mean=0.5, eda_std=0.1, eda_slope=0.005, eda_min=0.4, eda_max=0.7,
        )
    },
    {
        "name": "Medium Stress Example",
        "features": ModelFeatures(
            acc_mag_mean=1.2, acc_mag_std=0.2, acc_hf_mean=0.08,
            bvp_mean=-45.0, bvp_std=110.0,
            hr_mean=90.0, hr_std=8.0, hr_slope=0.03, hr_min=80.0, hr_max=100.0,
            temp_mean=34.0, temp_slope=0.002, temp_delta=0.05,
            eda_mean=1.8, eda_std=0.5, eda_slope=0.015, eda_min=1.2, eda_max=2.5,
        )
    },
]

results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"[{i}/{len(test_cases)}] Testing: {test_case['name']}")

    result = predict_stress(test_case['features'])

    results.append({
        "name": test_case['name'],
        "risk": result['risk_level'],
        "confidence": result['probability']
    })

    print(f"    → Prediction: {result['risk_level']} (confidence={result['probability']:.1%})")
    print()

    # Small delay between predictions
    time.sleep(0.5)

print("=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print()

for r in results:
    print(f"  {r['name']:<25} → {r['risk']:<6} ({r['confidence']:.1%})")

print()
print("=" * 60)
print("VERIFICATION STEPS:")
print("=" * 60)
print()
print("1. Check the output above:")
print("   - Look for '[ARIZE] Logged prediction' messages")
print("   - If you see them, Arize is working! ✅")
print()
print("2. Check your Arize dashboard:")
print("   https://app.arize.com/")
print()
print("3. Navigate to your 'shift-guard' project")
print()
print("4. You should see 3 new predictions with timestamps")
print()
print("If you DON'T see '[ARIZE] Logged prediction' messages:")
print("   - Check .env file has ARIZE_API_KEY and ARIZE_SPACE_ID")
print("   - Make sure you're connected to the internet")
print()
