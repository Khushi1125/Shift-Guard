"""
test_full_integration.py — Full pipeline integration test for Shift-Guard.

Runs both a "worst case" (stressed) and "best case" (calm) scenario through
the complete four-function chain and prints a side-by-side summary for a
quick visual sanity check before handoff.

Usage
-----
    cd /path/to/Shift-Guard
    HF_HOME=.hf_cache python tests/test_full_integration.py

Expected output
---------------
  • Worst case final_score meaningfully higher than best case.
  • Worst case triggers the stress intervention; best case does not.
  • No exceptions — all failures degrade to neutral defaults.
"""

import json
import os
import pathlib
import sys

# Make src/ importable from the tests/ directory
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from final_scoring import (
    check_and_get_intervention,
    compute_final_risk_score,
    predict_sensor_score,
    run_voice_checkin,
)

# ─────────────────────────────────────────────────────────────────────────────
# Test scenarios
# ─────────────────────────────────────────────────────────────────────────────

# Sensor values from real WESAD windows (confirmed ONNX predictions in brackets):
#   worst: subject S2, t=2280 s, label=stressed  → model predicts P(stress)=0.90
#   best:  subject S13, t=1095 s, label=calm     → model predicts P(stress)=0.49
#
# Feature order in predict_sensor_score: temp_mean, temp_slope, bpm_mean, bpm_std
SCENARIOS = [
    {
        "label":       "WORST CASE — stressed",
        "sensor":      {"temp_mean": 34.08, "temp_slope": 7.78e-06, "bpm_mean": 94.02, "bpm_std": 5.93},
        "wav_path":    str(ROOT / "tests" / "Stressed.wav"),
        "expect_trigger": True,
    },
    {
        "label":       "BEST CASE  — calm",
        "sensor":      {"temp_mean": 34.88, "temp_slope": 8.28e-05, "bpm_mean": 92.04, "bpm_std": 2.90},
        "wav_path":    str(ROOT / "tests" / "chill_3.wav"),
        "expect_trigger": False,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 30) -> str:
    """Render a simple ASCII progress bar for a [0, 1] value."""
    filled = round(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:.2f}"


def _divider(char: str = "─", width: int = 62) -> str:
    return char * width


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(scenario: dict) -> dict:
    """Run the full four-step pipeline for one scenario and return all results."""
    s = scenario["sensor"]

    # Step 1: sensor score
    sensor_proba = predict_sensor_score(
        temp_mean  = s["temp_mean"],
        temp_slope = s["temp_slope"],
        bpm_mean   = s["bpm_mean"],
        bpm_std    = s["bpm_std"],
    )

    # Step 2: voice check-in
    voice = run_voice_checkin(scenario["wav_path"])

    # Step 3: final blend
    risk = compute_final_risk_score(sensor_proba, voice["combined_voice"])

    # Step 4: intervention decision
    alert = check_and_get_intervention(risk["final_score"])

    return {**voice, **risk, "alert": alert}


def print_scenario(scenario: dict, result: dict) -> None:
    print(_divider("═"))
    print(f"  {scenario['label']}")
    print(_divider("═"))

    sensor_proba   = result["sensor_proba"]
    combined_voice = result["combined_voice"]
    final_score    = result["final_score"]
    alert          = result["alert"]

    # ── Voice channel ──────────────────────────────────────────────────────
    print(f"\n  Voice check-in")
    print(_divider())
    transcript = result["transcript"] or "(empty — Deepgram key not set or API error)"
    # Wrap long transcripts at 55 chars
    words = transcript.split()
    line, lines = "", []
    for word in words:
        if len(line) + len(word) + 1 > 55:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        lines.append(line)
    for i, l in enumerate(lines):
        prefix = "  Transcript  : " if i == 0 else "               "
        print(f"{prefix}{l}")

    print(f"  semantic_score  : {result['semantic_score']:+.4f}  "
          f"(-1=negative  0=neutral  +1=positive)")
    print(f"  tone_stress     : {_bar(result['tone_stress'])}")
    print(f"  combined_voice  : {_bar(combined_voice)}")

    # ── Sensor channel ─────────────────────────────────────────────────────
    print(f"\n  Sensor model")
    print(_divider())
    print(f"  sensor_proba    : {_bar(sensor_proba)}")

    # ── Final score ────────────────────────────────────────────────────────
    print(f"\n  Final risk score")
    print(_divider())
    print(f"  final_score     : {_bar(final_score)}")
    level = (
        "HIGH RISK  ⚠" if final_score >= 0.80 else
        "ELEVATED   ⚠" if final_score >= 0.60 else
        "BORDERLINE"   if final_score >= 0.40 else
        "LOW RISK   ✓"
    )
    print(f"  risk level      : {level}")

    # ── Intervention ───────────────────────────────────────────────────────
    print(f"\n  Intervention")
    print(_divider())
    triggered = alert["triggered"]
    print(f"  triggered       : {'YES ⚠' if triggered else 'NO  ✓'}")
    if triggered and alert["text"]:
        words2 = alert["text"].split()
        line2, lines2 = "", []
        for word in words2:
            if len(line2) + len(word) + 1 > 52:
                lines2.append(line2)
                line2 = word
            else:
                line2 = (line2 + " " + word).strip()
        if line2:
            lines2.append(line2)
        for i, l in enumerate(lines2):
            prefix = "  text          : " if i == 0 else "                  "
            print(f"{prefix}{l}")
    print()


def main():
    print()
    print("  Shift-Guard — Full Integration Test")
    print("  Four-function pipeline end-to-end sanity check")
    print()

    results = []
    for scenario in SCENARIOS:
        print(f"  Running: {scenario['label']} ...")
        r = run_scenario(scenario)
        results.append((scenario, r))

    print()
    for scenario, result in results:
        print_scenario(scenario, result)

    # ── Assertions ────────────────────────────────────────────────────────
    worst_score = results[0][1]["final_score"]
    best_score  = results[1][1]["final_score"]
    worst_alert = results[0][1]["alert"]["triggered"]
    best_alert  = results[1][1]["alert"]["triggered"]

    print(_divider("═"))
    print("  SANITY CHECKS")
    print(_divider("═"))
    print()

    checks = [
        (
            "worst_case final_score > best_case final_score",
            worst_score > best_score,
            f"{worst_score:.4f} > {best_score:.4f}",
        ),
        (
            "worst_case triggers intervention",
            worst_alert is True,
            f"triggered={worst_alert}",
        ),
        (
            "best_case does NOT trigger intervention",
            best_alert is False,
            f"triggered={best_alert}",
        ),
    ]

    all_pass = True
    for desc, passed, detail in checks:
        mark = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {mark}  {desc}")
        print(f"         ({detail})")
        print()
        if not passed:
            all_pass = False

    print(_divider("═"))
    if all_pass:
        print("  ALL CHECKS PASSED — pipeline is ready for handoff.")
    else:
        print("  SOME CHECKS FAILED — review scores above before handoff.")
    print(_divider("═"))
    print()

    return 0 if all_pass else 1


def test_cadence_independence():
    """Verify that sensor and voice updates combine correctly on independent clocks.

    Simulates the real runtime scenario without live hardware or API calls:

    Step 1 — Sensor updates 3× with no voice click.
              Final score should change each time; voice stays at default (0.5).

    Step 2 — Voice click happens once.
              Final score should immediately use new voice + LATEST sensor value.

    Step 3 — Sensor updates again after the voice click.
              Should combine with the voice value from Step 2, NOT the default.
    """
    import final_scoring as fs
    from unittest.mock import patch

    # ── Reset module state so this test is isolated ────────────────────────
    fs._state["latest_sensor_proba"] = 0.5
    fs._state["latest_voice_result"] = {"combined_voice": 0.5}
    fs._score_update_callback = None

    # ── Register a callback to verify dashboard pushes happen ──────────────
    push_log = []   # every push recorded here
    fs.register_score_callback(lambda result: push_log.append(result))

    print()
    print(_divider("═"))
    print("  CADENCE INDEPENDENCE TEST")
    print("  Verifies sensor + voice combine on independent clocks")
    print(_divider("═"))

    # ── Step 1: Three sensor updates, no voice click ───────────────────────
    print()
    print("  Step 1 — sensor updates 3×, no voice click")
    print(_divider())

    sensor_side_effects = [0.55, 0.65, 0.75]
    voice_default = 0.5

    with patch.object(fs, "predict_sensor_score", side_effect=sensor_side_effects):
        r1 = fs.on_sensor_update(34.0, 0.0, 80.0, 4.0)
        r2 = fs.on_sensor_update(34.5, 0.0, 90.0, 5.0)
        r3 = fs.on_sensor_update(35.0, 0.0, 100.0, 6.0)

    checks_1 = [
        ("r1 combined_voice == 0.5 (default, no voice click yet)",
         r1["combined_voice"] == voice_default,
         f"combined_voice={r1['combined_voice']}"),
        ("r2 combined_voice == 0.5 (still no voice click)",
         r2["combined_voice"] == voice_default,
         f"combined_voice={r2['combined_voice']}"),
        ("r3 combined_voice == 0.5 (still no voice click)",
         r3["combined_voice"] == voice_default,
         f"combined_voice={r3['combined_voice']}"),
        ("final_score increases as sensor_proba rises",
         r1["final_score"] < r2["final_score"] < r3["final_score"],
         f"{r1['final_score']:.4f} < {r2['final_score']:.4f} < {r3['final_score']:.4f}"),
        ("dashboard callback fired 3× (one push per sensor update)",
         len(push_log) == 3,
         f"push_log has {len(push_log)} entries"),
    ]

    all_pass = True
    for desc, passed, detail in checks_1:
        mark = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {mark}  {desc}")
        print(f"         ({detail})")
        if not passed:
            all_pass = False

    sensor_proba_before_voice = fs._state["latest_sensor_proba"]  # should be 0.75

    # ── Step 2: Voice click ────────────────────────────────────────────────
    print()
    print("  Step 2 — voice click, uses latest sensor (0.75)")
    print(_divider())

    fake_voice = {
        "transcript":     "I'm completely overwhelmed right now.",
        "semantic_score": -0.60,
        "tone_stress":    0.90,
        "combined_voice": 0.80,
    }
    with patch.object(fs, "run_voice_checkin", return_value=fake_voice):
        r4 = fs.on_voice_update("tests/Stressed.wav")

    checks_2 = [
        ("r4 sensor_proba == latest sensor before the click (0.75)",
         r4["sensor_proba"] == sensor_proba_before_voice,
         f"sensor_proba={r4['sensor_proba']}, expected {sensor_proba_before_voice}"),
        ("r4 combined_voice == 0.80 (new voice value)",
         r4["combined_voice"] == 0.80,
         f"combined_voice={r4['combined_voice']}"),
        ("dashboard callback fired again (total 4)",
         len(push_log) == 4,
         f"push_log has {len(push_log)} entries"),
    ]

    for desc, passed, detail in checks_2:
        mark = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {mark}  {desc}")
        print(f"         ({detail})")
        if not passed:
            all_pass = False

    # ── Step 3: Sensor update after voice click ────────────────────────────
    print()
    print("  Step 3 — sensor updates again, must use voice from Step 2 (0.80)")
    print(_divider())

    with patch.object(fs, "predict_sensor_score", return_value=0.60):
        r5 = fs.on_sensor_update(34.2, 0.0, 85.0, 4.5)

    expected_final = round(0.7 * 0.60 + 0.3 * 0.80, 4)

    checks_3 = [
        ("r5 combined_voice == 0.80 (voice from Step 2, not default 0.5)",
         r5["combined_voice"] == 0.80,
         f"combined_voice={r5['combined_voice']}"),
        (f"r5 final_score == {expected_final} (0.7×0.60 + 0.3×0.80)",
         abs(r5["final_score"] - expected_final) < 1e-4,
         f"final_score={r5['final_score']}, expected {expected_final}"),
        ("dashboard callback fired again (total 5)",
         len(push_log) == 5,
         f"push_log has {len(push_log)} entries"),
    ]

    for desc, passed, detail in checks_3:
        mark = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {mark}  {desc}")
        print(f"         ({detail})")
        if not passed:
            all_pass = False

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print(_divider("═"))
    if all_pass:
        print("  ALL CADENCE CHECKS PASSED")
        print("  → sensor and voice combine correctly on independent clocks.")
        print("  → dashboard callback fires on every update from either source.")
    else:
        print("  SOME CADENCE CHECKS FAILED — review output above.")
    print(_divider("═"))

    return all_pass


def main():
    print()
    print("  Shift-Guard — Full Integration Test")
    print("  Four-function pipeline end-to-end sanity check")
    print()

    results = []
    for scenario in SCENARIOS:
        print(f"  Running: {scenario['label']} ...")
        r = run_scenario(scenario)
        results.append((scenario, r))

    print()
    for scenario, result in results:
        print_scenario(scenario, result)

    # ── Assertions ────────────────────────────────────────────────────────
    worst_score = results[0][1]["final_score"]
    best_score  = results[1][1]["final_score"]
    worst_alert = results[0][1]["alert"]["triggered"]
    best_alert  = results[1][1]["alert"]["triggered"]

    print(_divider("═"))
    print("  SANITY CHECKS")
    print(_divider("═"))
    print()

    checks = [
        (
            "worst_case final_score > best_case final_score",
            worst_score > best_score,
            f"{worst_score:.4f} > {best_score:.4f}",
        ),
        (
            "worst_case triggers intervention",
            worst_alert is True,
            f"triggered={worst_alert}",
        ),
        (
            "best_case does NOT trigger intervention",
            best_alert is False,
            f"triggered={best_alert}",
        ),
    ]

    all_pass = True
    for desc, passed, detail in checks:
        mark = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {mark}  {desc}")
        print(f"         ({detail})")
        print()
        if not passed:
            all_pass = False

    print(_divider("═"))
    if all_pass:
        print("  ALL CHECKS PASSED — pipeline is ready for handoff.")
    else:
        print("  SOME CHECKS FAILED — review scores above before handoff.")
    print(_divider("═"))
    print()

    # ── Cadence independence test (no live API/model calls needed) ────────
    cadence_ok = test_cadence_independence()
    print()

    return 0 if (all_pass and cadence_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
