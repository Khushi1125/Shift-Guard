"""
eval_semantic.py — Offline evaluation of get_semantic_score() for Shift-Guard.

What this script does
---------------------
Measures how well VADER sentiment scoring separates "calm" from "stressed"
language using a labeled sentence dataset that mirrors speech patterns seen
in first-responder / caregiver contexts.

No Deepgram API key or audio files are needed — evaluations run against the
text output of `get_semantic_score()` directly, which is the component most
critical to validate before using it downstream.

Three evaluations are run:
  1. Direction accuracy  — does the sign of the score match the label?
  2. Calibration check   — are scores well-separated between calm and stressed?
  3. Edge-case audit     — how does the scorer handle tricky / borderline inputs?

Usage:
    python src/eval_semantic.py           # prints a report to stdout
    python src/eval_semantic.py --json    # also writes outputs/eval_semantic.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import List

sys.path.insert(0, os.path.dirname(__file__))
from semantic_analysis import get_semantic_score  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Labeled dataset
# Each entry: (text, expected_label)  where label ∈ {"calm", "stressed"}
# Sourced to reflect first-responder / caregiver occupational language.
# ──────────────────────────────────────────────────────────────────────────────

LABELED_SENTENCES: List[tuple[str, str]] = [
    # ── Stressed / negative ──────────────────────────────────────────────────
    ("I'm completely overwhelmed and I can't keep up anymore.", "stressed"),
    ("This is terrible. Nothing is working and I'm falling apart.", "stressed"),
    ("I feel trapped and exhausted. I don't know how much longer I can do this.", "stressed"),
    ("I hate these shifts. Everyone is demanding more than I can give.", "stressed"),
    ("I'm running on empty. I haven't slept in two days.", "stressed"),
    ("I'm so frustrated. The patient deteriorated and I feel like I failed.", "stressed"),
    ("Everything hurts and I don't care anymore. I'm done.", "stressed"),
    ("This job is destroying me. I feel hopeless and burned out.", "stressed"),
    ("Another awful day. I messed up twice and I can't forgive myself.", "stressed"),
    ("I'm panicking. There's too much to do and not enough time.", "stressed"),
    ("I feel disconnected and numb. Nothing feels meaningful anymore.", "stressed"),
    ("It was chaos today. I was scared and I didn't know what to do.", "stressed"),
    # ── Calm / positive ──────────────────────────────────────────────────────
    ("Today was a good day. I handled everything well and I feel proud.", "calm"),
    ("I feel calm and focused. The shift went smoothly.", "calm"),
    ("I'm grateful for the support from my team. We work well together.", "calm"),
    ("I helped a patient recover today. It was really rewarding.", "calm"),
    ("I feel rested and ready. I slept well and I'm looking forward to the day.", "calm"),
    ("Everything went according to plan. I'm satisfied with how I handled things.", "calm"),
    ("I feel confident and in control. The team was great today.", "calm"),
    ("I managed to stay composed even during a tough call. I'm proud of that.", "calm"),
    ("I love this work. Even when it's hard, I know it matters.", "calm"),
    ("I feel at peace with today. I did my best and that's enough.", "calm"),
    ("The handover was smooth and I left on time. Good day overall.", "calm"),
    ("I had a moment of clarity today. I know why I do this job.", "calm"),
    # ── Neutral (score should be between -0.05 and +0.05) ───────────────────
    ("The patient is in room four.", "neutral"),
    ("Shift starts at seven in the morning.", "neutral"),
    ("The report was submitted at the end of the day.", "neutral"),
    ("I attended a briefing about the new protocol.", "neutral"),
]

# Thresholds that define polarity buckets (mirrors VADER convention).
STRESSED_THRESHOLD = -0.05   # score below this → classified as "stressed"
CALM_THRESHOLD = 0.05        # score above this → classified as "calm"
                             # between the two → "neutral"


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SentenceResult:
    text: str
    expected: str
    score: float
    predicted: str
    correct: bool


@dataclass
class EvalReport:
    n_total: int
    n_correct: int
    direction_accuracy: float          # among calm+stressed only (excludes neutral)
    mean_score_calm: float
    mean_score_stressed: float
    score_gap: float                   # mean_calm − mean_stressed (want > 0.4)
    neutral_mean_abs_score: float      # want < 0.2 (close to zero)
    edge_case_results: List[dict]
    per_sentence: List[dict]


# ──────────────────────────────────────────────────────────────────────────────
# Eval 1 — Direction accuracy
# ──────────────────────────────────────────────────────────────────────────────

def _classify(score: float) -> str:
    if score < STRESSED_THRESHOLD:
        return "stressed"
    if score > CALM_THRESHOLD:
        return "calm"
    return "neutral"


def run_direction_accuracy(sentences: List[tuple[str, str]]) -> List[SentenceResult]:
    """Score each sentence and check whether the predicted polarity matches the label."""
    results = []
    for text, expected in sentences:
        score = get_semantic_score(text)
        predicted = _classify(score)
        correct = predicted == expected
        results.append(SentenceResult(text=text, expected=expected, score=score,
                                      predicted=predicted, correct=correct))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Eval 2 — Calibration: are the two classes well separated?
# ──────────────────────────────────────────────────────────────────────────────

def run_calibration(results: List[SentenceResult]) -> dict:
    calm_scores = [r.score for r in results if r.expected == "calm"]
    stressed_scores = [r.score for r in results if r.expected == "stressed"]
    neutral_scores = [r.score for r in results if r.expected == "neutral"]

    mean_calm = sum(calm_scores) / len(calm_scores) if calm_scores else 0.0
    mean_stressed = sum(stressed_scores) / len(stressed_scores) if stressed_scores else 0.0
    mean_neutral_abs = (
        sum(abs(s) for s in neutral_scores) / len(neutral_scores) if neutral_scores else 0.0
    )
    gap = mean_calm - mean_stressed

    return {
        "mean_score_calm": round(mean_calm, 4),
        "mean_score_stressed": round(mean_stressed, 4),
        "score_gap": round(gap, 4),
        "neutral_mean_abs_score": round(mean_neutral_abs, 4),
        "gap_pass": gap > 0.4,
        "neutral_pass": mean_neutral_abs < 0.2,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Eval 3 — Edge cases
# ──────────────────────────────────────────────────────────────────────────────

EDGE_CASES: List[tuple[str, str, str]] = [
    # (text, expected_direction, description)
    ("", "neutral", "empty string"),
    ("   ", "neutral", "whitespace only"),
    ("help", "neutral", "single word"),
    ("help help help help help help", "neutral", "repeated single word"),
    ("I'm not stressed at all", "calm",
     "negation — VADER handles this; should read as calm"),
    ("fine", "neutral",
     "ambiguous word — could be dismissive ('I'm fine') or genuinely okay"),
    ("It was not a bad day", "calm",
     "double negation — should come out slightly positive"),
    ("!!!!", "neutral", "punctuation only — no meaningful content"),
    ("The patient coded.", "neutral",
     "clinical jargon — neutral factual statement"),
    ("I'm so tired but I did it.", "neutral",
     "mixed valence — tired (neg) vs achievement (pos)"),
]


def run_edge_cases() -> List[dict]:
    results = []
    for text, expected_dir, description in EDGE_CASES:
        score = get_semantic_score(text)
        predicted = _classify(score)
        # Edge cases don't all have hard correct/incorrect answers — we flag
        # unexpected direction only for cases with a clear expected direction.
        expected_direction_str = expected_dir
        correct = predicted == expected_dir
        results.append({
            "description": description,
            "text": text,
            "score": round(score, 4),
            "predicted": predicted,
            "expected": expected_direction_str,
            "pass": correct,
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report printing
# ──────────────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    """Render a [-1, 1] score as an ASCII bar centred at 0."""
    mid = width // 2
    pos = int((value + 1) / 2 * width)
    bar = ["-"] * width
    if 0 <= pos < width:
        bar[pos] = "█"
    bar[mid] = "|"
    return "".join(bar)


def print_report(results: List[SentenceResult], calibration: dict,
                 edge_results: List[dict]) -> None:
    # ── Direction accuracy ────────────────────────────────────────────────────
    non_neutral = [r for r in results if r.expected != "neutral"]
    n_correct = sum(1 for r in non_neutral if r.correct)
    accuracy = n_correct / len(non_neutral) if non_neutral else 0.0

    print("\n" + "═" * 70)
    print("  EVAL 1 — Direction Accuracy (calm / stressed only)")
    print("═" * 70)
    print(f"  Sentences evaluated : {len(non_neutral)}")
    print(f"  Correct             : {n_correct}")
    print(f"  Accuracy            : {accuracy:.1%}  "
          f"{'✓ PASS (≥ 80%)' if accuracy >= 0.80 else '✗ FAIL (< 80%)'}")
    print()

    failures = [r for r in non_neutral if not r.correct]
    if failures:
        print("  Misclassified sentences:")
        for r in failures:
            print(f"    [{r.expected:>8} → {r.predicted:<8}] score={r.score:+.3f}  {r.text[:60]}")
    else:
        print("  All sentences classified correctly.")

    # ── Calibration ──────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  EVAL 2 — Score Calibration")
    print("═" * 70)
    print(f"  Mean score (calm)     : {calibration['mean_score_calm']:+.4f}  "
          f"{_bar(calibration['mean_score_calm'])}")
    print(f"  Mean score (stressed) : {calibration['mean_score_stressed']:+.4f}  "
          f"{_bar(calibration['mean_score_stressed'])}")
    print(f"  Score gap (calm−stressed): {calibration['score_gap']:+.4f}  "
          f"{'✓ PASS (> 0.40)' if calibration['gap_pass'] else '✗ FAIL (≤ 0.40)'}")
    print(f"  Neutral mean |score|  : {calibration['neutral_mean_abs_score']:.4f}  "
          f"{'✓ PASS (< 0.20)' if calibration['neutral_pass'] else '✗ FAIL (≥ 0.20)'}")

    # Per-sentence breakdown
    print()
    print(f"  {'Label':>8}  {'Score':>7}  {'Pred':>8}  {'OK':>3}  Text")
    print("  " + "-" * 66)
    for r in results:
        ok = "✓" if r.correct else "✗"
        label = r.expected.upper()[:3]
        print(f"  {label:>8}  {r.score:>+7.3f}  {r.predicted:<8}  {ok:>3}  {r.text[:44]}")

    # ── Edge cases ────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  EVAL 3 — Edge Cases")
    print("═" * 70)
    for ec in edge_results:
        ok = "✓" if ec["pass"] else "△"   # △ = unexpected but not a hard failure
        print(f"  {ok}  score={ec['score']:+.4f}  [{ec['predicted']:<8}]  {ec['description']}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    overall = (
        accuracy >= 0.80
        and calibration["gap_pass"]
        and calibration["neutral_pass"]
    )
    print("═" * 70)
    print(f"  OVERALL: {'✓ ALL EVALS PASS' if overall else '✗ ONE OR MORE EVALS FAILED'}")
    print("═" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(write_json: bool = False) -> EvalReport:
    logger.info("Starting semantic analysis evaluation")

    sentence_results = run_direction_accuracy(LABELED_SENTENCES)
    calibration = run_calibration(sentence_results)
    edge_results = run_edge_cases()

    print_report(sentence_results, calibration, edge_results)

    non_neutral = [r for r in sentence_results if r.expected != "neutral"]
    n_correct = sum(1 for r in non_neutral if r.correct)
    accuracy = n_correct / len(non_neutral) if non_neutral else 0.0

    report = EvalReport(
        n_total=len(sentence_results),
        n_correct=n_correct,
        direction_accuracy=round(accuracy, 4),
        mean_score_calm=calibration["mean_score_calm"],
        mean_score_stressed=calibration["mean_score_stressed"],
        score_gap=calibration["score_gap"],
        neutral_mean_abs_score=calibration["neutral_mean_abs_score"],
        edge_case_results=edge_results,
        per_sentence=[asdict(r) for r in sentence_results],
    )

    if write_json:
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "outputs", "eval_semantic.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("Eval results written to %s", os.path.abspath(out_path))

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate get_semantic_score()")
    parser.add_argument("--json", action="store_true",
                        help="Write results to outputs/eval_semantic.json")
    args = parser.parse_args()
    main(write_json=args.json)
