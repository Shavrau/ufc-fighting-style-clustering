"""
validate_golden_set.py

Validation step for the fighting-style clustering pipeline. Scores a fixed set of
well-known fighters (3 per target style, styles agreed on by the MMA community) against
the currently trained model and reports accuracy. Exists because eyeballing 3-4
hand-picked examples gave false confidence early in this project (see
docs/model_limitations.md) -- a fixed, checked-in golden set makes the validation
reproducible and auditable every time the model changes.

Not part of the Airflow DAG -- this is a manual check to run after retraining, not a
production step. Run it whenever preprocessing.py, feature_engineering.py, or the
weights in scoring.py change, to see whether accuracy improved or regressed.

Usage:
    python3 src/validate_golden_set.py

Input : models/scaler.pkl, models/pca.pkl, models/kmeans.pkl, reports/cluster_labels.json
        (all loaded via scoring.py)
Output: reports/golden_set_validation.json
"""
import json
from pathlib import Path

from scoring import score_fighter

DAGS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = DAGS_DIR / "reports" / "golden_set_validation.json"

# Stats pulled from CBS Sports fighter profiles. 3 fighters per target style, chosen
# for broad community agreement on their style label -- not chosen to make the model
# look good (that was the mistake with the earlier 3-4-fighter ad hoc checks).
GOLDEN_SET = {
    # Striker
    'Alex Pereira "Poatan"': {
        "expected": "Striker",
        "stats": {
            "significant_strikes_landed_per_minute": 5.01,
            "significant_striking_accuracy": 61.4,
            "significant_strikes_absorbed_per_minute": 3.72,
            "significant_strike_defence": 53,
            "average_takedowns_landed_per_15_minutes": 0.11,
            "takedown_accuracy": 50,
            "takedown_defense": 80,
            "average_submissions_attempted_per_15_minutes": 0.22,
        },
    },
    "Max Holloway": {
        "expected": "Striker",
        "stats": {
            "significant_strikes_landed_per_minute": 6.91,
            "significant_striking_accuracy": 48.1,
            "significant_strikes_absorbed_per_minute": 4.61,
            "significant_strike_defence": 59,
            "average_takedowns_landed_per_15_minutes": 0.23,
            "takedown_accuracy": 53.3,
            "takedown_defense": 81,
            "average_submissions_attempted_per_15_minutes": 0.28,
        },
    },
    "Justin Gaethje": {
        "expected": "Striker",
        "stats": {
            "significant_strikes_landed_per_minute": 6.29,
            "significant_striking_accuracy": 57.6,
            "significant_strikes_absorbed_per_minute": 6.83,
            "significant_strike_defence": 51,
            "average_takedowns_landed_per_15_minutes": 0.3,
            "takedown_accuracy": 26.7,
            "takedown_defense": 73,
            "average_submissions_attempted_per_15_minutes": 0.15,
        },
    },
    # Wrestler
    "Khabib Nurmagomedov": {
        "expected": "Wrestler",
        "stats": {
            "significant_strikes_landed_per_minute": 4.1,
            "significant_striking_accuracy": 48.8,
            "significant_strikes_absorbed_per_minute": 1.75,
            "significant_strike_defence": 65,
            "average_takedowns_landed_per_15_minutes": 5.32,
            "takedown_accuracy": 48.0,
            "takedown_defense": 85,
            "average_submissions_attempted_per_15_minutes": 0.79,
        },
    },
    "Kamaru Usman": {
        "expected": "Wrestler",
        "stats": {
            "significant_strikes_landed_per_minute": 4.13,
            "significant_striking_accuracy": 50.9,
            "significant_strikes_absorbed_per_minute": 2.86,
            "significant_strike_defence": 55,
            "average_takedowns_landed_per_15_minutes": 2.6,
            "takedown_accuracy": 42.0,
            "takedown_defense": 90,
            "average_submissions_attempted_per_15_minutes": 0.08,
        },
    },
    "Islam Makhachev": {
        "expected": "Wrestler",
        "stats": {
            "significant_strikes_landed_per_minute": 2.45,
            "significant_striking_accuracy": 58.4,
            "significant_strikes_absorbed_per_minute": 1.45,
            "significant_strike_defence": 62,
            "average_takedowns_landed_per_15_minutes": 3.1,
            "takedown_accuracy": 56.2,
            "takedown_defense": 91,
            "average_submissions_attempted_per_15_minutes": 0.98,
        },
    },
    # Grappler
    "Charles Oliveira": {
        "expected": "Grappler",
        "stats": {
            "significant_strikes_landed_per_minute": 3.23,
            "significant_striking_accuracy": 55.6,
            "significant_strikes_absorbed_per_minute": 3.05,
            "significant_strike_defence": 49,
            "average_takedowns_landed_per_15_minutes": 2.29,
            "takedown_accuracy": 39.8,
            "takedown_defense": 55,
            "average_submissions_attempted_per_15_minutes": 2.59,
        },
    },
    "Demian Maia": {
        "expected": "Grappler",
        "stats": {
            "significant_strikes_landed_per_minute": 1.8,
            "significant_striking_accuracy": 43.6,
            "significant_strikes_absorbed_per_minute": 2.07,
            "significant_strike_defence": 64,
            "average_takedowns_landed_per_15_minutes": 2.49,
            "takedown_accuracy": 24.8,
            "takedown_defense": 62,
            "average_submissions_attempted_per_15_minutes": 0.99,
        },
    },
    "Gilbert Burns": {
        "expected": "Grappler",
        "stats": {
            "significant_strikes_landed_per_minute": 3.16,
            "significant_striking_accuracy": 48.1,
            "significant_strikes_absorbed_per_minute": 3.68,
            "significant_strike_defence": 53,
            "average_takedowns_landed_per_15_minutes": 2.04,
            "takedown_accuracy": 35.6,
            "takedown_defense": 54,
            "average_submissions_attempted_per_15_minutes": 0.44,
        },
    },
    # Well-rounded
    "Georges St-Pierre": {
        "expected": "Well-rounded",
        "stats": {
            "significant_strikes_landed_per_minute": 3.83,
            "significant_striking_accuracy": 53.2,
            "significant_strikes_absorbed_per_minute": 1.5,
            "significant_strike_defence": 73,
            "average_takedowns_landed_per_15_minutes": 3.94,
            "takedown_accuracy": 73.8,
            "takedown_defense": 84,
            "average_submissions_attempted_per_15_minutes": 1.05,
        },
    },
    "Jon Jones": {
        "expected": "Well-rounded",
        "stats": {
            "significant_strikes_landed_per_minute": 4.38,
            "significant_striking_accuracy": 58.9,
            "significant_strikes_absorbed_per_minute": 2.24,
            "significant_strike_defence": 64,
            "average_takedowns_landed_per_15_minutes": 1.89,
            "takedown_accuracy": 45.9,
            "takedown_defense": 95,
            "average_submissions_attempted_per_15_minutes": 0.46,
        },
    },
    "Daniel Cormier": {
        "expected": "Well-rounded",
        "stats": {
            "significant_strikes_landed_per_minute": 4.39,
            "significant_striking_accuracy": 54.7,
            "significant_strikes_absorbed_per_minute": 4.03,
            "significant_strike_defence": 55,
            "average_takedowns_landed_per_15_minutes": 1.66,
            "takedown_accuracy": 46.0,
            "takedown_defense": 80,
            "average_submissions_attempted_per_15_minutes": 0.58,
        },
    },
}


def validate(golden_set: dict = GOLDEN_SET, output_path: Path = OUTPUT_PATH) -> dict:
    results = []
    for name, info in golden_set.items():
        r = score_fighter(info["stats"])
        expected, predicted = info["expected"], r["label"]
        # loose match: production labels can carry extra qualifiers
        # (e.g. "Low Output / Weak Defense") that a plain expected/predicted
        # string-equality check would miss even when directionally related.
        correct = expected.lower() in predicted.lower() or predicted.lower() in expected.lower()
        results.append({
            "name": name,
            "expected": expected,
            "predicted": predicted,
            "distance": r["distance_to_assigned_centroid"],
            "borderline": r["borderline"],
            "correct": correct,
        })

    accuracy = sum(r["correct"] for r in results) / len(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return {"results": results, "accuracy": accuracy}


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    outcome = validate()
    print(f"{'Fighter':<26} {'Expected':<14} {'Predicted':<26} Correct")
    for r in outcome["results"]:
        mark = "✓" if r["correct"] else "✗"
        print(f"{r['name']:<26} {r['expected']:<14} {r['predicted']:<26} {mark}")

    n_correct = sum(r["correct"] for r in outcome["results"])
    print(f"\nAccuracy: {n_correct}/{len(outcome['results'])} = {outcome['accuracy']*100:.0f}%")
    print(f"Saved to {OUTPUT_PATH}")
