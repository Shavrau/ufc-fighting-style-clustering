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
    "Tai Tuivasa": {
        "expected": "Striker",
        "stats": {
            "significant_strikes_landed_per_minute": 3.98,
            "significant_striking_accuracy": 49.0,
            "significant_strikes_absorbed_per_minute": 4.98,
            "significant_strike_defence": 43.0,
            "average_takedowns_landed_per_15_minutes": 0.0,
            "takedown_accuracy": 0.0,
            "takedown_defense": 54.0,
            "average_submissions_attempted_per_15_minutes": 0.0,
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
    "Ben Askren": {
        "expected": "Wrestler",
        "stats": {
            "significant_strikes_landed_per_minute": 3.64,
            "significant_striking_accuracy": 49.0,
            "significant_strikes_absorbed_per_minute": 5.66,
            "significant_strike_defence": 36.0,
            "average_takedowns_landed_per_15_minutes": 4.33,
            "takedown_accuracy": 55.0,
            "takedown_defense": 0.0,
            "average_submissions_attempted_per_15_minutes": 0.9,
        },
    },
    "Mark Coleman": {
        "expected": "Wrestler",
        "stats": {
            "significant_strikes_landed_per_minute": 1.88,
            "significant_striking_accuracy": 52.0,
            "significant_strikes_absorbed_per_minute": 2.62,
            "significant_strike_defence": 40.0,
            "average_takedowns_landed_per_15_minutes": 3.89,
            "takedown_accuracy": 40.0,
            "takedown_defense": 50.0,
            "average_submissions_attempted_per_15_minutes": 0.8,
        },
    },
    "Yoel Romero": {
        "expected": "Wrestler",
        "stats": {
            "significant_strikes_landed_per_minute": 3.44,
            "significant_striking_accuracy": 49.0,
            "significant_strikes_absorbed_per_minute": 3.05,
            "significant_strike_defence": 61.0,
            "average_takedowns_landed_per_15_minutes": 1.77,
            "takedown_accuracy": 35.0,
            "takedown_defense": 78.0,
            "average_submissions_attempted_per_15_minutes": 0.0,
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
    "Kamaru Usman": {
        "expected": "Well-rounded",
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
        "expected": "Well-rounded",
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
    # Submission Specialist
    "Robert Drysdale": {
        "expected": "Submission Specialist",
        "stats": {
            "significant_strikes_landed_per_minute": 0.0,
            "significant_striking_accuracy": 0.0,
            "significant_strikes_absorbed_per_minute": 0.0,
            "significant_strike_defence": 0.0,
            "average_takedowns_landed_per_15_minutes": 7.32,
            "takedown_accuracy": 100.0,
            "takedown_defense": 0.0,
            "average_submissions_attempted_per_15_minutes": 21.9,
        },
    },
    "Megumi Fujii": {
        "expected": "Submission Specialist",
        "stats": {
            "significant_strikes_landed_per_minute": 2.4,
            "significant_striking_accuracy": 42.0,
            "significant_strikes_absorbed_per_minute": 0.0,
            "significant_strike_defence": 100.0,
            "average_takedowns_landed_per_15_minutes": 0.0,
            "takedown_accuracy": 0.0,
            "takedown_defense": 0.0,
            "average_submissions_attempted_per_15_minutes": 12.0,
        },
    },
    # Low Output / Weak Defense
    "CM Punk": {
        "expected": "Low Output / Weak Defense",
        "stats": {
            "significant_strikes_landed_per_minute": 1.1,
            "significant_striking_accuracy": 23.0,
            "significant_strikes_absorbed_per_minute": 4.87,
            "significant_strike_defence": 40.0,
            "average_takedowns_landed_per_15_minutes": 0.87,
            "takedown_accuracy": 11.0,
            "takedown_defense": 0.0,
            "average_submissions_attempted_per_15_minutes": 0.0,
        },
    },
    "Kris Moutinho": {
        "expected": "Low Output / Weak Defense",
        "stats": {
            "significant_strikes_landed_per_minute": 4.86,
            "significant_striking_accuracy": 32.0,
            "significant_strikes_absorbed_per_minute": 15.48,
            "significant_strike_defence": 28.0,
            "average_takedowns_landed_per_15_minutes": 0.0,
            "takedown_accuracy": 0.0,
            "takedown_defense": 0.0,
            "average_submissions_attempted_per_15_minutes": 0.0,
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
