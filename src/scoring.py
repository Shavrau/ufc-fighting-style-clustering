"""
scoring.py

Scoring step for the fighting-style clustering pipeline. Takes a single
fighter's raw stats and tells you which trained cluster (style group) they
belong to. Runs the exact same transform chain as training
(preprocessing.py -> feature_engineering.py) but only *applies* the already
-fitted scaler/PCA/K-Means instead of refitting them.

Usage from the command line (interactive prompts):
    python3 scoring.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DAGS_DIR = Path(__file__).resolve().parent.parent

SCALER_PATH = DAGS_DIR / "models" / "scaler.pkl"
PCA_PATH = DAGS_DIR / "models" / "pca.pkl"
KMEANS_PATH = DAGS_DIR / "models" / "kmeans.pkl"
LABELS_PATH = DAGS_DIR / "reports" / "cluster_labels.json"

STYLE_FEATURES = [
    "significant_strikes_landed_per_minute",
    "significant_striking_accuracy",
    "significant_strikes_absorbed_per_minute",
    "significant_strike_defence",
    "average_takedowns_landed_per_15_minutes",
    "takedown_accuracy",
    "takedown_defense",
    "average_submissions_attempted_per_15_minutes",
]

SKEWED_FEATURES = [
    "average_submissions_attempted_per_15_minutes",
    "average_takedowns_landed_per_15_minutes",
]


def load_artifacts():
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    with open(LABELS_PATH) as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    return scaler, pca, kmeans, labels


def transform(stats: dict, scaler, pca) -> np.ndarray:
    row = pd.DataFrame([{f: stats[f] for f in STYLE_FEATURES}])

    for col in SKEWED_FEATURES:
        row[col] = np.log1p(row[col])

    scaled = scaler.transform(row[STYLE_FEATURES])
    scaled_cols = [f"{c}_scaled" for c in STYLE_FEATURES]
    scaled_df = pd.DataFrame(scaled, columns=scaled_cols)

    scaled_df["striking_net_scaled"] = (
        scaled_df["significant_strikes_landed_per_minute_scaled"]
        - scaled_df["significant_strikes_absorbed_per_minute_scaled"]
    )
    scaled_df["grappling_pressure_scaled"] = (
        scaled_df["average_takedowns_landed_per_15_minutes_scaled"]
        + scaled_df["average_submissions_attempted_per_15_minutes_scaled"]
    ) / 2
    scaled_df["defensive_rating_scaled"] = (
        scaled_df["significant_strike_defence_scaled"] + scaled_df["takedown_defense_scaled"]
    ) / 2

    feature_cols = scaled_cols + ["striking_net_scaled", "grappling_pressure_scaled", "defensive_rating_scaled"]
    return pca.transform(scaled_df[feature_cols])


def score_fighter(stats: dict, artifacts=None) -> dict:
    scaler, pca, kmeans, labels = artifacts or load_artifacts()

    X = transform(stats, scaler, pca)
    cluster_id = int(kmeans.predict(X)[0])
    distances = kmeans.transform(X)[0]

    ranked = sorted(
        [{"cluster": i, "label": labels[i]["label"], "distance": round(float(d), 3)}
         for i, d in enumerate(distances)],
        key=lambda r: r["distance"],
    )

    return {
        "cluster": cluster_id,
        "label": labels[cluster_id]["label"],
        "distance_to_assigned_centroid": round(float(distances[cluster_id]), 3),
        "ranked_clusters": ranked,
        "borderline": ranked[1]["distance"] - ranked[0]["distance"] < 0.5,
    }


def prompt_for_stats() -> dict:
    print("Enter the fighter's stats (original units, same as the raw CSV):\n")
    stats = {}
    for feature in STYLE_FEATURES:
        while True:
            raw = input(f"  {feature}: ").strip()
            try:
                stats[feature] = float(raw)
                break
            except ValueError:
                print("    please enter a number")
    return stats


if __name__ == "__main__":
    fighter_stats = prompt_for_stats()
    result = score_fighter(fighter_stats)

    print(f"\nPredicted style group: cluster {result['cluster']} - {result['label']}")
    print(f"Distance to assigned centroid: {result['distance_to_assigned_centroid']}")
    if result["borderline"]:
        print("Note: this fighter is borderline - close to more than one style cluster.")
    print("\nAll clusters ranked by distance (closest first):")
    for r in result["ranked_clusters"]:
        print(f"  cluster {r['cluster']} ({r['label']}): {r['distance']}")