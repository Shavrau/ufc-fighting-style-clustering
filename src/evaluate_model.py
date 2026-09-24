"""
evaluate_model.py

Evaluates clusters, assigns human-readable labels based on raw unscaled metrics,
and generates summary reports.
"""
import json
import logging
from pathlib import Path
import pandas as pd

DAGS_DIR = Path(__file__).resolve().parent.parent

CLUSTERED_PATH = DAGS_DIR / "data" / "processed" / "fighters_clustered.parquet"
PROCESSED_PATH = DAGS_DIR / "data" / "processed" / "fighters_processed.parquet"
OUTPUT_PATH = DAGS_DIR / "data" / "processed" / "fighters_clustered_labeled.parquet"
PROFILE_REPORT_PATH = DAGS_DIR / "reports" / "cluster_profile.csv"
LABELS_REPORT_PATH = DAGS_DIR / "reports" / "cluster_labels.json"

logger = logging.getLogger(__name__)

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

PHYSICAL_CONTEXT_COLS = ["height_cm", "weight_in_kg", "reach_in_cm", "wins", "losses"]

def suggest_style_labels(profile: pd.DataFrame) -> dict:
    z = (profile[STYLE_FEATURES] - profile[STYLE_FEATURES].mean()) / profile[STYLE_FEATURES].std()

    signal_groups = {
        "Striker": ["significant_strikes_landed_per_minute", "significant_striking_accuracy"],
        "Wrestler": ["average_takedowns_landed_per_15_minutes", "takedown_accuracy", "takedown_defense"],
        "Grappler": ["average_submissions_attempted_per_15_minutes"],
    }
    # NOTE: "Pressure Striker" used to be its own signal group keyed only on
    # significant_strikes_absorbed_per_minute. That's ambiguous: absorbing a lot
    # of strikes could mean an aggressive volume-trading fighter (real "pressure"
    # style) OR just a fighter with weak defense and low output -- the raw metric
    # can't tell those apart. Confirmed on the real cluster profile: the cluster
    # that used to win this label had the LOWEST landed-strikes rate of all
    # clusters, not a high one -- it was mislabeling a low-quality/weak-defense
    # cluster as an aggressive offensive style. Handled below as a special case
    # that checks landed volume too, instead of a plain z-score signal group.

    labels = {}
    for cluster_id in profile.index:
        scores = {label: z.loc[cluster_id, cols].mean() for label, cols in signal_groups.items()}
        best_label = max(scores, key=scores.get)

        landed_z = z.loc[cluster_id, "significant_strikes_landed_per_minute"]
        absorbed_z = z.loc[cluster_id, "significant_strikes_absorbed_per_minute"]
        defence_z = z.loc[cluster_id, "significant_strike_defence"]

        if absorbed_z > 0.5 and landed_z > 0.3:
            # High output AND high absorption together = genuine volume-trading
            # "pressure" style, not just poor defense.
            best_label = "Pressure Striker"
            scores["Pressure Striker"] = (landed_z + absorbed_z) / 2
        elif absorbed_z > 0.5 and landed_z < 0 and defence_z < 0:
            # High absorption WITHOUT matching output = weak defense, not an
            # offensive style. Name it for what it is.
            best_label = "Low Output / Weak Defense"
            scores["Low Output / Weak Defense"] = absorbed_z
        elif scores[best_label] < 0.3:
            best_label = "Well-rounded"
        elif best_label == "Wrestler" and profile.loc[cluster_id, "average_submissions_attempted_per_15_minutes"] > 1.0:
            best_label = "Wrestler / Grappler"
        elif best_label == "Grappler" and profile.loc[cluster_id, "average_submissions_attempted_per_15_minutes"] > 5.0:
            best_label = "Submission Specialist (Grappler)"

        labels[int(cluster_id)] = {
            "label": best_label,
            "scores": {k: round(v, 2) for k, v in scores.items()},
        }
    return labels

def evaluate() -> Path:
    logger.info(f"Loading data from {CLUSTERED_PATH}")
    clustered = pd.read_parquet(CLUSTERED_PATH)
    processed = pd.read_parquet(PROCESSED_PATH)

    if len(clustered) != len(processed):
        raise ValueError(f"Row count mismatch: Clustered ({len(clustered)}) vs Processed ({len(processed)})")

    df = clustered.copy()
    df[STYLE_FEATURES] = processed[STYLE_FEATURES].values

    profile = df.groupby("cluster")[STYLE_FEATURES + PHYSICAL_CONTEXT_COLS].mean().round(2)
    profile["count"] = df["cluster"].value_counts()
    profile["pct"] = (profile["count"] / len(df) * 100).round(1)

    labels = suggest_style_labels(profile)
    df["cluster_label"] = df["cluster"].map(lambda c: labels[int(c)]["label"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_PATH, index=False)
    profile.to_csv(PROFILE_REPORT_PATH)

    with open(LABELS_REPORT_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    logger.info(f"Success! Evaluation complete. Labeled data saved to: {OUTPUT_PATH}")
    return OUTPUT_PATH

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    evaluate()
