"""
evaluate_model.py

Evaluates clusters, assigns human-readable labels based on raw unscaled metrics,
and generates summary reports.
"""
import json
import logging
from pathlib import Path
import pandas as pd

# --- PATH FIXES ---
DAGS_DIR = Path(__file__).resolve().parent.parent

CLUSTERED_PATH = DAGS_DIR / "data" / "processed" / "fighters_clustered.parquet"
PROCESSED_PATH = DAGS_DIR / "data" / "processed" / "fighters_processed.parquet"
OUTPUT_PATH = DAGS_DIR / "data" / "processed" / "fighters_clustered_labeled.parquet"
PROFILE_REPORT_PATH = DAGS_DIR / "reports" / "cluster_profile.csv"
LABELS_REPORT_PATH = DAGS_DIR / "reports" / "cluster_labels.json"

logger = logging.getLogger(__name__)

# Constants for features
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

# These are already in the clustered dataset (carried over by train_model.py)
PHYSICAL_CONTEXT_COLS = ["height_cm", "weight_in_kg", "reach_in_cm", "wins", "losses"]


def suggest_style_labels(profile: pd.DataFrame) -> dict:
    """Heuristic auto-labeling based on z-scored metrics."""
    # Calculate how many std deviations each cluster is from the mean for each metric
    z = (profile[STYLE_FEATURES] - profile[STYLE_FEATURES].mean()) / profile[STYLE_FEATURES].std()

    signal_groups = {
        "Striker": ["significant_strikes_landed_per_minute", "significant_striking_accuracy"],
        "Pressure Striker": ["significant_strikes_absorbed_per_minute"],
        "Wrestler": ["average_takedowns_landed_per_15_minutes", "takedown_accuracy", "takedown_defense"],
        "Grappler": ["average_submissions_attempted_per_15_minutes"],
    }

    labels = {}
    for cluster_id in profile.index:
        scores = {label: z.loc[cluster_id, cols].mean() for label, cols in signal_groups.items()}
        best_label = max(scores, key=scores.get)
        
        # Threshold for "Well-rounded" if no strong signal exists
        if scores[best_label] < 0.3:
            best_label = "Well-rounded"
            
        labels[int(cluster_id)] = {
            "label": best_label,
            "scores": {k: round(v, 2) for k, v in scores.items()},
        }
    return labels


def evaluate() -> Path:
    """Evaluates clusters, generates labels, and saves reports."""
    
    # 1. Load Data
    logger.info(f"Loading data from {CLUSTERED_PATH}")
    clustered = pd.read_parquet(CLUSTERED_PATH)
    processed = pd.read_parquet(PROCESSED_PATH)

    if len(clustered) != len(processed):
        raise ValueError(f"Row count mismatch: Clustered ({len(clustered)}) vs Processed ({len(processed)})")

    # 2. Combine Data
    # Clustered already has physical context. Just bring over the raw unscaled style 
    # features from 'processed' for human-readable profiling.
    df = clustered.copy()
    df[STYLE_FEATURES] = processed[STYLE_FEATURES].values

    # 3. Profiling
    # Calculate means of the raw stats for each cluster
    profile = df.groupby("cluster")[STYLE_FEATURES + PHYSICAL_CONTEXT_COLS].mean().round(2)
    profile["count"] = df["cluster"].value_counts()
    profile["pct"] = (profile["count"] / len(df) * 100).round(1)

    # 4. Labeling
    labels = suggest_style_labels(profile)
    df["cluster_label"] = df["cluster"].map(lambda c: labels[int(c)]["label"])

    # 5. Save Outputs
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