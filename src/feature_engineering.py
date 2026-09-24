"""
feature_engineering.py

Feature engineering step for the fighting-style clustering pipeline.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA

DAGS_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = DAGS_DIR / "data" / "processed" / "fighters_processed.parquet"
OUTPUT_PATH = DAGS_DIR / "data" / "processed" / "fighters_features.parquet"
PCA_PATH = DAGS_DIR / "models" / "pca.pkl"

logger = logging.getLogger(__name__)

SCALED_FEATURES = [
    "significant_strikes_landed_per_minute_scaled",
    "significant_striking_accuracy_scaled",
    "significant_strikes_absorbed_per_minute_scaled",
    "significant_strike_defence_scaled",
    "average_takedowns_landed_per_15_minutes_scaled",
    "takedown_accuracy_scaled",
    "takedown_defense_scaled",
    "average_submissions_attempted_per_15_minutes_scaled",
]

def engineer_features(fit_pca: bool = True) -> Path:
    logger.info(f"Loading processed data from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    df["striking_net_scaled"] = (
        df["significant_strikes_landed_per_minute_scaled"] -
        df["significant_strikes_absorbed_per_minute_scaled"]
    )
    df["grappling_pressure_scaled"] = (
        df["average_takedowns_landed_per_15_minutes_scaled"] +
        df["average_submissions_attempted_per_15_minutes_scaled"]
    ) / 2
    df["defensive_rating_scaled"] = (
        df["significant_strike_defence_scaled"] +
        df["takedown_defense_scaled"]
    ) / 2

    derived_cols = ["striking_net_scaled", "grappling_pressure_scaled", "defensive_rating_scaled"]
    feature_matrix = df[SCALED_FEATURES + derived_cols].copy()

    # Grappling signal (takedowns, submissions) tends to get diluted once mixed
    # with 7 other features and compressed through PCA. Verified by testing known
    # wrestlers (Khabib Nurmagomedov, 5.3 takedowns/15min) and known submission
    # specialists (Robert Drysdale, Megumi Fujii, 10+ submission attempts/15min) --
    # both archetypes were landing in the generic striker cluster despite dominant
    # grappling volume. Submissions need a bigger boost than takedowns since the
    # submission-hunter archetype is a smaller, more extreme subset of fighters.
    TD_WEIGHT = 1.5
    SUB_WEIGHT = 2.5
    feature_matrix["average_takedowns_landed_per_15_minutes_scaled"] *= TD_WEIGHT
    feature_matrix["takedown_accuracy_scaled"] *= TD_WEIGHT
    feature_matrix["average_submissions_attempted_per_15_minutes_scaled"] *= SUB_WEIGHT
    feature_matrix["grappling_pressure_scaled"] *= (TD_WEIGHT + SUB_WEIGHT) / 2

    if fit_pca:
        full_pca = PCA().fit(feature_matrix)
        cumulative = full_pca.explained_variance_ratio_.cumsum()
        n_components = int((cumulative < 0.90).sum() + 1)

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(feature_matrix)

        PCA_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, PCA_PATH)
        logger.info(f"PCA fitted: {n_components} components (90% variance) saved to {PCA_PATH}")
    else:
        pca = joblib.load(PCA_PATH)
        components = pca.transform(feature_matrix)
        logger.info(f"Loaded existing PCA from {PCA_PATH}")

    pca_cols = [f"pc_{i+1}" for i in range(components.shape[1])]
    final_features = pd.DataFrame(components, columns=pca_cols, index=df.index)

    profiling_cols = [
        "name", "nickname", "wins", "losses", "draws",
        "height_cm", "weight_in_kg", "reach_in_cm", "stance"
    ]
    profiling_cols = [col for col in profiling_cols if col in df.columns]

    result = pd.concat([df[profiling_cols], final_features], axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    logger.info(f"Success! Features written to {OUTPUT_PATH} ({len(result)} rows)")
    return OUTPUT_PATH

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    engineer_features(fit_pca=True)
