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
    """
    Full feature engineering pipeline.
    fit_pca=True -> Fits new PCA (for training)
    fit_pca=False -> Loads existing PCA (for scoring/inference)
    """
    logger.info(f"Loading processed data from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    # 1. Create Derived Features
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

    # 2. Isolate final feature matrix
    derived_cols = ["striking_net_scaled", "grappling_pressure_scaled", "defensive_rating_scaled"]
    feature_matrix = df[SCALED_FEATURES + derived_cols]

    # 3. PCA Dimensionality Reduction
    if fit_pca:
        # Fit full PCA to dynamically find components for 90% variance
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

    # Build DataFrame from PCA components
    pca_cols = [f"pc_{i+1}" for i in range(components.shape[1])]
    final_features = pd.DataFrame(components, columns=pca_cols, index=df.index)

    # 4. Combine with Identifier/Profiling Columns
    profiling_cols = [
        "name", "nickname", "wins", "losses", "draws", 
        "height_cm", "weight_in_kg", "reach_in_cm", "stance"
    ]
    # Filter to only keep columns that actually exist in the dataframe to prevent errors
    profiling_cols = [col for col in profiling_cols if col in df.columns]
    
    result = pd.concat([df[profiling_cols], final_features], axis=1)

    # 5. Save and Return
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    
    logger.info(f"Success! Features written to {OUTPUT_PATH} ({len(result)} rows)")
    return OUTPUT_PATH


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    engineer_features(fit_pca=True)