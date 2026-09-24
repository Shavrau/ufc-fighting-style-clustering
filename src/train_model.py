"""
train_model.py

Training step for the fighting-style clustering pipeline.
"""
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

DAGS_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = DAGS_DIR / "data" / "processed" / "fighters_features.parquet"
OUTPUT_PATH = DAGS_DIR / "data" / "processed" / "fighters_clustered.parquet"
KMEANS_PATH = DAGS_DIR / "models" / "kmeans.pkl"
MODEL_CHOICE_PATH = DAGS_DIR / "models" / "model_choice.json"

logger = logging.getLogger(__name__)

PROFILING_COLS = [
    "name", "nickname", "wins", "losses", "draws",
    "height_cm", "weight_in_kg", "reach_in_cm", "stance",
]

K_CANDIDATES = range(3, 9)
K_FINAL_OVERRIDE = 4

def train_model(k_override=K_FINAL_OVERRIDE) -> Path:
    logger.info(f"Loading features from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    profiling_cols = [c for c in PROFILING_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in profiling_cols]

    X = df[feature_cols].values
    logger.info(f"Loaded {len(df)} rows, {len(feature_cols)} feature columns")

    scores = {}
    for k in K_CANDIDATES:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))

    best_k = max(scores, key=scores.get)
    logger.info(f"Silhouette scores by k: {scores}")

    k_final = k_override if k_override is not None else best_k
    logger.info(f"Training with k={k_final} (override={k_override}, optimal={best_k})")

    kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10).fit(X)
    gmm = GaussianMixture(n_components=k_final, random_state=42, n_init=5).fit(X)

    kmeans_sil = float(silhouette_score(X, kmeans.labels_))
    gmm_sil = float(silhouette_score(X, gmm.predict(X)))

    logger.info(f"Final K-Means silhouette: {kmeans_sil:.4f} | GMM silhouette: {gmm_sil:.4f}")

    KMEANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, KMEANS_PATH)

    model_choice = {
        "k_final": int(k_final),
        "k_silhouette_optimal": int(best_k),
        "silhouette_by_k": scores,
        "kmeans_silhouette": kmeans_sil,
        "gmm_silhouette": gmm_sil,
    }
    with open(MODEL_CHOICE_PATH, "w") as f:
        json.dump(model_choice, f, indent=2)
    logger.info(f"Model choice metadata saved to {MODEL_CHOICE_PATH}")

    df["cluster"] = kmeans.labels_
    df.to_parquet(OUTPUT_PATH, index=False)

    logger.info(f"Success! Clustered data written to {OUTPUT_PATH} ({len(df)} rows)")
    return OUTPUT_PATH

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_model()
