"""
fighting_style_clustering_dag.py

Orchestrates the fighting-style clustering pipeline. Each task calls the
already-tested entry point from src/ — no business logic here, only wiring.

Flow: preprocess -> engineer_features -> train_model -> evaluate_model
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/opt/airflow/src")

from airflow.decorators import dag, task

BASE = Path("/opt/airflow")

default_args = {
    "owner": "shavraul",
    "retries": 1,
}


@dag(
    dag_id="fighting_style_clustering",
    description="Cluster UFC fighters by fighting style",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ml", "clustering", "ufc"],
)
def fighting_style_clustering_dag():

    @task()
    def preprocess():
        import preprocessing

        return str(preprocessing.preprocess(
            raw_path=BASE / "data/raw/ufc-fighters-statistics.csv",
            output_path=BASE / "data/processed/fighters_processed.parquet",
            scaler_path=BASE / "models/scaler.pkl",
            fit=True,
        ))

    @task()
    def engineer_features(processed_path: str):
        import feature_engineering

        return str(feature_engineering.engineer_features(fit_pca=True))

    @task()
    def train(features_path: str):
        import train_model

        return str(train_model.train_model(k_override=4))

    @task()
    def evaluate(clustered_path: str):
        import evaluate_model

        return str(evaluate_model.evaluate())

    processed_path = preprocess()
    features_path = engineer_features(processed_path)
    clustered_path = train(features_path)
    evaluate(clustered_path)


fighting_style_clustering_dag()