import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

SKEWED_FEATURES = [
    # NOTE: both average_submissions_attempted_per_15_minutes and
    # average_takedowns_landed_per_15_minutes used to be log-transformed here.
    # Log-compressing them masked two real archetypes: elite wrestlers (5+
    # takedowns/15min, e.g. Khabib Nurmagomedov) and submission specialists
    # (10+ submission attempts/15min, e.g. known BJJ competitors like Robert
    # Drysdale) both got diluted into the generic striker cluster. Left raw
    # (still standard-scaled below) so these extremes survive; the extra
    # weighting that separates them into their own clusters happens in
    # feature_engineering.py.
]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in STYLE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df.dropna(subset=STYLE_FEATURES).copy()

    all_zero_mask = (df[STYLE_FEATURES] == 0).all(axis=1)
    dropped_count = all_zero_mask.sum()
    if dropped_count > 0:
        logger.warning(f"Dropped {dropped_count} rows where all style stats were 0.")
        df = df[~all_zero_mask].copy()

    return df

def scale_features(df: pd.DataFrame, scaler_path: Path, fit: bool = True) -> pd.DataFrame:
    df = df.copy()

    for col in SKEWED_FEATURES:
        df[col] = np.log1p(df[col])

    if fit:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[STYLE_FEATURES])
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        scaled_data = scaler.transform(df[STYLE_FEATURES])

    scaled_cols = [f"{c}_scaled" for c in STYLE_FEATURES]
    scaled_df = pd.DataFrame(scaled_data, columns=scaled_cols, index=df.index)

    return pd.concat([df, scaled_df], axis=1)

def preprocess(raw_path: Path, output_path: Path, scaler_path: Path, fit: bool = True) -> Path:
    logger.info(f"Starting preprocessing for {raw_path}")
    df = pd.read_csv(raw_path)
    df = clean_data(df)
    df = scale_features(df, scaler_path, fit=fit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Preprocessing complete. Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    BASE = Path(__file__).resolve().parents[1]
    preprocess(
        raw_path=BASE / "data/raw/ufc-fighters-statistics.csv",
        output_path=BASE / "data/processed/fighters_processed.parquet",
        scaler_path=BASE / "models/scaler.pkl",
        fit=True
    )
