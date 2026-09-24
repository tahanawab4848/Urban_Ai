"""
Data Preparation, Feature Engineering, and Preprocessing Pipeline Module.
Implements ColumnTransformer, physical aerosol ratio engineering,
temporal seasonality extraction, and training-fold SMOTE rebalancing.
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.data_loader import load_raw_data, ORDERED_BUCKETS
from src.utils import ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Core CPCB Pollutants
RAW_POLLUTANT_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]

# Target mapping dictionary
LABEL_TO_INT = {cat: idx for idx, cat in enumerate(ORDERED_BUCKETS)}
INT_TO_LABEL = {idx: cat for idx, cat in enumerate(ORDERED_BUCKETS)}


def get_season(month: int) -> str:
    """Classify month into Indian meteorological season."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


class AirQualityFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer that derives domain-specific aerosol ratios
    and temporal seasonality indicators.
    """
    def __init__(self):
        self.engineered_feature_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Derive:
        1. PM_Ratio = PM2.5 / (PM10 + 1e-4) -> fine-to-coarse ratio
        2. Month, DayOfWeek, Season indicators (if Date or temporal cols present)
        """
        df = pd.DataFrame(X).copy()

        # Aerosol ratio: PM2.5 to PM10
        if "PM2.5" in df.columns and "PM10" in df.columns:
            pm10_safe = np.where(df["PM10"] <= 0, 1e-4, df["PM10"])
            df["PM_Ratio"] = (df["PM2.5"] / pm10_safe).clip(0.0, 2.0)
            # Impute NaN ratio where PM2.5 or PM10 was NaN
            df["PM_Ratio"] = df["PM_Ratio"].fillna(0.5)
        else:
            df["PM_Ratio"] = 0.5

        # Handle Date or temporal features
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Month"] = df["Date"].dt.month.fillna(6).astype(int)
            df["DayOfWeek"] = df["Date"].dt.dayofweek.fillna(2).astype(int)
        else:
            if "Month" not in df.columns:
                df["Month"] = 6
            if "DayOfWeek" not in df.columns:
                df["DayOfWeek"] = 2

        # Seasonality flags
        seasons = df["Month"].apply(get_season)
        df["Is_Winter"] = (seasons == "Winter").astype(float)
        df["Is_Summer"] = (seasons == "Summer").astype(float)
        df["Is_Monsoon"] = (seasons == "Monsoon").astype(float)
        df["Is_PostMonsoon"] = (seasons == "Post-Monsoon").astype(float)

        # Drop non-feature columns if present
        cols_to_drop = [c for c in ["City", "Date", "AQI", "AQI_Bucket", "NO", "NOx", "Benzene", "Toluene", "Xylene"] if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        self.engineered_feature_names_ = list(df.columns)
        return df


def build_preprocessor_pipeline(feature_cols: list) -> ColumnTransformer:
    """Build a ColumnTransformer with median imputation and StandardScaler."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, feature_cols)
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_capstone_data(
    raw_path: str = "data/raw/city_day.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Execute complete end-to-end data preparation:
    1. Filter labeled rows (AQI_Bucket present).
    2. Extract engineered features.
    3. Stratified 80/20 train/test split.
    4. Fit ColumnTransformer on train split ONLY.
    5. Apply SMOTE to training partition ONLY.
    6. Serialize fitted preprocessor artifact.
    """
    ensure_dir(output_dir)
    ensure_dir("artifacts")

    logger.info("Loading raw data for preprocessing...")
    df = load_raw_data(raw_path)

    # 1. Filter labeled records
    clean_df = df.dropna(subset=["AQI_Bucket"]).copy()
    logger.info(f"Retained {len(clean_df):,} records with valid AQI_Bucket labels.")

    # 2. Extract features via AirQualityFeatureEngineer
    engineer = AirQualityFeatureEngineer()
    X_engineered = engineer.transform(clean_df)
    feature_columns = list(X_engineered.columns)
    logger.info(f"Engineered feature set ({len(feature_columns)} features): {feature_columns}")

    # Map target to integer labels
    y_raw = clean_df["AQI_Bucket"].map(LABEL_TO_INT).values

    # 3. Stratified Train-Test Split (80/20)
    logger.info(f"Performing Stratified Train-Test Split (test_size={test_size}, random_state={random_state})...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_engineered,
        y_raw,
        test_size=test_size,
        stratify=y_raw,
        random_state=random_state
    )

    # 4. Fit Preprocessor on X_train_raw ONLY
    preprocessor = build_preprocessor_pipeline(feature_columns)
    logger.info("Fitting preprocessor on training data...")
    X_train_scaled = preprocessor.fit_transform(X_train_raw)
    X_test_scaled = preprocessor.transform(X_test_raw)

    # Save artifacts for inference
    preprocessor_artifact = {
        "preprocessor": preprocessor,
        "feature_columns": feature_columns,
        "engineer": engineer,
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL,
        "cpcb_categories": ORDERED_BUCKETS
    }
    artifact_path = "artifacts/preprocessor.joblib"
    joblib.dump(preprocessor_artifact, artifact_path)
    logger.info(f"Serialized preprocessor artifact to {artifact_path}")

    # 5. Apply SMOTE strictly on training split
    logger.info("Applying SMOTE on training split only to rebalance minority classes...")
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    train_dist_before = pd.Series(y_train).value_counts().sort_index().to_dict()
    train_dist_after = pd.Series(y_train_resampled).value_counts().sort_index().to_dict()
    logger.info(f"Train class distribution before SMOTE: {train_dist_before}")
    logger.info(f"Train class distribution after SMOTE: {train_dist_after}")

    # 6. Save processed datasets for reproducible experiments
    train_save_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
    train_save_df["Target"] = y_train
    train_save_df.to_csv(os.path.join(output_dir, "train_scaled_original.csv"), index=False)

    train_resampled_df = pd.DataFrame(X_train_resampled, columns=feature_columns)
    train_resampled_df["Target"] = y_train_resampled
    train_resampled_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)

    test_save_df = pd.DataFrame(X_test_scaled, columns=feature_columns)
    test_save_df["Target"] = y_test
    test_save_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    logger.info(f"Saved processed train ({X_train_resampled.shape[0]} rows) and test ({X_test_scaled.shape[0]} rows) to {output_dir}")

    return {
        "X_train": X_train_resampled,
        "y_train": y_train_resampled,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "feature_columns": feature_columns
    }


if __name__ == "__main__":
    prepare_capstone_data()
