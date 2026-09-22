"""
Data Acquisition and Ingestion Module for Urban Air Quality Dataset.

Dataset: Air Quality Data in India (city_day.csv)
Original Author/Curator: Rohan Rao (Kaggle: rohanrao/air-quality-data-in-india)
License: CC0: Public Domain
Standard: CPCB / National Air Quality Index (NAQI) Breakpoints
"""

import os
import urllib.request
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_URLS = [
    "https://raw.githubusercontent.com/shrutisbhosale14/Analysis-Of-Air-Pollution/main/city_day.csv",
    "https://raw.githubusercontent.com/shrutisbhosale14/Analysis-Of-Air-Pollution/master/city_day.csv",
]

# Official CPCB / NAQI Breakpoints
AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]

ORDERED_BUCKETS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def download_dataset(target_path: str = "data/raw/city_day.csv") -> str:
    """Download the raw dataset if not already present."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        logger.info(f"Dataset already exists at {target_path} ({os.path.getsize(target_path):,} bytes).")
        return target_path

    logger.info("Downloading city_day.csv from remote mirrors...")
    last_err = None
    for url in DATA_URLS:
        try:
            logger.info(f"Attempting download from {url}...")
            urllib.request.urlretrieve(url, target_path)
            if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
                logger.info(f"Successfully downloaded to {target_path} ({os.path.getsize(target_path):,} bytes).")
                return target_path
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            last_err = e

    raise RuntimeError(f"Could not download dataset from any mirror. Last error: {last_err}")


def map_aqi_to_bucket(aqi: float) -> str:
    """Map numeric AQI to CPCB standardized category."""
    if pd.isna(aqi):
        return np.nan
    for low, high, bucket in AQI_CATEGORIES:
        if low <= aqi <= high:
            return bucket
    if aqi > 500:
        return "Severe"
    return np.nan


def load_raw_data(data_path: str = "data/raw/city_day.csv") -> pd.DataFrame:
    """Load raw dataset and ensure basic typing."""
    if not os.path.exists(data_path):
        download_dataset(data_path)
    
    logger.info(f"Loading raw data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape {df.shape[0]:,} rows and {df.shape[1]} columns.")
    
    # Ensure Date parsing
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Verify or reconcile AQI_Bucket
    if "AQI" in df.columns and "AQI_Bucket" in df.columns:
        # Reconcile missing AQI_Bucket if AQI is present
        reconciled = df["AQI"].apply(map_aqi_to_bucket)
        df["AQI_Bucket"] = df["AQI_Bucket"].fillna(reconciled)

    return df


if __name__ == "__main__":
    download_dataset()
    df = load_raw_data()
    print("Columns:", list(df.columns))
    print("Null counts:\n", df.isnull().sum())
    print("\nAQI_Bucket distribution:\n", df["AQI_Bucket"].value_counts(dropna=False))
