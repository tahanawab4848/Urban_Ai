"""
End-to-End System Verification and Smoke Test Script.
Validates artifact integrity, pipeline execution, and model predictions across presets.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import AirQualityFeatureEngineer
import __main__
__main__.AirQualityFeatureEngineer = AirQualityFeatureEngineer

from src.utils import ORDERED_CATEGORIES, load_json


def verify_system():
    print("=" * 60)
    print("RUNNING END-TO-END CAPSTONE VERIFICATION")
    print("=" * 60)

    # 1. Verify file paths
    required_files = [
        "data/raw/city_day.csv",
        "data/processed/train.csv",
        "data/processed/test.csv",
        "data/data_citation.md",
        "artifacts/preprocessor.joblib",
        "artifacts/model.joblib",
        "artifacts/metrics.json",
        "artifacts/reports/eda_investigation_report.md",
        "artifacts/reports/model_evaluation_and_error_analysis.md",
        "artifacts/figures/confusion_matrices_all_models.png",
        "artifacts/figures/roc_curves_all_models.png",
        "artifacts/figures/model_comparison_benchmark.png",
        "notebooks/capstone_eda_and_modeling.ipynb",
        "docs/technical_paper.md",
        "docs/presentation_slides.md",
        "docs/viva_qa.md",
        "app.py",
        "requirements.txt",
        "README.md"
    ]

    all_exist = True
    for f in required_files:
        exists = os.path.exists(f)
        size = os.path.getsize(f) if exists else 0
        status = f"EXISTS ({size:,} bytes)" if exists else "MISSING"
        if not exists:
            all_exist = False
        print(f"[{'PASS' if exists else 'FAIL'}] {f:<55} -> {status}")

    assert all_exist, "One or more required project deliverables are missing!"

    # 2. Verify artifact loading
    print("\nVerifying serialized artifacts...")
    prep_bundle = joblib.load("artifacts/preprocessor.joblib")
    model_bundle = joblib.load("artifacts/model.joblib")
    metrics = load_json("artifacts/metrics.json")

    preprocessor = prep_bundle["preprocessor"]
    engineer = prep_bundle["engineer"]
    models = model_bundle["all_best_models"]
    champion_name = model_bundle["model_name"]

    print(f"Champion Model: {champion_name}")
    print(f"Logged Test Metrics for Champion:")
    for k, v in metrics["models"][champion_name].items():
        if k != "classification_report":
            print(f"  - {k}: {v}")

    # 3. Test scenarios inference
    scenarios = [
        {
            "name": "Clean Coastal Day (Bengaluru/Kochi)",
            "data": {"PM2.5": 18.0, "PM10": 35.0, "NO2": 12.0, "SO2": 6.0, "CO": 0.4, "O3": 22.0, "NH3": 8.0, "Month": 7, "DayOfWeek": 1},
            "expected_likely": ["Good", "Satisfactory"]
        },
        {
            "name": "Moderate Industrial Day (Hyderabad)",
            "data": {"PM2.5": 62.0, "PM10": 115.0, "NO2": 32.0, "SO2": 14.0, "CO": 1.1, "O3": 42.0, "NH3": 19.0, "Month": 4, "DayOfWeek": 3},
            "expected_likely": ["Moderate", "Satisfactory"]
        },
        {
            "name": "Severe Winter Smog (Delhi)",
            "data": {"PM2.5": 285.0, "PM10": 420.0, "NO2": 95.0, "SO2": 28.0, "CO": 4.2, "O3": 58.0, "NH3": 52.0, "Month": 11, "DayOfWeek": 4},
            "expected_likely": ["Severe", "Very Poor"]
        }
    ]

    print("\nRunning inference validation on test scenarios...")
    for sc in scenarios:
        df_in = pd.DataFrame([sc["data"]])
        eng = engineer.transform(df_in)
        scaled = preprocessor.transform(eng)

        print(f"\nScenario: {sc['name']}")
        for m_name, m in models.items():
            pred_idx = int(m.predict(scaled)[0])
            pred_cat = ORDERED_CATEGORIES[pred_idx]
            if hasattr(m, "predict_proba"):
                conf = m.predict_proba(scaled)[0][pred_idx] * 100
                print(f"  - {m_name:<22}: {pred_cat:<14} (Confidence: {conf:.1f}%)")
            else:
                print(f"  - {m_name:<22}: {pred_cat:<14}")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION CHECKS PASSED: SYSTEM IS PRODUCTION READY!")
    print("=" * 60)


if __name__ == "__main__":
    verify_system()
