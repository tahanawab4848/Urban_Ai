"""
Model Training and Hyperparameter Optimization Module.
Orchestrates 5-Fold Stratified Cross-Validation with GridSearchCV for:
1. Multinomial Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree Classifier
Logs comprehensive performance metrics and exports the champion model.
"""

import os
import sys
import time
import logging
import joblib
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from src.utils import ensure_dir, save_json
from src.data_loader import ORDERED_BUCKETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_processed_splits(data_dir: str = "data/processed"):
    """Load train and test splits."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Processed datasets not found in {data_dir}. Run preprocess.py first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c != "Target"]
    X_train = train_df[feature_cols].values
    y_train = train_df["Target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["Target"].values

    return X_train, y_train, X_test, y_test, feature_cols


def run_training_and_tuning(random_state: int = 42):
    """Execute grid search across LR, KNN, and Decision Tree."""
    ensure_dir("artifacts")
    X_train, y_train, X_test, y_test, feature_cols = load_processed_splits()
    logger.info(f"Loaded training set: {X_train.shape[0]:,} samples, test set: {X_test.shape[0]:,} samples.")

    # Define hyperparameter search spaces
    models_config = {
        "Logistic_Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=random_state),
            "param_grid": {
                "C": [0.1, 1.0, 10.0],
                "solver": ["lbfgs"],
                "class_weight": [None, "balanced"]
            }
        },
        "K_Nearest_Neighbors": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [5, 11, 21],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        },
        "Decision_Tree": {
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "param_grid": {
                "max_depth": [8, 12, 16],
                "min_samples_split": [5, 10],
                "criterion": ["gini", "entropy"]
            }
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}
    best_estimators = {}

    for name, config in models_config.items():
        logger.info(f"\n{'='*50}\nStarting 5-Fold Stratified Tuning for {name}...\n{'='*50}")
        start_time = time.time()
        grid = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=2,  # Stable concurrency on Windows
            verbose=1
        )
        grid.fit(X_train, y_train)
        tuning_duration = time.time() - start_time

        best_model = grid.best_estimator_
        best_estimators[name] = best_model

        logger.info(f"{name} Best Params: {grid.best_params_}")
        logger.info(f"{name} Best CV Macro-F1: {grid.best_score_:.4f} (elapsed {tuning_duration:.2f}s)")

        # Evaluate on held-out test set
        test_start = time.time()
        y_pred = best_model.predict(X_test)
        inference_latency_ms = ((time.time() - test_start) / len(X_test)) * 1000

        # Predict probabilities if supported
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(X_test)
            try:
                ovr_roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC for {name}: {e}")
                ovr_roc_auc = None
        else:
            ovr_roc_auc = None

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        report_dict = classification_report(y_test, y_pred, target_names=ORDERED_BUCKETS, output_dict=True, zero_division=0)

        results[name] = {
            "best_params": grid.best_params_,
            "best_cv_macro_f1": float(grid.best_score_),
            "tuning_time_seconds": float(round(tuning_duration, 2)),
            "inference_latency_per_sample_ms": float(round(inference_latency_ms, 4)),
            "test_accuracy": round(acc, 4),
            "test_macro_precision": round(prec, 4),
            "test_macro_recall": round(rec, 4),
            "test_macro_f1": round(f1, 4),
            "test_ovr_roc_auc": round(ovr_roc_auc, 4) if ovr_roc_auc is not None else None,
            "classification_report": report_dict
        }

        logger.info(f"{name} Test Set Evaluation:")
        logger.info(f"  Accuracy:  {acc:.4f}")
        logger.info(f"  Macro-F1:  {f1:.4f}")
        logger.info(f"  Macro-Rec: {rec:.4f}")
        logger.info(f"  ROC-AUC:   {ovr_roc_auc if ovr_roc_auc is not None else 'N/A'}")

    # Determine Champion Model based on held-out Macro-F1
    champion_name = max(results.keys(), key=lambda k: results[k]["test_macro_f1"])
    champion_model = best_estimators[champion_name]
    logger.info(f"\n{'*'*50}\nCHAMPION MODEL SELECTED: {champion_name} (Macro-F1: {results[champion_name]['test_macro_f1']})\n{'*'*50}")

    # Export metrics JSON
    metrics_path = "artifacts/metrics.json"
    save_json({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion_model": champion_name,
        "models": results
    }, metrics_path)
    logger.info(f"Exported metrics log to {metrics_path}")

    # Export Champion Model artifact
    champion_artifact = {
        "model_name": champion_name,
        "model": champion_model,
        "all_best_models": best_estimators,
        "feature_names": feature_cols,
        "target_names": ORDERED_BUCKETS
    }
    model_artifact_path = "artifacts/model.joblib"
    joblib.dump(champion_artifact, model_artifact_path)
    logger.info(f"Serialized champion model to {model_artifact_path}")

    return champion_artifact, results


if __name__ == "__main__":
    run_training_and_tuning()
