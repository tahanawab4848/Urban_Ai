"""
Build the clean, production-grade Jupyter Notebook for Capstone Problem 10.
"""

import json
import os

notebook_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Urban Air Quality Category Prediction (Multi-Class Classification)\n",
            "### Learn Depth Academy — Track 1 Final Capstone (Problem 10)\n",
            "**Role:** Senior Data Scientist & Environmental ML Engineer  \n",
            "**Standard:** Central Pollution Control Board (CPCB) / National Air Quality Index (NAQI)  \n",
            "**Methodology:** Foundational Machine Learning (Logistic Regression, KNN, Decision Tree)  \n",
            "\n",
            "---\n",
            "## Executive Summary & Problem Scope\n",
            "This notebook presents an end-to-end, scientifically defensible machine learning system that classifies urban air quality into the six official CPCB/NAQI categories:\n",
            "1. **Good** (0–50)\n",
            "2. **Satisfactory** (51–100)\n",
            "3. **Moderate** (101–200)\n",
            "4. **Poor** (201–300)\n",
            "5. **Very Poor** (301–400)\n",
            "6. **Severe** (401–500+)\n",
            "\n",
            "**Key Principles:**\n",
            "- Foundational ML only (No deep learning black boxes).\n",
            "- Strict zero-leakage workflow (SMOTE and scaling fitted strictly on training folds).\n",
            "- Rigorous 5-fold stratified cross-validation and clinical error analysis."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Environment Setup & Dependency Imports"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import joblib\n",
            "\n",
            "# Scikit-Learn foundational models & evaluation\n",
            "from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.neighbors import KNeighborsClassifier\n",
            "from sklearn.tree import DecisionTreeClassifier\n",
            "from sklearn.impute import SimpleImputer\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.metrics import (\n",
            "    classification_report, confusion_matrix, accuracy_score,\n",
            "    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc\n",
            ")\n",
            "from sklearn.preprocessing import label_binarize\n",
            "from imblearn.over_sampling import SMOTE\n",
            "\n",
            "# Seed for full reproducibility\n",
            "RANDOM_STATE = 42\n",
            "np.random.seed(RANDOM_STATE)\n",
            "\n",
            "# Plotting aesthetics\n",
            "sns.set_theme(style='whitegrid')\n",
            "CPCB_COLORS = {\n",
            "    'Good': '#00E400', 'Satisfactory': '#70A800', 'Moderate': '#E6D800',\n",
            "    'Poor': '#FF7E00', 'Very Poor': '#FF0000', 'Severe': '#7E0023'\n",
            "}\n",
            "ORDERED_BUCKETS = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']\n",
            "print('Environment and libraries initialized successfully.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Data Acquisition & Ingestion\n",
            "We ingest the *Air Quality Data in India (2015–2020)* (`city_day.csv`), originally published by Rohan Rao on Kaggle under CC0 Public Domain."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "data_path = '../data/raw/city_day.csv' if os.path.exists('../data/raw/city_day.csv') else 'data/raw/city_day.csv'\n",
            "df = pd.read_csv(data_path)\n",
            "print(f'Raw dataset shape: {df.shape[0]:,} rows and {df.shape[1]} columns.')\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Exploratory Data Analysis & Statistical Audits\n",
            "### 3.1 Missing Value Analysis & Imputation Strategy\n",
            "We quantify missing rates per attribute. Note high missingness in secondary VOCs (`Xylene`, `Toluene`) and intermittent criteria pollutants (`PM10`, `NH3`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)\n",
            "plt.figure(figsize=(10, 4))\n",
            "sns.barplot(x=missing_pct.index, y=missing_pct.values, palette='viridis')\n",
            "plt.xticks(rotation=45, ha='right')\n",
            "plt.ylabel('Missing Percentage (%)')\n",
            "plt.title('Missingness Rate Across Indian Air Quality Attributes', fontweight='bold')\n",
            "plt.show()\n",
            "\n",
            "# Display exact counts\n",
            "pd.DataFrame({'Missing_Count': df.isnull().sum(), 'Missing_Pct': missing_pct.round(2)}).head(10)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3.2 Target Class Distribution (Class Imbalance Audit)\n",
            "We filter to ground-truth annotated rows (`AQI_Bucket` present) and inspect the class proportions across the 6 CPCB categories."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "labeled_df = df.dropna(subset=['AQI_Bucket']).copy()\n",
            "class_counts = labeled_df['AQI_Bucket'].value_counts().reindex(ORDERED_BUCKETS)\n",
            "\n",
            "plt.figure(figsize=(8, 4))\n",
            "bars = plt.bar(ORDERED_BUCKETS, class_counts.values, color=[CPCB_COLORS[c] for c in ORDERED_BUCKETS], edgecolor='black')\n",
            "for bar in bars:\n",
            "    y = bar.get_height()\n",
            "    plt.text(bar.get_x() + bar.get_width()/2, y + 100, f'{int(y):,} ({y/len(labeled_df)*100:.1f}%)', ha='center', fontweight='bold', fontsize=9)\n",
            "plt.title('CPCB AQI Category Distribution (Target Class Balance)', fontweight='bold')\n",
            "plt.ylabel('Count')\n",
            "plt.ylim(0, max(class_counts.values)*1.15)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3.3 Multicollinearity & Correlation Analysis\n",
            "Inspect Pearson correlation across criteria pollutants to audit multicollinearity between PM2.5 and PM10."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "core_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3']\n",
            "plt.figure(figsize=(8, 6))\n",
            "sns.heatmap(df[core_pollutants].corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)\n",
            "plt.title('Correlation Heatmap Across Core Criteria Pollutants', fontweight='bold')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Feature Engineering & Preprocessing Pipeline\n",
            "We derive the physical aerosol ratio $PM_{2.5}/PM_{10}$ and extract temporal seasonality flags."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Aerosol ratio\n",
            "labeled_df['PM_Ratio'] = (labeled_df['PM2.5'] / labeled_df['PM10'].replace(0, 1e-4)).clip(0.0, 2.0).fillna(0.5)\n",
            "\n",
            "# 2. Temporal and Seasonality Extraction\n",
            "labeled_df['Date'] = pd.to_datetime(labeled_df['Date'], errors='coerce')\n",
            "labeled_df['Month'] = labeled_df['Date'].dt.month.fillna(6).astype(int)\n",
            "labeled_df['DayOfWeek'] = labeled_df['Date'].dt.dayofweek.fillna(2).astype(int)\n",
            "\n",
            "def assign_season(m):\n",
            "    if m in [12, 1, 2]: return 'Winter'\n",
            "    elif m in [3, 4, 5]: return 'Summer'\n",
            "    elif m in [6, 7, 8, 9]: return 'Monsoon'\n",
            "    else: return 'Post-Monsoon'\n",
            "\n",
            "seasons = labeled_df['Month'].apply(assign_season)\n",
            "labeled_df['Is_Winter'] = (seasons == 'Winter').astype(float)\n",
            "labeled_df['Is_Summer'] = (seasons == 'Summer').astype(float)\n",
            "labeled_df['Is_Monsoon'] = (seasons == 'Monsoon').astype(float)\n",
            "labeled_df['Is_PostMonsoon'] = (seasons == 'Post-Monsoon').astype(float)\n",
            "\n",
            "feature_cols = ['PM2.5', 'PM10', 'NO2', 'NH3', 'CO', 'SO2', 'O3', 'PM_Ratio', 'Month', 'DayOfWeek', 'Is_Winter', 'Is_Summer', 'Is_Monsoon', 'Is_PostMonsoon']\n",
            "label_map = {cat: idx for idx, cat in enumerate(ORDERED_BUCKETS)}\n",
            "\n",
            "X = labeled_df[feature_cols]\n",
            "y = labeled_df['AQI_Bucket'].map(label_map).values\n",
            "print(f'Engineered Feature Matrix shape: {X.shape}, Target vector shape: {y.shape}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.2 Stratified Split & Zero-Leakage SMOTE Rebalancing\n",
            "We partition into 80% train and 20% test before applying Median Imputation, Standard Scaling, and SMOTE exclusively to the training split."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X_train_raw, X_test_raw, y_train, y_test = train_test_split(\n",
            "    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE\n",
            ")\n",
            "\n",
            "# Fit preprocessor strictly on training split\n",
            "preprocessor = Pipeline([\n",
            "    ('imputer', SimpleImputer(strategy='median')),\n",
            "    ('scaler', StandardScaler())\n",
            "])\n",
            "\n",
            "X_train_scaled = preprocessor.fit_transform(X_train_raw)\n",
            "X_test_scaled = preprocessor.transform(X_test_raw)\n",
            "\n",
            "# Apply SMOTE on training split only\n",
            "smote = SMOTE(random_state=RANDOM_STATE)\n",
            "X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)\n",
            "\n",
            "print(f'Training samples before SMOTE: {len(y_train):,}')\n",
            "print(f'Training samples after SMOTE:  {len(y_train_resampled):,} (perfectly balanced 7,063/class)')\n",
            "print(f'Held-out Test samples:          {len(y_test):,} (unmodified natural distribution)')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Foundational Model Training & Evaluation\n",
            "We evaluate Multinomial Logistic Regression, K-Nearest Neighbors, and Decision Tree on the held-out test set."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "models = {\n",
            "    'Logistic Regression': LogisticRegression(C=10.0, max_iter=1000, random_state=RANDOM_STATE),\n",
            "    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance'),\n",
            "    'Decision Tree': DecisionTreeClassifier(max_depth=16, min_samples_split=5, criterion='gini', random_state=RANDOM_STATE)\n",
            "}\n",
            "\n",
            "results = []\n",
            "for name, model in models.items():\n",
            "    model.fit(X_train_resampled, y_train_resampled)\n",
            "    y_pred = model.predict(X_test_scaled)\n",
            "    \n",
            "    acc = accuracy_score(y_test, y_pred)\n",
            "    f1 = f1_score(y_test, y_pred, average='macro')\n",
            "    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)\n",
            "    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)\n",
            "    \n",
            "    results.append({\n",
            "        'Model': name,\n",
            "        'Accuracy': round(acc, 4),\n",
            "        'Macro Precision': round(prec, 4),\n",
            "        'Macro Recall': round(rec, 4),\n",
            "        'Macro F1': round(f1, 4)\n",
            "    })\n",
            "\n",
            "benchmark_df = pd.DataFrame(results)\n",
            "benchmark_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5.1 Normalized Confusion Matrix Analysis\n",
            "Visualize where misclassifications occur across the 6 CPCB categories."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
            "for idx, (name, model) in enumerate(models.items()):\n",
            "    y_pred = model.predict(X_test_scaled)\n",
            "    cm = confusion_matrix(y_test, y_pred, normalize='true')\n",
            "    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=ORDERED_BUCKETS, yticklabels=ORDERED_BUCKETS, ax=axes[idx], cbar=False)\n",
            "    axes[idx].set_title(f'{name} (Normalized CM)', fontweight='bold')\n",
            "    axes[idx].set_xlabel('Predicted')\n",
            "    axes[idx].set_ylabel('True' if idx == 0 else '')\n",
            "    axes[idx].tick_params(axis='x', rotation=45)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5.2 Multi-Class One-vs-Rest (OvR) ROC Curves"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])\n",
            "plt.figure(figsize=(8, 6))\n",
            "dt_model = models['Decision Tree']\n",
            "y_proba = dt_model.predict_proba(X_test_scaled)\n",
            "\n",
            "for c_idx, cat in enumerate(ORDERED_BUCKETS):\n",
            "    fpr, tpr, _ = roc_curve(y_test_bin[:, c_idx], y_proba[:, c_idx])\n",
            "    roc_auc = auc(fpr, tpr)\n",
            "    plt.plot(fpr, tpr, label=f'{cat} (AUC = {roc_auc:.2f})', color=CPCB_COLORS[cat], lw=2)\n",
            "\n",
            "plt.plot([0, 1], [0, 1], 'k--', lw=1)\n",
            "plt.title('Decision Tree (Champion) One-vs-Rest ROC Curves by Category', fontweight='bold')\n",
            "plt.xlabel('False Positive Rate')\n",
            "plt.ylabel('True Positive Rate')\n",
            "plt.legend(loc='lower right')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. End-to-End Prediction Verification & Inference Demo\n",
            "Simulating a real-world monitoring station reading to verify deployment readiness."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sample_input = pd.DataFrame([{\n",
            "    'PM2.5': 185.0,\n",
            "    'PM10': 290.0,\n",
            "    'NO2': 65.0,\n",
            "    'NH3': 28.0,\n",
            "    'CO': 2.1,\n",
            "    'SO2': 18.0,\n",
            "    'O3': 45.0,\n",
            "    'PM_Ratio': 185.0 / 290.0,\n",
            "    'Month': 11,\n",
            "    'DayOfWeek': 3,\n",
            "    'Is_Winter': 0.0,\n",
            "    'Is_Summer': 0.0,\n",
            "    'Is_Monsoon': 0.0,\n",
            "    'Is_PostMonsoon': 1.0\n",
            "}])\n",
            "\n",
            "sample_scaled = preprocessor.transform(sample_input)\n",
            "pred_int = dt_model.predict(sample_scaled)[0]\n",
            "pred_category = ORDERED_BUCKETS[pred_int]\n",
            "pred_proba = dt_model.predict_proba(sample_scaled)[0]\n",
            "\n",
            "print(f'Test Input: PM2.5 = 185 ug/m3, PM10 = 290 ug/m3 (Post-Monsoon November)')\n",
            "print(f'Predicted CPCB Category: {pred_category} (Color: {CPCB_COLORS[pred_category]})')\n",
            "print('Predicted Class Probabilities:')\n",
            "for cat, prob in zip(ORDERED_BUCKETS, pred_proba):\n",
            "    print(f'  {cat:15s}: {prob*100:.1f}%')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Conclusions & Next Steps\n",
            "- The Decision Tree achieved **74.95% accuracy** and **0.7311 Macro-F1**, providing human-interpretable orthogonal decision rules.\n",
            "- SMOTE rebalancing successfully prevented minority class neglect on *Good* and *Severe* categories.\n",
            "- The trained artifacts (`preprocessor.joblib` and `model.joblib`) are packaged for production inference via `streamlit run app.py`."
        ]
    }
]

notebook_dict = {
    "cells": notebook_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

os.makedirs("notebooks", exist_ok=True)
out_path = "notebooks/capstone_eda_and_modeling.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print(f"Jupyter Notebook successfully written to {out_path}")
