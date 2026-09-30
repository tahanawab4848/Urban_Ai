# Urban Air Quality Category Prediction (Multi-Class Classification)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![License: CC0](https://img.shields.io/badge/License-CC0-green.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)

**Learn Depth Academy — Track 1 Final Capstone (Problem 10)**  
**Target Standard:** Central Pollution Control Board (CPCB) / National Air Quality Index (NAQI)  
**Methodology:** Foundational Machine Learning (Logistic Regression, KNN, Decision Trees)  
**Constraints:** No Deep Learning, 100% Free & Open-Source Tools, Full Reproducibility (`random_state=42`).

---

## 1. Executive Summary & Problem Scope

Ambient air pollution in rapidly urbanizing regions poses acute public health hazards. Municipal authorities enforce environmental interventions (e.g. Graded Response Action Plan - GRAP, traffic odd-even rationing, school closures) based on categorized air quality indices rather than raw numerical readings. 

This project delivers a complete, reproducible, end-to-end Machine Learning system that predicts CPCB NAQI air quality categories across Indian metropolitan areas:
1. **Good** (AQI 0–50) — Minimal impact
2. **Satisfactory** (AQI 51–100) — Minor breathing discomfort to sensitive individuals
3. **Moderate** (AQI 101–200) — Breathing discomfort with lung/heart disease
4. **Poor** (AQI 201–300) — Breathing discomfort to most people on prolonged exposure
5. **Very Poor** (AQI 301–400) — Respiratory illness on prolonged exposure
6. **Severe** (AQI 401–500+) — Severe health emergency affecting healthy populations

---

## 2. Dataset Ingestion & Citation

- **Dataset:** *Air Quality Data in India (2015–2020)*, `city_day.csv`
- **Curator:** Rohan Rao (Kaggle: `rohanrao/air-quality-data-in-india`)
- **Authority:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest & Climate Change, Government of India.
- **License:** Creative Commons CC0: Public Domain
- **Scale:** 29,531 daily records across 26 major Indian cities (Delhi, Bengaluru, Hyderabad, Mumbai, etc.).
- **Supervised Cohort:** 24,850 ground-truth annotated records.

```bibtex
@misc{rao2020airqualityindia,
  author = {Rao, Rohan},
  title = {Air Quality Data in India (2015-2020)},
  year = {2020},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india}}
}
```

---

## 3. Empirical Results & Benchmark Comparison

The three foundational models were tuned using **5-Fold Stratified Cross-Validation** on 42,378 balanced training instances and evaluated on **4,970 held-out real-world test instances** (un-resampled 20% stratified test partition):

| Model Architecture | Optimal Hyperparameters | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | OvR ROC-AUC | Inference Latency |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | `C=10.0, solver='lbfgs', weight=None` | **69.88%** | 0.6542 | **0.7429** | **0.6896** | **0.9414** | 0.003 ms |
| **K-Nearest Neighbors** | `n_neighbors=5, metric='manhattan', weights='distance'` | **70.38%** | 0.6698 | **0.7288** | **0.6900** | **0.9041** | 2.500 ms |
| **Decision Tree ⭐** | `max_depth=16, min_samples_split=5, criterion='gini'` | **74.95%** | **0.7214** | **0.7532** | **0.7311** | **0.8789** | **0.040 ms** |

### Key Diagnostic Findings:
- **Champion Model:** **Decision Tree Classifier** achieved **74.95% Accuracy** and **0.7311 Macro-F1**, providing explicit, human-auditable orthogonal decision rules aligned with environmental regulations.
- **Zero-Leakage SMOTE Impact:** Minority classes (*Good* and *Severe*, each ~5.4% in the raw dataset) achieved high recall ($> 75\%$), eliminating dangerous false-negative alerts on severe toxic episodes.
- **Atmospheric Boundary Overlap:** The highest confusion occurs between contiguous categories (*Moderate* and *Poor*), where atmospheric pollutant concentrations transition along a continuous chemical gradient. Zero catastrophic misclassifications (*Severe* predicted as *Good*) occurred.

---

## 4. Project Architecture & Directory Layout

```
urban-air-quality-capstone/
├── data/
│   ├── raw/
│   │   └── city_day.csv          # Raw CPCB monitoring dataset (29,531 records)
│   ├── processed/
│   │   ├── train.csv             # SMOTE-balanced training split (42,378 records)
│   │   └── test.csv              # Unpolluted held-out test split (4,970 records)
│   └── data_citation.md          # Formal provenance, BibTeX and CPCB metadata
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Automated download & CPCB breakpoint mapping
│   ├── eda.py                    # 8-point statistical investigation & VIF auditing
│   ├── preprocess.py             # ColumnTransformer, PM ratio & SMOTE pipeline
│   ├── train.py                  # 5-fold Stratified GridSearchCV for LR, KNN, DT
│   ├── evaluate.py               # Normalized CM, OvR ROC curves, clinical analysis
│   └── utils.py                  # Color maps, plotting aesthetics & JSON helpers
├── artifacts/
│   ├── preprocessor.joblib       # Serialized ColumnTransformer & metadata
│   ├── model.joblib              # Serialized Champion Model bundle
│   ├── metrics.json              # Full performance logs across all models
│   ├── figures/                  # 10 publication-quality diagnostic plots
│   └── reports/                  # Phase 1 EDA & Phase 3 Evaluation reports
├── notebooks/
│   └── capstone_eda_and_modeling.ipynb # Fully annotated reproducible notebook
├── docs/
│   ├── technical_paper.md        # Academic conference/journal-style paper
│   ├── presentation_slides.md    # 15-slide capstone defense presentation deck
│   └── viva_qa.md                # 20+ oral defense / viva voce technical Q&A
├── app.py                        # Interactive Streamlit Web Application
├── requirements.txt              # Pinned production dependencies
└── README.md                     # Documentation & project manual
```

---

## 5. Quick Start & Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Git

### 1. Clone & Install Dependencies
```bash
git clone <repo-url>
cd "urban-air-quality-capstone"
pip install -r requirements.txt
```

### 2. Run Complete End-to-End Pipeline
Each module can be executed independently or sequentially:

```bash
# Phase 1: Ingest dataset and run exhaustive statistical investigation
python src/eda.py

# Phase 2: Feature engineering, ColumnTransformer, and training-fold SMOTE
python src/preprocess.py

# Phase 3: Train and tune Logistic Regression, KNN, and Decision Tree
python src/train.py

# Phase 3 Evaluation: Generate confusion matrices, ROC curves, and reports
python src/evaluate.py
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The app features:
- Interactive criteria pollutant sliders with physical validation bounds.
- Real-time classification with official CPCB hex color coding.
- Posterior category probability distribution bars.
- Actionable clinical health advisories for general and sensitive populations.
- Interactive "What-If" emission reduction simulator.

---

## 6. Scientific Governance & Reproducibility Guarantee

1. **Fixed Random State:** All stochastic operations (splits, SMOTE, estimators, grid search) are locked to `random_state=42`.
2. **Leakage Firewall:**
   - Numerical `SimpleImputer(strategy='median')` and `StandardScaler` are fitted strictly on $X_{\text{train}}$.
   - SMOTE is applied **strictly** to the training fold after train-test partitioning.
   - The test set reflects genuine real-world atmospheric class distributions.
3. **Open Access:** Built entirely using free, open-source scientific Python libraries (`scikit-learn`, `imbalanced-learn`, `pandas`, `streamlit`, `seaborn`).
