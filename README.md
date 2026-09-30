# Urban Air Quality Category Prediction

> An explainable, reproducible, and defensible foundational machine learning system for multi-class urban air quality classification following Central Pollution Control Board (CPCB) NAQI standards.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64%2B-FF4B4B.svg)](https://streamlit.io/)

---

## 1. Project Title & Mission
**Urban Air Quality Category Prediction (Multi-Class Classification)**  
Developed as an educational prototype for **Learn Depth Academy's Track 1 Final Capstone (Problem 10)**. Built strictly with foundational machine learning algorithms (no deep learning) using free and open-source tools.

---

## 2. Overview
Ambient air pollution in rapidly urbanizing regions poses acute public health hazards and drives severe pulmonary, cardiovascular, and systemic diseases. Municipal environmental governance relies on the **Air Quality Index (AQI)** to convert complex chemical concentration telemetry into discrete, actionable alert levels. In India, the Central Pollution Control Board (CPCB) and Ministry of Environment, Forest and Climate Change define six standardized categories: **Good (0–50)**, **Satisfactory (51–100)**, **Moderate (101–200)**, **Poor (201–300)**, **Very Poor (301–400)**, and **Severe (401–500+)**.

While traditional municipal assessment relies on manual piecewise interpolation across reporting monitors, monitoring stations frequently suffer from sensor dropouts, missing channels, and calibration drift. This project formulates an empirical machine learning pipeline that predicts regulatory CPCB categories directly from criteria particulate and gaseous measurements, domain-engineered aerosol ratios, and temporal seasonality signals.

Unlike opaque deep neural networks, this capstone emphasizes **interpretability, regulatory defensibility, and mathematical transparency**. By deploying foundational algorithms (Multinomial Logistic Regression, K-Nearest Neighbors, and Decision Trees) alongside a zero-leakage training-fold SMOTE pipeline, this project delivers high sensitivity on life-threatening atmospheric emergencies while providing sub-millisecond inference and human-auditable decision rules for municipal decision-makers.

---

## 3. Dataset
- **Title:** Air Quality Data in India (2015–2020)
- **Primary Data File:** `city_day.csv`
- **Host / Curator:** Rohan Rao (Kaggle: [`rohanrao/air-quality-data-in-india`](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india))
- **Original Source:** Central Pollution Control Board (CPCB), Government of India ([https://cpcb.nic.in](https://cpcb.nic.in))
- **Scale:** 29,531 daily observations across 26 major Indian urban centers (e.g., Delhi, Bengaluru, Hyderabad, Mumbai, Kolkata, Chennai).
- **Supervised Universe:** 24,850 ground-truth annotated records.
- **Criteria Pollutants:** PM2.5, PM10, NO2, SO2, CO, O3, NH3.
- **Secondary Species:** NO, NOx, Benzene, Toluene, Xylene.
- **Target Feature:** `AQI_Bucket` (6-class ordinal label: Good, Satisfactory, Moderate, Poor, Very Poor, Severe).
- **License:** Creative Commons CC0: Public Domain.
- **Citation:** See [docs/dataset_citation.md](docs/dataset_citation.md) for APA, MLA, and BibTeX citations.

---

## 4. Project Structure

```
Urban_Ai/
├── data/
│   ├── raw/
│   │   └── city_day.csv          # Raw CPCB monitoring dataset (29,531 records)
│   ├── processed/
│   │   ├── train.csv             # Final SMOTE-balanced training split (42,378 records)
│   │   └── test.csv              # Natural held-out test split (4,970 records)
│   └── data_citation.md          # Provenance, BibTeX citation, and CC0 license
├── src/
│   ├── __init__.py               # Package initializer
│   ├── data_loader.py            # Automated download & CPCB breakpoint mapping
│   ├── eda.py                    # 8-point statistical investigation & VIF auditing
│   ├── preprocess.py             # ColumnTransformer, aerosol ratio & SMOTE pipeline
│   ├── train.py                  # 5-fold Stratified GridSearchCV for LR, KNN, DT
│   ├── evaluate.py               # Normalized CM, OvR ROC curves & clinical error analysis
│   ├── utils.py                  # Color maps, plotting aesthetics & JSON helpers
│   └── verify_system.py          # End-to-end verification and smoke testing script
├── artifacts/
│   ├── preprocessor.joblib       # Serialized ColumnTransformer & metadata
│   ├── model.joblib              # Serialized Champion Model bundle
│   ├── metrics.json              # Full performance logs across all models
│   ├── figures/                  # 10 diagnostic plots (CMs, ROC curves, VIF, distributions)
│   └── reports/                  # Phase 1 EDA & Phase 3 Evaluation reports
├── notebooks/
│   └── capstone_eda_and_modeling.ipynb # Fully annotated reproducible notebook
├── docs/
│   ├── technical_paper.md        # 12-section IEEE/ACM style research paper
│   ├── presentation_slides.md    # 15-slide capstone defense presentation deck
│   ├── viva_questions.md         # 50 viva voce technical defense questions and answers
│   ├── submission_checklist.md   # Rubric compliance verification checklist
│   ├── milestone_log.md          # Daily execution log across D1-D10 milestones
│   ├── dataset_citation.md       # Comprehensive multi-format dataset citations
│   └── CHANGELOG.md              # 188-commit progressive audit trail
├── app.py                        # Interactive Streamlit Web Application
├── requirements.txt              # Pinned production dependencies
├── .gitignore                    # Python, checkpoint, and cache exclusions
└── README.md                     # Comprehensive project documentation manual
```

---

## 5. Installation

### Prerequisites
- Python >= 3.10 (Tested on Python 3.14)
- Git

### Setup Instructions
```bash
# 1. Clone the repository
git clone https://github.com/tahanawab4848/Urban_Ai.git
cd Urban_Ai

# 2. Create and activate a clean virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install pinned production dependencies
pip install -r requirements.txt
```

---

## 6. Usage

### Option A: Launch Interactive Streamlit Application
Run the web dashboard for real-time risk assessment, CPCB color badging, and policy sensitivity simulation:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Option B: Run End-to-End Jupyter Notebook
Launch the step-by-step interactive workflow covering EDA, feature engineering, cross-validation, and scenario testing:
```bash
jupyter notebook notebooks/capstone_eda_and_modeling.ipynb
```

### Option C: Execute Pipeline Modules Sequentially
```bash
# Phase 1: Ingest dataset and run exhaustive statistical investigation
python src/eda.py

# Phase 2: Feature engineering, ColumnTransformer, and training-fold SMOTE
python src/preprocess.py

# Phase 3: Train and tune Logistic Regression, KNN, and Decision Tree
python src/train.py

# Phase 3 Evaluation: Generate confusion matrices, ROC curves, and reports
python src/evaluate.py

# Run complete system verification smoke test
python src/verify_system.py
```

---

## 7. Model Summary

Three foundational classification paradigms were systematically optimized using **5-Fold Stratified Cross-Validation with `GridSearchCV`** on 42,378 balanced training instances, targeting **Macro-averaged F1-Score**:

| Model Architecture | Mathematical Paradigm | Key Hyperparameters Evaluated | Best Configuration | Tuning Latency |
|:---|:---|:---|:---|:---:|
| **Logistic Regression** | Parametric Linear Hyperplane (Softmax) | $C \in \{0.1, 1.0, 10.0\}$, solver, class_weight | `C=10.0, solver='lbfgs', class_weight=None` | ~42s |
| **K-Nearest Neighbors (KNN)** | Non-parametric Spatial Metric Learning | $k \in \{5, 11, 21\}$, metric, weights | `n_neighbors=5, metric='manhattan', weights='distance'` | ~216s |
| **Decision Tree ⭐** | Non-linear Recursive Orthogonal Partitioning | `max_depth` $\in \{8, 12, 16\}$, `min_samples_split`, criterion | `max_depth=16, min_samples_split=5, criterion='gini'` | **~49s** |

---

## 8. Results & Empirical Benchmark

Evaluation on the **4,970 held-out real-world test instances** (un-resampled 20% stratified test partition):

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | OvR ROC-AUC | Inference Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 69.88% | 0.6542 | 0.7429 | 0.6896 | **0.9414** | 0.003 ms |
| **K-Nearest Neighbors** | 70.38% | 0.6698 | 0.7288 | 0.6900 | 0.9041 | 2.500 ms |
| **Decision Tree ⭐ (Champion)** | **74.95%** | **0.7160** | **0.7532** | **0.7311** | 0.8789 | **0.0007 ms** |

> **Champion Selection Justification:** The **Decision Tree Classifier** achieved the highest test accuracy ([METRIC_VALUE, e.g., 74.95%]), highest Macro-F1 ([METRIC_VALUE, e.g., 0.7311]), and leading Macro-Recall ([METRIC_VALUE, e.g., 0.7532]). Crucially, it translates into human-interpretable orthogonal decision rules that can be audited in civic hearings and executes in sub-millisecond latency.

### Confusion Matrix & Error Highlights
- **High Sensitivity on Life-Threatening Air:** *Good* and *Severe* categories achieve high diagonal recall ($> 75\%$), confirming that training-fold SMOTE prevented minority class neglect.
- **Adjacent Boundary Overlap:** ~15% confusion occurs between contiguous classes (*Moderate* and *Poor*), where atmospheric particulate accumulation transitions along an unbroken chemical continuum.
- **Zero Catastrophic Errors:** Zero instances of *Severe* air days were misclassified as *Good*.
- *Figure Reference:* See `artifacts/figures/confusion_matrices_all_models.png` and `artifacts/figures/roc_curves_all_models.png`.

---

## 9. Application Screenshots

| Scenario / Feature | Interface Preview |
|:---|:---|
| **Real-Time Classification & CPCB Badge** | `[SCREENSHOT_PLACEHOLDER: Main Dashboard showing Severe Winter Smog prediction with maroon badge and probability distribution]` |
| **Clinical Health Advisories** | `[SCREENSHOT_PLACEHOLDER: Three-column advisory panel detailing precautions for General Public, Sensitive Populations, and Protective Gear]` |
| **What-If Emission Simulator** | `[SCREENSHOT_PLACEHOLDER: Interactive slider demonstrating how a 25% particulate reduction shifts category from Poor to Moderate]` |

---

## 10. Reproducibility & Seed Governance
- **Random Seed:** Universally fixed to `random_state=42` across train-test splitting, SMOTE synthesis, cross-validation fold generation, and tree splitting.
- **Leakage Firewall:**
  1. Target features (`AQI`, `AQI_Bucket`) isolated from $X$ before transformation.
  2. 80/20 train-test split executed before computing imputation medians or standard deviations.
  3. Preprocessor fitted strictly on training data; test set remained 100% natural and un-synthesized.
- **Artifact Serialization:** Pinned preprocessing pipeline (`preprocessor.joblib`) and model estimators (`model.joblib`) guarantee identical predictions across platforms.

---

## 11. Limitations
1. **Educational Prototype Scope:** Designed as an educational prototype operating on station-level daily aggregates rather than continuous sub-hourly IoT telemetry.
2. **Missing Volatile Compounds:** Secondary VOCs (`Xylene`, `Toluene`) experienced high historical missingness in early monitoring phases, requiring median imputation.
3. **Temporal Independence:** Treats daily records as independent observations, omitting continuous multi-day time-series lag structures.
4. **Spatial Generalization:** Topographical microclimates vary between coastal boundaries (Mumbai/Chennai) and continental thermal inversion basins (Delhi), which may introduce minor local calibration offsets.

---

## 12. Future Scope
1. **Live IoT Sensor Ingestion:** Integrating real-time streaming APIs from open government air quality portals.
2. **Spatio-Temporal Graph Networks:** Modeling regional wind dispersion vectors across adjacent geographic monitoring stations.
3. **Conformal Prediction:** Generating mathematically guaranteed prediction sets ($1 - \alpha$ coverage) to quantify uncertainty during critical civic alerts.
4. **Edge Microcontroller Deployment:** Compiling Decision Tree logic into embedded C++ firmware for solar-powered ESP32/ARM Cortex field sensors.

---

## 13. Citation

```bibtex
@misc{rao2020airqualityindia,
  author       = {Rohan Rao},
  title        = {Air Quality Data in India (2015--2020)},
  year         = {2020},
  publisher    = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india}},
  note         = {Data sourced originally from Central Pollution Control Board (CPCB), India. CC0 Public Domain License}
}
```

---

## 14. License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. The underlying monitoring dataset is distributed under the **Creative Commons CC0: Public Domain** license.

---

## 15. Acknowledgments
- **Learn Depth Academy:** For curriculum guidance, problem formulation, and evaluation rubric structure for Track 1 Capstone (Problem 10).
- **Central Pollution Control Board (CPCB), India:** For public environmental monitoring telemetry and National Air Quality Index (NAQI) technical standards.
- **Rohan Rao:** For curating and hosting the *Air Quality Data in India* dataset on Kaggle.
- **Open-Source Python Community:** The developers of `scikit-learn`, `imbalanced-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and `streamlit`.

---

## 16. Contact & Author Information
- **Author:** Muhammad Taha Nawab
- **Role:** Senior Data Scientist & ML Engineer
- **GitHub:** [https://github.com/tahanawab4848](https://github.com/tahanawab4848)
- **Repository:** [https://github.com/tahanawab4848/Urban_Ai](https://github.com/tahanawab4848/Urban_Ai)
- **Email:** `tahanawab.official@gmail.com`
