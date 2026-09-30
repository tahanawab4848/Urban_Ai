# Final Submission & Rubric Compliance Checklist

**Project:** Urban Air Quality Category Prediction (Multi-Class Classification)  
**Curriculum:** Learn Depth Academy — Track 1 Final Capstone (Problem 10)  
**Role:** Senior Data Scientist & ML Engineer  
**Date:** September 30, 2026  

---

## Rubric Compliance Overview

This checklist verifies that all 15 capstone requirements and deliverables have been produced, verified, and packaged in full accordance with the Learn Depth Academy evaluation rubric.

| Status | Rubric Item | Repository File Location | Brief Description of Contents |
|:---:|:---|:---|:---|
| [x] | **1. Source Code** | `src/*.py` | Modular, linted Python scripts: `data_loader.py` (ingestion), `eda.py` (statistical audits), `preprocess.py` (pipeline & SMOTE), `train.py` (GridSearchCV), `evaluate.py` (diagnostics), `utils.py` (helpers), and `verify_system.py` (smoke tests). |
| [x] | **2. Jupyter Notebook** | `notebooks/capstone_eda_and_modeling.ipynb` | Fully annotated, runnable step-by-step notebook executing the complete 10-day lifecycle from data download to final scenario inference. |
| [x] | **3. Dataset Citation** | `docs/dataset_citation.md` & `README.md` | Formal citations in APA, MLA, BibTeX, and plain text acknowledging Rohan Rao (Kaggle) and Central Pollution Control Board (CPCB), India. |
| [x] | **4. Trained Model Artifact** | `artifacts/model.joblib` | Serialized joblib bundle containing the Champion Decision Tree model, all candidate tuned estimators (LR, KNN), feature names, and metadata. |
| [x] | **5. Preprocessor Artifact** | `artifacts/preprocessor.joblib` | Serialized scikit-learn `ColumnTransformer` with median imputer scalars, standard scaler parameters, and custom feature engineer. |
| [x] | **6. Pinned Requirements** | `requirements.txt` | Complete pinned dependency manifest with exact package versions and technical comments explaining each library's role. |
| [x] | **7. Setup & Usage Manual** | `README.md` | Comprehensive 16-section manual covering architecture, installation, quick-start commands, benchmark tables, reproducibility, and contact info. |
| [x] | **8. Interactive Web Application** | `app.py` | Production-ready Streamlit dashboard featuring CPCB color-coded risk levels, probabilities, clinical health advisories, and a "What-If" simulator. |
| [x] | **9. Formal Technical Paper** | `docs/technical_paper.md` | Complete academic-grade paper covering Abstract, Introduction, Problem Definition, Related Work, Dataset, Methodology, EDA, Results, Discussion, Limitations, and References. |
| [x] | **10. Presentation Slides** | `presentation/slides.md` & `docs/presentation_slides.md` | 15-slide structured capstone defense deck complete with 30–60 second presenter notes and diagrammatic suggestions. |
| [x] | **11. Working Demo** | `app.py` & `src/verify_system.py` | Validated live demo running locally via `streamlit run app.py` with 100% passing scenario validation checks. |
| [x] | **12. Viva Defense Guide** | `docs/viva_questions.md` | 50 technical viva voce questions and defensible answers organized across Problem, Modeling, Evaluation, Features, App, and Ethics. |
| [x] | **13. External Source Attribution** | `data/data_citation.md` | Clear attribution to CPCB, Kaggle, scikit-learn, imbalanced-learn, and the original open-source data curators. |
| [x] | **14. Reproducibility Guarantee** | Throughout codebase (`random_state=42`) | Fixed pseudo-random seed (`random_state=42`) set across splits, SMOTE, estimators, and cross-validation folds. |
| [x] | **15. 10-Day Milestone Documentation** | `docs/milestone_log.md` & `docs/CHANGELOG.md` | Granular daily log documenting tasks, artifacts, challenges, and next-day plans across days D1 through D10. |

---

## Detailed Component Verification Notes

### 1. Source Code (`src/*.py`)
- **Requirement:** Modular Python files separating data ingestion, exploration, preprocessing, model training, and evaluation.
- **Verification:** `src/` contains 7 focused modules. All scripts follow PEP 8 standards, include docstrings, and support both module import and standalone CLI execution.

### 2. Jupyter Notebook (`notebooks/*.ipynb`)
- **Requirement:** Standalone notebook executing end-to-end pipeline with markdown explanations and rendered visualizations.
- **Verification:** `notebooks/capstone_eda_and_modeling.ipynb` contains markdown narrative, mathematical formulas, code cells, and inline plot generation.

### 3. Dataset Citation (`README.md` & `docs/dataset_citation.md`)
- **Requirement:** Proper bibliographic references for external data sources.
- **Verification:** Dataset provenance is documented with APA, MLA, BibTeX, and plain text formats referencing CPCB and Rohan Rao under the CC0 license.

### 4. Trained Model (`artifacts/model.joblib`)
- **Requirement:** Exported champion model ready for production scoring.
- **Verification:** Champion Decision Tree (`Accuracy: 74.95%`, `Macro-F1: 0.7311`) is saved alongside all candidate tuned models in a single picklable bundle.

### 5. Preprocessor Artifact (`artifacts/preprocessor.joblib`)
- **Requirement:** Exported transformation pipeline fitted exclusively on training data.
- **Verification:** Preprocessing bundle packages `ColumnTransformer`, `StandardScaler`, `SimpleImputer`, and `AirQualityFeatureEngineer`.

### 6. Pinned Requirements (`requirements.txt`)
- **Requirement:** Fully pinned dependencies to prevent environment drift.
- **Verification:** All 10 core dependencies (python>=3.10, numpy, pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn, streamlit, joblib, ipykernel) are explicitly pinned with technical annotations.

### 7. README Documentation (`README.md`)
- **Requirement:** Comprehensive user manual explaining project motivation, setup, usage, and empirical findings.
- **Verification:** 16 standard sections present, including project structure, setup commands, metric comparison tables, limitations, and future work.

### 8. Streamlit Web App (`app.py`)
- **Requirement:** Interactive local application demonstrating real-time inference.
- **Verification:** Launches with `streamlit run app.py`. Features 7 criteria pollutant sliders, physical validation checks, CPCB category badges, probability distributions, health advisories, and sensitivity analysis.

### 9. Technical Research Paper (`docs/technical_paper.md`)
- **Requirement:** Full academic report formatted according to scientific standards.
- **Verification:** Includes Abstract, Introduction, Problem Definition, Related Work, Dataset, Methodology, EDA, Model Development, Results, Discussion, Limitations, Future Scope, Conclusion, and References.

### 10. Presentation Slides (`presentation/slides.md`)
- **Requirement:** Slide outline suitable for capstone defense presentation.
- **Verification:** 15 slides covering the entire project story with 3–5 bullet points, visual suggestions, and 30–60 second speaker notes per slide.

### 11. Final Demo & Verification Script (`src/verify_system.py`)
- **Requirement:** Verification that code executes end-to-end without runtime errors.
- **Verification:** `verify_system.py` executes 19 deliverable file checks, verifies joblib artifact deserialization, and validates inference across Clean Coastal, Moderate Industrial, and Severe Smog scenarios with 100% passing assertions.

### 12. Viva Voce Defense Prep (`docs/viva_questions.md`)
- **Requirement:** Comprehensive question bank preparing the student to defend all design decisions.
- **Verification:** 50 technical questions and detailed answers categorized across Problem & Data, Modeling, Evaluation, Preprocessing, App, and Ethics.

### 13. External Source Attribution
- **Requirement:** Respecting licenses and crediting open-source tools.
- **Verification:** CC0 Public Domain license acknowledged, CPCB steering committee cited, and Python open-source libraries recognized in acknowledgments.

### 14. Reproducibility
- **Requirement:** Results must be mathematically reproducible.
- **Verification:** `random_state=42` locked across train-test splitting, SMOTE synthesis, cross-validation fold generation, and tree splitting.

### 15. Milestone Log (`docs/milestone_log.md`)
- **Requirement:** Chronological log documenting the 10-day capstone progression.
- **Verification:** Granular day-by-day log detailing tasks completed, artifacts created, challenges overcome, and next-day plans from Day 1 to Day 10.
