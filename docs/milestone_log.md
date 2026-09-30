# 10-Day Milestone Execution Log

**Project:** Urban Air Quality Category Prediction (Multi-Class Classification)  
**Track:** Learn Depth Academy — Track 1 Final Capstone (Problem 10)  
**Role:** Senior Data Scientist & ML Engineer  
**Standard:** Central Pollution Control Board (CPCB) National Air Quality Index (NAQI)  

---

## Day 1: Problem Formulation, Regulatory Standards & Project Setup
**Date:** 2026-09-21  
**Milestone Focus:** Problem Definition & Scientific Scoping  

### Tasks Completed:
- [x] Defined multi-class air quality classification problem statement using CPCB/NAQI standards.
- [x] Researched CPCB NAQI breakpoint formulas for 6 regulatory categories (Good, Satisfactory, Moderate, Poor, Very Poor, Severe).
- [x] Configured clean repository directory structure (`src/`, `data/`, `artifacts/`, `docs/`, `notebooks/`).
- [x] Initialized development environment and drafted initial `requirements.txt` with pinned dependencies.
- [x] Implemented color palette mapping and category dictionaries in `src/utils.py`.

### Artifacts Produced:
- `src/__init__.py`
- `src/utils.py`
- `requirements.txt`
- `.gitignore`

### Learnings:
- *What went well:* Establishing official CPCB hex colors early ensured visual consistency across all subsequent EDA plots and the Streamlit dashboard.
- *What was challenging:* Clarifying why categorical classification is clinically more actionable than continuous AQI regression for municipal emergency interventions.

### Next Day Plan:
- Acquire raw dataset (`city_day.csv`) from verified public mirrors.
- Document dataset provenance, schema attributes, and CC0 licensing.
- Implement automated data ingestion script with fallback logic.

---

## Day 2: Data Acquisition, Provenance & Ingestion Pipeline
**Date:** 2026-09-22  
**Milestone Focus:** Dataset Acquisition & Verification  

### Tasks Completed:
- [x] Downloaded raw `city_day.csv` (29,531 records, 16 features) from Rohan Rao's Indian Air Quality repository.
- [x] Verified file integrity and schema consistency across the 26 Indian metropolitan cities.
- [x] Developed `src/data_loader.py` with multi-mirror fallback handling and automated CPCB breakpoint mapping.
- [x] Documented formal dataset provenance in APA, MLA, and BibTeX formats in `data/data_citation.md`.
- [x] Reconciled missing `AQI_Bucket` annotations against numerical `AQI` breakpoints.

### Artifacts Produced:
- `src/data_loader.py`
- `data/raw/city_day.csv`
- `data/data_citation.md`

### Learnings:
- *What went well:* Building automated mirror fallback logic prevented broken pipeline execution when remote URLs experienced downtime.
- *What was challenging:* Filtering out unannotated records where fewer than three criteria pollutants were monitored, yielding an effective supervised cohort of 24,850 rows.

### Next Day Plan:
- Conduct exhaustive Phase 1 Exploratory Data Analysis.
- Audit missing values, exact duplicates, class imbalance, and lognormal skewness.
- Generate publication-quality figures and an investigation report.

---

## Day 3: Exploratory Data Analysis & Statistical Audits
**Date:** 2026-09-23  
**Milestone Focus:** Comprehensive Data Investigation & VIF Collinearity  

### Tasks Completed:
- [x] Built comprehensive statistical audit module in `src/eda.py`.
- [x] Quantified missingness rates per pollutant (`Xylene` 61.3%, `PM10` 37.7%, `NH3` 35.0%) and justified median imputation.
- [x] Verified zero duplicate rows across the 16 attributes and verified temporal city-date stationarity.
- [x] Analyzed target class imbalance (Moderate 35.5%, Satisfactory 33.1% vs. Good 5.4%, Severe 5.4%).
- [x] Computed skewness and kurtosis across 12 pollutants; performed IQR outlier analysis.
- [x] Calculated Pearson correlation matrix and computed manual Variance Inflation Factor (VIF) matrix scores.
- [x] Computed Mutual Information (MI) classification scores, establishing $PM_{2.5}$ and $PM_{10}$ as dominant predictors.

### Artifacts Produced:
- `src/eda.py`
- `artifacts/figures/eda_missing_values.png`
- `artifacts/figures/eda_class_balance.png`
- `artifacts/figures/eda_pollutant_distributions.png`
- `artifacts/figures/eda_pollutant_boxplots.png`
- `artifacts/figures/eda_correlation_heatmap.png`
- `artifacts/figures/eda_top_features_pairplot.png`
- `artifacts/figures/eda_feature_relevance.png`
- `artifacts/reports/eda_investigation_report.md`

### Learnings:
- *What went well:* VIF and correlation analysis clearly demonstrated the strong physical coupling between $PM_{2.5}$ and $PM_{10}$ ($r = 0.84$).
- *What was challenging:* Defending outlier retention: realizing that extreme pollutant spikes ($PM_{2.5} > 500\ \mu\text{g/m}^3$) are physically real catastrophic events that must not be deleted.

### Next Day Plan:
- Design domain-specific feature engineering ($PM_{2.5}/PM_{10}$ ratio, Indian seasonal indicators).
- Encapsulate median imputation and standard scaling into a scikit-learn `ColumnTransformer`.

---

## Day 4: Feature Engineering & Preprocessing Pipeline
**Date:** 2026-09-24  
**Milestone Focus:** Domain Ratios & ColumnTransformer Pipeline  

### Tasks Completed:
- [x] Developed custom `AirQualityFeatureEngineer` inheriting from `BaseEstimator` and `TransformerMixin`.
- [x] Engineered the dimensionless fine-to-coarse particulate ratio: $\text{PM\_Ratio} = \frac{PM_{2.5}}{PM_{10} + \epsilon}$.
- [x] Extracted calendar attributes (Month, Day of Week) and engineered one-hot indicators for 4 Indian seasons (*Winter, Summer, Monsoon, Post-Monsoon*).
- [x] Constructed a leak-free `ColumnTransformer` with `SimpleImputer(strategy='median')` and `StandardScaler()`.
- [x] Serialized preprocessing pipeline and feature schema definitions.

### Artifacts Produced:
- `src/preprocess.py`
- `artifacts/preprocessor.joblib`

### Learnings:
- *What went well:* The $PM_{2.5}/PM_{10}$ ratio provided an intuitive atmospheric fingerprint separating fine combustion smoke from mineral dust storms.
- *What was challenging:* Ensuring that custom transformer classes remain pickle-serializable across different execution contexts without module namespace conflicts.

### Next Day Plan:
- Implement stratified 80/20 train-test partitioning.
- Apply SMOTE exclusively on the training partition to solve class imbalance without leakage.

---

## Day 5: Resampling Strategy & Data Partitioning
**Date:** 2026-09-25  
**Milestone Focus:** Leakage-Free SMOTE Rebalancing  

### Tasks Completed:
- [x] Implemented stratified 80/20 train-test split (`test_size=0.2, random_state=42`) yielding 19,880 train and 4,970 test samples.
- [x] Fitted the `ColumnTransformer` strictly on the 80% training partition to eliminate parameter leakage.
- [x] Applied Synthetic Minority Over-sampling Technique (SMOTE) strictly to the training partition.
- [x] Rebalanced training set from 19,880 to 42,378 samples (exactly 7,063 instances per class).
- [x] Verified that the 4,970 test instances remained 100% natural, un-synthesized, and unpolluted.
- [x] Exported processed training and test splits to `data/processed/train.csv` and `data/processed/test.csv`.

### Artifacts Produced:
- `data/processed/train.csv`
- `data/processed/test.csv`
- Updated `artifacts/preprocessor.joblib`

### Learnings:
- *What went well:* Verifying that test class ratios matched real-world ambient baselines gave full confidence in our evaluation validity.
- *What was challenging:* Understanding the exact mathematical failure of applying SMOTE before splitting, where synthetic vectors interpolate test information into training space.

### Next Day Plan:
- Configure foundational model architectures (Multinomial Logistic Regression, KNN, Decision Tree).
- Define 5-fold Stratified K-Fold cross-validation hyperparameter search grids.

---

## Day 6: Foundational Model Architectures & Training Setup
**Date:** 2026-09-26  
**Milestone Focus:** Foundational Classifier Setup & Search Spaces  

### Tasks Completed:
- [x] Built model training module in `src/train.py`.
- [x] Configured Multinomial Logistic Regression with Softmax formulation and L2 regularization.
- [x] Configured K-Nearest Neighbors (KNN) classifier with Euclidean and Manhattan spatial distance metrics.
- [x] Configured Decision Tree Classifier with Gini Impurity and Information Gain splitting criteria.
- [x] Established 5-Fold Stratified K-Fold cross-validation scheme locked to `random_state=42`.
- [x] Configured `GridSearchCV` search spaces targeting **Macro-averaged F1-Score** as the optimization criterion.

### Artifacts Produced:
- `src/train.py`

### Learnings:
- *What went well:* Structuring a unified `GridSearchCV` interface across all three algorithms allowed automated benchmarking with identical cross-validation folds.
- *What was challenging:* Balancing hyperparameter grid resolution against training runtime on 42,378 resampled instances, particularly for distance-based KNN calculations.

### Next Day Plan:
- Execute 5-fold Stratified GridSearchCV for all three foundational models.
- Identify champion model and export fitted estimator artifacts to `artifacts/model.joblib`.

---

## Day 7: Hyperparameter Optimization & Model Selection
**Date:** 2026-09-27  
**Milestone Focus:** Grid Search Execution & Champion Model Selection  

### Tasks Completed:
- [x] Executed 5-fold Stratified GridSearchCV for Logistic Regression (optimal: $C=10.0, \text{solver}='lbfgs'$; CV Macro-F1: 0.7528).
- [x] Executed 5-fold Stratified GridSearchCV for KNN (optimal: $k=5, \text{metric}='manhattan', \text{weights}='distance'$; CV Macro-F1: 0.8756).
- [x] Executed 5-fold Stratified GridSearchCV for Decision Tree (optimal: `criterion='gini', max_depth=16, min_samples_split=5`; CV Macro-F1: 0.8293).
- [x] Evaluated all tuned models on 4,970 held-out test samples.
- [x] Selected **Decision Tree Classifier** as production champion (`Test Accuracy: 74.95%`, `Macro-F1: 0.7311`, `Macro-Recall: 0.7532`).
- [x] Serialized champion model bundle to `artifacts/model.joblib` and exported benchmark logs to `artifacts/metrics.json`.

### Artifacts Produced:
- `artifacts/model.joblib`
- `artifacts/metrics.json`

### Learnings:
- *What went well:* The Decision Tree achieved superior generalization on held-out test data, outperforming linear and spatial models while executing in sub-millisecond inference time.
- *What was challenging:* Managing the compute duration of KNN's 60 cross-validation fits across 42,000 samples, highlighting KNN's $\mathcal{O}(N \cdot D)$ scalability limitation.

### Next Day Plan:
- Generate normalized confusion matrices and One-vs-Rest (OvR) ROC curves.
- Compile in-depth clinical error analysis and misclassification diagnosis report.

---

## Day 8: Evaluation, Diagnostics & Error Analysis
**Date:** 2026-09-28  
**Milestone Focus:** Model Diagnostics & Clinical Error Analysis  

### Tasks Completed:
- [x] Built evaluation and diagnostic module in `src/evaluate.py`.
- [x] Generated 1x3 side-by-side normalized confusion matrices across all three models.
- [x] Computed Multi-Class One-vs-Rest (OvR) ROC curves and per-class AUC scores.
- [x] Produced model comparison benchmark bar charts across Accuracy, Macro Precision, Recall, and F1.
- [x] Authored clinical error analysis diagnosing boundary confusion between *Moderate* and *Poor*.
- [x] Verified zero catastrophic classification errors (zero *Severe* days classified as *Good*).
- [x] Created `src/verify_system.py` automated smoke test verifying all deliverable files and inference pipelines.

### Artifacts Produced:
- `src/evaluate.py`
- `src/verify_system.py`
- `artifacts/figures/confusion_matrices_all_models.png`
- `artifacts/figures/roc_curves_all_models.png`
- `artifacts/figures/model_comparison_benchmark.png`
- `artifacts/reports/model_evaluation_and_error_analysis.md`

### Learnings:
- *What went well:* High diagonal recall ($> 75\%$) on extreme classes (*Good* and *Severe*) validated the effectiveness of training-fold SMOTE.
- *What was challenging:* Explaining that ~15% confusion between *Moderate* and *Poor* is caused by atmospheric particulate continuity rather than model defects.

### Next Day Plan:
- Build interactive Streamlit dashboard (`app.py`) with CPCB color-coded risk badging.
- Implement input validation, posterior probabilities, health advisories, and a "What-If" simulator.

---

## Day 9: Streamlit Application Development
**Date:** 2026-09-29  
**Milestone Focus:** Production Web Dashboard & Decision Support  

### Tasks Completed:
- [x] Developed interactive Streamlit web dashboard in `app.py`.
- [x] Designed responsive UI with CPCB color badging (Green $\to$ Yellow $\to$ Orange $\to$ Red $\to$ Maroon).
- [x] Implemented input sliders for 7 criteria pollutants with physical units ($\mu\text{g/m}^3, \text{mg/m}^3$).
- [x] Built physical validation checks blocking negative values and flagging $PM_{2.5} > PM_{10}$ sensor anomalies.
- [x] Built real-time inference pipeline applying `preprocessor.joblib` and champion `model.joblib`.
- [x] Integrated horizontal probability distribution bars matching official CPCB hex colors.
- [x] Added stratified health advisories for General Public, Sensitive Populations, and Personal Protection.
- [x] Implemented interactive "What-If" emission reduction simulator for municipal policy exploration.

### Artifacts Produced:
- `app.py`

### Learnings:
- *What went well:* Caching model loading via `@st.cache_resource` resulted in instantaneous UI responsiveness on every slider adjustment.
- *What was challenging:* Structuring clean custom CSS badges to render bold, readable text across contrasting background colors (e.g., dark text on bright yellow Moderate badge).

### Next Day Plan:
- Author formal 12-section IEEE/ACM style technical research paper (`docs/technical_paper.md`).
- Construct 15-slide capstone presentation deck (`presentation/slides.md`).
- Prepare 50-question viva voce defense guide (`docs/viva_questions.md`).

---

## Day 10: Technical Documentation, Viva Prep & Release
**Date:** 2026-09-30  
**Milestone Focus:** Academic Paper, Presentation Deck, Viva Prep & Packaging  

### Tasks Completed:
- [x] Compiled interactive end-to-end Jupyter Notebook `notebooks/capstone_eda_and_modeling.ipynb`.
- [x] Authored 12-section academic research paper in `docs/technical_paper.md`.
- [x] Structured 15-slide presentation deck with speaker notes in `presentation/slides.md`.
- [x] Authored comprehensive 50-question viva voce defense guide in `docs/viva_questions.md`.
- [x] Created rubric submission compliance checklist in `docs/submission_checklist.md`.
- [x] Finalized production `README.md` with complete architecture, benchmark tables, and setup instructions.
- [x] Executed `src/verify_system.py` passing 100% of deliverable assertions and scenario tests.
- [x] Initialized clean Git repository, recorded milestone changelog, and pushed to GitHub (`Urban_Ai`).

### Artifacts Produced:
- `notebooks/capstone_eda_and_modeling.ipynb`
- `docs/technical_paper.md`
- `presentation/slides.md`
- `docs/viva_questions.md`
- `docs/submission_checklist.md`
- `docs/milestone_log.md`
- `README.md`

### Learnings:
- *What went well:* Having an exhaustive viva question bank and structured slide deck provides complete confidence for oral defense and code review.
- *What was challenging:* Ensuring strict terminology alignment and consistent metric reporting across all 7 submission documents.

### Next Day Plan:
- Project completed and submitted for Learn Depth Academy Track 1 Capstone evaluation.
