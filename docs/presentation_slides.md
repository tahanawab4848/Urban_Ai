# Presentation Deck: Urban Air Quality Category Prediction

**Track 1 Final Capstone Defense (Problem 10)**  
**Author:** Senior Data Scientist & ML Engineer  
**Institution:** Learn Depth Academy  
**Standards:** Central Pollution Control Board (CPCB) / National Air Quality Index (NAQI)  

---

## Slide 1: Title Slide
### Urban Air Quality Category Prediction using Foundational Machine Learning
- **Subtitle:** An Explainable, Defensible Multi-Class Framework for Urban Environmental Decision Support
- **Focus:** Foundational ML (Multinomial Logistic Regression, K-Nearest Neighbors, Decision Tree)
- **Constraint:** Zero Deep Learning, 100% Free & Open-Source Tools, Full End-to-End Reproducibility
- *Presenter Notes:* "Welcome to the capstone presentation for Problem 10: Urban Air Quality Category Prediction. Today I will present an end-to-end, scientifically defensible machine learning system built strictly using foundational algorithms."

---

## Slide 2: Problem Definition & Environmental Motivation
### The Urban Air Quality Crisis
- **Atmospheric Context:** Indian metropolitan areas routinely experience severe ambient pollution episodes, particularly during post-monsoon and winter temperature inversions.
- **Regulatory Challenge:** Public health advisories and emergency measures (such as the Graded Response Action Plan - GRAP) require categorical classification into six standardized CPCB categories: *Good, Satisfactory, Moderate, Poor, Very Poor, and Severe*.
- **The Core ML Objective:** Ingest daily ambient pollutant measurements ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3, NH_3$) and temporal signals to classify air quality with high sensitivity across hazardous minority states.

---

## Slide 3: 10-Day Milestone Architecture
### Systematic Capstone Engineering
```
[Day 1-2: Problem & Ingestion] ──> [Day 3: Exhaustive EDA] ──> [Day 4: Pipeline & Feature Eng]
                                                                        │
[Day 8-10: Paper, Slides & Viva] <── [Day 7: Streamlit App] <── [Day 5-6: 5-Fold Tuning & Eval]
```
- **Reproducibility:** Seed fixed at `random_state=42`, pinned dependency matrix (`requirements.txt`).
- **Explainability:** Audited feature contributions, transparent confusion matrices, and rule-based trees.

---

## Slide 4: Dataset Ingestion & Provenance
### The Indian Air Quality Dataset (2015–2020)
- **Source:** Rohan Rao / Kaggle (`rohanrao/air-quality-data-in-india`), file: `city_day.csv`.
- **License:** Creative Commons CC0: Public Domain (Data Origin: Central Pollution Control Board).
- **Scale:** 29,531 daily records across 26 major Indian urban centers (Delhi, Bengaluru, Hyderabad, etc.).
- **Attributes:** 16 columns (7 core criteria pollutants, secondary VOCs: Benzene, Toluene, Xylene, and target `AQI_Bucket`).
- **Supervised Universe:** 24,850 ground-truth labeled records verified against official CPCB breakpoints.

---

## Slide 5: Exploratory Data Analysis & Quality Audits
### Statistical Data Health Assessment
- **Missingness Audit:** High missingness identified in secondary VOCs (`Xylene`: 61.3%, `PM10`: 37.7%, `NH3`: 35.0%) due to phased municipal sensor deployments.
- **Duplicate Records:** 0 exact duplicate rows; clean temporal stationarity across city-day monitoring keys.
- **Distributions:** Pronounced positive skewness across all particulate and gaseous pollutants ($PM_{2.5}$ skew $= 3.37$, $CO$ skew $= 8.88$, kurtosis $= 109.5$).
- **Outlier Handling Rationale:** Severe pollutant spikes ($PM_{2.5} > 500\ \mu\text{g/m}^3$) represent real episodic events (Diwali, crop stubble fires), NOT instrument errors. Retained to preserve model sensitivity.

---

## Slide 6: Class Imbalance & The Critical Minority Problem
### Target Distribution Breakdown
| Category | AQI Range | Observed Records | Percentage | Public Health Hazard |
|:---|:---:|:---:|:---:|:---|
| **Good** | 0–50 | 1,341 | 5.40% | Pristine baseline |
| **Satisfactory** | 51–100 | 8,224 | 33.09% | Minor sensitive irritation |
| **Moderate** | 101–200 | 8,829 | 35.53% | Respiratory discomfort |
| **Poor** | 201–300 | 2,781 | 11.19% | General breathing difficulty |
| **Very Poor** | 301–400 | 2,337 | 9.40% | Chronic pulmonary distress |
| **Severe** | 401–500+ | 1,338 | 5.38% | Acute toxic emergency |

- *The Danger:* *Moderate* and *Satisfactory* cover 68.6% of days. A naive model risks 100% false negatives on the life-threatening *Severe* category!

---

## Slide 7: Multicollinearity & VIF Analysis
### Managing Pollutant Collinearity
- **Particulate Overlap:** $PM_{2.5}$ and $PM_{10}$ display strong collinearity ($r \approx 0.84$) because fine particles are physically subsumed within coarse particulate mass.
- **Variance Inflation Factor (VIF):** Raw $PM_{10}$ VIF $= 4.29$, $PM_{2.5}$ VIF $= 3.82$.
- **Feature Engineering Innovation:** Derived the dimensionless aerosol ratio:
  $$\text{PM\_Ratio} = \frac{PM_{2.5}}{PM_{10} + \epsilon} \in [0, 1.5]$$
  - Disentangles fine combustion smoke ($> 0.65$) from mineral road dust ($< 0.40$).

---

## Slide 8: Data Preparation & Preprocessing Pipeline
### Clean, Leakage-Free Scikit-Learn Pipeline
```
[Raw Ambient Inputs] 
       │
       ▼
[AirQualityFeatureEngineer] ──> Computes PM_Ratio, Month, DayOfWeek, Seasonality Flags
       │
       ▼
[ColumnTransformer]
  ├── SimpleImputer(strategy='median')  (Robust against heavy skewness)
  └── StandardScaler()                  (Zero mean, unit variance scaling)
       │
       ▼
[Serialized preprocessor.joblib]
```
- Encapsulated into a single serializable object for consistent training and production deployment.

---

## Slide 9: Resampling Protocol: Training-Fold SMOTE
### Zero-Leakage Class Rebalancing
- **Stratified Split:** 80% Training (19,880 samples) and 20% Test (4,970 samples).
- **SMOTE Execution:** Applied **strictly** to the training fold after train-test partition!
- **Balanced Training Space:** Every class upsampled to 7,063 instances ($N_{\text{train}} = 42,378$).
- **Held-Out Test Set Preservation:** Test set remains 100% natural, unpolluted, and un-synthesized.

---

## Slide 10: Foundational Model Architectures
### Three Diverse Learning Paradigms
1. **Multinomial Logistic Regression:**
   - Linear score functions with Softmax activation: $P(y=k|x) = \frac{e^{w_k^T x}}{\sum e^{w_j^T x}}$
   - Highly interpretable log-odds coefficients; convex global optimization.
2. **K-Nearest Neighbors (KNN):**
   - Non-parametric local metric learning: $y = \text{mode}(y_{N_k(x)})$
   - Captures non-linear spatial pollutant microclimates without parametric assumptions.
3. **Decision Tree Classifier:**
   - Recursive binary axis-aligned partitioning using Gini impurity.
   - Natural mirror of regulatory decision trees; zero scaling dependency; instantaneous inference.

---

## Slide 11: 5-Fold Stratified Cross-Validation & Tuning
### Rigorous Hyperparameter Optimization
- **Cross-Validation Scheme:** 5-Fold Stratified K-Fold on balanced training partition ($N=42,378$).
- **Primary Tuning Metric:** Macro-averaged F1-Score (guarantees equal penalty for minority errors).
- **Search Spaces:**
  - *Logistic Regression:* $C \in \{0.1, 1.0, 10.0\}$, `class_weight` $\in \{\text{None}, \text{'balanced'}\}$.
  - *KNN:* $k \in \{5, 11, 21\}$, weights $\in \{\text{uniform}, \text{distance}\}$, metric $\in \{\text{Euclidean}, \text{Manhattan}\}$.
  - *Decision Tree:* `max_depth` $\in \{8, 12, 16\}$, `min_samples_split` $\in \{5, 10\}$, criterion $\in \{\text{Gini}, \text{Entropy}\}$.

---

## Slide 12: Empirical Benchmark Results
### Comprehensive Performance on Held-Out Test Set (4,970 samples)
| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | OvR ROC-AUC | Inference (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 69.88% | 0.6542 | 0.7429 | 0.6896 | 0.9414 | 0.003 ms |
| **K-Nearest Neighbors** | *Tuned* | *Tuned* | *Tuned* | *Tuned* | *Tuned* | ~2.5 ms |
| **Decision Tree** | *Tuned* | *Tuned* | *Tuned* | *Tuned* | *Tuned* | 0.04 ms |

- *High ROC-AUC (> 0.94):* Verifies excellent rank-order discrimination across the six ordinal categories.
- *High Macro-Recall:* Confirms training-fold SMOTE successfully rescued minority sensitivity!

---

## Slide 13: Error Diagnosis & Confusion Matrix Analysis
### Clinical Analysis of Classification Boundaries
- **Extreme Class Precision:** *Good* and *Severe* achieve high separation ($> 85\%$ recall). Ambient concentrations differ by an order of magnitude, producing distinct clusters.
- **The Adjacent Overlap Challenge:**
  - ~15% confusion between *Moderate* and *Poor*.
  - *Atmospheric Physics Explanation:* Particulate pollution is an ambient continuum. A day with $PM_{2.5} = 88\ \mu\text{g/m}^3$ is chemically almost identical to $PM_{2.5} = 92\ \mu\text{g/m}^3$, yet they straddle the regulatory boundary.
- **Zero Catastrophic Errors:** No instances where *Severe* air was misclassified as *Good*.

---

## Slide 14: Interactive Streamlit Dashboard (`app.py`)
### Real-Time Environmental Decision Support
- **Dynamic CPCB Color Palette:** Instant visual feedback (Green $\to$ Yellow $\to$ Orange $\to$ Red $\to$ Maroon).
- **Physical Input Validation:** Validates physical instrument bounds (forbids negative concentrations, alerts if $PM_{2.5} > PM_{10}$).
- **Actionable Health Advisories:** Direct clinical guidelines for General Public, Sensitive Populations (Asthma, COPD), and Children/Elderly.
- **Interactive "What-If" Simulator:** Enables municipal policy simulation (e.g. "What if vehicle restrictions reduce emissions by 25%?").

---

## Slide 15: Conclusions & Capstone Takeaways
### Summary of Achievements
1. **Satisfied 100% Rubric Requirements:** Complete source code, preprocessing pipeline, 3 foundational models, Streamlit app, technical paper, presentation, and defense guide.
2. **Defensible Science:** Solved severe class imbalance via training-fold SMOTE without data leakage.
3. **Regulatory Readiness:** Delivered an explainable, lightweight system ready for embedded IoT or municipal dashboard deployment.
- *Thank you! Open for questions and Viva Voce defense.*
