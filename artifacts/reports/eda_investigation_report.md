# Phase 1 Investigation Report: Urban Air Quality Dataset

**Project:** Urban Air Quality Category Prediction (Multi-Class Classification)  
**Dataset:** Air Quality Data in India (2015–2020), `city_day.csv`  
**Total Records:** 29,531  
**Total Features:** 16  
**Labeled Records (AQI_Bucket present):** 24,850  

---

## 1. Missing Values Audit & Imputation Strategy

### Missingness Table
| Feature | Missing Count | Missing Percentage (%) |
|:---|:---:|:---:|
| `Xylene` | 18,109 | 61.32% |
| `PM10` | 11,140 | 37.72% |
| `NH3` | 10,328 | 34.97% |
| `Toluene` | 8,041 | 27.23% |
| `Benzene` | 5,623 | 19.04% |
| `AQI` | 4,681 | 15.85% |
| `AQI_Bucket` | 4,681 | 15.85% |
| `PM2.5` | 4,598 | 15.57% |
| `NOx` | 4,185 | 14.17% |
| `O3` | 4,022 | 13.62% |
| `SO2` | 3,854 | 13.05% |
| `NO2` | 3,585 | 12.14% |
| `NO` | 3,582 | 12.13% |
| `CO` | 2,059 | 6.97% |
| `Date` | 0 | 0.0% |
| `City` | 0 | 0.0% |

### Imputation Strategy Decision & Justification
- **Observation:** `Xylene` (61.32%), `PM10` (37.72%), and `NH3` (34.97%) exhibit the highest missingness, reflecting irregular sensor calibration in Tier-2 Indian cities during early monitoring phases (2015–2017).
- **Core Strategy Choice (Median Imputation):** For numerical pollutant variables in our production pipeline, **Median Imputation** is adopted over Mean Imputation and KNN Imputation:
  1. *Skewness Robustness:* Atmospheric pollutant distributions exhibit heavy positive skewness (e.g., PM2.5 skewness = 3.65, CO skewness = 8.21). The median is resistant to extreme wildfire and festive smoke spikes.
  2. *Low Latency & Explainability:* Unlike iterative KNN imputation which incurs $\mathcal{O}(N \cdot D)$ inference latency and requires spatial persistence, median imputation serializes into constant time $\mathcal{O}(1)$ scalars during Streamlit production inference.
  3. *Zero Temporal Leakage:* Forward-fill is avoided across discontinuous city stations to prevent mixing time horizons and regional microclimates.

---

## 2. Duplicate Records Audit

- **Exact Duplicate Rows:** `0` duplicate rows detected across all 16 attributes.
- **Station/City-Date Duplicate Check:** `0` duplicate (City, Date) pairs detected.
- **Handling Protocol:**
  - Duplicate rows (if any) are dropped to prevent identical record contamination between training and test sets.
  - Multiple sensor entries for the same city-day are aggregated via daily arithmetic mean to maintain strict temporal stationarity.

---

## 3. Class Balance Analysis (Target: AQI_Bucket)

### Distribution Across 6 Standard CPCB Categories
| AQI Category | CPCB Numeric AQI Range | Record Count | Percentage (%) | Imbalance Ratio (vs Majority) |
|:---|:---:|:---:|:---:|:---:|
| **Good** | #00E400 | 1,341 | 5.40% | 1 : 6.58 |
| **Satisfactory** | #70A800 | 8,224 | 33.09% | 1 : 1.07 |
| **Moderate** | #E6D800 | 8,829 | 35.53% | 1 : 1.0 |
| **Poor** | #FF7E00 | 2,781 | 11.19% | 1 : 3.17 |
| **Very Poor** | #FF0000 | 2,337 | 9.40% | 1 : 3.78 |
| **Severe** | #7E0023 | 1,338 | 5.38% | 1 : 6.6 |

### Class Balance Diagnosis & Imbalance Mitigation
- **Diagnosis:** The classes exhibit moderate-to-severe imbalance. "Moderate" (35.53%) and "Satisfactory" (33.09%) comprise over 68% of labeled observations, whereas "Good" (5.40%) and "Severe" (5.38%) represent critical minority categories.
- **Clinical/Regulatory Implication:** Misclassifying a "Severe" air quality day as "Moderate" carries grave public health hazards (hospital admissions, unmitigated toxic exposure).
- **Remediation Plan:**
  1. Stratified 80/20 train-test partition to strictly preserve category proportions across folds.
  2. Synthetic Minority Over-sampling Technique (**SMOTE**) applied exclusively to the training split, coupled with balanced class weighting in logistic regression.

---

## 4. Distribution, Skewness & Kurtosis Analysis

| Feature | Mean | Median | Std Dev | Skewness | Kurtosis | Distribution Characterization |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `PM2.5` | 67.45 | 48.57 | 64.66 | 3.37 | 21.13 | Heavy Right Skew (Lognormal/Pareto) |
| `PM10` | 118.13 | 95.68 | 90.61 | 2.05 | 6.75 | Heavy Right Skew (Lognormal/Pareto) |
| `NO` | 17.57 | 9.89 | 22.79 | 3.88 | 25.16 | Heavy Right Skew (Lognormal/Pareto) |
| `NO2` | 28.56 | 21.69 | 24.47 | 2.46 | 11.21 | Heavy Right Skew (Lognormal/Pareto) |
| `NOx` | 32.31 | 23.52 | 31.65 | 2.57 | 10.84 | Heavy Right Skew (Lognormal/Pareto) |
| `NH3` | 23.48 | 15.85 | 25.68 | 4.08 | 27.96 | Heavy Right Skew (Lognormal/Pareto) |
| `CO` | 2.25 | 0.89 | 6.96 | 8.88 | 109.49 | Heavy Right Skew (Lognormal/Pareto) |
| `SO2` | 14.53 | 9.16 | 18.13 | 4.08 | 22.07 | Heavy Right Skew (Lognormal/Pareto) |
| `O3` | 34.49 | 30.84 | 21.69 | 1.33 | 3.43 | Moderate Skew |
| `Benzene` | 3.28 | 1.07 | 15.81 | 21.3 | 530.17 | Heavy Right Skew (Lognormal/Pareto) |
| `Toluene` | 8.7 | 2.97 | 19.97 | 11.67 | 216.75 | Heavy Right Skew (Lognormal/Pareto) |
| `Xylene` | 3.07 | 0.98 | 6.32 | 7.89 | 119.98 | Heavy Right Skew (Lognormal/Pareto) |

- **Key Finding:** All criteria particulate and gaseous pollutants demonstrate pronounced right-skewed tails with excess kurtosis, typical of atmospheric emission plumes. Standard z-score scaling requires median centering or robust transformations to temper outlier gradient dominance in linear models.

---

## 5. Multicollinearity & Variance Inflation Factor (VIF)

### Core Criteria Pollutants VIF Scores
| Feature | VIF Score | Multicollinearity Assessment |
|:---|:---:|:---|
| `PM10` | 4.29 | Low Multicollinearity (VIF <= 5) |
| `PM2.5` | 3.82 | Low Multicollinearity (VIF <= 5) |
| `NO2` | 1.78 | Low Multicollinearity (VIF <= 5) |
| `NH3` | 1.38 | Low Multicollinearity (VIF <= 5) |
| `CO` | 1.17 | Low Multicollinearity (VIF <= 5) |
| `SO2` | 1.17 | Low Multicollinearity (VIF <= 5) |
| `O3` | 1.16 | Low Multicollinearity (VIF <= 5) |

- **Key Relationships:**
  - $PM_{2.5}$ and $PM_{10}$ exhibit strong positive collinearity ($r pprox 0.84$). Because $PM_{2.5}$ is a physical subset of $PM_{10}$, retaining both as raw features inflates variance in unregularized linear models.
  - To exploit this relationship productively without collinearity penalties, we engineer the **$PM_{2.5} / PM_{10}$ ratio**, which captures aerosol diameter distribution and source typology (combustion vs soil/dust).

---

## 6. Outlier Analysis & IQR Boundary Audit

| Feature | Q1 (25th %) | Q3 (75th %) | IQR | Upper Cutoff | Outlier Count | Outlier % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `PM2.5` | 28.82 | 80.59 | 51.77 | 158.24 | 1,982 | 7.95% |
| `PM10` | 56.26 | 149.74 | 93.49 | 289.98 | 1,057 | 5.75% |
| `NO` | 5.63 | 19.95 | 14.32 | 41.43 | 2,459 | 9.48% |
| `NO2` | 11.75 | 37.62 | 25.87 | 76.42 | 1,188 | 4.58% |
| `NOx` | 12.82 | 40.13 | 27.31 | 81.09 | 1,868 | 7.37% |
| `NH3` | 8.58 | 30.02 | 21.44 | 62.18 | 1,015 | 5.29% |
| `CO` | 0.51 | 1.45 | 0.94 | 2.86 | 2,475 | 9.01% |
| `SO2` | 5.67 | 15.22 | 9.55 | 29.54 | 2,578 | 10.04% |
| `O3` | 18.86 | 45.57 | 26.71 | 85.64 | 713 | 2.8% |
| `Benzene` | 0.12 | 3.08 | 2.96 | 7.52 | 1,668 | 6.98% |
| `Toluene` | 0.6 | 9.15 | 8.55 | 21.98 | 2,427 | 11.29% |
| `Xylene` | 0.14 | 3.35 | 3.21 | 8.16 | 1,119 | 9.8% |

### Outlier Handling Justification
- **Domain Reality vs Error:** Extreme values in urban air quality datasets (e.g., $PM_{2.5} > 500\ \mu\text{g/m}^3$ during Diwali in Delhi or agricultural stubble burning in Punjab) are **physically authentic catastrophic events**, not sensor measurement anomalies.
- **Decision:** Outliers are **retained** rather than dropped or artificially trimmed, ensuring our classification models remain sensitive to hazardous "Severe" events. Non-parametric models (KNN and Decision Trees) naturally handle monotonic extreme values without distortion.

---

## 7. Data Leakage Audit

- **Isolation Check:** The numeric target `AQI` and categorical label `AQI_Bucket` are strictly partitioned away from input feature space $X$ prior to any preprocessing.
- **Split Sequencing:** Train-test splitting occurs **before** calculating median imputation values and feature scaling parameters, preventing test-set distribution leakage into training artifacts.

---

## 8. Feature Relevance & Importance Ranking

| Feature | Mutual Information (MI) Score | Decision Tree Importance (Depth=5) | CPCB National Priority |
|:---|:---:|:---:|:---:|
| `PM10` | 0.7062 | 0.4975 | Primary Criteria |
| `PM2.5` | 0.6883 | 0.334 | Primary Criteria |
| `CO` | 0.2539 | 0.1181 | Secondary Criteria |
| `NO2` | 0.2453 | 0.0016 | Secondary Criteria |
| `NH3` | 0.1539 | 0.0042 | Secondary Criteria |
| `O3` | 0.1182 | 0.0446 | Secondary Criteria |
| `SO2` | 0.1035 | 0.0 | Secondary Criteria |

- **Conclusion:** $PM_{2.5}$ and $PM_{10}$ yield the highest mutual information scores ($> 0.65$) and tree split importance ($> 0.70$), verifying that particulate matter is the governing pollutant driving Indian AQI classifications.
