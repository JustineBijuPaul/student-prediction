# Data Analysis — Student Performance Dataset

## 1. Dataset Source

| Attribute | Detail |
|-----------|--------|
| **Name** | Student Performance Data Set |
| **Repository** | UCI Machine Learning Repository |
| **Authors** | Paulo Cortez, University of Minho; Alice Silva, Alcochete School |
| **Instances** | 649 (Portuguese) + 395 (Math) = **1,044** combined |
| **Features** | 33 attributes (demographic, social, academic) |
| **Target** | G3 — final period grade (0–20) |

---

## 2. Feature Categories

### Demographic
- `school`, `sex`, `age`, `address`, `famsize`, `Pstatus`
- `Medu`, `Fedu` — parental education (0–4 ordinal)
- `Mjob`, `Fjob` — parental occupation

### Academic Support
- `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`

### Behavioral / Lifestyle
- `studytime`, `traveltime`, `failures`, `absences`
- `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `romantic`

### Grades
- `G1` — first period (0–20)
- `G2` — second period (0–20)
- `G3` — final grade (0–20) → mapped to performance category

---

## 3. Summary Statistics

Key numeric features (combined dataset):

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| age | ~16.7 | ~1.3 | 15 | 22 |
| studytime | ~2.0 | ~0.8 | 1 | 4 |
| failures | ~0.3 | ~0.6 | 0 | 3 |
| absences | ~5.5 | ~7.0 | 0 | 93 |
| G1 | ~11.4 | ~3.0 | 0 | 19 |
| G2 | ~11.6 | ~3.0 | 0 | 19 |
| G3 | ~11.9 | ~3.2 | 0 | 19 |

---

## 4. Missing Values & Data Quality

- **Missing values**: None in original UCI files (verified)
- **Duplicates**: Minimal; removed during preprocessing
- **Delimiter**: Semicolon (`;`) — handled automatically
- **Encoding**: Categorical strings (`yes`/`no`, `M`/`F`)

---

## 5. Target Distribution (Performance Categories)

| Category | G3 Range | Approx. % |
|----------|----------|-----------|
| Low | 0–9 | ~15% |
| Medium | 10–13 | ~35% |
| High | 14–20 | ~50% |

### Class Imbalance Analysis
The **Low** category is underrepresented (~15%). Mitigation strategies:
- `class_weight='balanced_subsample'` in Random Forest
- Stratified train/test split
- Focus on **recall** for Low class in evaluation
- SMOTE considered but not applied (interpretability priority)

---

## 6. Correlation Analysis

### Strong Positive Correlations with G3
| Pair | r ≈ | Interpretation |
|------|-----|----------------|
| G2 ↔ G3 | 0.90+ | Second period grade is strongest predictor |
| G1 ↔ G3 | 0.80+ | First period grade highly predictive |
| G1 ↔ G2 | 0.85+ | Grades stable across periods |

### Negative Correlations with G3
| Pair | r ≈ | Interpretation |
|------|-----|----------------|
| failures ↔ G3 | -0.35 | Past failures predict lower finals |
| absences ↔ G3 | -0.10 to -0.20 | More absences, lower grades |

### Covariance Insight
G1, G2, G3 covary strongly — students who start well tend to finish well. Interventions should target students with **declining G1→G2 trajectories** even if G2 alone looks acceptable.

---

## 7. Distribution Analysis

### G3 Distribution
- Approximately normal, centered near 12
- Slight left skew due to failing students
- No grades at exactly 20 (max observed: 19)

### Absences
- Right-skewed: most students have 0–10 absences
- Outliers up to 93 (capped via IQR in preprocessing)

### Study Time
- Ordinal 1–4; mode at 2 (2–5 hours/week)
- Higher studytime associated with higher performance in count plots

---

## 8. Categorical Analysis

### Gender (`sex`)
- Balanced M/F distribution
- Slight performance differences not statistically dominant alone

### Internet Access
- Students with `internet=yes` show marginally better outcomes
- Important for remote learning equity analysis

### School Support (`schoolsup`)
- Students receiving extra support more prevalent in lower grades
- **Confounding**: support is assigned to struggling students

---

## 9. Outlier Detection

**Method**: IQR rule (Q1 − 1.5×IQR, Q3 + 1.5×IQR)

| Feature | Outlier Action |
|---------|----------------|
| absences | Winsorized |
| failures | Winsorized |
| G1, G2 | Retained (legitimate extremes) |

---

## 10. Feature Relationships (Pairplot)

Pairwise plots of `age`, `studytime`, `absences`, `G1`, `G2`, `G3` reveal:
- Clear grade clusters by performance category
- absences vs G3: scattered negative trend
- studytime vs G3: weak positive trend

---

## 11. Feature Importance (Post-Modeling)

Top predictors from Random Forest:
1. **G2** — most recent assessment
2. **G1** — prior assessment
3. **failures** — historical failure count
4. **absences** — attendance proxy
5. **studytime** — self-reported effort
6. **Medu/Fedu** — parental education (encoded)

**Interpretation**: Academic history dominates; demographic features modulate risk.

---

## 12. Patterns & Insights

1. **Grade momentum matters**: G1→G2 trajectory predicts G3
2. **Failure history is persistent**: failures > 0 strongly associated with Low category
3. **Attendance counts**: Chronic absence (>15) warrants intervention regardless of ML score
4. **Support paradox**: schoolsup=yes correlates with lower grades because support is reactive
5. **Course effect**: Math vs Portuguese students show slightly different distributions

---

## 13. Data Limitations

- Self-reported study time and alcohol consumption
- Single school system (Portugal) — limited geographic generalization
- 382 students appear in both Math and Portuguese datasets (related records)
- No timestamp — cross-sectional snapshot, not full time series

---

## 14. Visualization Index

| File | Content |
|------|---------|
| `outputs/correlation_heatmap.png` | Pearson correlation matrix |
| `outputs/class_distribution.png` | Target bar chart |
| `outputs/pairplot.png` | Pairwise relationships |
| `outputs/feature_importance.png` | Random Forest importances |

See notebook Section 4 for interactive exploration.
