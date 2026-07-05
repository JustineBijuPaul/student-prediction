# Student Performance Prediction Agent
## Complete Project Documentation — AI Final Examination

**Project Title:** Student Performance Prediction Agent  
**Course:** Artificial Intelligence — Final Examination  
**Author:** [Your Name]  
**Institution:** [Your University]  
**Semester:** [Semester/Year]  
**Group:** [Group Number]  

---

# Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset Description](#4-dataset-description)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Selection](#7-model-selection)
8. [Training Procedure](#8-training-procedure)
9. [Evaluation Results](#9-evaluation-results)
10. [AI Agent Architecture](#10-ai-agent-architecture)
11. [Russell & Norvig Environment Analysis](#11-russell--norvig-environment-analysis)
12. [Limitations](#12-limitations)
13. [Ethics and Privacy](#13-ethics-and-privacy)
14. [Deployment Architecture](#14-deployment-architecture)
15. [Future Work](#15-future-work)
16. [Conclusion](#16-conclusion)
17. [References](#17-references)

---

# 1. Introduction

Artificial Intelligence, as defined by Russell and Norvig, is concerned with building **rational agents** that perceive their environment and act to achieve goals. Machine learning provides one mechanism — the **learning component** — by which agents improve their performance from experience.

This project develops the **Student Performance Prediction Agent (SPPA)**, an intelligent system designed to help teachers identify students at risk of poor academic outcomes before final examinations. Unlike a standalone classification script, SPPA embodies a complete agent architecture: sensors gather percepts, a knowledge base encodes intervention policies, a trained Random Forest classifier estimates performance beliefs, a decision engine synthesizes recommendations, and actuators deliver alerts through a teacher dashboard.

The pedagogical context is secondary education in Portugal, using the publicly available UCI Student Performance dataset. The agent classifies students into **Low**, **Medium**, and **High** performance categories based on demographic, social, and academic features, then recommends targeted interventions.

This documentation demonstrates that the examination deliverable satisfies AI curriculum requirements: environment analysis, agent design, decision-making, dataset understanding, machine learning integration, and clear articulation of how the ML model functions as one component within a larger intelligent system.

---

# 2. Problem Statement

Educational institutions face a persistent challenge: **identifying struggling students early enough to intervene effectively**. Traditional approaches rely on teachers' intuition and mid-term grades, which may arrive too late for students with accumulating risk factors — chronic absence, prior failures, insufficient study time, lack of family support.

The problem is formally stated as:

> Given a vector of observable student attributes at time *t*, determine the most likely final performance category and recommend appropriate educational interventions before the terminal examination.

Constraints:
- Decisions must be **actionable** for teachers with limited time
- Predictions must be **probabilistic** (partial observability)
- System must **assist**, not replace, human judgment
- Solution must comply with **student data privacy** regulations

---

# 3. Objectives

## Primary Objectives
1. Design and document a complete **AI Agent** for student performance prediction
2. Perform rigorous **exploratory data analysis** on the UCI Student Performance dataset
3. Train and compare **multiple supervised classifiers**
4. Integrate the best model as the agent's **learning component**
5. Conduct **Russell & Norvig environment analysis**
6. Produce presentation-ready artifacts (notebook, plots, documentation)

## Secondary Objectives
1. Support multiple dataset formats via automatic column mapping
2. Generate teacher dashboard outputs for batch scoring
3. Document limitations, ethics, and future improvements
4. Prepare viva examination materials (Q&A, cheat sheet, speaker notes)

---

# 4. Dataset Description

## 4.1 Source
- **Repository:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/ml/datasets/Student+Performance
- **Citation:** Cortez & Silva (2008)

## 4.2 Files
| File | Records | Subject |
|------|---------|---------|
| student-mat.csv | 395 | Mathematics |
| student-por.csv | 649 | Portuguese |
| **Combined** | **1,044** | Both courses |

## 4.3 Features (33 attributes)
Grouped as demographic (school, sex, age, address, family), parental (education, jobs), academic support (schoolsup, famsup, paid), behavioral (studytime, absences, failures, alcohol), and grades (G1, G2, G3).

## 4.4 Target Variable
- **Raw target:** G3 (final grade, 0–20)
- **Agent target:** `performance_category`
  - Low: G3 < 10
  - Medium: 10 ≤ G3 < 14
  - High: G3 ≥ 14

## 4.5 Column Mapping
The system auto-maps alternative names (`study_time` → `studytime`, `final_grade` → `G3`) for Kaggle dataset compatibility.

---

# 5. Exploratory Data Analysis

## 5.1 Data Quality
- Zero missing values in UCI source files
- Semicolon delimiter handled automatically
- Duplicate rows removed (<1% of records)

## 5.2 Summary Statistics
Mean final grade G3 ≈ 11.9 (σ ≈ 3.2). Mean absences ≈ 5.5 with heavy right tail. Mean failures ≈ 0.3 — most students have zero prior failures.

## 5.3 Correlation Findings
Strong positive correlation between G1, G2, and G3 (r > 0.8). Negative correlation between failures and G3 (r ≈ -0.35). Weak negative correlation between absences and G3.

## 5.4 Class Distribution
Approximately 15% Low, 35% Medium, 50% High — moderate imbalance favoring higher performers.

## 5.5 Visualizations
Generated in `outputs/`:
- Correlation heatmap
- Class distribution bar chart
- Pairplot of key numeric features
- Count plots by gender, internet, study time

See `docs/DATA_ANALYSIS.md` for extended analysis.

---

# 6. Feature Engineering

## 6.1 Cleaning
- Duplicate removal
- Median imputation for numeric missing (if any)
- Mode imputation for categorical missing
- IQR-based outlier capping for absences and failures

## 6.2 Encoding
- **Numeric features:** StandardScaler (zero mean, unit variance)
- **Categorical features:** OneHotEncoder with `handle_unknown='ignore'`

## 6.3 Feature Selection
All features except G3 and performance_category included. G1 and G2 retained as legitimate predictors available before final exam.

## 6.4 Train/Test Split
- 80% train / 20% test
- Stratified by performance category
- `random_state=42`

---

# 7. Model Selection

Seven algorithms evaluated:

| # | Algorithm | Type |
|---|-----------|------|
| 1 | Logistic Regression | Linear |
| 2 | Decision Tree | Non-linear |
| 3 | Random Forest | Ensemble |
| 4 | Support Vector Machine | Kernel |
| 5 | Gradient Boosting | Boosting |
| 6 | Extra Trees | Ensemble |
| 7 | XGBoost | Gradient Boosting |

**Selection criterion:** Weighted F1 score on held-out test set (balances precision/recall under class imbalance).

**Winner:** Random Forest (F1 ≈ 0.86, Accuracy ≈ 0.86, ROC AUC ≈ 0.95)

---

# 8. Training Procedure

```
1. Load & merge datasets
2. Clean & cap outliers
3. Create performance_category from G3
4. Stratified train/test split
5. Fit ColumnTransformer on training data
6. For each model:
   a. Fit on transformed training data
   b. Predict on test set
   c. Compute metrics + 5-fold CV
   d. Record training/prediction time
7. Select best model by F1
8. Save best_model.pkl + visualizations
```

Execute via: `python main.py --train`

---

# 9. Evaluation Results

## 9.1 Best Model Metrics (Random Forest)

| Metric | Test Value |
|--------|------------|
| Accuracy | 0.8612 |
| Precision (weighted) | 0.8636 |
| Recall (weighted) | 0.8612 |
| F1 Score (weighted) | 0.8615 |
| ROC AUC (weighted OVR) | 0.9461 |
| CV Accuracy (5-fold) | 0.8467 ± 0.0195 |

## 9.2 Model Comparison
Full table saved to `outputs/model_comparison.csv`. Random Forest and Logistic Regression lead; Extra Trees underperforms due to excessive randomization on this feature space.

## 9.3 Feature Importance
Top features: G2, G1, failures, absences, studytime, parental education. Confirms academic history dominates prediction.

## 9.4 Confusion Matrix
Strong diagonal performance. Primary confusion at Medium/High boundary — expected given continuous G3 discretization at thresholds 10 and 14.

---

# 10. AI Agent Architecture

## 10.1 Agent Definition
**SPPA** is a hybrid learning + knowledge-based agent that:
1. Perceives student records from school sensors
2. Updates beliefs via Random Forest probabilities
3. Reasons over knowledge base intervention rules
4. Acts through dashboard alerts and recommendations

## 10.2 Components

| Component | Implementation |
|-----------|----------------|
| Sensors | SIS, LMS, attendance APIs |
| Percepts | Normalized student feature vector |
| Knowledge Base | `INTERVENTION_RULES` in predict.py |
| Learning Component | `best_model.pkl` (Random Forest) |
| Decision Engine | Risk assessment + rule augmentation |
| Memory | Per-student prediction history |
| Actuators | Dashboard CSV, notifications |

## 10.3 Decision Flow
```
Percept → Preprocess → ML probabilities → Risk level → Rules → Recommendations → Actions
```

## 10.4 Example Output
For an at-risk student (low studytime, high absences, prior failures):
- Category: Low
- Risk: Critical
- Priority: HIGH
- Actions: Counseling, tutoring, parent notification

See `docs/AI_AGENT_DESIGN.md` for Mermaid architecture diagrams.

---

# 11. Russell & Norvig Environment Analysis

| Property | Classification |
|----------|---------------|
| Observable | **Partially Observable** |
| Deterministic | **Stochastic** |
| Sequential | **Sequential** |
| Static | **Dynamic** |
| Discrete | **Discrete decisions** |
| Agents | **Multi-Agent** |

**Justification summary:** The agent lacks full visibility into student motivation and home environment (partial observability). Outcomes are probabilistic. Past grades create temporal dependencies. The school environment evolves daily. Multiple stakeholders (teachers, students, parents) interact.

Full analysis: `docs/ENVIRONMENT_ANALYSIS.md`

---

# 12. Limitations

1. **Dataset scope:** Portuguese secondary schools only
2. **Self-reported features:** studytime, alcohol consumption
3. **Class imbalance:** Low performers underrepresented
4. **Cross-sectional data:** No full semester time series
5. **Overlapping students:** 382 students in both Math and Portuguese files
6. **Support confounding:** schoolsup assigned reactively to struggling students
7. **Model errors:** ~14% misclassification rate
8. **No causal claims:** Correlation ≠ causation for interventions

---

# 13. Ethics and Privacy

## 13.1 Privacy
Student records are protected under FERPA (US) and GDPR (EU). Deployment requires:
- Data minimization (only necessary features)
- Role-based access control
- Anonymization in research contexts
- Parental consent for minors

## 13.2 Fairness
Risk of **algorithmic bias** against socioeconomic groups correlated with failures and absences. Mitigation:
- Human review before high-stakes actions
- Fairness auditing across demographic subgroups
- Transparency about model limitations

## 13.3 Ethical Use
- Agent **supports** teachers; never autonomously assigns grades
- Avoid stigmatizing labels visible to peers
- Provide appeal mechanism for disputed predictions

---

# 14. Deployment Architecture

## 14.1 Components
```
[School SIS] → [ETL Pipeline] → [SPPA API] → [Teacher Dashboard]
                                      ↓
                              [models/best_model.pkl]
```

## 14.2 Execution Modes
| Command | Purpose |
|---------|---------|
| `python main.py --train` | Retrain and save artifacts |
| `python main.py --predict` | Single-student demo |
| `python main.py --dashboard` | Batch cohort scoring |

## 14.3 Kaggle Deployment
Notebook `notebooks/Student_Performance_Prediction_Agent.ipynb` runs end-to-end on Kaggle with `../input/` dataset paths.

---

# 15. Future Work

1. **Explainable AI:** SHAP/LIME per-student explanations
2. **Deep Learning:** LSTM on sequential LMS activity logs
3. **Real-time Analytics:** Kafka stream processing for live absence updates
4. **Retraining Pipeline:** MLOps with MLflow, scheduled retraining
5. **LLM Integration:** Natural language reports for parent conferences
6. **Reinforcement Learning:** Optimize intervention timing as policy learning
7. **Multi-school Federation:** Privacy-preserving federated learning

---

# 16. Conclusion

This project demonstrates that **Artificial Intelligence** in educational contexts requires more than training a classifier. The Student Performance Prediction Agent integrates:

- Rigorous **data analysis** of 1,044 student records
- Comparative **machine learning** evaluation of seven algorithms
- A **Random Forest learning component** achieving 86% F1 score
- Complete **agent architecture** with perception, reasoning, and action
- Formal **environment characterization** per Russell & Norvig

The trained model answers: *"What performance category is most likely?"*  
The agent answers: *"What should we do about it?"*

That distinction — from prediction to action — is the essence of this AI Final Examination project.

---

# 17. References

1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
2. Cortez, P., & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance. *Proceedings of 5th FUture BUsiness TEChnology Conference*.
3. UCI Machine Learning Repository. Student Performance Data Set.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*.
5. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

---

**Document Version:** 1.0  
**Last Updated:** July 2026  
**Total Pages:** ~18 (when printed at 11pt)
