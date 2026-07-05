# Presentation Content — AI Final Examination
## Student Performance Prediction Agent

*Copy each slide section into PowerPoint / Google Slides. Replace placeholders before submission.*

---

## Slide 1: Agent

### Title
**Student Performance Prediction Agent**

### Content

| Field | Value |
|-------|-------|
| **Agent Name** | Student Performance Prediction Agent (SPPA) |
| **Goal** | Predict student academic performance category (Low / Medium / High) and recommend early interventions before final examinations |
| **Student Name** | [Your Name] |
| **Semester** | [e.g., Spring 2026] |
| **Group** | [Group Number] |
| **Course** | Artificial Intelligence — Final Examination |

### Visual Suggestion
Agent icon + school illustration. Include university logo.

### Key Message
> An intelligent AI Agent — not merely a machine learning script — that helps teachers identify at-risk students early.

---

## Slide 2: Russell & Norvig Environment Analysis

### Title
**Environment Properties**

### Table

| Property | Classification | Justification |
|----------|---------------|---------------|
| **Observable vs Partially Observable** | **Partially Observable** | Motivation, home stress, learning disabilities not in dataset |
| **Deterministic vs Stochastic** | **Stochastic** | Probabilistic outcomes; Random Forest outputs P(class) |
| **Sequential vs Episodic** | **Sequential** | G1, G2 prior grades affect G3; interventions affect future |
| **Static vs Dynamic** | **Dynamic** | Absences and grades change throughout semester |
| **Discrete vs Continuous** | **Discrete** (decisions) | Categories: Low, Medium, High |
| **Single-Agent vs Multi-Agent** | **Multi-Agent** | Teachers, students, parents, counselors interact |

### Visual Suggestion
6-row table with color-coded cells. Optional: small PEAS diagram.

---

## Slide 3: Dataset

### Title
**Dataset & Sensors**

### Content

| Item | Detail |
|------|--------|
| **Source** | UCI Machine Learning Repository — Student Performance (Cortez & Silva, 2008) |
| **Description** | 1,044 secondary school student records from Portugal (Math + Portuguese courses), 33 features |
| **Sensors** | Student Information System (grades), Attendance System (absences), Enrollment Forms (demographics), Surveys (study habits) |
| **Target** | G3 final grade (0–20) → performance category |

### Sample Data

| age | sex | studytime | failures | absences | G1 | G2 | G3 | Category |
|-----|-----|-----------|----------|----------|----|----|-----|----------|
| 15 | F | 2 | 3 | 10 | 7 | 8 | 10 | Medium |
| 18 | M | 1 | 2 | 20 | 8 | 7 | 6 | Low |
| 17 | F | 4 | 0 | 2 | 15 | 14 | 15 | High |

### Problems Identified
- Class imbalance (Low 22%, Medium 50%, High 28%)
- Outliers in absences (max 93)
- Self-reported behavioral features
- 382 students overlap between Math and Portuguese files

### Visual Suggestion
Sample data table + UCI logo + sensor diagram (SIS → Agent).

---

## Slide 4: Exploratory Data Analysis (EDA)

### Title
**Data Analysis**

### Correlation Highlights
- G2 ↔ G3: r ≈ **0.90** (strongest predictor)
- G1 ↔ G3: r ≈ **0.80**
- failures ↔ G3: r ≈ **-0.35**

### Statistics
- Mean G3: **11.9** (σ = 3.2)
- Mean absences: **5.5** (right-skewed)
- Missing values: **0** in UCI source

### Distributions
- G3: approximately normal, centered at 12
- Performance categories: Low 22% · Medium 50% · High 28%

### Charts (insert images)
1. `outputs/correlation_heatmap.png`
2. `outputs/class_distribution.png`
3. `outputs/pairplot.png` (optional)

### Key Insight
> Prior grades (G1, G2) and failure history are the strongest predictors of final performance.

---

## Slide 5: Model

### Title
**Machine Learning — Learning Component**

### Architecture
```
Student Features → StandardScaler + OneHotEncoder → Random Forest (200 trees) → P(Low), P(Med), P(High)
```

### Learning Type
**Supervised Multi-Class Classification**

### Hyperparameters (Random Forest — Best Model)
| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 12 |
| min_samples_split | 5 |
| class_weight | balanced_subsample |
| random_state | 42 |

### Model Comparison (insert `outputs/model_comparison.png`)

| Model | F1 Score | Accuracy |
|-------|----------|----------|
| **Random Forest** | **0.8615** | **0.8612** |
| Logistic Regression | 0.8516 | 0.8517 |
| XGBoost | 0.8468 | 0.8469 |
| Decision Tree | 0.8467 | 0.8469 |
| Gradient Boosting | 0.8371 | 0.8373 |
| SVM | 0.8220 | 0.8230 |
| Extra Trees | 0.7525 | 0.7560 |

### Why Best?
Highest F1, stable CV, handles mixed features, provides feature importance.

---

## Slide 6: Results

### Title
**Evaluation Results — Random Forest**

### Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 86.12% |
| **Precision** | 86.36% |
| **Recall** | 86.12% |
| **F1 Score** | 86.15% |
| **ROC AUC** | 94.61% |
| **CV Accuracy** | 84.67% ± 1.95% |
| **Train / Test split** | 835 / 209 students |

### Confusion Matrix
Insert: `outputs/confusion_matrix.png`

### Feature Importance
Insert: `outputs/feature_importance.png`

### Insights
1. G2 (second period grade) is the #1 predictor
2. Main errors at Medium ↔ High boundary (grades 13–14)
3. Model catches majority of Low performers — critical for intervention
4. Training time: 0.52s | Prediction: 46ms (scalable to full school)

---

## Slide 7: Agent Integration

### Title
**From Prediction to Action**

### Decision Making Flow
```
Percept → Preprocess → ML Probabilities → Risk Assessment → Intervention Rules → Teacher Dashboard
```

### Real-World Use
- **Teacher Dashboard:** Scores all **1,044 students**; **233 HIGH priority**, 525 MEDIUM, 286 LOW
- **Notifications:** Alerts counselor when risk = Critical (270 students)
- **Recommendations:** Tutoring, counseling, parent meetings based on category + features

### Agent Components
| Component | Implementation |
|-----------|----------------|
| Learning | Random Forest (`best_model.pkl`) |
| Knowledge Base | Intervention rule templates |
| Decision Engine | Risk escalation + rule merge |
| Memory | Per-student prediction history |
| Actions | Dashboard CSV, alerts |

### Limitations
- Portuguese school data only
- ~14% misclassification rate
- Potential socioeconomic bias
- Privacy requirements (FERPA/GDPR)
- Cannot capture hidden motivation factors

### Future Improvements
- SHAP explainability per student
- Real-time absence streaming
- Automated retraining pipeline
- LLM-generated parent conference summaries
- Federated learning across schools

### Closing
> **Thank you. Questions welcome.**

### Visual Suggestion
Architecture diagram from `docs/AI_AGENT_DESIGN.md` (Mermaid export).

---

## Appendix: Demo Command (for Q&A)

```bash
python main.py --predict
```

Sample output: Category=Low, Risk=Critical, 6 personalized recommendations.
