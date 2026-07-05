# Model Explanation — Student Performance Prediction Agent

## 1. Learning Paradigm

### Supervised Multi-Class Classification

We use **supervised learning** because:
- Historical records include **labeled outcomes** (G3 → performance category)
- The agent must learn a mapping \( f: \mathbf{X} \rightarrow \{Low, Medium, High\} \)
- Teacher feedback and grade book data provide ground truth

This is not unsupervised (no labels), not reinforcement learning (no sequential reward signal), though RL could optimize intervention policies in future work.

---

## 2. Selected Model: Random Forest

### Why Random Forest Won

From our comparison of 7 algorithms on the held-out test set:

| Model | F1 Score | ROC AUC |
|-------|----------|---------|
| **Random Forest** | **~0.86** | **~0.95** |
| Logistic Regression | ~0.85 | ~0.95 |
| XGBoost | ~0.85 | ~0.95 |
| Decision Tree | ~0.85 | ~0.94 |
| Gradient Boosting | ~0.84 | ~0.95 |
| SVM | ~0.82 | ~0.93 |
| Extra Trees | ~0.75 | ~0.89 |

**Random Forest** achieved the highest F1 with strong stability across 5-fold cross-validation.

---

## 3. Why Random Forest (Detailed)

### Advantages
1. **Handles mixed data**: Numeric + categorical (via preprocessing pipeline)
2. **Non-linear interactions**: Captures failure × absences without manual feature crosses
3. **Robust to outliers**: Tree splits insensitive to extreme absences after capping
4. **Feature importance**: Interpretable rankings for teacher explanations
5. **Ensemble variance reduction**: 200 bagged trees reduce overfitting vs single tree
6. **Class imbalance**: `class_weight='balanced_subsample'` boosts minority Low class
7. **Fast inference**: ~45ms batch prediction — suitable for real-time dashboard

### Disadvantages
1. Less interpretable than single decision tree or logistic regression
2. Larger model size than linear models
3. Extrapolation poor outside training grade ranges
4. Correlated trees if features highly collinear (G1, G2)

---

## 4. Why Not Other Models?

| Model | Reason Not Selected |
|-------|---------------------|
| **Logistic Regression** | Assumes linear decision boundaries; G1/G2 interactions need manual engineering |
| **Decision Tree** | Higher variance; single tree overfits without ensemble |
| **SVM** | Slower on larger feature spaces; probability calibration overhead |
| **Gradient Boosting** | Comparable accuracy but slower training; more hyperparameter sensitivity |
| **Extra Trees** | Underperformed on our split (F1 ~0.75) — excessive randomization |
| **XGBoost** | Tie with RF on accuracy; RF selected for simplicity and sklearn-native integration |

---

## 5. Hyperparameters

```python
RandomForestClassifier(
    n_estimators=200,        # Number of trees in forest
    max_depth=12,            # Limit tree depth to prevent overfitting
    min_samples_split=5,     # Minimum samples to split internal node
    min_samples_leaf=2,      # Minimum samples per leaf
    class_weight='balanced_subsample',  # Auto-balance classes per bootstrap
    random_state=42,         # Reproducibility
    n_jobs=-1                # Parallel training
)
```

### Rationale
- **n_estimators=200**: Diminishing returns beyond 200 on this dataset size
- **max_depth=12**: Deep enough for interactions, shallow enough to generalize
- **class_weight**: Addresses Low class underrepresentation

---

## 6. Model Complexity

| Metric | Value |
|--------|-------|
| Training samples | ~835 |
| Test samples | ~209 |
| Input features (raw) | ~32 |
| Encoded features | ~60+ (after one-hot) |
| Trees | 200 |
| Approx. model size | ~5–15 MB (joblib) |

**Bias-Variance Tradeoff**: Ensemble reduces variance compared to single Decision Tree while maintaining low bias via depth-12 trees.

---

## 7. Preprocessing Pipeline

```
Raw Features
    → ColumnTransformer
        → Numeric: StandardScaler
        → Categorical: OneHotEncoder(handle_unknown='ignore')
    → Transformed matrix → Random Forest
```

**No leakage**: G3 excluded from features; G1/G2 retained as legitimate predictors available before final exam.

---

## 8. Evaluation Metrics Explained

| Metric | Value | Meaning |
|--------|-------|---------|
| **Accuracy** | ~86% | Correct predictions overall |
| **Precision** | ~86% | When agent says Low, usually correct |
| **Recall** | ~86% | Agent catches most actual Low students |
| **F1** | ~86% | Harmonic mean — balances precision/recall |
| **ROC AUC** | ~95% | Strong class separability |

**Why F1 for selection?** Class imbalance makes accuracy misleading; F1 weights Low class performance.

---

## 9. Confusion Matrix Interpretation

Typical pattern:
- Strong diagonal — most predictions correct
- Main errors: Medium ↔ High boundary (G3 near 13–14)
- Fewer Low ↔ High confusions — model discriminates extremes well

**Actionable insight**: Borderline Medium students need human review before high-stakes interventions.

---

## 10. Cross-Validation

5-fold CV on training set:
- Mean CV accuracy ~85%
- Std ~2%
- Low variance → model stable across data partitions

---

## 11. Integration as Learning Component

The Random Forest is **not the agent** — it is the **belief update mechanism**:

```
P(category | percepts) = RandomForest.predict_proba(X)
```

The **Decision Engine** (`src/predict.py`) consumes these probabilities to:
1. Select performance category
2. Assess risk level
3. Query knowledge base for interventions
4. Emit prioritized actions

This separation embodies the Russell & Norvig distinction between **learning** (inducing beliefs from data) and **acting** (selecting actions from beliefs).

---

## 12. Retraining Considerations

Retrain when:
- New semester data exceeds 20% of training set
- CV accuracy drops >5% on validation
- School policy changes (grading scale)

Pipeline: `python main.py --train` — fully automated.

---

## 13. Explainability Roadmap

Current: Global feature importance plot  
Future: SHAP values per student for parent-teacher conferences

---

## 14. Conclusion

Random Forest provides the optimal balance of **accuracy**, **robustness**, **interpretability**, and **inference speed** for tabular student data. Embedded within the Student Performance Prediction Agent, it transforms raw educational percepts into calibrated beliefs that drive rational intervention decisions.
