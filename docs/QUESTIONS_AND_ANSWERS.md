# Viva Questions & Answers
## Student Performance Prediction Agent — 55 Questions

---

## Section A: Artificial Intelligence & Agent Design (1–15)

### Q1. What is an AI Agent?
**A:** An AI agent is anything that perceives its environment through sensors and acts upon that environment through actuators to achieve a goal. Our Student Performance Prediction Agent perceives student records and acts by generating intervention recommendations for teachers.

### Q2. Why is this project considered AI and not just Machine Learning?
**A:** Machine learning is only the **learning component**. The complete agent includes sensors (SIS, attendance), a knowledge base (intervention rules), a decision engine (risk assessment), memory (prediction history), and actuators (dashboard, notifications). ML provides beliefs; the agent converts beliefs into actions.

### Q3. What is the agent's goal?
**A:** To predict student performance category (Low/Medium/High) and recommend timely educational interventions before final examinations, helping teachers identify at-risk students early.

### Q4. What are percepts in your agent?
**A:** Percepts are the processed inputs the agent receives about a student: age, sex, study time, absences, prior grades (G1, G2), failures, family support, internet access, and other features from the student record vector.

### Q5. What sensors does the agent use?
**A:** In deployment: Student Information System (grades), attendance tracking system (absences), enrollment forms (demographics), and surveys/LMS (study habits). In our project, these are simulated by CSV data from UCI.

### Q6. What is the knowledge base?
**A:** Stored domain knowledge including performance category thresholds (G3 < 10 = Low), intervention rule templates per category, feature semantics, and privacy/escalation policies. Implemented in `INTERVENTION_RULES` in `src/predict.py`.

### Q7. What is the decision engine?
**A:** The component that combines ML probability outputs with knowledge base rules to determine risk level, intervention priority, and specific recommendations. It escalates risk when P(Low) ≥ 0.35 regardless of top predicted class.

### Q8. What actions can the agent take?
**A:** Update teacher dashboard, send counselor notifications, recommend tutoring enrollment, suggest parent conferences, and prioritize intervention queues (HIGH/MEDIUM/LOW).

### Q9. What is agent memory in your system?
**A:** `StudentPerformanceAgent.memory` stores a list of past `AgentPrediction` objects per student ID, enabling longitudinal tracking and future trend analysis.

### Q10. What type of agent architecture is this?
**A:** A **hybrid learning + knowledge-based agent**: the Random Forest learns from data; the rule base handles intervention logic that is not easily learned from labels alone.

### Q11. What is the PEAS description?
**A:** **P**erformance: maximize early at-risk detection. **E**nvironment: schools, classrooms, homes. **A**ctuators: dashboards, notifications, referrals. **S**ensors: SIS, attendance, surveys.

### Q12. Is your agent autonomous?
**A:** Partially. It autonomously processes student data and generates recommendations, but human teachers validate and execute interventions — appropriate human-in-the-loop design for high-stakes education.

### Q13. What is the difference between a model and an agent?
**A:** A model maps inputs to outputs (classification). An agent operates in an environment over time, maintains state (memory), and selects actions to achieve goals. Our Random Forest is a model; SPPA is the agent wrapping it.

### Q14. What is rationality in this context?
**A:** A rational agent selects actions that maximize expected performance given its percepts and knowledge. We approximate this by maximizing correct risk identification (high F1) and applying utility-maximizing interventions.

### Q15. How does feedback improve the agent?
**A:** Teachers mark prediction accuracy; new semester grades provide updated labels. Periodic retraining (`python main.py --train`) improves the learning component; rule updates refine the knowledge base.

---

## Section B: Russell & Norvig Environment (16–25)

### Q16. Is the environment fully observable?
**A:** No, it is **partially observable**. Hidden variables include student motivation, sleep quality, undiagnosed learning disabilities, and domestic stress — none fully captured in the dataset.

### Q17. Is the environment deterministic or stochastic?
**A:** **Stochastic.** Identical feature vectors do not guarantee identical outcomes. The Random Forest outputs probability distributions reflecting this uncertainty.

### Q18. Is the environment episodic or sequential?
**A:** **Sequential.** Decisions depend on history — G1 and G2 grades from prior periods influence G3 prediction. Interventions today affect future percepts.

### Q19. Is the environment static or dynamic?
**A:** **Dynamic.** Absences accumulate, grades update, and family circumstances change during the semester while the agent operates.

### Q20. Is the state space discrete or continuous?
**A:** **Mixed.** Agent decisions are discrete (three categories) but internal beliefs are continuous probabilities. Raw features include both discrete categories and continuous integers.

### Q21. Is this a single-agent or multi-agent environment?
**A:** **Multi-agent.** Teachers, students, parents, counselors, and administrators all interact. SPPA's outputs influence other agents' behaviors.

### Q22. Why does partial observability matter for model design?
**A:** It requires probabilistic outputs (`predict_proba`) rather than hard labels, and risk escalation rules when failure probability is significant even if not the top class.

### Q23. Give an example of environmental dynamics.
**A:** A student has 5 absences at mid-semester (Medium risk). Two weeks later, absences reach 20 (elevated risk). The agent must re-score with updated percepts.

### Q24. How would the environment change in a different country?
**A:** Grading scales, cultural factors, and feature distributions differ. The agent would need retraining and knowledge base updates — demonstrating generalization limits.

### Q25. What is the accessibility of the environment state?
**A:** Recorded features are accessible via school systems; psychological and home environment states are inaccessible, requiring inference from partial percepts.

---

## Section C: Machine Learning (26–40)

### Q26. What type of machine learning is used?
**A:** **Supervised multi-class classification.** We train on labeled historical data where G3 grades define the class labels.

### Q27. Why supervised and not unsupervised?
**A:** We have ground truth labels (performance categories from G3). Unsupervised clustering would not align with educational outcome definitions.

### Q28. Why not reinforcement learning?
**A:** RL requires sequential reward signals from intervention outcomes. Our current data is cross-sectional with final grades, not intervention feedback loops. RL is planned future work.

### Q29. What is the target variable?
**A:** `performance_category`: Low (G3<10), Medium (10≤G3<14), High (G3≥14), derived from final grade G3.

### Q30. Why discretize G3 into categories instead of regression?
**A:** Categories align with actionable intervention tiers. Teachers think in risk bands, not exact grade point predictions. Classification also handles threshold-based policy decisions naturally.

### Q31. What features were excluded and why?
**A:** G3 was excluded from training features to prevent **target leakage** — it is the raw source of the label. G1 and G2 were retained as they are available before the final exam.

### Q32. What preprocessing was applied?
**A:** Duplicate removal, missing value imputation, IQR outlier capping, StandardScaler for numeric features, OneHotEncoder for categoricals, stratified 80/20 train-test split.

### Q33. Why stratified splitting?
**A:** To preserve the class distribution (especially the minority Low class) in both train and test sets, ensuring reliable evaluation.

### Q34. What is random_state=42?
**A:** A fixed random seed ensuring reproducible train/test splits and model initialization across runs — critical for scientific reproducibility.

### Q35. What models were compared?
**A:** Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting, Extra Trees, and XGBoost — seven classifiers total.

### Q36. Why did Random Forest win?
**A:** Highest F1 (86.15%), strong ROC AUC (94.61%), stable cross-validation (84.67% ± 1.95%), handles non-linear feature interactions, provides feature importance, and fast inference.

### Q37. What are Random Forest hyperparameters?
**A:** n_estimators=200, max_depth=12, min_samples_split=5, min_samples_leaf=2, class_weight='balanced_subsample', random_state=42.

### Q38. What is class_weight='balanced_subsample'?
**A:** Automatically adjusts weights inversely proportional to class frequencies in each bootstrap sample, addressing the underrepresented Low performance class.

### Q39. What is cross-validation and what did it show?
**A:** 5-fold CV splits training data into 5 parts, training on 4 and validating on 1 iteratively. Mean accuracy ~84.7% with low std (~2%) indicates stable generalization.

### Q40. What is overfitting and how was it prevented?
**A:** Overfitting is when a model memorizes training data. We prevented it via max_depth limits, ensemble averaging (200 trees), cross-validation monitoring, and comparing train vs test metrics.

---

## Section D: Evaluation Metrics (41–48)

### Q41. What is accuracy?
**A:** Proportion of correct predictions: (TP+TN)/Total. Our Random Forest achieves **86.12%** on the test set (209 students).

### Q42. What is precision?
**A:** Of all students predicted as Low (or any class), what fraction truly belong to that class: TP/(TP+FP). Important to avoid false alarms.

### Q43. What is recall?
**A:** Of all truly Low students, what fraction did we correctly identify: TP/(TP+FN). Critical for intervention — we must not miss at-risk students.

### Q44. What is F1 score?
**A:** Harmonic mean of precision and recall: 2×P×R/(P+R). Used as our model selection metric because it balances both under class imbalance.

### Q45. What is ROC AUC?
**A:** Area Under the Receiver Operating Characteristic curve. Measures separability between classes across thresholds. Our model achieves ~0.95 — excellent discrimination.

### Q46. What is a confusion matrix?
**A:** A table showing true vs predicted class counts. Diagonal elements are correct predictions; off-diagonal are errors. Main errors occur at Medium-High boundary.

### Q47. Why use weighted metrics?
**A:** Weighted precision/recall/F1 account for class imbalance by computing per-class metrics and averaging weighted by class support.

### Q48. Why is accuracy alone insufficient?
**A:** With 28.2% High class, a naive "always predict High" model gets limited accuracy but poor recall for Low students. F1 and recall for minority class matter more.

---

## Section E: EDA & Dataset (49–52)

### Q49. Describe the dataset source and size.
**A:** UCI Student Performance dataset by Cortez & Silva (2008). 1,044 records combining 395 Math and 649 Portuguese students with 33 features each.

### Q50. What is the strongest predictor of G3?
**A:** G2 (second period grade), with Pearson correlation r≈0.90 with G3. G1 is second. Prior academic performance dominates.

### Q51. How was class imbalance handled?
**A:** Stratified splitting, balanced class weights in Random Forest, F1-based model selection, and reporting per-class recall in classification report.

### Q52. What outliers exist and how were they treated?
**A:** Absences range up to 93 (extreme). IQR winsorization capped absences and failures to reduce undue influence on tree splits while preserving signal.

---

## Section F: Ethics, Limitations & Deployment (53–55)

### Q53. What ethical concerns apply?
**A:** Student privacy (FERPA/GDPR), algorithmic bias against disadvantaged groups, stigmatization from labels, and over-reliance on automated decisions. Mitigated by human-in-the-loop and transparency.

### Q54. What are the main limitations?
**A:** Portuguese schools only, self-reported features, cross-sectional data, ~14% error rate, class imbalance, overlapping students in dual-course records, and inability to capture hidden variables.

### Q55. How would you deploy this in production?
**A:** ETL pipeline from school SIS → SPPA API (loads `best_model.pkl`) → teacher dashboard with role-based access → scheduled retraining each semester → audit logging for privacy compliance.

---

## Bonus Rapid-Fire Questions

| Q | Short Answer |
|---|-------------|
| What is a percept sequence? | History of percepts over time for one student |
| What algorithm is in best_model.pkl? | Random Forest Classifier |
| How many trees in the forest? | 200 |
| What Python library for ML? | scikit-learn |
| How is the model saved? | joblib.dump() |
| What command trains the model? | python main.py --train |
| What is one-hot encoding? | Binary columns per category value |
| What is StandardScaler? | Z-score normalization (mean=0, std=1) |
| What is the G3 scale? | 0 to 20 |
| Low performance threshold? | G3 < 10 |

---

*Study Sections A–B for AI focus; C–D for ML focus; E–F for practical deployment.*
