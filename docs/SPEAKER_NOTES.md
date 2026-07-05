# Speaker Notes — 5-Minute Presentation

**Total time: 5 minutes (~42 seconds per slide)**

Replace [Your Name], [Group], and [University] before presenting.

---

## Slide 1: Agent Introduction (~40 sec)

> "Good morning/afternoon. I am **[Your Name]** from **[Group]** presenting our AI Final Examination project: the **Student Performance Prediction Agent**.
>
> The agent's **goal** is to help teachers identify students at risk of poor academic performance **before** final examinations — enabling early intervention such as tutoring or counseling.
>
> This is explicitly an **AI Agent project**, not just machine learning. The ML model is one component inside a larger architecture that perceives student data, reasons about risk, and recommends actions.
>
> I am in **[Semester]** at **[University]**. Let's begin."

---

## Slide 2: Russell & Norvig Environment (~45 sec)

> "We characterize our agent's environment using Russell and Norvig's taxonomy.
>
> It is **partially observable** because we cannot see hidden factors like motivation or home stress — only school records.
>
> It is **stochastic** — the same student profile does not guarantee a fixed outcome; our Random Forest outputs probabilities.
>
> It is **sequential** because past grades G1 and G2 influence the final prediction.
>
> It is **dynamic** — absences and grades change throughout the semester.
>
> Decisions are **discrete** — Low, Medium, or High performance categories.
>
> And it is **multi-agent** — teachers, students, parents, and counselors all interact with the system's outputs."

---

## Slide 3: Dataset (~40 sec)

> "Our data comes from the **UCI Student Performance dataset** by Cortez and Silva, collected from Portuguese secondary schools.
>
> We merged Mathematics — 395 students — and Portuguese — 649 students — for **1,044 records** and **33 features** including age, study time, absences, prior failures, and family support.
>
> **Sensors** in a deployed system would be the Student Information System, attendance logs, and grade books — feeding percepts to the agent.
>
> Here is a sample row showing features like studytime equals 2 and G3 equals the final grade.
>
> **Data problems** we handled: no missing values in UCI files, class imbalance in the Low category, and outliers in absences up to 93 days."

---

## Slide 4: EDA (~45 sec)

> "Exploratory analysis revealed three key patterns.
>
> First, **correlation**: G1 and G2 grades correlate strongly with the final grade G3 — above 0.8 Pearson coefficient. Failures correlate negatively.
>
> Second, **distributions**: Most students score between 10 and 14 on a 0–20 scale. Absences are right-skewed.
>
> Third, **class imbalance**: only about 15 percent are Low performers — so we used stratified splitting and balanced class weights.
>
> The heatmap and pairplot on screen show these relationships visually. Study time and absences clearly separate performance clusters."

---

## Slide 5: Model (~45 sec)

> "We trained **seven supervised classifiers**: Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting, Extra Trees, and XGBoost.
>
> This is **supervised multi-class classification** — we learn from labeled historical grades mapped to Low, Medium, and High categories.
>
> **Random Forest** won with hyperparameters: 200 trees, max depth 12, balanced class weights, random state 42.
>
> The bar chart compares all models. Random Forest achieved the highest F1 score at approximately **86 percent**, with strong cross-validation stability.
>
> We chose Random Forest over Logistic Regression because it captures non-linear interactions — like failures combined with absences — without manual feature engineering."

---

## Slide 6: Results (~45 sec)

> "On the held-out test set, Random Forest achieved **86 percent accuracy**, **86 percent precision**, **86 percent recall**, and **86 percent F1 score**. ROC AUC is approximately **95 percent**.
>
> The confusion matrix shows a strong diagonal — most predictions are correct. Main errors occur at the Medium–High boundary near grade 14.
>
> **Key insight**: G2 — the second period grade — is the most important feature. Teachers should watch students whose G1-to-G2 trajectory is declining, even if the current grade looks acceptable.
>
> Feature importance confirms academic history dominates over demographics."

---

## Slide 7: Agent Integration (~50 sec)

> "Here is how the ML model becomes part of the **intelligent agent**.
>
> **Decision making**: The Random Forest outputs class probabilities. Our decision engine escalates risk if the probability of Low performance exceeds 35 percent, then applies intervention rules — counseling, tutoring, parent contact.
>
> **Real-world use**: Teachers run a dashboard that batch-scores their class and prioritizes HIGH intervention students.
>
> **Limitations**: The model may reflect historical bias, struggles with class imbalance, and was trained on Portuguese schools only — limiting generalization.
>
> **Future improvements** include SHAP explainability, real-time absence streaming, and LLM-generated parent reports.
>
> Thank you. I welcome your questions."

---

## Timing Checklist

| Slide | Target | Content |
|-------|--------|---------|
| 1 | 0:00–0:40 | Intro + goal |
| 2 | 0:40–1:25 | Environment table |
| 3 | 1:25–2:05 | Dataset |
| 4 | 2:05–2:50 | EDA |
| 5 | 2:50–3:35 | Models |
| 6 | 3:35–4:20 | Results |
| 7 | 4:20–5:00 | Agent + close |

**Tip:** Practice once with a timer. If running long, shorten Slide 4; if short, expand Slide 7 with a demo mention.
