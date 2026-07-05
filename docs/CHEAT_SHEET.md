# Cheat Sheet — Student Performance Prediction Agent
*One-page revision for AI Final Examination viva*

---

## Core Definitions

| Term | Definition |
|------|------------|
| **AI Agent** | Entity that perceives environment via sensors and acts via actuators to maximize performance measure |
| **Percept** | Agent's perceptual input at any instant (one student record vector) |
| **Sensor** | Component gathering data (SIS, LMS, attendance) |
| **Actuator** | Component executing actions (dashboard, notifications) |
| **Knowledge Base** | Stored facts + rules (thresholds, interventions) |
| **Learning Component** | Random Forest trained on historical labeled data |
| **Supervised Learning** | Learn f(X)→Y from labeled examples |

---

## Russell & Norvig Table

| Dimension | Our Agent |
|-----------|-----------|
| Observable | **Partially** — hidden motivation/home factors |
| Deterministic | **Stochastic** — probabilistic outcomes |
| Sequential | **Sequential** — G1,G2 history matters |
| Static | **Dynamic** — grades/absences change daily |
| Discrete | **Discrete** — Low/Med/High categories |
| Agents | **Multi** — teachers, students, parents |

---

## Agent Architecture (Memory Aid)

```
Sensors → Percepts → Preprocess → ML Model → Decision Engine → Actions
                         ↑              ↑
                    Knowledge Base    Memory
```

---

## Dataset Quick Facts

- **Source:** UCI Student Performance (Cortez & Silva, 2008)
- **Size:** 1,044 students (395 Math + 649 Portuguese)
- **Features:** 33 (demographic, social, academic)
- **Target:** G3 (0–20) → Low (<10), Medium (10–13), High (≥14)

---

## Key Metrics (Random Forest)

| Metric | Value |
|--------|-------|
| Accuracy | 86.12% |
| Precision | 86.36% |
| Recall | 86.12% |
| F1 | 86.15% |
| ROC AUC | 94.61% |

**F1** = 2×(Precision×Recall)/(Precision+Recall) — used for model selection due to class imbalance.

---

## Model Comparison Winner

**Random Forest** (200 trees, depth 12, balanced weights) beats Logistic Regression (non-linear interactions), SVM (slower), Extra Trees (underfit).

**Top features:** G2 > G1 > failures > absences > studytime

---

## Why AI ≠ Just ML

| ML Alone | Full Agent |
|----------|------------|
| `model.predict()` | Perceive + reason + act |
| Single output | Recommendations + priority |
| No memory | Tracks student history |
| No rules | Knowledge base interventions |

---

## Common Viva One-Liners

- **Why supervised?** We have labeled G3 grades from past students.
- **Why partially observable?** Cannot measure motivation, sleep, home stress.
- **Why stochastic?** Same features ≠ guaranteed outcome; we output probabilities.
- **Ethics?** FERPA/GDPR, human-in-the-loop, avoid stigmatizing labels.
- **Limitations?** Portuguese schools only, ~14% error, class imbalance.
- **Future work?** SHAP, real-time streaming, LLM reports, retraining pipeline.

---

## Commands

```bash
python main.py --train      # Train + save model
python main.py --predict    # Demo agent
python main.py --dashboard  # Batch teacher CSV
```

**Files:** `models/best_model.pkl` | `outputs/*.png` | `src/predict.py` = agent

---

## Performance Thresholds

| Category | G3 Range | Action |
|----------|----------|--------|
| Low | 0–9 | HIGH priority (~233 students in dashboard) |
| Medium | 10–13 | Monitor (~525 students) |
| High | 14–20 | Enrichment (~286 students) |

## Dashboard Summary (1,044 students)

| Output | Count |
|--------|-------|
| HIGH priority | 233 |
| MEDIUM priority | 525 |
| LOW priority | 286 |
| Critical risk | 270 |

---

*Print this page for exam day.*
