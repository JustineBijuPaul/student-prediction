# Presentation Guide — AI Final Examination

## Overview

This guide helps you deliver a **5-minute** presentation matching the professor's required 7-slide structure. All slide content is in `ppt/PPT_CONTENT.md`. Speaker script is in `docs/SPEAKER_NOTES.md`.

---

## Before Presentation Day

### 1. Personalize Placeholders
Replace in `ppt/PPT_CONTENT.md`, notebook header, and `PROJECT_DOCUMENTATION.md`:
- [Your Name]
- [Semester/Year]
- [Group Number]
- [University]

### 2. Generate Fresh Artifacts
```bash
python main.py --train
```
Ensures `outputs/*.png` matches your live results.

### 3. Prepare Visual Aids
Copy these images to your slides:
| Slide | Image |
|-------|-------|
| 4 | `outputs/correlation_heatmap.png`, `outputs/class_distribution.png` |
| 5 | `outputs/model_comparison.png` |
| 6 | `outputs/confusion_matrix.png`, `outputs/feature_importance.png` |
| 7 | Architecture diagram from `docs/AI_AGENT_DESIGN.md` |

### 4. Optional Live Demo
```bash
python main.py --predict
```
Shows agent output in terminal (~10 seconds). Only if time permits in Q&A.

---

## Slide-by-Slide Guide

### Slide 1 — Agent (30–45 sec)
**Must include:** Agent name, goal, your name, semester, group  
**Emphasize:** This is an AI Agent project, not just ML  
**Do not:** Spend time on dataset details yet

### Slide 2 — Environment (30–45 sec)
**Must include:** Full Russell & Norvig table  
**Emphasize:** Partially observable + multi-agent — unique to AI course  
**Do not:** Read every cell verbatim — hit the highlights

### Slide 3 — Dataset (30–45 sec)
**Must include:** Source, description, sensors, sample data, problems  
**Emphasize:** Sensors connect data to agent percepts  
**Visual:** One table row from `student-mat.csv`

### Slide 4 — EDA (30–45 sec)
**Must include:** Correlation, statistics, distributions, charts  
**Emphasize:** G1/G2 → G3 relationship and class imbalance  
**Visual:** Heatmap + class distribution

### Slide 5 — Model (30–45 sec)
**Must include:** Architecture, learning type, hyperparameters, comparison  
**Emphasize:** Why Random Forest won  
**Visual:** Model comparison bar chart

### Slide 6 — Results (30–45 sec)
**Must include:** Accuracy, precision, recall, F1, confusion matrix, insights  
**Emphasize:** G2 as top feature — actionable for teachers  
**Visual:** Confusion matrix + feature importance

### Slide 7 — Integration (30–45 sec)
**Must include:** Decision making, real-world use, limitations, future  
**Emphasize:** ML model → decision engine → actions  
**Close:** "Thank you, questions welcome"

---

## Viva Preparation

Study these documents in order:
1. `CHEAT_SHEET.md` — night before
2. `QUESTIONS_AND_ANSWERS.md` — full preparation
3. `ENVIRONMENT_ANALYSIS.md` — likely viva focus
4. `AI_AGENT_DESIGN.md` — architecture questions

---

## Common Examiner Questions

| Question | Quick Answer Location |
|----------|----------------------|
| Why is this AI not ML? | Agent has KB + decision engine + actions |
| Environment type? | Partially obs., stochastic, sequential, dynamic, multi-agent |
| Why Random Forest? | Best F1, handles mixed features, feature importance |
| What is a percept? | Normalized student feature vector from sensors |
| Ethical concerns? | Privacy, bias, human-in-the-loop |

---

## Presentation Tools

- **PowerPoint / Google Slides:** Copy content from `PPT_CONTENT.md`
- **Canva / Gamma:** Import images from `outputs/`
- **Live Jupyter:** Open notebook on Kaggle if internet available

---

## Grading Alignment Checklist

- [ ] Agent name and goal stated clearly (Slide 1)
- [ ] Russell & Norvig table complete (Slide 2)
- [ ] Dataset + sensors explained (Slide 3)
- [ ] EDA with visualizations (Slide 4)
- [ ] Multiple models compared (Slide 5)
- [ ] Metrics + confusion matrix (Slide 6)
- [ ] Agent integration + limitations (Slide 7)
- [ ] Under 5 minutes total
- [ ] Can explain how ML fits inside agent (viva)

---

## Emergency Fallbacks

| Problem | Solution |
|---------|----------|
| Projector fails | Walk through `CHEAT_SHEET.md` verbally |
| Forgot metric | Say "approximately 86% F1" from Random Forest |
| Tough viva question | Refer to `docs/` — offer to show diagram |
| Demo fails | Describe `main.py --predict` output from docs |

Good luck!
