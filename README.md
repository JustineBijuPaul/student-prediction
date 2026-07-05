# Student Performance Prediction Agent

An **intelligent AI Agent** for early identification of students at academic risk, built for the Artificial Intelligence Final Examination. The trained machine learning model is one component within a full agent architecture featuring perception, reasoning, decision-making, and action.

## Highlights

- **AI Agent focus** — not just ML: percepts, knowledge base, decision engine, memory, actions
- **Russell & Norvig environment analysis** — fully documented
- **UCI Student Performance dataset** — Math + Portuguese courses (1,044 records)
- **7 classifiers compared** — Random Forest selected as learning component
- **Kaggle-ready notebook** — runs from `../input/` without modification
- **Production pipeline** — `src/` modules, CLI, visualizations, model persistence

## Project Structure

```
student-performance-agent/
├── README.md
├── requirements.txt
├── main.py
├── dataset/          # UCI CSV files
├── notebooks/        # Kaggle notebook
├── src/              # preprocess, train, predict, evaluation, visualization
├── outputs/          # Generated plots and dashboards
├── models/           # best_model.pkl
├── docs/             # Full examination documentation
└── ppt/              # Presentation content
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train models, generate plots, save best_model.pkl
python main.py --train

# Demo agent inference with recommendations
python main.py --predict

# Generate teacher dashboard CSV
python main.py --dashboard
```

## Performance Categories

| Category | G3 Range | Meaning |
|----------|----------|---------|
| Low      | 0–9      | At-risk — immediate intervention |
| Medium   | 10–13    | Monitor — structured support |
| High     | 14–20    | On track — enrichment opportunities |

## Best Model Results (Test Set)

| Metric    | Random Forest |
|-----------|---------------|
| Accuracy  | 86.12%        |
| Precision | 86.36%        |
| Recall    | 86.12%        |
| F1 Score  | 86.15%        |
| ROC AUC   | 94.61%        |
| CV Accuracy | 84.67% ± 1.95% |

*Training set: 835 students | Test set: 209 students | See `outputs/model_comparison.csv`.*

## Teacher Dashboard (Full Cohort)

| Output | Count |
|--------|-------|
| **Total students scored** | **1,044** |
| HIGH priority (urgent) | 233 |
| MEDIUM priority (monitor) | 525 |
| LOW priority (on track) | 286 |

## AI Agent Architecture

```
Sensors (SIS/LMS) → Percepts → Preprocessor → ML Classifier
                                    ↓
              Knowledge Base ← Decision Engine → Actions
                     ↑              ↓
                  Memory    Teacher Dashboard / Notifications
```

See [docs/AI_AGENT_DESIGN.md](docs/AI_AGENT_DESIGN.md) for Mermaid diagrams and full design.

## Presentation

| File | Description |
|------|-------------|
| `ppt/Student_Performance_Prediction_Agent.pptx` | Ready-to-present PowerPoint (7 slides) |
| `ppt/PPT_CONTENT.md` | Slide text reference |
| `scripts/build_presentation.py` | Regenerate PPT from template |

```bash
# Regenerate presentation after updating outputs or metrics
python scripts/build_presentation.py
```

Edit `STUDENT_NAME`, `SEMESTER`, and `GROUP` at the top of `scripts/build_presentation.py` before presenting.

| Document | Description |
|----------|-------------|
| [PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) | Complete 15–20 page project report |
| [AI_AGENT_DESIGN.md](docs/AI_AGENT_DESIGN.md) | Agent architecture and decision flow |
| [ENVIRONMENT_ANALYSIS.md](docs/ENVIRONMENT_ANALYSIS.md) | Russell & Norvig analysis |
| [QUESTIONS_AND_ANSWERS.md](docs/QUESTIONS_AND_ANSWERS.md) | 50+ viva Q&A |
| [SPEAKER_NOTES.md](docs/SPEAKER_NOTES.md) | 5-minute presentation script |
| [CHEAT_SHEET.md](docs/CHEAT_SHEET.md) | One-page revision sheet |

## Dataset

**Source:** [UCI ML Repository — Student Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

Cortez, P. and Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance.

## License

MIT — see [LICENSE](LICENSE).

## Author

**[Your Name]** — AI Final Examination, [Semester/Year], [University]

Replace placeholder name/group details in `ppt/PPT_CONTENT.md` and notebook header before submission.
