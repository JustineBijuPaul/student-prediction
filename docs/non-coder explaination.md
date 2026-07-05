# How the Software Works — Explained for Non-Programmers

Think of this project as a **smart school assistant** for teachers. It does not replace the teacher. It reads student information, estimates who may struggle before final exams, and suggests what to do next.

---

## The Big Picture (One Sentence)

> The system looks at each student’s background and grades, predicts whether they are likely to be **Low**, **Medium**, or **High** performers, and gives the teacher a **priority list** with **recommended actions**.

---

## Real Numbers From Your Project

Your software currently works with **1,044 students** from a real public dataset (Portuguese secondary schools):

| Group | Number of students |
|-------|-------------------|
| **Total students** | **1,044** |
| Math course | 395 |
| Portuguese course | 649 |

When the teacher runs the dashboard, the system scores **all 1,044 students** and produces a report.

### What the system predicts for those 1,044 students

| Predicted performance | Students | What it means |
|----------------------|----------|---------------|
| **Low** (at risk) | **225** | May need urgent help |
| **Medium** (watch list) | **538** | Should be monitored |
| **High** (doing well) | **281** | On track |

### Intervention priority (what the teacher should focus on first)

| Priority | Students | Teacher action |
|----------|----------|----------------|
| **HIGH** | **225** | Act soon — counseling, tutoring, parent contact |
| **MEDIUM** | **538** | Monitor weekly, study plans |
| **LOW** | **281** | Maintain support, offer enrichment |

So if a teacher asks *“How many students do I have?”* — in this demo: **1,044 total**, with **225 flagged as high priority** for intervention.

---

## The Story: What Happens Step by Step

### Step 1 — The school collects information (Sensors)

In a real school, data comes from:

- **Grade books** → marks from term 1 (G1) and term 2 (G2)
- **Attendance system** → number of absences
- **Enrollment forms** → age, gender, family background
- **Surveys** → study time, internet at home, family support

In your project, this information is stored in CSV files (`student-mat.csv` and `student-por.csv`).

**Analogy:** These are the school’s “eyes and ears.”

---

### Step 2 — The agent “reads” each student (Percepts)

For every student, the system builds a profile, for example:

- Age: 17  
- Study time per week: low  
- Past failures: 2  
- Absences: 20  
- Grades so far: G1=8, G2=7  
- Family support: no  
- Internet at home: no  

**Analogy:** Each student file is one “case” the assistant opens.

---

### Step 3 — The AI learns from the past (Machine Learning part)

The system was trained on **historical students whose final grades are already known**.

It learned patterns such as:

- Students with **low G1/G2** often get low final grades  
- Students with **many failures** are more at risk  
- **High absences** often link to poor performance  

The “brain” that learned these patterns is called **Random Forest** — it is about **86% accurate** on test data.

**Important for your professor:**  
This ML model is only **one part** of the AI agent — the **learning component**. It answers: *“What category is this student likely in?”*

---

### Step 4 — The agent decides what to do (Decision Engine)

The system does **not** stop at a label. It also decides:

1. **How confident** is the prediction? (e.g. 99% sure = Low)  
2. **Risk level** — Critical / Moderate / Low  
3. **Priority** — HIGH / MEDIUM / LOW  
4. **Recommendations** — concrete actions for the teacher  

**Example for one at-risk student:**

| Output | Value |
|--------|-------|
| Predicted category | Low |
| Confidence | ~97% |
| Risk | Critical |
| Priority | HIGH |
| Top recommendation | Schedule one-on-one counseling within one week |

Other suggestions may include: notify parents, enroll in tutoring, address absences, increase study time.

**Analogy:** The ML model is the **diagnosis**; the agent is the **doctor who prescribes treatment**.

---

### Step 5 — The teacher sees a dashboard (Action)

When you run:

```bash
python main.py --dashboard
```

The system creates a file: **`outputs/teacher_dashboard.csv`**

Each row = one student, with columns like:

| Column | Meaning for teacher |
|--------|---------------------|
| student_index | Student ID in the list |
| predicted_category | Low / Medium / High |
| confidence | How sure the system is (0–100%) |
| risk_level | Critical / Moderate / Low |
| intervention_priority | HIGH / MEDIUM / LOW |
| top_recommendation | First suggested action |

The teacher can:

- Sort by **HIGH priority** → see the **225** students needing urgent help  
- Focus on **Critical risk** students first  
- Read specific recommendations per student  

**Analogy:** Like a hospital triage board — who needs attention first.

---

## The Three Performance Categories

Final exam grades (0–20) are grouped into:

| Category | Grade range | Plain meaning |
|----------|-------------|---------------|
| **Low** | Below 10 | At risk of failing |
| **Medium** | 10 to 13 | Passing but needs support |
| **High** | 14 and above | Strong performance |

---

## How the Teacher Would Use This in Real Life

1. **Beginning of term / mid-term** — Upload or sync class roster  
2. **System scores everyone** — A few seconds for 1,000+ students  
3. **Teacher reviews HIGH priority list** — e.g. 225 students in the demo  
4. **Teacher takes action** — counseling, tutoring, parent meetings  
5. **Teacher uses judgment** — the system **advises**; the teacher **decides**  

---

## What Each Main File Does (Simple Terms)

| Part | What it is | Non-programmer explanation |
|------|------------|----------------------------|
| `dataset/` | Student records | The school’s data files |
| `notebooks/...ipynb` | Analysis notebook | Full walkthrough with charts (for Kaggle/demo) |
| `src/train.py` | Training | Teaches the computer from past students |
| `src/predict.py` | The AI Agent | Reads a student → predicts → recommends |
| `models/best_model.pkl` | Saved brain | The trained model (memory of patterns) |
| `outputs/teacher_dashboard.csv` | Teacher report | Who needs help and what to do |
| `outputs/*.png` | Charts | Graphs for presentation (correlation, confusion matrix, etc.) |
| `main.py` | Control panel | Run train / predict / dashboard with simple commands |
| `docs/` | Documentation | Written explanations for exam and viva |

---

## Simple Diagram You Can Draw on a Board

```
SCHOOL DATA                    AI AGENT                         TEACHER
───────────                    ────────                         ───────

Grades      ──┐
Attendance  ──┼──►  Read student  ──►  Predict category  ──►  Dashboard
Family info ──┤         │                    │
Survey data ──┘         │                    ▼
                        │              Recommend actions
                        │              (counseling, tutoring…)
                        ▼
                   Learned from
                   1,044 past students
```

---

## How to Explain It to Your Professor (30-Second Version)

> “We built an intelligent **Student Performance Prediction Agent**. It uses data from **1,044 students** — age, study habits, absences, and prior grades — to predict whether each student will be a **Low**, **Medium**, or **High** performer. The machine learning model is the **learning component**; the full agent also includes a **decision engine** that assigns **risk levels** and **intervention priorities**. In our demo, **225 students** are flagged as **HIGH priority** for early intervention. The teacher receives a **dashboard** with recommendations such as counseling or tutoring — supporting decisions, not replacing the teacher.”

---

## Common Questions a Non-Programmer Might Ask

**Q: Does the teacher have 1,044 students in one class?**  
No. In the dataset, 1,044 is the **total across two subjects** (Math + Portuguese) and multiple school records. In a real school, a teacher might run the system for **their own class** (e.g. 30–40 students) — the software works the same way.

**Q: Is the prediction always correct?**  
No. It is about **86% accurate**. The teacher must always use professional judgment.

**Q: Can it read minds or home problems?**  
No. It only uses **recorded data**. Motivation and family stress are **hidden factors** — that is why we call the environment **partially observable** (Russell & Norvig).

**Q: What makes this “AI” and not just “ML”?**  
ML only classifies. The **agent** also **decides risk**, **applies rules**, **recommends actions**, and **outputs a teacher dashboard** — perception → reasoning → action.

**Q: What should the teacher do first?**  
Open the dashboard, filter **intervention_priority = HIGH**, start with **Critical risk** students.

---

## Demo Commands (If Someone Asks You to Show It)

```bash
# Score all 1,044 students → teacher dashboard
python main.py --dashboard

# Show one example student with full recommendations
python main.py --predict
```

---

If you want, I can also write a **1-page “Teacher User Guide”** or a **60-second viva script** focused only on “how many students” and “what the teacher sees” — tailored to your actual class size if you tell me how many students your professor expects in the example.