# AI Agent Design — Student Performance Prediction Agent

## 1. Agent Identity

| Attribute | Value |
|-----------|-------|
| **Name** | Student Performance Prediction Agent (SPPA) |
| **Type** | Learning + Knowledge-Based Hybrid Agent |
| **Goal** | Predict academic performance category and recommend timely interventions |
| **Environment** | Secondary school educational ecosystem |
| **Users** | Teachers, counselors, administrators, parents |

---

## 2. PEAS Description

| Component | Description |
|-----------|-------------|
| **Performance** | Maximize early detection of at-risk students; minimize false negatives |
| **Environment** | School information systems, classrooms, homes, examination periods |
| **Actuators** | Dashboard alerts, notification emails, tutoring referrals, parent conferences |
| **Sensors** | SIS records, LMS logs, attendance scanners, survey forms, grade books |

---

## 3. System Architecture

```mermaid
flowchart TB
    subgraph Environment
        SIS[Student Information System]
        LMS[Learning Management System]
        ATT[Attendance System]
        SUR[Survey / Counselor Notes]
    end

    subgraph Sensors
        S1[Grade Sensor]
        S2[Demographic Sensor]
        S3[Behavioral Sensor]
    end

    subgraph Agent["Student Performance Prediction Agent"]
        P[Percept Processor]
        KB[(Knowledge Base)]
        LC[Learning Component<br/>Random Forest]
        DE[Decision Engine]
        MEM[(Memory)]
        ACT[Action Planner]
    end

    subgraph Actuators
        DASH[Teacher Dashboard]
        NOTIF[Notification System]
        REC[Recommendation Engine]
    end

    SIS --> S1
    LMS --> S2
    ATT --> S3
    SUR --> S3
    S1 --> P
    S2 --> P
    S3 --> P
    P --> LC
    P --> DE
    KB --> DE
    LC --> DE
    DE --> MEM
    DE --> ACT
    ACT --> DASH
    ACT --> NOTIF
    ACT --> REC
    DASH -.->|Feedback| MEM
```

---

## 4. AI Agent Architecture (Internal)

```mermaid
flowchart LR
    subgraph Input
        RAW[Raw Student Record]
    end

    subgraph Perception
        NORM[Column Normalizer]
        PRE[Preprocessor<br/>Scale + One-Hot]
    end

    subgraph Cognition
        ML[Classifier<br/>best_model.pkl]
        RISK[Risk Assessor]
        RULES[Intervention Rules]
    end

    subgraph Output
        PRED[Category + Confidence]
        RECS[Recommendations]
        PRIO[Priority Level]
    end

    RAW --> NORM --> PRE --> ML
    ML --> RISK
    RULES --> RISK
    RISK --> PRED
    RISK --> RECS
    RISK --> PRIO
```

---

## 5. Component Specifications

### 5.1 Percepts
Structured observations about a student at time *t*:
- Demographics: age, sex, address, family size
- Academic history: G1, G2, failures, study time
- Support factors: school support, family support, internet access
- Behavioral: absences, alcohol consumption, free time, health

### 5.2 Sensors
| Sensor | Data Source | Percepts Produced |
|--------|-------------|-------------------|
| Academic | Grade book API | G1, G2, failures |
| Attendance | RFID / manual logs | absences |
| Demographic | Enrollment forms | age, sex, parental education |
| Engagement | LMS analytics | study time proxies, activity flags |

### 5.3 Knowledge Base
Static and dynamic knowledge:
- **Performance thresholds**: Low (G3&lt;10), Medium (10–13), High (≥14)
- **Intervention rules**: category-specific action templates
- **Feature semantics**: UCI attribute definitions
- **Policy constraints**: privacy rules, escalation procedures

### 5.4 Learning Component
- **Algorithm**: Random Forest Classifier (200 trees, max_depth=12)
- **Training**: Supervised learning on 1,044 labeled student records
- **Input**: 32+ encoded features after preprocessing
- **Output**: P(Low), P(Medium), P(High)
- **Persistence**: `models/best_model.pkl` via Joblib

### 5.5 Decision Engine
Hybrid reasoning combining ML and rules:

1. **Classification**: `argmax(P(category))` from Random Forest
2. **Risk escalation**: If P(Low) ≥ 0.35, elevate risk regardless of top class
3. **Rule augmentation**: Append feature-specific recommendations (high absences, low study time)
4. **Priority assignment**: HIGH / MEDIUM / LOW intervention queue

```mermaid
flowchart TD
    A[Receive Student Percept] --> B[Preprocess Features]
    B --> C[ML Predict Probabilities]
    C --> D{P Low >= 0.35?}
    D -->|Yes| E[Risk = Critical]
    D -->|No| F{Top Class?}
    F -->|Low| E
    F -->|Medium| G[Risk = Moderate]
    F -->|High| H[Risk = Low]
    E --> I[Apply Intervention Rules]
    G --> I
    H --> I
    I --> J[Generate Recommendations]
    J --> K[Update Memory]
    K --> L[Emit Actions]
```

### 5.6 Memory
- **Short-term**: Current batch of students being scored
- **Long-term**: Per-student prediction history (`agent.memory` dict)
- **Episodic**: Timestamped interventions and outcomes (future extension)

### 5.7 Actions
| Action | Trigger | Recipient |
|--------|---------|-----------|
| Dashboard update | Any prediction | Teacher |
| High-priority alert | Low + Critical risk | Counselor |
| Parent notification | Low performance | Parents (with consent) |
| Tutoring referral | failures ≥ 2 | Academic support office |
| Enrichment offer | High performance | Student |

---

## 6. Prediction Flow

```mermaid
sequenceDiagram
    participant T as Teacher
    participant D as Dashboard
    participant A as SPP Agent
    participant M as ML Model
    participant K as Knowledge Base

    T->>D: Upload / sync student roster
    D->>A: Student percept vector
    A->>A: Normalize & preprocess
    A->>M: predict_proba(X)
    M-->>A: [P(Low), P(Med), P(High)]
    A->>K: Lookup intervention rules
    K-->>A: Rule templates
    A->>A: Decision engine merges ML + rules
    A-->>D: Category, risk, recommendations
    D-->>T: Visual alert + action items
```

---

## 7. Training Pipeline

```mermaid
flowchart LR
    A[Raw CSV] --> B[Merge Mat + Por]
    B --> C[Clean + Cap Outliers]
    C --> D[Create performance_category]
    D --> E[Train/Test Split]
    E --> F[Fit Preprocessor]
    F --> G[Train 7 Models]
    G --> H[Cross-Validate]
    H --> I[Select Best F1]
    I --> J[Save best_model.pkl]
    I --> K[Generate Plots]
```

---

## 8. Teacher Interaction

Teachers interact through:
1. **Batch dashboard** (`python main.py --dashboard`) — scores entire cohort
2. **Single-student query** — API or CLI demo
3. **Feedback loop** — teacher marks false alarms; data stored for retraining (future)

The agent **assists** teachers; it does not autonomously change grades or enforce disciplinary action.

---

## 9. Student Interaction

Students receive indirect benefits:
- Earlier support before failure
- Personalized study recommendations
- Privacy-preserving aggregation (no public ranking)

Direct student-facing features (future): mobile app showing study tips based on agent recommendations.

---

## 10. Feedback Loop

```
Predict → Act (intervene) → Observe outcome (next G1/G2/G3) → Store in Memory → Retrain periodically
```

This closes the **rational agent** cycle: the agent improves as more labeled outcomes become available.

---

## 11. Implementation Mapping

| Design Component | Code Location |
|------------------|---------------|
| Percept normalization | `src/utils.py` → `normalize_column_names()` |
| Preprocessing | `src/preprocess.py` |
| Learning component | `src/train.py` → Random Forest in `best_model.pkl` |
| Decision engine | `src/predict.py` → `StudentPerformanceAgent` |
| Knowledge base | `src/predict.py` → `INTERVENTION_RULES` |
| Memory | `src/predict.py` → `self.memory` |
| Actions / dashboard | `main.py --dashboard`, `outputs/teacher_dashboard.csv` |

---

## 12. Why This Is a True AI Agent

A standalone `model.predict()` call is **machine learning**. Our system exhibits **agency**:

1. **Autonomy**: Processes cohorts without per-student manual configuration
2. **Reactivity**: Responds to new student percepts in milliseconds
3. **Pro-activeness**: Recommends interventions before exams
4. **Social ability** (future): Multi-agent coordination with counselors and parents

The ML model provides **beliefs** about student state; the decision engine converts beliefs into **actions** — the defining characteristic of an intelligent agent per Russell & Norvig.
