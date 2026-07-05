#!/usr/bin/env python3
"""
Build the AI Final Examination presentation from the course template.

Usage:
    python scripts/build_presentation.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "AI Final Examination Presentation Template.pptx"
OUTPUT = ROOT / "ppt" / "Student_Performance_Prediction_Agent.pptx"
OUTPUTS = ROOT / "outputs"

# --- Customize before presenting ---
STUDENT_NAME = "[Your Name]"
SEMESTER = "[Spring 2026]"
GROUP = "[Group Number]"


def set_paragraphs(text_frame, lines: list[str], font_size: int = 14) -> None:
    """Replace all paragraphs in a text frame with new lines."""
    tf = text_frame
    # Clear existing paragraphs except first
    while len(tf.paragraphs) > 1:
        p = tf.paragraphs[-1]._element
        p.getparent().remove(p)

    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.level = 0
        for run in p.runs:
            run.font.size = Pt(font_size)
            run.font.name = "Calibri"


def set_table_cell(table, row: int, col: int, text: str, bold: bool = False) -> None:
    cell = table.cell(row, col)
    cell.text = text
    for paragraph in cell.text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.size = Pt(11)
            run.font.name = "Calibri"
            run.font.bold = bold


def replace_pictures(slide, image_paths: list[Path]) -> None:
    """Remove existing pictures on slide and insert project images at similar positions."""
    pic_shapes = [s for s in slide.shapes if s.shape_type == MSO_SHAPE_TYPE.PICTURE]
    positions = [(s.left, s.top, s.width, s.height) for s in pic_shapes]

    for shape in pic_shapes:
        el = shape._element
        el.getparent().remove(el)

    # Default layout if template pictures were removed
    if not positions:
        positions = [
            (Inches(0.5), Inches(2.0), Inches(4.5), Inches(3.0)),
            (Inches(5.2), Inches(2.0), Inches(4.3), Inches(3.0)),
        ]

    for path, (left, top, width, height) in zip(image_paths, positions):
        if path.exists():
            slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def add_picture_if_room(slide, path: Path, left, top, width, height) -> None:
    if path.exists():
        slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def create_architecture_diagram(path: Path) -> None:
    """Create a simple agent architecture diagram for slide 7."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 1.5, "Sensors\n(SIS, Attendance)"),
        (2.0, 1.5, "Percepts\n(Student Record)"),
        (3.7, 1.5, "Preprocessor"),
        (5.4, 1.5, "Random Forest\n(Learning)"),
        (7.1, 1.5, "Decision\nEngine"),
        (8.8, 1.5, "Teacher\nDashboard"),
    ]
    colors = ["#3498db", "#9b59b6", "#1abc9c", "#e67e22", "#e74c3c", "#27ae60"]
    for (x, y, label), color in zip(boxes, colors):
        box = FancyBboxPatch(
            (x, y), 1.4, 1.0,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            linewidth=1.5, edgecolor="#2c3e50", facecolor=color, alpha=0.85,
        )
        ax.add_patch(box)
        ax.text(x + 0.7, y + 0.5, label, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    for x in [1.7, 3.4, 5.1, 6.8, 8.5]:
        ax.annotate("", xy=(x + 0.3, 2.0), xytext=(x, 2.0),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2))

    ax.text(5, 3.3, "Student Performance Prediction Agent Architecture",
            ha="center", fontsize=14, fontweight="bold", color="#2c3e50")
    ax.text(7.1, 0.6, "Knowledge Base + Memory", ha="center", fontsize=9, color="#7f8c8d")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build() -> Path:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    prs = Presentation(str(TEMPLATE))

    # --- Slide 1: Agent ---
    s1 = prs.slides[0]
    shapes = [sh for sh in s1.shapes if sh.has_text_frame]
    if len(shapes) >= 2:
        shapes[0].text_frame.text = "Student Performance Prediction Agent"
        for run in shapes[0].text_frame.paragraphs[0].runs:
            run.font.size = Pt(28)
            run.font.bold = True

        set_paragraphs(
            shapes[1].text_frame,
            [
                "Agent name:",
                '"Student Performance Prediction Agent (SPPA)"',
                "",
                "Goal:",
                "Predict student performance (Low / Medium / High) and recommend "
                "early interventions before final examinations.",
                "",
                f"{STUDENT_NAME}  |  {SEMESTER}  |  Group {GROUP}",
                "Artificial Intelligence — Final Examination",
            ],
            font_size=13,
        )

    # --- Slide 2: Environment ---
    s2 = prs.slides[1]
    for shape in s2.shapes:
        if shape.has_text_frame:
            txt = shape.text_frame.text
            if "environment/context" in txt or "Russell" in txt:
                if "environment/context" in txt:
                    shape.text_frame.text = "Russell & Norvig Environment Analysis"
                    for run in shape.text_frame.paragraphs[0].runs:
                        run.font.size = Pt(24)
                        run.font.bold = True
                elif "Russell" in txt:
                    set_paragraphs(
                        shape.text_frame,
                        [
                            "Educational ecosystem: schools, classrooms, homes, examination periods",
                            "Agent perceives partial student records and acts via teacher dashboard",
                        ],
                        font_size=12,
                    )
        if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
            tbl = shape.table
            rows = [
                ("Observable", "Partially Observable ✓", "Fully Observable",
                 "Hidden motivation, home stress, undiagnosed learning needs"),
                ("Deterministic vs Stochastic", "Stochastic ✓", "Deterministic",
                 "Random Forest outputs probabilities; outcomes are uncertain"),
                ("Episodic vs Sequential", "Sequential ✓", "Episodic",
                 "G1/G2 history affects G3; interventions shape future state"),
                ("Static vs Dynamic", "Dynamic ✓", "Static",
                 "Grades and absences change throughout the semester"),
                ("Discrete vs Continuous", "Discrete ✓", "Continuous",
                 "Decisions: Low / Medium / High performance categories"),
                ("Single vs Multi-agent", "Multi-agent ✓", "Single",
                 "Teachers, students, parents, counselors interact"),
            ]
            for i, (prop, chosen, alt, reason) in enumerate(rows, start=1):
                set_table_cell(tbl, i, 0, prop, bold=True)
                set_table_cell(tbl, i, 1, chosen)
                set_table_cell(tbl, i, 2, alt)
                set_table_cell(tbl, i, 3, reason)

    # --- Slide 3: Dataset ---
    s3 = prs.slides[2]
    for shape in s3.shapes:
        if shape.has_text_frame and "One slide" in shape.text_frame.text:
            shape.text_frame.text = "Dataset & Sensors"
            for run in shape.text_frame.paragraphs[0].runs:
                run.font.size = Pt(24)
                run.font.bold = True
        if shape.has_text_frame and "Where did it come from" in shape.text_frame.text:
            set_paragraphs(
                shape.text_frame,
                [
                    "Source: UCI ML Repository — Student Performance (Cortez & Silva, 2008)",
                    "Description: 1,044 records (395 Math + 649 Portuguese), 33 features",
                    "Sensors: SIS grades, attendance logs, enrollment forms, study surveys",
                    "Target: G3 final grade (0–20) → Low / Medium / High",
                    "Issues: 22% Low class imbalance, absences outliers (max 93), self-reported fields",
                ],
                font_size=12,
            )
        if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
            tbl = shape.table
            set_table_cell(tbl, 0, 0, "Sample fields", bold=True)
            set_table_cell(tbl, 0, 1, "age, sex, studytime", bold=True)
            set_table_cell(tbl, 0, 2, "failures, absences, G1, G2", bold=True)
            set_table_cell(tbl, 0, 3, "G3 / Category", bold=True)
            set_table_cell(tbl, 1, 0, "Sample rows")
            set_table_cell(tbl, 1, 1, "15,F,2 | 18,M,1 | 17,F,4")
            set_table_cell(tbl, 1, 2, "failures, absences, G1, G2")
            set_table_cell(tbl, 1, 3, "10/Med · 6/Low · 15/High")

    # --- Slide 4: EDA ---
    s4 = prs.slides[3]
    for shape in s4.shapes:
        if shape.has_text_frame and "inspection" in shape.text_frame.text.lower():
            shape.text_frame.text = "Exploratory Data Analysis"
            for run in shape.text_frame.paragraphs[0].runs:
                run.font.size = Pt(24)
                run.font.bold = True
        if shape.has_text_frame and "covariance" in shape.text_frame.text.lower():
            set_paragraphs(
                shape.text_frame,
                [
                    "G2 ↔ G3 correlation: r ≈ 0.90 (strongest predictor)",
                    "G1 ↔ G3: r ≈ 0.80  |  failures ↔ G3: r ≈ -0.35",
                    "Mean G3 = 11.9 (σ = 3.2)  |  0 missing values",
                    "Categories: Low 22% · Medium 50% · High 28%",
                ],
                font_size=12,
            )
    replace_pictures(
        s4,
        [
            OUTPUTS / "correlation_heatmap.png",
            OUTPUTS / "class_distribution.png",
        ],
    )

    # --- Slide 5: Model ---
    s5 = prs.slides[4]
    for shape in s5.shapes:
        if shape.has_text_frame and "model/algorithm" in shape.text_frame.text.lower():
            shape.text_frame.text = "Machine Learning — Learning Component"
            for run in shape.text_frame.paragraphs[0].runs:
                run.font.size = Pt(24)
                run.font.bold = True
        if shape.has_text_frame and "supervised" in shape.text_frame.text.lower():
            set_paragraphs(
                shape.text_frame,
                [
                    "Type: Supervised Multi-Class Classification",
                    "Pipeline: Features → StandardScaler + OneHotEncoder → Random Forest",
                    "Best model: Random Forest (200 trees, depth 12, balanced weights)",
                    "Compared 7 models — RF wins: F1 = 86.15%, Accuracy = 86.12%",
                    "Runners-up: Logistic Regression (85.16%), XGBoost (84.68%)",
                ],
                font_size=12,
            )
    add_picture_if_room(
        s5,
        OUTPUTS / "model_comparison.png",
        Inches(5.0),
        Inches(1.8),
        Inches(4.5),
        Inches(3.2),
    )

    # --- Slide 6: Results ---
    s6 = prs.slides[5]
    for shape in s6.shapes:
        if shape.has_text_frame and "Results" in shape.text_frame.text:
            shape.text_frame.text = "Evaluation Results — Random Forest"
            for run in shape.text_frame.paragraphs[0].runs:
                run.font.size = Pt(24)
                run.font.bold = True
        if shape.has_text_frame and "Accuracy" in shape.text_frame.text:
            set_paragraphs(
                shape.text_frame,
                [
                    "Accuracy: 86.12%  |  Precision: 86.36%  |  Recall: 86.12%",
                    "F1 Score: 86.15%  |  ROC AUC: 94.61%",
                    "CV Accuracy: 84.67% ± 1.95%  |  Train/Test: 835 / 209",
                    "Top predictor: G2 (2nd period grade)",
                    "Training: 0.43s  |  Prediction: 34ms per batch",
                ],
                font_size=12,
            )
    add_picture_if_room(
        s6,
        OUTPUTS / "confusion_matrix.png",
        Inches(0.4),
        Inches(2.0),
        Inches(3.8),
        Inches(2.8),
    )
    add_picture_if_room(
        s6,
        OUTPUTS / "feature_importance.png",
        Inches(4.5),
        Inches(2.0),
        Inches(4.8),
        Inches(2.8),
    )
    add_picture_if_room(
        s6,
        OUTPUTS / "roc_curve.png",
        Inches(9.5),
        Inches(2.0),
        Inches(3.5),
        Inches(2.8),
    )

    # --- Slide 7: Agent Integration ---
    arch_path = OUTPUTS / "agent_architecture.png"
    create_architecture_diagram(arch_path)

    s7 = prs.slides[6]
    for shape in s7.shapes:
        if shape.has_text_frame and "Final slide" in shape.text_frame.text:
            shape.text_frame.text = "Agent Integration — From Prediction to Action"
            for run in shape.text_frame.paragraphs[0].runs:
                run.font.size = Pt(22)
                run.font.bold = True
        if shape.has_text_frame and "How could" in shape.text_frame.text:
            set_paragraphs(
                shape.text_frame,
                [
                    "ML model = Learning Component inside the intelligent agent",
                    "Flow: Percepts → Preprocess → RF probabilities → Risk engine → Actions",
                    "Dashboard scores 1,044 students: 233 HIGH · 525 MEDIUM · 286 LOW priority",
                    "Actions: counseling, tutoring, parent alerts, teacher dashboard CSV",
                    "Components: Sensors, Knowledge Base, Decision Engine, Memory, Actuators",
                    "Limitations: Portuguese data only, ~14% error, privacy & bias concerns",
                    "Future: SHAP explainability, real-time analytics, LLM parent reports",
                    "",
                    "Thank you — Questions welcome.",
                ],
                font_size=12,
            )
    add_picture_if_room(
        s7,
        arch_path,
        Inches(0.35),
        Inches(3.5),
        Inches(9.0),
        Inches(1.6),
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT))
    return OUTPUT


if __name__ == "__main__":
    out = build()
    print(f"Presentation saved to: {out}")
