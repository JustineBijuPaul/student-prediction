"""
Utility functions for the Student Performance Prediction Agent.

Provides path resolution, column mapping for multiple dataset formats,
performance category labeling, and project-wide constants.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reproducibility constant used across the project
RANDOM_STATE = 42

# Performance category thresholds (G3 scale: 0–20, UCI Student Performance)
PERFORMANCE_THRESHOLDS = {
    "Low": (0, 10),       # G3 < 10
    "Medium": (10, 14),   # 10 <= G3 < 14
    "High": (14, 21),     # G3 >= 14
}

PERFORMANCE_LABELS = ["Low", "Medium", "High"]

# Maps canonical column names to known aliases across public datasets
COLUMN_ALIASES: Dict[str, List[str]] = {
    "age": ["age", "Age", "student_age"],
    "sex": ["sex", "gender", "Gender"],
    "studytime": ["studytime", "study_time", "StudyTime", "weekly_study_time"],
    "absences": ["absences", "attendance", "Attendance", "school_absences"],
    "failures": ["failures", "failed_subjects", "past_failures", "class_failures"],
    "internet": ["internet", "internet_access", "Internet", "has_internet"],
    "schoolsup": ["schoolsup", "school_support", "SchoolSupport", "extra_school_support"],
    "famsup": ["famsup", "family_support", "FamilySupport", "familysup"],
    "activities": ["activities", "extra_activities", "extracurricular", "ExtraActivities"],
    "health": ["health", "Health", "health_status"],
    "Medu": ["Medu", "mother_education", "medu", "parent_education_mother"],
    "Fedu": ["Fedu", "father_education", "fedu", "parent_education_father"],
    "G1": ["G1", "grade1", "previous_grade_1", "first_period_grade"],
    "G2": ["G2", "grade2", "previous_grade_2", "second_period_grade"],
    "G3": ["G3", "score", "final_grade", "FinalGrade", "target", "performance"],
    "traveltime": ["traveltime", "travel_time", "commute_time"],
    "famrel": ["famrel", "family_relationship", "family_relations"],
    "freetime": ["freetime", "free_time", "leisure_time"],
    "goout": ["goout", "going_out", "social_outings"],
    "higher": ["higher", "wants_higher_education", "higher_education"],
    "paid": ["paid", "paid_classes", "extra_paid_classes"],
    "romantic": ["romantic", "romantic_relationship", "in_relationship"],
    "Dalc": ["Dalc", "workday_alcohol", "weekday_alcohol"],
    "Walc": ["Walc", "weekend_alcohol", "weekend_alcohol_consumption"],
}


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def get_paths() -> Dict[str, Path]:
    """Resolve standard project directories."""
    root = get_project_root()
    return {
        "root": root,
        "dataset": root / "dataset",
        "outputs": root / "outputs",
        "models": root / "models",
        "docs": root / "docs",
        "notebooks": root / "notebooks",
    }


def resolve_dataset_path(filename: Optional[str] = None) -> Path:
    """
    Locate dataset file for local execution or Kaggle environment.

    Search order:
    1. Kaggle input: ../input/**/
    2. Project dataset/ directory
    """
    paths = get_paths()
    candidates: List[Path] = []

    # Kaggle notebook environment
    kaggle_input = Path("../input")
    if kaggle_input.exists():
        for sub in kaggle_input.iterdir():
            if sub.is_dir():
                candidates.extend(sub.glob("*.csv"))
                candidates.extend(sub.glob("**/*.csv"))

    # Local project dataset
    if filename:
        candidates.append(paths["dataset"] / filename)
    else:
        candidates.extend([
            paths["dataset"] / "student-mat.csv",
            paths["dataset"] / "student-por.csv",
            paths["dataset"] / "student.csv",
            paths["dataset"] / "students.csv",
        ])

    for path in candidates:
        if path.exists() and path.suffix.lower() == ".csv":
            return path

    raise FileNotFoundError(
        "No student performance CSV found. Place UCI files in dataset/ "
        "or attach them on Kaggle under ../input/."
    )


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map alternative column names to canonical UCI-style names.

    Performs case-insensitive matching against COLUMN_ALIASES.
    """
    df = df.copy()
    lower_map = {c.lower().strip(): c for c in df.columns}

    rename: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            key = alias.lower()
            if key in lower_map:
                rename[lower_map[key]] = canonical
                break

    if rename:
        df = df.rename(columns=rename)

    # Attendance is sometimes stored as presence rate; convert to absences proxy if needed
    if "absences" not in df.columns and "attendance" in df.columns:
        att = pd.to_numeric(df["attendance"], errors="coerce")
        if att.max() is not None and att.max() <= 1.0:
            df["absences"] = ((1 - att) * 100).round().astype(int)
        elif att.max() is not None and att.max() <= 100:
            df["absences"] = (100 - att).clip(lower=0).astype(int)

    return df


def load_student_dataset(
    path: Optional[Path] = None,
    merge_math_portuguese: bool = True,
) -> pd.DataFrame:
    """
    Load and optionally merge UCI Math and Portuguese student datasets.

    Parameters
    ----------
    path : Path, optional
        Explicit CSV path. If None, auto-detect.
    merge_math_portuguese : bool
        When True and both mat/por files exist locally, concatenate with course label.

    Returns
    -------
    pd.DataFrame
        Normalized student records with canonical column names.
    """
    paths = get_paths()

    if path is not None:
        df = pd.read_csv(path, sep=None, engine="python")
        return normalize_column_names(df)

    mat_path = paths["dataset"] / "student-mat.csv"
    por_path = paths["dataset"] / "student-por.csv"

    frames = []
    if merge_math_portuguese and mat_path.exists() and por_path.exists():
        mat = pd.read_csv(mat_path, sep=";")
        por = pd.read_csv(por_path, sep=";")
        mat["course"] = "Math"
        por["course"] = "Portuguese"
        frames.extend([mat, por])
    else:
        detected = resolve_dataset_path()
        df = pd.read_csv(detected, sep=None, engine="python")
        return normalize_column_names(df)

    combined = pd.concat(frames, ignore_index=True)
    return normalize_column_names(combined)


def grade_to_performance_category(grades: pd.Series) -> pd.Series:
    """Convert numeric final grades (G3) to Low / Medium / High categories."""
    grades = pd.to_numeric(grades, errors="coerce")

    def _label(g: float) -> str:
        if pd.isna(g):
            return np.nan  # type: ignore[return-value]
        for label, (low, high) in PERFORMANCE_THRESHOLDS.items():
            if low <= g < high:
                return label
        return "High"

    return grades.apply(_label)


def ensure_directories() -> None:
    """Create outputs/ and models/ if they do not exist."""
    paths = get_paths()
    paths["outputs"].mkdir(parents=True, exist_ok=True)
    paths["models"].mkdir(parents=True, exist_ok=True)


def get_feature_columns(df: pd.DataFrame, target_col: str = "performance_category") -> List[str]:
    """Return predictor columns, excluding identifiers and leakage-prone grade columns."""
    exclude = {
        target_col,
        "G3",
        "performance",
        "final_grade",
        "score",
        "student_id",
        "id",
    }
    numeric_and_cat = [
        c for c in df.columns
        if c not in exclude and df[c].dtype != "object" or c in df.select_dtypes(include=["object"]).columns
    ]
    # Prefer features without direct target leakage; keep G1/G2 as legitimate predictors
    return [c for c in numeric_and_cat if c not in exclude]


def summary_class_distribution(y: pd.Series) -> pd.DataFrame:
    """Tabulate class counts and percentages."""
    counts = y.value_counts().reindex(PERFORMANCE_LABELS).fillna(0).astype(int)
    pct = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"count": counts, "percent": pct})
