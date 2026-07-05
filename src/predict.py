"""
Inference module for the Student Performance Prediction Agent.

Loads the trained model and provides predictions with recommended interventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.utils import PERFORMANCE_LABELS, get_paths, normalize_column_names


@dataclass
class AgentPrediction:
    """Structured output from the Student Performance Prediction Agent."""

    performance_category: str
    confidence: float
    probabilities: Dict[str, float]
    risk_level: str
    recommendations: List[str]
    intervention_priority: str


# Rule-based intervention knowledge base (Decision Engine supplement)
INTERVENTION_RULES: Dict[str, List[str]] = {
    "Low": [
        "Schedule one-on-one academic counseling within one week.",
        "Notify homeroom teacher and parents about elevated failure risk.",
        "Enroll student in peer tutoring for core subjects.",
        "Review attendance and study-time habits; set weekly check-in goals.",
        "Consider reduced extracurricular load until grades improve.",
    ],
    "Medium": [
        "Monitor weekly quiz scores and attendance trends.",
        "Recommend structured study plan (minimum 2 hours daily).",
        "Offer optional revision sessions before examinations.",
        "Encourage participation in subject-specific workshops.",
    ],
    "High": [
        "Maintain current support level; recognize positive performance.",
        "Offer advanced enrichment or mentorship opportunities.",
        "Use as peer tutor candidate for at-risk classmates.",
    ],
}


class StudentPerformanceAgent:
    """
    Intelligent Agent that wraps the ML classifier with decision-making logic.

    Architecture layers:
    - Percepts: student feature vector from sensors (SIS, LMS, surveys)
    - Learning Component: trained sklearn model (best_model.pkl)
    - Knowledge Base: intervention rules and performance thresholds
    - Decision Engine: classify → assess risk → generate recommendations
    - Memory: optional history of past predictions per student
    """

    def __init__(self, model_path: Optional[Path] = None):
        paths = get_paths()
        self.model_path = model_path or (paths["models"] / "best_model.pkl")
        self._artifact: Optional[Dict[str, Any]] = None
        self.memory: Dict[str, List[AgentPrediction]] = {}

    def load(self) -> None:
        """Load persisted model artifact from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run training first: python main.py --train"
            )
        self._artifact = joblib.load(self.model_path)

    @property
    def artifact(self) -> Dict[str, Any]:
        if self._artifact is None:
            self.load()
        return self._artifact  # type: ignore[return-value]

    def _get_column_types(self) -> tuple[List[str], List[str]]:
        """Return numeric and categorical column lists from artifact or preprocessor."""
        if "numeric_columns" in self.artifact and "categorical_columns" in self.artifact:
            return self.artifact["numeric_columns"], self.artifact["categorical_columns"]

        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        for name, _, cols in self.artifact["preprocessor"].transformers_:
            if name == "num":
                numeric_cols = list(cols)
            elif name == "cat":
                categorical_cols = list(cols)
        return numeric_cols, categorical_cols

    def preprocess_input(self, student_data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """Normalize columns and align with training feature schema."""
        if isinstance(student_data, dict):
            df = pd.DataFrame([student_data])
        else:
            df = student_data.copy()

        df = normalize_column_names(df)

        # Drop target/leakage columns if present
        for col in ["G3", "performance_category", "score", "final_grade"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        expected = self.artifact["input_columns"]
        numeric_cols, categorical_cols = self._get_column_types()

        for col in expected:
            if col not in df.columns:
                df[col] = 0 if col in numeric_cols else "missing"

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df[expected]

    def predict(
        self,
        student_data: Union[pd.DataFrame, Dict[str, Any]],
        student_id: Optional[str] = None,
    ) -> Union[AgentPrediction, List[AgentPrediction]]:
        """
        Run the full agent pipeline: perceive → classify → decide → act.

        Returns AgentPrediction with category, confidence, and recommendations.
        """
        df = self.preprocess_input(student_data)
        X = self.artifact["preprocessor"].transform(df)

        model = self.artifact["model"]
        encoder = self.artifact["target_encoder"]

        pred_idx = model.predict(X)
        proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        class_labels = list(encoder.classes_)

        predictions: List[AgentPrediction] = []
        for i in range(len(df)):
            label = encoder.inverse_transform([pred_idx[i]])[0]
            if proba is not None:
                conf = float(proba[i].max())
                prob_dict = {
                    class_labels[j]: float(proba[i][j])
                    for j in range(len(class_labels))
                }
            else:
                conf = 1.0
                prob_dict = {label: 1.0}

            risk = self._assess_risk(label, prob_dict)
            recs = self._generate_recommendations(label, student_data, i)
            priority = {"Low": "HIGH", "Medium": "MEDIUM", "High": "LOW"}[label]

            pred = AgentPrediction(
                performance_category=label,
                confidence=conf,
                probabilities=prob_dict,
                risk_level=risk,
                recommendations=recs,
                intervention_priority=priority,
            )
            predictions.append(pred)

            if student_id:
                self.memory.setdefault(student_id, []).append(pred)

        return predictions[0] if len(predictions) == 1 else predictions

    def _assess_risk(self, label: str, probabilities: Dict[str, float]) -> str:
        """Decision engine: combine prediction with probability spread."""
        low_prob = probabilities.get("Low", 0.0)
        if label == "Low" or low_prob >= 0.35:
            return "Critical"
        if label == "Medium" or low_prob >= 0.2:
            return "Moderate"
        return "Low"

    def _generate_recommendations(
        self,
        label: str,
        student_data: Union[pd.DataFrame, Dict[str, Any]],
        row_idx: int,
    ) -> List[str]:
        """Knowledge-base rules augmented with feature-specific hints."""
        recs = list(INTERVENTION_RULES.get(label, []))

        if isinstance(student_data, dict):
            row = student_data
        else:
            row = student_data.iloc[row_idx].to_dict()

        failures = row.get("failures", 0)
        absences = row.get("absences", 0)
        studytime = row.get("studytime", 2)

        try:
            failures = int(failures)
            absences = int(absences)
            studytime = int(studytime)
        except (TypeError, ValueError):
            pass

        if failures >= 2:
            recs.append("Prioritize remediation for previously failed subjects.")
        if absences > 15:
            recs.append("Address chronic absenteeism with attendance contract.")
        if studytime <= 1:
            recs.append("Increase weekly study time target to at least 2–5 hours.")

        return recs[:6]

    def batch_predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score an entire cohort and return dashboard-ready DataFrame."""
        df = normalize_column_names(df.copy())
        processed = self.preprocess_input(df)
        X = self.artifact["preprocessor"].transform(processed)

        model = self.artifact["model"]
        encoder = self.artifact["target_encoder"]
        pred_idx = model.predict(X)
        proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        class_labels = list(encoder.classes_)

        results = []
        for i in range(len(df)):
            idx = df.index[i]
            label = encoder.inverse_transform([pred_idx[i]])[0]
            if proba is not None:
                conf = float(proba[i].max())
            else:
                conf = 1.0

            prob_dict = (
                {class_labels[j]: float(proba[i][j]) for j in range(len(class_labels))}
                if proba is not None
                else {label: 1.0}
            )
            risk = self._assess_risk(label, prob_dict)
            recs = self._generate_recommendations(label, df, i)
            priority = {"Low": "HIGH", "Medium": "MEDIUM", "High": "LOW"}[label]

            results.append({
                "student_index": idx,
                "predicted_category": label,
                "confidence": conf,
                "risk_level": risk,
                "intervention_priority": priority,
                "top_recommendation": recs[0] if recs else "",
            })

        return pd.DataFrame(results)

    def batch_predict_csv(self, csv_path: Path) -> pd.DataFrame:
        """Score students from a CSV file and return dashboard-ready DataFrame."""
        df = pd.read_csv(csv_path, sep=";", engine="python")
        return self.batch_predict_df(df)


def load_agent(model_path: Optional[Path] = None) -> StudentPerformanceAgent:
    """Factory function to create a ready-to-use agent."""
    agent = StudentPerformanceAgent(model_path)
    agent.load()
    return agent
