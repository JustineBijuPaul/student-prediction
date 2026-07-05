"""
Model evaluation utilities for the Student Performance Prediction Agent.

Computes classification metrics, cross-validation scores, and comparison tables.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import label_binarize

from src.utils import RANDOM_STATE, get_paths


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """Calculate standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    if y_proba is not None:
        n_classes = y_proba.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes > 2:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_bin, y_proba, average="weighted", multi_class="ovr"
                )
            except ValueError:
                metrics["roc_auc"] = np.nan
        else:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])

    return metrics


def evaluate_model(
    model: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    class_labels: List[str],
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Train model, measure train/test metrics, timing, and cross-validation.

    Returns a dictionary suitable for aggregation into a comparison DataFrame.
    """
    # Training time
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Prediction time (batch)
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t1

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

    test_metrics = compute_metrics(y_test, y_pred, y_proba)
    train_pred = model.predict(X_train)
    train_metrics = compute_metrics(y_train, train_pred)

    # Cross-validation on training set
    scoring = {
        "cv_accuracy": "accuracy",
        "cv_f1": "f1_weighted",
        "cv_precision": "precision_weighted",
        "cv_recall": "recall_weighted",
    }
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    return {
        "model": model_name,
        "estimator": model,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics.get("roc_auc", np.nan),
        "train_accuracy": train_metrics["accuracy"],
        "train_f1": train_metrics["f1"],
        "cv_accuracy_mean": cv_results["test_cv_accuracy"].mean(),
        "cv_accuracy_std": cv_results["test_cv_accuracy"].std(),
        "cv_f1_mean": cv_results["test_cv_f1"].mean(),
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=class_labels,
            zero_division=0,
        ),
    }


def build_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-model evaluation dicts into a sorted comparison table."""
    rows = []
    for r in results:
        rows.append({
            "Model": r["model"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1 Score": round(r["f1"], 4),
            "ROC AUC": round(r.get("roc_auc", np.nan), 4),
            "CV Accuracy (mean)": round(r["cv_accuracy_mean"], 4),
            "CV Accuracy (std)": round(r["cv_accuracy_std"], 4),
            "Train Time (s)": round(r["train_time_sec"], 4),
            "Predict Time (s)": round(r["predict_time_sec"], 6),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    return df


def select_best_model(results: List[Dict[str, Any]], metric: str = "f1") -> Dict[str, Any]:
    """Return the result dict for the highest-scoring model on the given metric."""
    key_map = {
        "f1": "f1",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }
    key = key_map.get(metric, "f1")
    return max(results, key=lambda r: r.get(key, 0) or 0)
