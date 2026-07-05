"""
Publication-quality visualizations for EDA and model evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize

from src.utils import PERFORMANCE_LABELS, ensure_directories, get_paths

# Consistent plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
FIG_DPI = 150
FIG_SIZE = (10, 6)


def _savefig(name: str) -> Path:
    """Save figure to outputs/ directory."""
    ensure_directories()
    path = get_paths()["outputs"] / name
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_correlation_heatmap(df: pd.DataFrame, filename: str = "correlation_heatmap.png") -> Path:
    """Plot Pearson correlation matrix for numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("No numeric columns for correlation heatmap.")

    fig, ax = plt.subplots(figsize=(14, 10))
    corr = numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap (Pearson)", fontsize=14, fontweight="bold")
    return _savefig(filename)


def plot_class_distribution(y: pd.Series, filename: str = "class_distribution.png") -> Path:
    """Bar chart of performance category distribution."""
    counts = y.value_counts().reindex(PERFORMANCE_LABELS).fillna(0)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Performance Category", fontsize=12)
    ax.set_ylabel("Number of Students", fontsize=12)
    ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{int(val)}", ha="center", va="bottom", fontsize=11,
        )

    return _savefig(filename)


def plot_confusion_matrix(
    cm: np.ndarray,
    filename: str = "confusion_matrix.png",
) -> Path:
    """Annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=PERFORMANCE_LABELS,
        yticklabels=PERFORMANCE_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Best Model", fontsize=14, fontweight="bold")
    return _savefig(filename)


def plot_roc_curve_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    filename: str = "roc_curve.png",
) -> Path:
    """One-vs-rest ROC curves for multiclass classification."""
    from sklearn.metrics import roc_curve

    n_classes = len(PERFORMANCE_LABELS)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    for i, (label, color) in enumerate(zip(PERFORMANCE_LABELS, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    return _savefig(filename)


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    filename: str = "model_comparison.png",
) -> Path:
    """Grouped bar chart comparing model F1 scores."""
    fig, ax = plt.subplots(figsize=(12, 6))
    models = comparison_df["Model"]
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
    x = np.arange(len(models))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, comparison_df[metric], width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    return _savefig(filename)


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 15,
    filename: str = "feature_importance.png",
) -> Path:
    """Horizontal bar chart of top feature importances."""
    idx = np.argsort(importances)[-top_n:]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_names, top_vals, color="#3498db", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — Best Model", fontsize=14, fontweight="bold")
    return _savefig(filename)


def plot_pairplot_sample(
    df: pd.DataFrame,
    filename: str = "pairplot.png",
    sample_size: int = 300,
) -> Path:
    """Pairplot for key numeric features (sampled for performance)."""
    cols = [c for c in ["age", "studytime", "absences", "G1", "G2", "G3", "failures"] if c in df.columns]
    if "performance_category" in df.columns:
        plot_df = df[cols + ["performance_category"]].copy()
        hue = "performance_category"
    else:
        plot_df = df[cols].copy()
        hue = None

    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)

    g = sns.pairplot(plot_df, hue=hue, diag_kind="kde", corner=True, plot_kws={"alpha": 0.6, "s": 25})
    g.fig.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=14, fontweight="bold")
    ensure_directories()
    path = get_paths()["outputs"] / filename
    g.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close("all")
    return path


def generate_eda_figures(df: pd.DataFrame) -> Dict[str, Path]:
    """Generate all standard EDA output figures."""
    paths = {}
    if "performance_category" not in df.columns:
        from src.utils import grade_to_performance_category
        if "G3" in df.columns:
            df = df.copy()
            df["performance_category"] = grade_to_performance_category(df["G3"])

    paths["correlation_heatmap"] = plot_correlation_heatmap(df)
    if "performance_category" in df.columns:
        paths["class_distribution"] = plot_class_distribution(df["performance_category"])
    paths["pairplot"] = plot_pairplot_sample(df)
    return paths
