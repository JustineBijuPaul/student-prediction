"""
Model training pipeline for the Student Performance Prediction Agent.

Trains and compares multiple classifiers, selects the best model, and persists artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import build_comparison_table, evaluate_model, select_best_model
from src.preprocess import preprocess_pipeline, transform_features
from src.utils import RANDOM_STATE, ensure_directories, get_paths
from src.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curve_multiclass,
    generate_eda_figures,
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def get_model_registry() -> Dict[str, Any]:
    """
    Return sklearn (and optional XGBoost) classifiers with tuned hyperparameters.

    Hyperparameters chosen for stability on small tabular educational data.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    return models


def extract_feature_importance(
    model: Any,
    feature_names: List[str],
) -> Tuple[List[str], np.ndarray]:
    """Extract feature importances from tree-based models or use coefficient magnitudes."""
    if hasattr(model, "feature_importances_"):
        return feature_names, np.array(model.feature_importances_)

    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_).mean(axis=0)
        return feature_names, coef

    # Fallback: uniform importance
    n = len(feature_names)
    return feature_names, np.ones(n) / n


def train_all_models(
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end training: preprocess → train all models → evaluate → save best.

    Returns dictionary with comparison table, best model info, and paths.
    """
    ensure_directories()
    paths = get_paths()

    # Preprocess
    prep = preprocess_pipeline()
    X_train, X_test = transform_features(prep)
    y_train = prep.y_train.values
    y_test = prep.y_test.values

    # EDA figures on full cleaned dataframe
    if save_artifacts:
        generate_eda_figures(prep.raw_df)

    # Train and evaluate each model
    results: List[Dict[str, Any]] = []
    registry = get_model_registry()

    for name, estimator in registry.items():
        print(f"Training {name}...")
        result = evaluate_model(
            estimator, X_train, y_train, X_test, y_test, name,
            class_labels=list(prep.target_encoder.classes_),
        )
        results.append(result)

    comparison = build_comparison_table(results)
    best = select_best_model(results, metric="f1")
    best_name = best["model"]
    best_model = best["estimator"]

    print("\n=== Model Comparison ===")
    print(comparison.to_string(index=False))
    print(f"\nBest model: {best_name} (F1 = {best['f1']:.4f})")

    if save_artifacts:
        # Evaluation plots
        plot_model_comparison(comparison)
        plot_confusion_matrix(best["confusion_matrix"])
        if best["y_proba"] is not None:
            plot_roc_curve_multiclass(y_test, best["y_proba"])

        # Feature importance
        feat_names, importances = extract_feature_importance(best_model, prep.feature_names)
        plot_feature_importance(feat_names, importances)

        # Save full pipeline artifact
        artifact = {
            "model": best_model,
            "preprocessor": prep.preprocessor,
            "target_encoder": prep.target_encoder,
            "feature_names": prep.feature_names,
            "input_columns": list(prep.X_train.columns),
            "numeric_columns": prep.X_train.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": prep.X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
            "best_model_name": best_name,
            "metrics": {
                "accuracy": best["accuracy"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"],
                "roc_auc": best.get("roc_auc"),
            },
            "classification_report": best["classification_report"],
            "comparison_table": comparison.to_dict(orient="records"),
        }

        model_path = paths["models"] / "best_model.pkl"
        joblib.dump(artifact, model_path)

        meta_path = paths["models"] / "training_metadata.json"
        meta = {
            "best_model": best_name,
            "metrics": artifact["metrics"],
            "random_state": RANDOM_STATE,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        comparison.to_csv(paths["outputs"] / "model_comparison.csv", index=False)

    return {
        "preprocess": prep,
        "results": results,
        "comparison": comparison,
        "best": best,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    train_all_models()
