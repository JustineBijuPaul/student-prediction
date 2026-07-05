"""
Data preprocessing pipeline for the Student Performance Prediction Agent.

Handles cleaning, encoding, scaling, outlier treatment, and train/test splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.utils import (
    RANDOM_STATE,
    PERFORMANCE_LABELS,
    grade_to_performance_category,
    load_student_dataset,
    normalize_column_names,
)


@dataclass
class PreprocessResult:
    """Container for fitted preprocessor and split data."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: List[str]
    raw_df: pd.DataFrame
    target_encoder: LabelEncoder


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard cleaning steps: duplicates, missing values, type coercion.

    UCI Student Performance has no missing values; this remains robust for Kaggle variants.
    """
    df = normalize_column_names(df.copy())

    # Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Coerce numeric columns where possible
    numeric_candidates = [
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
        "G1", "G2", "G3",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute numeric missing with median; categorical with mode
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else "unknown"
                df[col] = df[col].fillna(fill)

    return df


def cap_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Winsorize extreme values using the IQR rule (1.5 × IQR).

    Applied only to continuous numeric features to preserve ordinal encodings.
    """
    df = df.copy()
    if columns is None:
        columns = ["absences", "G1", "G2", "failures"]

    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def build_target(df: pd.DataFrame, target_source: str = "G3") -> pd.DataFrame:
    """Create performance_category from final grade column."""
    df = df.copy()
    if target_source not in df.columns:
        raise ValueError(f"Target source column '{target_source}' not found after normalization.")

    df["performance_category"] = grade_to_performance_category(df[target_source])
    df = df.dropna(subset=["performance_category"])
    return df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Select feature matrix X and target y.

    Excludes G3 (direct target) but retains G1/G2 as prior performance signals.
    """
    target_col = "performance_category"
    leak_cols = {"G3", "performance_category", "score", "final_grade", "FinalGrade"}
    feature_cols = [c for c in df.columns if c not in leak_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, feature_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construct sklearn ColumnTransformer for mixed numeric/categorical data."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> List[str]:
    """Extract human-readable feature names after one-hot encoding."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return list(X.columns)


def preprocess_pipeline(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    cap_outliers: bool = True,
) -> PreprocessResult:
    """
    Full preprocessing workflow: clean → engineer target → split → fit preprocessor.

    Returns fitted preprocessor and stratified train/test sets.
    """
    if df is None:
        df = load_student_dataset()

    df = clean_dataframe(df)
    if cap_outliers:
        df = cap_outliers_iqr(df)
    df = build_target(df)

    X, y, _ = select_features(df)

    target_encoder = LabelEncoder()
    y_encoded = pd.Series(
        target_encoder.fit_transform(y),
        index=y.index,
        name="performance_category",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    feature_names = get_feature_names_from_preprocessor(preprocessor, X_train)

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        raw_df=df,
        target_encoder=target_encoder,
    )


def transform_features(result: PreprocessResult) -> Tuple[np.ndarray, np.ndarray]:
    """Apply fitted preprocessor to train and test features."""
    X_train_t = result.preprocessor.transform(result.X_train)
    X_test_t = result.preprocessor.transform(result.X_test)
    return X_train_t, X_test_t
