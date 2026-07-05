"""Data loading, schema validation, splitting, and probability helpers.

The pipeline is dataset-agnostic: any CSV with a binary ``churn`` target and at
least one feature column works. ``customer_id`` is treated as an identifier and
excluded from features when present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"


def load_dataset(path: str) -> pd.DataFrame:
    """Read the customer CSV at ``path`` into a DataFrame."""
    return pd.read_csv(path)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the modelling feature columns (everything but target and id)."""
    return [c for c in df.columns if c not in (TARGET_COLUMN, ID_COLUMN)]


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """Raise ``ValueError`` if the frame can't be used for training/inference.

    Requires a ``churn`` target (when ``require_target``) and at least one
    feature column, so real datasets fail fast with a clear message instead of
    erroring deep inside the pipeline.
    """
    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(f"Input is missing required target column: {TARGET_COLUMN}")
    if not feature_columns(df):
        raise ValueError("Input has no feature columns (only id/target present).")


def split_xy(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    """Split ``df`` into stratified train/val/test feature and target sets.

    ``customer_id`` is dropped (if present) and ``churn`` is used as the target.
    Returns ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN].astype(int)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def positive_proba(pipe, X) -> np.ndarray:
    """Return P(class=1) for ``X``, normalising decision scores when needed."""
    clf = pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    dec = pipe.decision_function(X)
    return (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
