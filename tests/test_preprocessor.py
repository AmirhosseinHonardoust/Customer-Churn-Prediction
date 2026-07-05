"""Tests for preprocessing and probability extraction."""

import numpy as np
from generate_customers import generate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from train_models import build_preprocessor, positive_proba
from utils import split_xy


def test_build_preprocessor_identifies_columns():
    df = generate(400, seed=21)
    X = df.drop(columns=["churn", "customer_id"])
    _, num_cols, cat_cols = build_preprocessor(X)
    assert "region" in cat_cols
    assert "age" in num_cols
    assert "region" not in num_cols


def test_preprocessor_fit_transform_shape():
    df = generate(400, seed=22)
    X = df.drop(columns=["churn", "customer_id"])
    pre, _, _ = build_preprocessor(X)
    transformed = pre.fit_transform(X)
    # One-hot on region (4 levels) adds columns beyond the numeric ones.
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] >= X.shape[1]


def test_positive_proba_range_and_shape():
    df = generate(600, seed=23)
    X_train, X_val, _, y_train, _, _ = split_xy(df, 0.2, 0.2, seed=23)
    pre, _, _ = build_preprocessor(X_train)
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_train, y_train)
    proba = positive_proba(pipe, X_val)
    assert proba.shape == (len(X_val),)
    assert np.all((proba >= 0) & (proba <= 1))
