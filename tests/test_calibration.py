"""Tests for optional probability calibration."""

import numpy as np
from generate_customers import generate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from train_models import build_preprocessor, calibrate_model
from utils import positive_proba, split_xy


def test_calibrate_model_returns_valid_probabilities():
    df = generate(800, seed=31)
    X_train, X_val, X_test, y_train, y_val, _ = split_xy(df, 0.2, 0.2, seed=31)
    pre, _, _ = build_preprocessor(X_train)
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_train, y_train)

    calibrated = calibrate_model(pipe, X_val, y_val)
    proba = positive_proba(calibrated, X_test)
    assert proba.shape == (len(X_test),)
    assert np.all((proba >= 0) & (proba <= 1))
