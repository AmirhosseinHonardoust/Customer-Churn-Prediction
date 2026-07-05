"""Tests for the inference CLI."""

import json

import numpy as np
from generate_customers import generate
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import predict as predict_mod
from train_models import build_preprocessor


def _train_tiny_model(tmp_path, seed=1):
    df = generate(400, seed=seed)
    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"]
    pre, _, _ = build_preprocessor(X)
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    model_path = tmp_path / "model.joblib"
    dump(pipe, model_path)
    return model_path


def test_predict_output_shape_and_values(tmp_path):
    model_path = _train_tiny_model(tmp_path)
    new = generate(50, seed=99)
    csv = tmp_path / "new.csv"
    new.to_csv(csv, index=False)

    out = predict_mod.predict(str(model_path), str(csv), threshold=0.5)
    assert list(out.columns) == ["customer_id", "churn_proba", "churn_pred"]
    assert len(out) == 50
    assert np.all((out["churn_proba"] >= 0) & (out["churn_proba"] <= 1))
    assert set(out["churn_pred"].unique()).issubset({0, 1})


def test_predict_without_customer_id(tmp_path):
    model_path = _train_tiny_model(tmp_path)
    new = generate(20, seed=5).drop(columns=["customer_id"])
    csv = tmp_path / "new.csv"
    new.to_csv(csv, index=False)

    out = predict_mod.predict(str(model_path), str(csv), threshold=0.5)
    assert "customer_id" not in out.columns
    assert len(out) == 20


def test_predict_missing_feature_raises(tmp_path):
    import pytest

    model_path = _train_tiny_model(tmp_path)
    new = generate(10, seed=2).drop(columns=["age"])
    csv = tmp_path / "bad.csv"
    new.to_csv(csv, index=False)

    with pytest.raises(ValueError, match="age"):
        predict_mod.predict(str(model_path), str(csv), threshold=0.5)


def test_load_threshold_reads_and_falls_back(tmp_path):
    assert predict_mod.load_threshold(str(tmp_path / "missing.json"), default=0.42) == 0.42
    metrics = {"selection": {"threshold": {"thr": 0.7}}}
    mpath = tmp_path / "metrics.json"
    mpath.write_text(json.dumps(metrics))
    assert predict_mod.load_threshold(str(mpath)) == 0.7


def test_predict_on_arbitrary_schema(tmp_path):
    # Dataset-agnostic: a schema unlike the synthetic one still trains and scores.
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from train_models import build_preprocessor

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "customer_id": range(200),
            "num_feat": rng.normal(size=200),
            "cat_feat": rng.choice(["a", "b", "c"], size=200),
            "churn": rng.integers(0, 2, size=200),
        }
    )
    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"]
    pre, _, _ = build_preprocessor(X)
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))]).fit(X, y)

    model_path = tmp_path / "m.joblib"
    dump(pipe, model_path)
    new = df.drop(columns=["churn"]).head(10)
    csv = tmp_path / "new.csv"
    new.to_csv(csv, index=False)

    out = predict_mod.predict(str(model_path), str(csv), threshold=0.5)
    assert list(out.columns) == ["customer_id", "churn_proba", "churn_pred"]
    assert len(out) == 10
