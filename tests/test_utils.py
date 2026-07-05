"""Tests for data loading and splitting helpers."""

import pandas as pd
from generate_customers import generate

from utils import load_dataset, split_xy


def test_load_dataset_roundtrip(tmp_path):
    df = generate(200, seed=5)
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    loaded = load_dataset(str(csv))
    pd.testing.assert_frame_equal(loaded, df)


def test_split_sizes_sum_and_no_target_leakage():
    df = generate(1000, seed=9)
    X_train, X_val, X_test, y_train, y_val, y_test = split_xy(df, 0.2, 0.2, seed=9)
    assert len(X_train) + len(X_val) + len(X_test) == len(df)
    assert len(y_train) + len(y_val) + len(y_test) == len(df)
    for X in (X_train, X_val, X_test):
        assert "churn" not in X.columns
        assert "customer_id" not in X.columns


def test_split_is_deterministic():
    df = generate(500, seed=11)
    first = split_xy(df, 0.2, 0.2, seed=11)
    second = split_xy(df, 0.2, 0.2, seed=11)
    for a, b in zip(first, second, strict=True):
        assert list(a.index) == list(b.index)


def test_split_stratification_roughly_preserved():
    df = generate(3000, seed=13)
    base = df["churn"].mean()
    _, _, _, y_train, y_val, y_test = split_xy(df, 0.2, 0.2, seed=13)
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - base) < 0.03
