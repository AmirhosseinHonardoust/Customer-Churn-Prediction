"""Tests for the synthetic data generator."""

import numpy as np
import pandas as pd
from generate_customers import generate

EXPECTED_COLUMNS = [
    "customer_id",
    "age",
    "region",
    "tenure_months",
    "is_premium",
    "monthly_spend",
    "avg_txn_value",
    "txns_last_30d",
    "days_since_last_purchase",
    "customer_service_calls",
    "discounts_used_90d",
    "complaints_90d",
    "churn",
]


def test_shape_and_columns():
    df = generate(500, seed=1)
    assert len(df) == 500
    assert list(df.columns) == EXPECTED_COLUMNS


def test_deterministic_for_same_seed():
    a = generate(300, seed=7)
    b = generate(300, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = generate(300, seed=1)
    b = generate(300, seed=2)
    assert not a["churn"].equals(b["churn"])


def test_churn_is_binary_and_plausible_rate():
    df = generate(2000, seed=42)
    assert set(df["churn"].unique()).issubset({0, 1})
    assert 0.05 < df["churn"].mean() < 0.5


def test_non_negativity_constraints():
    df = generate(1000, seed=3)
    for col in [
        "txns_last_30d",
        "days_since_last_purchase",
        "customer_service_calls",
        "discounts_used_90d",
        "complaints_90d",
    ]:
        assert (df[col] >= 0).all()
    assert (df["monthly_spend"] >= 5).all()
    assert np.isin(df["region"].unique(), ["North", "South", "East", "West"]).all()
