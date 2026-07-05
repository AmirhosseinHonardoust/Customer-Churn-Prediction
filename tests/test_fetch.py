"""Tests for Telco dataset normalisation (no network required)."""

import pandas as pd
from fetch_telco import normalise


def test_normalise_maps_columns_and_target():
    raw = pd.DataFrame(
        {
            "customerID": ["a1", "b2"],
            "gender": ["Male", "Female"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": ["29.85", " "],  # blank -> 0.0
            "Churn": ["No", "Yes"],
        }
    )
    out = normalise(raw)
    assert "customer_id" in out.columns
    assert "churn" in out.columns
    assert out["churn"].tolist() == [0, 1]
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])
    assert out["TotalCharges"].tolist() == [29.85, 0.0]


def test_normalise_target_is_binary():
    raw = pd.DataFrame(
        {
            "customerID": ["x"],
            "feat": [1.0],
            "TotalCharges": ["10.0"],
            "Churn": ["yes"],
        }
    )
    out = normalise(raw)
    assert set(out["churn"].unique()).issubset({0, 1})
