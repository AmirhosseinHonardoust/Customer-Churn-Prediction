"""Tests for threshold scoring and selection logic."""

import numpy as np

from train_models import _score_threshold, pick_threshold


def test_score_threshold_perfect_separation():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    precision, recall, fbeta = _score_threshold(y, proba, thr=0.5, beta=2.0)
    assert precision > 0.99
    assert recall > 0.99
    assert fbeta > 0.99


def test_pick_threshold_respects_precision_floor():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    proba = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, size=400), 0, 1)
    result = pick_threshold(y, proba, beta=2.0, min_precision=0.6)
    assert result["precision"] is not None
    # Either the floor is met, or we fell back to the default threshold.
    assert result["precision"] >= 0.6 or result["thr"] == 0.5


def test_pick_threshold_falls_back_when_floor_unreachable():
    # All-negative labels: no positive predictions can reach any precision floor.
    y = np.zeros(100, dtype=int)
    proba = np.linspace(0, 1, 100)
    result = pick_threshold(y, proba, beta=2.0, min_precision=0.9)
    assert result["thr"] == 0.5


def test_pick_threshold_returns_expected_keys():
    y = np.array([0, 1, 0, 1, 1, 0])
    proba = np.array([0.2, 0.7, 0.3, 0.8, 0.6, 0.1])
    result = pick_threshold(y, proba, beta=2.0, min_precision=0.0)
    assert set(result) == {"thr", "fbeta", "precision", "recall"}
    assert 0.0 <= result["thr"] <= 1.0
