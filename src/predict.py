"""Score new customers with a trained churn model.

Loads a saved pipeline (``best_model.joblib``), selects the exact feature columns
the model was trained on, validates they are present, and writes per-customer
churn probabilities and 0/1 predictions to a CSV. Works with any dataset the
model was trained on (synthetic or real).
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from joblib import load

from utils import feature_columns, load_dataset, positive_proba, validate_schema


def load_threshold(metrics_path: str, default: float = 0.5) -> float:
    """Read the tuned decision threshold from a metrics.json, falling back to ``default``."""
    if not metrics_path or not os.path.exists(metrics_path):
        return default
    with open(metrics_path) as f:
        metrics = json.load(f)
    return float(metrics.get("selection", {}).get("threshold", {}).get("thr", default))


def model_feature_names(pipe, df: pd.DataFrame) -> list[str]:
    """Feature columns the model expects (from the fitted pipeline, else inferred)."""
    names = getattr(pipe, "feature_names_in_", None)
    if names is not None:
        return list(names)
    return feature_columns(df)


def predict(model_path: str, input_path: str, threshold: float) -> pd.DataFrame:
    """Return a DataFrame with ``customer_id`` (if present), ``churn_proba``, ``churn_pred``."""
    pipe = load(model_path)
    df = load_dataset(input_path)
    validate_schema(df, require_target=False)

    features = model_feature_names(pipe, df)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required feature column(s): {', '.join(missing)}")

    proba = positive_proba(pipe, df[features])
    out = pd.DataFrame(
        {
            "churn_proba": proba,
            "churn_pred": (proba >= threshold).astype(int),
        }
    )
    if "customer_id" in df.columns:
        out.insert(0, "customer_id", df["customer_id"].to_numpy())
    return out


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="outputs/best_model.joblib")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="outputs/predictions.csv")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold; if omitted, read from --metrics, else 0.5.",
    )
    ap.add_argument("--metrics", default="outputs/metrics.json")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    """Load a model, score the input CSV, and write predictions."""
    args = parse_args(argv)
    threshold = args.threshold if args.threshold is not None else load_threshold(args.metrics)

    result = predict(args.model, args.input, threshold)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"[OK] Scored {len(result):,} rows at threshold={threshold:.3f}")
    print(f"Predicted churn rate: {result['churn_pred'].mean():.3f}")
    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
