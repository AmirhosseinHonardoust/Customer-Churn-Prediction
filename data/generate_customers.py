"""Generate a synthetic customer dataset with a plausible churn signal."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

REGIONS = ["North", "South", "East", "West"]


def sigmoid(x):
    """Numerically standard logistic sigmoid."""
    return 1 / (1 + np.exp(-x))


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate ``n`` synthetic customer rows with a ``churn`` label.

    The churn probability is a logistic function of behaviour signals (tenure,
    spend, recency, support calls, complaints, discounts, premium, region), so
    the resulting dataset has a learnable but noisy relationship.
    """
    rng = np.random.default_rng(seed)

    customer_id = np.arange(1, n + 1)
    age = rng.integers(18, 75, size=n)
    region = rng.choice(REGIONS, size=n, p=[0.28, 0.26, 0.24, 0.22])
    tenure_months = rng.integers(1, 72, size=n)
    is_premium = rng.integers(0, 2, size=n)

    monthly_spend = np.round(rng.normal(60 + 20 * is_premium + 0.5 * tenure_months, 15), 2)
    monthly_spend = np.clip(monthly_spend, 5, None)

    avg_txn_value = np.round(rng.normal(25 + 10 * is_premium, 8), 2)
    avg_txn_value = np.clip(avg_txn_value, 2, None)

    txns_last_30d = np.maximum(
        0, rng.poisson(lam=4 + 3 * is_premium - 0.03 * np.maximum(0, age - 45))
    )
    days_since_last_purchase = np.maximum(
        0, rng.integers(0, 90, size=n) + (1 - is_premium) * rng.integers(0, 30, size=n)
    )

    customer_service_calls = np.maximum(
        0, rng.poisson(lam=0.8 + 0.02 * np.maximum(0, 40 - tenure_months))
    )
    discounts_used_90d = np.maximum(0, rng.poisson(lam=1.5 + 1.0 * (1 - is_premium)))
    complaints_90d = rng.binomial(3, 0.05 + 0.02 * (1 - is_premium), size=n)

    logit = (
        -2.2
        - 0.02 * tenure_months
        - 0.015 * monthly_spend
        - 0.03 * txns_last_30d
        + 0.03 * days_since_last_purchase
        + 0.35 * customer_service_calls
        + 0.25 * discounts_used_90d
        + 0.5 * complaints_90d
        - 0.5 * is_premium
    )
    region_w = {"North": 0.0, "South": 0.1, "East": 0.05, "West": -0.05}
    logit += np.array([region_w[r] for r in region])

    p_churn = sigmoid(logit)
    churn = rng.binomial(1, p_churn)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "region": region,
            "tenure_months": tenure_months,
            "is_premium": is_premium,
            "monthly_spend": monthly_spend,
            "avg_txn_value": avg_txn_value,
            "txns_last_30d": txns_last_30d,
            "days_since_last_purchase": days_since_last_purchase,
            "customer_service_calls": customer_service_calls,
            "discounts_used_90d": discounts_used_90d,
            "complaints_90d": complaints_90d,
            "churn": churn.astype(int),
        }
    )
    return df


def main(argv=None) -> None:
    """Parse args and write a generated dataset to CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/customers.csv")
    args = ap.parse_args(argv)

    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} with {len(df):,} rows")


if __name__ == "__main__":
    main()
