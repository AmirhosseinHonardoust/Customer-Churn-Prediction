"""Download and normalise the IBM Telco customer-churn dataset.

Fetches the public IBM sample dataset and writes a pipeline-ready CSV with a
lowercase ``customer_id`` identifier and a binary ``churn`` target, so it can be
fed straight into ``src/train_models.py``. Requires network access; the output
CSV is git-ignored.

Source: https://github.com/IBM/telco-customer-churn-on-icp4d (IBM sample data).
"""

from __future__ import annotations

import argparse

import pandas as pd

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw Telco columns to the pipeline's ``customer_id`` / ``churn`` schema."""
    df = df.rename(columns={"customerID": "customer_id", "Churn": "churn"})
    df["churn"] = (df["churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    # TotalCharges ships as text with blanks for tenure-0 customers; coerce to numeric.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def main(argv=None) -> None:
    """Download, normalise, and save the Telco dataset."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=TELCO_URL)
    ap.add_argument("--out", default="data/telco_churn.csv")
    args = ap.parse_args(argv)

    raw = pd.read_csv(args.url)
    df = normalise(raw)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} with {len(df):,} rows, churn rate {df['churn'].mean():.3f}")


if __name__ == "__main__":
    main()
