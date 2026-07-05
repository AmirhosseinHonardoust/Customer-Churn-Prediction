# Model Card — Customer Churn Prediction

## Overview
A binary classifier that estimates the probability a customer will churn, trained
and selected among logistic regression, random forest, and gradient boosting.
The selected model is chosen by validation PR-AUC; its decision threshold is tuned
for F2 (recall-leaning) subject to a minimum-precision floor.

## Intended use
- **Intended:** prioritising retention outreach, exploratory analysis, and as a
  reference/education pipeline for churn modelling.
- **Not intended:** automated decisions that materially affect customers without
  human review, or use on populations unlike the training data.

## Training data
The pipeline is **dataset-agnostic**: any CSV with a binary `churn` target and at
least one feature column works (an optional `customer_id` is excluded from
features). Two datasets are supported out of the box:

- **Synthetic** (`data/generate_customers.py`) — churn is a logistic function of
  behavioural signals; useful for a fully reproducible, offline demo. Absolute
  metrics are illustrative, not a real-world claim.
- **IBM Telco** (`data/fetch_telco.py`) — the public IBM sample churn dataset
  (~7,043 customers, 26.5% churn), normalised to the `customer_id`/`churn` schema.
  Provenance: https://github.com/IBM/telco-customer-churn-on-icp4d.

## Evaluation
Metrics are computed on a held-out, stratified test split and written to
`outputs/metrics.json`, alongside ROC, precision-recall, confusion-matrix,
calibration (reliability), and permutation-importance figures.

Reference values (seed 42):

| Dataset | Model | ROC-AUC | PR-AUC | Recall (churn) | Brier |
|---|---|---|---|---|---|
| Synthetic | Logistic Regression | 0.823 | 0.562 | 0.50 | — |
| IBM Telco | Gradient Boosting | 0.847 | 0.666 | 0.77 | 0.135 |

## Calibration
Probabilities from the default model are **not** calibrated. Pass `--calibrate`
to `src/train_models.py` to isotonic-calibrate the selected model on the
validation set; a reliability curve and Brier score are always written for
inspection.

## Limitations & risks
- Synthetic training data limits external validity; retrain on representative data
  before operational use.
- Permutation importance reflects association, not causation.
- No fairness/subgroup analysis is performed; assess disparate impact before use
  in decisions affecting people.
- Distribution shift over time will degrade performance; monitor and retrain.

## Reproducing
Synthetic:
```bash
python data/generate_customers.py --n 10000 --seed 42 --out data/customers.csv
python src/train_models.py --input data/customers.csv --outdir outputs --seed 42
```
Real (IBM Telco):
```bash
python data/fetch_telco.py --out data/telco_churn.csv
python src/train_models.py --input data/telco_churn.csv --outdir outputs --seed 42
```
