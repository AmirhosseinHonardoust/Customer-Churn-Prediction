<div align="center">
   
# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-green)
![Calibration](https://img.shields.io/badge/Calibration-Probability%20Quality-orange)
![Status](https://img.shields.io/badge/Status-Portfolio%20MVP-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Customer-Churn-Prediction/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Customer-Churn-Prediction/actions/workflows/ci.yml)

</div>

A production-minded customer-churn workflow for turning behavioural signals into **calibrated churn probabilities**, **PR-AUC model selection**, **recall-focused threshold tuning**, **interpretable importances**, and **batch churn scoring**.

> **Important:** This project is a **portfolio and research demo**, not a production churn or customer-retention decision system.
>
> The models, thresholds, and reports are designed to demonstrate a professional modelling workflow. They should not be used for real retention, pricing, or customer-facing decisions without domain, fairness, security, and privacy review.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training and Evaluation](#training-and-evaluation)
- [Use Real Data (IBM Telco)](#use-real-data-ibm-telco)
- [Prediction](#prediction)
- [Data Schema](#data-schema)
- [Evaluation Metrics](#evaluation-metrics)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Churn prediction is not only a classification problem. In a real retention workflow, a model score is useful only if it can support a defensible action:

- rank customers by churn risk
- tune a decision threshold to the business cost of a missed churner
- report probability quality, not only ranking quality
- explain which signals drive the score
- run the same pipeline on real data, reproducibly

This project demonstrates an end-to-end churn modelling workflow on both a synthetic dataset and the public IBM Telco churn dataset. It includes schema validation, model training and selection, threshold tuning, optional probability calibration, visual diagnostics, and a batch scoring CLI.

The goal is to show how a churn model can be turned into a **decision-support tool**, not just a single accuracy or AUC score.

---

## What This Project Does

This project can:

- Generate a synthetic churn dataset with a realistic behavioural signal
- Fetch and normalise the public IBM Telco churn dataset
- Validate input schema so any compatible CSV works (dataset-agnostic)
- Train and tune Logistic Regression, Random Forest, and Gradient Boosting
- Select the best model by validation PR-AUC (average precision)
- Tune the decision threshold for F2 with a minimum-precision floor
- Optionally isotonic-calibrate the selected model's probabilities
- Evaluate ROC-AUC, PR-AUC, Brier score, and a full classification report
- Produce ROC, precision-recall, confusion-matrix, calibration, and permutation-importance figures
- Save the trained pipeline and metrics artifacts
- Score new customers in batch with a prediction CLI
- Run automated tests and CI across Python 3.10-3.12

---

## What This Project Does Not Do

This project does **not**:

- Make real retention, pricing, or customer-facing decisions
- Provide business, legal, or financial advice
- Guarantee fairness, compliance, or deployability
- Use a production customer dataset by default
- Provide real-time scoring infrastructure or an API
- Include live monitoring, drift detection, or retraining automation
- Perform fairness or subgroup safety analysis
- Certify that the model is safe for high-stakes deployment

A production churn system would need stronger governance, privacy controls, monitoring, fairness review, and expert validation.

---

## Key Features

- **Dataset-agnostic pipeline** that runs on any CSV with a binary `churn` target
- **Synthetic data generator** with a reproducible, readable churn signal
- **Real-data loader** for the public IBM Telco churn dataset
- **PR-AUC model selection** across Logistic Regression, Random Forest, and Gradient Boosting
- **Threshold tuning** for F2 (recall-leaning) with a precision floor
- **Optional isotonic calibration** with a reliability diagram and Brier score
- **Permutation feature importance** for model-agnostic, comparable importances
- **Batch scoring CLI** that reuses the model's own feature schema and tuned threshold
- **Schema validation** that fails fast with a clear message on bad input
- **Model card** documenting intended use, provenance, and limitations
- **Unit tests and GitHub Actions CI** with coverage across Python 3.10-3.12

---

## System Workflow

```text
Customer dataset (synthetic or real)
        ↓
Schema validation
        ↓
Stratified train / validation / test split
        ↓
Preprocessing + model search (LogReg / RF / GB)
        ↓
Model selection by validation PR-AUC
        ↓
Threshold tuning (F2 with precision floor)
        ↓
Optional probability calibration
        ↓
Evaluation, figures, and saved artifacts
        ↓
Batch scoring via predict.py
```

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── generate_customers.py
│   ├── fetch_telco.py
│   └── customers.csv
│
├── src/
│   ├── train_models.py
│   ├── predict.py
│   └── utils.py
│
├── tests/
│   ├── test_generate.py
│   ├── test_utils.py
│   ├── test_threshold.py
│   ├── test_preprocessor.py
│   ├── test_predict.py
│   ├── test_calibration.py
│   └── test_fetch.py
│
├── outputs/            (generated; git-ignored)
│
├── reports/
│   └── figures/        (versioned snapshots shown in the README)
│
├── .gitignore
├── README.md
├── MODEL_CARD.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── LICENSE
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

For the development tools (linting, formatting, type checking, tests):

```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

No dataset on hand? Generate a synthetic one and run the whole pipeline with zero external data:

```bash
python data/generate_customers.py --n 10000 --seed 42 --out data/customers.csv
python src/train_models.py --input data/customers.csv --outdir outputs --seed 42
```

Prefer real data? Fetch and train on the public IBM Telco churn dataset:

```bash
python data/fetch_telco.py --out data/telco_churn.csv
python src/train_models.py --input data/telco_churn.csv --outdir outputs --seed 42
```

Score new customers with a trained model:

```bash
python src/predict.py --model outputs/best_model.joblib --input new_customers.csv --output outputs/predictions.csv
```

---

## Training and Evaluation

The main pipeline splits the data, searches and selects the best model, tunes the decision threshold, evaluates on a held-out test set, and writes figures and artifacts.

```bash
python src/train_models.py \
  --input data/customers.csv \
  --outdir outputs \
  --test-size 0.2 \
  --val-size 0.2 \
  --seed 42
```

Add `--calibrate` to isotonic-calibrate the selected model on the validation set before evaluation.

Generated evaluation outputs include:

```text
outputs/metrics.json
outputs/classification_report.txt
outputs/roc_curve.png
outputs/pr_curve.png
outputs/confusion_matrix.png
outputs/calibration_curve.png
outputs/feature_importance.png
outputs/best_model.joblib
```

Files under `outputs/` are regenerated by the command above and are git-ignored.

---

## Use Real Data (IBM Telco)

The pipeline is **dataset-agnostic** — any CSV with a binary `churn` target and at least one feature column works, and an optional `customer_id` is passed through. `fetch_telco.py` downloads the [IBM sample dataset](https://github.com/IBM/telco-customer-churn-on-icp4d), maps `customerID`→`customer_id` and `Churn`→`churn` (0/1), and coerces `TotalCharges` to numeric:

```bash
python data/fetch_telco.py --out data/telco_churn.csv
python src/train_models.py --input data/telco_churn.csv --outdir outputs --seed 42
```

The downloaded CSV is git-ignored.

---

## Prediction

Score a CSV of customers with a trained model. The prediction CLI selects the exact feature columns the model was trained on, so it works on any schema the model was trained with. The decision threshold defaults to the tuned value in `outputs/metrics.json` (override with `--threshold`), and `churn` is not required in the input:

```bash
python src/predict.py \
  --model outputs/best_model.joblib \
  --input new_customers.csv \
  --output outputs/predictions.csv
```

Output columns: `customer_id` (if present), `churn_proba`, `churn_pred`.

> **`new_customers.csv` is a placeholder** — supply your own file, or create a quick one from existing data:
>
> ```bash
> python -c "import pandas as pd; pd.read_csv('data/customers.csv').head(20).to_csv('new_customers.csv', index=False)"
> ```
>
> **The input columns must match the schema the model was trained on.** A model trained on the synthetic data (`age`, `region`, `tenure_months`, ...) cannot score a Telco file (`gender`, `tenure`, `MonthlyCharges`, ...) and vice versa — retrain on the matching dataset first if the columns differ.

---

## Data Schema

The synthetic generator and the reference schema use the following columns:

<div align="center">

| column | description |
|---|---|
| customer_id | unique customer ID |
| age | customer age |
| region | {North, South, East, West} |
| tenure_months | months since signup |
| is_premium | premium plan (0/1) |
| monthly_spend | average monthly spend |
| avg_txn_value | average transaction value |
| txns_last_30d | transactions in last 30 days |
| days_since_last_purchase | recency (days) |
| customer_service_calls | support calls in last 90 days |
| discounts_used_90d | discounts used in last 90 days |
| complaints_90d | complaint count |
| churn | target label (0/1) |

</div>

> Real datasets may use a completely different set of columns; only a binary `churn` target and at least one feature are required.

---

## Evaluation Metrics

The evaluation layer reports metrics designed for imbalanced churn decision workflows.

<div align="center">

| Metric | Why it matters |
|---|---|
| Accuracy | Overall decision correctness at the tuned threshold |
| F1 | Balance between churn-class precision and recall |
| ROC-AUC | Ranking quality across thresholds |
| Average precision / PR-AUC | Positive-class ranking quality under class imbalance |
| Brier score | Measures probability quality (calibration) |
| Recall (churn) | Share of real churners the policy catches |
| Precision (churn) | Correctness among predicted churners |

</div>

Example results on **synthetic** data (Logistic Regression, seed 42):

<div align="center">

| Metric | Example value |
|---|---|
| Accuracy | 0.838 |
| ROC-AUC | 0.823 |
| Average precision / PR-AUC | 0.562 |
| Recall (churn) | 0.50 |
| Precision (churn) | 0.52 |

</div>

Example results on **real IBM Telco** data (Gradient Boosting, seed 42):

<div align="center">

| Metric | Example value |
|---|---|
| Accuracy | 0.756 |
| ROC-AUC | 0.847 |
| Average precision / PR-AUC | 0.666 |
| Recall (churn) | 0.77 |
| Precision (churn) | 0.53 |
| Brier score | 0.135 |

</div>

> These values are from demo datasets and should not be interpreted as real-world retention performance.

---

## Visual Reports

The figures below are from the **IBM Telco** run (Gradient Boosting, seed 42, threshold 0.29) and are stored in `reports/figures/`.

### Model evaluation charts

<div align="center">

| ROC Curve | Precision-Recall Curve |
|---|---|
| <img width="1280" height="960" alt="roc_curve" src="https://github.com/user-attachments/assets/861568e1-3eb4-45d0-aa1f-8b0f51640dc1" /> | <img width="1280" height="960" alt="pr_curve" src="https://github.com/user-attachments/assets/b0b2efd3-3e68-4d57-b520-04a9ed55a1cf" /> |
| **Analysis:** The model reaches ROC-AUC 0.847 and climbs steeply at low false-positive rates, so the highest-risk customers are ranked well above the rest. | **Analysis:** With churners ~27% of the data, PR-AUC (AP 0.666) is the primary selection metric; precision stays around 0.75-0.85 through the first third of recall before trading off. |

</div>

### Probability quality and decisions

<div align="center">

| Calibration Curve | Confusion Matrix |
|---|---|
| <img width="1280" height="960" alt="calibration_curve" src="https://github.com/user-attachments/assets/3685feb5-846a-42da-a217-0a67e3eb4457" /> | <img width="1120" height="1120" alt="confusion_matrix" src="https://github.com/user-attachments/assets/6e0e690a-8b84-4b3a-bc86-47e5e4b5dcda" /> |
| **Analysis:** Predicted probabilities track the diagonal closely (Brier 0.135), so a score of 0.6 means roughly a 60% observed churn rate — the scores are usable as probabilities, not just rankings. | **Analysis:** At the F2-tuned threshold the model catches 289 of 374 churners (recall ~0.77) while flagging 259 non-churners — the deliberate recall-leaning trade-off for retention outreach. |

</div>

<details>
<summary>Feature importance</summary>
        
### Feature importance

<div align="center">

<img width="1600" height="960" alt="feature_importance" src="https://github.com/user-attachments/assets/a084f599-be22-49e0-9976-08de6e17a97c" />

</div>

> **Analysis:** Permutation importance (drop in PR-AUC) makes numeric and categorical signals comparable on one axis. `tenure`, `Contract`, and `InternetService` dominate, matching the intuition that newer, month-to-month customers churn most. All figures are regenerated on every run into `outputs/`; the copies in `reports/figures/` are the versioned snapshots shown here.

</details>

---

## Testing and CI

Run the test suite locally:

```bash
pytest
```

Run the quality gate locally (matches CI):

```bash
ruff check .
black --check .
mypy src data
pytest --cov=src --cov=data --cov-report=term-missing
```

The GitHub Actions workflow checks:

- linting (ruff, selecting E, F, I, B, SIM, UP)
- formatting (black, line length 100)
- type checking (mypy)
- unit tests with coverage
- across a Python 3.10 / 3.11 / 3.12 matrix

CI is defined in:

```text
.github/workflows/ci.yml
```

---

## Code Quality

The project separates major responsibilities across modules:

<div align="center">

| Module | Purpose |
|---|---|
| `data/generate_customers.py` | Generates a synthetic churn dataset with a reproducible signal |
| `data/fetch_telco.py` | Downloads and normalises the IBM Telco churn dataset |
| `src/utils.py` | Loads data, validates schema, splits, and extracts probabilities |
| `src/train_models.py` | Builds the pipeline, searches models, tunes threshold, calibrates, and writes outputs |
| `src/predict.py` | Loads a trained model and scores new customers in batch |

</div>

---

## Limitations

This project has important limitations:

- The default dataset is synthetic, not a production customer dataset
- The project is not a retention or customer-decision engine
- Metrics do not prove real-world churn performance
- No fairness or subgroup safety analysis is included
- No real-time scoring service or API is included
- No live monitoring, drift detection, or retraining is included
- Probabilities are only calibrated when `--calibrate` is used
- Permutation importance reflects association, not causation

The project is strongest as a portfolio demonstration of a calibrated, dataset-agnostic modelling workflow.

---

## Responsible Use

This repository is intended for:

- learning about churn modelling and calibration
- demonstrating a reproducible modelling workflow
- practicing model selection and threshold tuning
- exploring probability-quality and interpretability
- portfolio demonstration

It should not be used as-is for:

- real retention, pricing, or eligibility decisions
- automated customer-facing workflows
- high-stakes financial decisions
- any deployment without fairness, privacy, and security review

Any real deployment would require expert review, monitoring, fairness analysis, privacy controls, and a human escalation process.

---

## Future Improvements

Potential next improvements:

- Add fairness metrics and group-specific calibration summaries
- Add a FastAPI scoring endpoint
- Add Docker support
- Add drift simulation and monitoring examples
- Add time-based validation splits
- Add SHAP-based local explanations
- Add a model registry-style metadata artifact
- Add configurable cost assumptions for threshold selection
- Add confidence intervals for reported metrics

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- joblib
- pytest
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is released under the terms of the [MIT License](LICENSE).

If you use or modify this project, please keep the responsible-use notes and limitations clear.
