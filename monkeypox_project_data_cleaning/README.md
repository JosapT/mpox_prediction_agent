# Monkeypox Prediction Data Cleaning Starter Repo

> Important: this project is for learning and exploratory modeling only. It should not be used as a medical diagnostic or public-health decision tool without clinical validation, bias checks, and review by qualified experts.

## Dataset summary

- Raw file: `data/raw/DATA.csv`
- Rows: 25,000
- Raw columns: 11
- Target: `monkeypox`
  - `0` = negative
  - `1` = positive
- Class balance:
  - Negative: 9,091 rows, about 36.36%
  - Positive: 15,909 rows, about 63.64%
- Missing values in raw data:
  - `Systemic Illness`: 6,216 missing values
  - Other columns: no missing values found
- Duplicate rows found: 0

## What was cleaned

The cleaning script does the following:

1. Converts column names to `snake_case`.
2. Converts symptom/risk-factor columns from `True`/`False` to `1`/`0`.
3. Converts the target column from `Positive`/`Negative` to `1`/`0`.
4. Fills missing `systemic_illness` values with the explicit category `none_reported_or_missing`.
5. Drops `patient_id` from the model-ready file so models do not learn from an identifier.
6. One-hot encodes `systemic_illness` in the model-ready dataset.
7. Creates a reproducible 80/20 train/test split stratified by the target.

## Folder structure

```text
monkeypox_project_data_cleaning/
├── data/
│   ├── raw/
│   │   └── DATA.csv
│   └── processed/
│       ├── monkeypox_clean.csv
│       ├── monkeypox_model_ready.csv
│       ├── train.csv
│       ├── test.csv
│       ├── feature_columns.txt
│       └── target_column.txt
├── reports/
│   └── data_profile.json
├── src/
│   └── data/
│       ├── make_dataset.py
│       └── load_dataset.py
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Rebuild the processed data

```bash
python src/data/make_dataset.py
```

## Load data in a notebook or script

```python
from src.data.load_dataset import load_train_test

X_train, y_train, X_test, y_test = load_train_test()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
```

## Recommended next steps

- Try multiple model types, such as logistic regression, random forest, gradient boosting, and calibrated classifiers.
- Use cross-validation instead of relying only on the provided train/test split.
- Track recall, precision, F1, ROC-AUC, and PR-AUC, not just accuracy.
- Consider false-negative cost carefully because this is a health-related prediction task.
- Keep `patient_id` out of model features.
