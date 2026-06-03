"""Load and clean the monkeypox dataset.

Run from the repository root:
    python src/data/make_dataset.py

Outputs:
    data/processed/monkeypox_clean.csv
    data/processed/monkeypox_model_ready.csv
    data/processed/train.csv
    data/processed/test.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/DATA.csv")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
TARGET_COLUMN = "monkeypox"
RANDOM_STATE = 42
TEST_SIZE = 0.20

BOOLEAN_COLUMNS = [
    "rectal_pain",
    "sore_throat",
    "penile_oedema",
    "oral_lesions",
    "solitary_lesion",
    "swollen_tonsils",
    "hiv_infection",
    "sexually_transmitted_infection",
]


def snake_case(name: str) -> str:
    """Convert a column/category name to stable snake_case."""
    name = str(name).strip().lower()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    return pd.read_csv(path)


def _map_boolean_column(series: pd.Series) -> pd.Series:
    """Map bool/string bool values to 0/1 integers."""
    mapping = {
        True: 1,
        False: 0,
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0,
    }
    mapped = series.map(mapping)
    if mapped.isna().any():
        bad_values = sorted(series[mapped.isna()].dropna().astype(str).unique())
        raise ValueError(f"Unexpected boolean values in {series.name}: {bad_values}")
    return mapped.astype("int8")


def clean_monkeypox_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw monkeypox dataset while preserving patient_id for traceability.

    The returned dataframe uses snake_case column names, encodes binary symptom
    fields as 0/1, fills missing systemic_illness values as a separate explicit
    category, and maps MonkeyPox labels to 0/1.
    """
    df = raw.copy()
    df.columns = [snake_case(col) for col in df.columns]

    required_columns = {"patient_id", "systemic_illness", TARGET_COLUMN, *BOOLEAN_COLUMNS}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["systemic_illness"] = (
        df["systemic_illness"]
        .fillna("none_reported_or_missing")
        .astype(str)
        .str.strip()
        .map(snake_case)
    )

    for col in BOOLEAN_COLUMNS:
        df[col] = _map_boolean_column(df[col])

    target_mapping = {"Negative": 0, "Positive": 1, "negative": 0, "positive": 1, 0: 0, 1: 1}
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(target_mapping)
    if df[TARGET_COLUMN].isna().any():
        bad_targets = sorted(df.loc[df[TARGET_COLUMN].isna(), TARGET_COLUMN].dropna().astype(str).unique())
        raise ValueError(f"Unexpected target labels: {bad_targets}")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")

    ordered_columns = ["patient_id", "systemic_illness", *BOOLEAN_COLUMNS, TARGET_COLUMN]
    return df[ordered_columns]


def make_model_ready(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Create a numeric, ML-ready dataframe.

    patient_id is dropped to avoid learning an identifier instead of clinical
    features. systemic_illness is one-hot encoded. The target is kept as the
    final column.
    """
    model_df = pd.get_dummies(
        clean_df.drop(columns=["patient_id"]),
        columns=["systemic_illness"],
        prefix="systemic_illness",
        dtype="int8",
    )
    feature_columns = [col for col in model_df.columns if col != TARGET_COLUMN]
    return model_df[feature_columns + [TARGET_COLUMN]]


def make_train_test_split(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible stratified train/test split."""
    train_df, test_df = train_test_split(
        model_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=model_df[TARGET_COLUMN],
    )
    return train_df, test_df


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.write_text("\n".join(values) + "\n")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw(RAW_PATH)
    clean_df = clean_monkeypox_dataset(raw_df)
    model_df = make_model_ready(clean_df)
    train_df, test_df = make_train_test_split(model_df)

    clean_df.to_csv(PROCESSED_DIR / "monkeypox_clean.csv", index=False)
    model_df.to_csv(PROCESSED_DIR / "monkeypox_model_ready.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    feature_columns = [col for col in model_df.columns if col != TARGET_COLUMN]
    write_lines(PROCESSED_DIR / "feature_columns.txt", feature_columns)
    write_lines(PROCESSED_DIR / "target_column.txt", [TARGET_COLUMN])

    profile = {
        "raw_shape": list(raw_df.shape),
        "clean_shape": list(clean_df.shape),
        "model_ready_shape": list(model_df.shape),
        "target_column": TARGET_COLUMN,
        "target_mapping": {"Negative": 0, "Positive": 1},
        "class_counts": clean_df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "missing_values_raw": raw_df.isna().sum().to_dict(),
        "duplicates_raw": int(raw_df.duplicated().sum()),
        "split": {
            "train_rows": int(train_df.shape[0]),
            "test_rows": int(test_df.shape[0]),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified_by": TARGET_COLUMN,
        },
    }
    (REPORTS_DIR / "data_profile.json").write_text(json.dumps(profile, indent=2))
    print("Dataset cleaning complete.")
    print(f"Clean data: {PROCESSED_DIR / 'monkeypox_clean.csv'}")
    print(f"Model-ready data: {PROCESSED_DIR / 'monkeypox_model_ready.csv'}")


if __name__ == "__main__":
    main()
