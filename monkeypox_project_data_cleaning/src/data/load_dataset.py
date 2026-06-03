"""Convenience loaders for modeling notebooks and scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_MODEL_READY_PATH = Path("data/processed/monkeypox_model_ready.csv")
DEFAULT_TRAIN_PATH = Path("data/processed/train.csv")
DEFAULT_TEST_PATH = Path("data/processed/test.csv")
TARGET_COLUMN = "monkeypox"


def load_model_ready(path: str | Path = DEFAULT_MODEL_READY_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return X, y from the numeric model-ready dataset."""
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def load_train_test(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return X_train, y_train, X_test, y_test from saved split CSVs."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN]
    X_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN]
    return X_train, y_train, X_test, y_test
