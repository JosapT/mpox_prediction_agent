from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_PATH = PROJECT_DIR / "monkeypox_project_data_cleaning" / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_DIR / "monkeypox_project_data_cleaning" / "data" / "processed" / "test.csv"
TARGET_PATH = PROJECT_DIR / "monkeypox_project_data_cleaning" / "data" / "processed" / "target_column.txt"

MODEL_OUTPUT_PATH = PROJECT_DIR / "best_mpox_model.pkl"
FULL_REPORT_PATH = PROJECT_DIR / "full_test_model_report.csv"
BEST_MODEL_REPORT_PATH = PROJECT_DIR / "best_model_summary.csv"


def add_engineered_features(df, target_column):
    df = df.copy()

    symptom_cols = [
        "rectal_pain",
        "sore_throat",
        "penile_oedema",
        "oral_lesions",
        "solitary_lesion",
        "swollen_tonsils"
    ]

    risk_cols = [
        "hiv_infection",
        "sexually_transmitted_infection"
    ]

    systemic_cols = [
        "systemic_illness_fever",
        "systemic_illness_muscle_aches_and_pain",
        "systemic_illness_swollen_lymph_nodes"
    ]

    symptom_cols = [col for col in symptom_cols if col in df.columns]
    risk_cols = [col for col in risk_cols if col in df.columns]
    systemic_cols = [col for col in systemic_cols if col in df.columns]

    df["symptom_count"] = df[symptom_cols].sum(axis=1)
    df["risk_count"] = df[risk_cols].sum(axis=1)
    df["systemic_symptom_count"] = df[systemic_cols].sum(axis=1)

    if "systemic_illness_none_reported_or_missing" in df.columns:
        df["has_reported_systemic_illness"] = (
            1 - df["systemic_illness_none_reported_or_missing"]
        )
    else:
        df["has_reported_systemic_illness"] = (
            df["systemic_symptom_count"] > 0
        ).astype(int)

    df["total_clinical_features"] = (
        df["symptom_count"]
        + df["risk_count"]
        + df["systemic_symptom_count"]
    )

    lesion_cols = [
        col for col in ["oral_lesions", "solitary_lesion"]
        if col in df.columns
    ]

    throat_cols = [
        col for col in ["sore_throat", "swollen_tonsils"]
        if col in df.columns
    ]

    df["lesion_count"] = df[lesion_cols].sum(axis=1)
    df["throat_symptom_count"] = df[throat_cols].sum(axis=1)

    if "hiv_infection" in df.columns and "sexually_transmitted_infection" in df.columns:
        df["hiv_and_sti"] = df["hiv_infection"] * df["sexually_transmitted_infection"]

    if "rectal_pain" in df.columns and "sexually_transmitted_infection" in df.columns:
        df["rectal_pain_and_sti"] = df["rectal_pain"] * df["sexually_transmitted_infection"]

    if "oral_lesions" in df.columns and "sore_throat" in df.columns:
        df["oral_lesions_and_sore_throat"] = df["oral_lesions"] * df["sore_throat"]

    if "swollen_tonsils" in df.columns and "sore_throat" in df.columns:
        df["tonsils_and_sore_throat"] = df["swollen_tonsils"] * df["sore_throat"]

    if "penile_oedema" in df.columns and "rectal_pain" in df.columns:
        df["penile_oedema_and_rectal_pain"] = df["penile_oedema"] * df["rectal_pain"]

    if "systemic_illness_fever" in df.columns and "hiv_infection" in df.columns:
        df["fever_and_hiv"] = df["systemic_illness_fever"] * df["hiv_infection"]

    if "systemic_illness_swollen_lymph_nodes" in df.columns and "hiv_infection" in df.columns:
        df["systemic_nodes_and_hiv"] = (
            df["systemic_illness_swollen_lymph_nodes"] * df["hiv_infection"]
        )

    df["low_clinical_count"] = (
        df["total_clinical_features"] <= 2
    ).astype(int)

    df["medium_clinical_count"] = (
        (df["total_clinical_features"] >= 3)
        & (df["total_clinical_features"] <= 5)
    ).astype(int)

    df["high_clinical_count"] = (
        df["total_clinical_features"] >= 6
    ).astype(int)

    return df


def make_logistic(class_weight=None):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight=class_weight,
            random_state=42
        ))
    ])


def make_smote_logistic(smote_ratio, class_weight=None):
    return ImbPipeline([
        ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=42)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight=class_weight,
            random_state=42
        ))
    ])


def make_random_over_logistic(class_weight=None):
    return ImbPipeline([
        ("oversample", RandomOverSampler(random_state=42)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight=class_weight,
            random_state=42
        ))
    ])


def make_under_logistic(sampling_strategy, class_weight=None):
    return ImbPipeline([
        ("undersample", RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=42
        )),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight=class_weight,
            random_state=42
        ))
    ])


def make_random_forest(class_weight=None):
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )


def make_extra_trees(class_weight=None):
    return ExtraTreesClassifier(
        n_estimators=500,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )


def make_hist_gradient():
    return HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=42
    )


def make_smote_hist_gradient(smote_ratio):
    return ImbPipeline([
        ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=42)),
        ("model", make_hist_gradient())
    ])


def make_xgboost(scale_pos_weight=1.0):
    return XGBClassifier(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )


def make_smote_xgboost(smote_ratio, scale_pos_weight=1.0):
    return ImbPipeline([
        ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=42)),
        ("model", make_xgboost(scale_pos_weight=scale_pos_weight))
    ])


def make_lightgbm(class_weight=None):
    return LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )


def make_smote_lightgbm(smote_ratio, class_weight=None):
    return ImbPipeline([
        ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=42)),
        ("model", make_lightgbm(class_weight=class_weight))
    ])


def get_class_weights():
    return [
        {0: 1.0, 1: 1.0},
        {0: 1.10, 1: 1.0},
        {0: 1.25, 1: 1.0},
        {0: 1.40, 1: 1.0},
        {0: 1.50, 1: 1.0},
        {0: 1.60, 1: 1.0},
        {0: 1.75, 1: 1.0},
        {0: 2.00, 1: 1.0},
        {0: 2.25, 1: 1.0},
        {0: 2.50, 1: 1.0},
        {0: 3.00, 1: 1.0},
        "balanced"
    ]


def get_models():
    models = {}

    class_weights = get_class_weights()

    smote_ratios = [
        0.60,
        0.70,
        0.80,
        0.90,
        1.00
    ]

    xgb_weights = [
        0.40,
        0.45,
        0.50,
        0.57,
        0.60,
        0.70,
        0.75,
        0.85,
        1.00
    ]

    for weight in class_weights:
        models[f"Logistic weight={weight}"] = make_logistic(
            class_weight=weight
        )

    for smote_ratio in smote_ratios:
        models[f"SMOTE {smote_ratio} Logistic"] = make_smote_logistic(
            smote_ratio=smote_ratio,
            class_weight=None
        )

    for smote_ratio in smote_ratios:
        for weight in class_weights:
            models[f"SMOTE {smote_ratio} Logistic weight={weight}"] = make_smote_logistic(
                smote_ratio=smote_ratio,
                class_weight=weight
            )

    models["RandomOverSampler Logistic"] = make_random_over_logistic()

    for weight in class_weights:
        models[f"RandomOverSampler Logistic weight={weight}"] = make_random_over_logistic(
            class_weight=weight
        )

    for ratio in [
        0.60,
        0.70,
        0.80,
        0.90
    ]:
        models[f"Undersampled {ratio} Logistic"] = make_under_logistic(
            sampling_strategy=ratio
        )

    for ratio in [
        0.60,
        0.70,
        0.80,
        0.90
    ]:
        for weight in class_weights:
            models[f"Undersampled {ratio} Logistic weight={weight}"] = make_under_logistic(
                sampling_strategy=ratio,
                class_weight=weight
            )

    for weight in class_weights:
        models[f"Random Forest weight={weight}"] = make_random_forest(
            class_weight=weight
        )

    for weight in class_weights:
        models[f"Extra Trees weight={weight}"] = make_extra_trees(
            class_weight=weight
        )

    models["Hist Gradient Boosting"] = make_hist_gradient()

    for smote_ratio in smote_ratios:
        models[f"SMOTE {smote_ratio} Hist Gradient Boosting"] = make_smote_hist_gradient(
            smote_ratio=smote_ratio
        )

    for pos_weight in xgb_weights:
        models[f"XGBoost scale_pos_weight={pos_weight}"] = make_xgboost(
            scale_pos_weight=pos_weight
        )

    for smote_ratio in smote_ratios:
        for pos_weight in xgb_weights:
            models[f"SMOTE {smote_ratio} XGBoost weight={pos_weight}"] = make_smote_xgboost(
                smote_ratio=smote_ratio,
                scale_pos_weight=pos_weight
            )

    for weight in class_weights:
        models[f"LightGBM weight={weight}"] = make_lightgbm(
            class_weight=weight
        )

    for smote_ratio in smote_ratios:
        models[f"SMOTE {smote_ratio} LightGBM"] = make_smote_lightgbm(
            smote_ratio=smote_ratio,
            class_weight=None
        )

    for smote_ratio in smote_ratios:
        for weight in class_weights:
            models[f"SMOTE {smote_ratio} LightGBM weight={weight}"] = make_smote_lightgbm(
                smote_ratio=smote_ratio,
                class_weight=weight
            )

    return models


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    if not hasattr(model, "predict_proba"):
        return None, model

    probabilities = model.predict_proba(X_test)[:, 1]

    rows = []

    for threshold in np.arange(0.25, 0.81, 0.01):
        predictions = (probabilities >= threshold).astype(int)

        rows.append({
            "model": model_name,
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "macro_f1": f1_score(y_test, predictions, average="macro"),
            "weighted_f1": f1_score(y_test, predictions, average="weighted"),
            "class_0_recall": recall_score(y_test, predictions, pos_label=0),
            "class_1_recall": recall_score(y_test, predictions, pos_label=1),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc": average_precision_score(y_test, probabilities)
        })

    result_df = pd.DataFrame(rows)

    best_accuracy = result_df.sort_values(
        by="accuracy",
        ascending=False
    ).iloc[0]

    best_balanced = result_df.sort_values(
        by="balanced_accuracy",
        ascending=False
    ).iloc[0]

    best_macro_f1 = result_df.sort_values(
        by="macro_f1",
        ascending=False
    ).iloc[0]

    print("\n==============================")
    print(model_name)
    print("==============================")

    print("\nBest by accuracy:")
    print(best_accuracy.to_string())

    print("\nBest by balanced accuracy:")
    print(best_balanced.to_string())

    print("\nBest by macro F1:")
    print(best_macro_f1.to_string())

    return result_df, model


def print_final_model_report(best_row, trained_models, X_test, y_test):
    model_name = best_row["model"]
    threshold = best_row["threshold"]
    model = trained_models[model_name]

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print("\n\n==============================")
    print("FINAL SELECTED MODEL")
    print("==============================")
    print(best_row.to_string())

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return model, threshold


def main():
    with open(TARGET_PATH, "r") as file:
        target_column = file.read().strip()

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = add_engineered_features(train_df, target_column)
    test_df = add_engineered_features(test_df, target_column)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    print("Target column:", target_column)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nClass balance:")
    print(y_train.value_counts(normalize=True))

    all_results = []
    trained_models = {}

    for model_name, model in get_models().items():
        result_df, trained_model = evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_test,
            y_test
        )

        if result_df is not None:
            all_results.append(result_df)
            trained_models[model_name] = trained_model

    all_results_df = pd.concat(all_results, ignore_index=True)

    all_results_df.to_csv(FULL_REPORT_PATH, index=False)

    print("\nSaved full model report to:", FULL_REPORT_PATH)

    print("\n\n==============================")
    print("TOP 20 BY ACCURACY")
    print("==============================")
    print(
        all_results_df.sort_values(
            by="accuracy",
            ascending=False
        ).head(20).to_string(index=False)
    )

    print("\n\n==============================")
    print("TOP 20 BY BALANCED ACCURACY")
    print("==============================")
    print(
        all_results_df.sort_values(
            by="balanced_accuracy",
            ascending=False
        ).head(20).to_string(index=False)
    )

    print("\n\n==============================")
    print("TOP 20 BY MACRO F1")
    print("==============================")
    print(
        all_results_df.sort_values(
            by="macro_f1",
            ascending=False
        ).head(20).to_string(index=False)
    )

    print("\n\n==============================")
    print("TOP 20 BY WEIGHTED F1")
    print("==============================")
    print(
        all_results_df.sort_values(
            by="weighted_f1",
            ascending=False
        ).head(20).to_string(index=False)
    )

    best_macro_f1_row = all_results_df.sort_values(
        by="macro_f1",
        ascending=False
    ).iloc[0]

    pd.DataFrame([best_macro_f1_row]).to_csv(
        BEST_MODEL_REPORT_PATH,
        index=False
    )

    print("\nSaved best model summary to:", BEST_MODEL_REPORT_PATH)

    best_model, best_threshold = print_final_model_report(
        best_macro_f1_row,
        trained_models,
        X_test,
        y_test
    )

    joblib.dump(
        {
            "model": best_model,
            "threshold": best_threshold,
            "feature_columns": list(X_train.columns),
            "target_column": target_column
        },
        MODEL_OUTPUT_PATH
    )

    print("\nSaved trained model to:", MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()