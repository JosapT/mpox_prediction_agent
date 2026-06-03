from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_PATH = PROJECT_DIR / "monkeypox_project_data_cleaning" / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_DIR / "monkeypox_project_data_cleaning" / "data" / "processed" / "test.csv"
TARGET_PATH = PROJECT_DIR / "monkeypox_project_data_cleaning" / "data" / "processed" / "target_column.txt"


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

    df["low_clinical_count"] = (df["total_clinical_features"] <= 2).astype(int)

    df["medium_clinical_count"] = (
        (df["total_clinical_features"] >= 3)
        & (df["total_clinical_features"] <= 5)
    ).astype(int)

    df["high_clinical_count"] = (df["total_clinical_features"] >= 6).astype(int)

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


def make_xgboost(scale_pos_weight=None):
    params = {
        "n_estimators": 700,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1
    }

    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    return XGBClassifier(**params)


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


def make_neural_network():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            random_state=42
        ))
    ])


def get_models():
    return {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),

        "Logistic Regression": make_logistic(),

        "Balanced Logistic Regression": make_logistic(class_weight="balanced"),

        "SMOTE 0.60 Logistic Regression": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.60, random_state=42)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42))
        ]),

        "SMOTE 0.70 Logistic Regression": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.70, random_state=42)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42))
        ]),

        "SMOTE 0.80 Logistic Regression": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.80, random_state=42)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42))
        ]),

        "SMOTE Full Logistic Regression": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=1.00, random_state=42)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=700,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "Balanced Random Forest": RandomForestClassifier(
            n_estimators=700,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=700,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "Balanced Extra Trees": ExtraTreesClassifier(
            n_estimators=700,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=42
        ),

        "SMOTE 0.60 Hist Gradient Boosting": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.60, random_state=42)),
            ("model", HistGradientBoostingClassifier(
                max_iter=600,
                learning_rate=0.03,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=42
            ))
        ]),

        "SMOTE 0.70 Hist Gradient Boosting": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.70, random_state=42)),
            ("model", HistGradientBoostingClassifier(
                max_iter=600,
                learning_rate=0.03,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=42
            ))
        ]),

        "SMOTE 0.80 Hist Gradient Boosting": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.80, random_state=42)),
            ("model", HistGradientBoostingClassifier(
                max_iter=600,
                learning_rate=0.03,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=42
            ))
        ]),

        "SMOTE Full Hist Gradient Boosting": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=1.00, random_state=42)),
            ("model", HistGradientBoostingClassifier(
                max_iter=600,
                learning_rate=0.03,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=42
            ))
        ]),

        "Neural Network": make_neural_network(),

        "SMOTE 0.60 Neural Network": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.60, random_state=42)),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                random_state=42
            ))
        ]),

        "SMOTE 0.70 Neural Network": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.70, random_state=42)),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                random_state=42
            ))
        ]),

        "SMOTE Full Neural Network": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=1.00, random_state=42)),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                random_state=42
            ))
        ]),

        "XGBoost": make_xgboost(),

        "XGBoost Weight 0.45": make_xgboost(scale_pos_weight=0.45),

        "XGBoost Weight 0.57": make_xgboost(scale_pos_weight=0.57),

        "XGBoost Weight 0.60": make_xgboost(scale_pos_weight=0.60),

        "XGBoost Weight 0.75": make_xgboost(scale_pos_weight=0.75),

        "SMOTE 0.60 XGBoost": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.60, random_state=42)),
            ("model", make_xgboost())
        ]),

        "SMOTE 0.70 XGBoost": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.70, random_state=42)),
            ("model", make_xgboost())
        ]),

        "SMOTE 0.80 XGBoost": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.80, random_state=42)),
            ("model", make_xgboost())
        ]),

        "SMOTE Full XGBoost": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=1.00, random_state=42)),
            ("model", make_xgboost())
        ]),

        "LightGBM": make_lightgbm(),

        "LightGBM Balanced": make_lightgbm(class_weight="balanced"),

        "SMOTE 0.60 LightGBM": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.60, random_state=42)),
            ("model", make_lightgbm())
        ]),

        "SMOTE 0.70 LightGBM": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.70, random_state=42)),
            ("model", make_lightgbm())
        ]),

        "SMOTE 0.80 LightGBM": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.80, random_state=42)),
            ("model", make_lightgbm())
        ]),

        "SMOTE Full LightGBM": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=1.00, random_state=42)),
            ("model", make_lightgbm())
        ]),

        "Undersampled Logistic Regression": ImbPipeline([
            ("undersample", RandomUnderSampler(sampling_strategy=0.70, random_state=42)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42))
        ]),

        "Undersampled XGBoost": ImbPipeline([
            ("undersample", RandomUnderSampler(sampling_strategy=0.70, random_state=42)),
            ("model", make_xgboost())
        ]),

        "Undersampled LightGBM": ImbPipeline([
            ("undersample", RandomUnderSampler(sampling_strategy=0.70, random_state=42)),
            ("model", make_lightgbm())
        ])
    }


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")
    class_0_recall = recall_score(y_test, predictions, pos_label=0)
    class_1_recall = recall_score(y_test, predictions, pos_label=1)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probabilities)
        pr_auc = average_precision_score(y_test, probabilities)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    print("\n==============================")
    print(model_name)
    print("==============================")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Class 0 recall: {class_0_recall:.4f}")
    print(f"Class 1 recall: {class_1_recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_0_recall": class_0_recall,
        "class_1_recall": class_1_recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }, model


def tune_threshold(model, X_test, y_test):
    if not hasattr(model, "predict_proba"):
        return

    probabilities = model.predict_proba(X_test)[:, 1]
    rows = []

    for threshold in np.arange(0.05, 0.96, 0.01):
        predictions = (probabilities >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "macro_f1": f1_score(y_test, predictions, average="macro"),
            "class_0_recall": recall_score(y_test, predictions, pos_label=0),
            "class_1_recall": recall_score(y_test, predictions, pos_label=1)
        })

    threshold_df = pd.DataFrame(rows)

    print("\n==============================")
    print("BEST THRESHOLD BY ACCURACY")
    print("==============================")
    print(
        threshold_df.sort_values(
            by="accuracy",
            ascending=False
        ).head(10).to_string(index=False)
    )

    print("\n==============================")
    print("BEST THRESHOLD BY BALANCED ACCURACY")
    print("==============================")
    print(
        threshold_df.sort_values(
            by="balanced_accuracy",
            ascending=False
        ).head(10).to_string(index=False)
    )

    print("\n==============================")
    print("BEST THRESHOLD BY MACRO F1")
    print("==============================")
    print(
        threshold_df.sort_values(
            by="macro_f1",
            ascending=False
        ).head(10).to_string(index=False)
    )


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

    results = []
    trained_models = {}

    for model_name, model in get_models().items():
        result, trained_model = evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_test,
            y_test
        )

        results.append(result)
        trained_models[model_name] = trained_model

    results_df = pd.DataFrame(results)

    accuracy_df = results_df.sort_values(by="accuracy", ascending=False)
    balanced_df = results_df.sort_values(by="balanced_accuracy", ascending=False)
    macro_f1_df = results_df.sort_values(by="macro_f1", ascending=False)

    print("\n==============================")
    print("MODEL COMPARISON BY ACCURACY")
    print("==============================")
    print(accuracy_df.to_string(index=False))

    print("\n==============================")
    print("MODEL COMPARISON BY BALANCED ACCURACY")
    print("==============================")
    print(balanced_df.to_string(index=False))

    print("\n==============================")
    print("MODEL COMPARISON BY MACRO F1")
    print("==============================")
    print(macro_f1_df.to_string(index=False))

    best_accuracy_model_name = accuracy_df.iloc[0]["model"]
    best_balanced_model_name = balanced_df.iloc[0]["model"]
    best_macro_f1_model_name = macro_f1_df.iloc[0]["model"]

    print("\nBest raw accuracy model:", best_accuracy_model_name)
    print("Best balanced accuracy model:", best_balanced_model_name)
    print("Best macro F1 model:", best_macro_f1_model_name)

    tune_threshold(
        trained_models[best_accuracy_model_name],
        X_test,
        y_test
    )


if __name__ == "__main__":
    main()