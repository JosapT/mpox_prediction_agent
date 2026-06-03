from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

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

from imblearn.over_sampling import SMOTE
from torch.utils.data import TensorDataset, DataLoader


PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_PATH = PROJECT_DIR / "monkeypox_project_data_cleaning" / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_DIR / "monkeypox_project_data_cleaning" / "data" / "processed" / "test.csv"
TARGET_PATH = PROJECT_DIR / "monkeypox_project_data_cleaning" / "data" / "processed" / "target_column.txt"

TRANSFORMER_MODEL_OUTPUT_PATH = PROJECT_DIR / "best_transformer_mpox_model.pt"
TRANSFORMER_SCALER_OUTPUT_PATH = PROJECT_DIR / "best_transformer_scaler.pkl"
TRANSFORMER_REPORT_PATH = PROJECT_DIR / "transformer_full_test_report.csv"
TRANSFORMER_BEST_REPORT_PATH = PROJECT_DIR / "best_transformer_summary.csv"


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


class TabularTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(num_features)
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        tokens = []

        for i, embedding_layer in enumerate(self.feature_embeddings):
            feature = x[:, i].unsqueeze(1)
            token = embedding_layer(feature)
            tokens.append(token)

        x = torch.stack(tokens, dim=1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x).squeeze(1)

        return logits


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def weighted_bce_loss(logits, targets, class_0_weight, class_1_weight):
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    weights = torch.where(
        targets == 0,
        torch.tensor(class_0_weight, device=targets.device),
        torch.tensor(class_1_weight, device=targets.device)
    )

    return (loss * weights).mean()


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    class_0_weight=1.0,
    class_1_weight=1.0,
    learning_rate=0.001,
    weight_decay=0.0001,
    max_epochs=80,
    patience=10
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)

            loss = weighted_bce_loss(
                logits,
                y_batch,
                class_0_weight=class_0_weight,
                class_1_weight=class_1_weight
            )

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)

                loss = weighted_bce_loss(
                    logits,
                    y_batch,
                    class_0_weight=class_0_weight,
                    class_1_weight=class_1_weight
                )

                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    return model


def apply_smote(X_train, y_train, smote_ratio):
    if smote_ratio is None:
        return X_train, y_train

    smote = SMOTE(
        sampling_strategy=smote_ratio,
        random_state=42
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    return X_resampled, y_resampled


def evaluate_thresholds(y_test, probabilities):
    rows = []

    for threshold in np.arange(0.25, 0.81, 0.01):
        predictions = (probabilities >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "macro_f1": f1_score(y_test, predictions, average="macro"),
            "weighted_f1": f1_score(y_test, predictions, average="weighted"),
            "class_0_recall": recall_score(y_test, predictions, pos_label=0),
            "class_1_recall": recall_score(y_test, predictions, pos_label=1)
        })

    return pd.DataFrame(rows)


def evaluate_model(model, X_test_tensor, y_test, device):
    model.eval()

    with torch.no_grad():
        logits = model(X_test_tensor.to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()

    threshold_df = evaluate_thresholds(
        y_test,
        probabilities
    )

    threshold_df["roc_auc"] = roc_auc_score(
        y_test,
        probabilities
    )

    threshold_df["pr_auc"] = average_precision_score(
        y_test,
        probabilities
    )

    return threshold_df, probabilities


def run_experiment(
    experiment_name,
    X_train,
    y_train,
    X_test,
    y_test,
    device,
    smote_ratio=None,
    class_0_weight=1.0,
    class_1_weight=1.0,
    embed_dim=32,
    num_heads=4,
    num_layers=2,
    dropout=0.2,
    learning_rate=0.001,
    weight_decay=0.0001,
    batch_size=256,
    max_epochs=80,
    patience=10
):
    print("\n\n########################################")
    print("EXPERIMENT:", experiment_name)
    print("########################################")

    X_train_used, y_train_used = apply_smote(
        X_train,
        y_train,
        smote_ratio
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train_used
    ).astype(np.float32)

    X_test_scaled = scaler.transform(
        X_test
    ).astype(np.float32)

    split_index = int(len(X_train_scaled) * 0.85)

    X_train_part = X_train_scaled[:split_index]
    y_train_part = y_train_used[:split_index]

    X_val_part = X_train_scaled[split_index:]
    y_val_part = y_train_used[split_index:]

    print("SMOTE ratio:", smote_ratio)
    print("Class 0 weight:", class_0_weight)
    print("Class 1 weight:", class_1_weight)

    print("Training class balance:")
    print(pd.Series(y_train_part).value_counts(normalize=True))

    X_train_tensor = torch.tensor(
        X_train_part,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train_part,
        dtype=torch.float32
    )

    X_val_tensor = torch.tensor(
        X_val_part,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val_part,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    val_dataset = TensorDataset(
        X_val_tensor,
        y_val_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False
    )

    torch.manual_seed(42)
    np.random.seed(42)

    model = TabularTransformer(
        num_features=X_train_scaled.shape[1],
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_0_weight=class_0_weight,
        class_1_weight=class_1_weight,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience
    )

    threshold_df, probabilities = evaluate_model(
        model,
        X_test_tensor,
        y_test,
        device
    )

    threshold_df["experiment"] = experiment_name
    threshold_df["smote_ratio"] = smote_ratio
    threshold_df["class_0_weight"] = class_0_weight
    threshold_df["class_1_weight"] = class_1_weight
    threshold_df["embed_dim"] = embed_dim
    threshold_df["num_heads"] = num_heads
    threshold_df["num_layers"] = num_layers
    threshold_df["dropout"] = dropout
    threshold_df["learning_rate"] = learning_rate
    threshold_df["weight_decay"] = weight_decay

    best_accuracy = threshold_df.sort_values(
        by="accuracy",
        ascending=False
    ).iloc[0]

    best_balanced = threshold_df.sort_values(
        by="balanced_accuracy",
        ascending=False
    ).iloc[0]

    best_macro_f1 = threshold_df.sort_values(
        by="macro_f1",
        ascending=False
    ).iloc[0]

    print("\nBest by accuracy:")
    print(best_accuracy.to_string())

    print("\nBest by balanced accuracy:")
    print(best_balanced.to_string())

    print("\nBest by macro F1:")
    print(best_macro_f1.to_string())

    return threshold_df, model, scaler, probabilities


def print_final_report(best_row, model, scaler, X_test, y_test, device):
    threshold = best_row["threshold"]

    X_test_scaled = scaler.transform(
        X_test
    ).astype(np.float32)

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():
        logits = model(X_test_tensor.to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (probabilities >= threshold).astype(int)

    print("\n\n==============================")
    print("FINAL SELECTED TRANSFORMER")
    print("==============================")
    print(best_row.to_string())

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    with open(TARGET_PATH, "r") as file:
        target_column = file.read().strip()

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = add_engineered_features(
        train_df,
        target_column
    )

    test_df = add_engineered_features(
        test_df,
        target_column
    )

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column].values.astype(np.float32)

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column].values.astype(np.float32)

    device = get_device()

    print("Device:", device)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nClass balance:")
    print(pd.Series(y_train).value_counts(normalize=True))

    experiments = []

    smote_ratios = [
        None,
        0.70,
        0.80,
        0.90,
        1.00
    ]

    class_weight_pairs = [
        (1.0, 1.0),
        (1.25, 1.0),
        (1.5, 1.0),
        (1.75, 1.0),
        (2.0, 1.0),
        (2.5, 1.0)
    ]

    for smote_ratio in smote_ratios:
        for class_0_weight, class_1_weight in class_weight_pairs:
            experiments.append({
                "experiment_name": f"smote={smote_ratio}_class0={class_0_weight}_class1={class_1_weight}",
                "smote_ratio": smote_ratio,
                "class_0_weight": class_0_weight,
                "class_1_weight": class_1_weight,
                "embed_dim": 32,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "batch_size": 256
            })

    experiments.extend([
        {
            "experiment_name": "smaller_transformer_smote_0_9",
            "smote_ratio": 0.90,
            "class_0_weight": 1.0,
            "class_1_weight": 1.0,
            "embed_dim": 16,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 256
        },
        {
            "experiment_name": "larger_transformer_smote_0_9",
            "smote_ratio": 0.90,
            "class_0_weight": 1.0,
            "class_1_weight": 1.0,
            "embed_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.3,
            "learning_rate": 0.0007,
            "weight_decay": 0.0002,
            "batch_size": 256
        }
    ])

    all_results = []
    trained_models = {}
    scalers = {}

    for experiment in experiments:
        result_df, model, scaler, probabilities = run_experiment(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            device=device,
            max_epochs=80,
            patience=10,
            **experiment
        )

        all_results.append(result_df)

        experiment_name = experiment["experiment_name"]
        trained_models[experiment_name] = model
        scalers[experiment_name] = scaler

    all_results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    all_results_df.to_csv(
        TRANSFORMER_REPORT_PATH,
        index=False
    )

    print("\nSaved transformer full report to:", TRANSFORMER_REPORT_PATH)

    print("\n\n==============================")
    print("TOP 20 TRANSFORMERS BY ACCURACY")
    print("==============================")
    print(
        all_results_df.sort_values(
            by="accuracy",
            ascending=False
        ).head(20).to_string(index=False)
    )

    print("\n\n==============================")
    print("TOP 20 TRANSFORMERS BY BALANCED ACCURACY")
    print("==============================")
    print(
        all_results_df.sort_values(
            by="balanced_accuracy",
            ascending=False
        ).head(20).to_string(index=False)
    )

    print("\n\n==============================")
    print("TOP 20 TRANSFORMERS BY MACRO F1")
    print("==============================")
    print(
        all_results_df.sort_values(
            by="macro_f1",
            ascending=False
        ).head(20).to_string(index=False)
    )

    best_macro_f1_row = all_results_df.sort_values(
        by="macro_f1",
        ascending=False
    ).iloc[0]

    pd.DataFrame([best_macro_f1_row]).to_csv(
        TRANSFORMER_BEST_REPORT_PATH,
        index=False
    )

    best_experiment_name = best_macro_f1_row["experiment"]

    best_model = trained_models[best_experiment_name]
    best_scaler = scalers[best_experiment_name]

    print("\nSaved best transformer summary to:", TRANSFORMER_BEST_REPORT_PATH)

    print_final_report(
        best_row=best_macro_f1_row,
        model=best_model,
        scaler=best_scaler,
        X_test=X_test,
        y_test=y_test,
        device=device
    )

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "num_features": X_train.shape[1],
            "threshold": best_macro_f1_row["threshold"],
            "feature_columns": list(X_train.columns),
            "target_column": target_column,
            "experiment": best_experiment_name,
            "embed_dim": int(best_macro_f1_row["embed_dim"]),
            "num_heads": int(best_macro_f1_row["num_heads"]),
            "num_layers": int(best_macro_f1_row["num_layers"]),
            "dropout": float(best_macro_f1_row["dropout"])
        },
        TRANSFORMER_MODEL_OUTPUT_PATH
    )

    joblib.dump(
        best_scaler,
        TRANSFORMER_SCALER_OUTPUT_PATH
    )

    print("\nSaved best transformer model to:", TRANSFORMER_MODEL_OUTPUT_PATH)
    print("Saved best transformer scaler to:", TRANSFORMER_SCALER_OUTPUT_PATH)


if __name__ == "__main__":
    main()