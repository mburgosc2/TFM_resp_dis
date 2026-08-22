"""Baseline reproducible de Stage 1: no tos (0) frente a tos (1).

Reproduce el Random Forest de ``rf_training.py`` utilizando los splits y las
features MFCC actuales. Las etiquetas dry/wet/unknown se colapsan a la clase
binaria tos. Guarda resultados estructurados para poder generar el Excel de
experimentos de Stage 1.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
FEATURES_DIR = ROOT / "features_extracted_multiclass_4c"
METADATA_DIR = ROOT / "metadata_splits_multi_experiment_random"
RESULTS_DIR = ROOT / "results_stage1_cough_no_cough" / "mfcc117_rf" / "full"
GRAPHS_DIR = ROOT / "graphs_results_stage1_cough_no_cough" / "mfcc117_rf" / "full"

RANDOM_STATE = 42
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

METADATA_FILES = {
    "train": "metadata_train_multiclass_4c.csv",
    "validation": "metadata_val_multiclass_4c.csv",
    "test": "metadata_test_multiclass_4c.csv",
}


def make_model(n_jobs: int = -1) -> Pipeline:
    """Crea el pipeline histórico, evitando leakage de normalización en CV."""
    params = dict(RF_PARAMS)
    params["n_jobs"] = n_jobs
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(**params)),
        ]
    )


def to_binary(y_multiclass: np.ndarray) -> np.ndarray:
    """0=no_cough; cualquier dry/wet/unknown=1 (cough)."""
    return (np.asarray(y_multiclass).astype(int) != 0).astype(np.int8)


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    npy_name = "val" if split == "validation" else split
    X = np.load(FEATURES_DIR / f"X_{npy_name}.npy")
    y = to_binary(np.load(FEATURES_DIR / f"y_{npy_name}.npy"))
    if X.ndim != 2 or X.shape[1] != 117:
        raise ValueError(f"X_{npy_name} inesperado: {X.shape}; se esperaban 117 features")
    if len(X) != len(y):
        raise ValueError(f"Longitudes distintas en {split}: X={len(X)}, y={len(y)}")
    if not np.isfinite(X).all():
        raise ValueError(f"Hay NaN o infinitos en X_{npy_name}")
    return X, y


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_samples": len(y_true),
        "n_no_cough": int(np.sum(y_true == 0)),
        "n_cough": int(np.sum(y_true == 1)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_cough": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_cough": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity_no_cough": recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "f1_cough": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "tn_no_cough_correct": int(tn),
        "fp_no_cough_as_cough": int(fp),
        "fn_cough_as_no_cough": int(fn),
        "tp_cough_correct": int(tp),
    }


def load_identifiers(split: str, expected_rows: int) -> pd.DataFrame:
    path = METADATA_DIR / METADATA_FILES[split]
    metadata = pd.read_csv(path)
    if len(metadata) != expected_rows:
        # Las antiguas extracciones podían descartar silencios. No asociamos UUIDs
        # por posición si no existe una correspondencia uno a uno demostrable.
        return pd.DataFrame({"row_index": np.arange(expected_rows)})
    keep = [
        col
        for col in ["original_uuid", "uuid_segmento", "dataset_origin", "cough_type_name"]
        if col in metadata.columns
    ]
    result = metadata[keep].reset_index(drop=True).copy()
    result.insert(0, "row_index", np.arange(expected_rows))
    return result


def save_predictions(
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    output = load_identifiers(split, len(y_true))
    output["y_true"] = y_true
    output["y_pred"] = y_pred
    output["probability_cough"] = y_prob
    output["threshold"] = 0.5
    output.to_csv(RESULTS_DIR / f"{split}_predictions.csv", index=False)


def save_graphs(
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    image = ax.imshow(cm, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No tos", "Tos"],
        yticklabels=["No tos", "Tos"],
        xlabel="Predicción",
        ylabel="Etiqueta real",
        title=f"Stage 1 RF — {split}",
    )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / f"confusion_matrix_{split}.png", dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.plot(fpr, tpr, label=f"RF (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="Recall tos", title=f"ROC — {split}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / f"roc_curve_{split}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STAGE 1 — CLASIFICACIÓN NO TOS / TOS — MFCC117 + RANDOM FOREST")
    print("=" * 78)
    print(f"Features actuales: {FEATURES_DIR}")
    print("Etiquetas: no_cough=0; dry/wet/unknown=1")
    print("Umbral fijo: 0.5 (sin optimizar con VALIDATION o TEST)")

    X_train, y_train = load_split("train")
    X_validation, y_validation = load_split("validation")
    X_test, y_test = load_split("test")
    folds = np.load(FEATURES_DIR / "folds_train.npy").astype(int)
    if len(folds) != len(y_train):
        raise ValueError("folds_train no está alineado con TRAIN")

    print(f"TRAIN:      {X_train.shape}; no_tos/tos={np.bincount(y_train).tolist()}")
    print(
        f"VALIDATION: {X_validation.shape}; "
        f"no_tos/tos={np.bincount(y_validation).tolist()}"
    )
    print(f"TEST:       {X_test.shape}; no_tos/tos={np.bincount(y_test).tolist()}")

    oof_pred = np.zeros(len(y_train), dtype=np.int8)
    oof_prob = np.zeros(len(y_train), dtype=np.float64)
    fold_rows: list[dict[str, float | int]] = []

    print("\nCV de TRAIN: scaler y RF se ajustan dentro de cada fold.")
    for fold in sorted(np.unique(folds)):
        inner_validation = folds == fold
        inner_train = ~inner_validation
        model = make_model(n_jobs=-1)
        started = time.perf_counter()
        model.fit(X_train[inner_train], y_train[inner_train])
        elapsed = time.perf_counter() - started
        probabilities = model.predict_proba(X_train[inner_validation])[:, 1]
        predictions = (probabilities >= 0.5).astype(np.int8)
        oof_prob[inner_validation] = probabilities
        oof_pred[inner_validation] = predictions
        row = {
            "fold": int(fold),
            "n_train": int(np.sum(inner_train)),
            "n_validation": int(np.sum(inner_validation)),
            "fit_seconds": elapsed,
            **classification_metrics(
                y_train[inner_validation], predictions, probabilities
            ),
        }
        fold_rows.append(row)
        print(
            f"  Fold {fold}: F1 tos={row['f1_cough']:.4f} | "
            f"macro-F1={row['macro_f1']:.4f} | recall tos={row['recall_cough']:.4f}"
        )

    metrics_rows: list[dict[str, float | int | str]] = [
        {
            "dataset": "train_oof",
            "threshold": 0.5,
            **classification_metrics(y_train, oof_pred, oof_prob),
        }
    ]
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "cv_fold_metrics.csv", index=False)

    oof_output = load_identifiers("train", len(y_train))
    oof_output["fold"] = folds
    oof_output["y_true"] = y_train
    oof_output["y_pred"] = oof_pred
    oof_output["probability_cough"] = oof_prob
    oof_output.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)

    print("\nAjustando modelo final exclusivamente con TRAIN...")
    final_model = make_model(n_jobs=-1)
    started = time.perf_counter()
    final_model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    for split, X, y in [
        ("train_fit", X_train, y_train),
        ("validation", X_validation, y_validation),
        ("test", X_test, y_test),
    ]:
        probability = final_model.predict_proba(X)[:, 1]
        prediction = (probability >= 0.5).astype(np.int8)
        row = {
            "dataset": split,
            "threshold": 0.5,
            **classification_metrics(y, prediction, probability),
        }
        metrics_rows.append(row)
        if split != "train_fit":
            save_predictions(split, y, prediction, probability)
            save_graphs(split, y, prediction, probability)
        print(
            f"{split:>10}: F1 tos={row['f1_cough']:.4f} | "
            f"macro-F1={row['macro_f1']:.4f} | recall tos={row['recall_cough']:.4f} | "
            f"AUC={row['roc_auc']:.4f}"
        )

    model_path = RESULTS_DIR / "stage1_mfcc117_rf.joblib"
    joblib.dump(final_model, model_path)
    rf = final_model.named_steps["classifier"]
    model_size_kb = model_path.stat().st_size / 1024
    pd.DataFrame(metrics_rows).to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    configuration = {
        "experiment": "stage1_cough_no_cough_mfcc117_rf",
        "feature_source": str(FEATURES_DIR),
        "target_mapping": "0=no_cough; 1=dry|wet|unknown",
        "n_features": X_train.shape[1],
        "normalization": "StandardScaler fitted inside each CV fold; final on TRAIN",
        "threshold": 0.5,
        "threshold_selection": "fixed; no threshold tuning",
        "n_estimators": RF_PARAMS["n_estimators"],
        "max_depth": RF_PARAMS["max_depth"],
        "min_samples_split": RF_PARAMS["min_samples_split"],
        "min_samples_leaf": RF_PARAMS["min_samples_leaf"],
        "class_weight": RF_PARAMS["class_weight"],
        "random_state": RANDOM_STATE,
        "fit_final_seconds": fit_seconds,
        "tree_nodes_total": sum(tree.tree_.node_count for tree in rf.estimators_),
        "model_size_kb": model_size_kb,
        "test_note": "TEST ya fue evaluado en el experimento histórico; no es virgen.",
    }
    pd.DataFrame(configuration.items(), columns=["parameter", "value"]).to_csv(
        RESULTS_DIR / "experiment_configuration.csv", index=False
    )

    print("\n" + "=" * 78)
    print("RESULTADOS STAGE 1 GUARDADOS")
    print("=" * 78)
    print(f"Resultados: {RESULTS_DIR}")
    print(f"Gráficas:   {GRAPHS_DIR}")
    print(f"Modelo:     {model_size_kb:.2f} KB")
    print("Nota: TEST ya se consultó en agosto; se conserva como comparación histórica.")

    try:
        from build_stage1_experiments_excel import build_workbook

        excel_path = build_workbook()
        print(f"Excel actualizado: {excel_path}")
    except Exception as exc:  # El entrenamiento no debe perderse por un fallo cosmético.
        print(f"AVISO: no se pudo actualizar el Excel automáticamente: {exc}")


if __name__ == "__main__":
    main()
