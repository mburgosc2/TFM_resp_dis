"""Entrena Stage 1 WST recording-level + Logistic Regression.

Busca exclusivamente mediante predicciones OOF de TRAIN:

* representacion sin PCA o PCA de 32, 64, 128 y 256 componentes;
* regularizacion C;
* penalizacion L1/L2;
* sin pesos de clase o ``class_weight='balanced'``;
* umbral de decision entre 0,10 y 0,90.

StandardScaler, PCA y LR se ajustan dentro de cada fold. Tras congelar la
configuracion ganadora se ajusta un modelo con todo TRAIN y se consulta
VALIDATION. TEST solo se lee con ``--action test``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_stage1_cough_no_cough_rf_audio_quality import classification_metrics


ROOT = Path(__file__).resolve().parent
PRESET_NAME = "paper_q8_q1_t500_recording_mean_std_max"
FEATURES_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_coughs_random"
    / PRESET_NAME
)
EXPERIMENT_NAME = "wst_recording_lr_fsd50k_coughs_random"
RESULTS_DIR = ROOT / "results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
GRAPHS_DIR = ROOT / "graphs_results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
MODEL_PATH = RESULTS_DIR / "stage1_wst_recording_lr.joblib"

RANDOM_STATE = 42
EXPECTED_FEATURES = 1932
PROJECTIONS: tuple[int | None, ...] = (None, 32, 64, 128, 256)
C_VALUES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0)
PENALTIES = ("l1", "l2")
CLASS_WEIGHTS: tuple[str | None, ...] = (None, "balanced")
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 WST recording-level + busqueda OOF de LR"
    )
    parser.add_argument(
        "--action",
        choices=["check", "train", "test"],
        default="check",
        help=(
            "check valida features; train selecciona con TRAIN y evalua "
            "VALIDATION; test evalua el modelo congelado"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar resultados de la misma accion",
    )
    return parser.parse_args()


def parse_bool_series(series: pd.Series, column: str) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        raise ValueError(
            f"Valores booleanos invalidos en {column}: "
            f"{sorted(normalized.loc[invalid].unique())[:5]}"
        )
    return normalized.map(mapping).astype(bool)


def load_feature_configuration() -> dict[str, str]:
    path = FEATURES_DIR / "wst_recording_configuration.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero la extraccion WST."
        )
    frame = pd.read_csv(path)
    if not {"parameter", "value"}.issubset(frame.columns):
        raise ValueError(f"Configuracion WST invalida: {path}")
    values = dict(zip(frame["parameter"].astype(str), frame["value"].astype(str)))
    if int(float(values.get("feature_dimension", "-1"))) != EXPECTED_FEATURES:
        raise ValueError("La extraccion WST no contiene 1932 features")
    if values.get("near_silence_policy") != "reject":
        raise ValueError(
            "Para comparar con MFCC, regenera WST con --near-silence-policy reject"
        )
    return values


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    npy_split = "val" if split == "validation" else split
    X = np.load(FEATURES_DIR / f"X_{npy_split}.npy")
    y = np.load(FEATURES_DIR / f"y_{npy_split}.npy").astype(np.int8)
    manifest = pd.read_csv(
        FEATURES_DIR / f"metadata_recordings_features_{split}.csv",
        low_memory=False,
    )
    if X.shape != (len(y), EXPECTED_FEATURES):
        raise ValueError(f"X_{npy_split} inesperado: {X.shape}")
    if len(manifest) != len(y):
        raise ValueError(f"Manifiesto desalineado en {split}")
    if not np.isfinite(X).all():
        raise ValueError(f"X_{npy_split} contiene NaN o infinito")
    required = {
        "dataset_origin",
        "fold",
        "is_new_fsd50k_cough",
        "original_uuid",
        "split_group",
        "stage1_target",
        "stage2_eligible",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {split}: {sorted(missing)}")
    if manifest["original_uuid"].astype(str).duplicated().any():
        raise ValueError(f"Hay grabaciones duplicadas en {split}")
    manifest_target = manifest["stage1_target"].to_numpy(np.int8)
    if not np.array_equal(y, manifest_target):
        raise ValueError(f"Las etiquetas no coinciden en {split}")
    is_new = parse_bool_series(
        manifest["is_new_fsd50k_cough"], "is_new_fsd50k_cough"
    )
    stage2 = parse_bool_series(manifest["stage2_eligible"], "stage2_eligible")
    invalid_new = is_new & (
        manifest["dataset_origin"].astype(str).ne("FSD50K")
        | pd.Series(y, index=manifest.index).ne(1)
        | stage2
    )
    if invalid_new.any():
        raise ValueError("Hay nuevas toses FSD50K mal configuradas")
    manifest["is_new_fsd50k_cough"] = is_new
    manifest["stage2_eligible"] = stage2
    return X, y, manifest


def validate_isolation(
    manifests: dict[str, pd.DataFrame],
    folds: np.ndarray | None = None,
) -> None:
    groups: dict[str, set[str]] = {}
    recordings: dict[str, set[str]] = {}
    for split, manifest in manifests.items():
        groups[split] = set(manifest["split_group"].astype(str))
        recordings[split] = set(manifest["original_uuid"].astype(str))
    names = list(manifests)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if groups[left].intersection(groups[right]):
                raise ValueError(f"Fuga de split_group entre {left} y {right}")
            if recordings[left].intersection(recordings[right]):
                raise ValueError(f"Fuga de original_uuid entre {left} y {right}")
    if folds is not None:
        train = manifests["train"]
        if len(folds) != len(train):
            raise ValueError("folds_train.npy no esta alineado")
        if set(folds.tolist()) != {0, 1, 2, 3, 4}:
            raise ValueError("TRAIN no contiene exactamente folds 0..4")
        if not np.array_equal(folds, train["fold"].to_numpy(int)):
            raise ValueError("Los folds no coinciden con el manifiesto")
        frame = pd.DataFrame(
            {"split_group": train["split_group"].astype(str), "fold": folds}
        )
        if (frame.groupby("split_group")["fold"].nunique() > 1).any():
            raise ValueError("Un split_group aparece en varios folds")


def projection_name(n_components: int | None) -> str:
    return "no_pca" if n_components is None else f"pca{n_components}"


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def candidate_name(
    n_components: int | None,
    C: float,
    penalty: str,
    class_weight: str | None,
) -> str:
    return (
        f"lr__{projection_name(n_components)}__C{format_number(C)}"
        f"__{penalty}__weight_{class_weight or 'none'}"
    )


def make_classifier(
    C: float,
    penalty: str,
    class_weight: str | None,
) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver="liblinear",
        class_weight=class_weight,
        max_iter=5_000,
        random_state=RANDOM_STATE,
    )


def transform_folds(
    X: np.ndarray,
    folds: np.ndarray,
    n_components: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    transformed: list[dict[str, Any]] = []
    variance_rows: list[dict[str, object]] = []
    for fold in sorted(np.unique(folds)):
        validation_mask = folds == fold
        train_mask = ~validation_mask
        scaler = StandardScaler()
        X_inner_train = scaler.fit_transform(X[train_mask])
        X_inner_validation = scaler.transform(X[validation_mask])
        explained_variance = np.nan
        if n_components is not None:
            pca = PCA(
                n_components=n_components,
                svd_solver="randomized",
                random_state=RANDOM_STATE,
            )
            X_inner_train = pca.fit_transform(X_inner_train)
            X_inner_validation = pca.transform(X_inner_validation)
            explained_variance = float(pca.explained_variance_ratio_.sum())
        transformed.append(
            {
                "fold": int(fold),
                "train_mask": train_mask,
                "validation_mask": validation_mask,
                "X_train": X_inner_train,
                "X_validation": X_inner_validation,
            }
        )
        variance_rows.append(
            {
                "projection": projection_name(n_components),
                "n_components": X_inner_train.shape[1],
                "fold": int(fold),
                "explained_variance": explained_variance,
            }
        )
        print(
            f"  Fold {fold}: scaler"
            + (
                f" + PCA{n_components} ({explained_variance * 100:.2f} %)"
                if n_components is not None
                else " sin PCA"
            )
            + " ajustados solo con TRAIN interno"
        )
    return transformed, variance_rows


def oof_probabilities(
    transformed_folds: list[dict[str, Any]],
    y: np.ndarray,
    C: float,
    penalty: str,
    class_weight: str | None,
) -> tuple[np.ndarray, float, int]:
    probability = np.zeros(len(y), dtype=np.float64)
    fit_seconds = 0.0
    max_iterations = 0
    for fold_data in transformed_folds:
        train_mask = fold_data["train_mask"]
        validation_mask = fold_data["validation_mask"]
        classifier = make_classifier(C, penalty, class_weight)
        started = time.perf_counter()
        classifier.fit(fold_data["X_train"], y[train_mask])
        fit_seconds += time.perf_counter() - started
        probability[validation_mask] = classifier.predict_proba(
            fold_data["X_validation"]
        )[:, 1]
        max_iterations = max(max_iterations, int(classifier.n_iter_.max()))
    if not np.isfinite(probability).all():
        raise ValueError("Probabilidades OOF no validas")
    return probability, fit_seconds, max_iterations


def best_threshold(
    y: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, dict[str, float | int]]:
    best: tuple[tuple[float, ...], float, dict[str, float | int]] | None = None
    for threshold in THRESHOLDS:
        prediction = (probability >= threshold).astype(np.int8)
        metrics = classification_metrics(y, prediction, probability)
        key = (
            float(metrics["macro_f1"]),
            float(metrics["balanced_accuracy"]),
            -abs(float(threshold) - 0.5),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    if best is None:
        raise RuntimeError("No se pudo seleccionar umbral")
    return best[1], best[2]


def subgroup_metrics(
    manifest: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    is_new = manifest["is_new_fsd50k_cough"].astype(bool).to_numpy()
    original_cough = (y == 1) & ~is_new
    new_cough = (y == 1) & is_new
    fsd = manifest["dataset_origin"].astype(str).eq("FSD50K").to_numpy()

    def recall(mask: np.ndarray) -> float:
        return float(np.mean(prediction[mask] == 1)) if mask.any() else np.nan

    output: dict[str, float | int] = {
        "original_cough_count": int(original_cough.sum()),
        "original_cough_recall": recall(original_cough),
        "new_fsd50k_cough_count": int(new_cough.sum()),
        "new_fsd50k_cough_recall": recall(new_cough),
    }
    if fsd.any() and np.unique(y[fsd]).size == 2:
        fsd_metrics = classification_metrics(
            y[fsd], prediction[fsd], np.zeros(int(fsd.sum()))
        )
        output["fsd50k_balanced_accuracy"] = fsd_metrics["balanced_accuracy"]
        output["fsd50k_specificity_no_cough"] = fsd_metrics[
            "specificity_no_cough"
        ]
    else:
        output["fsd50k_balanced_accuracy"] = np.nan
        output["fsd50k_specificity_no_cough"] = np.nan
    return output


def candidate_selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["macro_f1"]),
        float(row["balanced_accuracy"]),
        float(row["roc_auc"]),
        -float(row["n_components"]),
        -float(row["C"]),
    )


def build_final_pipeline(
    n_components: int | None,
    C: float,
    penalty: str,
    class_weight: str | None,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if n_components is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    random_state=RANDOM_STATE,
                ),
            )
        )
    steps.append(("classifier", make_classifier(C, penalty, class_weight)))
    return Pipeline(steps)


def save_predictions(
    filename: str,
    manifest: pd.DataFrame,
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> None:
    output = manifest.copy()
    output["y_true"] = y
    output["y_pred"] = (probability >= threshold).astype(np.int8)
    output["probability_cough"] = probability
    output["threshold"] = threshold
    output.to_csv(RESULTS_DIR / filename, index=False, encoding="utf-8-sig")


def save_validation_graph(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> None:
    prediction = (probability >= threshold).astype(np.int8)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[0].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No tos", "Tos"],
        yticklabels=["No tos", "Tos"],
        xlabel="Prediccion",
        ylabel="Etiqueta real",
        title=f"Validation · umbral {threshold:.2f}",
    )
    false_positive_rate, true_positive_rate, _ = roc_curve(y, probability)
    auc = roc_auc_score(y, probability)
    axes[1].plot(false_positive_rate, true_positive_rate, label=f"AUC={auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set(xlabel="FPR", ylabel="TPR", title="Curva ROC")
    axes[1].legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(GRAPHS_DIR / "validation_wst_recording_lr.png", dpi=180)
    plt.close(figure)


def benchmark_model(model: Pipeline, X: np.ndarray) -> float:
    sample = X[:1]
    for _ in range(5):
        model.predict_proba(sample)
    started = time.perf_counter()
    repetitions = 100
    for _ in range(repetitions):
        model.predict_proba(sample)
    return (time.perf_counter() - started) * 1_000 / repetitions


def print_dataset_check(
    X_train: np.ndarray,
    y_train: np.ndarray,
    manifest_train: pd.DataFrame,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    manifest_validation: pd.DataFrame,
) -> None:
    print("=" * 78)
    print("CHECK - STAGE 1 WST RECORDING + LR")
    print("=" * 78)
    print(f"Features: {FEATURES_DIR}")
    print(f"TRAIN: {X_train.shape}; no_tos/tos={np.bincount(y_train).tolist()}")
    print(
        f"VALIDATION: {X_validation.shape}; "
        f"no_tos/tos={np.bincount(y_validation).tolist()}"
    )
    print(
        "Nuevas toses FSD50K TRAIN/VALIDATION: "
        f"{int(manifest_train['is_new_fsd50k_cough'].sum())}/"
        f"{int(manifest_validation['is_new_fsd50k_cough'].sum())}"
    )
    print(
        f"Candidatos: {len(PROJECTIONS) * len(C_VALUES) * len(PENALTIES) * len(CLASS_WEIGHTS)}"
    )
    print("Scaler, PCA, LR y umbral se seleccionan exclusivamente con OOF de TRAIN.")
    print("TEST no sera leido durante --action train.")


def train(overwrite: bool) -> None:
    if MODEL_PATH.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {MODEL_PATH}. Usa --overwrite para repetir el experimento."
        )
    feature_config = load_feature_configuration()
    X_train, y_train, manifest_train = load_split("train")
    X_validation, y_validation, manifest_validation = load_split("validation")
    folds = np.load(FEATURES_DIR / "folds_train.npy").astype(int)
    validate_isolation(
        {"train": manifest_train, "validation": manifest_validation}, folds
    )
    print_dataset_check(
        X_train,
        y_train,
        manifest_train,
        X_validation,
        y_validation,
        manifest_validation,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, object]] = []
    winner_row: dict[str, Any] | None = None
    winner_probability: np.ndarray | None = None
    candidate_index = 0
    candidate_total = len(PROJECTIONS) * len(C_VALUES) * len(PENALTIES) * len(CLASS_WEIGHTS)

    for n_components in PROJECTIONS:
        print("\n" + "=" * 78)
        print(f"PROYECCION {projection_name(n_components).upper()}")
        print("=" * 78)
        transformed, projection_variance = transform_folds(
            X_train, folds, n_components
        )
        variance_rows.extend(projection_variance)
        output_dimension = EXPECTED_FEATURES if n_components is None else n_components
        for C in C_VALUES:
            for penalty in PENALTIES:
                for class_weight in CLASS_WEIGHTS:
                    candidate_index += 1
                    name = candidate_name(n_components, C, penalty, class_weight)
                    probability, fit_seconds, max_iterations = oof_probabilities(
                        transformed, y_train, C, penalty, class_weight
                    )
                    threshold, metrics = best_threshold(y_train, probability)
                    prediction = (probability >= threshold).astype(np.int8)
                    subgroups = subgroup_metrics(
                        manifest_train, y_train, prediction
                    )
                    row: dict[str, Any] = {
                        "candidate": name,
                        "projection": projection_name(n_components),
                        "n_components": output_dimension,
                        "C": C,
                        "penalty": penalty,
                        "class_weight": class_weight or "none",
                        "threshold": threshold,
                        "fit_seconds_cv": fit_seconds,
                        "max_iterations": max_iterations,
                        **metrics,
                        **subgroups,
                    }
                    candidate_rows.append(row)
                    if winner_row is None or candidate_selection_key(row) > candidate_selection_key(winner_row):
                        winner_row = row
                        winner_probability = probability.copy()
                    print(
                        f"[{candidate_index:03d}/{candidate_total:03d}] {name} | "
                        f"macro-F1={metrics['macro_f1']:.4f} | "
                        f"bal-acc={metrics['balanced_accuracy']:.4f} | "
                        f"recall FSD+={subgroups['new_fsd50k_cough_recall']:.4f} | "
                        f"AUC={metrics['roc_auc']:.4f} | t={threshold:.2f}"
                    )

    if winner_row is None or winner_probability is None:
        raise RuntimeError("La busqueda no produjo ganador")
    candidates = pd.DataFrame(candidate_rows)
    candidates["selected"] = candidates["candidate"].eq(winner_row["candidate"])
    candidates.sort_values(
        ["macro_f1", "balanced_accuracy", "roc_auc"], ascending=False
    ).to_csv(RESULTS_DIR / "candidate_cv_results.csv", index=False)
    pd.DataFrame(variance_rows).to_csv(
        RESULTS_DIR / "pca_explained_variance_by_fold.csv", index=False
    )

    winner_threshold = float(winner_row["threshold"])
    winner_prediction = (winner_probability >= winner_threshold).astype(np.int8)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        fold_metrics = classification_metrics(
            y_train[mask], winner_prediction[mask], winner_probability[mask]
        )
        fold_rows.append({"fold": int(fold), **fold_metrics})
    pd.DataFrame(fold_rows).to_csv(
        RESULTS_DIR / "best_cv_fold_metrics.csv", index=False
    )
    save_predictions(
        "best_oof_predictions.csv",
        manifest_train,
        y_train,
        winner_probability,
        winner_threshold,
    )

    selected_projection = str(winner_row["projection"])
    selected_components = (
        None
        if selected_projection == "no_pca"
        else int(selected_projection.removeprefix("pca"))
    )
    selected_weight = (
        None if winner_row["class_weight"] == "none" else str(winner_row["class_weight"])
    )
    final_model = build_final_pipeline(
        selected_components,
        float(winner_row["C"]),
        str(winner_row["penalty"]),
        selected_weight,
    )
    print("\nAjustando el modelo final exclusivamente con todo TRAIN...")
    started = time.perf_counter()
    final_model.fit(X_train, y_train)
    fit_final_seconds = time.perf_counter() - started

    metrics_rows: list[dict[str, Any]] = []
    oof_metrics = classification_metrics(
        y_train, winner_prediction, winner_probability
    )
    metrics_rows.append(
        {
            "dataset": "train_oof",
            "threshold": winner_threshold,
            **oof_metrics,
            **subgroup_metrics(manifest_train, y_train, winner_prediction),
        }
    )
    validation_probability: np.ndarray | None = None
    for split, X, y, manifest in (
        ("train_fit", X_train, y_train, manifest_train),
        ("validation", X_validation, y_validation, manifest_validation),
    ):
        probability = final_model.predict_proba(X)[:, 1]
        prediction = (probability >= winner_threshold).astype(np.int8)
        metrics = classification_metrics(y, prediction, probability)
        subgroups = subgroup_metrics(manifest, y, prediction)
        metrics_rows.append(
            {
                "dataset": split,
                "threshold": winner_threshold,
                **metrics,
                **subgroups,
            }
        )
        print(
            f"{split:>10}: macro-F1={metrics['macro_f1']:.4f} | "
            f"recall tos={metrics['recall_cough']:.4f} | "
            f"especificidad={metrics['specificity_no_cough']:.4f} | "
            f"recall FSD+={subgroups['new_fsd50k_cough_recall']:.4f} | "
            f"AUC={metrics['roc_auc']:.4f}"
        )
        if split == "validation":
            validation_probability = probability
            save_predictions(
                "validation_predictions.csv",
                manifest,
                y,
                probability,
                winner_threshold,
            )

    if validation_probability is None:
        raise RuntimeError("No se calcularon predicciones de validation")
    save_validation_graph(
        y_validation, validation_probability, winner_threshold
    )
    pd.DataFrame(metrics_rows).to_csv(
        RESULTS_DIR / "metrics_summary.csv", index=False
    )
    joblib.dump(final_model, MODEL_PATH)
    model_size_kb = MODEL_PATH.stat().st_size / 1024
    inference_ms = benchmark_model(final_model, X_validation)
    final_pca_variance = np.nan
    if "pca" in final_model.named_steps:
        final_pca_variance = float(
            final_model.named_steps["pca"].explained_variance_ratio_.sum()
        )
    configuration: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "feature_source": str(FEATURES_DIR),
        "feature_preset": PRESET_NAME,
        "winner": winner_row["candidate"],
        "selection_rule": (
            "maximize OOF TRAIN macro-F1; ties by balanced accuracy, AUC, "
            "smaller representation and stronger regularization"
        ),
        "n_candidates": candidate_total,
        "projection": selected_projection,
        "n_input_features": EXPECTED_FEATURES,
        "n_model_features": winner_row["n_components"],
        "final_pca_explained_variance": final_pca_variance,
        "C": winner_row["C"],
        "penalty": winner_row["penalty"],
        "class_weight": winner_row["class_weight"],
        "threshold": winner_threshold,
        "threshold_selection": "OOF TRAIN only",
        "split_protocol": "same grouped random Stage 1 FSD50K-cough splits",
        "sample_unit": "one row and one label per original_uuid",
        "stage2_use_of_new_fsd50k_coughs": False,
        "test_evaluated": False,
        "fit_final_seconds": fit_final_seconds,
        "model_size_kb": model_size_kb,
        "desktop_classifier_inference_ms": inference_ms,
        "wst_preprocessing_benchmark_included": False,
        "feature_extraction_elapsed_seconds_all_splits": feature_config.get(
            "elapsed_seconds", ""
        ),
    }
    pd.DataFrame(configuration.items(), columns=["parameter", "value"]).to_csv(
        RESULTS_DIR / "experiment_configuration.csv", index=False
    )

    print("\n" + "=" * 78)
    print("RESULTADO STAGE 1 - WST RECORDING + LR")
    print("=" * 78)
    print(f"Ganador OOF: {winner_row['candidate']}")
    print(f"Umbral OOF congelado: {winner_threshold:.2f}")
    print(f"Macro-F1 OOF: {oof_metrics['macro_f1']:.4f}")
    print(f"Modelo joblib: {model_size_kb:.2f} KB")
    print(f"Inferencia del clasificador: {inference_ms:.4f} ms/grabacion")
    print("El tiempo anterior NO incluye calcular WST desde el audio.")
    print(f"Resultados: {RESULTS_DIR}")
    print("TEST permanece reservado. Para evaluarlo: --action test")


def evaluate_test(overwrite: bool) -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No existe {MODEL_PATH}. Ejecuta primero --action train."
        )
    metrics_path = RESULTS_DIR / "metrics_summary.csv"
    configuration_path = RESULTS_DIR / "experiment_configuration.csv"
    if not metrics_path.is_file() or not configuration_path.is_file():
        raise FileNotFoundError("Faltan los resultados del entrenamiento")
    metrics_frame = pd.read_csv(metrics_path)
    if metrics_frame["dataset"].astype(str).eq("test").any() and not overwrite:
        raise FileExistsError("TEST ya fue evaluado; usa --overwrite para reproducirlo")

    configuration = pd.read_csv(configuration_path)
    config_map = dict(zip(configuration["parameter"], configuration["value"]))
    threshold = float(config_map["threshold"])
    model = joblib.load(MODEL_PATH)
    X_test, y_test, manifest_test = load_split("test")
    validate_isolation({"test": manifest_test})
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(np.int8)
    global_metrics = classification_metrics(y_test, prediction, probability)
    subgroups = subgroup_metrics(manifest_test, y_test, prediction)
    test_row = {
        "dataset": "test",
        "threshold": threshold,
        **global_metrics,
        **subgroups,
    }
    metrics_frame = metrics_frame.loc[
        ~metrics_frame["dataset"].astype(str).eq("test")
    ]
    pd.concat([metrics_frame, pd.DataFrame([test_row])], ignore_index=True).to_csv(
        metrics_path, index=False
    )
    save_predictions(
        "test_predictions.csv", manifest_test, y_test, probability, threshold
    )
    configuration.loc[
        configuration["parameter"].eq("test_evaluated"), "value"
    ] = True
    configuration.to_csv(configuration_path, index=False)
    print("=" * 78)
    print("EVALUACION FINAL TEST - STAGE 1 WST RECORDING + LR")
    print("=" * 78)
    print(
        f"TEST: macro-F1={global_metrics['macro_f1']:.4f} | "
        f"recall tos={global_metrics['recall_cough']:.4f} | "
        f"especificidad={global_metrics['specificity_no_cough']:.4f} | "
        f"AUC={global_metrics['roc_auc']:.4f}"
    )
    print(
        f"TEST original_cough: recall={subgroups['original_cough_recall']:.4f} "
        f"(n={subgroups['original_cough_count']})"
    )
    print(
        f"TEST new_fsd50k_cough: recall={subgroups['new_fsd50k_cough_recall']:.4f} "
        f"(n={subgroups['new_fsd50k_cough_count']})"
    )
    print(f"Resultados: {RESULTS_DIR}")


def check() -> None:
    load_feature_configuration()
    X_train, y_train, manifest_train = load_split("train")
    X_validation, y_validation, manifest_validation = load_split("validation")
    folds = np.load(FEATURES_DIR / "folds_train.npy").astype(int)
    validate_isolation(
        {"train": manifest_train, "validation": manifest_validation}, folds
    )
    print_dataset_check(
        X_train,
        y_train,
        manifest_train,
        X_validation,
        y_validation,
        manifest_validation,
    )
    print("Comprobacion completada. TEST no se ha leido.")


def main() -> None:
    args = parse_args()
    if args.action == "check":
        check()
    elif args.action == "train":
        train(args.overwrite)
    else:
        evaluate_test(args.overwrite)


if __name__ == "__main__":
    main()
