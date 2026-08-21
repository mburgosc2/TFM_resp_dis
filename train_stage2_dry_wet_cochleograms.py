"""Entrena clasificadores ligeros dry/wet con cocleogramas gammatone.

Se comparan dos unidades de modelado sin leer ni procesar TEST:

1. ``event``: se entrena con los vectores de cada evento. Cada grabacion
   aporta el mismo peso total y las puntuaciones se promedian por
   ``original_uuid`` antes de calcular cualquier metrica.
2. ``recording``: se entrena directamente con la agregacion
   media/desviacion/maximo de los eventos de cada grabacion.

La seleccion de modelo, C, PCA y umbral se hace exclusivamente mediante
los cinco folds de TRAIN. VALIDATION se evalua una sola vez con la
configuracion elegida. La normalizacion y PCA se ajustan dentro de cada
fold mediante un Pipeline, evitando fuga de informacion.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_ROOT = (
    SCRIPT_DIR / "features_extracted_stage2_dry_wet_cochleograms"
)
RESULTS_ROOT = SCRIPT_DIR / "results_stage2_dry_wet_cochleograms"
MODELS_ROOT = SCRIPT_DIR / "models_stage2_dry_wet_cochleograms"
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_cochleograms"
)

RANDOM_STATE = 42
EXPECTED_TRAIN_FOLDS = {0, 1, 2, 3, 4}
LABEL_TO_NAME = {0: "dry", 1: "wet"}
EXPERIMENTS = ("event", "recording")

FULL_C_VALUES = {
    "logistic_regression": (0.01, 0.1, 1.0, 10.0),
    "linear_svm": (0.001, 0.01, 0.1, 1.0),
}
FULL_PCA_OPTIONS = (None, 32, 64)


@dataclass(frozen=True)
class CandidateSpec:
    model_name: str
    c_value: float
    pca_components: int | None

    @property
    def key(self) -> str:
        pca_name = "none" if self.pca_components is None else str(
            self.pca_components
        )
        return f"{self.model_name}__C{self.c_value:g}__pca{pca_name}"

    @property
    def score_type(self) -> str:
        if self.model_name == "logistic_regression":
            return "wet_probability"
        return "wet_decision_score"

    @property
    def default_threshold(self) -> float:
        if self.model_name == "logistic_regression":
            return 0.5
        return 0.0


@dataclass
class DataView:
    experiment: str
    x_train: np.ndarray
    train_metadata: pd.DataFrame
    x_validation: np.ndarray
    validation_metadata: pd.DataFrame
    feature_names: list[str]


@dataclass
class CandidateEvaluation:
    spec: CandidateSpec
    threshold: float
    metrics: dict[str, float | int]
    oof_recordings: pd.DataFrame
    fold_metrics: pd.DataFrame
    elapsed_seconds: float


def feature_file_names(experiment: str) -> dict[str, str]:
    if experiment == "event":
        return {
            "x_train": "X_events_train.npy",
            "y_train": "y_events_train.npy",
            "folds_train": "folds_events_train.npy",
            "metadata_train": "metadata_events_features_train.csv",
            "x_validation": "X_events_validation.npy",
            "y_validation": "y_events_validation.npy",
            "folds_validation": "folds_events_validation.npy",
            "metadata_validation": (
                "metadata_events_features_validation.csv"
            ),
            "feature_names": "event_feature_names.csv",
        }
    if experiment == "recording":
        return {
            "x_train": "X_train.npy",
            "y_train": "y_train.npy",
            "folds_train": "folds_train.npy",
            "metadata_train": "metadata_recordings_features_train.csv",
            "x_validation": "X_validation.npy",
            "y_validation": "y_validation.npy",
            "folds_validation": "folds_validation.npy",
            "metadata_validation": (
                "metadata_recordings_features_validation.csv"
            ),
            "feature_names": "recording_feature_names.csv",
        }
    raise ValueError(f"Experimento desconocido: {experiment}")


def load_data_view(preset: str, experiment: str) -> DataView:
    input_dir = FEATURES_ROOT / preset
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"No se encuentra el directorio de features: {input_dir}"
        )

    names = feature_file_names(experiment)
    required_paths = {
        key: input_dir / filename for key, filename in names.items()
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos para el entrenamiento:\n" + "\n".join(missing)
        )

    x_train = np.load(required_paths["x_train"])
    y_train = np.load(required_paths["y_train"])
    folds_train = np.load(required_paths["folds_train"])
    train_metadata = pd.read_csv(required_paths["metadata_train"])

    x_validation = np.load(required_paths["x_validation"])
    y_validation = np.load(required_paths["y_validation"])
    folds_validation = np.load(required_paths["folds_validation"])
    validation_metadata = pd.read_csv(
        required_paths["metadata_validation"]
    )
    feature_names_df = pd.read_csv(required_paths["feature_names"])

    validate_arrays_and_metadata(
        experiment,
        "train",
        x_train,
        y_train,
        folds_train,
        train_metadata,
    )
    validate_arrays_and_metadata(
        experiment,
        "validation",
        x_validation,
        y_validation,
        folds_validation,
        validation_metadata,
    )

    train_uuids = set(train_metadata["original_uuid"].astype(str))
    validation_uuids = set(
        validation_metadata["original_uuid"].astype(str)
    )
    overlap = train_uuids & validation_uuids
    if overlap:
        raise ValueError(
            "Train y validation comparten original_uuid: "
            f"{sorted(overlap)[:10]}"
        )

    feature_names = feature_names_df["feature_name"].astype(str).tolist()
    if len(feature_names) != x_train.shape[1]:
        raise ValueError(
            "El numero de nombres de features no coincide con X_train."
        )
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Hay nombres de features duplicados.")

    return DataView(
        experiment=experiment,
        x_train=x_train,
        train_metadata=train_metadata,
        x_validation=x_validation,
        validation_metadata=validation_metadata,
        feature_names=feature_names,
    )


def validate_arrays_and_metadata(
    experiment: str,
    split_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    folds: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    required_columns = {
        "feature_row",
        "original_uuid",
        "cough_type",
        "cough_type_consensus",
        "stage2_target",
        "fold",
        "split",
    }
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas en metadata {split_name}/{experiment}: "
            f"{sorted(missing_columns)}"
        )
    if experiment == "event" and "event_id" not in metadata.columns:
        raise ValueError("El metadata de eventos no contiene event_id.")

    expected_rows = len(metadata)
    if x_values.ndim != 2 or x_values.shape[0] != expected_rows:
        raise ValueError(
            f"X {split_name}/{experiment} no coincide con metadata: "
            f"{x_values.shape} frente a {expected_rows} filas."
        )
    if y_values.shape != (expected_rows,):
        raise ValueError(f"Forma invalida de y: {y_values.shape}")
    if folds.shape != (expected_rows,):
        raise ValueError(f"Forma invalida de folds: {folds.shape}")
    if x_values.dtype != np.float32:
        raise ValueError(f"X debe ser float32, no {x_values.dtype}.")
    if not np.isfinite(x_values).all():
        raise ValueError(f"X {split_name}/{experiment} contiene NaN/Inf.")
    if metadata["feature_row"].tolist() != list(range(expected_rows)):
        raise ValueError("feature_row no esta alineado con las matrices.")
    if not np.array_equal(
        y_values.astype(int),
        metadata["stage2_target"].to_numpy(dtype=int),
    ):
        raise ValueError("y no esta alineado con stage2_target.")
    if not np.array_equal(
        folds.astype(int),
        metadata["fold"].to_numpy(dtype=int),
    ):
        raise ValueError("folds no esta alineado con metadata.")
    if set(y_values.astype(int)) != {0, 1}:
        raise ValueError("Las etiquetas deben ser exactamente dry=0/wet=1.")
    expected_targets = metadata["cough_type"].map(
        {"dry": 0, "wet": 1}
    )
    if expected_targets.isna().any() or not np.array_equal(
        y_values.astype(int), expected_targets.to_numpy(dtype=int)
    ):
        raise ValueError("cough_type no coincide con stage2_target.")
    if set(metadata["split"].astype(str)) != {split_name}:
        raise ValueError(f"La columna split no coincide con {split_name}.")

    unique_folds = set(metadata["fold"].astype(int))
    if split_name == "train" and unique_folds != EXPECTED_TRAIN_FOLDS:
        raise ValueError(f"Folds de train inesperados: {unique_folds}")
    if split_name == "validation" and unique_folds != {-1}:
        raise ValueError(f"Fold de validation inesperado: {unique_folds}")

    grouped = metadata.groupby("original_uuid")
    if grouped["stage2_target"].nunique().max() != 1:
        raise ValueError("Una grabacion contiene varias etiquetas.")
    if grouped["fold"].nunique().max() != 1:
        raise ValueError("Una grabacion aparece en varios folds.")
    if experiment == "recording" and metadata["original_uuid"].duplicated().any():
        raise ValueError("La vista recording contiene UUID duplicados.")
    if experiment == "event" and metadata["event_id"].duplicated().any():
        raise ValueError("La vista event contiene event_id duplicados.")


def candidate_specs(
    quick: bool,
    experiment: str,
) -> list[CandidateSpec]:
    if quick:
        c_values = {
            "logistic_regression": (1.0,),
            "linear_svm": (0.1,),
        }
        pca_options = (None,)
    else:
        c_values = FULL_C_VALUES
        pca_options = (
            (None,) if experiment == "event" else FULL_PCA_OPTIONS
        )

    return [
        CandidateSpec(model_name, c_value, pca_components)
        for model_name, model_c_values in c_values.items()
        for c_value in model_c_values
        for pca_components in pca_options
    ]


def build_pipeline(spec: CandidateSpec) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if spec.pca_components is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=spec.pca_components,
                    whiten=True,
                    svd_solver="randomized",
                    random_state=RANDOM_STATE,
                ),
            )
        )

    if spec.model_name == "logistic_regression":
        classifier = LogisticRegression(
            C=spec.c_value,
            penalty="l2",
            solver="liblinear",
            max_iter=5_000,
            random_state=RANDOM_STATE,
        )
    elif spec.model_name == "linear_svm":
        classifier = LinearSVC(
            C=spec.c_value,
            penalty="l2",
            dual="auto",
            max_iter=20_000,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Modelo desconocido: {spec.model_name}")

    steps.append(("classifier", classifier))
    return Pipeline(steps)


def compute_training_weights(metadata: pd.DataFrame) -> np.ndarray:
    """Iguala grabaciones y clases usando solo el subconjunto de ajuste."""

    recording_rows = metadata.drop_duplicates("original_uuid")
    class_counts = recording_rows["stage2_target"].value_counts()
    if set(class_counts.index.astype(int)) != {0, 1}:
        raise ValueError("El subconjunto de ajuste no contiene ambas clases.")

    recording_total = len(recording_rows)
    class_factors = {
        label: recording_total / (2.0 * int(class_counts[label]))
        for label in (0, 1)
    }
    events_per_recording = metadata.groupby("original_uuid")[
        "original_uuid"
    ].transform("size")
    class_factor_per_row = metadata["stage2_target"].map(class_factors)
    weights = (
        class_factor_per_row.to_numpy(dtype=float)
        / events_per_recording.to_numpy(dtype=float)
    )
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Se generaron sample_weight invalidos.")
    return weights


def compute_recording_equal_weights(metadata: pd.DataFrame) -> np.ndarray:
    """Hace que cada grabacion pese igual en el StandardScaler."""

    samples_per_recording = metadata.groupby("original_uuid")[
        "original_uuid"
    ].transform("size")
    weights = 1.0 / samples_per_recording.to_numpy(dtype=float)
    # StandardScaler solo depende de pesos relativos. Normalizar la media a
    # uno facilita su inspeccion sin cambiar media ni varianza ponderadas.
    weights /= np.mean(weights)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Se generaron pesos invalidos para StandardScaler.")
    return weights


def model_scores(model: Pipeline, x_values: np.ndarray) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        scores = model.predict_proba(x_values)[:, 1]
    else:
        scores = model.decision_function(x_values)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(x_values),) or not np.isfinite(scores).all():
        raise RuntimeError("El modelo produjo puntuaciones invalidas.")
    return scores


def aggregate_scores_by_recording(
    metadata: pd.DataFrame,
    sample_scores: np.ndarray,
) -> pd.DataFrame:
    if len(metadata) != len(sample_scores):
        raise ValueError("Scores y metadata tienen longitudes distintas.")

    rows = metadata[
        [
            "original_uuid",
            "stage2_target",
            "cough_type",
            "cough_type_consensus",
            "fold",
        ]
    ].copy()
    rows["score"] = sample_scores
    aggregated = (
        rows.groupby("original_uuid", sort=False)
        .agg(
            y_true=("stage2_target", "first"),
            cough_type=("cough_type", "first"),
            cough_type_consensus=("cough_type_consensus", "first"),
            fold=("fold", "first"),
            score=("score", "mean"),
            sample_count=("score", "size"),
        )
        .reset_index()
    )
    if not np.isfinite(aggregated["score"]).all():
        raise RuntimeError("La agregacion genero scores invalidos.")
    return aggregated


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def tune_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    default_threshold: float,
) -> float:
    unique_scores = np.unique(scores)
    if len(unique_scores) == 1:
        return default_threshold

    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    candidates = np.concatenate(
        [
            [np.nextafter(unique_scores[0], -np.inf)],
            midpoints,
            [np.nextafter(unique_scores[-1], np.inf), default_threshold],
        ]
    )
    candidates = np.unique(candidates)

    best_threshold = default_threshold
    best_key = (-np.inf, -np.inf, -np.inf)
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predicted,
            labels=[0, 1],
        ).ravel()
        dry_f1 = safe_divide(2.0 * tn, 2.0 * tn + fn + fp)
        wet_f1 = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
        macro_f1 = (dry_f1 + wet_f1) / 2.0
        dry_recall = safe_divide(tn, tn + fp)
        wet_recall = safe_divide(tp, tp + fn)
        balanced_accuracy = (dry_recall + wet_recall) / 2.0
        key = (
            macro_f1,
            balanced_accuracy,
            -abs(float(threshold) - default_threshold),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    ).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predicted)
        ),
        "macro_f1": float(
            f1_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "dry_precision": float(
            precision_score(
                y_true,
                predicted,
                pos_label=0,
                zero_division=0,
            )
        ),
        "wet_precision": float(
            precision_score(
                y_true,
                predicted,
                pos_label=1,
                zero_division=0,
            )
        ),
        "dry_recall": float(
            recall_score(
                y_true,
                predicted,
                pos_label=0,
                zero_division=0,
            )
        ),
        "wet_recall": float(
            recall_score(
                y_true,
                predicted,
                pos_label=1,
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision_wet": float(
            average_precision_score(y_true, scores)
        ),
        "tn_dry_correct": int(tn),
        "fp_dry_as_wet": int(fp),
        "fn_wet_as_dry": int(fn),
        "tp_wet_correct": int(tp),
    }


def evaluate_candidate_cv(
    data: DataView,
    spec: CandidateSpec,
) -> CandidateEvaluation:
    start_time = time.perf_counter()
    oof_sample_scores = np.full(len(data.x_train), np.nan, dtype=float)

    for fold in sorted(EXPECTED_TRAIN_FOLDS):
        validation_mask = data.train_metadata["fold"].to_numpy() == fold
        training_mask = ~validation_mask
        fold_training_metadata = data.train_metadata.loc[
            training_mask
        ].reset_index(drop=True)
        weights = compute_training_weights(fold_training_metadata)
        scaler_weights = compute_recording_equal_weights(
            fold_training_metadata
        )

        model = build_pipeline(spec)
        model.fit(
            data.x_train[training_mask],
            data.train_metadata.loc[
                training_mask, "stage2_target"
            ].to_numpy(dtype=int),
            scaler__sample_weight=scaler_weights,
            classifier__sample_weight=weights,
        )
        oof_sample_scores[validation_mask] = model_scores(
            model,
            data.x_train[validation_mask],
        )

    if not np.isfinite(oof_sample_scores).all():
        raise RuntimeError(f"OOF incompleto para {spec.key}.")

    oof_recordings = aggregate_scores_by_recording(
        data.train_metadata,
        oof_sample_scores,
    )
    y_true = oof_recordings["y_true"].to_numpy(dtype=int)
    scores = oof_recordings["score"].to_numpy(dtype=float)
    threshold = tune_threshold(y_true, scores, spec.default_threshold)
    metrics = binary_metrics(y_true, scores, threshold)
    oof_recordings["y_pred"] = (scores >= threshold).astype(int)
    oof_recordings["candidate_key"] = spec.key

    fold_rows = []
    for fold, fold_df in oof_recordings.groupby("fold", sort=True):
        fold_result = binary_metrics(
            fold_df["y_true"].to_numpy(dtype=int),
            fold_df["score"].to_numpy(dtype=float),
            threshold,
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "recording_count": len(fold_df),
                "threshold": threshold,
                **fold_result,
            }
        )

    return CandidateEvaluation(
        spec=spec,
        threshold=threshold,
        metrics=metrics,
        oof_recordings=oof_recordings,
        fold_metrics=pd.DataFrame(fold_rows),
        elapsed_seconds=time.perf_counter() - start_time,
    )


def selection_key(evaluation: CandidateEvaluation) -> tuple[float, ...]:
    # En empates se prioriza no usar PCA y despues regresion logistica,
    # porque requieren una inferencia mas sencilla y dan probabilidad directa.
    no_pca = float(evaluation.spec.pca_components is None)
    logistic = float(
        evaluation.spec.model_name == "logistic_regression"
    )
    return (
        float(evaluation.metrics["macro_f1"]),
        float(evaluation.metrics["balanced_accuracy"]),
        float(evaluation.metrics["roc_auc"]),
        no_pca,
        logistic,
    )


def fit_final_model(
    data: DataView,
    spec: CandidateSpec,
) -> Pipeline:
    weights = compute_training_weights(data.train_metadata)
    scaler_weights = compute_recording_equal_weights(data.train_metadata)
    model = build_pipeline(spec)
    model.fit(
        data.x_train,
        data.train_metadata["stage2_target"].to_numpy(dtype=int),
        scaler__sample_weight=scaler_weights,
        classifier__sample_weight=weights,
    )
    return model


def learned_inference_float_count(model: Pipeline) -> int:
    scaler = model.named_steps["scaler"]
    count = int(scaler.mean_.size + scaler.scale_.size)
    if "pca" in model.named_steps:
        pca = model.named_steps["pca"]
        count += int(
            pca.mean_.size
            + pca.components_.size
            + pca.explained_variance_.size
        )
    classifier = model.named_steps["classifier"]
    count += int(classifier.coef_.size + classifier.intercept_.size)
    return count


def create_validation_graph(
    predictions: pd.DataFrame,
    threshold: float,
    experiment: str,
    spec: CandidateSpec,
    output_path: Path,
) -> None:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    y_pred = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_percent = matrix / matrix.sum(axis=1, keepdims=True) * 100.0

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[0].text(
                column,
                row,
                f"{matrix[row, column]}\n({row_percent[row, column]:.1f}%)",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axes[0].set_xticks([0, 1], ["Dry", "Wet"])
    axes[0].set_yticks([0, 1], ["Dry", "Wet"])
    axes[0].set_xlabel("Prediccion")
    axes[0].set_ylabel("Etiqueta real")
    axes[0].set_title("Matriz de confusion")
    figure.colorbar(image, ax=axes[0], fraction=0.046)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    axes[1].plot(
        false_positive_rate,
        true_positive_rate,
        label=f"AUC={roc_auc:.3f}",
    )
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    average_precision = average_precision_score(y_true, scores)
    wet_prevalence = float(np.mean(y_true == 1))
    axes[2].plot(
        recall,
        precision,
        label=f"AP wet={average_precision:.3f}",
    )
    axes[2].axhline(
        wet_prevalence,
        linestyle="--",
        color="grey",
        label=f"Baseline={wet_prevalence:.3f}",
    )
    axes[2].set_xlabel("Recall wet")
    axes[2].set_ylabel("Precision wet")
    axes[2].set_title("Precision-recall")
    axes[2].legend()
    axes[2].grid(alpha=0.2)

    figure.suptitle(
        f"Validation | {experiment} | {spec.key}",
        fontsize=14,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train_experiment(
    data: DataView,
    preset: str,
    quick: bool,
) -> dict[str, Any]:
    run_name = "quick" if quick else "full"
    result_dir = RESULTS_ROOT / preset / run_name / data.experiment
    model_dir = MODELS_ROOT / preset / run_name
    graph_dir = GRAPHS_ROOT / preset / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    specs = candidate_specs(quick, data.experiment)
    candidate_rows = []
    best_evaluation: CandidateEvaluation | None = None

    print("\n" + "=" * 78)
    print(f"EXPERIMENTO: {data.experiment.upper()}")
    print("=" * 78)
    print(
        f"Train samples={len(data.x_train)}, "
        f"features={data.x_train.shape[1]}, candidatos={len(specs)}"
    )

    for candidate_index, spec in enumerate(specs, start=1):
        evaluation = evaluate_candidate_cv(data, spec)
        candidate_rows.append(
            {
                "experiment": data.experiment,
                "candidate_key": spec.key,
                "model_name": spec.model_name,
                "C": spec.c_value,
                "pca_components": (
                    "none"
                    if spec.pca_components is None
                    else spec.pca_components
                ),
                "score_type": spec.score_type,
                "threshold_oof": evaluation.threshold,
                "elapsed_seconds": evaluation.elapsed_seconds,
                **evaluation.metrics,
            }
        )
        print(
            f"[{candidate_index:02d}/{len(specs):02d}] {spec.key} | "
            f"macro-F1={evaluation.metrics['macro_f1']:.4f} | "
            f"bal-acc={evaluation.metrics['balanced_accuracy']:.4f} | "
            f"AUC={evaluation.metrics['roc_auc']:.4f}"
        )
        if (
            best_evaluation is None
            or selection_key(evaluation) > selection_key(best_evaluation)
        ):
            best_evaluation = evaluation

    if best_evaluation is None:
        raise RuntimeError("No se evaluo ningun candidato.")

    candidate_results = pd.DataFrame(candidate_rows).sort_values(
        ["macro_f1", "balanced_accuracy", "roc_auc"],
        ascending=False,
    )
    candidate_results.to_csv(
        result_dir / "candidate_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    spec = best_evaluation.spec
    best_evaluation.oof_recordings.to_csv(
        result_dir / "best_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_evaluation.fold_metrics.to_csv(
        result_dir / "best_cv_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof_summary = {
        "dataset": "train_oof",
        "experiment": data.experiment,
        "candidate_key": spec.key,
        "threshold": best_evaluation.threshold,
        "model_size_kb_joblib": np.nan,
        "learned_inference_float_count": np.nan,
        "estimated_float32_parameters_kb": np.nan,
        **best_evaluation.metrics,
    }
    if quick:
        pd.DataFrame([oof_summary]).to_csv(
            result_dir / "metrics_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\nMejor configuracion de la prueba rapida:")
        print(f"  {spec.key}")
        print(f"  Umbral OOF: {best_evaluation.threshold:.6f}")
        print(
            f"  OOF macro-F1: {best_evaluation.metrics['macro_f1']:.4f}"
        )
        print("  VALIDATION no se ha evaluado en modo --quick.")
        print(f"  Resultados OOF: {result_dir}")
        return oof_summary

    final_model = fit_final_model(data, spec)
    validation_sample_scores = model_scores(
        final_model,
        data.x_validation,
    )
    validation_recordings = aggregate_scores_by_recording(
        data.validation_metadata,
        validation_sample_scores,
    )
    validation_scores = validation_recordings["score"].to_numpy(float)
    validation_recordings["y_pred"] = (
        validation_scores >= best_evaluation.threshold
    ).astype(int)
    validation_recordings["candidate_key"] = spec.key
    validation_metrics = binary_metrics(
        validation_recordings["y_true"].to_numpy(dtype=int),
        validation_scores,
        best_evaluation.threshold,
    )

    model_path = model_dir / f"{data.experiment}_model.joblib"
    model_package = {
        "pipeline": final_model,
        "experiment": data.experiment,
        "preset": preset,
        "candidate_key": spec.key,
        "model_name": spec.model_name,
        "C": spec.c_value,
        "pca_components": spec.pca_components,
        "score_type": spec.score_type,
        "recording_aggregation": (
            "mean_event_score"
            if data.experiment == "event"
            else "single_recording_score"
        ),
        "threshold": best_evaluation.threshold,
        "label_mapping": LABEL_TO_NAME,
        "feature_names": data.feature_names,
        "unknown_rejection_calibrated": False,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(model_package, model_path, compress=3)
    model_size_kb = model_path.stat().st_size / 1024.0
    learned_float_count = learned_inference_float_count(final_model)

    validation_recordings.to_csv(
        result_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    deployment_fields = {
        "model_size_kb_joblib": model_size_kb,
        "learned_inference_float_count": learned_float_count,
        "estimated_float32_parameters_kb": (
            learned_float_count * 4.0 / 1024.0
        ),
    }
    oof_summary.update(deployment_fields)
    summary_rows = [
        oof_summary,
        {
            "dataset": "validation",
            "experiment": data.experiment,
            "candidate_key": spec.key,
            "threshold": best_evaluation.threshold,
            "model_size_kb_joblib": model_size_kb,
            "learned_inference_float_count": learned_float_count,
            "estimated_float32_parameters_kb": (
                learned_float_count * 4.0 / 1024.0
            ),
            **validation_metrics,
        },
    ]
    pd.DataFrame(summary_rows).to_csv(
        result_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    graph_path = graph_dir / f"validation_{data.experiment}.png"
    create_validation_graph(
        validation_recordings,
        best_evaluation.threshold,
        data.experiment,
        spec,
        graph_path,
    )

    print("\nMejor configuracion:")
    print(f"  {spec.key}")
    print(f"  Umbral OOF: {best_evaluation.threshold:.6f}")
    print(
        f"  OOF macro-F1: {best_evaluation.metrics['macro_f1']:.4f}"
    )
    print(
        f"  Validation macro-F1: {validation_metrics['macro_f1']:.4f}"
    )
    print(
        "  Validation recalls dry/wet: "
        f"{validation_metrics['dry_recall']:.4f} / "
        f"{validation_metrics['wet_recall']:.4f}"
    )
    print(f"  Modelo joblib: {model_size_kb:.2f} KB")
    print(f"  Resultados: {result_dir}")
    print(f"  Grafica: {graph_path}")

    return summary_rows[1]


def print_data_check(data: DataView) -> None:
    train_recordings = data.train_metadata.drop_duplicates(
        "original_uuid"
    )
    validation_recordings = data.validation_metadata.drop_duplicates(
        "original_uuid"
    )
    print("\n" + "=" * 78)
    print(f"CHECK: {data.experiment.upper()}")
    print("=" * 78)
    print(f"X train: {data.x_train.shape} {data.x_train.dtype}")
    print(f"X validation: {data.x_validation.shape} {data.x_validation.dtype}")
    print(
        "Grabaciones train dry/wet: "
        f"{(train_recordings['stage2_target'] == 0).sum()} / "
        f"{(train_recordings['stage2_target'] == 1).sum()}"
    )
    print(
        "Grabaciones validation dry/wet: "
        f"{(validation_recordings['stage2_target'] == 0).sum()} / "
        f"{(validation_recordings['stage2_target'] == 1).sum()}"
    )
    print(
        f"Folds train: {sorted(data.train_metadata['fold'].unique())}"
    )
    print("Validation permanece separada y TEST no ha sido leido.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena los experimentos event/recording de Stage 2 dry/wet."
        )
    )
    parser.add_argument(
        "--action",
        choices=["check", "train"],
        default="check",
        help="check valida entradas; train ejecuta CV y validation.",
    )
    parser.add_argument(
        "--experiment",
        choices=["event", "recording", "both"],
        default="both",
        help="Selecciona una rama o ejecuta ambas.",
    )
    parser.add_argument(
        "--preset",
        choices=["paper64", "compact32"],
        default="paper64",
        help="Preset de caracteristicas previamente extraido.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Prueba tecnica OOF: dos candidatos sin PCA y sin evaluar "
            "validation. Guarda resultados separados del estudio completo."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected_experiments = (
        EXPERIMENTS if args.experiment == "both" else (args.experiment,)
    )

    print("=" * 78)
    print("ENTRENAMIENTO STAGE 2 DRY/WET — COCLEOGRAMAS")
    print("=" * 78)
    print(f"Preset: {args.preset}")
    print(f"Accion: {args.action}")
    print(f"Experimentos: {', '.join(selected_experiments)}")
    print("TEST no sera leido ni procesado.")

    data_views = [
        load_data_view(args.preset, experiment)
        for experiment in selected_experiments
    ]
    for data in data_views:
        print_data_check(data)

    if args.action == "check":
        print("\nComprobacion completada. No se entreno ningun modelo.")
        return

    validation_summaries = [
        train_experiment(data, args.preset, args.quick)
        for data in data_views
    ]
    run_name = "quick" if args.quick else "full"
    comparison_dir = RESULTS_ROOT / args.preset / run_name
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_filename = (
        "oof_experiment_comparison.csv"
        if args.quick
        else "validation_experiment_comparison.csv"
    )
    pd.DataFrame(validation_summaries).to_csv(
        comparison_dir / comparison_filename,
        index=False,
        encoding="utf-8-sig",
    )
    print("\n" + "=" * 78)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 78)
    if args.quick:
        print(f"Comparacion OOF rapida: {comparison_dir}")
        print("VALIDATION no se ha evaluado.")
    else:
        print(f"Comparacion final en validation: {comparison_dir}")
    print("TEST continua completamente reservado.")


if __name__ == "__main__":
    main()
