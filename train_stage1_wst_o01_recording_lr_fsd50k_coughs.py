"""Stage 1: WST S0+S1 recording-level + Logistic Regression.

Entrenador paralelo para las salidas raw por ventana de
``feature_extraction_stage1_wavelet_scattering_recording.py``. Una grabacion
es una muestra: las ventanas se resumen mediante mean/std ponderadas por
``window_weight`` y max sin ponderar.

La seleccion usa exclusivamente los cinco folds de TRAIN. Primero compara
raw/log-WST y estadisticas de amplitud sin PCA; despues busca C, penalizacion,
class_weight y PCA opcional sobre las mejores representaciones. VALIDATION se
consulta solo tras congelar configuracion y umbral. TEST requiere una accion
separada y nunca participa en la seleccion.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
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
PRESET = "paper_q8_t500_window_raw_o01"
FEATURES_ROOT = ROOT / "features_extracted_stage1_fsd50k_coughs_random"
FEATURES_DIR = FEATURES_ROOT / PRESET
EXPERIMENT = "wst_o01_recording_enhanced_lr_fsd50k_coughs_random"
RESULTS_ROOT = ROOT / "results_stage1_cough_no_cough" / EXPERIMENT
GRAPHS_ROOT = ROOT / "graphs_results_stage1_cough_no_cough" / EXPERIMENT
MODELS_ROOT = ROOT / "models_stage1_cough_no_cough" / EXPERIMENT

RANDOM_STATE = 42
LOG_EPSILON = 1e-8
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)
FULL_PHASE_A_C = (1e-3, 1e-2, 1e-1)
QUICK_PHASE_A_C = (1e-2,)
FULL_PHASE_B_C = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
QUICK_PHASE_B_C = (1e-3, 1e-2, 1e-1)
FULL_PCA = (None, 32, 64, 128, 256)
QUICK_PCA = (None, 64, 128)
FULL_PENALTIES = ("l1", "l2")
QUICK_PENALTIES = ("l2",)
CLASS_WEIGHTS: tuple[str | None, ...] = (None, "balanced")
FULL_TOP_REPRESENTATIONS = 2
QUICK_TOP_REPRESENTATIONS = 2


@dataclass(frozen=True)
class RepresentationSpec:
    key: str
    transform: str
    amplitude_columns: tuple[str, ...]


@dataclass(frozen=True)
class CandidateSpec:
    representation_key: str
    pca_components: int | None
    c_value: float
    penalty: str
    class_weight: str | None

    @property
    def key(self) -> str:
        pca = "none" if self.pca_components is None else str(self.pca_components)
        weight = self.class_weight or "none"
        c_name = f"{self.c_value:g}".replace(".", "p")
        return (
            f"{self.representation_key}__pca{pca}__C{c_name}"
            f"__{self.penalty}__weight_{weight}"
        )


@dataclass
class SplitRaw:
    x_windows: np.ndarray
    windows: pd.DataFrame
    recordings: pd.DataFrame


@dataclass
class RawData:
    train: SplitRaw
    validation: SplitRaw
    path_orders: np.ndarray
    path_names: list[str]
    extraction_configuration: pd.DataFrame


@dataclass
class RecordingData:
    x_train: np.ndarray
    y_train: np.ndarray
    metadata_train: pd.DataFrame
    x_validation: np.ndarray
    y_validation: np.ndarray
    metadata_validation: pd.DataFrame
    feature_names: list[str]


@dataclass
class CandidateEvaluation:
    spec: CandidateSpec
    threshold: float
    metrics_tuned: dict[str, float | int]
    metrics_fixed: dict[str, float | int]
    subgroup_tuned: dict[str, float | int]
    probability: np.ndarray
    elapsed_seconds: float
    max_iterations: int
    input_dimension: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 WST S0+S1 recording-level + busqueda OOF de LR."
    )
    parser.add_argument(
        "--action", choices=("check", "train", "test"), default="check"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Rejilla reducida; no evalua VALIDATION.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Permite repetir una accion."
    )
    return parser.parse_args()


def representations() -> list[RepresentationSpec]:
    amplitude_sets = {
        "noamp": (),
        "peak_rms": ("log10_peak", "log10_rms"),
        "peak_rms_crest": (
            "log10_peak",
            "log10_rms",
            "log10_crest_factor",
        ),
    }
    return [
        RepresentationSpec(
            key=f"s0_s1__{transform}__{amplitude_key}",
            transform=transform,
            amplitude_columns=columns,
        )
        for transform in ("raw", "log")
        for amplitude_key, columns in amplitude_sets.items()
    ]


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


def require_split_files(split: str) -> None:
    required = (
        f"X_windows_raw_{split}.npy",
        f"metadata_windows_{split}.csv",
        f"metadata_recordings_{split}.csv",
    )
    missing = [name for name in required if not (FEATURES_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Faltan features WST de {split}: {', '.join(missing)}. "
            "Termina primero la extraccion correspondiente."
        )


def load_paths_and_configuration() -> tuple[np.ndarray, list[str], pd.DataFrame]:
    paths_path = FEATURES_DIR / "wst_paths.csv"
    config_path = FEATURES_DIR / "wst_extraction_configuration.csv"
    if not paths_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"No se encontro la configuracion del extractor en {FEATURES_DIR}."
        )
    paths = pd.read_csv(paths_path)
    config = pd.read_csv(config_path)
    config_map = dict(zip(config["parameter"].astype(str), config["value"].astype(str)))
    if int(float(config_map.get("max_order", "-1"))) != 1:
        raise ValueError("Esta prueba requiere max_order=1.")
    if int(float(config_map.get("path_count", "-1"))) != 95:
        raise ValueError("Se esperaban 95 paths S0+S1.")
    if config_map.get("recording_pooling_applied", "").lower() != "false":
        raise ValueError("La entrada debe estar antes del pooling recording-level.")
    if config_map.get("near_silence_policy") != "reject":
        raise ValueError(
            "Regenera con --near-silence-policy reject para comparar con MFCC."
        )
    orders = paths["scattering_order"].to_numpy(dtype=int)
    if len(orders) != 95 or np.bincount(orders, minlength=2).tolist()[:2] != [1, 94]:
        raise ValueError("Distribucion de paths inesperada; se esperaba S0=1, S1=94.")
    names = [f"path_{int(index):03d}" for index in paths["path_index"]]
    return orders, names, config


def load_raw_split(split: str, path_count: int) -> SplitRaw:
    require_split_files(split)
    x_windows = np.load(FEATURES_DIR / f"X_windows_raw_{split}.npy", mmap_mode="r")
    windows = pd.read_csv(
        FEATURES_DIR / f"metadata_windows_{split}.csv", low_memory=False
    )
    recordings = pd.read_csv(
        FEATURES_DIR / f"metadata_recordings_{split}.csv", low_memory=False
    )
    if x_windows.shape != (len(windows), path_count):
        raise ValueError(f"X_windows_raw_{split} inesperado: {x_windows.shape}.")
    if windows["feature_row"].tolist() != list(range(len(windows))):
        raise ValueError(f"feature_row no esta alineado en {split}.")
    if not np.isfinite(np.asarray(x_windows)).all():
        raise ValueError(f"X_windows_raw_{split} contiene NaN/Inf.")
    required_windows = {
        "original_uuid",
        "split_group",
        "stage1_target",
        "fold",
        "dataset_origin",
        "is_new_fsd50k_cough",
        "stage2_eligible",
        "window_weight",
        "valid_fraction",
        "log10_peak",
        "log10_rms",
        "log10_crest_factor",
    }
    required_recordings = required_windows - {
        "window_weight",
        "valid_fraction",
        "log10_peak",
        "log10_rms",
        "log10_crest_factor",
    }
    if required_windows - set(windows.columns):
        raise ValueError(
            f"Faltan columnas de ventanas en {split}: "
            f"{sorted(required_windows - set(windows.columns))}"
        )
    if required_recordings - set(recordings.columns):
        raise ValueError(
            f"Faltan columnas de grabaciones en {split}: "
            f"{sorted(required_recordings - set(recordings.columns))}"
        )
    if recordings["original_uuid"].astype(str).duplicated().any():
        raise ValueError(f"Hay grabaciones duplicadas en {split}.")
    window_ids = set(windows["original_uuid"].astype(str))
    recording_ids = set(recordings["original_uuid"].astype(str))
    if window_ids != recording_ids:
        raise ValueError(
            f"Ventanas/grabaciones no contienen los mismos UUID en {split}."
        )
    for frame in (windows, recordings):
        frame["is_new_fsd50k_cough"] = parse_bool_series(
            frame["is_new_fsd50k_cough"], "is_new_fsd50k_cough"
        )
        frame["stage2_eligible"] = parse_bool_series(
            frame["stage2_eligible"], "stage2_eligible"
        )
    invalid_new = recordings["is_new_fsd50k_cough"] & (
        recordings["dataset_origin"].astype(str).ne("FSD50K")
        | recordings["stage1_target"].astype(int).ne(1)
        | recordings["stage2_eligible"]
    )
    if invalid_new.any():
        raise ValueError(f"Nuevas toses FSD50K mal configuradas en {split}.")
    return SplitRaw(x_windows=x_windows, windows=windows, recordings=recordings)


def validate_isolation(train: SplitRaw, validation: SplitRaw) -> None:
    for column in ("original_uuid", "split_group"):
        overlap = set(train.recordings[column].astype(str)) & set(
            validation.recordings[column].astype(str)
        )
        if overlap:
            raise ValueError(f"Fuga de {column} TRAIN/VALIDATION: {len(overlap)}.")
    train_folds = set(train.recordings["fold"].astype(int))
    if train_folds != {0, 1, 2, 3, 4}:
        raise ValueError(f"Folds TRAIN inesperados: {sorted(train_folds)}.")
    if set(validation.recordings["fold"].astype(int)) != {-1}:
        raise ValueError("VALIDATION debe tener fold=-1.")
    grouped = train.recordings.groupby("split_group")["fold"].nunique()
    if (grouped > 1).any():
        raise ValueError("Un split_group de TRAIN aparece en varios folds.")


def load_raw_train_validation() -> RawData:
    orders, names, config = load_paths_and_configuration()
    train = load_raw_split("train", len(orders))
    validation = load_raw_split("validation", len(orders))
    validate_isolation(train, validation)
    return RawData(train, validation, orders, names, config)


def pool_split(
    split: SplitRaw,
    spec: RepresentationSpec,
    path_orders: np.ndarray,
    path_names: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    values = np.asarray(split.x_windows, dtype=np.float32).copy()
    if spec.transform == "log":
        first_order = path_orders == 1
        values[:, first_order] = np.log10(
            np.maximum(values[:, first_order], LOG_EPSILON)
        )
    elif spec.transform != "raw":
        raise ValueError(f"Transformacion desconocida: {spec.transform}.")

    pooled_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    recording_lookup = split.recordings.set_index(
        split.recordings["original_uuid"].astype(str), drop=False
    )
    for original_uuid, group in split.windows.groupby("original_uuid", sort=False):
        original_uuid = str(original_uuid)
        indices = group["feature_row"].to_numpy(dtype=int)
        group_values = values[indices]
        weights = group["window_weight"].to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError(f"window_weight invalido para {original_uuid}.")
        weights /= weights.sum()
        mean = np.sum(group_values * weights[:, None], axis=0)
        variance = np.sum(
            ((group_values - mean) ** 2) * weights[:, None], axis=0
        )
        blocks = [mean, np.sqrt(np.maximum(variance, 0.0)), group_values.max(axis=0)]
        if spec.amplitude_columns:
            amplitude = group[list(spec.amplitude_columns)].to_numpy(dtype=np.float32)
            if not np.isfinite(amplitude).all():
                raise ValueError(f"Amplitud no finita para {original_uuid}.")
            amp_mean = np.sum(amplitude * weights[:, None], axis=0)
            amp_variance = np.sum(
                ((amplitude - amp_mean) ** 2) * weights[:, None], axis=0
            )
            blocks.extend(
                [
                    amp_mean,
                    np.sqrt(np.maximum(amp_variance, 0.0)),
                    amplitude.max(axis=0),
                ]
            )
        pooled_rows.append(np.concatenate(blocks).astype(np.float32))
        recording = recording_lookup.loc[original_uuid]
        metadata_rows.append(
            {
                **recording.to_dict(),
                "original_uuid": original_uuid,
                "window_count_pooled": len(group),
                "window_weight_sum_pooled": float(group["window_weight"].sum()),
                "mean_valid_fraction": float(group["valid_fraction"].mean()),
            }
        )
    feature_names = [
        f"wst_{stat}__{path}"
        for stat in ("mean", "std", "max")
        for path in path_names
    ]
    feature_names.extend(
        f"amplitude_{stat}__{column}"
        for stat in ("mean", "std", "max")
        for column in spec.amplitude_columns
    )
    x = np.vstack(pooled_rows).astype(np.float32)
    metadata = pd.DataFrame(metadata_rows).reset_index(drop=True)
    y = metadata["stage1_target"].to_numpy(dtype=np.int8)
    if x.shape != (len(metadata), len(feature_names)):
        raise RuntimeError("Dimensiones recording-level inconsistentes.")
    return x, y, metadata, feature_names


def build_recording_data(raw: RawData, spec: RepresentationSpec) -> RecordingData:
    x_train, y_train, metadata_train, names = pool_split(
        raw.train, spec, raw.path_orders, raw.path_names
    )
    x_validation, y_validation, metadata_validation, validation_names = pool_split(
        raw.validation, spec, raw.path_orders, raw.path_names
    )
    if names != validation_names:
        raise RuntimeError("TRAIN/VALIDATION tienen features distintas.")
    return RecordingData(
        x_train,
        y_train,
        metadata_train,
        x_validation,
        y_validation,
        metadata_validation,
        names,
    )


def make_classifier(spec: CandidateSpec) -> LogisticRegression:
    return LogisticRegression(
        C=spec.c_value,
        penalty=spec.penalty,
        solver="liblinear",
        class_weight=spec.class_weight,
        max_iter=5_000,
        random_state=RANDOM_STATE,
    )


def subgroup_metrics(
    metadata: pd.DataFrame, y: np.ndarray, prediction: np.ndarray
) -> dict[str, float | int]:
    is_new = metadata["is_new_fsd50k_cough"].astype(bool).to_numpy()
    original_cough = (y == 1) & ~is_new
    new_cough = (y == 1) & is_new

    def recall(mask: np.ndarray) -> float:
        return float(np.mean(prediction[mask] == 1)) if mask.any() else np.nan

    return {
        "original_cough_count": int(original_cough.sum()),
        "original_cough_recall": recall(original_cough),
        "new_fsd50k_cough_count": int(new_cough.sum()),
        "new_fsd50k_cough_recall": recall(new_cough),
    }


def tune_threshold(
    y: np.ndarray, probability: np.ndarray
) -> tuple[float, dict[str, float | int]]:
    best = None
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
        raise RuntimeError("No se pudo seleccionar el umbral.")
    return best[1], best[2]


def transformed_folds(
    data: RecordingData,
    pca_options: tuple[int | None, ...],
    representation_key: str,
) -> tuple[
    dict[tuple[int, int | None], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    pd.DataFrame,
]:
    folds = data.metadata_train["fold"].to_numpy(dtype=int)
    cache = {}
    variance_rows = []
    for fold in sorted(np.unique(folds)):
        validation_mask = folds == fold
        training_mask = ~validation_mask
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(data.x_train[training_mask]).astype(np.float32)
        x_outer = scaler.transform(data.x_train[validation_mask]).astype(np.float32)
        cache[(fold, None)] = (x_fit, x_outer, training_mask, validation_mask)
        for components in pca_options:
            if components is None or components >= min(x_fit.shape):
                continue
            pca = PCA(
                n_components=components,
                whiten=False,
                svd_solver="randomized",
                random_state=RANDOM_STATE,
            )
            cache[(fold, components)] = (
                pca.fit_transform(x_fit).astype(np.float32),
                pca.transform(x_outer).astype(np.float32),
                training_mask,
                validation_mask,
            )
            variance_rows.append(
                {
                    "representation_key": representation_key,
                    "fold": fold,
                    "pca_components": components,
                    "explained_variance_ratio": float(
                        pca.explained_variance_ratio_.sum()
                    ),
                }
            )
    return cache, pd.DataFrame(variance_rows)


def evaluate_grid(
    data: RecordingData,
    representation: RepresentationSpec,
    c_values: tuple[float, ...],
    pca_options: tuple[int | None, ...],
    penalties: tuple[str, ...],
    class_weights: tuple[str | None, ...],
) -> tuple[list[CandidateEvaluation], pd.DataFrame]:
    cache, variances = transformed_folds(
        data, pca_options, representation.key
    )
    folds = data.metadata_train["fold"].to_numpy(dtype=int)
    results = []
    for components in pca_options:
        if any((fold, components) not in cache for fold in np.unique(folds)):
            continue
        for c_value in c_values:
            for penalty in penalties:
                for class_weight in class_weights:
                    spec = CandidateSpec(
                        representation.key,
                        components,
                        c_value,
                        penalty,
                        class_weight,
                    )
                    started = time.perf_counter()
                    probability = np.full(len(data.y_train), np.nan, dtype=float)
                    max_iterations = 0
                    for fold in sorted(np.unique(folds)):
                        x_fit, x_outer, training_mask, validation_mask = cache[
                            (fold, components)
                        ]
                        classifier = make_classifier(spec)
                        classifier.fit(x_fit, data.y_train[training_mask])
                        probability[validation_mask] = classifier.predict_proba(
                            x_outer
                        )[:, 1]
                        max_iterations = max(
                            max_iterations, int(classifier.n_iter_.max())
                        )
                    if not np.isfinite(probability).all():
                        raise RuntimeError(f"OOF incompleto para {spec.key}.")
                    threshold, metrics = tune_threshold(data.y_train, probability)
                    fixed_prediction = (probability >= 0.5).astype(np.int8)
                    prediction = (probability >= threshold).astype(np.int8)
                    result = CandidateEvaluation(
                        spec=spec,
                        threshold=threshold,
                        metrics_tuned=metrics,
                        metrics_fixed=classification_metrics(
                            data.y_train, fixed_prediction, probability
                        ),
                        subgroup_tuned=subgroup_metrics(
                            data.metadata_train, data.y_train, prediction
                        ),
                        probability=probability,
                        elapsed_seconds=time.perf_counter() - started,
                        max_iterations=max_iterations,
                        input_dimension=data.x_train.shape[1],
                    )
                    results.append(result)
                    print(
                        f"{spec.key} | macro-F1={metrics['macro_f1']:.4f} "
                        f"| bal-acc={metrics['balanced_accuracy']:.4f} "
                        f"| recall tos={metrics['recall_cough']:.4f} "
                        f"| recall FSD+={result.subgroup_tuned['new_fsd50k_cough_recall']:.4f} "
                        f"| AUC={metrics['roc_auc']:.4f} | t={threshold:.2f}"
                    )
    return results, variances


def selection_key(item: CandidateEvaluation) -> tuple[float, ...]:
    model_dimension = (
        item.input_dimension
        if item.spec.pca_components is None
        else item.spec.pca_components
    )
    return (
        float(item.metrics_tuned["macro_f1"]),
        float(item.metrics_tuned["balanced_accuracy"]),
        float(item.metrics_tuned["roc_auc"]),
        -float(model_dimension),
        -float(item.spec.c_value),
    )


def evaluation_row(
    item: CandidateEvaluation, selected: bool = False
) -> dict[str, object]:
    return {
        "candidate": item.spec.key,
        **asdict(item.spec),
        "input_dimension": item.input_dimension,
        "output_dimension": (
            item.input_dimension
            if item.spec.pca_components is None
            else item.spec.pca_components
        ),
        "threshold": item.threshold,
        "elapsed_seconds": item.elapsed_seconds,
        "max_iterations": item.max_iterations,
        "selected": selected,
        **{f"oof_tuned__{key}": value for key, value in item.metrics_tuned.items()},
        **{f"oof_0p5__{key}": value for key, value in item.metrics_fixed.items()},
        **{f"oof_tuned__{key}": value for key, value in item.subgroup_tuned.items()},
    }


def build_pipeline(spec: CandidateSpec) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if spec.pca_components is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=spec.pca_components,
                    whiten=False,
                    svd_solver="randomized",
                    random_state=RANDOM_STATE,
                ),
            )
        )
    steps.append(("classifier", make_classifier(spec)))
    return Pipeline(steps)


def save_predictions(
    path: Path,
    metadata: pd.DataFrame,
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> None:
    output = metadata.copy()
    output["y_true"] = y
    output["probability_cough"] = probability
    output["y_pred"] = (probability >= threshold).astype(np.int8)
    output["threshold"] = threshold
    output.to_csv(path, index=False, encoding="utf-8-sig")


def save_validation_graph(
    path: Path, y: np.ndarray, probability: np.ndarray, threshold: float
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
        title=f"Validation - umbral {threshold:.2f}",
    )
    fpr, tpr, _ = roc_curve(y, probability)
    auc = roc_auc_score(y, probability)
    axes[1].plot(fpr, tpr, label=f"AUC={auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set(xlabel="FPR", ylabel="Recall tos", title="Curva ROC")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def benchmark_model(model: Pipeline, x: np.ndarray) -> float:
    sample = x[:1]
    for _ in range(5):
        model.predict_proba(sample)
    started = time.perf_counter()
    repetitions = 200
    for _ in range(repetitions):
        model.predict_proba(sample)
    return (time.perf_counter() - started) * 1_000 / repetitions


def check(raw: RawData) -> None:
    print("=" * 78)
    print("CHECK - STAGE 1 WST S0+S1 RECORDING + LR")
    print("=" * 78)
    print(f"Features: {FEATURES_DIR}")
    print(f"Ventanas TRAIN:      {raw.train.x_windows.shape}")
    print(f"Ventanas VALIDATION: {raw.validation.x_windows.shape}")
    for name, split in (("TRAIN", raw.train), ("VALIDATION", raw.validation)):
        labels = split.recordings["stage1_target"].to_numpy(dtype=int)
        print(
            f"{name}: grabaciones={len(labels)}; no_tos/tos="
            f"{np.bincount(labels, minlength=2).tolist()}; "
            f"nuevas_toses_FSD50K="
            f"{int(split.recordings['is_new_fsd50k_cough'].sum())}"
        )
    print(f"Paths por orden: {dict(pd.Series(raw.path_orders).value_counts().sort_index())}")
    print("Una fila y una etiqueta por original_uuid despues del pooling.")
    print("TEST no se ha leido.")


def output_paths(quick: bool) -> tuple[Path, Path, Path, Path]:
    mode = "quick" if quick else "full"
    results = RESULTS_ROOT / mode
    graphs = GRAPHS_ROOT / mode
    models = MODELS_ROOT / mode
    return results, graphs, models, models / "stage1_wst_o01_recording_lr.joblib"


def update_excel() -> None:
    try:
        from build_stage1_experiments_excel import build_workbook

        path = build_workbook()
        print(f"Excel actualizado: {path}")
    except Exception as exc:
        print(f"AVISO: no se pudo actualizar el Excel automaticamente: {exc}")


def train(raw: RawData, quick: bool, overwrite: bool) -> None:
    results_dir, graphs_dir, models_dir, model_path = output_paths(quick)
    if model_path.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {model_path}. Usa --overwrite para repetirlo."
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rep_specs = representations()
    views: dict[str, RecordingData] = {}
    phase_a_results: list[CandidateEvaluation] = []
    phase_a_c = QUICK_PHASE_A_C if quick else FULL_PHASE_A_C
    print("\n" + "=" * 78)
    print("FASE A - RAW/LOG Y AMPLITUD (SIN PCA)")
    print("=" * 78)
    for index, rep in enumerate(rep_specs, start=1):
        print(f"\n[{index}/{len(rep_specs)}] {rep.key}")
        data = build_recording_data(raw, rep)
        views[rep.key] = data
        evaluations, _ = evaluate_grid(
            data,
            rep,
            phase_a_c,
            (None,),
            ("l2",),
            (None,),
        )
        phase_a_results.extend(evaluations)
    pd.DataFrame(evaluation_row(item) for item in phase_a_results).to_csv(
        results_dir / "phase_a_candidate_cv_results.csv", index=False
    )
    rep_winners = []
    for rep in rep_specs:
        candidates = [
            item for item in phase_a_results if item.spec.representation_key == rep.key
        ]
        rep_winners.append(max(candidates, key=selection_key))
    rep_winners.sort(key=selection_key, reverse=True)
    top_count = QUICK_TOP_REPRESENTATIONS if quick else FULL_TOP_REPRESENTATIONS
    selected_rep_keys = [item.spec.representation_key for item in rep_winners[:top_count]]
    pd.DataFrame(
        {
            **evaluation_row(item),
            "selected_for_phase_b": item.spec.representation_key in selected_rep_keys,
        }
        for item in rep_winners
    ).to_csv(results_dir / "phase_a_representation_winners.csv", index=False)
    print("\nRepresentaciones elegidas para Fase B:")
    for key in selected_rep_keys:
        print(f"  - {key}")

    c_values = QUICK_PHASE_B_C if quick else FULL_PHASE_B_C
    pca_options = QUICK_PCA if quick else FULL_PCA
    penalties = QUICK_PENALTIES if quick else FULL_PENALTIES
    phase_b_results: list[CandidateEvaluation] = []
    variance_frames = []
    rep_lookup = {rep.key: rep for rep in rep_specs}
    print("\n" + "=" * 78)
    print("FASE B - PCA, C, PENALIZACION Y CLASS_WEIGHT")
    print("=" * 78)
    for index, key in enumerate(selected_rep_keys, start=1):
        print(f"\n[{index}/{len(selected_rep_keys)}] {key}")
        evaluations, variances = evaluate_grid(
            views[key],
            rep_lookup[key],
            c_values,
            pca_options,
            penalties,
            CLASS_WEIGHTS,
        )
        phase_b_results.extend(evaluations)
        if not variances.empty:
            variance_frames.append(variances)
    winner = max(phase_b_results, key=selection_key)
    pd.DataFrame(
        evaluation_row(item, item.spec.key == winner.spec.key)
        for item in phase_b_results
    ).to_csv(results_dir / "phase_b_candidate_cv_results.csv", index=False)
    if variance_frames:
        pd.concat(variance_frames, ignore_index=True).to_csv(
            results_dir / "pca_explained_variance_by_fold.csv", index=False
        )

    winner_data = views[winner.spec.representation_key]
    winner_rep = rep_lookup[winner.spec.representation_key]
    oof_prediction = (winner.probability >= winner.threshold).astype(np.int8)
    save_predictions(
        results_dir / "best_oof_predictions.csv",
        winner_data.metadata_train,
        winner_data.y_train,
        winner.probability,
        winner.threshold,
    )
    fold_rows = []
    folds = winner_data.metadata_train["fold"].to_numpy(dtype=int)
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        fold_rows.append(
            {
                "fold": fold,
                **classification_metrics(
                    winner_data.y_train[mask],
                    oof_prediction[mask],
                    winner.probability[mask],
                ),
                **subgroup_metrics(
                    winner_data.metadata_train.loc[mask].reset_index(drop=True),
                    winner_data.y_train[mask],
                    oof_prediction[mask],
                ),
            }
        )
    pd.DataFrame(fold_rows).to_csv(
        results_dir / "best_cv_fold_metrics.csv", index=False
    )

    if quick:
        print("\nPrueba rapida completada.")
        print(f"Ganador OOF: {winner.spec.key}")
        print(f"Macro-F1 OOF: {winner.metrics_tuned['macro_f1']:.4f}")
        print(f"Recall tos OOF: {winner.metrics_tuned['recall_cough']:.4f}")
        print(
            "Recall nuevas toses FSD50K OOF: "
            f"{winner.subgroup_tuned['new_fsd50k_cough_recall']:.4f}"
        )
        print("VALIDATION y TEST no se han evaluado.")
        print(f"Resultados: {results_dir}")
        return

    print("\nAjustando scaler/PCA/LR final exclusivamente con todo TRAIN...")
    pipeline = build_pipeline(winner.spec)
    started = time.perf_counter()
    pipeline.fit(winner_data.x_train, winner_data.y_train)
    fit_seconds = time.perf_counter() - started
    validation_probability = pipeline.predict_proba(winner_data.x_validation)[:, 1]
    validation_prediction = (
        validation_probability >= winner.threshold
    ).astype(np.int8)
    validation_metrics = classification_metrics(
        winner_data.y_validation, validation_prediction, validation_probability
    )
    validation_subgroups = subgroup_metrics(
        winner_data.metadata_validation,
        winner_data.y_validation,
        validation_prediction,
    )
    save_predictions(
        results_dir / "validation_predictions.csv",
        winner_data.metadata_validation,
        winner_data.y_validation,
        validation_probability,
        winner.threshold,
    )
    save_validation_graph(
        graphs_dir / "validation_wst_o01_recording_lr.png",
        winner_data.y_validation,
        validation_probability,
        winner.threshold,
    )

    package = {
        "model": pipeline,
        "threshold": winner.threshold,
        "candidate": asdict(winner.spec),
        "representation": asdict(winner_rep),
        "feature_names": winner_data.feature_names,
        "pooling": "weighted_mean_weighted_std_unweighted_max",
        "window_weight_column": "window_weight",
        "log_epsilon": LOG_EPSILON,
        "preset": PRESET,
        "classes": {0: "no_cough", 1: "cough"},
    }
    joblib.dump(package, model_path, compress=3)
    model_size_kb = model_path.stat().st_size / 1024.0
    inference_ms = benchmark_model(pipeline, winner_data.x_validation)
    oof_metrics = classification_metrics(
        winner_data.y_train, oof_prediction, winner.probability
    )
    metrics_rows = [
        {
            "dataset": "train_oof",
            "threshold": winner.threshold,
            **oof_metrics,
            **winner.subgroup_tuned,
        },
        {
            "dataset": "validation",
            "threshold": winner.threshold,
            **validation_metrics,
            **validation_subgroups,
        },
    ]
    pd.DataFrame(metrics_rows).to_csv(
        results_dir / "metrics_summary.csv", index=False
    )
    final_pca_variance = np.nan
    if "pca" in pipeline.named_steps:
        final_pca_variance = float(
            pipeline.named_steps["pca"].explained_variance_ratio_.sum()
        )
    configuration = {
        "experiment": EXPERIMENT,
        "feature_source": str(FEATURES_DIR),
        "feature_preset": PRESET,
        "winner": winner.spec.key,
        "representation": asdict(winner_rep),
        "selection_rule": "OOF TRAIN macro-F1; ties balanced accuracy, AUC, size, C",
        "threshold": winner.threshold,
        "threshold_selection": "OOF TRAIN only",
        "sample_unit": "one recording / original_uuid",
        "pooling": "weighted mean/std and unweighted max",
        "input_dimension": winner_data.x_train.shape[1],
        "model_dimension": (
            winner_data.x_train.shape[1]
            if winner.spec.pca_components is None
            else winner.spec.pca_components
        ),
        "final_pca_explained_variance": final_pca_variance,
        "fit_final_seconds": fit_seconds,
        "model_size_kb": model_size_kb,
        "desktop_classifier_inference_ms": inference_ms,
        "wst_extraction_time_included_in_benchmark": False,
        "stage2_use_of_new_fsd50k_coughs": False,
        "test_evaluated": False,
    }
    pd.DataFrame(
        {
            "parameter": key,
            "value": json.dumps(value) if isinstance(value, dict) else value,
        }
        for key, value in configuration.items()
    ).to_csv(results_dir / "experiment_configuration.csv", index=False)
    print("\n" + "=" * 78)
    print("RESULTADO STAGE 1 - WST S0+S1 RECORDING + LR")
    print("=" * 78)
    print(f"Ganador OOF: {winner.spec.key}")
    print(f"Umbral OOF congelado: {winner.threshold:.2f}")
    print(f"Macro-F1 OOF: {oof_metrics['macro_f1']:.4f}")
    print(f"Macro-F1 validation: {validation_metrics['macro_f1']:.4f}")
    print(
        "Recall validation no_tos/tos: "
        f"{validation_metrics['specificity_no_cough']:.4f} / "
        f"{validation_metrics['recall_cough']:.4f}"
    )
    print(
        "Recall nuevas toses FSD50K validation: "
        f"{validation_subgroups['new_fsd50k_cough_recall']:.4f}"
    )
    print(f"Modelo joblib: {model_size_kb:.2f} KB")
    print(f"Inferencia del clasificador: {inference_ms:.4f} ms/grabacion")
    print("El benchmark no incluye calcular WST desde el audio.")
    print(f"Resultados: {results_dir}")
    print("TEST permanece reservado. Para evaluarlo: --action test")
    update_excel()


def evaluate_test(overwrite: bool) -> None:
    results_dir, _, _, model_path = output_paths(False)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No existe {model_path}. Ejecuta primero --action train."
        )
    metrics_path = results_dir / "metrics_summary.csv"
    configuration_path = results_dir / "experiment_configuration.csv"
    if not metrics_path.is_file() or not configuration_path.is_file():
        raise FileNotFoundError("Faltan resultados del entrenamiento completo.")
    metrics_frame = pd.read_csv(metrics_path)
    if metrics_frame["dataset"].astype(str).eq("test").any() and not overwrite:
        raise FileExistsError("TEST ya fue evaluado; usa --overwrite para repetir.")
    package = joblib.load(model_path)
    orders, names, _ = load_paths_and_configuration()
    raw_test = load_raw_split("test", len(orders))
    rep = RepresentationSpec(**package["representation"])
    x_test, y_test, metadata_test, feature_names = pool_split(
        raw_test, rep, orders, names
    )
    if feature_names != package["feature_names"]:
        raise ValueError("Las features de TEST no coinciden con el modelo.")
    model = package["model"]
    threshold = float(package["threshold"])
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= threshold).astype(np.int8)
    metrics = classification_metrics(y_test, prediction, probability)
    subgroups = subgroup_metrics(metadata_test, y_test, prediction)
    metrics_frame = metrics_frame.loc[
        ~metrics_frame["dataset"].astype(str).eq("test")
    ]
    pd.concat(
        [
            metrics_frame,
            pd.DataFrame(
                [{"dataset": "test", "threshold": threshold, **metrics, **subgroups}]
            ),
        ],
        ignore_index=True,
    ).to_csv(metrics_path, index=False)
    save_predictions(
        results_dir / "test_predictions.csv",
        metadata_test,
        y_test,
        probability,
        threshold,
    )
    configuration = pd.read_csv(configuration_path)
    configuration.loc[
        configuration["parameter"].eq("test_evaluated"), "value"
    ] = True
    configuration.to_csv(configuration_path, index=False)
    print("=" * 78)
    print("EVALUACION FINAL TEST - STAGE 1 WST S0+S1 RECORDING + LR")
    print("=" * 78)
    print(
        f"TEST: macro-F1={metrics['macro_f1']:.4f} | "
        f"recall tos={metrics['recall_cough']:.4f} | "
        f"especificidad={metrics['specificity_no_cough']:.4f} | "
        f"AUC={metrics['roc_auc']:.4f}"
    )
    print(
        f"TEST original_cough: recall={subgroups['original_cough_recall']:.4f} "
        f"(n={subgroups['original_cough_count']})"
    )
    print(
        f"TEST new_fsd50k_cough: recall={subgroups['new_fsd50k_cough_recall']:.4f} "
        f"(n={subgroups['new_fsd50k_cough_count']})"
    )
    print(f"Resultados: {results_dir}")
    update_excel()


def main() -> None:
    args = parse_args()
    print("=" * 78)
    print("STAGE 1 - WST S0+S1 RECORDING + LOGISTIC REGRESSION")
    print("=" * 78)
    print(f"Accion: {args.action}")
    if args.action == "test":
        evaluate_test(args.overwrite)
        return
    raw = load_raw_train_validation()
    check(raw)
    if args.action == "train":
        train(raw, args.quick, args.overwrite)


if __name__ == "__main__":
    main()
