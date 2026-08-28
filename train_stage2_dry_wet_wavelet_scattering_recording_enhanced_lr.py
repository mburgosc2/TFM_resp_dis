"""Stage 2: WST recording-level mejorado + Logistic Regression.

Consume la extraccion raw por evento creada por
``feature_extraction_stage2_wavelet_scattering_recording_enhanced.py`` y
construye una sola fila por grabacion. El pooling usa mean y desviacion
ponderadas por la fraccion real observada de cada evento, y max sin ponderar.

La busqueda se realiza exclusivamente con los cinco folds de TRAIN:

1. Fase A: compara ordenes WST, raw/log-WST y estadisticas de amplitud sin PCA.
2. Fase B: sobre las mejores representaciones busca C y PCA opcional.

StandardScaler, PCA y LR se ajustan dentro de cada fold. El umbral se obtiene
con las predicciones OOF de TRAIN. VALIDATION se evalua una sola vez con la
configuracion y el umbral congelados. TEST no se lee ni se procesa.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording


SCRIPT_DIR = Path(__file__).resolve().parent
PRESET = "paper_q8_q1_t500_window_raw_o012_weighted"
FEATURE_ROOT = (
    SCRIPT_DIR
    / "features_extracted_stage2_dry_wet_wavelet_scattering_recording_enhanced"
)
RESULTS_ROOT = (
    SCRIPT_DIR
    / "results_stage2_dry_wet_wavelet_scattering_recording_enhanced_lr"
)
MODELS_ROOT = (
    SCRIPT_DIR
    / "models_stage2_dry_wet_wavelet_scattering_recording_enhanced_lr"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wavelet_scattering_recording_enhanced_lr"
)

RANDOM_STATE = 42
LOG_EPSILON = 1e-8
FULL_PHASE_A_C = (3e-4, 1e-3, 3e-3)
QUICK_PHASE_A_C = (1e-3,)
FULL_PHASE_B_C = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
QUICK_PHASE_B_C = (1e-3, 1e-2)
FULL_PCA = (None, 32, 64, 128, 256)
QUICK_PCA = (None, 64, 128)
FULL_TOP_REPRESENTATIONS = 3
QUICK_TOP_REPRESENTATIONS = 2


@dataclass(frozen=True)
class RepresentationSpec:
    key: str
    orders: tuple[int, ...]
    transform: str
    amplitude_columns: tuple[str, ...]


@dataclass(frozen=True)
class CandidateSpec:
    representation_key: str
    c_value: float
    pca_components: int | None

    @property
    def key(self) -> str:
        pca = "none" if self.pca_components is None else str(self.pca_components)
        return f"{self.representation_key}__C{self.c_value:g}__pca{pca}"


@dataclass
class RawData:
    x_train: np.ndarray
    metadata_train: pd.DataFrame
    x_validation: np.ndarray
    metadata_validation: pd.DataFrame
    path_orders: np.ndarray
    path_names: list[str]
    extraction_configuration: pd.DataFrame


@dataclass
class CandidateEvaluation:
    spec: CandidateSpec
    threshold: float
    metrics_tuned: dict[str, float | int]
    metrics_native: dict[str, float | int]
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    elapsed_seconds: float
    input_dimension: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WST Stage 2 recording mejorado con PCA opcional y LR."
    )
    parser.add_argument("--action", choices=("check", "train"), default="check")
    parser.add_argument("--preset", default=PRESET)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Rejilla reducida; no evalua VALIDATION.",
    )
    return parser


def representations() -> list[RepresentationSpec]:
    order_sets = {
        "s1": (1,),
        "s0_s1": (0, 1),
        "s1_s2": (1, 2),
        "s0_s1_s2": (0, 1, 2),
    }
    amplitude_sets = {
        "noamp": (),
        "peak_rms": ("log10_peak", "log10_rms"),
        "peak_rms_crest": (
            "log10_peak",
            "log10_rms",
            "log10_crest_factor",
        ),
    }
    result = []
    for order_key, orders in order_sets.items():
        for transform in ("raw", "log"):
            for amplitude_key, columns in amplitude_sets.items():
                result.append(
                    RepresentationSpec(
                        key=f"{order_key}__{transform}__{amplitude_key}",
                        orders=orders,
                        transform=transform,
                        amplitude_columns=columns,
                    )
                )
    return result


def require_files(feature_dir: Path) -> None:
    required = (
        "X_events_raw_train.npy",
        "X_events_raw_validation.npy",
        "y_events_train.npy",
        "y_events_validation.npy",
        "folds_events_train.npy",
        "folds_events_validation.npy",
        "metadata_events_train.csv",
        "metadata_events_validation.csv",
        "wst_paths.csv",
        "wst_extraction_configuration.csv",
    )
    missing = [name for name in required if not (feature_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan salidas del extractor WST mejorado: " + ", ".join(missing)
        )


def load_raw_data(preset: str) -> RawData:
    feature_dir = FEATURE_ROOT / preset
    require_files(feature_dir)
    x_train = np.load(feature_dir / "X_events_raw_train.npy", mmap_mode="r")
    x_validation = np.load(
        feature_dir / "X_events_raw_validation.npy", mmap_mode="r"
    )
    y_train = np.load(feature_dir / "y_events_train.npy")
    y_validation = np.load(feature_dir / "y_events_validation.npy")
    folds_train = np.load(feature_dir / "folds_events_train.npy")
    folds_validation = np.load(feature_dir / "folds_events_validation.npy")
    metadata_train = pd.read_csv(feature_dir / "metadata_events_train.csv")
    metadata_validation = pd.read_csv(
        feature_dir / "metadata_events_validation.csv"
    )
    paths = pd.read_csv(feature_dir / "wst_paths.csv")
    configuration = pd.read_csv(
        feature_dir / "wst_extraction_configuration.csv"
    )

    for split, x_values, y_values, folds, metadata in (
        ("train", x_train, y_train, folds_train, metadata_train),
        (
            "validation",
            x_validation,
            y_validation,
            folds_validation,
            metadata_validation,
        ),
    ):
        if x_values.ndim != 2 or x_values.shape[0] != len(metadata):
            raise ValueError(f"Dimensiones inconsistentes en {split}: {x_values.shape}.")
        if x_values.shape[1] != len(paths):
            raise ValueError(f"Paths WST inconsistentes en {split}.")
        if len(y_values) != len(metadata) or len(folds) != len(metadata):
            raise ValueError(f"y/folds inconsistentes en {split}.")
        if not np.array_equal(
            y_values.astype(int), metadata["stage2_target"].to_numpy(dtype=int)
        ):
            raise ValueError(f"Etiquetas inconsistentes en {split}.")
        if not np.array_equal(folds.astype(int), metadata["fold"].to_numpy(dtype=int)):
            raise ValueError(f"Folds inconsistentes en {split}.")
        if metadata["feature_row"].tolist() != list(range(len(metadata))):
            raise ValueError(f"feature_row no es consecutivo en {split}.")
        if not np.isfinite(np.asarray(x_values)).all():
            raise ValueError(f"La matriz {split} contiene NaN/Inf.")
        origins = set(metadata["dataset_origin"].astype(str).str.upper())
        if origins != {"COUGHVID"}:
            raise ValueError(f"Stage 2 solo admite COUGHVID; {split}: {origins}")
        if metadata["source_audio_path"].astype(str).str.contains(
            "FSD50K", case=False, regex=False
        ).any():
            raise ValueError(f"{split} contiene rutas FSD50K.")

    train_folds = set(metadata_train["fold"].astype(int))
    if train_folds != set(common.EXPECTED_TRAIN_FOLDS):
        raise ValueError(f"Folds TRAIN inesperados: {sorted(train_folds)}")
    if set(metadata_validation["fold"].astype(int)) != {-1}:
        raise ValueError("VALIDATION debe tener fold=-1.")
    overlap = set(metadata_train["original_uuid"]) & set(
        metadata_validation["original_uuid"]
    )
    if overlap:
        raise ValueError(f"TRAIN/VALIDATION comparten {len(overlap)} UUID.")
    path_orders = paths["scattering_order"].to_numpy(dtype=int)
    if set(path_orders) != {0, 1, 2}:
        raise ValueError(f"Ordenes WST inesperados: {sorted(set(path_orders))}")
    path_names = [f"path_{int(value):03d}" for value in paths["path_index"]]
    return RawData(
        x_train=x_train,
        metadata_train=metadata_train,
        x_validation=x_validation,
        metadata_validation=metadata_validation,
        path_orders=path_orders,
        path_names=path_names,
        extraction_configuration=configuration,
    )


def validate_group_constant(group: pd.DataFrame, column: str) -> object:
    values = group[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"{column} no es constante para {group['original_uuid'].iloc[0]}."
        )
    return values[0]


def pool_recordings(
    x_events: np.ndarray,
    metadata: pd.DataFrame,
    spec: RepresentationSpec,
    path_orders: np.ndarray,
    path_names: list[str],
    split: str,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    mask = np.isin(path_orders, spec.orders)
    selected_orders = path_orders[mask]
    selected_names = [name for name, keep in zip(path_names, mask) if keep]
    event_values = np.asarray(x_events[:, mask], dtype=np.float32)
    if spec.transform == "log":
        log_mask = selected_orders > 0
        event_values[:, log_mask] = np.log10(
            np.maximum(event_values[:, log_mask], LOG_EPSILON)
        )
    elif spec.transform != "raw":
        raise ValueError(f"Transformacion desconocida: {spec.transform}")

    for column in (
        "original_uuid",
        "stage2_target",
        "fold",
        "window_weight",
        *spec.amplitude_columns,
    ):
        if column not in metadata.columns:
            raise ValueError(f"Falta {column} en metadata {split}.")

    rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    for original_uuid, group in metadata.groupby("original_uuid", sort=False):
        indices = group.index.to_numpy(dtype=int)
        values = event_values[indices]
        weights = group["window_weight"].to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError(f"Pesos invalidos para {original_uuid}.")
        weights /= weights.sum()
        mean = np.sum(values * weights[:, None], axis=0)
        variance = np.sum(((values - mean) ** 2) * weights[:, None], axis=0)
        std = np.sqrt(np.maximum(variance, 0.0))
        maximum = np.max(values, axis=0)
        blocks = [mean, std, maximum]

        if spec.amplitude_columns:
            amplitudes = group[list(spec.amplitude_columns)].to_numpy(dtype=np.float32)
            if not np.isfinite(amplitudes).all():
                raise ValueError(f"Amplitud NaN/Inf para {original_uuid}.")
            amp_mean = np.sum(amplitudes * weights[:, None], axis=0)
            amp_variance = np.sum(
                ((amplitudes - amp_mean) ** 2) * weights[:, None], axis=0
            )
            blocks.extend(
                [amp_mean, np.sqrt(np.maximum(amp_variance, 0.0)), amplitudes.max(axis=0)]
            )
        rows.append(np.concatenate(blocks).astype(np.float32))
        metadata_rows.append(
            {
                "original_uuid": original_uuid,
                "stage2_target": int(validate_group_constant(group, "stage2_target")),
                "fold": int(validate_group_constant(group, "fold")),
                "split": split,
                "cough_type": validate_group_constant(group, "cough_type"),
                "cough_type_consensus": validate_group_constant(
                    group, "cough_type_consensus"
                ),
                "event_count": len(group),
                "window_weight_sum": float(group["window_weight"].sum()),
                "mean_valid_fraction": float(group["valid_fraction"].mean()),
            }
        )

    feature_names = [
        f"wst_{stat}__{name}"
        for stat in ("mean", "std", "max")
        for name in selected_names
    ]
    if spec.amplitude_columns:
        feature_names.extend(
            f"amplitude_{stat}__{column}"
            for stat in ("mean", "std", "max")
            for column in spec.amplitude_columns
        )
    x_recordings = np.vstack(rows).astype(np.float32)
    recording_metadata = pd.DataFrame(metadata_rows)
    if x_recordings.shape[1] != len(feature_names):
        raise RuntimeError("Numero de nombres de features inconsistente.")
    return x_recordings, recording_metadata, feature_names


def build_representation(raw: RawData, spec: RepresentationSpec) -> common.DataView:
    x_train, metadata_train, names = pool_recordings(
        raw.x_train,
        raw.metadata_train,
        spec,
        raw.path_orders,
        raw.path_names,
        "train",
    )
    x_validation, metadata_validation, validation_names = pool_recordings(
        raw.x_validation,
        raw.metadata_validation,
        spec,
        raw.path_orders,
        raw.path_names,
        "validation",
    )
    if names != validation_names:
        raise RuntimeError("TRAIN/VALIDATION tienen features diferentes.")
    return common.DataView(
        experiment=spec.key,
        x_train=x_train,
        train_metadata=metadata_train,
        x_validation=x_validation,
        validation_metadata=metadata_validation,
        feature_names=names,
    )


def build_lr(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="liblinear",
        max_iter=5_000,
        random_state=RANDOM_STATE,
    )


def transformed_folds(
    data: common.DataView,
    pca_options: tuple[int | None, ...],
    representation_key: str,
) -> tuple[
    dict[tuple[int, int | None], tuple[np.ndarray, np.ndarray, np.ndarray]],
    pd.DataFrame,
]:
    cache = {}
    variance_rows = []
    folds = data.train_metadata["fold"].to_numpy(dtype=int)
    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = folds == fold
        training_mask = ~validation_mask
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(data.x_train[training_mask]).astype(np.float32)
        x_outer = scaler.transform(data.x_train[validation_mask]).astype(np.float32)
        cache[(fold, None)] = (x_fit, x_outer, training_mask)
        for components in pca_options:
            if components is None:
                continue
            if components >= min(x_fit.shape):
                continue
            pca = PCA(
                n_components=components,
                whiten=False,
                svd_solver="randomized",
                random_state=RANDOM_STATE,
            )
            x_fit_pca = pca.fit_transform(x_fit).astype(np.float32)
            x_outer_pca = pca.transform(x_outer).astype(np.float32)
            cache[(fold, components)] = (x_fit_pca, x_outer_pca, training_mask)
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
    data: common.DataView,
    representation: RepresentationSpec,
    c_values: tuple[float, ...],
    pca_options: tuple[int | None, ...],
) -> tuple[list[CandidateEvaluation], pd.DataFrame]:
    start = time.perf_counter()
    cache, variance = transformed_folds(data, pca_options, representation.key)
    metadata = data.train_metadata.reset_index(drop=True)
    y = metadata["stage2_target"].to_numpy(dtype=int)
    folds = metadata["fold"].to_numpy(dtype=int)
    evaluations = []
    for c_value in c_values:
        for pca_components in pca_options:
            if any(
                (fold, pca_components) not in cache
                for fold in common.EXPECTED_TRAIN_FOLDS
            ):
                continue
            candidate_start = time.perf_counter()
            oof = np.full(len(y), np.nan, dtype=float)
            for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
                x_fit, x_outer, training_mask = cache[(fold, pca_components)]
                validation_mask = folds == fold
                fit_metadata = metadata.loc[training_mask].reset_index(drop=True)
                weights = common.compute_training_weights(fit_metadata)
                classifier = build_lr(c_value)
                classifier.fit(x_fit, y[training_mask], sample_weight=weights)
                oof[validation_mask] = classifier.predict_proba(x_outer)[:, 1]
            if not np.isfinite(oof).all():
                raise RuntimeError(f"OOF incompleto para {representation.key}.")
            threshold = common.tune_threshold(y, oof, 0.5)
            predictions = metadata[
                [
                    "original_uuid",
                    "stage2_target",
                    "cough_type",
                    "cough_type_consensus",
                    "fold",
                    "event_count",
                ]
            ].copy()
            predictions = predictions.rename(columns={"stage2_target": "y_true"})
            predictions["score"] = oof
            predictions["y_pred_oof_threshold"] = (oof >= threshold).astype(int)
            predictions["y_pred_0p5"] = (oof >= 0.5).astype(int)
            spec = CandidateSpec(representation.key, c_value, pca_components)
            evaluation = CandidateEvaluation(
                spec=spec,
                threshold=threshold,
                metrics_tuned=common.binary_metrics(y, oof, threshold),
                metrics_native=common.binary_metrics(y, oof, 0.5),
                predictions=predictions,
                fold_metrics=recording.fold_metrics_at_threshold(
                    predictions, threshold
                ),
                elapsed_seconds=time.perf_counter() - candidate_start,
                input_dimension=data.x_train.shape[1],
            )
            evaluations.append(evaluation)
            print(
                f"{spec.key} | macro-F1={evaluation.metrics_tuned['macro_f1']:.4f} "
                f"| bal-acc={evaluation.metrics_tuned['balanced_accuracy']:.4f} "
                f"| AUC={evaluation.metrics_tuned['roc_auc']:.4f}"
            )
    print(
        f"  Tiempo representacion {representation.key}: "
        f"{time.perf_counter() - start:.1f} s"
    )
    return evaluations, variance


def selection_key(item: CandidateEvaluation) -> tuple[float, ...]:
    pca_dimension = (
        item.input_dimension
        if item.spec.pca_components is None
        else item.spec.pca_components
    )
    return (
        float(item.metrics_tuned["macro_f1"]),
        float(item.metrics_tuned["balanced_accuracy"]),
        float(item.metrics_tuned["roc_auc"]),
        -float(pca_dimension),
        -float(item.spec.c_value),
    )


def candidate_row(item: CandidateEvaluation, selected: bool = False) -> dict[str, object]:
    return {
        "candidate_key": item.spec.key,
        **asdict(item.spec),
        "input_dimension": item.input_dimension,
        "threshold_oof": item.threshold,
        "selected": selected,
        "elapsed_seconds": item.elapsed_seconds,
        **{f"oof_tuned__{key}": value for key, value in item.metrics_tuned.items()},
        **{f"oof_0p5__{key}": value for key, value in item.metrics_native.items()},
    }


def build_pipeline(spec: CandidateSpec) -> Pipeline:
    steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
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
    steps.append(("classifier", build_lr(spec.c_value)))
    return Pipeline(steps)


def model_float_count(model: Pipeline) -> int:
    count = 0
    scaler = model.named_steps["scaler"]
    count += int(scaler.mean_.size + scaler.scale_.size)
    if "pca" in model.named_steps:
        pca = model.named_steps["pca"]
        count += int(pca.mean_.size + pca.components_.size)
    classifier = model.named_steps["classifier"]
    count += int(classifier.coef_.size + classifier.intercept_.size)
    return count


def check(raw: RawData) -> None:
    order_counts = pd.Series(raw.path_orders).value_counts().sort_index()
    print("=" * 78)
    print("CHECK - WST RECORDING MEJORADO + LR")
    print("=" * 78)
    print(f"X eventos TRAIN:      {raw.x_train.shape} {raw.x_train.dtype}")
    print(f"X eventos VALIDATION: {raw.x_validation.shape} {raw.x_validation.dtype}")
    print(
        "Grabaciones TRAIN dry/wet:",
        raw.metadata_train.drop_duplicates("original_uuid")["stage2_target"]
        .value_counts()
        .sort_index()
        .to_dict(),
    )
    print(
        "Grabaciones VALIDATION dry/wet:",
        raw.metadata_validation.drop_duplicates("original_uuid")["stage2_target"]
        .value_counts()
        .sort_index()
        .to_dict(),
    )
    print(f"Paths WST por orden: {order_counts.to_dict()}")
    print(f"Representaciones de Fase A: {len(representations())}")
    print("TEST no existe en este pipeline y no sera leido.")


def train(raw: RawData, preset: str, quick: bool) -> None:
    mode = "quick" if quick else "full"
    result_dir = RESULTS_ROOT / preset / mode
    model_dir = MODELS_ROOT / preset / mode
    graph_dir = GRAPHS_ROOT / preset / mode
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    specs = representations()
    phase_a_c = QUICK_PHASE_A_C if quick else FULL_PHASE_A_C
    phase_a_results: list[CandidateEvaluation] = []
    views: dict[str, common.DataView] = {}
    print("\n" + "=" * 78)
    print("FASE A - REPRESENTACION WST (SIN PCA)")
    print("=" * 78)
    for index, spec in enumerate(specs, start=1):
        print(f"\n[{index:02d}/{len(specs):02d}] {spec.key}")
        view = build_representation(raw, spec)
        views[spec.key] = view
        evaluations, _ = evaluate_grid(view, spec, phase_a_c, (None,))
        phase_a_results.extend(evaluations)

    phase_a_frame = pd.DataFrame(candidate_row(item) for item in phase_a_results)
    phase_a_frame.to_csv(result_dir / "phase_a_candidate_cv_results.csv", index=False)
    representation_winners = []
    for spec in specs:
        candidates = [
            item for item in phase_a_results if item.spec.representation_key == spec.key
        ]
        representation_winners.append(max(candidates, key=selection_key))
    representation_winners.sort(key=selection_key, reverse=True)
    top_count = QUICK_TOP_REPRESENTATIONS if quick else FULL_TOP_REPRESENTATIONS
    selected_rep_keys = [
        item.spec.representation_key for item in representation_winners[:top_count]
    ]
    pd.DataFrame(
        {
            **candidate_row(item),
            "selected_for_phase_b": item.spec.representation_key in selected_rep_keys,
        }
        for item in representation_winners
    ).to_csv(result_dir / "phase_a_representation_winners.csv", index=False)
    print("\nRepresentaciones seleccionadas para Fase B:")
    for key in selected_rep_keys:
        print(f"  - {key}")

    phase_b_c = QUICK_PHASE_B_C if quick else FULL_PHASE_B_C
    pca_options = QUICK_PCA if quick else FULL_PCA
    phase_b_results: list[CandidateEvaluation] = []
    variance_frames = []
    print("\n" + "=" * 78)
    print("FASE B - C Y PCA OPCIONAL")
    print("=" * 78)
    spec_by_key = {spec.key: spec for spec in specs}
    for index, key in enumerate(selected_rep_keys, start=1):
        print(f"\n[{index}/{len(selected_rep_keys)}] {key}")
        evaluations, variance = evaluate_grid(
            views[key], spec_by_key[key], phase_b_c, pca_options
        )
        phase_b_results.extend(evaluations)
        if not variance.empty:
            variance_frames.append(variance)
    winner = max(phase_b_results, key=selection_key)
    pd.DataFrame(
        candidate_row(item, item.spec.key == winner.spec.key)
        for item in phase_b_results
    ).to_csv(result_dir / "phase_b_candidate_cv_results.csv", index=False)
    if variance_frames:
        pd.concat(variance_frames, ignore_index=True).to_csv(
            result_dir / "pca_explained_variance_by_fold.csv", index=False
        )
    winner.predictions.to_csv(result_dir / "best_oof_predictions.csv", index=False)
    winner.fold_metrics.to_csv(result_dir / "best_cv_fold_metrics.csv", index=False)

    winner_rep = spec_by_key[winner.spec.representation_key]
    winner_view = views[winner.spec.representation_key]
    if quick:
        print("\nPrueba rapida completada.")
        print(f"Ganador OOF: {winner.spec.key}")
        print(f"Macro-F1 OOF: {winner.metrics_tuned['macro_f1']:.4f}")
        print("VALIDATION no se ha evaluado. TEST no se ha leido.")
        print(f"Resultados: {result_dir}")
        return

    print("\nAjustando scaler/PCA/LR final exclusivamente con todo TRAIN...")
    pipeline = build_pipeline(winner.spec)
    sample_weights = common.compute_training_weights(winner_view.train_metadata)
    pipeline.fit(
        winner_view.x_train,
        winner_view.train_metadata["stage2_target"].to_numpy(dtype=int),
        classifier__sample_weight=sample_weights,
    )
    validation_scores = pipeline.predict_proba(winner_view.x_validation)[:, 1]
    y_validation = winner_view.validation_metadata["stage2_target"].to_numpy(dtype=int)
    validation_metrics = common.binary_metrics(
        y_validation, validation_scores, winner.threshold
    )
    validation_native = common.binary_metrics(y_validation, validation_scores, 0.5)
    validation_predictions = winner_view.validation_metadata[
        [
            "original_uuid",
            "stage2_target",
            "cough_type",
            "cough_type_consensus",
            "event_count",
        ]
    ].copy()
    validation_predictions = validation_predictions.rename(
        columns={"stage2_target": "y_true"}
    )
    validation_predictions["score"] = validation_scores
    validation_predictions["y_pred"] = (
        validation_scores >= winner.threshold
    ).astype(int)
    validation_predictions["threshold"] = winner.threshold
    validation_predictions.to_csv(
        result_dir / "validation_predictions.csv", index=False
    )

    package = {
        "model": pipeline,
        "threshold": winner.threshold,
        "candidate": asdict(winner.spec),
        "representation": asdict(winner_rep),
        "feature_names": winner_view.feature_names,
        "pooling": "weighted_mean_weighted_std_unweighted_max",
        "window_weight_column": "window_weight",
        "log_epsilon": LOG_EPSILON,
        "preset": preset,
        "classes": {0: "dry", 1: "wet"},
    }
    model_path = model_dir / "stage2_wst_recording_enhanced_lr.joblib"
    joblib.dump(package, model_path, compress=3)
    model_size_kb = model_path.stat().st_size / 1024.0

    metrics_rows = []
    for split, policy, metrics in (
        ("train_oof", "oof_threshold", winner.metrics_tuned),
        ("train_oof", "fixed_0.5", winner.metrics_native),
        ("validation", "oof_threshold", validation_metrics),
        ("validation", "fixed_0.5", validation_native),
    ):
        metrics_rows.append(
            {
                "split": split,
                "threshold_policy": policy,
                "threshold": winner.threshold if policy == "oof_threshold" else 0.5,
                "candidate_key": winner.spec.key,
                "representation_key": winner_rep.key,
                "input_dimension": winner_view.x_train.shape[1],
                "model_float_count": model_float_count(pipeline),
                "model_size_kb": model_size_kb,
                **metrics,
            }
        )
    pd.DataFrame(metrics_rows).to_csv(result_dir / "metrics_summary.csv", index=False)

    config = {
        "experiment": "stage2_wst_recording_enhanced_lr",
        "preset": preset,
        "selection_data": "TRAIN OOF only",
        "validation_role": "single final evaluation",
        "test_policy": "not read or processed",
        "pooling": "weighted mean/std + unweighted max",
        "phase_a_representations": len(specs),
        "phase_a_c_values": list(phase_a_c),
        "phase_b_c_values": list(phase_b_c),
        "phase_b_pca_options": ["none" if x is None else x for x in pca_options],
        "selected_candidate": asdict(winner.spec),
        "selected_representation": asdict(winner_rep),
        "threshold_oof": winner.threshold,
    }
    pd.DataFrame(
        {"parameter": key, "value": json.dumps(value) if isinstance(value, (dict, list, tuple)) else value}
        for key, value in config.items()
    ).to_csv(result_dir / "experiment_configuration.csv", index=False)

    graph_spec = common.CandidateSpec(
        "logistic_regression", winner.spec.c_value, winner.spec.pca_components
    )
    graph_path = graph_dir / "validation_wst_recording_enhanced_lr.png"
    common.create_validation_graph(
        validation_predictions,
        winner.threshold,
        winner_rep.key,
        graph_spec,
        graph_path,
    )

    print("\n" + "=" * 78)
    print("RESULTADO WST RECORDING MEJORADO + LOGISTIC REGRESSION")
    print("=" * 78)
    print(f"Ganador OOF: {winner.spec.key}")
    print(f"Pooling: weighted mean/std + unweighted max")
    print(f"Features por grabacion: {winner_view.x_train.shape[1]}")
    print(f"Umbral OOF congelado: {winner.threshold:.6f}")
    print(f"Macro-F1 OOF: {winner.metrics_tuned['macro_f1']:.4f}")
    print(f"Macro-F1 validation: {validation_metrics['macro_f1']:.4f}")
    print(
        "Recalls validation dry/wet: "
        f"{validation_metrics['dry_recall']:.4f} / "
        f"{validation_metrics['wet_recall']:.4f}"
    )
    print(f"Modelo joblib: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print(f"Grafica: {graph_path}")
    print("TEST permanece reservado y no ha sido leido.")


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST RECORDING MEJORADO + LR")
    print("=" * 78)
    print(f"Preset: {args.preset}")
    print(f"Accion: {args.action}")
    print("Solo COUGHVID. TEST no sera leido ni procesado.")
    raw = load_raw_data(args.preset)
    check(raw)
    if args.action == "train":
        train(raw, args.preset, args.quick)


if __name__ == "__main__":
    main()
