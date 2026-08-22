"""WST recording mean+std+max -> StandardScaler -> PCA -> Random Forest.

Compara el control sin PCA y 32, 64, 128 y 256 componentes. StandardScaler y
PCA se ajustan exclusivamente dentro del bloque de entrenamiento de cada fold.
La comparacion de componentes usa un RF fijo; despues se buscan
hiperparametros de RF solo para la proyeccion seleccionada.

La unidad es una grabacion (una fila por ``original_uuid``), se mantiene el
balanceo por ``sample_weight`` y no se aplica SMOTE. Se reportan umbral fijo
0.5 y umbral aprendido con OOF de TRAIN. VALIDATION se evalua una vez al final
y TEST no se lee ni se procesa.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording
import train_stage2_dry_wet_wavelet_scattering_rf as event_baseline


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (
    SCRIPT_DIR
    / "results_stage2_dry_wet_wavelet_scattering_recording_pca_rf"
)
MODELS_ROOT = (
    SCRIPT_DIR
    / "models_stage2_dry_wet_wavelet_scattering_recording_pca_rf"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wavelet_scattering_recording_pca_rf"
)

RANDOM_STATE = 42
PRESETS = event_baseline.PRESETS
PCA_COMPONENTS = (32, 64, 128, 256)
QUICK_COMPONENTS = (64, 128)
FIXED_THRESHOLD = 0.5
FIXED_PROJECTION_RF = event_baseline.RFCandidate(200, 8, 5, 0.1)
RECORDING_POOLING = next(
    spec for spec in recording.POOLING_SPECS if spec.key == "mean_std_max"
)


@dataclass(frozen=True)
class ProjectionSpec:
    components: int | None

    @property
    def key(self) -> str:
        return "no_pca" if self.components is None else f"pca{self.components}"

    @property
    def output_dimension(self) -> int:
        if self.components is None:
            return 1932
        return self.components


@dataclass
class ProjectionEvaluation:
    spec: ProjectionSpec
    evaluation: event_baseline.RFEvaluation
    metrics_fixed_0p5: dict[str, float | int]


def projection_specs(quick: bool) -> list[ProjectionSpec]:
    component_values = QUICK_COMPONENTS if quick else PCA_COMPONENTS
    return [ProjectionSpec(None), *[ProjectionSpec(n) for n in component_values]]


def rf_candidates(quick: bool) -> list[event_baseline.RFCandidate]:
    if quick:
        candidates = [
            FIXED_PROJECTION_RF,
            event_baseline.RFCandidate(100, 4, 15, "sqrt"),
            event_baseline.RFCandidate(150, 6, 10, "sqrt"),
            event_baseline.RFCandidate(200, 6, 10, 0.5),
        ]
        return list(dict.fromkeys(candidates))
    return [
        event_baseline.RFCandidate(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        for n_estimators in (100, 200, 300)
        for max_depth in (4, 6, 8, 10)
        for min_samples_leaf in (5, 10)
        for max_features in ("sqrt", 0.1, 0.5)
    ]


def build_recording_data(
    event_data: common.DataView,
) -> common.DataView:
    data = recording.build_recording_view(event_data, RECORDING_POOLING)
    if data.x_train.shape[1] != 1932:
        raise ValueError(
            f"mean_std_max deberia tener 1932 features: {data.x_train.shape}."
        )
    return data


def fit_scaler_pca(
    x_training: np.ndarray,
    n_components: int,
) -> tuple[StandardScaler, PCA, np.ndarray]:
    if n_components >= min(x_training.shape):
        raise ValueError(
            f"PCA{n_components} no es valida para {x_training.shape}."
        )
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_training).astype(np.float32)
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        n_oversamples=12,
        iterated_power=4,
        power_iteration_normalizer="auto",
        random_state=RANDOM_STATE,
        whiten=False,
    )
    transformed = pca.fit_transform(x_scaled).astype(np.float32)
    if transformed.shape != (len(x_training), n_components):
        raise RuntimeError("Forma PCA inesperada.")
    if not np.isfinite(transformed).all():
        raise RuntimeError("PCA produjo NaN/Inf.")
    return scaler, pca, transformed


def transform_projection(
    x_values: np.ndarray,
    spec: ProjectionSpec,
    scaler: StandardScaler | None,
    pca: PCA | None,
) -> np.ndarray:
    if spec.components is None:
        return np.asarray(x_values, dtype=np.float32)
    if scaler is None or pca is None:
        raise RuntimeError("Faltan StandardScaler/PCA para transformar.")
    transformed = pca.transform(scaler.transform(x_values)).astype(np.float32)
    if transformed.shape != (len(x_values), spec.components):
        raise RuntimeError("Transformacion PCA con forma inesperada.")
    if not np.isfinite(transformed).all():
        raise RuntimeError("Transformacion PCA produjo NaN/Inf.")
    return transformed


def make_evaluation(
    data: common.DataView,
    candidate: event_baseline.RFCandidate,
    event_scores: np.ndarray,
) -> event_baseline.RFEvaluation:
    if not np.isfinite(event_scores).all():
        raise RuntimeError(f"OOF incompleto para {candidate.key}.")
    predictions = common.aggregate_scores_by_recording(
        data.train_metadata,
        event_scores,
    )
    y_true = predictions["y_true"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    threshold = common.tune_threshold(y_true, scores, FIXED_THRESHOLD)
    metrics = common.binary_metrics(y_true, scores, threshold)
    predictions["y_pred"] = (scores >= threshold).astype(int)
    predictions["candidate_key"] = candidate.key
    fold_metrics = recording.fold_metrics_at_threshold(
        predictions,
        threshold,
    )
    return event_baseline.RFEvaluation(
        candidate=candidate,
        threshold=threshold,
        metrics=metrics,
        oof_recordings=predictions,
        fold_metrics=fold_metrics,
    )


def evaluate_projections(
    data: common.DataView,
    specs: list[ProjectionSpec],
) -> tuple[list[ProjectionEvaluation], pd.DataFrame]:
    oof = {
        spec.key: np.full(len(data.x_train), np.nan, dtype=float)
        for spec in specs
    }
    maximum_components = max(
        int(spec.components or 0) for spec in specs
    )
    variance_rows: list[dict[str, float | int]] = []

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = data.train_metadata["fold"].to_numpy() == fold
        training_mask = ~validation_mask
        metadata_fit = data.train_metadata.loc[training_mask].reset_index(
            drop=True
        )
        y_fit = metadata_fit["stage2_target"].to_numpy(dtype=int)
        sample_weights = common.compute_training_weights(metadata_fit)
        scaler, pca, x_fit_pca = fit_scaler_pca(
            data.x_train[training_mask],
            maximum_components,
        )
        x_validation_pca = pca.transform(
            scaler.transform(data.x_train[validation_mask])
        ).astype(np.float32)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        for spec in specs:
            classifier = event_baseline.build_rf(FIXED_PROJECTION_RF)
            if spec.components is None:
                x_fit = data.x_train[training_mask]
                x_validation = data.x_train[validation_mask]
            else:
                x_fit = x_fit_pca[:, : spec.components]
                x_validation = x_validation_pca[:, : spec.components]
                variance_rows.append(
                    {
                        "stage": "projection_selection",
                        "fold": fold,
                        "pca_components": spec.components,
                        "cumulative_explained_variance_ratio": float(
                            cumulative_variance[spec.components - 1]
                        ),
                        "cumulative_explained_variance_percent": float(
                            cumulative_variance[spec.components - 1] * 100.0
                        ),
                    }
                )
            classifier.fit(x_fit, y_fit, sample_weight=sample_weights)
            oof[spec.key][validation_mask] = event_baseline.rf_scores(
                classifier,
                x_validation,
            )
        print(f"  Fold {fold}: control y PCA ajustados solo con TRAIN interno.")

    results = []
    for spec in specs:
        evaluation = make_evaluation(
            data,
            FIXED_PROJECTION_RF,
            oof[spec.key],
        )
        results.append(
            ProjectionEvaluation(
                spec=spec,
                evaluation=evaluation,
                metrics_fixed_0p5=recording.metrics_at_fixed_threshold(
                    evaluation.oof_recordings
                ),
            )
        )
    return results, pd.DataFrame(variance_rows)


def projection_selection_key(
    item: ProjectionEvaluation,
) -> tuple[float, ...]:
    metrics = item.evaluation.metrics
    return (
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["roc_auc"]),
        -float(item.spec.output_dimension),
    )


def evaluate_rf_candidates(
    data: common.DataView,
    projection: ProjectionSpec,
    candidates: list[event_baseline.RFCandidate],
) -> tuple[list[event_baseline.RFEvaluation], pd.DataFrame]:
    oof = {
        candidate.key: np.full(len(data.x_train), np.nan, dtype=float)
        for candidate in candidates
    }
    variance_rows: list[dict[str, float | int]] = []

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = data.train_metadata["fold"].to_numpy() == fold
        training_mask = ~validation_mask
        metadata_fit = data.train_metadata.loc[training_mask].reset_index(
            drop=True
        )
        y_fit = metadata_fit["stage2_target"].to_numpy(dtype=int)
        sample_weights = common.compute_training_weights(metadata_fit)
        if projection.components is None:
            x_fit = data.x_train[training_mask]
            x_validation = data.x_train[validation_mask]
        else:
            scaler, pca, x_fit = fit_scaler_pca(
                data.x_train[training_mask],
                projection.components,
            )
            x_validation = transform_projection(
                data.x_train[validation_mask],
                projection,
                scaler,
                pca,
            )
            explained = float(np.sum(pca.explained_variance_ratio_))
            variance_rows.append(
                {
                    "stage": "rf_selection",
                    "fold": fold,
                    "pca_components": projection.components,
                    "cumulative_explained_variance_ratio": explained,
                    "cumulative_explained_variance_percent": explained * 100.0,
                }
            )

        print(f"  Fold {fold}: proyeccion {projection.key} ajustada.")
        for index, candidate in enumerate(candidates, start=1):
            classifier = event_baseline.build_rf(candidate)
            classifier.fit(x_fit, y_fit, sample_weight=sample_weights)
            oof[candidate.key][validation_mask] = event_baseline.rf_scores(
                classifier,
                x_validation,
            )
            print(f"    RF {index:02d}/{len(candidates):02d}: {candidate.key}")

    evaluations = [
        make_evaluation(data, candidate, oof[candidate.key])
        for candidate in candidates
    ]
    return evaluations, pd.DataFrame(variance_rows)


def prefixed_metrics(
    metrics: dict[str, float | int],
    prefix: str,
) -> dict[str, float | int]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def save_projection_results(
    results: list[ProjectionEvaluation],
    selected: ProjectionEvaluation,
    variance: pd.DataFrame,
    result_dir: Path,
) -> None:
    rows = []
    predictions = []
    for item in results:
        rows.append(
            {
                "projection_key": item.spec.key,
                "pca_components": item.spec.components,
                "output_dimension": item.spec.output_dimension,
                "fixed_rf": FIXED_PROJECTION_RF.key,
                "threshold_oof": item.evaluation.threshold,
                "selected_projection": item.spec.key == selected.spec.key,
                **prefixed_metrics(item.evaluation.metrics, "oof_tuned__"),
                **prefixed_metrics(item.metrics_fixed_0p5, "fixed_0p5__"),
            }
        )
        frame = item.evaluation.oof_recordings.copy()
        frame["projection_key"] = item.spec.key
        frame["y_pred_fixed_0p5"] = (
            frame["score"].to_numpy(dtype=float) >= FIXED_THRESHOLD
        ).astype(int)
        predictions.append(frame)
    pd.DataFrame(rows).sort_values(
        ["oof_tuned__macro_f1", "oof_tuned__balanced_accuracy", "oof_tuned__roc_auc"],
        ascending=False,
    ).to_csv(
        result_dir / "projection_cv_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        result_dir / "projection_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    variance.to_csv(
        result_dir / "pca_explained_variance_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_rf_results(
    evaluations: list[event_baseline.RFEvaluation],
    selected: event_baseline.RFEvaluation,
    result_dir: Path,
    elapsed_seconds: float,
) -> None:
    rows = []
    for item in evaluations:
        fixed_metrics = recording.metrics_at_fixed_threshold(
            item.oof_recordings
        )
        candidate = item.candidate
        rows.append(
            {
                "candidate_key": candidate.key,
                "n_estimators": candidate.n_estimators,
                "max_depth": candidate.max_depth,
                "min_samples_leaf": candidate.min_samples_leaf,
                "max_features": candidate.max_features,
                "threshold_oof": item.threshold,
                "selected_rf": candidate.key == selected.candidate.key,
                "elapsed_seconds_total_search": elapsed_seconds,
                **prefixed_metrics(item.metrics, "oof_tuned__"),
                **prefixed_metrics(fixed_metrics, "fixed_0p5__"),
            }
        )
    pd.DataFrame(rows).sort_values(
        ["oof_tuned__macro_f1", "oof_tuned__balanced_accuracy", "oof_tuned__roc_auc"],
        ascending=False,
    ).to_csv(
        result_dir / "rf_candidate_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions = selected.oof_recordings.copy()
    predictions["y_pred_fixed_0p5"] = (
        predictions["score"].to_numpy(dtype=float) >= FIXED_THRESHOLD
    ).astype(int)
    predictions.to_csv(
        result_dir / "best_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.fold_metrics.to_csv(
        result_dir / "best_cv_fold_metrics_oof_threshold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    recording.fold_metrics_at_threshold(predictions, FIXED_THRESHOLD).to_csv(
        result_dir / "best_cv_fold_metrics_fixed_0p5.csv",
        index=False,
        encoding="utf-8-sig",
    )


def metrics_rows(
    dataset: str,
    projection: ProjectionSpec,
    candidate: event_baseline.RFCandidate,
    tuned_threshold: float,
    predictions: pd.DataFrame,
    model_size_kb: float,
    complexity: dict[str, float | int] | None = None,
) -> list[dict[str, object]]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    shared = {
        "dataset": dataset,
        "experiment": "wavelet_scattering_recording_pca_rf",
        "pooling_key": RECORDING_POOLING.key,
        "projection_key": projection.key,
        "pca_components": projection.components,
        "output_dimension": projection.output_dimension,
        "candidate_key": candidate.key,
        "model_size_kb_joblib": model_size_kb,
        **(complexity or {}),
    }
    return [
        {
            **shared,
            "threshold_policy": "fixed_0p5",
            "threshold": FIXED_THRESHOLD,
            **common.binary_metrics(y_true, scores, FIXED_THRESHOLD),
        },
        {
            **shared,
            "threshold_policy": "oof_tuned_frozen",
            "threshold": tuned_threshold,
            **common.binary_metrics(y_true, scores, tuned_threshold),
        },
    ]


def train(
    data: common.DataView,
    extraction_configuration: pd.DataFrame,
    preset: str,
    quick: bool,
) -> None:
    run_name = "quick" if quick else "full"
    result_dir = RESULTS_ROOT / preset / run_name
    model_dir = MODELS_ROOT / preset / run_name
    graph_dir = GRAPHS_ROOT / preset / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    specs = projection_specs(quick)
    print("\n" + "=" * 78)
    print("FASE 1 - COMPARACION NO PCA / PCA CON RF FIJO")
    print("=" * 78)
    projection_results, projection_variance = evaluate_projections(data, specs)
    selected_projection = max(
        projection_results,
        key=projection_selection_key,
    )
    for item in projection_results:
        print(
            f"{item.spec.key} | dim={item.spec.output_dimension} | "
            f"macro-F1={item.evaluation.metrics['macro_f1']:.4f} | "
            f"macro-F1@0.5={item.metrics_fixed_0p5['macro_f1']:.4f} | "
            f"AUC={item.evaluation.metrics['roc_auc']:.4f}"
        )
    print(f"Proyeccion seleccionada: {selected_projection.spec.key}")
    save_projection_results(
        projection_results,
        selected_projection,
        projection_variance,
        result_dir,
    )

    candidates = rf_candidates(quick)
    print("\n" + "=" * 78)
    print("FASE 2 - BUSQUEDA RF SOBRE LA PROYECCION SELECCIONADA")
    print("=" * 78)
    print(f"Proyeccion: {selected_projection.spec.key}")
    print(f"Candidatos RF: {len(candidates)}")
    start = time.perf_counter()
    evaluations, rf_variance = evaluate_rf_candidates(
        data,
        selected_projection.spec,
        candidates,
    )
    elapsed_seconds = time.perf_counter() - start
    best = max(evaluations, key=event_baseline.selection_key)
    save_rf_results(evaluations, best, result_dir, elapsed_seconds)
    if not rf_variance.empty:
        rf_variance.to_csv(
            result_dir / "selected_pca_variance_by_fold.csv",
            index=False,
            encoding="utf-8-sig",
        )

    fixed_metrics = recording.metrics_at_fixed_threshold(best.oof_recordings)
    print(f"Mejor RF: {best.candidate.key}")
    print(f"Umbral OOF: {best.threshold:.6f}")
    print(f"Macro-F1 OOF: {best.metrics['macro_f1']:.4f}")
    print(f"Macro-F1 OOF @0.5: {fixed_metrics['macro_f1']:.4f}")
    oof_rows = metrics_rows(
        "train_oof",
        selected_projection.spec,
        best.candidate,
        best.threshold,
        best.oof_recordings,
        np.nan,
    )

    if quick:
        pd.DataFrame(oof_rows).to_csv(
            result_dir / "metrics_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\nPrueba rapida recording PCA + RF completada.")
        print("VALIDATION no se ha evaluado.")
        print(f"Resultados: {result_dir}")
        return

    print("\nAjustando transformacion y RF finales con todo TRAIN...")
    if selected_projection.spec.components is None:
        scaler = None
        pca = None
        x_train_final = data.x_train
        x_validation_final = data.x_validation
        final_explained = np.nan
    else:
        scaler, pca, x_train_final = fit_scaler_pca(
            data.x_train,
            selected_projection.spec.components,
        )
        x_validation_final = transform_projection(
            data.x_validation,
            selected_projection.spec,
            scaler,
            pca,
        )
        final_explained = float(np.sum(pca.explained_variance_ratio_))
        pd.DataFrame(
            {
                "component": np.arange(1, pca.n_components_ + 1),
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_explained_variance_ratio": np.cumsum(
                    pca.explained_variance_ratio_
                ),
            }
        ).to_csv(
            result_dir / "pca_explained_variance_final_train.csv",
            index=False,
            encoding="utf-8-sig",
        )

    classifier = event_baseline.build_rf(best.candidate)
    classifier.fit(
        x_train_final,
        data.train_metadata["stage2_target"].to_numpy(dtype=int),
        sample_weight=common.compute_training_weights(data.train_metadata),
    )
    validation_scores = event_baseline.rf_scores(
        classifier,
        x_validation_final,
    )
    validation_predictions = common.aggregate_scores_by_recording(
        data.validation_metadata,
        validation_scores,
    )
    validation_predictions["y_pred_oof_threshold"] = (
        validation_predictions["score"].to_numpy(dtype=float) >= best.threshold
    ).astype(int)
    validation_predictions["y_pred_fixed_0p5"] = (
        validation_predictions["score"].to_numpy(dtype=float) >= FIXED_THRESHOLD
    ).astype(int)
    validation_predictions["projection_key"] = selected_projection.spec.key
    validation_predictions["candidate_key"] = best.candidate.key

    complexity = event_baseline.forest_complexity(classifier)
    model_package = {
        "scaler": scaler,
        "pca": pca,
        "classifier": classifier,
        "preset": preset,
        "experiment": "wavelet_scattering_recording_pca_rf",
        "pooling_key": RECORDING_POOLING.key,
        "pooling_statistics": RECORDING_POOLING.statistics,
        "input_feature_count": data.x_train.shape[1],
        "projection_key": selected_projection.spec.key,
        "pca_components": selected_projection.spec.components,
        "pca_explained_variance_ratio": final_explained,
        "candidate_key": best.candidate.key,
        "threshold": best.threshold,
        "fixed_reference_threshold": FIXED_THRESHOLD,
        "label_mapping": common.LABEL_TO_NAME,
        "recording_feature_names": data.feature_names,
        "standardization_used": selected_projection.spec.components is not None,
        "pca_used": selected_projection.spec.components is not None,
        "pca_whiten": False,
        "smote_used": False,
        "classifier_weighting": "class_balanced_per_recording",
        "extraction_configuration": extraction_configuration.iloc[0].to_dict(),
        "random_state": RANDOM_STATE,
        **complexity,
    }
    model_path = model_dir / "wavelet_scattering_recording_pca_rf_model.joblib"
    joblib.dump(model_package, model_path, compress=3)
    model_size_kb = model_path.stat().st_size / 1024.0

    for row in oof_rows:
        row["model_size_kb_joblib"] = model_size_kb
        row.update(complexity)
    validation_rows = metrics_rows(
        "validation",
        selected_projection.spec,
        best.candidate,
        best.threshold,
        validation_predictions,
        model_size_kb,
        complexity,
    )
    pd.DataFrame([*oof_rows, *validation_rows]).to_csv(
        result_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_predictions.to_csv(
        result_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "input_feature_count": data.x_train.shape[1],
                "pooling_key": RECORDING_POOLING.key,
                "selected_projection": selected_projection.spec.key,
                "pca_components": selected_projection.spec.components,
                "pca_explained_variance_percent": (
                    final_explained * 100.0
                    if np.isfinite(final_explained)
                    else np.nan
                ),
                "pipeline_order": (
                    "standard_scaler_then_pca_then_rf"
                    if selected_projection.spec.components is not None
                    else "no_scaler_no_pca_then_rf"
                ),
                "selected_rf": best.candidate.key,
                "threshold_selection": "train_oof_macro_f1",
                "fixed_threshold_reported": FIXED_THRESHOLD,
                "smote_used": False,
                "validation_used_for_selection": False,
                "test_processed": False,
                **complexity,
            }
        ]
    ).to_csv(
        result_dir / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    graph_oof = graph_dir / "validation_recording_pca_rf_oof_threshold.png"
    graph_fixed = graph_dir / "validation_recording_pca_rf_fixed_0p5.png"
    common.create_validation_graph(
        validation_predictions,
        best.threshold,
        "wavelet_scattering_recording_pca_rf_oof_threshold",
        best.candidate,
        graph_oof,
    )
    common.create_validation_graph(
        validation_predictions,
        FIXED_THRESHOLD,
        "wavelet_scattering_recording_pca_rf_fixed_0p5",
        best.candidate,
        graph_fixed,
    )

    validation_oof_metrics = common.binary_metrics(
        validation_predictions["y_true"].to_numpy(dtype=int),
        validation_predictions["score"].to_numpy(dtype=float),
        best.threshold,
    )
    validation_fixed_metrics = common.binary_metrics(
        validation_predictions["y_true"].to_numpy(dtype=int),
        validation_predictions["score"].to_numpy(dtype=float),
        FIXED_THRESHOLD,
    )
    print("\n" + "=" * 78)
    print("RESULTADO WST RECORDING PCA + RANDOM FOREST")
    print("=" * 78)
    print(f"Proyeccion seleccionada: {selected_projection.spec.key}")
    if np.isfinite(final_explained):
        print(f"Varianza explicada: {final_explained * 100.0:.2f}%")
    print(f"Mejor RF: {best.candidate.key}")
    print(f"Umbral OOF congelado: {best.threshold:.6f}")
    print(f"Macro-F1 OOF: {best.metrics['macro_f1']:.4f}")
    print(f"Macro-F1 OOF @0.5: {fixed_metrics['macro_f1']:.4f}")
    print(
        "Macro-F1 validation OOF/@0.5: "
        f"{validation_oof_metrics['macro_f1']:.4f} / "
        f"{validation_fixed_metrics['macro_f1']:.4f}"
    )
    print(
        "Recalls validation dry/wet con umbral OOF: "
        f"{validation_oof_metrics['dry_recall']:.4f} / "
        f"{validation_oof_metrics['wet_recall']:.4f}"
    )
    print(f"Modelo joblib: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print(f"Graficas: {graph_dir}")
    print("TEST permanece reservado.")


def print_check(data: common.DataView, quick: bool) -> None:
    specs = projection_specs(quick)
    train_counts = data.train_metadata["stage2_target"].value_counts()
    validation_counts = data.validation_metadata["stage2_target"].value_counts()
    print("=" * 78)
    print("CHECK - WST RECORDING STANDARD SCALER + PCA + RF")
    print("=" * 78)
    print(f"X TRAIN/VALIDATION: {data.x_train.shape} / {data.x_validation.shape}")
    print(f"Grabaciones TRAIN dry/wet: {train_counts[0]} / {train_counts[1]}")
    print(
        "Grabaciones VALIDATION dry/wet: "
        f"{validation_counts[0]} / {validation_counts[1]}"
    )
    print(f"Pooling fijo: {RECORDING_POOLING.key}")
    print(
        "Proyecciones: "
        + ", ".join(
            f"{spec.key}={spec.output_dimension}" for spec in specs
        )
    )
    print(f"RF fijo para comparar PCA: {FIXED_PROJECTION_RF.key}")
    print(f"Candidatos RF tras seleccionar PCA: {len(rf_candidates(quick))}")
    print("Pipeline PCA: StandardScaler -> PCA randomized -> RF.")
    print("Scaler y PCA se ajustan dentro de cada fold.")
    print("Sin SMOTE; balanceo por grabacion mediante sample_weight.")
    print("VALIDATION no selecciona configuracion. TEST no se procesa.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WST recording mean+std+max con PCA y Random Forest."
    )
    parser.add_argument(
        "--action", choices=["check", "train"], default="check"
    )
    parser.add_argument(
        "--preset", choices=PRESETS, default="paper_q8_q1_t500_full"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Compara no-PCA/PCA64/PCA128 y cuatro RF; no evalua VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST RECORDING STANDARD SCALER + PCA + RF")
    print("=" * 78)
    print(f"Preset: {args.preset}")
    print(f"Accion: {args.action}")
    print("TEST no sera leido ni procesado.")
    event_data, extraction_configuration = event_baseline.load_wavelet_data(
        args.preset
    )
    data = build_recording_data(event_data)
    print_check(data, args.quick)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(data, extraction_configuration, args.preset, args.quick)


if __name__ == "__main__":
    main()
