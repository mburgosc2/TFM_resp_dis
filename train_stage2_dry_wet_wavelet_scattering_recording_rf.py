"""Random Forest recording-level sobre coeficientes Wavelet Scattering.

Parte de los 644 coeficientes WST ya extraidos por evento y construye una
unica fila por ``original_uuid``. Compara tres representaciones sin PCA,
SMOTE ni estandarizacion:

* ``mean``: media de cada camino WST entre los eventos.
* ``mean_std``: media y desviacion estandar.
* ``mean_std_max``: media, desviacion estandar y maximo.

La representacion se compara con un RF fijo. Despues se buscan los
hiperparametros del RF solamente para la representacion elegida. Todas las
decisiones y el umbral se aprenden con predicciones OOF de los cinco folds de
TRAIN. Se guardan tambien metricas con umbral fijo 0.5 para hacer transparente
el efecto del ajuste de umbral. VALIDATION se evalua una vez al final y TEST no
se lee ni se procesa.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_rf as event_baseline


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (
    SCRIPT_DIR / "results_stage2_dry_wet_wavelet_scattering_recording_rf"
)
MODELS_ROOT = (
    SCRIPT_DIR / "models_stage2_dry_wet_wavelet_scattering_recording_rf"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wavelet_scattering_recording_rf"
)

RANDOM_STATE = 42
PRESETS = event_baseline.PRESETS
FIXED_POOLING_RF = event_baseline.RFCandidate(300, 10, 5, 0.1)
FIXED_THRESHOLD = 0.5


@dataclass(frozen=True)
class PoolingSpec:
    key: str
    statistics: tuple[str, ...]

    @property
    def statistic_count(self) -> int:
        return len(self.statistics)


@dataclass
class PoolingEvaluation:
    spec: PoolingSpec
    data: common.DataView
    evaluation: event_baseline.RFEvaluation
    metrics_fixed_0p5: dict[str, float | int]


POOLING_SPECS = (
    PoolingSpec("mean", ("mean",)),
    PoolingSpec("mean_std", ("mean", "std")),
    PoolingSpec("mean_std_max", ("mean", "std", "max")),
)


def statistic_block(values: np.ndarray, statistic: str) -> np.ndarray:
    if statistic == "mean":
        return np.mean(values, axis=0, dtype=np.float64)
    if statistic == "std":
        # ddof=0 permite representar grabaciones con un solo evento: su
        # variabilidad observada es cero en vez de NaN.
        return np.std(values, axis=0, ddof=0, dtype=np.float64)
    if statistic == "max":
        return np.max(values, axis=0)
    raise ValueError(f"Estadistico recording desconocido: {statistic}")


def recording_feature_names(
    event_feature_names: list[str],
    spec: PoolingSpec,
) -> list[str]:
    return [
        f"event_{statistic}__{feature_name}"
        for statistic in spec.statistics
        for feature_name in event_feature_names
    ]


def aggregate_split(
    x_events: np.ndarray,
    event_metadata: pd.DataFrame,
    event_feature_names: list[str],
    spec: PoolingSpec,
    split_name: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(x_events) != len(event_metadata):
        raise ValueError("WST y metadata de eventos tienen longitudes distintas.")

    vectors: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    for original_uuid, positions in event_metadata.groupby(
        "original_uuid", sort=False
    ).indices.items():
        indices = np.asarray(positions, dtype=int)
        group = event_metadata.iloc[indices]
        values = x_events[indices]

        for column in (
            "stage2_target",
            "cough_type",
            "cough_type_consensus",
            "fold",
            "split",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"{original_uuid} contiene valores distintos en {column}."
                )

        blocks = [
            statistic_block(values, statistic)
            for statistic in spec.statistics
        ]
        vector = np.concatenate(blocks).astype(np.float32)
        if not np.isfinite(vector).all():
            raise RuntimeError(
                f"Agregacion {spec.key} produjo NaN/Inf en {original_uuid}."
            )
        vectors.append(vector)
        metadata_rows.append(
            {
                "feature_row": len(metadata_rows),
                "original_uuid": str(original_uuid),
                "cough_type": str(group["cough_type"].iloc[0]),
                "cough_type_consensus": str(
                    group["cough_type_consensus"].iloc[0]
                ),
                "stage2_target": int(group["stage2_target"].iloc[0]),
                "fold": int(group["fold"].iloc[0]),
                "split": str(group["split"].iloc[0]),
                "event_count": len(group),
                "pooling_key": spec.key,
            }
        )

    matrix = np.asarray(vectors, dtype=np.float32)
    metadata = pd.DataFrame(metadata_rows)
    expected_features = len(event_feature_names) * spec.statistic_count
    if matrix.shape != (len(metadata), expected_features):
        raise RuntimeError(
            f"Forma recording {spec.key} inesperada: {matrix.shape}."
        )

    y_values = metadata["stage2_target"].to_numpy(dtype=np.int64)
    folds = metadata["fold"].to_numpy(dtype=np.int32)
    common.validate_arrays_and_metadata(
        "recording",
        split_name,
        matrix,
        y_values,
        folds,
        metadata,
    )
    return matrix, metadata


def build_recording_view(
    event_data: common.DataView,
    spec: PoolingSpec,
) -> common.DataView:
    names = recording_feature_names(event_data.feature_names, spec)
    x_train, metadata_train = aggregate_split(
        event_data.x_train,
        event_data.train_metadata,
        event_data.feature_names,
        spec,
        "train",
    )
    x_validation, metadata_validation = aggregate_split(
        event_data.x_validation,
        event_data.validation_metadata,
        event_data.feature_names,
        spec,
        "validation",
    )
    train_uuids = set(metadata_train["original_uuid"])
    validation_uuids = set(metadata_validation["original_uuid"])
    overlap = train_uuids & validation_uuids
    if overlap:
        raise ValueError(
            "TRAIN y VALIDATION recording comparten UUID: "
            f"{sorted(overlap)[:10]}"
        )
    if len(names) != x_train.shape[1]:
        raise RuntimeError("Nombres y matriz recording no coinciden.")
    return common.DataView(
        experiment=f"wavelet_scattering_recording_{spec.key}",
        x_train=x_train,
        train_metadata=metadata_train,
        x_validation=x_validation,
        validation_metadata=metadata_validation,
        feature_names=names,
    )


def metrics_at_fixed_threshold(
    predictions: pd.DataFrame,
) -> dict[str, float | int]:
    return common.binary_metrics(
        predictions["y_true"].to_numpy(dtype=int),
        predictions["score"].to_numpy(dtype=float),
        FIXED_THRESHOLD,
    )


def pooling_selection_key(item: PoolingEvaluation) -> tuple[float, ...]:
    metrics = item.evaluation.metrics
    return (
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["roc_auc"]),
        -float(item.data.x_train.shape[1]),
    )


def evaluate_poolings(
    event_data: common.DataView,
) -> tuple[list[PoolingEvaluation], PoolingEvaluation]:
    results = []
    for index, spec in enumerate(POOLING_SPECS, start=1):
        print("\n" + "-" * 78)
        print(
            f"POOLING {index}/{len(POOLING_SPECS)}: {spec.key} "
            f"con RF fijo {FIXED_POOLING_RF.key}"
        )
        data = build_recording_view(event_data, spec)
        evaluation = event_baseline.evaluate_candidates_oof(
            data,
            [FIXED_POOLING_RF],
        )[0]
        results.append(
            PoolingEvaluation(
                spec=spec,
                data=data,
                evaluation=evaluation,
                metrics_fixed_0p5=metrics_at_fixed_threshold(
                    evaluation.oof_recordings
                ),
            )
        )
    return results, max(results, key=pooling_selection_key)


def rf_grid(quick: bool) -> list[event_baseline.RFCandidate]:
    candidates = event_baseline.rf_candidates(quick)
    # El ganador event-level siempre queda como punto de referencia incluso
    # en --quick. En full ya forma parte de la rejilla y se elimina duplicado.
    return list(dict.fromkeys([FIXED_POOLING_RF, *candidates]))


def rf_selection_key(
    evaluation: event_baseline.RFEvaluation,
) -> tuple[float, ...]:
    return event_baseline.selection_key(evaluation)


def fold_metrics_at_threshold(
    predictions: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for fold, fold_data in predictions.groupby("fold", sort=True):
        rows.append(
            {
                "fold": int(fold),
                "recording_count": len(fold_data),
                "threshold": threshold,
                **common.binary_metrics(
                    fold_data["y_true"].to_numpy(dtype=int),
                    fold_data["score"].to_numpy(dtype=float),
                    threshold,
                ),
            }
        )
    return pd.DataFrame(rows)


def prefixed_metrics(
    metrics: dict[str, float | int],
    prefix: str,
) -> dict[str, float | int]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def save_pooling_results(
    results: list[PoolingEvaluation],
    selected: PoolingEvaluation,
    result_dir: Path,
) -> None:
    rows = []
    all_predictions = []
    for item in results:
        evaluation = item.evaluation
        rows.append(
            {
                "pooling_key": item.spec.key,
                "statistics": "|".join(item.spec.statistics),
                "feature_count": item.data.x_train.shape[1],
                "fixed_rf": FIXED_POOLING_RF.key,
                "threshold_oof": evaluation.threshold,
                "selected_pooling": item.spec.key == selected.spec.key,
                **prefixed_metrics(evaluation.metrics, "oof_tuned__"),
                **prefixed_metrics(item.metrics_fixed_0p5, "fixed_0p5__"),
            }
        )
        predictions = evaluation.oof_recordings.copy()
        predictions["pooling_key"] = item.spec.key
        predictions["y_pred_fixed_0p5"] = (
            predictions["score"].to_numpy(dtype=float) >= FIXED_THRESHOLD
        ).astype(int)
        all_predictions.append(predictions)
    pd.DataFrame(rows).sort_values(
        [
            "oof_tuned__macro_f1",
            "oof_tuned__balanced_accuracy",
            "oof_tuned__roc_auc",
        ],
        ascending=False,
    ).to_csv(
        result_dir / "pooling_cv_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_predictions, ignore_index=True).to_csv(
        result_dir / "pooling_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_rf_results(
    evaluations: list[event_baseline.RFEvaluation],
    best: event_baseline.RFEvaluation,
    result_dir: Path,
    elapsed_seconds: float,
) -> None:
    rows = []
    for evaluation in evaluations:
        candidate = evaluation.candidate
        metrics_fixed = metrics_at_fixed_threshold(
            evaluation.oof_recordings
        )
        rows.append(
            {
                "candidate_key": candidate.key,
                "n_estimators": candidate.n_estimators,
                "max_depth": candidate.max_depth,
                "min_samples_leaf": candidate.min_samples_leaf,
                "max_features": candidate.max_features,
                "max_samples": event_baseline.MAX_SAMPLES,
                "threshold_oof": evaluation.threshold,
                "selected_rf": candidate.key == best.candidate.key,
                "elapsed_seconds_total_search": elapsed_seconds,
                **prefixed_metrics(evaluation.metrics, "oof_tuned__"),
                **prefixed_metrics(metrics_fixed, "fixed_0p5__"),
            }
        )
    pd.DataFrame(rows).sort_values(
        [
            "oof_tuned__macro_f1",
            "oof_tuned__balanced_accuracy",
            "oof_tuned__roc_auc",
        ],
        ascending=False,
    ).to_csv(
        result_dir / "rf_candidate_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions = best.oof_recordings.copy()
    predictions["y_pred_fixed_0p5"] = (
        predictions["score"].to_numpy(dtype=float) >= FIXED_THRESHOLD
    ).astype(int)
    predictions.to_csv(
        result_dir / "best_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best.fold_metrics.to_csv(
        result_dir / "best_cv_fold_metrics_oof_threshold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_metrics_at_threshold(predictions, FIXED_THRESHOLD).to_csv(
        result_dir / "best_cv_fold_metrics_fixed_0p5.csv",
        index=False,
        encoding="utf-8-sig",
    )


def summary_rows(
    dataset: str,
    pooling: PoolingSpec,
    candidate: event_baseline.RFCandidate,
    feature_count: int,
    tuned_threshold: float,
    predictions: pd.DataFrame,
    model_size_kb: float,
    complexity: dict[str, float | int] | None = None,
) -> list[dict[str, object]]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    shared = {
        "dataset": dataset,
        "experiment": "wavelet_scattering_recording_rf",
        "pooling_key": pooling.key,
        "candidate_key": candidate.key,
        "feature_count": feature_count,
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
    event_data: common.DataView,
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

    print("\n" + "=" * 78)
    print("FASE 1 - COMPARACION DE AGREGACIONES RECORDING CON RF FIJO")
    print("=" * 78)
    pooling_results, selected_pooling = evaluate_poolings(event_data)
    for item in pooling_results:
        print(
            f"{item.spec.key} | features={item.data.x_train.shape[1]} | "
            f"macro-F1 OOF={item.evaluation.metrics['macro_f1']:.4f} | "
            f"macro-F1 @0.5={item.metrics_fixed_0p5['macro_f1']:.4f} | "
            f"AUC={item.evaluation.metrics['roc_auc']:.4f}"
        )
    print(f"Agregacion seleccionada: {selected_pooling.spec.key}")
    save_pooling_results(
        pooling_results,
        selected_pooling,
        result_dir,
    )

    selected_data = selected_pooling.data
    candidates = rf_grid(quick)
    print("\n" + "=" * 78)
    print("FASE 2 - BUSQUEDA RF PARA LA AGREGACION SELECCIONADA")
    print("=" * 78)
    print(f"Pooling: {selected_pooling.spec.key}")
    print(f"Candidatos RF: {len(candidates)}")
    start = time.perf_counter()
    rf_evaluations = event_baseline.evaluate_candidates_oof(
        selected_data,
        candidates,
    )
    elapsed_seconds = time.perf_counter() - start
    best = max(rf_evaluations, key=rf_selection_key)
    save_rf_results(
        rf_evaluations,
        best,
        result_dir,
        elapsed_seconds,
    )

    best_fixed_metrics = metrics_at_fixed_threshold(best.oof_recordings)
    print(f"Mejor RF: {best.candidate.key}")
    print(f"Umbral OOF: {best.threshold:.6f}")
    print(f"Macro-F1 OOF: {best.metrics['macro_f1']:.4f}")
    print(f"Macro-F1 OOF @0.5: {best_fixed_metrics['macro_f1']:.4f}")

    feature_count = selected_data.x_train.shape[1]
    oof_rows = summary_rows(
        "train_oof",
        selected_pooling.spec,
        best.candidate,
        feature_count,
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
        print("\nPrueba rapida recording WST + RF completada.")
        print("VALIDATION no se ha evaluado.")
        print(f"Resultados: {result_dir}")
        return

    print("\nAjustando RF final con una fila por grabacion de TRAIN...")
    classifier = event_baseline.build_rf(best.candidate)
    classifier.fit(
        selected_data.x_train,
        selected_data.train_metadata["stage2_target"].to_numpy(dtype=int),
        sample_weight=common.compute_training_weights(
            selected_data.train_metadata
        ),
    )
    validation_scores = event_baseline.rf_scores(
        classifier,
        selected_data.x_validation,
    )
    validation_predictions = common.aggregate_scores_by_recording(
        selected_data.validation_metadata,
        validation_scores,
    )
    validation_predictions["y_pred_oof_threshold"] = (
        validation_predictions["score"].to_numpy(dtype=float) >= best.threshold
    ).astype(int)
    validation_predictions["y_pred_fixed_0p5"] = (
        validation_predictions["score"].to_numpy(dtype=float) >= FIXED_THRESHOLD
    ).astype(int)
    validation_predictions["pooling_key"] = selected_pooling.spec.key
    validation_predictions["candidate_key"] = best.candidate.key

    complexity = event_baseline.forest_complexity(classifier)
    feature_table = pd.DataFrame(
        {
            "feature_index": np.arange(feature_count),
            "feature_name": selected_data.feature_names,
        }
    )
    feature_table.to_csv(
        result_dir / "recording_feature_names.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_data.train_metadata.to_csv(
        result_dir / "metadata_recordings_train.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_data.validation_metadata.to_csv(
        result_dir / "metadata_recordings_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_package = {
        "classifier": classifier,
        "preset": preset,
        "experiment": "wavelet_scattering_recording_rf",
        "pooling_key": selected_pooling.spec.key,
        "pooling_statistics": selected_pooling.spec.statistics,
        "candidate_key": best.candidate.key,
        "threshold": best.threshold,
        "fixed_reference_threshold": FIXED_THRESHOLD,
        "label_mapping": common.LABEL_TO_NAME,
        "event_feature_count": event_data.x_train.shape[1],
        "recording_feature_count": feature_count,
        "event_feature_names": event_data.feature_names,
        "recording_feature_names": selected_data.feature_names,
        "within_event_temporal_pooling": "mean_scattering_positions",
        "between_event_pooling": selected_pooling.spec.statistics,
        "prediction_unit": "one_original_uuid",
        "pca_used": False,
        "standardization_used": False,
        "smote_used": False,
        "classifier_weighting": "class_balanced_per_recording",
        "extraction_configuration": extraction_configuration.iloc[0].to_dict(),
        "random_state": RANDOM_STATE,
        **complexity,
    }
    model_path = model_dir / "wavelet_scattering_recording_rf_model.joblib"
    joblib.dump(model_package, model_path, compress=3)
    model_size_kb = model_path.stat().st_size / 1024.0

    for row in oof_rows:
        row["model_size_kb_joblib"] = model_size_kb
        row.update(complexity)
    validation_rows = summary_rows(
        "validation",
        selected_pooling.spec,
        best.candidate,
        feature_count,
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
                "event_feature_count": event_data.x_train.shape[1],
                "recording_feature_count": feature_count,
                "pooling_key": selected_pooling.spec.key,
                "pooling_statistics": "|".join(
                    selected_pooling.spec.statistics
                ),
                "pooling_selection_rf": FIXED_POOLING_RF.key,
                "pooling_selection_metric": "macro_f1_oof_tuned",
                "selected_rf": best.candidate.key,
                "threshold_selection": "train_oof_macro_f1",
                "fixed_threshold_reported": FIXED_THRESHOLD,
                "pca_used": False,
                "standardization_used": False,
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

    graph_oof_threshold = (
        graph_dir / "validation_recording_rf_oof_threshold.png"
    )
    graph_fixed_threshold = (
        graph_dir / "validation_recording_rf_fixed_0p5.png"
    )
    common.create_validation_graph(
        validation_predictions,
        best.threshold,
        "wavelet_scattering_recording_rf_oof_threshold",
        best.candidate,
        graph_oof_threshold,
    )
    common.create_validation_graph(
        validation_predictions,
        FIXED_THRESHOLD,
        "wavelet_scattering_recording_rf_fixed_0p5",
        best.candidate,
        graph_fixed_threshold,
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
    print("RESULTADO WST RECORDING-LEVEL + RANDOM FOREST")
    print("=" * 78)
    print(f"Pooling seleccionado: {selected_pooling.spec.key}")
    print(f"Features por grabacion: {feature_count}")
    print(f"Mejor RF: {best.candidate.key}")
    print(f"Umbral OOF congelado: {best.threshold:.6f}")
    print(f"Macro-F1 OOF: {best.metrics['macro_f1']:.4f}")
    print(f"Macro-F1 OOF @0.5: {best_fixed_metrics['macro_f1']:.4f}")
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
    print(
        f"RF: {complexity['rf_tree_count']} arboles, "
        f"{complexity['rf_total_nodes']} nodos"
    )
    print(f"Modelo joblib: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print(f"Graficas: {graph_dir}")
    print("TEST permanece reservado.")


def print_check(event_data: common.DataView, quick: bool) -> None:
    train_recordings = event_data.train_metadata.drop_duplicates(
        "original_uuid"
    )
    validation_recordings = event_data.validation_metadata.drop_duplicates(
        "original_uuid"
    )
    print("=" * 78)
    print("CHECK - WST RECORDING-LEVEL + RANDOM FOREST")
    print("=" * 78)
    print(f"Eventos TRAIN/VALIDATION: {len(event_data.x_train)} / "
          f"{len(event_data.x_validation)}")
    print(f"Grabaciones TRAIN/VALIDATION: {len(train_recordings)} / "
          f"{len(validation_recordings)}")
    print(f"Coeficientes WST por evento: {event_data.x_train.shape[1]}")
    for spec in POOLING_SPECS:
        print(
            f"  {spec.key}: {spec.statistics} -> "
            f"{event_data.x_train.shape[1] * spec.statistic_count} features"
        )
    print(f"RF fijo para comparar poolings: {FIXED_POOLING_RF.key}")
    print(f"Candidatos RF tras elegir pooling: {len(rf_grid(quick))}")
    print("Se reportaran umbral OOF y umbral fijo 0.5.")
    print("Sin PCA, SMOTE ni estandarizacion.")
    print("VALIDATION no selecciona configuracion. TEST no se procesa.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RF recording-level con agregaciones de coeficientes WST."
    )
    parser.add_argument(
        "--action",
        choices=["check", "train"],
        default="check",
    )
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default="paper_q8_q1_t500_full",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Compara poolings y cuatro RF mediante OOF; no evalua VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST RECORDING-LEVEL + RANDOM FOREST")
    print("=" * 78)
    print(f"Preset: {args.preset}")
    print(f"Accion: {args.action}")
    print("TEST no sera leido ni procesado.")
    event_data, extraction_configuration = event_baseline.load_wavelet_data(
        args.preset
    )
    print_check(event_data, args.quick)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(
        event_data,
        extraction_configuration,
        args.preset,
        args.quick,
    )


if __name__ == "__main__":
    main()
