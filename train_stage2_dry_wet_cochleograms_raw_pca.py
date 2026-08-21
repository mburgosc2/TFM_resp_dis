"""Experimento Stage 2: cocleograma 4096 -> PCA -> modelo lineal.

La PCA se ajusta dentro de cada fold y pondera cada evento con
``1 / numero_de_eventos_de_su_grabacion``. De este modo, cada grabacion
influye igual en la representacion no supervisada. El clasificador usa
ademas pesos de balanceo dry/wet calculados solo con el train del fold.

Las predicciones se calculan por evento y sus scores se promedian por
``original_uuid``. TEST no se lee ni se procesa.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import randomized_svd

import train_stage2_dry_wet_cochleograms as common


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_ROOT = (
    SCRIPT_DIR / "features_extracted_stage2_dry_wet_cochleograms_raw"
)
RESULTS_ROOT = (
    SCRIPT_DIR / "results_stage2_dry_wet_cochleograms_raw_pca"
)
MODELS_ROOT = SCRIPT_DIR / "models_stage2_dry_wet_cochleograms_raw_pca"
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_cochleograms_raw_pca"
)

RANDOM_STATE = 42
RAW_DIMENSION = 64 * 64
FULL_PCA_COMPONENTS = (16, 32, 64, 128)
FULL_C_VALUES = {
    "logistic_regression": (0.01, 0.1, 1.0, 10.0),
    "linear_svm": (0.001, 0.01, 0.1, 1.0),
}


@dataclass
class WeightedProjection:
    weighted_mean: np.ndarray
    components: np.ndarray
    component_mean: np.ndarray
    component_scale: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(
        self,
        x_values: np.ndarray,
        n_components: int | None = None,
    ) -> np.ndarray:
        component_count = (
            self.components.shape[0]
            if n_components is None
            else n_components
        )
        if component_count > self.components.shape[0]:
            raise ValueError("Se solicitaron mas componentes de las ajustadas.")
        centered = x_values - self.weighted_mean
        projected = centered @ self.components[:component_count].T
        standardized = (
            projected - self.component_mean[:component_count]
        ) / self.component_scale[:component_count]
        result = np.asarray(standardized, dtype=np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("La proyeccion PCA contiene NaN o infinito.")
        return result

    def selected(self, n_components: int) -> "WeightedProjection":
        return WeightedProjection(
            weighted_mean=self.weighted_mean.copy(),
            components=self.components[:n_components].copy(),
            component_mean=self.component_mean[:n_components].copy(),
            component_scale=self.component_scale[:n_components].copy(),
            singular_values=self.singular_values[:n_components].copy(),
            explained_variance_ratio=(
                self.explained_variance_ratio[:n_components].copy()
            ),
        )


def load_raw_data(preset: str) -> common.DataView:
    input_dir = FEATURES_ROOT / preset
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"No se encuentra la extraccion raw: {input_dir}\n"
            "Ejecuta primero feature_extraction_stage2_cochleograms_raw.py."
        )

    def load_split(
        split_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        x_values = np.load(input_dir / f"X_events_{split_name}.npy")
        y_values = np.load(input_dir / f"y_events_{split_name}.npy")
        folds = np.load(input_dir / f"folds_events_{split_name}.npy")
        metadata = pd.read_csv(
            input_dir / f"metadata_events_features_{split_name}.csv"
        )
        common.validate_arrays_and_metadata(
            "event",
            split_name,
            x_values,
            y_values,
            folds,
            metadata,
        )
        if x_values.shape[1] != RAW_DIMENSION:
            raise ValueError(
                f"Se esperaban {RAW_DIMENSION} features raw, "
                f"no {x_values.shape[1]}."
            )
        if np.min(x_values) < 0.0 or np.max(x_values) > 1.0:
            raise ValueError("Los cocleogramas raw deben estar en [0, 1].")
        return x_values, y_values, folds, metadata

    x_train, _, _, train_metadata = load_split("train")
    x_validation, _, _, validation_metadata = load_split("validation")
    overlap = set(train_metadata["original_uuid"]) & set(
        validation_metadata["original_uuid"]
    )
    if overlap:
        raise ValueError(
            "Train y validation comparten original_uuid: "
            f"{sorted(overlap)[:10]}"
        )

    feature_names_path = input_dir / "raw_feature_names.csv"
    feature_names_df = pd.read_csv(feature_names_path)
    feature_names = feature_names_df["feature_name"].astype(str).tolist()
    if len(feature_names) != RAW_DIMENSION:
        raise ValueError("raw_feature_names.csv no contiene 4096 nombres.")

    return common.DataView(
        experiment="raw_pca_event",
        x_train=x_train,
        train_metadata=train_metadata,
        x_validation=x_validation,
        validation_metadata=validation_metadata,
        feature_names=feature_names,
    )


def candidate_specs(quick: bool) -> list[common.CandidateSpec]:
    if quick:
        return [
            common.CandidateSpec("logistic_regression", 1.0, 32),
            common.CandidateSpec("linear_svm", 0.1, 32),
        ]
    return [
        common.CandidateSpec(model_name, c_value, pca_components)
        for model_name, c_values in FULL_C_VALUES.items()
        for c_value in c_values
        for pca_components in FULL_PCA_COMPONENTS
    ]


def build_classifier(spec: common.CandidateSpec) -> Any:
    if spec.model_name == "logistic_regression":
        return LogisticRegression(
            C=spec.c_value,
            penalty="l2",
            solver="liblinear",
            max_iter=5_000,
            random_state=RANDOM_STATE,
        )
    if spec.model_name == "linear_svm":
        return LinearSVC(
            C=spec.c_value,
            penalty="l2",
            dual="auto",
            max_iter=20_000,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Modelo desconocido: {spec.model_name}")


def classifier_scores(classifier: Any, x_values: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(x_values)[:, 1]
    else:
        scores = classifier.decision_function(x_values)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(x_values),) or not np.isfinite(scores).all():
        raise RuntimeError("El clasificador genero scores invalidos.")
    return scores


def fit_weighted_projection(
    x_values: np.ndarray,
    metadata: pd.DataFrame,
    n_components: int,
) -> tuple[WeightedProjection, np.ndarray]:
    """PCA por SVD ponderada y escalado ponderado de sus componentes."""

    if n_components >= min(x_values.shape):
        raise ValueError(
            f"n_components={n_components} no es valido para {x_values.shape}."
        )
    recording_weights = common.compute_recording_equal_weights(metadata)
    weighted_mean = np.average(
        x_values,
        axis=0,
        weights=recording_weights,
    ).astype(np.float32)
    centered = np.asarray(
        x_values - weighted_mean,
        dtype=np.float32,
    )
    weighted_for_svd = centered * np.sqrt(recording_weights).astype(
        np.float32
    )[:, np.newaxis]
    _, singular_values, components = randomized_svd(
        weighted_for_svd,
        n_components=n_components,
        n_oversamples=12,
        n_iter=4,
        power_iteration_normalizer="auto",
        random_state=RANDOM_STATE,
        flip_sign=True,
    )
    components = np.asarray(components, dtype=np.float32)
    total_weighted_sum_squares = float(
        np.sum(weighted_for_svd.astype(np.float64) ** 2)
    )
    explained_variance_ratio = (
        np.asarray(singular_values, dtype=np.float64) ** 2
        / total_weighted_sum_squares
    ).astype(np.float32)
    raw_projection = centered @ components.T

    scaler = StandardScaler()
    scaler.fit(raw_projection, sample_weight=recording_weights)
    component_scale = np.asarray(scaler.scale_, dtype=np.float32)
    if np.any(component_scale <= 0) or not np.isfinite(component_scale).all():
        raise RuntimeError("PCA produjo componentes sin varianza valida.")
    projection = WeightedProjection(
        weighted_mean=weighted_mean,
        components=components,
        component_mean=np.asarray(scaler.mean_, dtype=np.float32),
        component_scale=component_scale,
        singular_values=np.asarray(singular_values, dtype=np.float32),
        explained_variance_ratio=explained_variance_ratio,
    )
    transformed = (
        raw_projection - projection.component_mean
    ) / projection.component_scale
    transformed = np.asarray(transformed, dtype=np.float32)
    if not np.isfinite(transformed).all():
        raise RuntimeError("El train transformado contiene NaN o infinito.")
    return projection, transformed


def evaluate_candidates_oof(
    data: common.DataView,
    specs: list[common.CandidateSpec],
) -> tuple[list[common.CandidateEvaluation], pd.DataFrame]:
    maximum_components = max(
        int(spec.pca_components or 0) for spec in specs
    )
    oof_scores = {
        spec.key: np.full(len(data.x_train), np.nan, dtype=float)
        for spec in specs
    }
    start_times = {spec.key: time.perf_counter() for spec in specs}
    variance_rows = []

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        print(f"  Ajustando PCA ponderada del fold {fold}...")
        validation_mask = data.train_metadata["fold"].to_numpy() == fold
        training_mask = ~validation_mask
        training_metadata = data.train_metadata.loc[
            training_mask
        ].reset_index(drop=True)
        projection, x_training_projected = fit_weighted_projection(
            data.x_train[training_mask],
            training_metadata,
            maximum_components,
        )
        cumulative_variance = np.cumsum(
            projection.explained_variance_ratio
        )
        for component_count in FULL_PCA_COMPONENTS:
            if component_count <= maximum_components:
                variance_rows.append(
                    {
                        "fold": fold,
                        "pca_components": component_count,
                        "cumulative_explained_variance_ratio": float(
                            cumulative_variance[component_count - 1]
                        ),
                        "cumulative_explained_variance_percent": float(
                            cumulative_variance[component_count - 1] * 100.0
                        ),
                    }
                )
        x_fold_validation_projected = projection.transform(
            data.x_train[validation_mask]
        )
        classifier_weights = common.compute_training_weights(
            training_metadata
        )
        y_training = training_metadata["stage2_target"].to_numpy(dtype=int)

        for spec in specs:
            component_count = int(spec.pca_components or 0)
            classifier = build_classifier(spec)
            classifier.fit(
                x_training_projected[:, :component_count],
                y_training,
                sample_weight=classifier_weights,
            )
            oof_scores[spec.key][validation_mask] = classifier_scores(
                classifier,
                x_fold_validation_projected[:, :component_count],
            )

    evaluations = []
    for spec in specs:
        sample_scores = oof_scores[spec.key]
        if not np.isfinite(sample_scores).all():
            raise RuntimeError(f"OOF incompleto para {spec.key}.")
        recordings = common.aggregate_scores_by_recording(
            data.train_metadata,
            sample_scores,
        )
        y_true = recordings["y_true"].to_numpy(dtype=int)
        recording_scores = recordings["score"].to_numpy(dtype=float)
        threshold = common.tune_threshold(
            y_true,
            recording_scores,
            spec.default_threshold,
        )
        metrics = common.binary_metrics(
            y_true,
            recording_scores,
            threshold,
        )
        recordings["y_pred"] = (
            recording_scores >= threshold
        ).astype(int)
        recordings["candidate_key"] = spec.key

        fold_rows = []
        for fold, fold_df in recordings.groupby("fold", sort=True):
            fold_rows.append(
                {
                    "fold": int(fold),
                    "recording_count": len(fold_df),
                    "threshold": threshold,
                    **common.binary_metrics(
                        fold_df["y_true"].to_numpy(dtype=int),
                        fold_df["score"].to_numpy(dtype=float),
                        threshold,
                    ),
                }
            )
        evaluations.append(
            common.CandidateEvaluation(
                spec=spec,
                threshold=threshold,
                metrics=metrics,
                oof_recordings=recordings,
                fold_metrics=pd.DataFrame(fold_rows),
                elapsed_seconds=(
                    time.perf_counter() - start_times[spec.key]
                ),
            )
        )
    return evaluations, pd.DataFrame(variance_rows)


def save_oof_results(
    evaluations: list[common.CandidateEvaluation],
    best: common.CandidateEvaluation,
    variance_by_fold: pd.DataFrame,
    result_dir: Path,
) -> None:
    rows = []
    for evaluation in evaluations:
        spec = evaluation.spec
        rows.append(
            {
                "experiment": "raw_pca_event",
                "candidate_key": spec.key,
                "model_name": spec.model_name,
                "C": spec.c_value,
                "pca_components": spec.pca_components,
                "score_type": spec.score_type,
                "threshold_oof": evaluation.threshold,
                "elapsed_seconds_total_run": evaluation.elapsed_seconds,
                **evaluation.metrics,
            }
        )
    pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy", "roc_auc"],
        ascending=False,
    ).to_csv(
        result_dir / "candidate_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best.oof_recordings.to_csv(
        result_dir / "best_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best.fold_metrics.to_csv(
        result_dir / "best_cv_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    variance_by_fold.to_csv(
        result_dir / "pca_explained_variance_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )


def train(
    data: common.DataView,
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

    specs = candidate_specs(quick)
    print("\n" + "=" * 78)
    print("CV RAW 4096 -> PCA PONDERADA -> CLASIFICADOR LINEAL")
    print("=" * 78)
    print(
        f"Candidatos={len(specs)}, PCA={sorted(set(s.pca_components for s in specs))}"
    )
    evaluations, variance_by_fold = evaluate_candidates_oof(data, specs)
    for evaluation in evaluations:
        print(
            f"{evaluation.spec.key} | "
            f"macro-F1={evaluation.metrics['macro_f1']:.4f} | "
            f"bal-acc={evaluation.metrics['balanced_accuracy']:.4f} | "
            f"AUC={evaluation.metrics['roc_auc']:.4f}"
        )
    best = max(evaluations, key=common.selection_key)
    save_oof_results(evaluations, best, variance_by_fold, result_dir)
    print("\nVarianza explicada acumulada de PCA (media de folds):")
    variance_summary = variance_by_fold.groupby("pca_components")[
        "cumulative_explained_variance_percent"
    ].agg(["mean", "std"])
    for component_count, row in variance_summary.iterrows():
        print(
            f"  {component_count:3d} componentes: "
            f"{row['mean']:.2f}% +/- {row['std']:.2f}%"
        )

    oof_summary = {
        "dataset": "train_oof",
        "experiment": "raw_pca_event",
        "candidate_key": best.spec.key,
        "threshold": best.threshold,
        "model_size_kb_joblib": np.nan,
        "learned_inference_float_count": np.nan,
        "estimated_float32_parameters_kb": np.nan,
        **best.metrics,
    }
    if quick:
        pd.DataFrame([oof_summary]).to_csv(
            result_dir / "metrics_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\nPrueba rapida completada.")
        print(f"Mejor OOF: {best.spec.key}")
        print(f"Macro-F1 OOF: {best.metrics['macro_f1']:.4f}")
        print("VALIDATION no se ha evaluado.")
        print(f"Resultados: {result_dir}")
        return

    maximum_components = max(FULL_PCA_COMPONENTS)
    print("\nAjustando PCA ponderada final con todo TRAIN...")
    projection, x_train_projected = fit_weighted_projection(
        data.x_train,
        data.train_metadata,
        maximum_components,
    )
    selected_components = int(best.spec.pca_components or 0)
    selected_projection = projection.selected(selected_components)
    final_cumulative_variance = np.cumsum(
        projection.explained_variance_ratio
    )
    pd.DataFrame(
        {
            "pca_components": FULL_PCA_COMPONENTS,
            "cumulative_explained_variance_ratio": [
                float(final_cumulative_variance[count - 1])
                for count in FULL_PCA_COMPONENTS
            ],
            "cumulative_explained_variance_percent": [
                float(final_cumulative_variance[count - 1] * 100.0)
                for count in FULL_PCA_COMPONENTS
            ],
        }
    ).to_csv(
        result_dir / "pca_explained_variance_final_train.csv",
        index=False,
        encoding="utf-8-sig",
    )
    x_validation_projected = selected_projection.transform(
        data.x_validation
    )
    classifier = build_classifier(best.spec)
    classifier.fit(
        x_train_projected[:, :selected_components],
        data.train_metadata["stage2_target"].to_numpy(dtype=int),
        sample_weight=common.compute_training_weights(
            data.train_metadata
        ),
    )
    validation_sample_scores = classifier_scores(
        classifier,
        x_validation_projected,
    )
    validation_recordings = common.aggregate_scores_by_recording(
        data.validation_metadata,
        validation_sample_scores,
    )
    validation_scores = validation_recordings["score"].to_numpy(float)
    validation_recordings["y_pred"] = (
        validation_scores >= best.threshold
    ).astype(int)
    validation_recordings["candidate_key"] = best.spec.key
    validation_metrics = common.binary_metrics(
        validation_recordings["y_true"].to_numpy(dtype=int),
        validation_scores,
        best.threshold,
    )

    model_path = model_dir / "raw_pca_event_model.joblib"
    model_package = {
        # Se guardan arrays simples en lugar de la dataclass para que el
        # artefacto pueda cargarse sin depender de ejecutar este archivo
        # como modulo con el mismo nombre.
        "projection": {
            "weighted_mean": selected_projection.weighted_mean,
            "components": selected_projection.components,
            "component_mean": selected_projection.component_mean,
            "component_scale": selected_projection.component_scale,
            "singular_values": selected_projection.singular_values,
            "explained_variance_ratio": (
                selected_projection.explained_variance_ratio
            ),
        },
        "classifier": classifier,
        "preset": preset,
        "experiment": "raw_pca_event",
        "candidate_key": best.spec.key,
        "model_name": best.spec.model_name,
        "C": best.spec.c_value,
        "pca_components": selected_components,
        "score_type": best.spec.score_type,
        "recording_aggregation": "mean_event_score",
        "threshold": best.threshold,
        "label_mapping": common.LABEL_TO_NAME,
        "raw_input_shape": (64, 64),
        "flatten_order": "C_filter_then_time",
        "unknown_rejection_calibrated": False,
        "pca_weighting": "each_recording_equal_via_1_over_event_count",
        "random_state": RANDOM_STATE,
    }
    joblib.dump(model_package, model_path, compress=3)
    model_size_kb = model_path.stat().st_size / 1024.0
    classifier_float_count = int(
        classifier.coef_.size + classifier.intercept_.size
    )
    learned_float_count = int(
        selected_projection.weighted_mean.size
        + selected_projection.components.size
        + selected_projection.component_mean.size
        + selected_projection.component_scale.size
        + classifier_float_count
    )

    validation_recordings.to_csv(
        result_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    deployment = {
        "model_size_kb_joblib": model_size_kb,
        "learned_inference_float_count": learned_float_count,
        "estimated_float32_parameters_kb": (
            learned_float_count * 4.0 / 1024.0
        ),
    }
    oof_summary.update(deployment)
    validation_summary = {
        "dataset": "validation",
        "experiment": "raw_pca_event",
        "candidate_key": best.spec.key,
        "threshold": best.threshold,
        **deployment,
        **validation_metrics,
    }
    pd.DataFrame([oof_summary, validation_summary]).to_csv(
        result_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "raw_dimension": RAW_DIMENSION,
                "maximum_components_fitted": maximum_components,
                "selected_components": selected_components,
                "pca_fit_weighting": "1_over_events_per_recording",
                "component_scaling": "weighted_standard_scaler",
                "classifier_weighting": (
                    "class_balanced_and_1_over_events_per_recording"
                ),
                "validation_used_for_selection": False,
                "test_processed": False,
            }
        ]
    ).to_csv(
        result_dir / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    graph_path = graph_dir / "validation_raw_pca_event.png"
    common.create_validation_graph(
        validation_recordings,
        best.threshold,
        "raw_pca_event",
        best.spec,
        graph_path,
    )
    print("\n" + "=" * 78)
    print("RESULTADO RAW PCA")
    print("=" * 78)
    print(f"Mejor configuracion: {best.spec.key}")
    print(f"Umbral OOF: {best.threshold:.6f}")
    print(f"Macro-F1 OOF: {best.metrics['macro_f1']:.4f}")
    print(f"Macro-F1 validation: {validation_metrics['macro_f1']:.4f}")
    print(
        "Recalls validation dry/wet: "
        f"{validation_metrics['dry_recall']:.4f} / "
        f"{validation_metrics['wet_recall']:.4f}"
    )
    print(f"Modelo: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print(f"Grafica: {graph_path}")
    print("TEST permanece reservado.")


def print_check(data: common.DataView) -> None:
    train_recordings = data.train_metadata.drop_duplicates("original_uuid")
    validation_recordings = data.validation_metadata.drop_duplicates(
        "original_uuid"
    )
    print("=" * 78)
    print("CHECK — ENTRENAMIENTO RAW PCA")
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
    print("PCA y umbral se seleccionaran solo con los folds de TRAIN.")
    print("TEST no sera leido ni procesado.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena Stage 2 con cocleograma raw y PCA ponderada."
    )
    parser.add_argument(
        "--action",
        choices=["check", "train"],
        default="check",
        help="check valida la extraccion; train ejecuta el experimento.",
    )
    parser.add_argument(
        "--preset",
        choices=["paper64"],
        default="paper64",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Prueba OOF con PCA=32 y dos modelos. No evalua validation."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 — COCLEOGRAMA RAW + PCA PONDERADA")
    print("=" * 78)
    print(f"Preset: {args.preset}")
    print(f"Accion: {args.action}")
    print("TEST no sera leido ni procesado.")
    data = load_raw_data(args.preset)
    print_check(data)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(data, args.preset, args.quick)


if __name__ == "__main__":
    main()
