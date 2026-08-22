"""Stage 2: WST recording -> StandardScaler -> PCA whiten -> SVM RBF.

La unidad es una grabacion con mean+std+max de los 644 caminos WST. Se
comparan PCA32/64/128, C y gamma mediante los cinco folds de TRAIN. PCA se
ajusta dentro de cada fold. El ganador y el umbral se congelan antes de
evaluar VALIDATION. TEST no se lee.

Se usa ``probability=False``: evita el coste adicional de Platt scaling y
permite medir el SVM RBF puro. Se guardan numero de support vectors, tamano y
coste aproximado de kernel para valorar su viabilidad en movil.
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
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording
import train_stage2_dry_wet_wavelet_scattering_rf as event_baseline


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (
    SCRIPT_DIR / "results_stage2_dry_wet_wavelet_scattering_recording_rbf_svm"
)
MODELS_ROOT = (
    SCRIPT_DIR / "models_stage2_dry_wet_wavelet_scattering_recording_rbf_svm"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wavelet_scattering_recording_rbf_svm"
)

RANDOM_STATE = 42
PRESETS = event_baseline.PRESETS
FIXED_THRESHOLD = 0.0
FULL_PCA_COMPONENTS = (32, 64, 128)
QUICK_PCA_COMPONENTS = (64, 128)
FULL_C_VALUES = (0.1, 1.0, 10.0, 100.0)
QUICK_C_VALUES = (1.0, 10.0)
FULL_GAMMA_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)
QUICK_GAMMA_MULTIPLIERS = (0.5, 1.0)
RECORDING_POOLING = next(
    spec for spec in recording.POOLING_SPECS if spec.key == "mean_std_max"
)


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class RBFCandidate:
    pca_components: int
    c_value: float
    gamma_multiplier: float

    @property
    def gamma(self) -> float:
        # Tras PCA whiten, gamma='scale' es aproximadamente 1/dimension.
        return self.gamma_multiplier / self.pca_components

    @property
    def key(self) -> str:
        return (
            f"rbf_svm__C{format_number(self.c_value)}"
            f"__gamma{format_number(self.gamma_multiplier)}overD"
            f"__pca{self.pca_components}"
        )


@dataclass
class CandidateEvaluation:
    candidate: RBFCandidate
    threshold: float
    metrics_tuned: dict[str, float | int]
    metrics_native: dict[str, float | int]
    predictions: pd.DataFrame
    fold_metrics_tuned: pd.DataFrame
    elapsed_seconds: float
    support_vectors_mean: float
    support_vectors_max: int
    support_vector_fraction_mean: float


def candidates(quick: bool) -> list[RBFCandidate]:
    components = QUICK_PCA_COMPONENTS if quick else FULL_PCA_COMPONENTS
    c_values = QUICK_C_VALUES if quick else FULL_C_VALUES
    multipliers = (
        QUICK_GAMMA_MULTIPLIERS if quick else FULL_GAMMA_MULTIPLIERS
    )
    return [
        RBFCandidate(component_count, c_value, multiplier)
        for component_count in components
        for c_value in c_values
        for multiplier in multipliers
    ]


def build_recording_data(event_data: common.DataView) -> common.DataView:
    data = recording.build_recording_view(event_data, RECORDING_POOLING)
    if data.x_train.shape[1] != 1932:
        raise ValueError(f"Se esperaban 1932 features: {data.x_train.shape}.")
    return data


def build_svc(candidate: RBFCandidate) -> SVC:
    return SVC(
        C=candidate.c_value,
        kernel="rbf",
        gamma=candidate.gamma,
        probability=False,
        shrinking=True,
        tol=1e-3,
        cache_size=1024,
        class_weight=None,
        max_iter=-1,
        random_state=RANDOM_STATE,
    )


def decision_scores(model: SVC | Pipeline, x_values: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(x_values), dtype=float)
    if scores.shape != (len(x_values),) or not np.isfinite(scores).all():
        raise RuntimeError("El SVM produjo scores invalidos.")
    return scores


def prepare_fold_projections(
    data: common.DataView,
    component_values: tuple[int, ...],
) -> tuple[
    dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    pd.DataFrame,
]:
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    variance_rows: list[dict[str, float | int]] = []
    fold_values = data.train_metadata["fold"].to_numpy(dtype=int)

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = fold_values == fold
        training_mask = ~validation_mask
        scaler = StandardScaler()
        x_fit_scaled = scaler.fit_transform(
            data.x_train[training_mask]
        ).astype(np.float32)
        x_validation_scaled = scaler.transform(
            data.x_train[validation_mask]
        ).astype(np.float32)

        for component_count in component_values:
            pca = PCA(
                n_components=component_count,
                svd_solver="randomized",
                n_oversamples=12,
                iterated_power=4,
                power_iteration_normalizer="auto",
                whiten=True,
                random_state=RANDOM_STATE,
            )
            x_fit = pca.fit_transform(x_fit_scaled).astype(np.float32)
            x_validation = pca.transform(x_validation_scaled).astype(np.float32)
            cache[(fold, component_count)] = (x_fit, x_validation)
            explained = float(np.sum(pca.explained_variance_ratio_))
            variance_rows.append(
                {
                    "fold": fold,
                    "pca_components": component_count,
                    "pca_whiten": True,
                    "cumulative_explained_variance_ratio": explained,
                    "cumulative_explained_variance_percent": explained * 100.0,
                }
            )
        print(f"  Fold {fold}: scaler y PCA independientes ajustados en TRAIN interno.")
    return cache, pd.DataFrame(variance_rows)


def evaluate_candidates(
    data: common.DataView,
    candidate_list: list[RBFCandidate],
) -> tuple[list[CandidateEvaluation], pd.DataFrame]:
    components = tuple(
        dict.fromkeys(candidate.pca_components for candidate in candidate_list)
    )
    projections, variance = prepare_fold_projections(data, components)
    oof = {
        candidate.key: np.full(len(data.x_train), np.nan, dtype=float)
        for candidate in candidate_list
    }
    elapsed = {candidate.key: 0.0 for candidate in candidate_list}
    support_counts: dict[str, list[int]] = {
        candidate.key: [] for candidate in candidate_list
    }
    support_fractions: dict[str, list[float]] = {
        candidate.key: [] for candidate in candidate_list
    }
    fold_values = data.train_metadata["fold"].to_numpy(dtype=int)

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = fold_values == fold
        training_mask = ~validation_mask
        metadata_fit = data.train_metadata.loc[training_mask].reset_index(drop=True)
        y_fit = metadata_fit["stage2_target"].to_numpy(dtype=int)
        weights = common.compute_training_weights(metadata_fit)
        for candidate in candidate_list:
            x_fit, x_validation = projections[
                (fold, candidate.pca_components)
            ]
            classifier = build_svc(candidate)
            started = time.perf_counter()
            classifier.fit(x_fit, y_fit, sample_weight=weights)
            elapsed[candidate.key] += time.perf_counter() - started
            oof[candidate.key][validation_mask] = decision_scores(
                classifier, x_validation
            )
            support_count = int(np.sum(classifier.n_support_))
            support_counts[candidate.key].append(support_count)
            support_fractions[candidate.key].append(
                support_count / len(x_fit)
            )

    evaluations = []
    for index, candidate in enumerate(candidate_list, start=1):
        scores = oof[candidate.key]
        if not np.isfinite(scores).all():
            raise RuntimeError(f"OOF incompleto para {candidate.key}.")
        predictions = common.aggregate_scores_by_recording(
            data.train_metadata, scores
        )
        y_true = predictions["y_true"].to_numpy(dtype=int)
        recording_scores = predictions["score"].to_numpy(dtype=float)
        threshold = common.tune_threshold(
            y_true, recording_scores, FIXED_THRESHOLD
        )
        metrics_tuned = common.binary_metrics(
            y_true, recording_scores, threshold
        )
        metrics_native = common.binary_metrics(
            y_true, recording_scores, FIXED_THRESHOLD
        )
        predictions["y_pred_oof_threshold"] = (
            recording_scores >= threshold
        ).astype(int)
        predictions["y_pred_native_threshold"] = (
            recording_scores >= FIXED_THRESHOLD
        ).astype(int)
        predictions["candidate_key"] = candidate.key
        evaluation = CandidateEvaluation(
            candidate=candidate,
            threshold=threshold,
            metrics_tuned=metrics_tuned,
            metrics_native=metrics_native,
            predictions=predictions,
            fold_metrics_tuned=recording.fold_metrics_at_threshold(
                predictions, threshold
            ),
            elapsed_seconds=elapsed[candidate.key],
            support_vectors_mean=float(np.mean(support_counts[candidate.key])),
            support_vectors_max=int(np.max(support_counts[candidate.key])),
            support_vector_fraction_mean=float(
                np.mean(support_fractions[candidate.key])
            ),
        )
        evaluations.append(evaluation)
        print(
            f"[{index:02d}/{len(candidate_list):02d}] {candidate.key} | "
            f"macro-F1={metrics_tuned['macro_f1']:.4f} | "
            f"AUC={metrics_tuned['roc_auc']:.4f} | "
            f"SV mean={evaluation.support_vectors_mean:.0f}"
        )
    return evaluations, variance


def selection_key(evaluation: CandidateEvaluation) -> tuple[float, ...]:
    # Rendimiento primero; en empates, menor coste de inferencia.
    return (
        float(evaluation.metrics_tuned["macro_f1"]),
        float(evaluation.metrics_tuned["balanced_accuracy"]),
        float(evaluation.metrics_tuned["roc_auc"]),
        -float(evaluation.support_vectors_mean * evaluation.candidate.pca_components),
    )


def candidate_frame(evaluations: list[CandidateEvaluation]) -> pd.DataFrame:
    rows = []
    for item in evaluations:
        candidate = item.candidate
        rows.append(
            {
                "candidate_key": candidate.key,
                "C": candidate.c_value,
                "gamma": candidate.gamma,
                "gamma_multiplier_over_dimension": candidate.gamma_multiplier,
                "pca_components": candidate.pca_components,
                "pca_whiten": True,
                "threshold_oof": item.threshold,
                "elapsed_fit_seconds": item.elapsed_seconds,
                "support_vectors_mean_cv": item.support_vectors_mean,
                "support_vectors_max_cv": item.support_vectors_max,
                "support_vector_fraction_mean_cv": item.support_vector_fraction_mean,
                "kernel_feature_products_mean_cv": (
                    item.support_vectors_mean * candidate.pca_components
                ),
                **{f"oof_tuned__{key}": value for key, value in item.metrics_tuned.items()},
                **{f"native__{key}": value for key, value in item.metrics_native.items()},
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["oof_tuned__macro_f1", "oof_tuned__balanced_accuracy", "oof_tuned__roc_auc"],
        ascending=False,
    )


def build_final_pipeline(candidate: RBFCandidate) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=candidate.pca_components,
                    svd_solver="randomized",
                    n_oversamples=12,
                    iterated_power=4,
                    power_iteration_normalizer="auto",
                    whiten=True,
                    random_state=RANDOM_STATE,
                ),
            ),
            ("classifier", build_svc(candidate)),
        ]
    )


def model_complexity(model: Pipeline) -> dict[str, float | int]:
    scaler = model.named_steps["scaler"]
    pca = model.named_steps["pca"]
    classifier = model.named_steps["classifier"]
    support_vectors = int(np.sum(classifier.n_support_))
    dimension = int(pca.n_components_)
    learned_float_count = int(
        scaler.mean_.size
        + scaler.scale_.size
        + pca.mean_.size
        + pca.components_.size
        + pca.explained_variance_.size
        + classifier.support_vectors_.size
        + classifier.dual_coef_.size
        + classifier.intercept_.size
    )
    return {
        "support_vectors_total": support_vectors,
        "support_vectors_dry": int(classifier.n_support_[0]),
        "support_vectors_wet": int(classifier.n_support_[1]),
        "support_vector_fraction_train": support_vectors / int(classifier.shape_fit_[0]),
        "kernel_feature_products_per_recording": support_vectors * dimension,
        "learned_inference_float_count": learned_float_count,
        "estimated_float32_parameters_kb": learned_float_count * 4.0 / 1024.0,
    }


def metric_rows(
    dataset: str,
    best: CandidateEvaluation,
    predictions: pd.DataFrame,
    model_size_kb: float,
    complexity: dict[str, float | int] | None,
) -> list[dict[str, Any]]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    candidate = best.candidate
    shared = {
        "dataset": dataset,
        "experiment": "wavelet_scattering_recording_rbf_svm",
        "pooling_key": RECORDING_POOLING.key,
        "candidate_key": candidate.key,
        "C": candidate.c_value,
        "gamma": candidate.gamma,
        "gamma_multiplier_over_dimension": candidate.gamma_multiplier,
        "pca_components": candidate.pca_components,
        "pca_whiten": True,
        "output_dimension": candidate.pca_components,
        "score_type": "wet_decision_score",
        "model_size_kb_joblib": model_size_kb,
        **(complexity or {}),
    }
    return [
        {
            **shared,
            "threshold_policy": "fixed_native",
            "threshold": FIXED_THRESHOLD,
            **common.binary_metrics(y_true, scores, FIXED_THRESHOLD),
        },
        {
            **shared,
            "threshold_policy": "oof_tuned_frozen",
            "threshold": best.threshold,
            **common.binary_metrics(y_true, scores, best.threshold),
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

    candidate_list = candidates(quick)
    print("\n" + "=" * 78)
    print("CV — WST RECORDING PCA WHITEN + SVM RBF")
    print("=" * 78)
    print(f"Candidatos: {len(candidate_list)}")
    evaluations, variance = evaluate_candidates(data, candidate_list)
    best = max(evaluations, key=selection_key)
    candidate_frame(evaluations).to_csv(
        result_dir / "candidate_cv_results.csv", index=False, encoding="utf-8-sig"
    )
    variance.to_csv(
        result_dir / "pca_explained_variance_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    best.predictions.to_csv(
        result_dir / "best_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    best.fold_metrics_tuned.to_csv(
        result_dir / "best_cv_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    recording.fold_metrics_at_threshold(
        best.predictions, FIXED_THRESHOLD
    ).to_csv(
        result_dir / "best_cv_fold_metrics_native.csv", index=False, encoding="utf-8-sig"
    )
    oof_rows = metric_rows("train_oof", best, best.predictions, np.nan, None)

    print(f"Mejor OOF: {best.candidate.key}")
    print(f"Macro-F1 OOF: {best.metrics_tuned['macro_f1']:.4f}")
    print(f"Umbral OOF: {best.threshold:.6f}")
    if quick:
        pd.DataFrame(oof_rows).to_csv(
            result_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig"
        )
        print("VALIDATION no se ha evaluado en modo --quick.")
        print(f"Resultados: {result_dir}")
        return

    print("\nAjustando PCA y SVM RBF finales con todo TRAIN...")
    model = build_final_pipeline(best.candidate)
    model.fit(
        data.x_train,
        data.train_metadata["stage2_target"].to_numpy(dtype=int),
        classifier__sample_weight=common.compute_training_weights(data.train_metadata),
    )
    validation_scores = decision_scores(model, data.x_validation)
    validation_predictions = common.aggregate_scores_by_recording(
        data.validation_metadata, validation_scores
    )
    validation_predictions["y_pred_oof_threshold"] = (
        validation_predictions["score"].to_numpy(dtype=float) >= best.threshold
    ).astype(int)
    validation_predictions["y_pred_native_threshold"] = (
        validation_predictions["score"].to_numpy(dtype=float) >= FIXED_THRESHOLD
    ).astype(int)
    validation_predictions["candidate_key"] = best.candidate.key

    complexity = model_complexity(model)
    model_package = {
        "pipeline": model,
        "experiment": "wavelet_scattering_recording_rbf_svm",
        "preset": preset,
        "pooling_key": RECORDING_POOLING.key,
        "pooling_statistics": RECORDING_POOLING.statistics,
        "candidate_key": best.candidate.key,
        "C": best.candidate.c_value,
        "gamma": best.candidate.gamma,
        "gamma_multiplier_over_dimension": best.candidate.gamma_multiplier,
        "pca_components": best.candidate.pca_components,
        "pca_whiten": True,
        "score_type": "wet_decision_score",
        "probability_calibrated": False,
        "threshold": best.threshold,
        "native_threshold": FIXED_THRESHOLD,
        "label_mapping": common.LABEL_TO_NAME,
        "recording_feature_names": data.feature_names,
        "smote_used": False,
        "extraction_configuration": extraction_configuration.iloc[0].to_dict(),
        "random_state": RANDOM_STATE,
        **complexity,
    }
    model_path = model_dir / "wavelet_scattering_recording_rbf_svm_model.joblib"
    joblib.dump(model_package, model_path, compress=3)
    model_size_kb = model_path.stat().st_size / 1024.0
    for row in oof_rows:
        row["model_size_kb_joblib"] = model_size_kb
        row.update(complexity)
    validation_rows = metric_rows(
        "validation", best, validation_predictions, model_size_kb, complexity
    )
    pd.DataFrame([*oof_rows, *validation_rows]).to_csv(
        result_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig"
    )
    validation_predictions.to_csv(
        result_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {
                "input_feature_count": data.x_train.shape[1],
                "pooling_key": RECORDING_POOLING.key,
                "pca_options": "32|64|128",
                "pca_whiten": True,
                "selected_candidate": best.candidate.key,
                "threshold_selection": "train_oof_macro_f1",
                "native_threshold": FIXED_THRESHOLD,
                "probability_calibrated": False,
                "smote_used": False,
                "validation_used_for_selection": False,
                "test_processed": False,
                **complexity,
            }
        ]
    ).to_csv(
        result_dir / "experiment_configuration.csv", index=False, encoding="utf-8-sig"
    )

    common.create_validation_graph(
        validation_predictions,
        best.threshold,
        "wavelet_scattering_recording_rbf_svm",
        best.candidate,
        graph_dir / "validation_rbf_svm_oof_threshold.png",
    )
    common.create_validation_graph(
        validation_predictions,
        FIXED_THRESHOLD,
        "wavelet_scattering_recording_rbf_svm",
        best.candidate,
        graph_dir / "validation_rbf_svm_native_threshold.png",
    )
    validation_tuned = common.binary_metrics(
        validation_predictions["y_true"].to_numpy(dtype=int),
        validation_predictions["score"].to_numpy(dtype=float),
        best.threshold,
    )
    validation_native = common.binary_metrics(
        validation_predictions["y_true"].to_numpy(dtype=int),
        validation_predictions["score"].to_numpy(dtype=float),
        FIXED_THRESHOLD,
    )
    print("\n" + "=" * 78)
    print("RESULTADO WST RECORDING PCA + SVM RBF")
    print("=" * 78)
    print(f"Mejor configuracion: {best.candidate.key}")
    print(f"Umbral OOF congelado: {best.threshold:.6f}")
    print(f"Macro-F1 OOF: {best.metrics_tuned['macro_f1']:.4f}")
    print(
        "Macro-F1 validation OOF/native: "
        f"{validation_tuned['macro_f1']:.4f} / {validation_native['macro_f1']:.4f}"
    )
    print(
        "Recalls validation dry/wet: "
        f"{validation_tuned['dry_recall']:.4f} / {validation_tuned['wet_recall']:.4f}"
    )
    print(
        "Support vectors total dry/wet: "
        f"{complexity['support_vectors_total']} "
        f"({complexity['support_vectors_dry']}/{complexity['support_vectors_wet']})"
    )
    print(
        "Productos kernel-feature por grabacion: "
        f"{complexity['kernel_feature_products_per_recording']}"
    )
    print(f"Modelo joblib: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print(f"Graficas: {graph_dir}")
    print("TEST permanece reservado.")


def print_check(data: common.DataView, quick: bool) -> None:
    train_counts = data.train_metadata["stage2_target"].value_counts()
    validation_counts = data.validation_metadata["stage2_target"].value_counts()
    print("=" * 78)
    print("CHECK — WST RECORDING PCA WHITEN + SVM RBF")
    print("=" * 78)
    print(f"X TRAIN/VALIDATION: {data.x_train.shape} / {data.x_validation.shape}")
    print(f"TRAIN dry/wet: {train_counts[0]} / {train_counts[1]}")
    print(f"VALIDATION dry/wet: {validation_counts[0]} / {validation_counts[1]}")
    print(f"Pooling: {RECORDING_POOLING.key}; entrada=1932 features")
    print(f"Candidatos: {len(candidates(quick))}")
    print("Pipeline: StandardScaler -> PCA whiten -> SVM RBF.")
    print("PCA raw1932 se excluye por coste de inferencia movil.")
    print("Sin SMOTE. VALIDATION no selecciona. TEST no se procesa.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WST recording mean+std+max con PCA y SVM RBF."
    )
    parser.add_argument("--action", choices=["check", "train"], default="check")
    parser.add_argument("--preset", choices=PRESETS, default="paper_q8_q1_t500_full")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="8 candidatos OOF; no evalua VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 — WST RECORDING — SVM RBF")
    print("=" * 78)
    print(f"Preset: {args.preset} | Accion: {args.action} | Quick: {args.quick}")
    print("TEST no sera leido ni procesado.")
    event_data, extraction_configuration = event_baseline.load_wavelet_data(args.preset)
    data = build_recording_data(event_data)
    print_check(data, args.quick)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(data, extraction_configuration, args.preset, args.quick)


if __name__ == "__main__":
    main()
