"""Stage 2 WST recording: StandardScaler/PCA + LR y SVM lineal.

Usa una fila por grabacion con mean+std+max de los 644 caminos WST (1932
features). Compara sin PCA y PCA 32/64/128/256. Todos los transformadores,
hiperparametros y umbrales se seleccionan exclusivamente con los cinco folds
de TRAIN. En modo completo se evalua en VALIDATION el ganador OOF de cada
familia; TEST no se lee ni se procesa.
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording
import train_stage2_dry_wet_wavelet_scattering_rf as event_baseline


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (
    SCRIPT_DIR / "results_stage2_dry_wet_wavelet_scattering_recording_linear"
)
MODELS_ROOT = (
    SCRIPT_DIR / "models_stage2_dry_wet_wavelet_scattering_recording_linear"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wavelet_scattering_recording_linear"
)

RANDOM_STATE = 42
PRESETS = event_baseline.PRESETS
DEFAULT_PRESET = "paper_q8_q1_t500_full"
EXPECTED_RECORDING_FEATURE_COUNT = 1932
EXPERIMENT_KEY = "wavelet_scattering_recording_linear"
DISPLAY_TITLE = "STAGE 2 — WST RECORDING — REGRESION LOGISTICA Y SVM LINEAL"
CV_TITLE = "CV — WST RECORDING + LR/SVM LINEAL"
RESULT_TITLE = "RESULTADO WST RECORDING + CLASIFICADORES LINEALES"
CHECK_TITLE = "CHECK — WST RECORDING + LR/SVM LINEAL"
PARSER_DESCRIPTION = "WST recording mean+std+max con LR y SVM lineal."
FEATURE_LOADER = event_baseline.load_wavelet_data
FULL_PCA_OPTIONS = (None, 32, 64, 128, 256)
QUICK_PCA_OPTIONS = (None, 64, 128)
FULL_C_VALUES = {
    "logistic_regression": (0.001, 0.01, 0.1, 1.0, 10.0),
    "linear_svm": (0.0001, 0.001, 0.01, 0.1, 1.0),
}
QUICK_C_VALUES = {
    "logistic_regression": (0.01, 0.1, 1.0),
    "linear_svm": (0.001, 0.01, 0.1),
}
RECORDING_POOLING = next(
    spec for spec in recording.POOLING_SPECS if spec.key == "mean_std_max"
)


@dataclass
class CandidateResult:
    spec: common.CandidateSpec
    threshold: float
    metrics_tuned: dict[str, float | int]
    metrics_native: dict[str, float | int]
    predictions: pd.DataFrame
    fold_metrics_tuned: pd.DataFrame
    elapsed_seconds: float


def build_recording_data(event_data: common.DataView) -> common.DataView:
    data = recording.build_recording_view(event_data, RECORDING_POOLING)
    expected = EXPECTED_RECORDING_FEATURE_COUNT
    if data.x_train.shape[1] != expected:
        raise ValueError(
            f"Se esperaban {expected} features mean+std+max: {data.x_train.shape}."
        )
    return data


def candidate_specs(quick: bool) -> list[common.CandidateSpec]:
    pca_options = QUICK_PCA_OPTIONS if quick else FULL_PCA_OPTIONS
    c_values = QUICK_C_VALUES if quick else FULL_C_VALUES
    return [
        common.CandidateSpec(model_name, c_value, pca_components)
        for model_name, values in c_values.items()
        for c_value in values
        for pca_components in pca_options
    ]


def build_classifier(spec: common.CandidateSpec):
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
    raise ValueError(f"Clasificador desconocido: {spec.model_name}")


def classifier_scores(classifier, x_values: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(x_values)[:, 1]
    else:
        scores = classifier.decision_function(x_values)
    result = np.asarray(scores, dtype=float)
    if result.shape != (len(x_values),) or not np.isfinite(result).all():
        raise RuntimeError("El clasificador produjo scores invalidos.")
    return result


def transformed_folds(
    data: common.DataView,
    pca_options: tuple[int | None, ...],
) -> tuple[dict[tuple[int, int | None], tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """Ajusta scaler/PCA una vez por fold y reutiliza las proyecciones."""
    cache: dict[tuple[int, int | None], tuple[np.ndarray, np.ndarray]] = {}
    variance_rows: list[dict[str, float | int]] = []
    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = data.train_metadata["fold"].to_numpy(dtype=int) == fold
        training_mask = ~validation_mask
        x_fit_raw = data.x_train[training_mask]
        x_validation_raw = data.x_train[validation_mask]

        scaler = StandardScaler()
        x_fit_scaled = scaler.fit_transform(x_fit_raw).astype(np.float32)
        x_validation_scaled = scaler.transform(x_validation_raw).astype(np.float32)
        cache[(fold, None)] = (x_fit_scaled, x_validation_scaled)

        # Cada dimensionalidad se ajusta por separado. Asi PCA64 en CV y la
        # PCA64 final representan exactamente el mismo procedimiento, sin
        # depender de recortar una PCA256 aproximada.
        for components in pca_options:
            if components is None:
                continue
            pca = PCA(
                n_components=components,
                svd_solver="randomized",
                n_oversamples=12,
                iterated_power=4,
                power_iteration_normalizer="auto",
                whiten=False,
                random_state=RANDOM_STATE,
            )
            x_fit_pca = pca.fit_transform(x_fit_scaled).astype(np.float32)
            x_validation_pca = pca.transform(x_validation_scaled).astype(np.float32)
            cache[(fold, components)] = (x_fit_pca, x_validation_pca)
            explained = float(np.sum(pca.explained_variance_ratio_))
            variance_rows.append(
                {
                    "fold": fold,
                    "pca_components": components,
                    "cumulative_explained_variance_ratio": explained,
                    "cumulative_explained_variance_percent": explained * 100.0,
                }
            )
        print(f"  Fold {fold}: StandardScaler y PCA ajustados solo con TRAIN interno.")
    return cache, pd.DataFrame(variance_rows)


def make_result(
    data: common.DataView,
    spec: common.CandidateSpec,
    scores: np.ndarray,
    elapsed_seconds: float,
) -> CandidateResult:
    predictions = common.aggregate_scores_by_recording(data.train_metadata, scores)
    y_true = predictions["y_true"].to_numpy(dtype=int)
    recording_scores = predictions["score"].to_numpy(dtype=float)
    threshold = common.tune_threshold(
        y_true, recording_scores, spec.default_threshold
    )
    metrics_tuned = common.binary_metrics(y_true, recording_scores, threshold)
    metrics_native = common.binary_metrics(
        y_true, recording_scores, spec.default_threshold
    )
    predictions["y_pred_oof_threshold"] = (
        recording_scores >= threshold
    ).astype(int)
    predictions["y_pred_native_threshold"] = (
        recording_scores >= spec.default_threshold
    ).astype(int)
    predictions["candidate_key"] = spec.key
    return CandidateResult(
        spec=spec,
        threshold=threshold,
        metrics_tuned=metrics_tuned,
        metrics_native=metrics_native,
        predictions=predictions,
        fold_metrics_tuned=recording.fold_metrics_at_threshold(
            predictions, threshold
        ),
        elapsed_seconds=elapsed_seconds,
    )


def evaluate_candidates(
    data: common.DataView,
    specs: list[common.CandidateSpec],
) -> tuple[list[CandidateResult], pd.DataFrame]:
    pca_options = tuple(dict.fromkeys(spec.pca_components for spec in specs))
    cache, variance = transformed_folds(data, pca_options)
    oof = {
        spec.key: np.full(len(data.x_train), np.nan, dtype=float) for spec in specs
    }
    elapsed = {spec.key: 0.0 for spec in specs}

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = data.train_metadata["fold"].to_numpy(dtype=int) == fold
        training_mask = ~validation_mask
        metadata_fit = data.train_metadata.loc[training_mask].reset_index(drop=True)
        y_fit = metadata_fit["stage2_target"].to_numpy(dtype=int)
        sample_weights = common.compute_training_weights(metadata_fit)
        for spec in specs:
            x_fit, x_validation = cache[(fold, spec.pca_components)]
            classifier = build_classifier(spec)
            started = time.perf_counter()
            classifier.fit(x_fit, y_fit, sample_weight=sample_weights)
            oof[spec.key][validation_mask] = classifier_scores(
                classifier, x_validation
            )
            elapsed[spec.key] += time.perf_counter() - started

    results = []
    for index, spec in enumerate(specs, start=1):
        if not np.isfinite(oof[spec.key]).all():
            raise RuntimeError(f"OOF incompleto para {spec.key}.")
        result = make_result(data, spec, oof[spec.key], elapsed[spec.key])
        results.append(result)
        print(
            f"[{index:02d}/{len(specs):02d}] {spec.key} | "
            f"macro-F1={result.metrics_tuned['macro_f1']:.4f} | "
            f"macro-F1 native={result.metrics_native['macro_f1']:.4f} | "
            f"AUC={result.metrics_tuned['roc_auc']:.4f}"
        )
    return results, variance


def selection_key(result: CandidateResult) -> tuple[float, ...]:
    dimension = (
        EXPECTED_RECORDING_FEATURE_COUNT
        if result.spec.pca_components is None
        else result.spec.pca_components
    )
    return (
        float(result.metrics_tuned["macro_f1"]),
        float(result.metrics_tuned["balanced_accuracy"]),
        float(result.metrics_tuned["roc_auc"]),
        -float(dimension),
    )


def candidate_results_frame(results: list[CandidateResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        spec = result.spec
        rows.append(
            {
                "candidate_key": spec.key,
                "model_name": spec.model_name,
                "C": spec.c_value,
                "pca_components": spec.pca_components,
                "output_dimension": (
                    EXPECTED_RECORDING_FEATURE_COUNT
                    if spec.pca_components is None
                    else spec.pca_components
                ),
                "score_type": spec.score_type,
                "native_threshold": spec.default_threshold,
                "threshold_oof": result.threshold,
                "elapsed_fit_seconds": result.elapsed_seconds,
                **{f"oof_tuned__{k}": v for k, v in result.metrics_tuned.items()},
                **{f"native__{k}": v for k, v in result.metrics_native.items()},
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["oof_tuned__macro_f1", "oof_tuned__balanced_accuracy", "oof_tuned__roc_auc"],
        ascending=False,
    )


def build_final_pipeline(
    data: common.DataView,
    spec: common.CandidateSpec,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if spec.pca_components is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=spec.pca_components,
                    svd_solver="randomized",
                    n_oversamples=12,
                    iterated_power=4,
                    power_iteration_normalizer="auto",
                    whiten=False,
                    random_state=RANDOM_STATE,
                ),
            )
        )
    steps.append(("classifier", build_classifier(spec)))
    pipeline = Pipeline(steps)
    pipeline.fit(
        data.x_train,
        data.train_metadata["stage2_target"].to_numpy(dtype=int),
        classifier__sample_weight=common.compute_training_weights(data.train_metadata),
    )
    return pipeline


def learned_float_count(model: Pipeline) -> int:
    scaler = model.named_steps["scaler"]
    count = int(scaler.mean_.size + scaler.scale_.size)
    if "pca" in model.named_steps:
        pca = model.named_steps["pca"]
        count += int(pca.mean_.size + pca.components_.size)
    classifier = model.named_steps["classifier"]
    count += int(classifier.coef_.size + classifier.intercept_.size)
    return count


def metric_rows(
    dataset: str,
    winner: CandidateResult,
    predictions: pd.DataFrame,
    model_size_kb: float,
    float_count: int,
) -> list[dict[str, Any]]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    spec = winner.spec
    shared = {
        "dataset": dataset,
        "experiment": EXPERIMENT_KEY,
        "pooling_key": RECORDING_POOLING.key,
        "model_family": spec.model_name,
        "candidate_key": spec.key,
        "C": spec.c_value,
        "pca_components": spec.pca_components,
        "output_dimension": (
            EXPECTED_RECORDING_FEATURE_COUNT
            if spec.pca_components is None
            else spec.pca_components
        ),
        "score_type": spec.score_type,
        "model_size_kb_joblib": model_size_kb,
        "learned_inference_float_count": float_count,
        "estimated_float32_parameters_kb": float_count * 4.0 / 1024.0,
    }
    return [
        {
            **shared,
            "threshold_policy": "fixed_native",
            "threshold": spec.default_threshold,
            **common.binary_metrics(y_true, scores, spec.default_threshold),
        },
        {
            **shared,
            "threshold_policy": "oof_tuned_frozen",
            "threshold": winner.threshold,
            **common.binary_metrics(y_true, scores, winner.threshold),
        },
    ]


def save_family_result(
    data: common.DataView,
    extraction_configuration: pd.DataFrame,
    preset: str,
    run_name: str,
    winner: CandidateResult,
    quick: bool,
) -> dict[str, Any]:
    family = winner.spec.model_name
    result_dir = RESULTS_ROOT / preset / run_name / family
    result_dir.mkdir(parents=True, exist_ok=True)
    winner.predictions.to_csv(
        result_dir / "best_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    winner.fold_metrics_tuned.to_csv(
        result_dir / "best_cv_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    native_folds = recording.fold_metrics_at_threshold(
        winner.predictions, winner.spec.default_threshold
    )
    native_folds.to_csv(
        result_dir / "best_cv_fold_metrics_native.csv", index=False, encoding="utf-8-sig"
    )
    oof_rows = metric_rows("train_oof", winner, winner.predictions, np.nan, 0)
    if quick:
        pd.DataFrame(oof_rows).to_csv(
            result_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig"
        )
        return oof_rows[-1]

    model = build_final_pipeline(data, winner.spec)
    validation_scores = common.model_scores(model, data.x_validation)
    validation_predictions = common.aggregate_scores_by_recording(
        data.validation_metadata, validation_scores
    )
    validation_predictions["y_pred_oof_threshold"] = (
        validation_predictions["score"].to_numpy(dtype=float) >= winner.threshold
    ).astype(int)
    validation_predictions["y_pred_native_threshold"] = (
        validation_predictions["score"].to_numpy(dtype=float)
        >= winner.spec.default_threshold
    ).astype(int)
    validation_predictions["candidate_key"] = winner.spec.key

    model_dir = MODELS_ROOT / preset / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{family}_model.joblib"
    package = {
        "pipeline": model,
        "experiment": EXPERIMENT_KEY,
        "preset": preset,
        "pooling_key": RECORDING_POOLING.key,
        "pooling_statistics": RECORDING_POOLING.statistics,
        "candidate_key": winner.spec.key,
        "model_family": family,
        "C": winner.spec.c_value,
        "pca_components": winner.spec.pca_components,
        "score_type": winner.spec.score_type,
        "threshold": winner.threshold,
        "native_threshold": winner.spec.default_threshold,
        "label_mapping": common.LABEL_TO_NAME,
        "recording_feature_names": data.feature_names,
        "smote_used": False,
        "extraction_configuration": extraction_configuration.iloc[0].to_dict(),
        "random_state": RANDOM_STATE,
    }
    joblib.dump(package, model_path, compress=3)
    model_size_kb = model_path.stat().st_size / 1024.0
    float_count = learned_float_count(model)
    for row in oof_rows:
        row["model_size_kb_joblib"] = model_size_kb
        row["learned_inference_float_count"] = float_count
        row["estimated_float32_parameters_kb"] = float_count * 4.0 / 1024.0
    validation_rows = metric_rows(
        "validation", winner, validation_predictions, model_size_kb, float_count
    )
    pd.DataFrame([*oof_rows, *validation_rows]).to_csv(
        result_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig"
    )
    validation_predictions.to_csv(
        result_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )

    graph_dir = GRAPHS_ROOT / preset / run_name
    graph_dir.mkdir(parents=True, exist_ok=True)
    common.create_validation_graph(
        validation_predictions,
        winner.threshold,
        EXPERIMENT_KEY,
        winner.spec,
        graph_dir / f"validation_{family}_oof_threshold.png",
    )
    common.create_validation_graph(
        validation_predictions,
        winner.spec.default_threshold,
        EXPERIMENT_KEY,
        winner.spec,
        graph_dir / f"validation_{family}_native_threshold.png",
    )
    return validation_rows[-1]


def train(
    data: common.DataView,
    extraction_configuration: pd.DataFrame,
    preset: str,
    quick: bool,
) -> None:
    run_name = "quick" if quick else "full"
    root = RESULTS_ROOT / preset / run_name
    root.mkdir(parents=True, exist_ok=True)
    specs = candidate_specs(quick)

    print("\n" + "=" * 78)
    print(CV_TITLE)
    print("=" * 78)
    print(f"Candidatos: {len(specs)}")
    results, variance = evaluate_candidates(data, specs)
    candidate_results_frame(results).to_csv(
        root / "candidate_cv_results.csv", index=False, encoding="utf-8-sig"
    )
    variance.to_csv(
        root / "pca_explained_variance_by_fold.csv", index=False, encoding="utf-8-sig"
    )

    family_winners = {
        family: max(
            [result for result in results if result.spec.model_name == family],
            key=selection_key,
        )
        for family in FULL_C_VALUES
    }
    overall = max(family_winners.values(), key=selection_key)
    comparison_rows = []
    for family, winner in family_winners.items():
        comparison_rows.append(
            {
                "model_family": family,
                "candidate_key": winner.spec.key,
                "selected_overall_oof": winner is overall,
                "threshold_oof": winner.threshold,
                **winner.metrics_tuned,
            }
        )
        print(
            f"Ganador {family}: {winner.spec.key} | "
            f"macro-F1 OOF={winner.metrics_tuned['macro_f1']:.4f}"
        )
    pd.DataFrame(comparison_rows).to_csv(
        root / "family_winners_oof.csv", index=False, encoding="utf-8-sig"
    )

    validation_rows = []
    for winner in family_winners.values():
        validation_rows.append(
            save_family_result(
                data, extraction_configuration, preset, run_name, winner, quick
            )
        )
    pd.DataFrame(validation_rows).to_csv(
        root / ("oof_family_comparison.csv" if quick else "validation_family_comparison.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "input_feature_count": data.x_train.shape[1],
                "pooling_key": RECORDING_POOLING.key,
                "pca_options": "|".join(
                    "none" if value is None else str(value)
                    for value in (
                        QUICK_PCA_OPTIONS if quick else FULL_PCA_OPTIONS
                    )
                ),
                "standard_scaler_used": True,
                "pca_whiten": False,
                "smote_used": False,
                "class_balance": "sample_weight balanced per recording",
                "selection_metric": "train_oof_macro_f1",
                "validation_used_for_selection": False,
                "test_processed": False,
                "overall_oof_winner": overall.spec.key,
            }
        ]
    ).to_csv(root / "experiment_configuration.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(RESULT_TITLE)
    print("=" * 78)
    print(f"Ganador global OOF: {overall.spec.key}")
    if quick:
        print("VALIDATION no se ha evaluado en modo --quick.")
    else:
        for row in validation_rows:
            print(
                f"{row['model_family']}: macro-F1 validation={row['macro_f1']:.4f} | "
                f"recalls dry/wet={row['dry_recall']:.4f}/{row['wet_recall']:.4f} | "
                f"modelo={row['model_size_kb_joblib']:.2f} KB"
            )
    print(f"Resultados: {root}")
    print("TEST permanece reservado.")


def print_check(data: common.DataView, quick: bool) -> None:
    train_counts = data.train_metadata["stage2_target"].value_counts()
    validation_counts = data.validation_metadata["stage2_target"].value_counts()
    print("=" * 78)
    print(CHECK_TITLE)
    print("=" * 78)
    print(f"X TRAIN/VALIDATION: {data.x_train.shape} / {data.x_validation.shape}")
    print(f"TRAIN dry/wet: {train_counts[0]} / {train_counts[1]}")
    print(f"VALIDATION dry/wet: {validation_counts[0]} / {validation_counts[1]}")
    print(f"Pooling: {RECORDING_POOLING.key}; features={data.x_train.shape[1]}")
    print(f"Candidatos: {len(candidate_specs(quick))}")
    print("StandardScaler siempre; PCA se ajusta dentro de cada fold.")
    print("Sin SMOTE. VALIDATION no selecciona. TEST no se procesa.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=PARSER_DESCRIPTION
    )
    parser.add_argument("--action", choices=["check", "train"], default="check")
    parser.add_argument("--preset", choices=PRESETS, default=DEFAULT_PRESET)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="18 candidatos OOF; no evalua VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print(DISPLAY_TITLE)
    print("=" * 78)
    print(f"Preset: {args.preset} | Accion: {args.action} | Quick: {args.quick}")
    print("TEST no sera leido ni procesado.")
    event_data, extraction_configuration = FEATURE_LOADER(args.preset)
    data = build_recording_data(event_data)
    print_check(data, args.quick)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(data, extraction_configuration, args.preset, args.quick)


if __name__ == "__main__":
    main()
