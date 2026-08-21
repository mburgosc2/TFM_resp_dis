"""Random Forest ligero sobre Wavelet Scattering completo para dry/wet.

Consume todos los caminos WST de orden 0, 1 y 2 por evento, después de
promediar sus posiciones temporales. No aplica PCA, seleccion de variables,
SMOTE ni estandarizacion. Cada grabacion aporta
el mismo peso total; las probabilidades de sus eventos se promedian antes de
calcular metricas. La seleccion de hiperparametros y del umbral usa solo los
cinco folds de TRAIN. VALIDATION se evalua una vez y TEST no se lee.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import train_stage2_dry_wet_cochleograms as common


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_ROOT = SCRIPT_DIR / "features_extracted_stage2_dry_wet_wavelet_scattering"
RESULTS_ROOT = SCRIPT_DIR / "results_stage2_dry_wet_wavelet_scattering_rf"
MODELS_ROOT = SCRIPT_DIR / "models_stage2_dry_wet_wavelet_scattering_rf"
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wavelet_scattering_rf"
)

RANDOM_STATE = 42
MAX_SAMPLES = 0.8
PRESETS = ("paper_q8_q1_t500_full",)


@dataclass(frozen=True)
class RFCandidate:
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    max_features: str | float

    @property
    def key(self) -> str:
        features_name = str(self.max_features).replace(".", "p")
        return (
            f"rf__trees{self.n_estimators}__depth{self.max_depth}__"
            f"leaf{self.min_samples_leaf}__features{features_name}"
        )

    @property
    def theoretical_max_nodes(self) -> int:
        return self.n_estimators * (2 ** (self.max_depth + 1) - 1)


@dataclass
class RFEvaluation:
    candidate: RFCandidate
    threshold: float
    metrics: dict[str, float | int]
    oof_recordings: pd.DataFrame
    fold_metrics: pd.DataFrame


def required_feature_paths(input_dir: Path) -> dict[str, Path]:
    return {
        "x_train": input_dir / "X_events_train.npy",
        "y_train": input_dir / "y_events_train.npy",
        "folds_train": input_dir / "folds_events_train.npy",
        "metadata_train": input_dir / "metadata_events_features_train.csv",
        "x_validation": input_dir / "X_events_validation.npy",
        "y_validation": input_dir / "y_events_validation.npy",
        "folds_validation": input_dir / "folds_events_validation.npy",
        "metadata_validation": (
            input_dir / "metadata_events_features_validation.csv"
        ),
        "feature_layout": (
            input_dir / "wavelet_scattering_feature_layout.csv"
        ),
        "configuration": (
            input_dir / "wavelet_scattering_configuration.csv"
        ),
    }


def load_wavelet_data(preset: str) -> tuple[common.DataView, pd.DataFrame]:
    input_dir = FEATURES_ROOT / preset
    paths = required_feature_paths(input_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos WST para el entrenamiento:\n" + "\n".join(missing)
        )

    x_train = np.load(paths["x_train"])
    y_train = np.load(paths["y_train"])
    folds_train = np.load(paths["folds_train"])
    train_metadata = pd.read_csv(paths["metadata_train"])
    x_validation = np.load(paths["x_validation"])
    y_validation = np.load(paths["y_validation"])
    folds_validation = np.load(paths["folds_validation"])
    validation_metadata = pd.read_csv(paths["metadata_validation"])
    feature_layout = pd.read_csv(paths["feature_layout"])
    configuration = pd.read_csv(paths["configuration"])

    common.validate_arrays_and_metadata(
        "event",
        "train",
        x_train,
        y_train,
        folds_train,
        train_metadata,
    )
    common.validate_arrays_and_metadata(
        "event",
        "validation",
        x_validation,
        y_validation,
        folds_validation,
        validation_metadata,
    )
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("TRAIN y VALIDATION tienen distinta dimension WST.")
    if len(feature_layout) != x_train.shape[1]:
        raise ValueError("El layout WST no coincide con las matrices.")
    if feature_layout["feature_index"].tolist() != list(range(len(feature_layout))):
        raise ValueError("feature_index no es consecutivo en el layout WST.")
    if set(feature_layout["scattering_order"].astype(int)) != {0, 1, 2}:
        raise ValueError("No estan presentes exactamente los ordenes WST 0, 1 y 2.")
    if len(configuration) != 1:
        raise ValueError("La configuracion WST debe contener exactamente una fila.")
    config_row = configuration.iloc[0]
    representation = {
        "temporal_aggregation": str(config_row["temporal_aggregation"]),
        "dimensionality_reduction": str(config_row["dimensionality_reduction"]),
    }
    if representation != {
        "temporal_aggregation": "mean_over_scattering_time_positions",
        "dimensionality_reduction": "none",
    }:
        raise ValueError(
            "La extraccion no corresponde a WST con media temporal: "
            f"{representation}"
        )

    train_uuids = set(train_metadata["original_uuid"].astype(str))
    validation_uuids = set(validation_metadata["original_uuid"].astype(str))
    overlap = train_uuids & validation_uuids
    if overlap:
        raise ValueError(
            "TRAIN y VALIDATION comparten original_uuid: "
            f"{sorted(overlap)[:10]}"
        )

    data = common.DataView(
        experiment="wavelet_scattering_full_event",
        x_train=x_train,
        train_metadata=train_metadata,
        x_validation=x_validation,
        validation_metadata=validation_metadata,
        feature_names=feature_layout["feature_name"].astype(str).tolist(),
    )
    return data, configuration


def rf_candidates(quick: bool) -> list[RFCandidate]:
    quick_candidates = [
        RFCandidate(100, 4, 15, "sqrt"),
        RFCandidate(150, 6, 10, "sqrt"),
        RFCandidate(200, 6, 10, 0.1),
    ]
    if quick:
        return quick_candidates

    grid_candidates = [
        RFCandidate(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        for n_estimators in (100, 200, 300)
        for max_depth in (4, 6, 8, 10)
        for min_samples_leaf in (5, 10)
        for max_features in ("sqrt", 0.1)
    ]
    # La busqueda completa siempre conserva los puntos de la prueba rapida,
    # incluido el eventual ganador de 150 arboles.
    return list(dict.fromkeys([*quick_candidates, *grid_candidates]))


def build_rf(candidate: RFCandidate) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=candidate.n_estimators,
        criterion="gini",
        max_depth=candidate.max_depth,
        min_samples_split=2 * candidate.min_samples_leaf,
        min_samples_leaf=candidate.min_samples_leaf,
        max_features=candidate.max_features,
        bootstrap=True,
        max_samples=MAX_SAMPLES,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def rf_scores(
    classifier: RandomForestClassifier,
    x_values: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(classifier.predict_proba(x_values)[:, 1], dtype=float)
    if scores.shape != (len(x_values),) or not np.isfinite(scores).all():
        raise RuntimeError("Random Forest produjo probabilidades invalidas.")
    return scores


def evaluate_candidates_oof(
    data: common.DataView,
    candidates: list[RFCandidate],
) -> list[RFEvaluation]:
    oof_scores = {
        candidate.key: np.full(len(data.x_train), np.nan, dtype=float)
        for candidate in candidates
    }

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        print(f"  Fold {fold}: WST completo sin PCA ni seleccion...")
        validation_mask = data.train_metadata["fold"].to_numpy() == fold
        training_mask = ~validation_mask
        training_metadata = data.train_metadata.loc[training_mask].reset_index(
            drop=True
        )
        y_training = training_metadata["stage2_target"].to_numpy(dtype=int)
        sample_weights = common.compute_training_weights(training_metadata)

        for candidate_index, candidate in enumerate(candidates, start=1):
            classifier = build_rf(candidate)
            classifier.fit(
                data.x_train[training_mask],
                y_training,
                sample_weight=sample_weights,
            )
            oof_scores[candidate.key][validation_mask] = rf_scores(
                classifier,
                data.x_train[validation_mask],
            )
            print(
                f"    RF {candidate_index:02d}/{len(candidates):02d}: "
                f"{candidate.key}"
            )

    evaluations: list[RFEvaluation] = []
    for candidate in candidates:
        sample_scores = oof_scores[candidate.key]
        if not np.isfinite(sample_scores).all():
            raise RuntimeError(f"OOF incompleto para {candidate.key}.")
        recordings = common.aggregate_scores_by_recording(
            data.train_metadata,
            sample_scores,
        )
        y_true = recordings["y_true"].to_numpy(dtype=int)
        scores = recordings["score"].to_numpy(dtype=float)
        threshold = common.tune_threshold(y_true, scores, 0.5)
        metrics = common.binary_metrics(y_true, scores, threshold)
        recordings["y_pred"] = (scores >= threshold).astype(int)
        recordings["candidate_key"] = candidate.key

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
            RFEvaluation(
                candidate=candidate,
                threshold=threshold,
                metrics=metrics,
                oof_recordings=recordings,
                fold_metrics=pd.DataFrame(fold_rows),
            )
        )
    return evaluations


def selection_key(evaluation: RFEvaluation) -> tuple[float, ...]:
    return (
        float(evaluation.metrics["macro_f1"]),
        float(evaluation.metrics["balanced_accuracy"]),
        float(evaluation.metrics["roc_auc"]),
        -float(evaluation.candidate.theoretical_max_nodes),
    )


def forest_complexity(
    classifier: RandomForestClassifier,
) -> dict[str, float | int]:
    node_counts = np.asarray(
        [estimator.tree_.node_count for estimator in classifier.estimators_]
    )
    leaf_counts = np.asarray(
        [estimator.tree_.n_leaves for estimator in classifier.estimators_]
    )
    depths = np.asarray(
        [estimator.tree_.max_depth for estimator in classifier.estimators_]
    )
    return {
        "rf_tree_count": len(classifier.estimators_),
        "rf_total_nodes": int(node_counts.sum()),
        "rf_total_leaves": int(leaf_counts.sum()),
        "rf_mean_nodes_per_tree": float(node_counts.mean()),
        "rf_max_observed_depth": int(depths.max()),
    }


def save_oof_results(
    evaluations: list[RFEvaluation],
    best: RFEvaluation,
    result_dir: Path,
    elapsed_seconds: float,
) -> None:
    rows = []
    for evaluation in evaluations:
        candidate = evaluation.candidate
        rows.append(
            {
                "candidate_key": candidate.key,
                "n_estimators": candidate.n_estimators,
                "max_depth": candidate.max_depth,
                "min_samples_leaf": candidate.min_samples_leaf,
                "max_features": candidate.max_features,
                "max_samples": MAX_SAMPLES,
                "theoretical_max_nodes": candidate.theoretical_max_nodes,
                "threshold_oof": evaluation.threshold,
                "elapsed_seconds_total_search": elapsed_seconds,
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

    candidates = rf_candidates(quick)
    print("\n" + "=" * 78)
    print("CV WAVELET SCATTERING COMPLETO -> RANDOM FOREST")
    print("=" * 78)
    print(f"Candidatos RF: {len(candidates)}")
    start_time = time.perf_counter()
    evaluations = evaluate_candidates_oof(data, candidates)
    elapsed_seconds = time.perf_counter() - start_time
    for evaluation in evaluations:
        print(
            f"{evaluation.candidate.key} | "
            f"macro-F1={evaluation.metrics['macro_f1']:.4f} | "
            f"bal-acc={evaluation.metrics['balanced_accuracy']:.4f} | "
            f"AUC={evaluation.metrics['roc_auc']:.4f}"
        )

    best = max(evaluations, key=selection_key)
    save_oof_results(
        evaluations,
        best,
        result_dir,
        elapsed_seconds,
    )
    oof_summary = {
        "dataset": "train_oof",
        "experiment": "wavelet_scattering_full_rf_event",
        "candidate_key": best.candidate.key,
        "threshold": best.threshold,
        "input_feature_count": data.x_train.shape[1],
        "model_size_kb_joblib": np.nan,
        **best.metrics,
    }

    if quick:
        pd.DataFrame([oof_summary]).to_csv(
            result_dir / "metrics_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\nPrueba rapida WST + RF completada.")
        print(f"Mejor OOF: {best.candidate.key}")
        print(f"Macro-F1 OOF: {best.metrics['macro_f1']:.4f}")
        print("VALIDATION no se ha evaluado.")
        print(f"Resultados: {result_dir}")
        return

    print("\nAjustando Random Forest final con todo TRAIN...")
    classifier = build_rf(best.candidate)
    classifier.fit(
        data.x_train,
        data.train_metadata["stage2_target"].to_numpy(dtype=int),
        sample_weight=common.compute_training_weights(data.train_metadata),
    )
    validation_event_scores = rf_scores(classifier, data.x_validation)
    validation_recordings = common.aggregate_scores_by_recording(
        data.validation_metadata,
        validation_event_scores,
    )
    validation_scores = validation_recordings["score"].to_numpy(dtype=float)
    validation_recordings["y_pred"] = (
        validation_scores >= best.threshold
    ).astype(int)
    validation_recordings["candidate_key"] = best.candidate.key
    validation_metrics = common.binary_metrics(
        validation_recordings["y_true"].to_numpy(dtype=int),
        validation_scores,
        best.threshold,
    )

    complexity = forest_complexity(classifier)
    model_package = {
        "classifier": classifier,
        "preset": preset,
        "experiment": "wavelet_scattering_full_rf_event",
        "candidate_key": best.candidate.key,
        "threshold": best.threshold,
        "label_mapping": common.LABEL_TO_NAME,
        "input_feature_count": data.x_train.shape[1],
        "feature_names": data.feature_names,
        "recording_aggregation": "mean_event_probability",
        "within_event_temporal_pooling": "mean_scattering_positions",
        "pca_used": False,
        "feature_selection_used": False,
        "standardization_used": False,
        "smote_used": False,
        "classifier_weighting": (
            "class_balanced_and_1_over_events_per_recording"
        ),
        "extraction_configuration": extraction_configuration.iloc[0].to_dict(),
        "random_state": RANDOM_STATE,
        **complexity,
    }
    model_path = model_dir / "wavelet_scattering_full_rf_event_model.joblib"
    joblib.dump(model_package, model_path, compress=3)
    model_size_kb = model_path.stat().st_size / 1024.0

    oof_summary.update(
        {"model_size_kb_joblib": model_size_kb, **complexity}
    )
    validation_summary = {
        "dataset": "validation",
        "experiment": "wavelet_scattering_full_rf_event",
        "candidate_key": best.candidate.key,
        "threshold": best.threshold,
        "input_feature_count": data.x_train.shape[1],
        "model_size_kb_joblib": model_size_kb,
        **complexity,
        **validation_metrics,
    }
    pd.DataFrame([oof_summary, validation_summary]).to_csv(
        result_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_recordings.to_csv(
        result_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "input_feature_count": data.x_train.shape[1],
                "pca_used": False,
                "feature_selection_used": False,
                "temporal_aggregation_used": True,
                "temporal_aggregation": "mean_scattering_positions",
                "standardization_used": False,
                "smote_used": False,
                "max_samples": MAX_SAMPLES,
                "validation_used_for_hyperparameter_selection": False,
                "test_processed": False,
                **best.candidate.__dict__,
                **complexity,
            }
        ]
    ).to_csv(
        result_dir / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    graph_path = graph_dir / "validation_wavelet_scattering_full_rf_event.png"
    common.create_validation_graph(
        validation_recordings,
        best.threshold,
        "wavelet_scattering_full_rf_event",
        best.candidate,
        graph_path,
    )
    print("\n" + "=" * 78)
    print("RESULTADO WAVELET SCATTERING COMPLETO + RANDOM FOREST")
    print("=" * 78)
    print(f"Mejor configuracion: {best.candidate.key}")
    print(f"Umbral OOF: {best.threshold:.6f}")
    print(f"Macro-F1 OOF: {best.metrics['macro_f1']:.4f}")
    print(f"Macro-F1 validation: {validation_metrics['macro_f1']:.4f}")
    print(
        "Recalls validation dry/wet: "
        f"{validation_metrics['dry_recall']:.4f} / "
        f"{validation_metrics['wet_recall']:.4f}"
    )
    print(
        f"RF: {complexity['rf_tree_count']} arboles, "
        f"{complexity['rf_total_nodes']} nodos"
    )
    print(f"Modelo joblib: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print(f"Grafica: {graph_path}")
    print("TEST permanece reservado.")


def print_check(data: common.DataView, quick: bool) -> None:
    train_recordings = data.train_metadata.drop_duplicates("original_uuid")
    validation_recordings = data.validation_metadata.drop_duplicates(
        "original_uuid"
    )
    print("=" * 78)
    print("CHECK - WAVELET SCATTERING COMPLETO + RANDOM FOREST")
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
        "Caminos WST por evento tras media temporal: "
        f"{data.x_train.shape[1]}"
    )
    print(f"Candidatos RF: {len(rf_candidates(quick))}")
    print("Media temporal por camino; sin PCA, seleccion, estandarizacion ni SMOTE.")
    print("Balanceo mediante sample_weight dentro de cada fold.")
    print("TEST no sera leido ni procesado.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena RF sobre todos los coeficientes Wavelet Scattering."
    )
    parser.add_argument(
        "--action",
        choices=["check", "train"],
        default="check",
        help="check valida datos; train ejecuta CV y, salvo quick, validation.",
    )
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default="paper_q8_q1_t500_full",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Prueba tres RF mediante OOF sin evaluar VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WAVELET SCATTERING COMPLETO + RANDOM FOREST")
    print("=" * 78)
    print(f"Preset: {args.preset}")
    print(f"Accion: {args.action}")
    print("TEST no sera leido ni procesado.")
    data, extraction_configuration = load_wavelet_data(args.preset)
    print_check(data, args.quick)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(data, extraction_configuration, args.preset, args.quick)


if __name__ == "__main__":
    main()
