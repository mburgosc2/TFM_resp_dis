"""Experimento WST completo + SMOTE + Random Forest para dry/wet.

La seleccion por informacion mutua previa conservo los 644 caminos WST. Este
script mantiene esa representacion y estudia solamente el efecto de SMOTE.
StandardScaler y SMOTE se ajustan dentro del bloque de entrenamiento de cada
fold. Las probabilidades de evento se promedian por ``original_uuid`` antes de
calcular metricas. VALIDATION se usa una vez al final y TEST no se lee.

Se incluyen dos controles: el RF ponderado actual y el mismo RF escalado sin
pesos ni SMOTE. Los candidatos SMOTE tampoco usan pesos de clase, para no
corregir el desbalance dos veces.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_rf as baseline


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results_stage2_dry_wet_wavelet_scattering_smote_rf"
MODELS_ROOT = SCRIPT_DIR / "models_stage2_dry_wet_wavelet_scattering_smote_rf"
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wavelet_scattering_smote_rf"
)
RANDOM_STATE = 42
PRESETS = baseline.PRESETS
FULL_RATIOS = (0.50, 0.75, 1.00)
FULL_NEIGHBORS = (3, 5)
QUICK_RATIOS = (0.50, 1.00)
QUICK_NEIGHBORS = (3,)
FIXED_RF = baseline.RFCandidate(300, 10, 5, 0.1)


@dataclass(frozen=True)
class Candidate:
    kind: str
    ratio: float | None = None
    neighbors: int | None = None

    @property
    def key(self) -> str:
        if self.kind == "reference":
            return "reference_weighted__no_smote"
        if self.kind == "control":
            return "control_unweighted__no_smote"
        ratio_name = str(self.ratio).replace(".", "p")
        return f"smote__ratio{ratio_name}__k{self.neighbors}"

    @property
    def uses_smote(self) -> bool:
        return self.kind == "smote"


@dataclass
class Evaluation:
    candidate: Candidate
    threshold: float
    metrics: dict[str, float | int]
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame


def candidate_grid(quick: bool) -> list[Candidate]:
    ratios = QUICK_RATIOS if quick else FULL_RATIOS
    neighbors = QUICK_NEIGHBORS if quick else FULL_NEIGHBORS
    return [
        Candidate("reference"),
        Candidate("control"),
        *[Candidate("smote", ratio, k) for ratio in ratios for k in neighbors],
    ]


def counts(y_values: np.ndarray) -> tuple[int, int]:
    values = np.bincount(np.asarray(y_values, dtype=int), minlength=2)
    if len(values) != 2 or np.any(values <= 0):
        raise ValueError("El subconjunto debe contener dry=0 y wet=1.")
    return int(values[0]), int(values[1])


def fit_scaler(
    x_values: np.ndarray,
    metadata: pd.DataFrame,
) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(
        x_values,
        sample_weight=common.compute_recording_equal_weights(metadata),
    )
    return scaler


def apply_smote(
    x_values: np.ndarray,
    y_values: np.ndarray,
    candidate: Candidate,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    dry_before, wet_before = counts(y_values)
    sampler = SMOTE(
        sampling_strategy=float(candidate.ratio),
        k_neighbors=int(candidate.neighbors),
        random_state=RANDOM_STATE,
    )
    x_resampled, y_resampled = sampler.fit_resample(x_values, y_values)
    dry_after, wet_after = counts(y_resampled)
    details = {
        "candidate_key": candidate.key,
        "sampling_strategy": candidate.ratio,
        "k_neighbors": candidate.neighbors,
        "dry_events_before": dry_before,
        "wet_events_before": wet_before,
        "dry_events_after": dry_after,
        "wet_events_after": wet_after,
        "synthetic_wet_events_added": wet_after - wet_before,
        "wet_to_dry_ratio_after": wet_after / dry_after,
    }
    return (
        np.asarray(x_resampled, dtype=np.float32),
        np.asarray(y_resampled, dtype=np.int64),
        details,
    )


def evaluate_oof(
    data: common.DataView,
    candidate_list: list[Candidate],
) -> tuple[list[Evaluation], pd.DataFrame]:
    oof = {
        candidate.key: np.full(len(data.x_train), np.nan, dtype=float)
        for candidate in candidate_list
    }
    resampling_rows: list[dict[str, object]] = []

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = data.train_metadata["fold"].to_numpy() == fold
        training_mask = ~validation_mask
        metadata_fit = data.train_metadata.loc[training_mask].reset_index(
            drop=True
        )
        x_fit_raw = data.x_train[training_mask]
        x_validation_raw = data.x_train[validation_mask]
        y_fit = metadata_fit["stage2_target"].to_numpy(dtype=int)

        scaler = fit_scaler(x_fit_raw, metadata_fit)
        x_fit_scaled = scaler.transform(x_fit_raw).astype(np.float32)
        x_validation_scaled = scaler.transform(x_validation_raw).astype(
            np.float32
        )
        dry_before, wet_before = counts(y_fit)
        print(f"  Fold {fold}: scaler/SMOTE solo sobre TRAIN interno...")

        for index, candidate in enumerate(candidate_list, start=1):
            classifier = baseline.build_rf(FIXED_RF)
            if candidate.kind == "reference":
                classifier.fit(
                    x_fit_raw,
                    y_fit,
                    sample_weight=common.compute_training_weights(metadata_fit),
                )
                scores = baseline.rf_scores(classifier, x_validation_raw)
                details = {
                    "candidate_key": candidate.key,
                    "sampling_strategy": np.nan,
                    "k_neighbors": np.nan,
                    "dry_events_before": dry_before,
                    "wet_events_before": wet_before,
                    "dry_events_after": dry_before,
                    "wet_events_after": wet_before,
                    "synthetic_wet_events_added": 0,
                    "wet_to_dry_ratio_after": wet_before / dry_before,
                }
            elif candidate.kind == "control":
                classifier.fit(x_fit_scaled, y_fit)
                scores = baseline.rf_scores(classifier, x_validation_scaled)
                details = {
                    "candidate_key": candidate.key,
                    "sampling_strategy": np.nan,
                    "k_neighbors": np.nan,
                    "dry_events_before": dry_before,
                    "wet_events_before": wet_before,
                    "dry_events_after": dry_before,
                    "wet_events_after": wet_before,
                    "synthetic_wet_events_added": 0,
                    "wet_to_dry_ratio_after": wet_before / dry_before,
                }
            else:
                x_resampled, y_resampled, details = apply_smote(
                    x_fit_scaled,
                    y_fit,
                    candidate,
                )
                classifier.fit(x_resampled, y_resampled)
                scores = baseline.rf_scores(classifier, x_validation_scaled)

            oof[candidate.key][validation_mask] = scores
            details["fold"] = fold
            resampling_rows.append(details)
            print(f"    {index:02d}/{len(candidate_list):02d}: {candidate.key}")

    evaluations = []
    for candidate in candidate_list:
        event_scores = oof[candidate.key]
        if not np.isfinite(event_scores).all():
            raise RuntimeError(f"OOF incompleto para {candidate.key}.")
        recordings = common.aggregate_scores_by_recording(
            data.train_metadata,
            event_scores,
        )
        y_true = recordings["y_true"].to_numpy(dtype=int)
        scores = recordings["score"].to_numpy(dtype=float)
        threshold = common.tune_threshold(y_true, scores, 0.5)
        metrics = common.binary_metrics(y_true, scores, threshold)
        recordings["y_pred"] = (scores >= threshold).astype(int)
        recordings["candidate_key"] = candidate.key

        fold_rows = []
        for fold, fold_data in recordings.groupby("fold", sort=True):
            fold_rows.append(
                {
                    "fold": int(fold),
                    "recording_count": len(fold_data),
                    "candidate_key": candidate.key,
                    "threshold": threshold,
                    **common.binary_metrics(
                        fold_data["y_true"].to_numpy(dtype=int),
                        fold_data["score"].to_numpy(dtype=float),
                        threshold,
                    ),
                }
            )
        evaluations.append(
            Evaluation(
                candidate,
                threshold,
                metrics,
                recordings,
                pd.DataFrame(fold_rows),
            )
        )
    return evaluations, pd.DataFrame(resampling_rows)


def get_control(evaluations: list[Evaluation], kind: str) -> Evaluation:
    matches = [item for item in evaluations if item.candidate.kind == kind]
    if len(matches) != 1:
        raise RuntimeError(f"Numero inesperado de controles {kind}.")
    return matches[0]


def select_smote(evaluations: list[Evaluation]) -> Evaluation:
    candidates_smote = [item for item in evaluations if item.candidate.uses_smote]
    return max(
        candidates_smote,
        key=lambda item: (
            float(item.metrics["macro_f1"]),
            float(item.metrics["balanced_accuracy"]),
            float(item.metrics["wet_recall"]),
            float(item.metrics["roc_auc"]),
            -float(item.candidate.ratio),
            -float(item.candidate.neighbors),
        ),
    )


def save_oof(
    evaluations: list[Evaluation],
    selected: Evaluation,
    resampling: pd.DataFrame,
    result_dir: Path,
    elapsed_seconds: float,
) -> None:
    rows = []
    for item in evaluations:
        rows.append(
            {
                "candidate_key": item.candidate.key,
                "candidate_kind": item.candidate.kind,
                "sampling_strategy": item.candidate.ratio,
                "k_neighbors": item.candidate.neighbors,
                "selected_smote_candidate": (
                    item.candidate.key == selected.candidate.key
                ),
                "fixed_rf": FIXED_RF.key,
                "threshold_oof": item.threshold,
                "elapsed_seconds_total_search": elapsed_seconds,
                **item.metrics,
            }
        )
    pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy", "wet_recall", "roc_auc"],
        ascending=False,
    ).to_csv(
        result_dir / "candidate_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.predictions.to_csv(
        result_dir / "selected_smote_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.fold_metrics.to_csv(
        result_dir / "selected_smote_cv_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    resampling.to_csv(
        result_dir / "resampling_counts_by_fold.csv",
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

    grid = candidate_grid(quick)
    print("\n" + "=" * 78)
    print("CV WST COMPLETO -> STANDARD SCALER -> SMOTE -> RF FIJO")
    print("=" * 78)
    start = time.perf_counter()
    evaluations, resampling = evaluate_oof(data, grid)
    elapsed = time.perf_counter() - start

    for item in evaluations:
        print(
            f"{item.candidate.key} | macro-F1={item.metrics['macro_f1']:.4f} | "
            f"bal-acc={item.metrics['balanced_accuracy']:.4f} | "
            f"wet-recall={item.metrics['wet_recall']:.4f} | "
            f"AUC={item.metrics['roc_auc']:.4f}"
        )
    selected = select_smote(evaluations)
    reference = get_control(evaluations, "reference")
    control = get_control(evaluations, "control")
    save_oof(evaluations, selected, resampling, result_dir, elapsed)

    pd.DataFrame(
        [
            {"role": "weighted_reference", "key": reference.candidate.key,
             **reference.metrics},
            {"role": "unweighted_control", "key": control.candidate.key,
             **control.metrics},
            {"role": "selected_smote", "key": selected.candidate.key,
             **selected.metrics},
        ]
    ).to_csv(
        result_dir / "oof_experiment_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof_summary = {
        "dataset": "train_oof",
        "experiment": "wavelet_scattering_smote_rf_event",
        "candidate_key": selected.candidate.key,
        "threshold": selected.threshold,
        "feature_count": data.x_train.shape[1],
        "sampling_strategy": selected.candidate.ratio,
        "k_neighbors": selected.candidate.neighbors,
        "model_size_kb_joblib": np.nan,
        **selected.metrics,
    }
    if quick:
        pd.DataFrame([oof_summary]).to_csv(
            result_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig"
        )
        print("\nPrueba rapida WST + SMOTE + RF completada.")
        print(f"Mejor SMOTE OOF: {selected.candidate.key}")
        print(f"Macro-F1 OOF: {selected.metrics['macro_f1']:.4f}")
        print(
            "Diferencia frente al RF ponderado: "
            f"{selected.metrics['macro_f1'] - reference.metrics['macro_f1']:+.4f}"
        )
        print("VALIDATION no se ha evaluado.")
        print(f"Resultados: {result_dir}")
        return

    print("\nAjustando scaler, SMOTE y RF finales con todo TRAIN...")
    y_train = data.train_metadata["stage2_target"].to_numpy(dtype=int)
    scaler = fit_scaler(data.x_train, data.train_metadata)
    x_train_scaled = scaler.transform(data.x_train).astype(np.float32)
    x_resampled, y_resampled, final_counts = apply_smote(
        x_train_scaled,
        y_train,
        selected.candidate,
    )
    classifier = baseline.build_rf(FIXED_RF)
    classifier.fit(x_resampled, y_resampled)
    validation_scaled = scaler.transform(data.x_validation).astype(np.float32)
    event_scores = baseline.rf_scores(classifier, validation_scaled)
    validation = common.aggregate_scores_by_recording(
        data.validation_metadata,
        event_scores,
    )
    validation_scores = validation["score"].to_numpy(dtype=float)
    validation["y_pred"] = (validation_scores >= selected.threshold).astype(int)
    validation["candidate_key"] = selected.candidate.key
    validation_metrics = common.binary_metrics(
        validation["y_true"].to_numpy(dtype=int),
        validation_scores,
        selected.threshold,
    )

    complexity = baseline.forest_complexity(classifier)
    package = {
        "scaler": scaler,
        "classifier": classifier,
        "feature_names": data.feature_names,
        "feature_count": data.x_train.shape[1],
        "preset": preset,
        "experiment": "wavelet_scattering_smote_rf_event",
        "candidate_key": selected.candidate.key,
        "sampling_strategy": selected.candidate.ratio,
        "k_neighbors": selected.candidate.neighbors,
        "threshold": selected.threshold,
        "label_mapping": common.LABEL_TO_NAME,
        "recording_aggregation": "mean_event_probability",
        "feature_selection": "mi_selected_100_percent",
        "standardization_used": True,
        "smote_used_during_training_only": True,
        "classifier_weighting": "none_after_smote",
        "extraction_configuration": extraction_configuration.iloc[0].to_dict(),
        "random_state": RANDOM_STATE,
        **complexity,
    }
    model_path = model_dir / "wavelet_scattering_smote_rf_event_model.joblib"
    joblib.dump(package, model_path, compress=3)
    model_size_kb = model_path.stat().st_size / 1024.0

    oof_summary.update({"model_size_kb_joblib": model_size_kb, **complexity})
    validation_summary = {
        "dataset": "validation",
        "experiment": "wavelet_scattering_smote_rf_event",
        "candidate_key": selected.candidate.key,
        "threshold": selected.threshold,
        "feature_count": data.x_train.shape[1],
        "sampling_strategy": selected.candidate.ratio,
        "k_neighbors": selected.candidate.neighbors,
        "model_size_kb_joblib": model_size_kb,
        **complexity,
        **validation_metrics,
    }
    pd.DataFrame([oof_summary, validation_summary]).to_csv(
        result_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig"
    )
    validation.to_csv(
        result_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([final_counts]).to_csv(
        result_dir / "final_resampling_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [{
            "feature_count": data.x_train.shape[1],
            "feature_selection": "mi_selected_100_percent",
            "selected_smote_candidate": selected.candidate.key,
            "smote_scope": "training_partition_inside_each_fold",
            "scaler_scope": "training_partition_inside_each_fold",
            "scaler_weighting": "1_over_events_per_recording",
            "classifier_weighting_after_smote": "none",
            "validation_used_for_selection": False,
            "test_processed": False,
            **complexity,
        }]
    ).to_csv(
        result_dir / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    graph_path = graph_dir / "validation_wavelet_scattering_smote_rf_event.png"
    common.create_validation_graph(
        validation,
        selected.threshold,
        "wavelet_scattering_smote_rf_event",
        selected.candidate,
        graph_path,
    )
    print("\n" + "=" * 78)
    print("RESULTADO WST COMPLETO + SMOTE + RANDOM FOREST")
    print("=" * 78)
    print(f"Mejor SMOTE OOF: {selected.candidate.key}")
    print(f"Umbral OOF: {selected.threshold:.6f}")
    print(f"Macro-F1 OOF: {selected.metrics['macro_f1']:.4f}")
    print(
        "Diferencia OOF frente al RF ponderado: "
        f"{selected.metrics['macro_f1'] - reference.metrics['macro_f1']:+.4f}"
    )
    print(f"Macro-F1 validation: {validation_metrics['macro_f1']:.4f}")
    print(
        "Recalls validation dry/wet: "
        f"{validation_metrics['dry_recall']:.4f} / "
        f"{validation_metrics['wet_recall']:.4f}"
    )
    print(
        "Eventos finales dry/wet: "
        f"{final_counts['dry_events_after']} / {final_counts['wet_events_after']}"
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
    dry_events, wet_events = counts(
        data.train_metadata["stage2_target"].to_numpy(dtype=int)
    )
    print("=" * 78)
    print("CHECK - WST COMPLETO + SMOTE + RF FIJO")
    print("=" * 78)
    print(f"X train: {data.x_train.shape} {data.x_train.dtype}")
    print(f"X validation: {data.x_validation.shape} {data.x_validation.dtype}")
    print(f"Eventos train dry/wet: {dry_events} / {wet_events}")
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
    print(f"RF fijo: {FIXED_RF.key}")
    print("Candidatos:")
    for candidate in candidate_grid(quick):
        print(f"  - {candidate.key}")
    print("Se conservan los 644 caminos WST seleccionados por MI.")
    print("Scaler y SMOTE se ajustan dentro de cada fold.")
    print("Los candidatos SMOTE no usan pesos adicionales de clase.")
    print("VALIDATION no selecciona configuracion. TEST no se procesa.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WST completo con SMOTE y Random Forest fijo."
    )
    parser.add_argument("--action", choices=["check", "train"], default="check")
    parser.add_argument(
        "--preset", choices=PRESETS, default="paper_q8_q1_t500_full"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Prueba ratios 0.5/1.0 con k=3 sin evaluar VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST COMPLETO + SMOTE + RANDOM FOREST")
    print("=" * 78)
    print(f"Preset: {args.preset}")
    print(f"Accion: {args.action}")
    print("TEST no sera leido ni procesado.")
    data, extraction_configuration = baseline.load_wavelet_data(args.preset)
    print_check(data, args.quick)
    if args.action == "train":
        train(data, extraction_configuration, args.preset, args.quick)
    else:
        print("Comprobacion completada. No se entreno ningun modelo.")


if __name__ == "__main__":
    main()
