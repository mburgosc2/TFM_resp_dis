"""Ablacion del orden log/pooling temporal en WST Stage 2 + LR.

Consume los tensores WST temporales T=8000 ya extraidos con forma
``(evento, path, tiempo)`` y compara tres representaciones de igual tamano:

1. ``raw_mean``: ``mean(S)`` (baseline actual).
2. ``log_after_mean``: ``log(abs(mean(S)) + eps)``.
3. ``log_before_mean``: ``mean(log(abs(S) + eps))``.

Se mantienen los 644 paths, incluido S0. El valor absoluto es necesario para
que las variantes log sean finitas porque S0 puede tener signo; S1 y S2 son
no negativos. Esta decision se registra expresamente para no confundir esta
prueba con una ablacion que elimine S0.

Despues se aplica exactamente el pooling recording-level mean/std/max entre
eventos. Para aislar el momento de la compresion logaritmica se congela la
configuracion ganadora previa: StandardScaler, PCA128 sin whitening y LR L2
con C=0.001, gold_multiplier=1 y wet_cost=1. Scaler y PCA se vuelven a ajustar
por separado dentro de cada fold para cada representacion.

La seleccion depende solo de OOF de TRAIN. Las tres variantes preespecificadas
se reportan en VALIDATION como analisis de sensibilidad. TEST no se carga.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_linear as linear_base
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording
import train_stage2_dry_wet_wst_recording_consensus_weighted_lr as consensus_base
import train_stage2_dry_wet_wst_recording_lr_two_phase_search as lr_base
import train_stage2_dry_wet_wst_temporal_statistics_lr as temporal_base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRESET = "paper_q8_q1_t500_full"
RESULTS_ROOT = SCRIPT_DIR / "results_stage2_dry_wet_wst_log_before_mean_lr"
MODELS_ROOT = SCRIPT_DIR / "models_stage2_dry_wet_wst_log_before_mean_lr"
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wst_log_before_mean_lr"
)

LOG_EPSILON = 1e-8
FROZEN_C = 1e-3
FROZEN_PCA_COMPONENTS = 128
FROZEN_PCA_WHITEN = False
FROZEN_GOLD_MULTIPLIER = 1.0
FROZEN_WET_COST = 1.0

BETWEEN_EVENT_POOLING = next(
    item for item in recording.POOLING_SPECS if item.key == "mean_std_max"
)


@dataclass(frozen=True)
class LogRepresentation:
    key: str
    formula: str


REPRESENTATIONS = (
    LogRepresentation("raw_mean", "mean(S)"),
    LogRepresentation(
        "log_after_mean",
        "log(abs(mean(S)) + epsilon)",
    ),
    LogRepresentation(
        "log_before_mean",
        "mean(log(abs(S) + epsilon))",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara raw_mean, log_after_mean y log_before_mean en WST + LR."
        )
    )
    parser.add_argument("--action", choices=("check", "train"), default="check")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser


def frozen_candidate() -> lr_base.SearchCandidate:
    return lr_base.SearchCandidate(
        spec=common.CandidateSpec(
            "logistic_regression",
            FROZEN_C,
            FROZEN_PCA_COMPONENTS,
        ),
        pca_whiten=FROZEN_PCA_WHITEN,
        gold_multiplier=FROZEN_GOLD_MULTIPLIER,
        wet_cost=FROZEN_WET_COST,
    )


def transform_temporal_tensor(
    tensor: np.ndarray,
    representation: LogRepresentation,
) -> np.ndarray:
    if representation.key == "raw_mean":
        result = np.mean(tensor, axis=2, dtype=np.float32)
    elif representation.key == "log_after_mean":
        temporal_mean = np.mean(tensor, axis=2, dtype=np.float32)
        result = np.log(
            np.abs(temporal_mean).astype(np.float32) + LOG_EPSILON
        )
    elif representation.key == "log_before_mean":
        logged = np.log(
            np.abs(np.asarray(tensor, dtype=np.float32)) + LOG_EPSILON
        )
        result = np.mean(logged, axis=2, dtype=np.float32)
    else:
        raise ValueError(
            f"Representacion log desconocida: {representation.key}."
        )
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError(
            f"{representation.key} produjo valores NaN o infinitos."
        )
    return result


def build_recording_view(
    raw: temporal_base.TemporalData,
    representation: LogRepresentation,
) -> common.DataView:
    names = [
        f"{representation.key}__{path_name}"
        for path_name in raw.path_names
    ]
    event_data = common.DataView(
        experiment=representation.key,
        x_train=transform_temporal_tensor(raw.x_train, representation),
        train_metadata=raw.metadata_train.copy(),
        x_validation=transform_temporal_tensor(
            raw.x_validation, representation
        ),
        validation_metadata=raw.metadata_validation.copy(),
        feature_names=names,
    )
    data = recording.build_recording_view(
        event_data,
        BETWEEN_EVENT_POOLING,
    )
    data.experiment = representation.key
    return consensus_base.attach_vote_metadata(data)


def log_input_diagnostics(
    raw: temporal_base.TemporalData,
) -> pd.DataFrame:
    orders = raw.path_layout["scattering_order"].to_numpy(dtype=int)
    rows = []
    for order in sorted(np.unique(orders)):
        values = np.asarray(raw.x_train[:, orders == order, :])
        absolute = np.abs(values)
        rows.append(
            {
                "scattering_order": int(order),
                "value_count": int(values.size),
                "negative_count": int(np.sum(values < 0)),
                "negative_fraction": float(np.mean(values < 0)),
                "zero_count": int(np.sum(values == 0)),
                "absolute_minimum": float(absolute.min()),
                "absolute_median": float(np.median(absolute)),
                "absolute_maximum": float(absolute.max()),
                "fraction_abs_below_epsilon": float(
                    np.mean(absolute < LOG_EPSILON)
                ),
                "epsilon": LOG_EPSILON,
            }
        )
    return pd.DataFrame(rows)


def check(raw: temporal_base.TemporalData, preset: str) -> None:
    config = raw.extraction_configuration.iloc[0]
    if int(config["invariance_samples"]) != 8000:
        raise ValueError(
            "Esta primera ablacion log mantiene T=8000; recibido "
            f"T={int(config['invariance_samples'])}."
        )
    if raw.x_train.shape[2] != 5:
        raise ValueError(
            "El preset T=8000 debe contener cinco posiciones temporales."
        )
    equivalence = temporal_base.baseline_equivalence(raw, preset)
    if not equivalence["baseline_available"]:
        raise FileNotFoundError(
            "No se encontro el baseline raw necesario para verificar la "
            "primera representacion."
        )
    if not equivalence["event_allclose"]:
        raise RuntimeError("raw_mean no reproduce el baseline existente.")

    diagnostics = log_input_diagnostics(raw)
    train_recordings = raw.metadata_train.drop_duplicates("original_uuid")
    validation_recordings = raw.metadata_validation.drop_duplicates(
        "original_uuid"
    )
    print("=" * 78)
    print("CHECK - WST LOG BEFORE/AFTER MEAN + LR")
    print("=" * 78)
    print(f"Preset: {preset}")
    print(
        f"J={int(config['J'])} | Q={config['Q']} | "
        f"T={int(config['invariance_samples'])} | "
        f"tensor={raw.x_train.shape[1]}x{raw.x_train.shape[2]} por evento"
    )
    print(
        f"Grabaciones TRAIN/VALIDATION: {len(train_recordings)} / "
        f"{len(validation_recordings)}"
    )
    print(f"Epsilon logaritmico: {LOG_EPSILON:g}")
    print("Diagnostico por orden:")
    for row in diagnostics.to_dict("records"):
        print(
            f"  S{row['scattering_order']}: "
            f"negativos={row['negative_count']} "
            f"({row['negative_fraction']:.4%}); "
            f"abs<eps={row['fraction_abs_below_epsilon']:.2%}"
        )
    for representation in REPRESENTATIONS:
        print(
            f"  {representation.key}: {representation.formula} -> "
            "644/evento -> 1932/grabacion"
        )
    print(
        "Equivalencia raw_mean con baseline: "
        f"allclose={equivalence['event_allclose']} | "
        f"max_diff={equivalence['event_max_absolute_difference']}"
    )
    print(
        "Cada representacion reajusta StandardScaler y PCA128 dentro de cada "
        "fold."
    )
    print("TEST no se carga ni se procesa.")


def prefixed_metrics(
    metrics: dict[str, float | int],
    prefix: str,
) -> dict[str, float | int]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def train(raw: temporal_base.TemporalData, preset: str) -> None:
    result_dir = RESULTS_ROOT / preset / "full"
    model_dir = MODELS_ROOT / preset / "full"
    graph_dir = GRAPHS_ROOT / preset / "full"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    candidate = frozen_candidate()
    comparison_rows: list[dict[str, object]] = []
    evaluations: dict[str, lr_base.SearchResult] = {}
    oof_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    pca_frames: list[pd.DataFrame] = []

    diagnostics = log_input_diagnostics(raw)
    diagnostics.to_csv(
        result_dir / "log_input_diagnostics_by_order.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 78)
    print("ABLACION - ORDEN DE LOG Y MEDIA TEMPORAL WST")
    print("=" * 78)
    for index, representation in enumerate(REPRESENTATIONS, start=1):
        print(
            f"\n[{index}/{len(REPRESENTATIONS)}] "
            f"{representation.key}: {representation.formula}"
        )
        data = build_recording_view(raw, representation)
        results, pca_variance, weight_diagnostics = lr_base.evaluate_candidates(
            data,
            [candidate],
        )
        evaluation = results[0]
        evaluations[representation.key] = evaluation

        pipeline = lr_base.build_final_pipeline(data, candidate)
        validation = lr_base.validation_predictions(data, pipeline, evaluation)
        y_validation = validation["y_true"].to_numpy(dtype=int)
        validation_scores = validation["score"].to_numpy(dtype=float)
        validation_metrics = common.binary_metrics(
            y_validation,
            validation_scores,
            evaluation.threshold,
        )
        validation_native = common.binary_metrics(
            y_validation,
            validation_scores,
            candidate.spec.default_threshold,
        )

        oof = evaluation.predictions.copy()
        oof["log_representation"] = representation.key
        validation["log_representation"] = representation.key
        fold_metrics = evaluation.fold_metrics.copy()
        fold_metrics["log_representation"] = representation.key
        pca_variance = pca_variance.copy()
        pca_variance["log_representation"] = representation.key
        oof_frames.append(oof)
        validation_frames.append(validation)
        fold_frames.append(fold_metrics)
        pca_frames.append(pca_variance)
        weight_diagnostics.to_csv(
            result_dir / f"weight_diagnostics_{representation.key}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        package = {
            "pipeline": pipeline,
            "experiment": "stage2_wst_log_before_mean_lr",
            "preset": preset,
            "representation": asdict(representation),
            "log_epsilon": LOG_EPSILON,
            "s0_policy": "retain_and_use_absolute_value_only_in_log_variants",
            "between_event_pooling": {
                "key": BETWEEN_EVENT_POOLING.key,
                "statistics": BETWEEN_EVENT_POOLING.statistics,
            },
            "candidate_key": candidate.key,
            "C": FROZEN_C,
            "penalty": lr_base.PENALTY,
            "pca_components": FROZEN_PCA_COMPONENTS,
            "pca_whiten": FROZEN_PCA_WHITEN,
            "gold_multiplier": FROZEN_GOLD_MULTIPLIER,
            "wet_cost": FROZEN_WET_COST,
            "threshold": evaluation.threshold,
            "minimum_wet_recall_oof": lr_base.MIN_WET_RECALL_OOF,
            "recording_feature_names": data.feature_names,
            "label_mapping": common.LABEL_TO_NAME,
            "selection_data": "TRAIN OOF only",
            "validation_role": "pre-specified ablation evaluation",
            "test_processed": False,
            "extraction_configuration": (
                raw.extraction_configuration.iloc[0].to_dict()
            ),
            "random_state": lr_base.RANDOM_STATE,
        }
        model_path = model_dir / f"{representation.key}_lr.joblib"
        joblib.dump(package, model_path, compress=3)
        model_size_kb = model_path.stat().st_size / 1024.0

        common.create_validation_graph(
            validation,
            evaluation.threshold,
            f"wst_{representation.key}",
            candidate.spec,
            graph_dir / f"validation_{representation.key}.png",
        )

        comparison_rows.append(
            {
                "log_representation": representation.key,
                "formula": representation.formula,
                "epsilon": LOG_EPSILON,
                "s0_policy": (
                    "retain_signed_raw; absolute_value_in_log_variants"
                ),
                "T_samples": int(
                    raw.extraction_configuration.iloc[0]["invariance_samples"]
                ),
                "time_position_count": raw.x_train.shape[2],
                "event_feature_count": len(raw.path_names),
                "recording_feature_count": data.x_train.shape[1],
                "pca_components": FROZEN_PCA_COMPONENTS,
                "pca_whiten": FROZEN_PCA_WHITEN,
                "mean_pca_explained_variance_ratio": float(
                    pca_variance[
                        "cumulative_explained_variance_ratio"
                    ].mean()
                ),
                "C": FROZEN_C,
                "threshold_oof": evaluation.threshold,
                "model_size_kb": model_size_kb,
                "learned_float_count": linear_base.learned_float_count(
                    pipeline
                ),
                **prefixed_metrics(evaluation.metrics_tuned, "oof__"),
                **prefixed_metrics(evaluation.metrics_native, "oof_0p5__"),
                **prefixed_metrics(validation_metrics, "validation__"),
                **prefixed_metrics(validation_native, "validation_0p5__"),
            }
        )
        print(
            f"  OOF: macro-F1={evaluation.metrics_tuned['macro_f1']:.4f} | "
            f"bal-acc={evaluation.metrics_tuned['balanced_accuracy']:.4f} | "
            f"wet-recall={evaluation.metrics_tuned['wet_recall']:.4f} | "
            f"AUC={evaluation.metrics_tuned['roc_auc']:.4f}"
        )
        print(
            f"  VALIDATION: macro-F1={validation_metrics['macro_f1']:.4f} | "
            f"bal-acc={validation_metrics['balanced_accuracy']:.4f} | "
            f"dry/wet recall={validation_metrics['dry_recall']:.4f}/"
            f"{validation_metrics['wet_recall']:.4f} | "
            f"AUC={validation_metrics['roc_auc']:.4f}"
        )

    winner_key = max(
        evaluations,
        key=lambda key: lr_base.selection_key(evaluations[key]),
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison["selected_by_oof"] = comparison["log_representation"].eq(
        winner_key
    )
    comparison = comparison.sort_values(
        ["selected_by_oof", "oof__macro_f1", "oof__balanced_accuracy"],
        ascending=False,
    )
    comparison.to_csv(
        result_dir / "log_order_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_oof = pd.concat(oof_frames, ignore_index=True)
    all_validation = pd.concat(validation_frames, ignore_index=True)
    all_oof.to_csv(
        result_dir / "oof_predictions_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_validation.to_csv(
        result_dir / "validation_predictions_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(fold_frames, ignore_index=True).to_csv(
        result_dir / "cv_fold_metrics_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(pca_frames, ignore_index=True).to_csv(
        result_dir / "pca_explained_variance_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_oof[all_oof["log_representation"].eq(winner_key)].to_csv(
        result_dir / "best_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_validation[
        all_validation["log_representation"].eq(winner_key)
    ].to_csv(
        result_dir / "best_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    experiment_configuration = {
        "experiment": "stage2_wst_log_before_mean_lr",
        "preset": preset,
        "representations": [asdict(item) for item in REPRESENTATIONS],
        "log_epsilon": LOG_EPSILON,
        "s0_policy": "retain_signed_raw; absolute_value_in_log_variants",
        "between_event_pooling": list(BETWEEN_EVENT_POOLING.statistics),
        "classifier_configuration_frozen": True,
        "C": FROZEN_C,
        "pca_components": FROZEN_PCA_COMPONENTS,
        "pca_whiten": FROZEN_PCA_WHITEN,
        "gold_multiplier": FROZEN_GOLD_MULTIPLIER,
        "wet_cost": FROZEN_WET_COST,
        "selection": "TRAIN OOF only",
        "selected_representation": winner_key,
        "validation_used_for_selection": False,
        "test_processed": False,
    }
    pd.DataFrame(
        {
            "parameter": key,
            "value": json.dumps(value)
            if isinstance(value, (dict, list, tuple))
            else value,
        }
        for key, value in experiment_configuration.items()
    ).to_csv(
        result_dir / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    winner = comparison.iloc[0]
    print("\n" + "=" * 78)
    print("RESULTADO - ORDEN DE LOG Y MEDIA TEMPORAL WST")
    print("=" * 78)
    print(f"Ganador segun OOF: {winner_key}")
    print(f"Macro-F1 OOF: {winner['oof__macro_f1']:.4f}")
    print(f"Macro-F1 validation: {winner['validation__macro_f1']:.4f}")
    print(
        "Recalls validation dry/wet: "
        f"{winner['validation__dry_recall']:.4f} / "
        f"{winner['validation__wet_recall']:.4f}"
    )
    print(f"Comparacion: {result_dir / 'log_order_comparison.csv'}")
    print(f"Resultados: {result_dir}")
    print("TEST no se ha cargado ni evaluado.")


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST LOG BEFORE/AFTER MEAN + LR")
    print("=" * 78)
    print(f"Preset: {args.preset} | accion: {args.action}")
    print("TEST no sera leido ni procesado.")
    raw = temporal_base.load_temporal_data(args.preset)
    check(raw, args.preset)
    if args.action == "train":
        train(raw, args.preset)
    else:
        print("\nComprobacion completada. No se entreno ningun modelo.")


if __name__ == "__main__":
    main()
