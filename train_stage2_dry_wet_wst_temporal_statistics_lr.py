"""Ablacion de estadisticas temporales WST para Stage 2 con LR.

Reutiliza los tensores WST temporales guardados con forma
``(evento, path, tiempo)``. No vuelve a calcular scattering. Compara:

* ``temporal_mean``: media temporal por path, igual al baseline existente.
* ``temporal_mean_std_max``: media, desviacion y maximo temporales por path.

En ambos casos se conserva el pooling recording-level usado por el mejor
baseline WST + LR: mean, std y max entre los eventos de cada grabacion. Para
aislar el efecto de las estadisticas temporales se congela la configuracion
del clasificador: StandardScaler, PCA128 sin whitening, LR L2 con C=0.001,
gold_multiplier=1 y wet_cost=1.

Los cinco folds de TRAIN producen predicciones OOF y el umbral se obtiene
exclusivamente de ellas. Las dos representaciones preespecificadas se
reportan tambien en VALIDATION, pero la marca de ganador depende solo de OOF.
TEST no se carga ni se procesa.
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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRESET = "paper_q8_q1_t500_full"
FEATURE_ROOT = (
    SCRIPT_DIR
    / "features_extracted_stage2_dry_wet_wavelet_scattering_temporal"
)
POOLED_BASELINE_ROOT = (
    SCRIPT_DIR / "features_extracted_stage2_dry_wet_wavelet_scattering"
)
RESULTS_ROOT = (
    SCRIPT_DIR
    / "results_stage2_dry_wet_wst_temporal_statistics_lr"
)
MODELS_ROOT = (
    SCRIPT_DIR
    / "models_stage2_dry_wet_wst_temporal_statistics_lr"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wst_temporal_statistics_lr"
)

FROZEN_C = 1e-3
FROZEN_PCA_COMPONENTS = 128
FROZEN_PCA_WHITEN = False
FROZEN_GOLD_MULTIPLIER = 1.0
FROZEN_WET_COST = 1.0

BETWEEN_EVENT_POOLING = next(
    item for item in recording.POOLING_SPECS if item.key == "mean_std_max"
)


@dataclass(frozen=True)
class TemporalRepresentation:
    key: str
    statistics: tuple[str, ...]


@dataclass
class TemporalData:
    x_train: np.ndarray
    metadata_train: pd.DataFrame
    x_validation: np.ndarray
    metadata_validation: pd.DataFrame
    path_names: list[str]
    path_layout: pd.DataFrame
    extraction_configuration: pd.DataFrame


REPRESENTATIONS = (
    TemporalRepresentation("temporal_mean", ("mean",)),
    TemporalRepresentation(
        "temporal_mean_std_max",
        ("mean", "std", "max"),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara mean frente a mean+std+max temporal de WST con LR "
            "recording-level."
        )
    )
    parser.add_argument("--action", choices=("check", "train"), default="check")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser


def required_files(feature_dir: Path) -> tuple[str, ...]:
    return (
        "X_events_train.npy",
        "X_events_validation.npy",
        "y_events_train.npy",
        "y_events_validation.npy",
        "folds_events_train.npy",
        "folds_events_validation.npy",
        "metadata_events_features_train.csv",
        "metadata_events_features_validation.csv",
        "wavelet_scattering_temporal_path_layout.csv",
        "wavelet_scattering_temporal_configuration.csv",
    )


def load_temporal_data(preset: str) -> TemporalData:
    feature_dir = FEATURE_ROOT / preset
    missing = [
        name for name in required_files(feature_dir)
        if not (feature_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Faltan salidas WST temporales en "
            f"{feature_dir}: {', '.join(missing)}"
        )

    x_train = np.load(feature_dir / "X_events_train.npy", mmap_mode="r")
    x_validation = np.load(
        feature_dir / "X_events_validation.npy", mmap_mode="r"
    )
    y_train = np.load(feature_dir / "y_events_train.npy")
    y_validation = np.load(feature_dir / "y_events_validation.npy")
    folds_train = np.load(feature_dir / "folds_events_train.npy")
    folds_validation = np.load(feature_dir / "folds_events_validation.npy")
    metadata_train = pd.read_csv(
        feature_dir / "metadata_events_features_train.csv"
    )
    metadata_validation = pd.read_csv(
        feature_dir / "metadata_events_features_validation.csv"
    )
    path_layout = pd.read_csv(
        feature_dir / "wavelet_scattering_temporal_path_layout.csv"
    )
    configuration = pd.read_csv(
        feature_dir / "wavelet_scattering_temporal_configuration.csv"
    )

    if len(configuration) != 1:
        raise ValueError("La configuracion temporal debe contener una fila.")
    if configuration.iloc[0]["stored_axis_order"] != "event,path,time":
        raise ValueError("El tensor no usa el orden esperado event,path,time.")
    if int(configuration.iloc[0]["J"]) != 13:
        raise ValueError("Esta ablacion mantiene J=13.")
    q_text = str(configuration.iloc[0]["Q"]).replace(" ", "")
    if q_text not in {"(8,1)", "[8,1]"}:
        raise ValueError(f"Esta ablacion mantiene Q=(8,1); recibido {q_text}.")
    if int(configuration.iloc[0]["max_order"]) != 2:
        raise ValueError("Se esperaban coeficientes hasta orden 2.")

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
        if x_values.ndim != 3:
            raise ValueError(
                f"{split}: se esperaba (evento,path,tiempo), recibido "
                f"{x_values.shape}."
            )
        if x_values.shape[0] != len(metadata):
            raise ValueError(f"{split}: tensor y metadata no coinciden.")
        if x_values.shape[1] != len(path_layout):
            raise ValueError(f"{split}: numero de paths inconsistente.")
        if len(y_values) != len(metadata) or len(folds) != len(metadata):
            raise ValueError(f"{split}: y/folds no coinciden con metadata.")
        if metadata["feature_row"].tolist() != list(range(len(metadata))):
            raise ValueError(f"{split}: feature_row no es consecutivo.")
        if not np.array_equal(
            y_values.astype(int),
            metadata["stage2_target"].to_numpy(dtype=int),
        ):
            raise ValueError(f"{split}: etiquetas desalineadas.")
        if not np.array_equal(
            folds.astype(int), metadata["fold"].to_numpy(dtype=int)
        ):
            raise ValueError(f"{split}: folds desalineados.")
        if not np.isfinite(np.asarray(x_values)).all():
            raise ValueError(f"{split}: el tensor contiene NaN o infinito.")
        if set(metadata["split"].astype(str)) != {split}:
            raise ValueError(f"{split}: columna split inesperada.")

    if set(metadata_train["fold"].astype(int)) != set(
        common.EXPECTED_TRAIN_FOLDS
    ):
        raise ValueError("TRAIN no contiene los cinco folds esperados.")
    if set(metadata_validation["fold"].astype(int)) != {-1}:
        raise ValueError("VALIDATION debe tener fold=-1.")
    overlap = set(metadata_train["original_uuid"].astype(str)) & set(
        metadata_validation["original_uuid"].astype(str)
    )
    if overlap:
        raise ValueError(
            f"TRAIN y VALIDATION comparten {len(overlap)} grabaciones."
        )
    if path_layout["path_index"].tolist() != list(range(len(path_layout))):
        raise ValueError("path_index no es consecutivo.")
    expected_orders = {0, 1, 2}
    orders = set(path_layout["scattering_order"].astype(int))
    if orders != expected_orders:
        raise ValueError(f"Ordenes WST inesperados: {sorted(orders)}.")

    return TemporalData(
        x_train=x_train,
        metadata_train=metadata_train,
        x_validation=x_validation,
        metadata_validation=metadata_validation,
        path_names=path_layout["path_name"].astype(str).tolist(),
        path_layout=path_layout,
        extraction_configuration=configuration,
    )


def temporal_statistic(x_values: np.ndarray, statistic: str) -> np.ndarray:
    if statistic == "mean":
        return np.mean(x_values, axis=2, dtype=np.float32)
    if statistic == "std":
        return np.std(x_values, axis=2, ddof=0, dtype=np.float32)
    if statistic == "max":
        return np.max(x_values, axis=2)
    raise ValueError(f"Estadistico temporal desconocido: {statistic}")


def event_view(
    raw: TemporalData,
    representation: TemporalRepresentation,
) -> common.DataView:
    def transform(x_values: np.ndarray) -> np.ndarray:
        blocks = [
            temporal_statistic(x_values, statistic)
            for statistic in representation.statistics
        ]
        result = np.concatenate(blocks, axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError(
                f"{representation.key} produjo NaN o infinito."
            )
        return result

    names = [
        f"temporal_{statistic}__{path_name}"
        for statistic in representation.statistics
        for path_name in raw.path_names
    ]
    return common.DataView(
        experiment=representation.key,
        x_train=transform(raw.x_train),
        train_metadata=raw.metadata_train.copy(),
        x_validation=transform(raw.x_validation),
        validation_metadata=raw.metadata_validation.copy(),
        feature_names=names,
    )


def recording_view(
    raw: TemporalData,
    representation: TemporalRepresentation,
) -> common.DataView:
    events = event_view(raw, representation)
    recordings = recording.build_recording_view(
        events,
        BETWEEN_EVENT_POOLING,
    )
    recordings.experiment = representation.key
    return consensus_base.attach_vote_metadata(recordings)


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


def baseline_equivalence(
    raw: TemporalData,
    preset: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "baseline_available": False,
        "event_shape_matches": False,
        "event_allclose": False,
        "event_max_absolute_difference": np.nan,
    }
    baseline_path = POOLED_BASELINE_ROOT / preset / "X_events_train.npy"
    if not baseline_path.exists():
        return result
    baseline = np.load(baseline_path, mmap_mode="r")
    temporal_mean = temporal_statistic(raw.x_train, "mean")
    result["baseline_available"] = True
    result["event_shape_matches"] = baseline.shape == temporal_mean.shape
    if baseline.shape == temporal_mean.shape:
        difference = np.abs(np.asarray(baseline) - temporal_mean)
        result["event_max_absolute_difference"] = float(difference.max())
        result["event_allclose"] = bool(
            np.allclose(baseline, temporal_mean, rtol=1e-5, atol=1e-7)
        )
    return result


def check(raw: TemporalData, preset: str) -> None:
    config = raw.extraction_configuration.iloc[0]
    equivalence = baseline_equivalence(raw, preset)
    train_recordings = raw.metadata_train.drop_duplicates("original_uuid")
    validation_recordings = raw.metadata_validation.drop_duplicates(
        "original_uuid"
    )

    print("=" * 78)
    print("CHECK - WST TEMPORAL STATISTICS + LR")
    print("=" * 78)
    print(f"Preset: {preset}")
    print(
        f"J={int(config['J'])} | Q={config['Q']} | "
        f"T={int(config['invariance_samples'])} | "
        f"posiciones={raw.x_train.shape[2]}"
    )
    print(f"Tensor TRAIN:      {raw.x_train.shape} {raw.x_train.dtype}")
    print(f"Tensor VALIDATION: {raw.x_validation.shape} {raw.x_validation.dtype}")
    print(
        "Grabaciones TRAIN dry/wet: "
        f"{train_recordings['stage2_target'].value_counts().sort_index().to_dict()}"
    )
    print(
        "Grabaciones VALIDATION dry/wet: "
        f"{validation_recordings['stage2_target'].value_counts().sort_index().to_dict()}"
    )
    print(
        "Paths por orden: "
        f"{raw.path_layout['scattering_order'].value_counts().sort_index().to_dict()}"
    )
    for representation in REPRESENTATIONS:
        event_dimension = len(raw.path_names) * len(representation.statistics)
        recording_dimension = event_dimension * len(
            BETWEEN_EVENT_POOLING.statistics
        )
        print(
            f"{representation.key}: evento={event_dimension}; "
            f"grabacion={recording_dimension} features"
        )
    print(
        "Equivalencia mean con baseline: "
        f"available={equivalence['baseline_available']} | "
        f"shape={equivalence['event_shape_matches']} | "
        f"allclose={equivalence['event_allclose']} | "
        f"max_diff={equivalence['event_max_absolute_difference']}"
    )
    if equivalence["baseline_available"] and not equivalence["event_allclose"]:
        raise RuntimeError(
            "La media temporal no reproduce el baseline WST existente."
        )
    print(
        "LR congelada: StandardScaler + PCA128 no-whiten + "
        "L2 C=0.001; gold=1; wet_cost=1."
    )
    print("TEST no se carga ni se procesa.")


def prefixed_metrics(
    metrics: dict[str, float | int],
    prefix: str,
) -> dict[str, float | int]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def model_float_count(model: object) -> int:
    return linear_base.learned_float_count(model)


def train(raw: TemporalData, preset: str) -> None:
    result_dir = RESULTS_ROOT / preset / "full"
    model_dir = MODELS_ROOT / preset / "full"
    graph_dir = GRAPHS_ROOT / preset / "full"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    candidate = frozen_candidate()
    comparison_rows: list[dict[str, object]] = []
    all_oof: list[pd.DataFrame] = []
    all_validation: list[pd.DataFrame] = []
    all_fold_metrics: list[pd.DataFrame] = []
    all_pca_variance: list[pd.DataFrame] = []
    evaluations: dict[str, lr_base.SearchResult] = {}

    print("\n" + "=" * 78)
    print("ABLACION T=FIJO - ESTADISTICAS TEMPORALES WST")
    print("=" * 78)
    for index, representation in enumerate(REPRESENTATIONS, start=1):
        print(
            f"\n[{index}/{len(REPRESENTATIONS)}] {representation.key} "
            f"{representation.statistics}"
        )
        data = recording_view(raw, representation)
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
        oof["temporal_representation"] = representation.key
        validation["temporal_representation"] = representation.key
        folds = evaluation.fold_metrics.copy()
        folds["temporal_representation"] = representation.key
        all_oof.append(oof)
        all_validation.append(validation)
        all_fold_metrics.append(folds)

        pca_variance = pca_variance.copy()
        pca_variance["temporal_representation"] = representation.key
        all_pca_variance.append(pca_variance)
        weight_diagnostics.to_csv(
            result_dir / f"weight_diagnostics_{representation.key}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        package = {
            "pipeline": pipeline,
            "experiment": "stage2_wst_temporal_statistics_lr",
            "preset": preset,
            "temporal_representation": asdict(representation),
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

        explained = pca_variance[
            "cumulative_explained_variance_ratio"
        ].mean()
        comparison_rows.append(
            {
                "temporal_representation": representation.key,
                "temporal_statistics": "|".join(representation.statistics),
                "T_samples": int(
                    raw.extraction_configuration.iloc[0]["invariance_samples"]
                ),
                "T_seconds": float(
                    raw.extraction_configuration.iloc[0][
                        "invariance_scale_seconds"
                    ]
                ),
                "time_position_count": raw.x_train.shape[2],
                "path_count": len(raw.path_names),
                "event_feature_count": (
                    len(raw.path_names) * len(representation.statistics)
                ),
                "recording_feature_count": data.x_train.shape[1],
                "pca_components": FROZEN_PCA_COMPONENTS,
                "pca_whiten": FROZEN_PCA_WHITEN,
                "mean_pca_explained_variance_ratio": float(explained),
                "C": FROZEN_C,
                "gold_multiplier": FROZEN_GOLD_MULTIPLIER,
                "wet_cost": FROZEN_WET_COST,
                "threshold_oof": evaluation.threshold,
                "model_size_kb": model_size_kb,
                "learned_float_count": model_float_count(pipeline),
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
            f"{validation_metrics['wet_recall']:.4f}"
        )

    winner_key = max(
        evaluations,
        key=lambda key: lr_base.selection_key(evaluations[key]),
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison["selected_by_oof"] = (
        comparison["temporal_representation"] == winner_key
    )
    comparison = comparison.sort_values(
        ["selected_by_oof", "oof__macro_f1", "oof__balanced_accuracy"],
        ascending=False,
    )
    comparison.to_csv(
        result_dir / "temporal_statistics_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_oof, ignore_index=True).to_csv(
        result_dir / "oof_predictions_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_validation, ignore_index=True).to_csv(
        result_dir / "validation_predictions_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_fold_metrics, ignore_index=True).to_csv(
        result_dir / "cv_fold_metrics_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_pca_variance, ignore_index=True).to_csv(
        result_dir / "pca_explained_variance_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    winner_oof = pd.concat(all_oof, ignore_index=True)
    winner_oof = winner_oof[
        winner_oof["temporal_representation"] == winner_key
    ]
    winner_validation = pd.concat(all_validation, ignore_index=True)
    winner_validation = winner_validation[
        winner_validation["temporal_representation"] == winner_key
    ]
    winner_oof.to_csv(
        result_dir / "best_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    winner_validation.to_csv(
        result_dir / "best_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    configuration = {
        "experiment": "stage2_wst_temporal_statistics_lr",
        "preset": preset,
        "T_samples": int(
            raw.extraction_configuration.iloc[0]["invariance_samples"]
        ),
        "J": int(raw.extraction_configuration.iloc[0]["J"]),
        "Q": str(raw.extraction_configuration.iloc[0]["Q"]),
        "representations": [asdict(item) for item in REPRESENTATIONS],
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
        for key, value in configuration.items()
    ).to_csv(
        result_dir / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    winner_row = comparison.iloc[0]
    print("\n" + "=" * 78)
    print("RESULTADO - ABLACION DE ESTADISTICAS TEMPORALES WST")
    print("=" * 78)
    print(f"Ganador segun OOF: {winner_key}")
    print(f"Macro-F1 OOF: {winner_row['oof__macro_f1']:.4f}")
    print(f"Macro-F1 validation: {winner_row['validation__macro_f1']:.4f}")
    print(
        "Recalls validation dry/wet: "
        f"{winner_row['validation__dry_recall']:.4f} / "
        f"{winner_row['validation__wet_recall']:.4f}"
    )
    print(f"Comparacion: {result_dir / 'temporal_statistics_comparison.csv'}")
    print(f"Resultados: {result_dir}")
    print("TEST no se ha cargado ni evaluado.")


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST TEMPORAL STATISTICS + LR")
    print("=" * 78)
    print(f"Preset: {args.preset} | accion: {args.action}")
    print("TEST no sera leido ni procesado.")
    raw = load_temporal_data(args.preset)
    check(raw, args.preset)
    if args.action == "train":
        train(raw, args.preset)
    else:
        print("\nComprobacion completada. No se entreno ningun modelo.")


if __name__ == "__main__":
    main()
