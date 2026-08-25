"""WST recording + LR en dos fases incorporando augmentation gold.

Repite la busqueda del experimento ``lr_two_phase_search`` y anade al ajuste
las 48 grabaciones logicas creadas a partir de 6 wet gold y 6 dry gold. Cada
variante se agrega a nivel recording con mean+std+max de sus eventos WST.

Regla de validacion esencial
----------------------------
El OOF contiene solamente grabaciones originales. En el fold f se excluyen
del ajuste tanto los originales de f como todas las variantes cuyo
``parent_original_uuid`` pertenece a f. VALIDATION permanece externa a la
seleccion y TEST no se lee.

Las variantes conservan ``gold_expert`` y reciben el mismo multiplicador gold
que cualquier otra grabacion gold del candidato. Los diagnosticos guardan por
separado el peso de originales y aumentadas para hacer visible su influencia.
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

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_linear as linear_base
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording
import train_stage2_dry_wet_wst_recording_consensus_weighted_lr as consensus_base
import train_stage2_dry_wet_wst_recording_lr_two_phase_search as search_base


SCRIPT_DIR = Path(__file__).resolve().parent
AUGMENTED_FEATURES_ROOT = (
    SCRIPT_DIR / "features_extracted_stage2_dry_wet_wst_gold_augmentation"
)
RESULTS_ROOT = (
    SCRIPT_DIR
    / "results_stage2_dry_wet_wst_recording_lr_gold_augmentation"
)
MODELS_ROOT = (
    SCRIPT_DIR
    / "models_stage2_dry_wet_wst_recording_lr_gold_augmentation"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wst_recording_lr_gold_augmentation"
)

PRESETS = linear_base.PRESETS
DEFAULT_PRESET = linear_base.DEFAULT_PRESET
RANDOM_STATE = linear_base.RANDOM_STATE
EXPERIMENT_KEY = "wst_recording_lr_gold_augmentation"
EXPECTED_AUGMENTED_RECORDINGS = 48
EXPECTED_AUGMENTATIONS_PER_PARENT = 4


@dataclass
class AugmentedExperimentData:
    original: common.DataView
    x_augmented: np.ndarray
    augmented_metadata: pd.DataFrame
    augmentation_configuration: pd.DataFrame


def augmented_paths(preset: str) -> dict[str, Path]:
    root = AUGMENTED_FEATURES_ROOT / preset
    return {
        "x": root / "X_events_train_augmented.npy",
        "y": root / "y_events_train_augmented.npy",
        "folds": root / "folds_events_train_augmented.npy",
        "metadata": root / "metadata_events_features_train_augmented.csv",
        "layout": root / "wavelet_scattering_feature_layout.csv",
        "configuration": root / "augmentation_configuration.csv",
    }


def add_original_lineage(data: common.DataView) -> common.DataView:
    train = data.train_metadata.copy()
    validation = data.validation_metadata.copy()
    for metadata in (train, validation):
        metadata["is_augmented"] = False
        metadata["parent_original_uuid"] = metadata["original_uuid"].astype(str)
        metadata["family_uuid"] = metadata["original_uuid"].astype(str)
        metadata["augmentation_type"] = "original"
        metadata["selection_role"] = "original"
    return common.DataView(
        experiment=f"{data.experiment}_gold_augmentation",
        x_train=data.x_train,
        train_metadata=train,
        x_validation=data.x_validation,
        validation_metadata=validation,
        feature_names=data.feature_names,
    )


def load_augmented_recordings(
    preset: str,
    original_event_data: common.DataView,
    original_recording_data: common.DataView,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    paths = augmented_paths(preset)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan features del augmentation gold:\n" + "\n".join(missing)
        )

    x_events = np.load(paths["x"])
    y_events = np.load(paths["y"])
    folds_events = np.load(paths["folds"])
    event_metadata = pd.read_csv(paths["metadata"])
    layout = pd.read_csv(paths["layout"])
    configuration = pd.read_csv(paths["configuration"])

    common.validate_arrays_and_metadata(
        "event",
        "train",
        x_events,
        y_events,
        folds_events,
        event_metadata,
    )
    if x_events.shape[1] != original_event_data.x_train.shape[1]:
        raise ValueError("Las WST aumentadas no tienen 644 features compatibles.")
    if layout["feature_name"].astype(str).tolist() != original_event_data.feature_names:
        raise ValueError("El layout WST aumentado no coincide con el original.")

    required_lineage = {
        "parent_original_uuid",
        "augmentation_type",
        "selection_role",
        "is_augmented",
        "maximum_agreement_count",
        "expert_evaluator_count",
        "dry_vote_count",
        "wet_vote_count",
        "unknown_vote_count",
        "vote_pattern",
    }
    missing_lineage = required_lineage - set(event_metadata.columns)
    if missing_lineage:
        raise ValueError(
            f"Falta lineage en las WST aumentadas: {sorted(missing_lineage)}"
        )
    if not event_metadata["is_augmented"].astype(bool).all():
        raise ValueError("Las matrices aumentadas contienen filas originales.")

    x_recordings, metadata = recording.aggregate_split(
        x_events,
        event_metadata,
        original_event_data.feature_names,
        linear_base.RECORDING_POOLING,
        "train",
    )
    lineage_columns = [
        "original_uuid",
        "parent_original_uuid",
        "augmentation_type",
        "selection_role",
        "is_augmented",
        "expert_evaluator_count",
        "dry_vote_count",
        "wet_vote_count",
        "unknown_vote_count",
        "maximum_agreement_count",
        "vote_pattern",
    ]
    lineage = event_metadata[lineage_columns].drop_duplicates()
    if lineage["original_uuid"].duplicated().any():
        raise ValueError("El lineage aumentado no es constante por grabacion.")
    metadata = metadata.merge(
        lineage,
        on="original_uuid",
        how="left",
        validate="one_to_one",
    )
    metadata["family_uuid"] = metadata["parent_original_uuid"].astype(str)
    metadata["consensus_group"] = np.where(
        metadata["maximum_agreement_count"].astype(int).eq(4),
        "gold_4_of_4",
        "gold_3_of_4",
    )

    validate_augmented_recordings(
        x_recordings,
        metadata,
        original_recording_data,
    )
    return x_recordings, metadata, configuration


def validate_augmented_recordings(
    x_augmented: np.ndarray,
    metadata: pd.DataFrame,
    original: common.DataView,
) -> None:
    if x_augmented.shape != (
        EXPECTED_AUGMENTED_RECORDINGS,
        linear_base.EXPECTED_RECORDING_FEATURE_COUNT,
    ):
        raise ValueError(
            f"Forma recording aumentada inesperada: {x_augmented.shape}."
        )
    if len(metadata) != EXPECTED_AUGMENTED_RECORDINGS:
        raise ValueError("Se esperaban exactamente 48 grabaciones aumentadas.")
    if not np.isfinite(x_augmented).all():
        raise ValueError("Las features recording aumentadas contienen NaN/Inf.")
    if not metadata["cough_type_consensus"].eq("gold_expert").all():
        raise ValueError("Alguna variante aumentada no conserva gold_expert.")
    if set(metadata["selection_role"]) != {"wet_gold", "dry_gold_control"}:
        raise ValueError("Roles inesperados en el augmentation.")
    if metadata["original_uuid"].duplicated().any():
        raise ValueError("UUID de grabaciones aumentadas duplicados.")

    original_meta = original.train_metadata.set_index("original_uuid")
    parent_ids = metadata["parent_original_uuid"].astype(str)
    if not set(parent_ids).issubset(set(original_meta.index.astype(str))):
        raise ValueError("Hay variantes cuyo padre no existe en TRAIN original.")
    inherited_folds = parent_ids.map(original_meta["fold"].astype(int))
    if not np.array_equal(
        inherited_folds.to_numpy(dtype=int),
        metadata["fold"].to_numpy(dtype=int),
    ):
        raise ValueError("Alguna variante no conserva el fold de su padre.")
    inherited_targets = parent_ids.map(original_meta["stage2_target"].astype(int))
    if not np.array_equal(
        inherited_targets.to_numpy(dtype=int),
        metadata["stage2_target"].to_numpy(dtype=int),
    ):
        raise ValueError("Alguna variante no conserva la etiqueta de su padre.")

    variants_per_parent = metadata.groupby("parent_original_uuid").size()
    if set(variants_per_parent) != {EXPECTED_AUGMENTATIONS_PER_PARENT}:
        raise ValueError("Cada padre debe aportar exactamente cuatro variantes.")
    if metadata.groupby("parent_original_uuid")["augmentation_type"].nunique().ne(
        EXPECTED_AUGMENTATIONS_PER_PARENT
    ).any():
        raise ValueError("Un padre no contiene las cuatro transformaciones.")
    if set(metadata["original_uuid"]) & set(original.train_metadata["original_uuid"]):
        raise ValueError("Un UUID aumentado colisiona con un original.")


def load_experiment_data(
    preset: str,
) -> tuple[AugmentedExperimentData, pd.DataFrame]:
    event_data, original_extraction_configuration = linear_base.FEATURE_LOADER(
        preset
    )
    recording_data = linear_base.build_recording_data(event_data)
    recording_data = consensus_base.attach_vote_metadata(recording_data)
    recording_data = add_original_lineage(recording_data)
    x_augmented, augmented_metadata, aug_config = load_augmented_recordings(
        preset,
        event_data,
        recording_data,
    )
    return (
        AugmentedExperimentData(
            original=recording_data,
            x_augmented=x_augmented,
            augmented_metadata=augmented_metadata,
            augmentation_configuration=aug_config,
        ),
        original_extraction_configuration,
    )


def fit_metadata_for_fold(
    data: AugmentedExperimentData,
    fold: int,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    original_metadata = data.original.train_metadata
    original_fit_mask = original_metadata["fold"].to_numpy(dtype=int) != fold
    original_validation_mask = ~original_fit_mask
    augmented_fit_mask = (
        data.augmented_metadata["fold"].to_numpy(dtype=int) != fold
    )

    validation_parent_ids = set(
        original_metadata.loc[original_validation_mask, "original_uuid"].astype(str)
    )
    fit_augmented_parents = set(
        data.augmented_metadata.loc[
            augmented_fit_mask, "parent_original_uuid"
        ].astype(str)
    )
    leaked = validation_parent_ids & fit_augmented_parents
    if leaked:
        raise RuntimeError(
            f"Fold {fold}: leakage de padres aumentados: {sorted(leaked)[:5]}"
        )

    x_fit = np.vstack(
        [
            data.original.x_train[original_fit_mask],
            data.x_augmented[augmented_fit_mask],
        ]
    ).astype(np.float32, copy=False)
    metadata_fit = pd.concat(
        [
            original_metadata.loc[original_fit_mask],
            data.augmented_metadata.loc[augmented_fit_mask],
        ],
        ignore_index=True,
        sort=False,
    )
    if len(x_fit) != len(metadata_fit):
        raise RuntimeError("TRAIN interno y metadata no quedaron alineados.")
    return x_fit, metadata_fit, original_validation_mask, augmented_fit_mask


def transformed_folds(
    data: AugmentedExperimentData,
    candidates: list[search_base.SearchCandidate],
) -> tuple[
    dict[tuple[int, int, bool], tuple[np.ndarray, np.ndarray]],
    dict[int, pd.DataFrame],
    pd.DataFrame,
]:
    representations = sorted(
        {
            (int(candidate.spec.pca_components), candidate.pca_whiten)
            for candidate in candidates
        }
    )
    component_values = sorted({components for components, _ in representations})
    cache: dict[tuple[int, int, bool], tuple[np.ndarray, np.ndarray]] = {}
    metadata_by_fold: dict[int, pd.DataFrame] = {}
    variance_rows: list[dict[str, Any]] = []

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        x_fit_raw, metadata_fit, validation_mask, augmented_fit_mask = (
            fit_metadata_for_fold(data, fold)
        )
        x_validation_raw = data.original.x_train[validation_mask]
        metadata_by_fold[fold] = metadata_fit

        scaler = StandardScaler()
        x_fit_scaled = scaler.fit_transform(x_fit_raw).astype(np.float32)
        x_validation_scaled = scaler.transform(x_validation_raw).astype(np.float32)

        for components in component_values:
            pca = PCA(
                n_components=components,
                svd_solver="randomized",
                n_oversamples=12,
                iterated_power=4,
                power_iteration_normalizer="auto",
                whiten=False,
                random_state=RANDOM_STATE,
            )
            pca.fit(x_fit_scaled)
            explained = float(np.sum(pca.explained_variance_ratio_))
            whiten_values = sorted(
                whiten
                for current_components, whiten in representations
                if current_components == components
            )
            for whiten in whiten_values:
                pca.whiten = whiten
                cache[(fold, components, whiten)] = (
                    pca.transform(x_fit_scaled).astype(np.float32),
                    pca.transform(x_validation_scaled).astype(np.float32),
                )
                variance_rows.append(
                    {
                        "fold": fold,
                        "pca_components": components,
                        "pca_whiten": whiten,
                        "cumulative_explained_variance_ratio": explained,
                        "cumulative_explained_variance_percent": explained * 100.0,
                        "original_fit_recordings": int(
                            (~validation_mask).sum()
                        ),
                        "augmented_fit_recordings": int(augmented_fit_mask.sum()),
                        "oof_original_recordings": int(validation_mask.sum()),
                    }
                )
        print(
            f"  Fold {fold}: fit={len(metadata_fit)} "
            f"({(~validation_mask).sum()} originales + "
            f"{augmented_fit_mask.sum()} aumentadas); "
            f"OOF={validation_mask.sum()} originales."
        )
    return cache, metadata_by_fold, pd.DataFrame(variance_rows)


def weight_diagnostics_by_source(
    metadata: pd.DataFrame,
    fold: int,
    gold_multiplier: float,
    wet_cost: float,
) -> list[dict[str, Any]]:
    weights = search_base.consensus_and_wet_weights(
        metadata, gold_multiplier, wet_cost
    )
    diagnostic = metadata.copy()
    diagnostic["sample_weight"] = weights
    diagnostic["source_type"] = np.where(
        diagnostic["is_augmented"].astype(bool), "augmented", "original"
    )
    rows: list[dict[str, Any]] = []
    for keys, group in diagnostic.groupby(
        ["source_type", "cough_type", "consensus_group"], sort=True
    ):
        source_type, class_name, consensus_group = keys
        rows.append(
            {
                "fold_excluded": fold,
                "gold_multiplier": gold_multiplier,
                "wet_cost": wet_cost,
                "source_type": source_type,
                "class_name": class_name,
                "consensus_group": consensus_group,
                "recording_count": len(group),
                "weight_sum": float(group["sample_weight"].sum()),
                "weight_mean": float(group["sample_weight"].mean()),
            }
        )
    return rows


def evaluate_candidates(
    data: AugmentedExperimentData,
    candidates: list[search_base.SearchCandidate],
) -> tuple[list[search_base.SearchResult], pd.DataFrame, pd.DataFrame]:
    cache, metadata_by_fold, variance = transformed_folds(data, candidates)
    original = data.original
    oof = {
        candidate.key: np.full(len(original.x_train), np.nan, dtype=float)
        for candidate in candidates
    }
    elapsed = {candidate.key: 0.0 for candidate in candidates}
    weight_rows: list[dict[str, Any]] = []
    weight_configurations = sorted(
        {
            (candidate.gold_multiplier, candidate.wet_cost)
            for candidate in candidates
        }
    )

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = (
            original.train_metadata["fold"].to_numpy(dtype=int) == fold
        )
        metadata_fit = metadata_by_fold[fold]
        y_fit = metadata_fit["stage2_target"].to_numpy(dtype=int)
        weights_by_configuration: dict[tuple[float, float], np.ndarray] = {}
        for gold_multiplier, wet_cost in weight_configurations:
            key = (gold_multiplier, wet_cost)
            weights_by_configuration[key] = (
                search_base.consensus_and_wet_weights(
                    metadata_fit, gold_multiplier, wet_cost
                )
            )
            weight_rows.extend(
                weight_diagnostics_by_source(
                    metadata_fit,
                    fold,
                    gold_multiplier,
                    wet_cost,
                )
            )

        for candidate in candidates:
            x_fit, x_validation = cache[
                (
                    fold,
                    int(candidate.spec.pca_components),
                    candidate.pca_whiten,
                )
            ]
            classifier = linear_base.build_classifier(candidate.spec)
            started = time.perf_counter()
            classifier.fit(
                x_fit,
                y_fit,
                sample_weight=weights_by_configuration[
                    (candidate.gold_multiplier, candidate.wet_cost)
                ],
            )
            oof[candidate.key][validation_mask] = linear_base.classifier_scores(
                classifier, x_validation
            )
            elapsed[candidate.key] += time.perf_counter() - started

    results: list[search_base.SearchResult] = []
    for index, candidate in enumerate(candidates, start=1):
        scores = oof[candidate.key]
        if not np.isfinite(scores).all():
            raise RuntimeError(f"OOF original incompleto para {candidate.key}.")
        result = search_base.make_result(
            data.original,
            candidate,
            scores,
            elapsed[candidate.key],
        )
        results.append(result)
        print(
            f"[{index:03d}/{len(candidates):03d}] {candidate.key} | "
            f"macro-F1={result.metrics_tuned['macro_f1']:.4f} | "
            f"wet-recall={result.metrics_tuned['wet_recall']:.4f} | "
            f"AUC={result.metrics_tuned['roc_auc']:.4f}"
        )
    return results, variance, pd.DataFrame(weight_rows)


def combined_training_data(
    data: AugmentedExperimentData,
) -> tuple[np.ndarray, pd.DataFrame]:
    x_values = np.vstack(
        [data.original.x_train, data.x_augmented]
    ).astype(np.float32, copy=False)
    metadata = pd.concat(
        [data.original.train_metadata, data.augmented_metadata],
        ignore_index=True,
        sort=False,
    )
    if len(x_values) != len(metadata):
        raise RuntimeError("TRAIN final combinado no quedo alineado.")
    return x_values, metadata


def build_final_pipeline(
    data: AugmentedExperimentData,
    candidate: search_base.SearchCandidate,
) -> Pipeline:
    x_fit, metadata_fit = combined_training_data(data)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=candidate.spec.pca_components,
                    svd_solver="randomized",
                    n_oversamples=12,
                    iterated_power=4,
                    power_iteration_normalizer="auto",
                    whiten=candidate.pca_whiten,
                    random_state=RANDOM_STATE,
                ),
            ),
            ("classifier", linear_base.build_classifier(candidate.spec)),
        ]
    )
    weights = search_base.consensus_and_wet_weights(
        metadata_fit,
        candidate.gold_multiplier,
        candidate.wet_cost,
    )
    model.fit(
        x_fit,
        metadata_fit["stage2_target"].to_numpy(dtype=int),
        classifier__sample_weight=weights,
    )
    return model


def train(
    data: AugmentedExperimentData,
    original_configuration: pd.DataFrame,
    preset: str,
    quick: bool,
) -> None:
    run_name = "quick" if quick else "full"
    result_dir = RESULTS_ROOT / preset / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("FASE A - REPRESENTACION CON AUGMENTATION GOLD")
    print("=" * 78)
    phase_a_candidates = search_base.phase_a_candidate_grid(quick)
    phase_a_results, phase_a_variance, phase_a_weights = evaluate_candidates(
        data, phase_a_candidates
    )
    top_phase_a = search_base.ranked_results(phase_a_results)[:3]
    search_base.candidate_results_frame(phase_a_results).to_csv(
        result_dir / "phase_a_candidate_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    search_base.candidate_results_frame(top_phase_a).to_csv(
        result_dir / "phase_a_top3_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )
    phase_a_variance.to_csv(
        result_dir / "phase_a_pca_explained_variance_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    phase_a_weights.to_csv(
        result_dir / "phase_a_weight_diagnostics_by_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nTop 3 fase A:")
    for result in top_phase_a:
        print(
            f"  {result.candidate.representation_key} | "
            f"macro-F1={result.metrics_tuned['macro_f1']:.4f} | "
            f"wet-recall={result.metrics_tuned['wet_recall']:.4f}"
        )

    print("\n" + "=" * 78)
    print("FASE B - PESO GOLD Y COSTE WET CON AUGMENTATION")
    print("=" * 78)
    phase_b_candidates = search_base.phase_b_candidate_grid(top_phase_a, quick)
    phase_b_results, phase_b_variance, phase_b_weights = evaluate_candidates(
        data, phase_b_candidates
    )
    winner = search_base.ranked_results(phase_b_results)[0]
    search_base.candidate_results_frame(phase_b_results).to_csv(
        result_dir / "phase_b_candidate_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    phase_b_variance.to_csv(
        result_dir / "phase_b_pca_explained_variance_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    phase_b_weights.to_csv(
        result_dir / "phase_b_weight_diagnostics_by_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    search_base.save_candidate_subgroups(
        phase_b_results,
        result_dir / "phase_b_metrics_by_consensus_subgroup_oof.csv",
    )
    winner.predictions.to_csv(
        result_dir / "best_oof_predictions_original_only.csv",
        index=False,
        encoding="utf-8-sig",
    )
    winner.fold_metrics.to_csv(
        result_dir / "best_cv_fold_metrics_original_only.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation = None
    model_size_kb = np.nan
    if not quick:
        print("\nAjustando LR final con TRAIN original + augmentation gold...")
        model = build_final_pipeline(data, winner.candidate)
        validation = search_base.validation_predictions(
            data.original, model, winner
        )
        validation.to_csv(
            result_dir / "validation_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        model_dir = MODELS_ROOT / preset / run_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "logistic_regression_model.joblib"
        package = {
            "pipeline": model,
            "experiment": EXPERIMENT_KEY,
            "preset": preset,
            "candidate_key": winner.candidate.key,
            "threshold": winner.threshold,
            "minimum_wet_recall_oof": search_base.MIN_WET_RECALL_OOF,
            "gold_multiplier": winner.candidate.gold_multiplier,
            "wet_cost": winner.candidate.wet_cost,
            "recording_feature_names": data.original.feature_names,
            "original_extraction_configuration": (
                original_configuration.iloc[0].to_dict()
            ),
            "augmentation_configuration": (
                data.augmentation_configuration.to_dict(orient="records")
            ),
            "oof_evaluated_on_originals_only": True,
            "validation_used_for_selection": False,
            "test_processed": False,
            "random_state": RANDOM_STATE,
        }
        joblib.dump(package, model_path, compress=3)
        model_size_kb = model_path.stat().st_size / 1024.0

        graph_dir = GRAPHS_ROOT / preset / run_name
        graph_dir.mkdir(parents=True, exist_ok=True)
        common.create_validation_graph(
            validation,
            winner.threshold,
            EXPERIMENT_KEY,
            winner.candidate.spec,
            graph_dir / "validation_wst_lr_gold_augmentation.png",
        )

    metric_rows = [
        {
            "dataset": "train_oof_original_only",
            "candidate_key": winner.candidate.key,
            "threshold": winner.threshold,
            "model_size_kb_joblib": model_size_kb,
            **winner.metrics_tuned,
        }
    ]
    if validation is not None:
        metric_rows.append(
            {
                "dataset": "validation_original_only",
                "candidate_key": winner.candidate.key,
                "threshold": winner.threshold,
                "model_size_kb_joblib": model_size_kb,
                **common.binary_metrics(
                    validation["y_true"].to_numpy(dtype=int),
                    validation["score"].to_numpy(dtype=float),
                    winner.threshold,
                ),
            }
        )
    pd.DataFrame(metric_rows).to_csv(
        result_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "experiment": EXPERIMENT_KEY,
                "original_train_recordings": len(data.original.x_train),
                "augmented_train_recordings": len(data.x_augmented),
                "augmented_dry_recordings": int(
                    data.augmented_metadata["stage2_target"].eq(0).sum()
                ),
                "augmented_wet_recordings": int(
                    data.augmented_metadata["stage2_target"].eq(1).sum()
                ),
                "augmentation_parent_recordings": int(
                    data.augmented_metadata["parent_original_uuid"].nunique()
                ),
                "oof_contains_augmented_recordings": False,
                "augmented_excluded_with_parent_fold": True,
                "augmented_consensus": "gold_expert",
                "phase_a_candidate_count": len(phase_a_candidates),
                "phase_b_candidate_count": len(phase_b_candidates),
                "winner": winner.candidate.key,
                "validation_used_for_selection": False,
                "test_processed": False,
            }
        ]
    ).to_csv(
        result_dir / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 78)
    print("RESULTADO WST RECORDING + LR + AUGMENTATION GOLD")
    print("=" * 78)
    print(f"Ganador OOF: {winner.candidate.key}")
    print(f"Umbral OOF restringido: {winner.threshold:.6f}")
    print(f"Macro-F1 OOF original: {winner.metrics_tuned['macro_f1']:.4f}")
    print(f"Recall wet OOF original: {winner.metrics_tuned['wet_recall']:.4f}")
    if validation is None:
        print("VALIDATION no se ha evaluado en modo --quick.")
    else:
        metrics = metric_rows[-1]
        print(f"Macro-F1 validation: {metrics['macro_f1']:.4f}")
        print(
            "Recalls validation dry/wet: "
            f"{metrics['dry_recall']:.4f} / {metrics['wet_recall']:.4f}"
        )
        print(f"Modelo: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print("OOF y VALIDATION contienen solo originales. TEST permanece reservado.")


def print_check(data: AugmentedExperimentData, quick: bool) -> None:
    phase_a_count = len(search_base.phase_a_candidate_grid(quick))
    phase_b_gold = (
        search_base.QUICK_GOLD_MULTIPLIERS
        if quick
        else search_base.PHASE_B_GOLD_MULTIPLIERS
    )
    phase_b_wet = (
        search_base.QUICK_WET_COSTS
        if quick
        else search_base.PHASE_B_WET_COSTS
    )
    parents_by_fold = (
        data.augmented_metadata.drop_duplicates("parent_original_uuid")
        .groupby("fold")
        .size()
        .to_dict()
    )
    print("=" * 78)
    print("CHECK - WST RECORDING + LR + AUGMENTATION GOLD")
    print("=" * 78)
    print(
        f"TRAIN original: {data.original.x_train.shape}; "
        f"augmentation: {data.x_augmented.shape}; "
        f"VALIDATION: {data.original.x_validation.shape}"
    )
    print(
        "Grabaciones aumentadas dry/wet: "
        f"{data.augmented_metadata.groupby('cough_type').size().to_dict()}"
    )
    print(
        "Padres aumentados por fold: "
        f"{parents_by_fold}"
    )
    print(f"Fase A: {phase_a_count} candidatos.")
    print(
        "Fase B: "
        f"{3 * len(phase_b_gold) * len(phase_b_wet)} candidatos."
    )
    print("Las aumentadas son gold y reciben gold_multiplier=1 o 1.5.")
    print("OOF: solo originales; variantes excluidas junto a su padre.")
    print("VALIDATION no selecciona. TEST no se lee.")

    # Verificacion explicita para los cinco folds antes de entrenar.
    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        _, metadata_fit, validation_mask, augmented_mask = fit_metadata_for_fold(
            data, fold
        )
        weights = search_base.consensus_and_wet_weights(
            metadata_fit, 1.5, 1.0
        )
        augmented_weight = float(
            weights[metadata_fit["is_augmented"].astype(bool).to_numpy()].sum()
        )
        print(
            f"  Fold {fold}: OOF originales={validation_mask.sum()}, "
            f"aumentadas en fit={augmented_mask.sum()}, "
            f"peso aumentado total @gold1.5={augmented_weight:.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LR WST recording en dos fases con augmentation gold."
    )
    parser.add_argument("--action", choices=["check", "train"], default="check")
    parser.add_argument("--preset", choices=PRESETS, default=DEFAULT_PRESET)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Rejilla reducida OOF; no evalua VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST RECORDING + LR + AUGMENTATION GOLD")
    print("=" * 78)
    print(f"Preset: {args.preset} | Accion: {args.action} | Quick: {args.quick}")
    print("TEST no sera leido ni procesado.")
    data, original_configuration = load_experiment_data(args.preset)
    print_check(data, args.quick)
    if args.action == "check":
        print("\nComprobacion completada. No se entreno ningun modelo.")
        return
    train(data, original_configuration, args.preset, args.quick)


if __name__ == "__main__":
    main()
