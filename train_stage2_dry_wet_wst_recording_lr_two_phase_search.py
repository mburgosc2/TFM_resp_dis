"""Busqueda en dos fases para WST recording + regresion logistica.

FASE A
------
Mantiene gold_multiplier=1, wet_cost=1 y penalizacion L2. Busca C, numero de
componentes PCA y whitening usando exclusivamente OOF de TRAIN.

FASE B
------
Conserva las tres mejores representaciones de la fase A y busca un peso gold
moderado (1 o 1.5) y un coste wet (1, 1.15, 1.30 o 1.50).

El umbral de cada candidato maximiza macro-F1 OOF sujeto a que recall wet OOF
no sea inferior al baseline (0.5505319). Scaler, PCA, pesos y clasificadores se
ajustan dentro de cada fold. VALIDATION solo se evalua una vez con el ganador
final de TRAIN. TEST no se lee ni se procesa.
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
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_linear as linear_base
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording
import train_stage2_dry_wet_wst_recording_consensus_weighted_lr as consensus_base


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results_stage2_dry_wet_wst_recording_lr_two_phase_search"
MODELS_ROOT = SCRIPT_DIR / "models_stage2_dry_wet_wst_recording_lr_two_phase_search"
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wst_recording_lr_two_phase_search"
)

RANDOM_STATE = linear_base.RANDOM_STATE
PRESETS = linear_base.PRESETS
DEFAULT_PRESET = linear_base.DEFAULT_PRESET
EXPERIMENT_KEY = "wst_recording_lr_two_phase_search"

PHASE_A_C_VALUES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
PHASE_A_PCA_OPTIONS = (64, 128, 192, 256, 384, 512)
PHASE_A_WHITEN_OPTIONS = (False, True)

PHASE_B_GOLD_MULTIPLIERS = (1.0, 1.5)
PHASE_B_WET_COSTS = (1.0, 1.15, 1.30, 1.50)

QUICK_C_VALUES = (1e-4, 1e-3)
QUICK_PCA_OPTIONS = (128, 256)
QUICK_WHITEN_OPTIONS = (False, True)
QUICK_GOLD_MULTIPLIERS = (1.0, 1.5)
QUICK_WET_COSTS = (1.0, 1.30)

MIN_WET_RECALL_OOF = 0.550531914893617
PENALTY = "l2"


def number_key(value: float) -> str:
    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class SearchCandidate:
    spec: common.CandidateSpec
    pca_whiten: bool
    gold_multiplier: float
    wet_cost: float

    @property
    def key(self) -> str:
        whiten_key = "whiten" if self.pca_whiten else "no_whiten"
        return (
            f"{self.spec.key}__{whiten_key}"
            f"__gold{number_key(self.gold_multiplier)}"
            f"__wetcost{number_key(self.wet_cost)}"
        )

    @property
    def representation_key(self) -> str:
        whiten_key = "whiten" if self.pca_whiten else "no_whiten"
        return f"{self.spec.key}__{whiten_key}"


@dataclass
class SearchResult:
    candidate: SearchCandidate
    threshold: float
    metrics_tuned: dict[str, float | int]
    metrics_native: dict[str, float | int]
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    elapsed_seconds: float


def phase_a_candidate_grid(quick: bool) -> list[SearchCandidate]:
    c_values = QUICK_C_VALUES if quick else PHASE_A_C_VALUES
    pca_options = QUICK_PCA_OPTIONS if quick else PHASE_A_PCA_OPTIONS
    whiten_options = QUICK_WHITEN_OPTIONS if quick else PHASE_A_WHITEN_OPTIONS
    return [
        SearchCandidate(
            spec=common.CandidateSpec(
                "logistic_regression",
                c_value,
                pca_components,
            ),
            pca_whiten=pca_whiten,
            gold_multiplier=1.0,
            wet_cost=1.0,
        )
        for c_value in c_values
        for pca_components in pca_options
        for pca_whiten in whiten_options
    ]


def phase_b_candidate_grid(
    top_phase_a: list[SearchResult],
    quick: bool,
) -> list[SearchCandidate]:
    gold_values = (
        QUICK_GOLD_MULTIPLIERS if quick else PHASE_B_GOLD_MULTIPLIERS
    )
    wet_costs = QUICK_WET_COSTS if quick else PHASE_B_WET_COSTS
    return [
        SearchCandidate(
            spec=result.candidate.spec,
            pca_whiten=result.candidate.pca_whiten,
            gold_multiplier=gold_multiplier,
            wet_cost=wet_cost,
        )
        for result in top_phase_a
        for gold_multiplier in gold_values
        for wet_cost in wet_costs
    ]


def transformed_folds(
    data: common.DataView,
    candidates: list[SearchCandidate],
) -> tuple[
    dict[tuple[int, int, bool], tuple[np.ndarray, np.ndarray]],
    pd.DataFrame,
]:
    """Ajusta scaler/PCA solo con train interno y cachea cada proyeccion."""
    representations = sorted(
        {
            (int(candidate.spec.pca_components), candidate.pca_whiten)
            for candidate in candidates
        }
    )
    components_to_whiten = sorted(
        {components for components, _ in representations}
    )
    cache: dict[
        tuple[int, int, bool], tuple[np.ndarray, np.ndarray]
    ] = {}
    variance_rows: list[dict[str, Any]] = []

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = data.train_metadata["fold"].to_numpy(dtype=int) == fold
        training_mask = ~validation_mask
        x_fit_raw = data.x_train[training_mask]
        x_validation_raw = data.x_train[validation_mask]

        scaler = StandardScaler()
        x_fit_scaled = scaler.fit_transform(x_fit_raw).astype(np.float32)
        x_validation_scaled = scaler.transform(x_validation_raw).astype(np.float32)

        for components in components_to_whiten:
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

            requested_whiten = sorted(
                whiten
                for current_components, whiten in representations
                if current_components == components
            )
            for pca_whiten in requested_whiten:
                # Whitening no cambia los componentes aprendidos; unicamente
                # la escala aplicada por transform. Se evita ajustar dos PCA
                # identicas para cada fold.
                pca.whiten = pca_whiten
                x_fit_pca = pca.transform(x_fit_scaled).astype(np.float32)
                x_validation_pca = pca.transform(
                    x_validation_scaled
                ).astype(np.float32)
                cache[(fold, components, pca_whiten)] = (
                    x_fit_pca,
                    x_validation_pca,
                )
                variance_rows.append(
                    {
                        "fold": fold,
                        "pca_components": components,
                        "pca_whiten": pca_whiten,
                        "cumulative_explained_variance_ratio": explained,
                        "cumulative_explained_variance_percent": explained * 100.0,
                    }
                )
        print(
            f"  Fold {fold}: scaler y {len(components_to_whiten)} PCA "
            "ajustados solo con TRAIN interno."
        )
    return cache, pd.DataFrame(variance_rows)


def consensus_and_wet_weights(
    metadata: pd.DataFrame,
    gold_multiplier: float,
    wet_cost: float,
) -> np.ndarray:
    """Pondera consenso dentro de clase y fija peso wet/dry=wet_cost."""
    if metadata["original_uuid"].duplicated().any():
        raise ValueError("Se esperaba exactamente una fila por grabacion.")
    if gold_multiplier <= 0 or wet_cost <= 0:
        raise ValueError("gold_multiplier y wet_cost deben ser positivos.")

    labels = metadata["stage2_target"].to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("El train interno no contiene dry y wet.")
    is_gold = metadata["cough_type_consensus"].eq("gold_expert").to_numpy()
    confidence = np.where(is_gold, gold_multiplier, 1.0).astype(float)

    total = float(len(metadata))
    target_class_weights = {
        0: total / (1.0 + wet_cost),
        1: total * wet_cost / (1.0 + wet_cost),
    }
    weights = np.empty(len(metadata), dtype=float)
    for label in (0, 1):
        mask = labels == label
        weights[mask] = (
            confidence[mask]
            * target_class_weights[label]
            / float(np.sum(confidence[mask]))
        )

    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("Se generaron sample_weight invalidos.")
    if not np.isclose(np.mean(weights), 1.0, atol=1e-10):
        raise RuntimeError("Los pesos no quedaron normalizados.")
    dry_sum = float(weights[labels == 0].sum())
    wet_sum = float(weights[labels == 1].sum())
    if not np.isclose(wet_sum / dry_sum, wet_cost, rtol=1e-10):
        raise RuntimeError("El cociente de pesos wet/dry no coincide con wet_cost.")
    return weights


def constrained_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    default_threshold: float,
) -> float:
    """Maximiza macro-F1 sujeto al recall wet minimo definido en TRAIN."""
    unique_scores = np.unique(np.asarray(scores, dtype=float))
    if len(unique_scores) == 1:
        return default_threshold
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    candidates = np.unique(
        np.concatenate(
            [
                [np.nextafter(unique_scores[0], -np.inf)],
                midpoints,
                [default_threshold],
                [np.nextafter(unique_scores[-1], np.inf)],
            ]
        )
    )

    best_threshold: float | None = None
    best_key: tuple[float, ...] | None = None
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predicted, labels=[0, 1]
        ).ravel()
        wet_recall = tp / (tp + fn) if tp + fn else 0.0
        if wet_recall + 1e-12 < MIN_WET_RECALL_OOF:
            continue
        dry_recall = tn / (tn + fp) if tn + fp else 0.0
        macro_f1 = f1_score(
            y_true,
            predicted,
            average="macro",
            zero_division=0,
        )
        balanced_accuracy = (dry_recall + wet_recall) / 2.0
        key = (
            float(macro_f1),
            float(balanced_accuracy),
            -abs(float(threshold) - default_threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)

    if best_threshold is None:
        raise RuntimeError("No existe umbral que alcance el recall wet minimo.")
    return best_threshold


def make_result(
    data: common.DataView,
    candidate: SearchCandidate,
    scores: np.ndarray,
    elapsed_seconds: float,
) -> SearchResult:
    predictions = common.aggregate_scores_by_recording(data.train_metadata, scores)
    predictions = consensus_base.enrich_predictions(
        predictions, data.train_metadata
    )
    y_true = predictions["y_true"].to_numpy(dtype=int)
    recording_scores = predictions["score"].to_numpy(dtype=float)
    threshold = constrained_threshold(
        y_true,
        recording_scores,
        candidate.spec.default_threshold,
    )
    metrics_tuned = common.binary_metrics(y_true, recording_scores, threshold)
    if metrics_tuned["wet_recall"] + 1e-12 < MIN_WET_RECALL_OOF:
        raise RuntimeError("El umbral ajustado incumple el recall wet minimo.")
    metrics_native = common.binary_metrics(
        y_true,
        recording_scores,
        candidate.spec.default_threshold,
    )
    predictions["y_pred_oof_threshold"] = (
        recording_scores >= threshold
    ).astype(int)
    predictions["y_pred_native_threshold"] = (
        recording_scores >= candidate.spec.default_threshold
    ).astype(int)
    predictions["candidate_key"] = candidate.key
    predictions["pca_whiten"] = candidate.pca_whiten
    predictions["gold_multiplier"] = candidate.gold_multiplier
    predictions["wet_cost"] = candidate.wet_cost
    return SearchResult(
        candidate=candidate,
        threshold=threshold,
        metrics_tuned=metrics_tuned,
        metrics_native=metrics_native,
        predictions=predictions,
        fold_metrics=recording.fold_metrics_at_threshold(predictions, threshold),
        elapsed_seconds=elapsed_seconds,
    )


def weight_diagnostics(
    metadata: pd.DataFrame,
    fold: int,
    gold_multiplier: float,
    wet_cost: float,
) -> list[dict[str, Any]]:
    weights = consensus_and_wet_weights(
        metadata, gold_multiplier, wet_cost
    )
    labels = metadata["stage2_target"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for label, class_name in ((0, "dry"), (1, "wet")):
        for group_name, positions in metadata.groupby(
            "consensus_group", sort=True
        ).indices.items():
            indices = np.asarray(positions, dtype=int)
            indices = indices[labels[indices] == label]
            if len(indices) == 0:
                continue
            rows.append(
                {
                    "fold_excluded": fold,
                    "gold_multiplier": gold_multiplier,
                    "wet_cost": wet_cost,
                    "class_name": class_name,
                    "consensus_group": group_name,
                    "recording_count": len(indices),
                    "weight_sum": float(weights[indices].sum()),
                    "weight_mean": float(weights[indices].mean()),
                }
            )
    return rows


def evaluate_candidates(
    data: common.DataView,
    candidates: list[SearchCandidate],
) -> tuple[list[SearchResult], pd.DataFrame, pd.DataFrame]:
    cache, variance = transformed_folds(data, candidates)
    oof = {
        candidate.key: np.full(len(data.x_train), np.nan, dtype=float)
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
        validation_mask = data.train_metadata["fold"].to_numpy(dtype=int) == fold
        training_mask = ~validation_mask
        metadata_fit = data.train_metadata.loc[training_mask].reset_index(drop=True)
        y_fit = metadata_fit["stage2_target"].to_numpy(dtype=int)
        weights_by_configuration = {}
        for gold_multiplier, wet_cost in weight_configurations:
            configuration = (gold_multiplier, wet_cost)
            weights_by_configuration[configuration] = consensus_and_wet_weights(
                metadata_fit,
                gold_multiplier,
                wet_cost,
            )
            weight_rows.extend(
                weight_diagnostics(
                    metadata_fit,
                    fold,
                    gold_multiplier,
                    wet_cost,
                )
            )

        for candidate in candidates:
            components = int(candidate.spec.pca_components)
            x_fit, x_validation = cache[
                (fold, components, candidate.pca_whiten)
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

    results = []
    for index, candidate in enumerate(candidates, start=1):
        if not np.isfinite(oof[candidate.key]).all():
            raise RuntimeError(f"OOF incompleto para {candidate.key}.")
        result = make_result(
            data,
            candidate,
            oof[candidate.key],
            elapsed[candidate.key],
        )
        results.append(result)
        print(
            f"[{index:03d}/{len(candidates):03d}] {candidate.key} | "
            f"macro-F1={result.metrics_tuned['macro_f1']:.4f} | "
            f"bal-acc={result.metrics_tuned['balanced_accuracy']:.4f} | "
            f"wet-recall={result.metrics_tuned['wet_recall']:.4f} | "
            f"AUC={result.metrics_tuned['roc_auc']:.4f}"
        )
    return results, variance, pd.DataFrame(weight_rows)


def selection_key(result: SearchResult) -> tuple[float, ...]:
    return (
        float(result.metrics_tuned["macro_f1"]),
        float(result.metrics_tuned["balanced_accuracy"]),
        float(result.metrics_tuned["roc_auc"]),
        -float(result.candidate.wet_cost),
        -float(result.candidate.gold_multiplier),
        -float(result.candidate.spec.pca_components),
    )


def ranked_results(results: list[SearchResult]) -> list[SearchResult]:
    eligible = [
        result
        for result in results
        if result.metrics_tuned["wet_recall"] + 1e-12
        >= MIN_WET_RECALL_OOF
    ]
    if not eligible:
        raise RuntimeError("Ningun candidato mantiene el recall wet minimo.")
    return sorted(eligible, key=selection_key, reverse=True)


def candidate_results_frame(results: list[SearchResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        candidate = result.candidate
        rows.append(
            {
                "candidate_key": candidate.key,
                "representation_key": candidate.representation_key,
                "C": candidate.spec.c_value,
                "penalty": PENALTY,
                "pca_components": candidate.spec.pca_components,
                "pca_whiten": candidate.pca_whiten,
                "gold_multiplier": candidate.gold_multiplier,
                "wet_cost": candidate.wet_cost,
                "threshold_oof_constrained": result.threshold,
                "minimum_wet_recall_oof": MIN_WET_RECALL_OOF,
                "elapsed_fit_seconds": result.elapsed_seconds,
                **{
                    f"oof_tuned__{key}": value
                    for key, value in result.metrics_tuned.items()
                },
                **{
                    f"native__{key}": value
                    for key, value in result.metrics_native.items()
                },
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "oof_tuned__macro_f1",
            "oof_tuned__balanced_accuracy",
            "oof_tuned__roc_auc",
        ],
        ascending=False,
    )


def build_final_pipeline(
    data: common.DataView,
    candidate: SearchCandidate,
) -> Pipeline:
    pca = PCA(
        n_components=candidate.spec.pca_components,
        svd_solver="randomized",
        n_oversamples=12,
        iterated_power=4,
        power_iteration_normalizer="auto",
        whiten=candidate.pca_whiten,
        random_state=RANDOM_STATE,
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", pca),
            ("classifier", linear_base.build_classifier(candidate.spec)),
        ]
    )
    weights = consensus_and_wet_weights(
        data.train_metadata,
        candidate.gold_multiplier,
        candidate.wet_cost,
    )
    model.fit(
        data.x_train,
        data.train_metadata["stage2_target"].to_numpy(dtype=int),
        classifier__sample_weight=weights,
    )
    return model


def validation_predictions(
    data: common.DataView,
    model: Pipeline,
    winner: SearchResult,
) -> pd.DataFrame:
    scores = common.model_scores(model, data.x_validation)
    predictions = common.aggregate_scores_by_recording(
        data.validation_metadata, scores
    )
    predictions = consensus_base.enrich_predictions(
        predictions, data.validation_metadata
    )
    predictions["y_pred_oof_threshold"] = (
        predictions["score"].to_numpy(dtype=float) >= winner.threshold
    ).astype(int)
    predictions["y_pred_native_threshold"] = (
        predictions["score"].to_numpy(dtype=float)
        >= winner.candidate.spec.default_threshold
    ).astype(int)
    predictions["candidate_key"] = winner.candidate.key
    predictions["pca_whiten"] = winner.candidate.pca_whiten
    predictions["gold_multiplier"] = winner.candidate.gold_multiplier
    predictions["wet_cost"] = winner.candidate.wet_cost
    return predictions


def save_candidate_subgroups(
    results: list[SearchResult],
    output_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        for row in consensus_base.subgroup_metric_rows(
            "train_oof", result.predictions, result.threshold
        ):
            rows.append(
                {
                    "candidate_key": result.candidate.key,
                    "C": result.candidate.spec.c_value,
                    "pca_components": result.candidate.spec.pca_components,
                    "pca_whiten": result.candidate.pca_whiten,
                    "gold_multiplier": result.candidate.gold_multiplier,
                    "wet_cost": result.candidate.wet_cost,
                    **row,
                }
            )
    pd.DataFrame(rows).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )


def train(
    data: common.DataView,
    extraction_configuration: pd.DataFrame,
    preset: str,
    quick: bool,
) -> None:
    run_name = "quick" if quick else "full"
    result_dir = RESULTS_ROOT / preset / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("FASE A - REGULARIZACION Y REPRESENTACION")
    print("=" * 78)
    phase_a_candidates = phase_a_candidate_grid(quick)
    print(f"Candidatos fase A: {len(phase_a_candidates)}")
    phase_a_results, phase_a_variance, phase_a_weights = evaluate_candidates(
        data, phase_a_candidates
    )
    phase_a_ranking = ranked_results(phase_a_results)
    top_phase_a = phase_a_ranking[:3]

    candidate_results_frame(phase_a_results).to_csv(
        result_dir / "phase_a_candidate_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidate_results_frame(top_phase_a).to_csv(
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
        result_dir / "phase_a_weight_diagnostics_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nTop 3 de la fase A:")
    for result in top_phase_a:
        print(
            f"  {result.candidate.representation_key} | "
            f"macro-F1={result.metrics_tuned['macro_f1']:.4f} | "
            f"wet-recall={result.metrics_tuned['wet_recall']:.4f}"
        )

    print("\n" + "=" * 78)
    print("FASE B - PESO GOLD Y COSTE WET")
    print("=" * 78)
    phase_b_candidates = phase_b_candidate_grid(top_phase_a, quick)
    print(f"Candidatos fase B: {len(phase_b_candidates)}")
    phase_b_results, phase_b_variance, phase_b_weights = evaluate_candidates(
        data, phase_b_candidates
    )
    phase_b_ranking = ranked_results(phase_b_results)
    winner = phase_b_ranking[0]

    candidate_results_frame(phase_b_results).to_csv(
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
        result_dir / "phase_b_weight_diagnostics_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_candidate_subgroups(
        phase_b_results,
        result_dir / "phase_b_metrics_by_consensus_subgroup_oof.csv",
    )

    winner.predictions.to_csv(
        result_dir / "best_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    winner.fold_metrics.to_csv(
        result_dir / "best_cv_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subgroup_rows = consensus_base.subgroup_metric_rows(
        "train_oof", winner.predictions, winner.threshold
    )

    validation = None
    model_size_kb = np.nan
    learned_float_count = 0
    if not quick:
        print("\nAjustando LR final exclusivamente con todo TRAIN...")
        model = build_final_pipeline(data, winner.candidate)
        validation = validation_predictions(data, model, winner)
        validation.to_csv(
            result_dir / "validation_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        subgroup_rows.extend(
            consensus_base.subgroup_metric_rows(
                "validation", validation, winner.threshold
            )
        )

        model_dir = MODELS_ROOT / preset / run_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "logistic_regression_model.joblib"
        package = {
            "pipeline": model,
            "experiment": EXPERIMENT_KEY,
            "preset": preset,
            "candidate_key": winner.candidate.key,
            "C": winner.candidate.spec.c_value,
            "penalty": PENALTY,
            "pca_components": winner.candidate.spec.pca_components,
            "pca_whiten": winner.candidate.pca_whiten,
            "gold_multiplier": winner.candidate.gold_multiplier,
            "wet_cost": winner.candidate.wet_cost,
            "threshold": winner.threshold,
            "minimum_wet_recall_oof": MIN_WET_RECALL_OOF,
            "label_mapping": common.LABEL_TO_NAME,
            "recording_feature_names": data.feature_names,
            "extraction_configuration": extraction_configuration.iloc[0].to_dict(),
            "validation_used_for_selection": False,
            "test_processed": False,
            "random_state": RANDOM_STATE,
        }
        joblib.dump(package, model_path, compress=3)
        model_size_kb = model_path.stat().st_size / 1024.0
        learned_float_count = linear_base.learned_float_count(model)

        graph_dir = GRAPHS_ROOT / preset / run_name
        graph_dir.mkdir(parents=True, exist_ok=True)
        common.create_validation_graph(
            validation,
            winner.threshold,
            EXPERIMENT_KEY,
            winner.candidate.spec,
            graph_dir / "validation_wst_lr_two_phase.png",
        )

    pd.DataFrame(subgroup_rows).to_csv(
        result_dir / "metrics_by_consensus_subgroup.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_rows = [
        {
            "dataset": "train_oof",
            "candidate_key": winner.candidate.key,
            "threshold": winner.threshold,
            "minimum_wet_recall_oof": MIN_WET_RECALL_OOF,
            "model_size_kb_joblib": model_size_kb,
            "learned_inference_float_count": learned_float_count,
            **winner.metrics_tuned,
        }
    ]
    if validation is not None:
        metric_rows.append(
            {
                "dataset": "validation",
                "candidate_key": winner.candidate.key,
                "threshold": winner.threshold,
                "minimum_wet_recall_oof": MIN_WET_RECALL_OOF,
                "model_size_kb_joblib": model_size_kb,
                "learned_inference_float_count": learned_float_count,
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
                "phase_a_candidate_count": len(phase_a_candidates),
                "phase_b_candidate_count": len(phase_b_candidates),
                "phase_a_C_values": "|".join(map(str, PHASE_A_C_VALUES)),
                "phase_a_PCA": "|".join(map(str, PHASE_A_PCA_OPTIONS)),
                "phase_a_whiten": "False|True",
                "phase_b_gold": "|".join(map(str, PHASE_B_GOLD_MULTIPLIERS)),
                "phase_b_wet_cost": "|".join(map(str, PHASE_B_WET_COSTS)),
                "penalty": PENALTY,
                "selection": "max_oof_macro_f1_subject_to_min_wet_recall",
                "minimum_wet_recall_oof": MIN_WET_RECALL_OOF,
                "validation_used_for_selection": False,
                "test_processed": False,
                "winner": winner.candidate.key,
            }
        ]
    ).to_csv(
        result_dir / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 78)
    print("RESULTADO WST RECORDING + LR - BUSQUEDA EN DOS FASES")
    print("=" * 78)
    print(f"Ganador OOF: {winner.candidate.key}")
    print(f"Umbral OOF restringido: {winner.threshold:.6f}")
    print(f"Macro-F1 OOF: {winner.metrics_tuned['macro_f1']:.4f}")
    print(f"Recall wet OOF: {winner.metrics_tuned['wet_recall']:.4f}")
    if validation is None:
        print("VALIDATION no se ha evaluado en modo --quick.")
    else:
        validation_metrics = metric_rows[-1]
        print(f"Macro-F1 validation: {validation_metrics['macro_f1']:.4f}")
        print(
            "Recalls validation dry/wet: "
            f"{validation_metrics['dry_recall']:.4f} / "
            f"{validation_metrics['wet_recall']:.4f}"
        )
        print(f"Modelo: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print("TEST permanece reservado.")


def print_check(data: common.DataView, quick: bool) -> None:
    phase_a_count = len(phase_a_candidate_grid(quick))
    phase_b_gold = QUICK_GOLD_MULTIPLIERS if quick else PHASE_B_GOLD_MULTIPLIERS
    phase_b_wet = QUICK_WET_COSTS if quick else PHASE_B_WET_COSTS
    expected_phase_b = 3 * len(phase_b_gold) * len(phase_b_wet)
    print("=" * 78)
    print("CHECK - WST RECORDING + LR - BUSQUEDA EN DOS FASES")
    print("=" * 78)
    print(f"X TRAIN/VALIDATION: {data.x_train.shape} / {data.x_validation.shape}")
    print(f"Fase A: {phase_a_count} candidatos.")
    print(f"Fase B: {expected_phase_b} candidatos tras seleccionar top 3.")
    print(f"Penalizacion fija: {PENALTY}.")
    print(f"Recall wet OOF minimo: {MIN_WET_RECALL_OOF:.6f}.")
    print("Scaler, PCA, whitening y pesos se ajustan dentro de cada fold.")
    print("VALIDATION no selecciona. TEST no se procesa.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Busqueda LR WST recording en dos fases."
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
    print("STAGE 2 - WST RECORDING + LR - BUSQUEDA EN DOS FASES")
    print("=" * 78)
    print(f"Preset: {args.preset} | Accion: {args.action} | Quick: {args.quick}")
    print("TEST no sera leido ni procesado.")
    event_data, extraction_configuration = linear_base.FEATURE_LOADER(args.preset)
    recording_data = linear_base.build_recording_data(event_data)
    data = consensus_base.attach_vote_metadata(recording_data)
    print_check(data, args.quick)
    if args.action == "check":
        print("\nComprobacion completada. No se entreno ningun modelo.")
        return
    train(data, extraction_configuration, args.preset, args.quick)


if __name__ == "__main__":
    main()
