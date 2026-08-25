"""Stage 2 dry/wet: WST recording + LR ponderada por consenso experto.

Este experimento conserva el pipeline WST recording mean+std+max existente y
anade una unica variable: las grabaciones ``gold_expert`` reciben un peso
relativo 1, 1.5, 2 o 3 frente a las ``weak_expert``. La ponderacion se calcula
exclusivamente con el subconjunto de ajuste de cada fold y se normaliza dentro
de cada clase, por lo que dry y wet mantienen el mismo peso total.

En modo completo se buscan conjuntamente C, PCA y multiplicador usando solo
OOF de TRAIN. El ganador se congela y se evalua una vez en VALIDATION. TEST no
se lee ni se procesa. El script original de WST + LR permanece intacto.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import analyze_stage2_expert_votes as vote_audit
import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_linear as baseline
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (
    SCRIPT_DIR
    / "results_stage2_dry_wet_wst_recording_consensus_weighted_lr"
)
MODELS_ROOT = (
    SCRIPT_DIR
    / "models_stage2_dry_wet_wst_recording_consensus_weighted_lr"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wst_recording_consensus_weighted_lr"
)

RANDOM_STATE = baseline.RANDOM_STATE
PRESETS = baseline.PRESETS
DEFAULT_PRESET = baseline.DEFAULT_PRESET
GOLD_MULTIPLIERS = (1.0, 1.5, 2.0, 3.0)
QUICK_SPEC = common.CandidateSpec("logistic_regression", 0.001, 128)
EXPERIMENT_KEY = "wst_recording_consensus_weighted_lr"

VOTE_METADATA_COLUMNS = (
    "expert_evaluator_count",
    "dry_vote_count",
    "wet_vote_count",
    "unknown_vote_count",
    "maximum_agreement_count",
    "agreement_ratio",
    "vote_pattern",
    "consensus_reason",
)


def number_key(value: float) -> str:
    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class WeightedCandidate:
    spec: common.CandidateSpec
    gold_multiplier: float

    @property
    def key(self) -> str:
        return f"{self.spec.key}__gold{number_key(self.gold_multiplier)}"


@dataclass
class WeightedResult:
    candidate: WeightedCandidate
    threshold: float
    metrics_tuned: dict[str, float | int]
    metrics_native: dict[str, float | int]
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    elapsed_seconds: float


def candidate_grid(quick: bool) -> list[WeightedCandidate]:
    if quick:
        specs = [QUICK_SPEC]
    else:
        specs = [
            common.CandidateSpec("logistic_regression", c_value, pca_components)
            for c_value in baseline.FULL_C_VALUES["logistic_regression"]
            for pca_components in baseline.FULL_PCA_OPTIONS
        ]
    return [
        WeightedCandidate(spec, multiplier)
        for multiplier in GOLD_MULTIPLIERS
        for spec in specs
    ]


def attach_vote_metadata(data: common.DataView) -> common.DataView:
    """Anade los votos solo a UUID presentes en TRAIN/VALIDATION."""
    votes = vote_audit.load_original_votes().rename(columns={"uuid": "original_uuid"})
    vote_columns = ["original_uuid", *VOTE_METADATA_COLUMNS]
    votes = votes[vote_columns].copy()
    votes["original_uuid"] = votes["original_uuid"].astype(str).str.strip()

    def enrich(metadata: pd.DataFrame, split_name: str) -> pd.DataFrame:
        result = metadata.copy()
        result["original_uuid"] = result["original_uuid"].astype(str).str.strip()
        result = result.merge(
            votes,
            on="original_uuid",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if (result["_merge"] != "both").any():
            missing = result.loc[
                result["_merge"] != "both", "original_uuid"
            ].head(10)
            raise ValueError(
                f"{split_name}: UUID sin votos originales: {missing.tolist()}"
            )
        result = result.drop(columns="_merge")
        if result["expert_evaluator_count"].isna().any():
            raise ValueError(f"{split_name}: faltan recuentos expertos.")
        if not result["cough_type_consensus"].isin(
            ["gold_expert", "weak_expert"]
        ).all():
            raise ValueError(
                f"{split_name}: Stage 2 contiene consensos no permitidos."
            )

        result["consensus_group"] = "weak_other"
        weak = result["cough_type_consensus"].eq("weak_expert")
        result.loc[
            weak & result["expert_evaluator_count"].eq(1),
            "consensus_group",
        ] = "weak_1_vote"
        result.loc[
            weak
            & result["consensus_reason"].eq("4_evaluators_2_vs_1_vs_1"),
            "consensus_group",
        ] = "weak_4_eval_2_1_1"
        gold = result["cough_type_consensus"].eq("gold_expert")
        result.loc[
            gold & result["maximum_agreement_count"].eq(3),
            "consensus_group",
        ] = "gold_3_of_4"
        result.loc[
            gold & result["maximum_agreement_count"].eq(4),
            "consensus_group",
        ] = "gold_4_of_4"
        invalid_gold = gold & ~result["consensus_group"].isin(
            ["gold_3_of_4", "gold_4_of_4"]
        )
        if invalid_gold.any():
            raise ValueError(f"{split_name}: gold sin acuerdo 3/4 o 4/4.")
        if (weak & result["consensus_group"].eq("weak_other")).any():
            unexpected = result.loc[
                weak & result["consensus_group"].eq("weak_other"),
                "consensus_reason",
            ].value_counts()
            raise ValueError(
                f"{split_name}: patron weak no contemplado: "
                f"{unexpected.to_dict()}"
            )
        return result

    return common.DataView(
        experiment=f"{data.experiment}_consensus_weighted",
        x_train=data.x_train,
        train_metadata=enrich(data.train_metadata, "train"),
        x_validation=data.x_validation,
        validation_metadata=enrich(data.validation_metadata, "validation"),
        feature_names=data.feature_names,
    )


def consensus_balanced_weights(
    metadata: pd.DataFrame,
    gold_multiplier: float,
) -> np.ndarray:
    """Mantiene peso total dry=wet y pondera gold dentro de cada clase."""
    if metadata["original_uuid"].duplicated().any():
        raise ValueError("Se esperaba exactamente una fila por grabacion.")
    if gold_multiplier <= 0:
        raise ValueError("El multiplicador gold debe ser positivo.")

    labels = metadata["stage2_target"].to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("El subconjunto interno no contiene dry y wet.")
    is_gold = metadata["cough_type_consensus"].eq("gold_expert").to_numpy()
    confidence = np.where(is_gold, gold_multiplier, 1.0).astype(float)
    weights = np.empty(len(metadata), dtype=float)
    target_class_weight = len(metadata) / 2.0
    for label in (0, 1):
        mask = labels == label
        weights[mask] = (
            confidence[mask]
            * target_class_weight
            / float(np.sum(confidence[mask]))
        )

    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("Se generaron sample_weight invalidos.")
    if not np.isclose(np.mean(weights), 1.0, atol=1e-10):
        raise RuntimeError("Los pesos no quedaron normalizados.")
    class_sums = [float(weights[labels == label].sum()) for label in (0, 1)]
    if not np.isclose(class_sums[0], class_sums[1], rtol=1e-10):
        raise RuntimeError("Los pesos dejaron de balancear dry y wet.")
    return weights


def enrich_predictions(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    extra_columns = [
        "original_uuid",
        "consensus_group",
        *VOTE_METADATA_COLUMNS,
    ]
    extra = metadata[extra_columns].drop_duplicates("original_uuid")
    result = predictions.merge(
        extra,
        on="original_uuid",
        how="left",
        validate="one_to_one",
    )
    if result["consensus_group"].isna().any():
        raise RuntimeError("No se pudieron asociar votos a las predicciones.")
    return result


def make_result(
    data: common.DataView,
    candidate: WeightedCandidate,
    scores: np.ndarray,
    elapsed_seconds: float,
) -> WeightedResult:
    predictions = common.aggregate_scores_by_recording(data.train_metadata, scores)
    predictions = enrich_predictions(predictions, data.train_metadata)
    y_true = predictions["y_true"].to_numpy(dtype=int)
    recording_scores = predictions["score"].to_numpy(dtype=float)
    threshold = common.tune_threshold(
        y_true,
        recording_scores,
        candidate.spec.default_threshold,
    )
    metrics_tuned = common.binary_metrics(y_true, recording_scores, threshold)
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
    predictions["gold_multiplier"] = candidate.gold_multiplier
    return WeightedResult(
        candidate=candidate,
        threshold=threshold,
        metrics_tuned=metrics_tuned,
        metrics_native=metrics_native,
        predictions=predictions,
        fold_metrics=recording.fold_metrics_at_threshold(predictions, threshold),
        elapsed_seconds=elapsed_seconds,
    )


def weight_diagnostic_rows(
    metadata: pd.DataFrame,
    fold: int | str,
    multiplier: float,
) -> list[dict[str, Any]]:
    weights = consensus_balanced_weights(metadata, multiplier)
    rows: list[dict[str, Any]] = []
    labels = metadata["stage2_target"].to_numpy(dtype=int)
    for label, class_name in ((0, "dry"), (1, "wet")):
        for group_name, group in metadata.groupby("consensus_group", sort=True):
            mask = (labels == label) & metadata.index.isin(group.index)
            if not mask.any():
                continue
            rows.append(
                {
                    "fold_excluded": fold,
                    "gold_multiplier": multiplier,
                    "class_name": class_name,
                    "consensus_group": group_name,
                    "recording_count": int(mask.sum()),
                    "weight_sum": float(weights[mask].sum()),
                    "weight_mean": float(weights[mask].mean()),
                }
            )
    return rows


def evaluate_candidates(
    data: common.DataView,
    candidates: list[WeightedCandidate],
) -> tuple[list[WeightedResult], pd.DataFrame, pd.DataFrame]:
    specs = list(dict.fromkeys(candidate.spec for candidate in candidates))
    pca_options = tuple(dict.fromkeys(spec.pca_components for spec in specs))
    cache, variance = baseline.transformed_folds(data, pca_options)
    oof = {
        candidate.key: np.full(len(data.x_train), np.nan, dtype=float)
        for candidate in candidates
    }
    elapsed = {candidate.key: 0.0 for candidate in candidates}
    diagnostic_rows: list[dict[str, Any]] = []

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = data.train_metadata["fold"].to_numpy(dtype=int) == fold
        training_mask = ~validation_mask
        metadata_fit = data.train_metadata.loc[training_mask].reset_index(drop=True)
        y_fit = metadata_fit["stage2_target"].to_numpy(dtype=int)
        weights_by_multiplier = {}
        for multiplier in GOLD_MULTIPLIERS:
            weights_by_multiplier[multiplier] = consensus_balanced_weights(
                metadata_fit, multiplier
            )
            diagnostic_rows.extend(
                weight_diagnostic_rows(metadata_fit, fold, multiplier)
            )

        for candidate in candidates:
            x_fit, x_validation = cache[(fold, candidate.spec.pca_components)]
            classifier = baseline.build_classifier(candidate.spec)
            started = time.perf_counter()
            classifier.fit(
                x_fit,
                y_fit,
                sample_weight=weights_by_multiplier[candidate.gold_multiplier],
            )
            oof[candidate.key][validation_mask] = baseline.classifier_scores(
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
            f"AUC={result.metrics_tuned['roc_auc']:.4f}"
        )
    return results, variance, pd.DataFrame(diagnostic_rows)


def selection_key(result: WeightedResult) -> tuple[float, ...]:
    dimension = (
        baseline.EXPECTED_RECORDING_FEATURE_COUNT
        if result.candidate.spec.pca_components is None
        else result.candidate.spec.pca_components
    )
    return (
        float(result.metrics_tuned["macro_f1"]),
        float(result.metrics_tuned["balanced_accuracy"]),
        float(result.metrics_tuned["roc_auc"]),
        -float(result.candidate.gold_multiplier),
        -float(dimension),
    )


def candidate_results_frame(results: list[WeightedResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        candidate = result.candidate
        spec = candidate.spec
        rows.append(
            {
                "candidate_key": candidate.key,
                "gold_multiplier": candidate.gold_multiplier,
                "C": spec.c_value,
                "pca_components": spec.pca_components,
                "output_dimension": (
                    baseline.EXPECTED_RECORDING_FEATURE_COUNT
                    if spec.pca_components is None
                    else spec.pca_components
                ),
                "threshold_oof": result.threshold,
                "native_threshold": spec.default_threshold,
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


def best_by_multiplier_frame(results: list[WeightedResult]) -> pd.DataFrame:
    rows = []
    for multiplier in GOLD_MULTIPLIERS:
        winner = max(
            [r for r in results if r.candidate.gold_multiplier == multiplier],
            key=selection_key,
        )
        rows.append(
            {
                "gold_multiplier": multiplier,
                "best_candidate_key": winner.candidate.key,
                "threshold_oof": winner.threshold,
                **winner.metrics_tuned,
            }
        )
    return pd.DataFrame(rows)


def build_final_pipeline(
    data: common.DataView,
    candidate: WeightedCandidate,
) -> Pipeline:
    spec = candidate.spec
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
    steps.append(("classifier", baseline.build_classifier(spec)))
    model = Pipeline(steps)
    weights = consensus_balanced_weights(
        data.train_metadata,
        candidate.gold_multiplier,
    )
    model.fit(
        data.x_train,
        data.train_metadata["stage2_target"].to_numpy(dtype=int),
        classifier__sample_weight=weights,
    )
    return model


def safe_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    dry_total = int(np.sum(y_true == 0))
    wet_total = int(np.sum(y_true == 1))
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(
            np.mean(
                [
                    tn / dry_total if dry_total else np.nan,
                    tp / wet_total if wet_total else np.nan,
                ]
            )
        ) if dry_total and wet_total else np.nan,
        "macro_f1": float(
            f1_score(
                y_true,
                predicted,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "dry_precision": float(
            precision_score(y_true, predicted, pos_label=0, zero_division=0)
        ),
        "wet_precision": float(
            precision_score(y_true, predicted, pos_label=1, zero_division=0)
        ),
        "dry_recall": float(
            recall_score(y_true, predicted, pos_label=0, zero_division=0)
        ) if dry_total else np.nan,
        "wet_recall": float(
            recall_score(y_true, predicted, pos_label=1, zero_division=0)
        ) if wet_total else np.nan,
        "roc_auc": float(roc_auc_score(y_true, scores))
        if dry_total and wet_total
        else np.nan,
        "average_precision_wet": float(average_precision_score(y_true, scores))
        if wet_total
        else np.nan,
        "tn_dry_correct": int(tn),
        "fp_dry_as_wet": int(fp),
        "fn_wet_as_dry": int(fn),
        "tp_wet_correct": int(tp),
    }


def subgroup_metric_rows(
    dataset: str,
    predictions: pd.DataFrame,
    threshold: float,
) -> list[dict[str, Any]]:
    selectors = {
        "all": np.ones(len(predictions), dtype=bool),
        "weak_all": predictions["cough_type_consensus"].eq("weak_expert"),
        "gold_all": predictions["cough_type_consensus"].eq("gold_expert"),
        "gold_3_of_4": predictions["consensus_group"].eq("gold_3_of_4"),
        "gold_4_of_4": predictions["consensus_group"].eq("gold_4_of_4"),
    }
    rows = []
    for subgroup, selector in selectors.items():
        subset = predictions.loc[np.asarray(selector)]
        if subset.empty:
            continue
        y_true = subset["y_true"].to_numpy(dtype=int)
        scores = subset["score"].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "subgroup": subgroup,
                "recording_count": len(subset),
                "dry_count": int(np.sum(y_true == 0)),
                "wet_count": int(np.sum(y_true == 1)),
                "threshold": threshold,
                **safe_metrics(y_true, scores, threshold),
            }
        )
    return rows


def prepare_validation_predictions(
    data: common.DataView,
    model: Pipeline,
    winner: WeightedResult,
) -> pd.DataFrame:
    scores = common.model_scores(model, data.x_validation)
    predictions = common.aggregate_scores_by_recording(
        data.validation_metadata,
        scores,
    )
    predictions = enrich_predictions(predictions, data.validation_metadata)
    predictions["y_pred_oof_threshold"] = (
        predictions["score"].to_numpy(dtype=float) >= winner.threshold
    ).astype(int)
    predictions["y_pred_native_threshold"] = (
        predictions["score"].to_numpy(dtype=float)
        >= winner.candidate.spec.default_threshold
    ).astype(int)
    predictions["candidate_key"] = winner.candidate.key
    predictions["gold_multiplier"] = winner.candidate.gold_multiplier
    return predictions


def train(
    data: common.DataView,
    extraction_configuration: pd.DataFrame,
    preset: str,
    quick: bool,
) -> None:
    run_name = "quick" if quick else "full"
    result_dir = RESULTS_ROOT / preset / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_grid(quick)

    print("\n" + "=" * 78)
    print("CV - WST RECORDING + LR PONDERADA POR CONSENSO")
    print("=" * 78)
    print(f"Candidatos: {len(candidates)}")
    results, variance, weight_diagnostics = evaluate_candidates(data, candidates)
    candidate_results_frame(results).to_csv(
        result_dir / "candidate_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_by_multiplier_frame(results).to_csv(
        result_dir / "gold_multiplier_comparison_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )
    variance.to_csv(
        result_dir / "pca_explained_variance_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    weight_diagnostics.to_csv(
        result_dir / "consensus_weight_diagnostics_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    winner = max(results, key=selection_key)
    all_candidate_subgroups: list[dict[str, Any]] = []
    for result in results:
        for row in subgroup_metric_rows(
            "train_oof",
            result.predictions,
            result.threshold,
        ):
            all_candidate_subgroups.append(
                {
                    "candidate_key": result.candidate.key,
                    "gold_multiplier": result.candidate.gold_multiplier,
                    "C": result.candidate.spec.c_value,
                    "pca_components": result.candidate.spec.pca_components,
                    **row,
                }
            )
    pd.DataFrame(all_candidate_subgroups).to_csv(
        result_dir / "candidate_metrics_by_consensus_subgroup_oof.csv",
        index=False,
        encoding="utf-8-sig",
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
    subgroup_rows = subgroup_metric_rows(
        "train_oof",
        winner.predictions,
        winner.threshold,
    )

    validation_predictions = None
    model_size_kb = np.nan
    learned_float_count = 0
    if not quick:
        print("\nAjustando LR final exclusivamente con todo TRAIN...")
        model = build_final_pipeline(data, winner.candidate)
        validation_predictions = prepare_validation_predictions(data, model, winner)
        validation_predictions.to_csv(
            result_dir / "validation_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        subgroup_rows.extend(
            subgroup_metric_rows(
                "validation",
                validation_predictions,
                winner.threshold,
            )
        )

        model_dir = MODELS_ROOT / preset / run_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "logistic_regression_model.joblib"
        package = {
            "pipeline": model,
            "experiment": EXPERIMENT_KEY,
            "preset": preset,
            "pooling_key": baseline.RECORDING_POOLING.key,
            "candidate_key": winner.candidate.key,
            "C": winner.candidate.spec.c_value,
            "pca_components": winner.candidate.spec.pca_components,
            "gold_multiplier": winner.candidate.gold_multiplier,
            "weak_multiplier": 1.0,
            "class_balance": "equal total weight per class after consensus",
            "threshold": winner.threshold,
            "native_threshold": winner.candidate.spec.default_threshold,
            "label_mapping": common.LABEL_TO_NAME,
            "recording_feature_names": data.feature_names,
            "extraction_configuration": extraction_configuration.iloc[0].to_dict(),
            "test_processed": False,
            "random_state": RANDOM_STATE,
        }
        joblib.dump(package, model_path, compress=3)
        model_size_kb = model_path.stat().st_size / 1024.0
        learned_float_count = baseline.learned_float_count(model)

        graph_dir = GRAPHS_ROOT / preset / run_name
        graph_dir.mkdir(parents=True, exist_ok=True)
        common.create_validation_graph(
            validation_predictions,
            winner.threshold,
            EXPERIMENT_KEY,
            winner.candidate.spec,
            graph_dir / "validation_consensus_weighted_lr.png",
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
            "gold_multiplier": winner.candidate.gold_multiplier,
            "threshold": winner.threshold,
            "model_size_kb_joblib": model_size_kb,
            "learned_inference_float_count": learned_float_count,
            **winner.metrics_tuned,
        }
    ]
    if validation_predictions is not None:
        metric_rows.append(
            {
                "dataset": "validation",
                "candidate_key": winner.candidate.key,
                "gold_multiplier": winner.candidate.gold_multiplier,
                "threshold": winner.threshold,
                "model_size_kb_joblib": model_size_kb,
                "learned_inference_float_count": learned_float_count,
                **common.binary_metrics(
                    validation_predictions["y_true"].to_numpy(dtype=int),
                    validation_predictions["score"].to_numpy(dtype=float),
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
                "input_feature_count": data.x_train.shape[1],
                "pooling_key": baseline.RECORDING_POOLING.key,
                "gold_multipliers": "|".join(map(str, GOLD_MULTIPLIERS)),
                "weak_multiplier": 1.0,
                "class_balance": "equal total weight per class after consensus",
                "standard_scaler_used": True,
                "selection_metric": "train_oof_macro_f1",
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
    print("RESULTADO WST RECORDING + LR PONDERADA POR CONSENSO")
    print("=" * 78)
    print(f"Ganador OOF: {winner.candidate.key}")
    print(f"Multiplicador gold: {winner.candidate.gold_multiplier:g}")
    print(f"Umbral OOF congelado: {winner.threshold:.6f}")
    print(f"Macro-F1 OOF: {winner.metrics_tuned['macro_f1']:.4f}")
    if validation_predictions is None:
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
    print("=" * 78)
    print("CHECK - WST RECORDING + LR PONDERADA POR CONSENSO")
    print("=" * 78)
    print(f"X TRAIN/VALIDATION: {data.x_train.shape} / {data.x_validation.shape}")
    for split_name, metadata in (
        ("TRAIN", data.train_metadata),
        ("VALIDATION", data.validation_metadata),
    ):
        table = pd.crosstab(
            metadata["cough_type"],
            metadata["consensus_group"],
        )
        print(f"\n{split_name} por clase y consenso:")
        print(table.to_string())
    print(f"Candidatos: {len(candidate_grid(quick))}")
    print(f"Multiplicadores gold: {GOLD_MULTIPLIERS}; weak=1.")
    print("El peso total dry/wet se iguala dentro de cada train interno.")
    print("VALIDATION no selecciona hiperparametros. TEST no se procesa.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WST recording + LR con pesos por consenso experto."
    )
    parser.add_argument("--action", choices=["check", "train"], default="check")
    parser.add_argument("--preset", choices=PRESETS, default=DEFAULT_PRESET)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Compara los cuatro pesos con la LR baseline; no usa VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST RECORDING + LR PONDERADA POR CONSENSO")
    print("=" * 78)
    print(f"Preset: {args.preset} | Accion: {args.action} | Quick: {args.quick}")
    print("TEST no sera leido ni procesado.")
    event_data, extraction_configuration = baseline.FEATURE_LOADER(args.preset)
    data = attach_vote_metadata(baseline.build_recording_data(event_data))
    print_check(data, args.quick)
    if args.action == "check":
        print("\nComprobacion completada. No se entreno ningun modelo.")
        return
    train(data, extraction_configuration, args.preset, args.quick)


if __name__ == "__main__":
    main()
