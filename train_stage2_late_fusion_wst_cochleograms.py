"""Late fusion ponderada: WST recording LR + cocleograma event RF/XGBoost.

Lee exclusivamente predicciones ya existentes y alineadas por original_uuid:

    p_fusion = alpha * p_wst + (1 - alpha) * p_coch

El candidato de cocleograma, alpha y umbral se seleccionan con OOF de TRAIN.
Se guarda ademas una sensibilidad leave-one-fold-out de la seleccion. La
configuracion final se congela y se aplica una vez a VALIDATION. TEST no se
lee ni se procesa.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import train_stage2_dry_wet_cochleograms as common

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_stage2_late_fusion_wst_cochleograms" / "full"
GRAPHS_DIR = (
    ROOT
    / "graphs_results_stage2_dry_wet_random"
    / "late_fusion_wst_cochleograms"
    / "full"
)
MODELS_DIR = ROOT / "models_stage2_late_fusion_wst_cochleograms" / "full"

ALPHAS = tuple(float(value) for value in np.round(np.arange(0.0, 1.0001, 0.05), 2))
DEFAULT_THRESHOLD = 0.5
EXPERIMENT_KEY = "late_fusion_wst_cochleograms_event"
DISPLAY_TITLE = "STAGE 2 — LATE FUSION WST-LR + COCLEOGRAMA EVENT RF/XGBOOST"
CHECK_TITLE = "CHECK — LATE FUSION WST-LR + COCLEOGRAMA EVENT RF/XGBOOST"
PARSER_DESCRIPTION = (
    "Late fusion WST recording LR + cocleograma event RF/XGBoost."
)
VALIDATION_GRAPH_FILENAME = "validation_late_fusion_oof_threshold.png"
SECONDARY_DISPLAY_LABEL = "cocleograma"
CANDIDATE_COMPARISON_FILENAME = "coch_candidate_comparison.csv"
RESULT_TITLE = "RESULTADO LATE FUSION WST + COCLEOGRAMA"
SECONDARY_WEIGHT_LABEL = "peso cocleograma"

WST_RESULT_DIR = (
    ROOT
    / "results_stage2_dry_wet_wavelet_scattering_recording_linear"
    / "paper_q8_q1_t500_full"
    / "full"
    / "logistic_regression"
)
WST_MODEL_PATH = (
    ROOT
    / "models_stage2_dry_wet_wavelet_scattering_recording_linear"
    / "paper_q8_q1_t500_full"
    / "full"
    / "logistic_regression_model.joblib"
)


@dataclass(frozen=True)
class CochCandidate:
    key: str
    display_name: str
    result_dir: Path
    model_path: Path


COCH_CANDIDATES = (
    CochCandidate(
        key="coch_raw_pca64_rf",
        display_name="Cocleograma raw PCA64 + RF",
        result_dir=(
            ROOT
            / "results_stage2_dry_wet_cochleograms_raw_pca_rf"
            / "paper64"
            / "full"
        ),
        model_path=(
            ROOT
            / "models_stage2_dry_wet_cochleograms_raw_pca_rf"
            / "paper64"
            / "full"
            / "raw_pca64_rf_event_model.joblib"
        ),
    ),
    CochCandidate(
        key="coch_raw_pca64_xgb",
        display_name="Cocleograma raw PCA64 + XGBoost",
        result_dir=(
            ROOT
            / "results_stage2_dry_wet_cochleograms_raw_pca_xgb"
            / "paper64"
            / "full"
        ),
        model_path=(
            ROOT
            / "models_stage2_dry_wet_cochleograms_raw_pca_xgb"
            / "paper64"
            / "full"
            / "raw_pca64_xgb_event_model.joblib"
        ),
    ),
)


@dataclass
class SearchResult:
    coch_key: str
    alpha: float
    threshold: float
    metrics_tuned: dict[str, float | int]
    metrics_fixed: dict[str, float | int]

    @property
    def key(self) -> str:
        alpha_name = f"{self.alpha:.2f}".replace(".", "p")
        effective = "none" if np.isclose(self.alpha, 1.0) else self.coch_key
        return f"late_fusion__wst_lr__{effective}__alpha{alpha_name}"


@dataclass(frozen=True)
class GraphSpec:
    key: str


def prediction_path(result_dir: Path, split: str) -> Path:
    filename = (
        "best_oof_predictions.csv"
        if split == "train_oof"
        else "validation_predictions.csv"
    )
    return result_dir / filename


def require_inputs() -> None:
    paths = [
        prediction_path(WST_RESULT_DIR, "train_oof"),
        prediction_path(WST_RESULT_DIR, "validation"),
        WST_RESULT_DIR / "metrics_summary.csv",
        WST_MODEL_PATH,
    ]
    for candidate in COCH_CANDIDATES:
        paths.extend(
            [
                prediction_path(candidate.result_dir, "train_oof"),
                prediction_path(candidate.result_dir, "validation"),
                candidate.result_dir / "metrics_summary.csv",
                candidate.model_path,
            ]
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan entradas para la fusion:\n" + "\n".join(missing))


def compact_predictions(path: Path, score_name: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "original_uuid",
        "y_true",
        "cough_type",
        "cough_type_consensus",
        "fold",
        "score",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")
    if data["original_uuid"].duplicated().any():
        raise ValueError(f"Hay UUID duplicados en {path}.")
    result = data[
        [
            "original_uuid",
            "y_true",
            "cough_type",
            "cough_type_consensus",
            "fold",
            "score",
        ]
    ].copy()
    result = result.rename(columns={"score": score_name})
    scores = result[score_name].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(f"Scores no finitos en {path}.")
    if np.any(scores < -1e-9) or np.any(scores > 1.0 + 1e-9):
        raise ValueError(
            f"{path} no contiene probabilidades [0,1]; requiere calibracion."
        )
    return result


def load_aligned_split(split: str) -> pd.DataFrame:
    wst = compact_predictions(
        prediction_path(WST_RESULT_DIR, split), "wst_probability"
    )
    base_columns = [
        "original_uuid",
        "y_true",
        "cough_type",
        "cough_type_consensus",
        "fold",
    ]
    aligned = wst.copy()
    for candidate in COCH_CANDIDATES:
        coch = compact_predictions(
            prediction_path(candidate.result_dir, split),
            f"{candidate.key}_probability",
        )
        merged = aligned.merge(
            coch,
            on="original_uuid",
            how="inner",
            suffixes=("", "__coch"),
            validate="one_to_one",
        )
        for column in base_columns[1:]:
            other = f"{column}__coch"
            if not merged[column].equals(merged[other]):
                raise ValueError(
                    f"Desalineacion de {column} entre WST y {candidate.key}."
                )
            merged = merged.drop(columns=other)
        aligned = merged
    if len(aligned) != len(wst):
        raise ValueError(f"No todos los UUID de {split} aparecen en los modelos.")
    expected_folds = {0, 1, 2, 3, 4} if split == "train_oof" else {-1}
    if set(aligned["fold"].astype(int)) != expected_folds:
        raise ValueError(f"Folds inesperados en {split}.")
    return aligned.sort_values("original_uuid").reset_index(drop=True)


def explicit_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "dry_precision": float(precision_score(y_true, predictions, pos_label=0, zero_division=0)),
        "wet_precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "dry_recall": float(recall_score(y_true, predictions, pos_label=0, zero_division=0)),
        "wet_recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision_wet": float(average_precision_score(y_true, scores)),
        "tn_dry_correct": int(tn),
        "fp_dry_as_wet": int(fp),
        "fn_wet_as_dry": int(fn),
        "tp_wet_correct": int(tp),
    }


def fused_scores(data: pd.DataFrame, coch_key: str, alpha: float) -> np.ndarray:
    wst = data["wst_probability"].to_numpy(dtype=float)
    coch = data[f"{coch_key}_probability"].to_numpy(dtype=float)
    scores = alpha * wst + (1.0 - alpha) * coch
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise RuntimeError("La fusion produjo probabilidades invalidas.")
    return scores


def tune_threshold_fast(
    y_true: np.ndarray,
    scores: np.ndarray,
    default_threshold: float,
) -> float:
    """Equivalente vectorizado de common.tune_threshold.

    Ordena una vez los scores y calcula la matriz de confusion de todos los
    cortes mediante sumas acumuladas, evitando una confusion_matrix completa
    por cada umbral candidato.
    """
    unique_scores = np.unique(scores)
    if len(unique_scores) == 1:
        return default_threshold
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    thresholds = np.unique(
        np.concatenate(
            [
                [np.nextafter(unique_scores[0], -np.inf)],
                midpoints,
                [np.nextafter(unique_scores[-1], np.inf), default_threshold],
            ]
        )
    )
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y_true[order].astype(np.int64)
    prefix_positive = np.concatenate([[0], np.cumsum(sorted_y)])
    cut = np.searchsorted(sorted_scores, thresholds, side="left")
    fn = prefix_positive[cut].astype(float)
    tn = cut.astype(float) - fn
    positives = float(np.sum(y_true == 1))
    negatives = float(np.sum(y_true == 0))
    tp = positives - fn
    fp = negatives - tn

    dry_denominator = 2.0 * tn + fn + fp
    wet_denominator = 2.0 * tp + fp + fn
    dry_f1 = np.divide(
        2.0 * tn,
        dry_denominator,
        out=np.zeros_like(tn),
        where=dry_denominator != 0,
    )
    wet_f1 = np.divide(
        2.0 * tp,
        wet_denominator,
        out=np.zeros_like(tp),
        where=wet_denominator != 0,
    )
    macro_f1 = (dry_f1 + wet_f1) / 2.0
    dry_recall = tn / negatives
    wet_recall = tp / positives
    balanced_accuracy = (dry_recall + wet_recall) / 2.0
    closeness = -np.abs(thresholds - default_threshold)

    candidates = np.flatnonzero(macro_f1 == np.max(macro_f1))
    best_balance = np.max(balanced_accuracy[candidates])
    candidates = candidates[balanced_accuracy[candidates] == best_balance]
    best_closeness = np.max(closeness[candidates])
    candidates = candidates[closeness[candidates] == best_closeness]
    return float(thresholds[int(candidates[0])])


def search_candidates(
    data: pd.DataFrame,
) -> tuple[SearchResult, pd.DataFrame, dict[str, SearchResult]]:
    y_true = data["y_true"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    results: list[SearchResult] = []
    for candidate_index, candidate in enumerate(COCH_CANDIDATES):
        for alpha in ALPHAS:
            # alpha=1 es WST puro y no depende del candidato cochleograma.
            if np.isclose(alpha, 1.0) and candidate_index > 0:
                continue
            scores = fused_scores(data, candidate.key, alpha)
            threshold = tune_threshold_fast(
                y_true, scores, DEFAULT_THRESHOLD
            )
            tuned = common.binary_metrics(y_true, scores, threshold)
            fixed = common.binary_metrics(y_true, scores, DEFAULT_THRESHOLD)
            result = SearchResult(candidate.key, alpha, threshold, tuned, fixed)
            results.append(result)
            rows.append(
                {
                    "candidate_key": result.key,
                    "coch_model": candidate.key,
                    "alpha_wst": alpha,
                    "weight_coch": 1.0 - alpha,
                    "threshold_oof": threshold,
                    **{f"oof_tuned__{key}": value for key, value in tuned.items()},
                    **{f"fixed_0p5__{key}": value for key, value in fixed.items()},
                }
            )

    def key(result: SearchResult) -> tuple[float, ...]:
        # En empates se prefiere depender mas del WST y despues menor tamano.
        candidate = next(item for item in COCH_CANDIDATES if item.key == result.coch_key)
        total_size = WST_MODEL_PATH.stat().st_size + (
            0 if np.isclose(result.alpha, 1.0) else candidate.model_path.stat().st_size
        )
        return (
            float(result.metrics_tuned["macro_f1"]),
            float(result.metrics_tuned["balanced_accuracy"]),
            float(result.metrics_tuned["roc_auc"]),
            float(result.alpha),
            -float(total_size),
        )

    best = max(results, key=key)
    best_by_coch = {
        candidate.key: max(
            [result for result in results if result.coch_key == candidate.key],
            key=key,
        )
        for candidate in COCH_CANDIDATES
    }
    frame = pd.DataFrame(rows).sort_values(
        ["oof_tuned__macro_f1", "oof_tuned__balanced_accuracy", "oof_tuned__roc_auc"],
        ascending=False,
    )
    return best, frame, best_by_coch


def meta_cv_sensitivity(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    fold_rows = []
    for fold in sorted(train["fold"].unique()):
        fit = train[train["fold"] != fold].reset_index(drop=True)
        held = train[train["fold"] == fold].copy()
        selected, _, _ = search_candidates(fit)
        scores = fused_scores(held, selected.coch_key, selected.alpha)
        y_true = held["y_true"].to_numpy(dtype=int)
        y_pred = (scores >= selected.threshold).astype(int)
        held["fused_probability"] = scores
        held["y_pred"] = y_pred
        held["selected_coch_model"] = selected.coch_key
        held["alpha_wst"] = selected.alpha
        held["threshold"] = selected.threshold
        predictions.append(held)
        fold_rows.append(
            {
                "fold": int(fold),
                "selected_coch_model": selected.coch_key,
                "alpha_wst": selected.alpha,
                "weight_coch": 1.0 - selected.alpha,
                "threshold": selected.threshold,
                "selection_train_rows": len(fit),
                "held_rows": len(held),
                **explicit_metrics(y_true, scores, y_pred),
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_rows)


def complementarity_summary(
    train: pd.DataFrame,
    wst_threshold: float,
    coch_thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    y = train["y_true"].to_numpy(dtype=int)
    wst_score = train["wst_probability"].to_numpy(dtype=float)
    wst_pred = (wst_score >= wst_threshold).astype(int)
    wst_correct = wst_pred == y
    for candidate in COCH_CANDIDATES:
        score = train[f"{candidate.key}_probability"].to_numpy(dtype=float)
        pred = (score >= coch_thresholds[candidate.key]).astype(int)
        correct = pred == y
        rows.append(
            {
                "coch_model": candidate.key,
                "secondary_model": candidate.key,
                "pearson_score": float(pd.Series(wst_score).corr(pd.Series(score), method="pearson")),
                "spearman_score": float(pd.Series(wst_score).corr(pd.Series(score), method="spearman")),
                "prediction_disagreement_pct": float(np.mean(wst_pred != pred) * 100.0),
                "both_correct": int(np.sum(wst_correct & correct)),
                "wst_only_correct": int(np.sum(wst_correct & ~correct)),
                "coch_only_correct": int(np.sum(~wst_correct & correct)),
                "both_wrong": int(np.sum(~wst_correct & ~correct)),
                "wet_wst_errors_recovered": int(np.sum((y == 1) & ~wst_correct & correct)),
                "dry_wst_errors_recovered": int(np.sum((y == 0) & ~wst_correct & correct)),
            }
        )
    return pd.DataFrame(rows)


def selected_model_size_kb(selected: SearchResult) -> float:
    total = WST_MODEL_PATH.stat().st_size
    if not np.isclose(selected.alpha, 1.0):
        candidate = next(item for item in COCH_CANDIDATES if item.key == selected.coch_key)
        total += candidate.model_path.stat().st_size
    return total / 1024.0


def metrics_rows(
    dataset: str,
    selected: SearchResult,
    data: pd.DataFrame,
    model_size_kb: float,
) -> list[dict[str, Any]]:
    scores = fused_scores(data, selected.coch_key, selected.alpha)
    y_true = data["y_true"].to_numpy(dtype=int)
    shared = {
        "dataset": dataset,
        "experiment": EXPERIMENT_KEY,
        "candidate_key": selected.key,
        "wst_model": "wst_recording_pca128_logistic_regression",
        "coch_model": selected.coch_key,
        "secondary_model": selected.coch_key,
        "alpha_wst": selected.alpha,
        "weight_coch": 1.0 - selected.alpha,
        "weight_secondary": 1.0 - selected.alpha,
        "score_calibration": "none_raw_probabilities",
        "model_size_kb_joblib": model_size_kb,
    }
    return [
        {
            **shared,
            "threshold_policy": "fixed_0p5",
            "threshold": DEFAULT_THRESHOLD,
            **common.binary_metrics(y_true, scores, DEFAULT_THRESHOLD),
        },
        {
            **shared,
            "threshold_policy": "oof_tuned_frozen",
            "threshold": selected.threshold,
            **common.binary_metrics(y_true, scores, selected.threshold),
        },
    ]


def load_threshold(result_dir: Path, dataset: str = "train_oof") -> float:
    metrics = pd.read_csv(result_dir / "metrics_summary.csv")
    rows = metrics[metrics["dataset"].astype(str) == dataset]
    if "threshold_policy" in rows.columns:
        preferred = rows[rows["threshold_policy"] == "oof_tuned_frozen"]
        if not preferred.empty:
            rows = preferred
    if rows.empty:
        raise ValueError(f"No hay threshold {dataset} en {result_dir}.")
    return float(rows.iloc[0]["threshold"])


def make_graphs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    search_frame: pd.DataFrame,
    selected: SearchResult,
) -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for candidate in COCH_CANDIDATES:
        rows = search_frame[search_frame["coch_model"] == candidate.key].sort_values("alpha_wst")
        axes[0].plot(rows["alpha_wst"], rows["oof_tuned__macro_f1"], marker="o", label=candidate.key)
        axes[1].plot(rows["alpha_wst"], rows["oof_tuned__wet_recall"], marker="o", label=candidate.key)
    axes[0].set(title="Macro-F1 OOF frente a alpha", xlabel="alpha WST", ylabel="Macro-F1")
    axes[1].set(title="Recall wet OOF frente a alpha", xlabel="alpha WST", ylabel="Recall wet")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(GRAPHS_DIR / "alpha_search_oof.png", dpi=180)
    plt.close(figure)

    figure, axes_grid = plt.subplots(
        1,
        len(COCH_CANDIDATES),
        figsize=(6 * len(COCH_CANDIDATES), 5),
        squeeze=False,
    )
    axes = axes_grid.ravel()
    for axis, candidate in zip(axes, COCH_CANDIDATES):
        scatter = axis.scatter(
            train["wst_probability"],
            train[f"{candidate.key}_probability"],
            c=train["y_true"],
            cmap="coolwarm",
            alpha=0.55,
            s=14,
        )
        axis.set(
            xlabel="P(wet) WST-LR",
            ylabel=f"P(wet) {candidate.key}",
            title=candidate.display_name,
        )
        axis.grid(alpha=0.2)
    figure.colorbar(scatter, ax=axes, label="0=dry, 1=wet")
    figure.savefig(GRAPHS_DIR / "score_scatter_oof.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    selected_scores = fused_scores(validation, selected.coch_key, selected.alpha)
    validation_graph = validation[
        ["original_uuid", "y_true", "cough_type", "cough_type_consensus", "fold"]
    ].copy()
    validation_graph["score"] = selected_scores
    common.create_validation_graph(
        validation_graph,
        selected.threshold,
        EXPERIMENT_KEY,
        GraphSpec(selected.key),
        GRAPHS_DIR / VALIDATION_GRAPH_FILENAME,
    )


def check_data(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    train_uuids = set(train["original_uuid"])
    validation_uuids = set(validation["original_uuid"])
    if train_uuids & validation_uuids:
        raise ValueError("TRAIN y VALIDATION comparten UUID.")
    print("=" * 78)
    print(CHECK_TITLE)
    print("=" * 78)
    print(f"TRAIN OOF: {len(train)} grabaciones; folds={sorted(train['fold'].unique())}")
    print(f"VALIDATION: {len(validation)} grabaciones")
    print(
        f"Candidatos {SECONDARY_DISPLAY_LABEL}: "
        f"{', '.join(item.key for item in COCH_CANDIDATES)}"
    )
    print(f"Alphas: {len(ALPHAS)} valores entre {ALPHAS[0]} y {ALPHAS[-1]}")
    print("Scores comprobados como probabilidades [0,1].")
    print("TEST no se ha leido ni se procesara.")


def train_fusion(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    selected, search_frame, best_by_coch = search_candidates(train)
    sensitivity_predictions, sensitivity_folds = meta_cv_sensitivity(train)
    model_size_kb = selected_model_size_kb(selected)

    search_frame.to_csv(RESULTS_DIR / "candidate_oof_results.csv", index=False, encoding="utf-8-sig")
    sensitivity_predictions.to_csv(
        RESULTS_DIR / "meta_cv_sensitivity_predictions.csv", index=False, encoding="utf-8-sig"
    )
    sensitivity_folds.to_csv(
        RESULTS_DIR / "meta_cv_sensitivity_by_fold.csv", index=False, encoding="utf-8-sig"
    )

    train_scores = fused_scores(train, selected.coch_key, selected.alpha)
    train_predictions = train.copy()
    train_predictions["fused_probability"] = train_scores
    train_predictions["y_pred_oof_threshold"] = (
        train_scores >= selected.threshold
    ).astype(int)
    train_predictions["y_pred_fixed_0p5"] = (
        train_scores >= DEFAULT_THRESHOLD
    ).astype(int)
    train_predictions["candidate_key"] = selected.key
    train_predictions.to_csv(
        RESULTS_DIR / "selected_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )

    validation_scores = fused_scores(validation, selected.coch_key, selected.alpha)
    validation_predictions = validation.copy()
    validation_predictions["fused_probability"] = validation_scores
    validation_predictions["score"] = validation_scores
    validation_predictions["y_pred_oof_threshold"] = (
        validation_scores >= selected.threshold
    ).astype(int)
    validation_predictions["y_pred_fixed_0p5"] = (
        validation_scores >= DEFAULT_THRESHOLD
    ).astype(int)
    validation_predictions["candidate_key"] = selected.key
    validation_predictions.to_csv(
        RESULTS_DIR / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )

    # Evaluacion descriptiva del ganador OOF de cada familia de cocleograma.
    # Ambos se seleccionan sin mirar VALIDATION; la seleccion oficial global
    # sigue siendo ``selected``.
    family_rows = []
    for coch_key, family_best in best_by_coch.items():
        family_dir = RESULTS_DIR / coch_key
        family_dir.mkdir(parents=True, exist_ok=True)
        family_size_kb = selected_model_size_kb(family_best)
        family_validation_scores = fused_scores(
            validation, family_best.coch_key, family_best.alpha
        )
        family_validation = validation.copy()
        family_validation["score"] = family_validation_scores
        family_validation["fused_probability"] = family_validation_scores
        family_validation["y_pred_oof_threshold"] = (
            family_validation_scores >= family_best.threshold
        ).astype(int)
        family_validation["y_pred_fixed_0p5"] = (
            family_validation_scores >= DEFAULT_THRESHOLD
        ).astype(int)
        family_validation["candidate_key"] = family_best.key
        family_validation.to_csv(
            family_dir / "validation_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        family_train_scores = fused_scores(
            train, family_best.coch_key, family_best.alpha
        )
        family_train = train.copy()
        family_train["score"] = family_train_scores
        family_train["fused_probability"] = family_train_scores
        family_train["candidate_key"] = family_best.key
        family_train.to_csv(
            family_dir / "selected_oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        family_metrics = [
            *metrics_rows(
                "train_oof", family_best, train, family_size_kb
            ),
            *metrics_rows(
                "validation", family_best, validation, family_size_kb
            ),
        ]
        pd.DataFrame(family_metrics).to_csv(
            family_dir / "metrics_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        validation_tuned_family = common.binary_metrics(
            validation["y_true"].to_numpy(dtype=int),
            family_validation_scores,
            family_best.threshold,
        )
        family_rows.append(
            {
                "coch_model": coch_key,
                "secondary_model": coch_key,
                "candidate_key": family_best.key,
                "selected_overall_oof": family_best.key == selected.key,
                "alpha_wst": family_best.alpha,
                "weight_coch": 1.0 - family_best.alpha,
                "threshold_oof": family_best.threshold,
                "model_size_kb_combined": family_size_kb,
                **{
                    f"train_oof__{key}": value
                    for key, value in family_best.metrics_tuned.items()
                },
                **{
                    f"validation__{key}": value
                    for key, value in validation_tuned_family.items()
                },
            }
        )
    pd.DataFrame(family_rows).to_csv(
        RESULTS_DIR / CANDIDATE_COMPARISON_FILENAME,
        index=False,
        encoding="utf-8-sig",
    )

    wst_threshold = load_threshold(WST_RESULT_DIR)
    coch_thresholds = {
        candidate.key: load_threshold(candidate.result_dir)
        for candidate in COCH_CANDIDATES
    }
    complementarity_summary(train, wst_threshold, coch_thresholds).to_csv(
        RESULTS_DIR / "complementarity_summary_oof.csv", index=False, encoding="utf-8-sig"
    )
    complementarity_summary(validation, wst_threshold, coch_thresholds).to_csv(
        RESULTS_DIR / "complementarity_summary_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sensitivity_y = sensitivity_predictions["y_true"].to_numpy(dtype=int)
    sensitivity_scores = sensitivity_predictions["fused_probability"].to_numpy(dtype=float)
    sensitivity_pred = sensitivity_predictions["y_pred"].to_numpy(dtype=int)
    sensitivity_metrics = explicit_metrics(
        sensitivity_y, sensitivity_scores, sensitivity_pred
    )
    summary_rows = [
        {
            "dataset": "train_oof_meta_cv_sensitivity",
            "experiment": EXPERIMENT_KEY,
            "candidate_key": "fold_specific_selection",
            "wst_model": "wst_recording_pca128_logistic_regression",
            "coch_model": "fold_specific",
            "secondary_model": "fold_specific",
            "alpha_wst": np.nan,
            "weight_coch": np.nan,
            "weight_secondary": np.nan,
            "score_calibration": "none_raw_probabilities",
            "model_size_kb_joblib": np.nan,
            "threshold_policy": "fold_specific_train_oof",
            "threshold": np.nan,
            **sensitivity_metrics,
        },
        *metrics_rows("train_oof", selected, train, model_size_kb),
        *metrics_rows("validation", selected, validation, model_size_kb),
    ]
    pd.DataFrame(summary_rows).to_csv(
        RESULTS_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig"
    )

    final_row = {
        "candidate_key": selected.key,
        "wst_model": "wst_recording_pca128_logistic_regression",
        "coch_model": selected.coch_key,
        "secondary_model": selected.coch_key,
        "alpha_wst": selected.alpha,
        "weight_coch": 1.0 - selected.alpha,
        "weight_secondary": 1.0 - selected.alpha,
        "threshold_oof": selected.threshold,
        "model_size_kb_combined": model_size_kb,
        **selected.metrics_tuned,
    }
    pd.DataFrame([final_row]).to_csv(
        RESULTS_DIR / "selected_fusion_configuration.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {
                "fusion_type": "late_weighted_probability_average",
                "formula": "alpha*p_wst + (1-alpha)*p_coch",
                "alpha_grid": "0.00:0.05:1.00",
                "candidate_coch_models": "|".join(item.key for item in COCH_CANDIDATES),
                "candidate_secondary_models": "|".join(
                    item.key for item in COCH_CANDIDATES
                ),
                "selected_candidate": selected.key,
                "selected_coch_model": selected.coch_key,
                "selected_secondary_model": selected.coch_key,
                "selected_alpha_wst": selected.alpha,
                "selected_weight_coch": 1.0 - selected.alpha,
                "selected_threshold": selected.threshold,
                "score_calibration": "none_raw_probabilities",
                "selection_data": "TRAIN OOF only",
                "meta_cv_sensitivity": "leave_one_existing_fold_out",
                "validation_used_for_selection": False,
                "test_processed": False,
                "wst_model_path": str(WST_MODEL_PATH),
                "coch_model_path": str(next(item.model_path for item in COCH_CANDIDATES if item.key == selected.coch_key)),
                "secondary_model_path": str(
                    next(
                        item.model_path
                        for item in COCH_CANDIDATES
                        if item.key == selected.coch_key
                    )
                ),
                "model_size_kb_combined": model_size_kb,
            }
        ]
    ).to_csv(
        RESULTS_DIR / "experiment_configuration.csv", index=False, encoding="utf-8-sig"
    )
    deployment = {
        "fusion_type": "late_weighted_probability_average",
        "formula": "alpha*p_wst + (1-alpha)*p_coch",
        "wst_model_path": str(WST_MODEL_PATH),
        "coch_model_key": selected.coch_key,
        "coch_model_path": str(next(item.model_path for item in COCH_CANDIDATES if item.key == selected.coch_key)),
        "secondary_model_key": selected.coch_key,
        "secondary_model_path": str(
            next(
                item.model_path
                for item in COCH_CANDIDATES
                if item.key == selected.coch_key
            )
        ),
        "alpha_wst": selected.alpha,
        "weight_coch": 1.0 - selected.alpha,
        "threshold_wet": selected.threshold,
        "label_mapping": {"0": "dry", "1": "wet"},
    }
    (MODELS_DIR / "late_fusion_configuration.json").write_text(
        json.dumps(deployment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_graphs(train, validation, search_frame, selected)

    validation_tuned = common.binary_metrics(
        validation["y_true"].to_numpy(dtype=int),
        validation_scores,
        selected.threshold,
    )
    validation_fixed = common.binary_metrics(
        validation["y_true"].to_numpy(dtype=int),
        validation_scores,
        DEFAULT_THRESHOLD,
    )
    print("\n" + "=" * 78)
    print(RESULT_TITLE)
    print("=" * 78)
    print(f"Candidato seleccionado: {selected.coch_key}")
    print(
        f"Alpha WST / {SECONDARY_WEIGHT_LABEL}: "
        f"{selected.alpha:.2f} / {1.0-selected.alpha:.2f}"
    )
    print(f"Umbral OOF congelado: {selected.threshold:.6f}")
    print(f"Macro-F1 OOF seleccion: {selected.metrics_tuned['macro_f1']:.4f}")
    print(f"Macro-F1 meta-CV sensibilidad: {sensitivity_metrics['macro_f1']:.4f}")
    print(
        "Macro-F1 validation OOF/@0.5: "
        f"{validation_tuned['macro_f1']:.4f} / {validation_fixed['macro_f1']:.4f}"
    )
    print(
        "Recalls validation dry/wet: "
        f"{validation_tuned['dry_recall']:.4f} / {validation_tuned['wet_recall']:.4f}"
    )
    print(f"Tamano combinado de modelos: {model_size_kb:.2f} KB")
    print(f"Resultados: {RESULTS_DIR}")
    print(f"Graficas: {GRAPHS_DIR}")
    print("TEST permanece reservado.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=PARSER_DESCRIPTION
    )
    parser.add_argument("--action", choices=["check", "train"], default="check")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print(DISPLAY_TITLE)
    print("=" * 78)
    print(f"Accion: {args.action}")
    print("TEST no sera leido ni procesado.")
    require_inputs()
    train = load_aligned_split("train_oof")
    validation = load_aligned_split("validation")
    check_data(train, validation)
    if args.action == "check":
        print("Comprobacion completada. No se genero ninguna fusion.")
        return
    train_fusion(train, validation)


if __name__ == "__main__":
    main()
