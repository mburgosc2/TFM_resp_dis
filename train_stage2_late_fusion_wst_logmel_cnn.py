"""Late fusion recording-level entre WST-LR y una tiny CNN Log-Mel.

No reentrena los modelos base. Combina sus probabilidades OOF, alineadas
uno-a-uno por ``original_uuid``, mediante:

    p_wet = alpha * p_wst + (1 - alpha) * p_logmel_cnn

Alpha y umbral se seleccionan solo con OOF de TRAIN. Se comparan dos
politicas predefinidas:

1. ``max_macro_f1``: maximiza macro-F1.
2. ``max_macro_f1_wet_recall_ge_0p55``: maximiza macro-F1 exigiendo recall
   wet >= 0,55 en los datos usados para seleccionar el umbral.

Un meta-CV leave-one-existing-fold-out mide la estabilidad de ambas
politicas. Las configuraciones finales se congelan antes de aplicarlas a
VALIDATION. TEST no se lee ni se procesa.
"""

from __future__ import annotations

import argparse
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
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train_stage2_dry_wet_cochleograms as common


ROOT = Path(__file__).resolve().parent

WST_RESULTS_DIR = (
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

CNN_RESULTS_DIR = (
    ROOT
    / "results_stage2_dry_wet_logmel_tiny_cnn"
    / "logmel64_win32_hop16"
    / "full"
)
CNN_MODEL_PATH = (
    ROOT
    / "models_stage2_dry_wet_logmel_tiny_cnn"
    / "logmel64_win32_hop16"
    / "full"
    / "logmel_tiny_cnn.keras"
)

RESULTS_DIR = ROOT / "results_stage2_late_fusion_wst_logmel_cnn" / "full"
GRAPHS_DIR = (
    ROOT
    / "graphs_results_stage2_dry_wet_random"
    / "late_fusion_wst_logmel_cnn"
    / "full"
)

ALPHAS = tuple(
    float(value)
    for value in np.round(np.arange(0.0, 1.0001, 0.05), 2)
)
DEFAULT_THRESHOLD = 0.5
WET_RECALL_MINIMUM = 0.55
POLICIES = (
    "max_macro_f1",
    "max_macro_f1_wet_recall_ge_0p55",
)


@dataclass(frozen=True)
class SearchResult:
    policy: str
    alpha_wst: float
    threshold: float
    metrics: dict[str, float | int]
    combined_model_size_kb: float

    @property
    def weight_cnn(self) -> float:
        return 1.0 - self.alpha_wst

    @property
    def key(self) -> str:
        alpha = f"{self.alpha_wst:.2f}".replace(".", "p")
        return f"late_fusion_wst_logmel_cnn__{self.policy}__alpha{alpha}"


@dataclass(frozen=True)
class GraphSpec:
    key: str


def prediction_path(result_dir: Path, split_name: str) -> Path:
    filename = (
        "best_oof_predictions.csv"
        if split_name == "train_oof"
        else "validation_predictions.csv"
    )
    return result_dir / filename


def require_inputs() -> None:
    required = [
        prediction_path(WST_RESULTS_DIR, "train_oof"),
        prediction_path(WST_RESULTS_DIR, "validation"),
        WST_RESULTS_DIR / "metrics_summary.csv",
        WST_MODEL_PATH,
        prediction_path(CNN_RESULTS_DIR, "train_oof"),
        prediction_path(CNN_RESULTS_DIR, "validation"),
        CNN_RESULTS_DIR / "metrics_summary.csv",
        CNN_MODEL_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan entradas para la late fusion:\n" + "\n".join(missing)
        )


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
        raise ValueError(f"Hay original_uuid duplicados en {path}.")

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
        raise ValueError(f"Los scores de {path} no son probabilidades.")
    return result


def load_aligned_split(split_name: str) -> pd.DataFrame:
    wst = compact_predictions(
        prediction_path(WST_RESULTS_DIR, split_name),
        "wst_probability",
    )
    cnn = compact_predictions(
        prediction_path(CNN_RESULTS_DIR, split_name),
        "cnn_probability",
    )
    aligned = wst.merge(
        cnn,
        on="original_uuid",
        how="inner",
        suffixes=("", "__cnn"),
        validate="one_to_one",
    )
    base_columns = [
        "y_true",
        "cough_type",
        "cough_type_consensus",
        "fold",
    ]
    for column in base_columns:
        cnn_column = f"{column}__cnn"
        if not aligned[column].equals(aligned[cnn_column]):
            raise ValueError(
                f"Desalineacion de {column} entre WST y CNN."
            )
        aligned = aligned.drop(columns=cnn_column)

    if len(aligned) != len(wst) or len(aligned) != len(cnn):
        raise ValueError(
            f"No coinciden todos los UUID de {split_name}: "
            f"WST={len(wst)}, CNN={len(cnn)}, comunes={len(aligned)}."
        )
    expected_folds = (
        common.EXPECTED_TRAIN_FOLDS
        if split_name == "train_oof"
        else {-1}
    )
    if set(aligned["fold"].astype(int)) != expected_folds:
        raise ValueError(f"Folds inesperados en {split_name}.")
    if set(aligned["y_true"].astype(int)) != {0, 1}:
        raise ValueError(f"Clases inesperadas en {split_name}.")
    return aligned.sort_values("original_uuid").reset_index(drop=True)


def fused_scores(data: pd.DataFrame, alpha_wst: float) -> np.ndarray:
    scores = (
        alpha_wst * data["wst_probability"].to_numpy(dtype=float)
        + (1.0 - alpha_wst)
        * data["cnn_probability"].to_numpy(dtype=float)
    )
    if (
        not np.isfinite(scores).all()
        or np.any(scores < -1e-9)
        or np.any(scores > 1.0 + 1e-9)
    ):
        raise RuntimeError("La fusion produjo probabilidades invalidas.")
    return scores


def combined_model_size_kb(alpha_wst: float) -> float:
    if np.isclose(alpha_wst, 1.0):
        size = WST_MODEL_PATH.stat().st_size
    elif np.isclose(alpha_wst, 0.0):
        size = CNN_MODEL_PATH.stat().st_size
    else:
        size = WST_MODEL_PATH.stat().st_size + CNN_MODEL_PATH.stat().st_size
    return size / 1024.0


def threshold_statistics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    """Calcula metricas de todos los cortes con una sola ordenacion."""

    unique_scores = np.unique(scores)
    if len(unique_scores) == 1:
        thresholds = np.asarray([DEFAULT_THRESHOLD], dtype=float)
    else:
        thresholds = np.unique(
            np.concatenate(
                [
                    [np.nextafter(unique_scores[0], -np.inf)],
                    (unique_scores[:-1] + unique_scores[1:]) / 2.0,
                    [
                        np.nextafter(unique_scores[-1], np.inf),
                        DEFAULT_THRESHOLD,
                    ],
                ]
            )
        )

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y_true[order].astype(np.int64)
    prefix_positive = np.concatenate([[0], np.cumsum(sorted_y)])
    cuts = np.searchsorted(sorted_scores, thresholds, side="left")
    false_negative = prefix_positive[cuts].astype(float)
    true_negative = cuts.astype(float) - false_negative
    positives = float(np.sum(y_true == 1))
    negatives = float(np.sum(y_true == 0))
    true_positive = positives - false_negative
    false_positive = negatives - true_negative

    dry_denominator = (
        2.0 * true_negative + false_negative + false_positive
    )
    wet_denominator = (
        2.0 * true_positive + false_positive + false_negative
    )
    dry_f1 = np.divide(
        2.0 * true_negative,
        dry_denominator,
        out=np.zeros_like(true_negative),
        where=dry_denominator != 0,
    )
    wet_f1 = np.divide(
        2.0 * true_positive,
        wet_denominator,
        out=np.zeros_like(true_positive),
        where=wet_denominator != 0,
    )
    dry_recall = true_negative / negatives
    wet_recall = true_positive / positives

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "macro_f1": (dry_f1 + wet_f1) / 2.0,
            "balanced_accuracy": (dry_recall + wet_recall) / 2.0,
            "dry_recall": dry_recall,
            "wet_recall": wet_recall,
            "distance_to_0p5": np.abs(thresholds - DEFAULT_THRESHOLD),
        }
    )


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    policy: str,
) -> float:
    candidates = threshold_statistics(y_true, scores)
    if policy == "max_macro_f1_wet_recall_ge_0p55":
        candidates = candidates[
            candidates["wet_recall"] >= WET_RECALL_MINIMUM - 1e-12
        ]
        if candidates.empty:
            raise RuntimeError(
                "No existe umbral que cumpla la restriccion de recall wet."
            )
    elif policy != "max_macro_f1":
        raise ValueError(f"Politica desconocida: {policy}")

    selected = candidates.sort_values(
        ["macro_f1", "balanced_accuracy", "distance_to_0p5"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    return float(selected["threshold"])


def search_policy(
    data: pd.DataFrame,
    policy: str,
) -> tuple[SearchResult, pd.DataFrame]:
    y_true = data["y_true"].to_numpy(dtype=int)
    results: list[SearchResult] = []
    rows: list[dict[str, Any]] = []

    for alpha in ALPHAS:
        scores = fused_scores(data, alpha)
        threshold = select_threshold(y_true, scores, policy)
        metrics = common.binary_metrics(y_true, scores, threshold)
        result = SearchResult(
            policy=policy,
            alpha_wst=alpha,
            threshold=threshold,
            metrics=metrics,
            combined_model_size_kb=combined_model_size_kb(alpha),
        )
        results.append(result)
        rows.append(
            {
                "policy": policy,
                "candidate_key": result.key,
                "alpha_wst": alpha,
                "weight_cnn": 1.0 - alpha,
                "threshold_oof": threshold,
                "combined_model_size_kb": result.combined_model_size_kb,
                **{f"oof__{key}": value for key, value in metrics.items()},
            }
        )

    def selection_key(result: SearchResult) -> tuple[float, ...]:
        return (
            float(result.metrics["macro_f1"]),
            float(result.metrics["balanced_accuracy"]),
            float(result.metrics["roc_auc"]),
            -float(result.combined_model_size_kb),
            float(result.alpha_wst),
        )

    return max(results, key=selection_key), pd.DataFrame(rows)


def explicit_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "dry_precision": float(
            precision_score(y_true, predictions, pos_label=0, zero_division=0)
        ),
        "wet_precision": float(
            precision_score(y_true, predictions, pos_label=1, zero_division=0)
        ),
        "dry_recall": float(
            recall_score(y_true, predictions, pos_label=0, zero_division=0)
        ),
        "wet_recall": float(
            recall_score(y_true, predictions, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision_wet": float(
            average_precision_score(y_true, scores)
        ),
        "tn_dry_correct": int(tn),
        "fp_dry_as_wet": int(fp),
        "fn_wet_as_dry": int(fn),
        "tp_wet_correct": int(tp),
    }


def meta_cv_policy(
    train: pd.DataFrame,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    fold_rows = []

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        fit = train[train["fold"] != fold].reset_index(drop=True)
        held = train[train["fold"] == fold].copy()
        selected, _ = search_policy(fit, policy)
        scores = fused_scores(held, selected.alpha_wst)
        y_true = held["y_true"].to_numpy(dtype=int)
        predictions = (scores >= selected.threshold).astype(int)

        held["policy"] = policy
        held["fused_probability"] = scores
        held["y_pred"] = predictions
        held["selected_alpha_wst"] = selected.alpha_wst
        held["selected_weight_cnn"] = selected.weight_cnn
        held["selected_threshold"] = selected.threshold
        prediction_frames.append(held)
        fold_rows.append(
            {
                "policy": policy,
                "fold": int(fold),
                "selection_rows": len(fit),
                "held_rows": len(held),
                "selected_alpha_wst": selected.alpha_wst,
                "selected_weight_cnn": selected.weight_cnn,
                "selected_threshold": selected.threshold,
                **explicit_metrics(y_true, scores, predictions),
            }
        )

    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(fold_rows),
    )


def meta_cv_global_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    return explicit_metrics(
        predictions["y_true"].to_numpy(dtype=int),
        predictions["fused_probability"].to_numpy(dtype=float),
        predictions["y_pred"].to_numpy(dtype=int),
    )


def prediction_frame(
    data: pd.DataFrame,
    selected: SearchResult,
) -> pd.DataFrame:
    result = data.copy()
    scores = fused_scores(result, selected.alpha_wst)
    result["policy"] = selected.policy
    result["candidate_key"] = selected.key
    result["fused_probability"] = scores
    result["score"] = scores
    result["alpha_wst"] = selected.alpha_wst
    result["weight_cnn"] = selected.weight_cnn
    result["threshold"] = selected.threshold
    result["y_pred"] = (scores >= selected.threshold).astype(int)
    return result


def metrics_row(
    dataset: str,
    selected: SearchResult,
    data: pd.DataFrame,
) -> dict[str, Any]:
    scores = fused_scores(data, selected.alpha_wst)
    return {
        "dataset": dataset,
        "policy": selected.policy,
        "candidate_key": selected.key,
        "alpha_wst": selected.alpha_wst,
        "weight_cnn": selected.weight_cnn,
        "threshold_policy": "oof_selected_frozen",
        "threshold": selected.threshold,
        "combined_model_size_kb": selected.combined_model_size_kb,
        **common.binary_metrics(
            data["y_true"].to_numpy(dtype=int),
            scores,
            selected.threshold,
        ),
    }


def load_base_threshold(result_dir: Path, dataset_name: str) -> float:
    metrics = pd.read_csv(result_dir / "metrics_summary.csv")
    split_column = "dataset" if "dataset" in metrics.columns else "split"
    rows = metrics[metrics[split_column].astype(str) == dataset_name]
    if "threshold_policy" in rows.columns:
        preferred = rows[
            rows["threshold_policy"].astype(str).str.contains(
                "oof", case=False, na=False
            )
        ]
        if not preferred.empty:
            rows = preferred
    if rows.empty:
        raise ValueError(
            f"No se encuentra threshold {dataset_name} en {result_dir}."
        )
    return float(rows.iloc[0]["threshold"])


def complementarity_summary(
    data: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    y_true = data["y_true"].to_numpy(dtype=int)
    wst = data["wst_probability"].to_numpy(dtype=float)
    cnn = data["cnn_probability"].to_numpy(dtype=float)
    wst_threshold = load_base_threshold(WST_RESULTS_DIR, "train_oof")
    cnn_threshold = load_base_threshold(CNN_RESULTS_DIR, "train_oof")
    wst_pred = (wst >= wst_threshold).astype(int)
    cnn_pred = (cnn >= cnn_threshold).astype(int)
    wst_correct = wst_pred == y_true
    cnn_correct = cnn_pred == y_true

    return pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "recording_count": len(data),
                "pearson_score": float(
                    pd.Series(wst).corr(pd.Series(cnn), method="pearson")
                ),
                "spearman_score": float(
                    pd.Series(wst).corr(pd.Series(cnn), method="spearman")
                ),
                "prediction_disagreement_pct": float(
                    np.mean(wst_pred != cnn_pred) * 100.0
                ),
                "both_correct": int(np.sum(wst_correct & cnn_correct)),
                "wst_only_correct": int(
                    np.sum(wst_correct & ~cnn_correct)
                ),
                "cnn_only_correct": int(
                    np.sum(~wst_correct & cnn_correct)
                ),
                "both_wrong": int(np.sum(~wst_correct & ~cnn_correct)),
                "wet_wst_errors_recovered_by_cnn": int(
                    np.sum((y_true == 1) & ~wst_correct & cnn_correct)
                ),
                "dry_wst_errors_recovered_by_cnn": int(
                    np.sum((y_true == 0) & ~wst_correct & cnn_correct)
                ),
                "wst_threshold_oof": wst_threshold,
                "cnn_threshold_oof": cnn_threshold,
            }
        ]
    )


def create_graphs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    search_frame: pd.DataFrame,
    selections: dict[str, SearchResult],
) -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for policy in POLICIES:
        rows = search_frame[search_frame["policy"] == policy].sort_values(
            "alpha_wst"
        )
        axes[0].plot(
            rows["alpha_wst"],
            rows["oof__macro_f1"],
            marker="o",
            label=policy,
        )
        axes[1].plot(
            rows["alpha_wst"],
            rows["oof__wet_recall"],
            marker="o",
            label=policy,
        )
    axes[0].set(
        title="Macro-F1 OOF frente a alpha",
        xlabel="alpha WST",
        ylabel="Macro-F1",
    )
    axes[1].set(
        title="Recall wet OOF frente a alpha",
        xlabel="alpha WST",
        ylabel="Recall wet",
    )
    axes[1].axhline(
        WET_RECALL_MINIMUM,
        color="gray",
        linestyle="--",
        label="minimo 0,55",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(GRAPHS_DIR / "alpha_search_oof.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    scatter = axis.scatter(
        train["wst_probability"],
        train["cnn_probability"],
        c=train["y_true"],
        cmap="coolwarm",
        alpha=0.55,
        s=15,
    )
    axis.set(
        xlabel="P(wet) WST + LR",
        ylabel="P(wet) CNN Log-Mel",
        title="Complementariedad OOF por grabacion",
    )
    axis.grid(alpha=0.2)
    figure.colorbar(scatter, ax=axis, label="0=dry, 1=wet")
    figure.tight_layout()
    figure.savefig(GRAPHS_DIR / "score_scatter_oof.png", dpi=180)
    plt.close(figure)

    for policy, selected in selections.items():
        graph_data = validation[
            [
                "original_uuid",
                "y_true",
                "cough_type",
                "cough_type_consensus",
                "fold",
            ]
        ].copy()
        graph_data["score"] = fused_scores(
            validation, selected.alpha_wst
        )
        common.create_validation_graph(
            graph_data,
            selected.threshold,
            "late_fusion_wst_logmel_cnn",
            GraphSpec(selected.key),
            GRAPHS_DIR / f"validation_{policy}.png",
        )


def check_data(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    overlap = set(train["original_uuid"]) & set(validation["original_uuid"])
    if overlap:
        raise ValueError("TRAIN y VALIDATION comparten original_uuid.")
    print("=" * 78)
    print("CHECK - LATE FUSION WST-LR + LOG-MEL TINY CNN")
    print("=" * 78)
    print(
        f"TRAIN OOF: {len(train)} grabaciones; "
        f"folds={sorted(train['fold'].unique())}"
    )
    print(f"VALIDATION: {len(validation)} grabaciones")
    print(f"Alphas: {len(ALPHAS)} valores de 0,00 a 1,00")
    print(
        "Politicas: max_macro_f1 y macro-F1 con recall wet OOF >= "
        f"{WET_RECALL_MINIMUM:.2f}."
    )
    print("alpha=0: CNN pura; alpha=1: WST puro.")
    print("Probabilidades alineadas uno-a-uno por original_uuid.")
    print("TEST no se ha leido ni se procesara.")


def run_experiment(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selections: dict[str, SearchResult] = {}
    search_frames = []
    meta_predictions = []
    meta_folds = []
    metrics_rows = []
    selection_rows = []

    for policy in POLICIES:
        selected, search_frame = search_policy(train, policy)
        selections[policy] = selected
        search_frames.append(search_frame)

        policy_meta_predictions, policy_meta_folds = meta_cv_policy(
            train, policy
        )
        meta_predictions.append(policy_meta_predictions)
        meta_folds.append(policy_meta_folds)
        meta_metrics = meta_cv_global_metrics(policy_meta_predictions)

        policy_dir = RESULTS_DIR / policy
        policy_dir.mkdir(parents=True, exist_ok=True)
        oof_predictions = prediction_frame(train, selected)
        validation_predictions = prediction_frame(validation, selected)
        oof_predictions.to_csv(
            policy_dir / "selected_oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        validation_predictions.to_csv(
            policy_dir / "validation_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        metrics_rows.extend(
            [
                {
                    "dataset": "train_oof_meta_cv",
                    "policy": policy,
                    "candidate_key": "fold_specific_selection",
                    "alpha_wst": np.nan,
                    "weight_cnn": np.nan,
                    "threshold_policy": "fold_specific_train_oof",
                    "threshold": np.nan,
                    "combined_model_size_kb": np.nan,
                    **meta_metrics,
                },
                metrics_row("train_oof", selected, train),
                metrics_row("validation", selected, validation),
            ]
        )
        selection_rows.append(
            {
                "policy": policy,
                "candidate_key": selected.key,
                "alpha_wst": selected.alpha_wst,
                "weight_cnn": selected.weight_cnn,
                "threshold_oof": selected.threshold,
                "wet_recall_constraint_oof": (
                    WET_RECALL_MINIMUM
                    if policy == "max_macro_f1_wet_recall_ge_0p55"
                    else np.nan
                ),
                "combined_model_size_kb": selected.combined_model_size_kb,
                **selected.metrics,
            }
        )

    all_search = pd.concat(search_frames, ignore_index=True)
    all_meta_predictions = pd.concat(meta_predictions, ignore_index=True)
    all_meta_folds = pd.concat(meta_folds, ignore_index=True)
    all_search.to_csv(
        RESULTS_DIR / "candidate_oof_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_meta_predictions.to_csv(
        RESULTS_DIR / "meta_cv_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_meta_folds.to_csv(
        RESULTS_DIR / "meta_cv_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metrics_rows).to_csv(
        RESULTS_DIR / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(selection_rows).to_csv(
        RESULTS_DIR / "selected_fusion_configurations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        [
            complementarity_summary(train, "train_oof"),
            complementarity_summary(validation, "validation"),
        ],
        ignore_index=True,
    ).to_csv(
        RESULTS_DIR / "complementarity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "fusion_type": "late_weighted_probability_average",
                "formula": "alpha*p_wst + (1-alpha)*p_logmel_cnn",
                "fusion_unit": "original_uuid_recording",
                "alpha_grid": "0.00:0.05:1.00",
                "selection_data": "TRAIN OOF only",
                "meta_cv": "leave_one_existing_fold_out",
                "wet_recall_constraint": WET_RECALL_MINIMUM,
                "score_calibration": "none_raw_probabilities",
                "validation_used_for_selection": False,
                "test_processed": False,
                "wst_results_dir": str(WST_RESULTS_DIR),
                "cnn_results_dir": str(CNN_RESULTS_DIR),
                "wst_model_path": str(WST_MODEL_PATH),
                "cnn_model_path": str(CNN_MODEL_PATH),
            }
        ]
    ).to_csv(
        RESULTS_DIR / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    create_graphs(train, validation, all_search, selections)

    print("\n" + "=" * 78)
    print("RESULTADO LATE FUSION WST-LR + LOG-MEL TINY CNN")
    print("=" * 78)
    for policy in POLICIES:
        selected = selections[policy]
        validation_metrics = common.binary_metrics(
            validation["y_true"].to_numpy(dtype=int),
            fused_scores(validation, selected.alpha_wst),
            selected.threshold,
        )
        meta_rows = pd.DataFrame(metrics_rows)
        meta_row = meta_rows[
            (meta_rows["policy"] == policy)
            & (meta_rows["dataset"] == "train_oof_meta_cv")
        ].iloc[0]
        print(f"\nPolitica: {policy}")
        print(
            f"Alpha WST/CNN: {selected.alpha_wst:.2f} / "
            f"{selected.weight_cnn:.2f}"
        )
        print(f"Umbral OOF: {selected.threshold:.6f}")
        print(f"Macro-F1 OOF seleccion: {selected.metrics['macro_f1']:.4f}")
        print(f"Macro-F1 meta-CV: {float(meta_row['macro_f1']):.4f}")
        print(f"Macro-F1 validation: {validation_metrics['macro_f1']:.4f}")
        print(
            "Recalls validation dry/wet: "
            f"{validation_metrics['dry_recall']:.4f} / "
            f"{validation_metrics['wet_recall']:.4f}"
        )
        print(f"AUC validation: {validation_metrics['roc_auc']:.4f}")
    print(f"\nResultados: {RESULTS_DIR}")
    print(f"Graficas: {GRAPHS_DIR}")
    print("TEST permanece reservado.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Late fusion WST-LR + tiny CNN Log-Mel."
    )
    parser.add_argument(
        "--action",
        choices=["check", "train"],
        default="check",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - LATE FUSION WST-LR + LOG-MEL TINY CNN")
    print("=" * 78)
    print(f"Accion: {args.action}")
    print("TEST no sera leido ni procesado.")
    require_inputs()
    train = load_aligned_split("train_oof")
    validation = load_aligned_split("validation")
    check_data(train, validation)
    if args.action == "check":
        print("Comprobacion completada. No se generaron resultados.")
        return
    run_experiment(train, validation)


if __name__ == "__main__":
    main()
