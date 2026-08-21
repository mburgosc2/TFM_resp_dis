"""XGBoost ligero sobre cocleogramas raw reducidos con PCA64.

La representacion, folds, pesos y agregacion por grabacion son identicos a
los experimentos PCA64 con SVM y Random Forest. Solo cambia el
clasificador. La busqueda no usa SMOTE, ``scale_pos_weight`` ni early
stopping, y TEST no se lee ni se procesa.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_cochleograms_raw_pca as raw_pca


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (
    SCRIPT_DIR / "results_stage2_dry_wet_cochleograms_raw_pca_xgb"
)
MODELS_ROOT = (
    SCRIPT_DIR / "models_stage2_dry_wet_cochleograms_raw_pca_xgb"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_cochleograms_raw_pca_xgb"
)

RANDOM_STATE = 42
PCA_COMPONENTS = 64
FULL_CANDIDATE_COUNT = 36


@dataclass(frozen=True)
class XGBCandidate:
    n_estimators: int
    max_depth: int
    learning_rate: float
    min_child_weight: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    gamma: float

    @property
    def key(self) -> str:
        def number(value: int | float) -> str:
            return f"{value:g}".replace(".", "p")

        return (
            f"xgb__trees{self.n_estimators}__depth{self.max_depth}__"
            f"lr{number(self.learning_rate)}__"
            f"child{self.min_child_weight}__"
            f"sub{number(self.subsample)}__"
            f"col{number(self.colsample_bytree)}__"
            f"lambda{number(self.reg_lambda)}__gamma{number(self.gamma)}"
        )

    @property
    def theoretical_max_nodes(self) -> int:
        return self.n_estimators * (2 ** (self.max_depth + 1) - 1)


@dataclass
class XGBEvaluation:
    candidate: XGBCandidate
    threshold: float
    metrics: dict[str, float | int]
    oof_recordings: pd.DataFrame
    fold_metrics: pd.DataFrame


def canonical_candidates() -> list[XGBCandidate]:
    return [
        XGBCandidate(75, 2, 0.08, 5, 0.85, 0.8, 5.0, 0.0),
        XGBCandidate(100, 3, 0.05, 5, 0.85, 0.8, 5.0, 0.0),
        XGBCandidate(150, 3, 0.05, 10, 0.85, 0.8, 5.0, 0.1),
        XGBCandidate(200, 3, 0.03, 10, 0.85, 0.8, 10.0, 0.1),
        XGBCandidate(100, 4, 0.05, 10, 0.85, 0.8, 10.0, 0.1),
        XGBCandidate(150, 4, 0.03, 15, 0.8, 0.7, 10.0, 0.3),
    ]


def xgb_candidates(quick: bool) -> list[XGBCandidate]:
    canonical = canonical_candidates()
    if quick:
        return canonical[:3]

    search_space = {
        "n_estimators": [50, 75, 100, 150, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "min_child_weight": [3, 5, 10, 15],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
        "gamma": [0.0, 0.1, 0.3],
    }
    sampled = ParameterSampler(
        search_space,
        n_iter=FULL_CANDIDATE_COUNT,
        random_state=RANDOM_STATE,
    )
    candidates = list(canonical)
    known_keys = {candidate.key for candidate in candidates}
    for params in sampled:
        candidate = XGBCandidate(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            min_child_weight=int(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            gamma=float(params["gamma"]),
        )
        if candidate.key not in known_keys:
            candidates.append(candidate)
            known_keys.add(candidate.key)
        if len(candidates) == FULL_CANDIDATE_COUNT:
            break

    if len(candidates) != FULL_CANDIDATE_COUNT:
        raise RuntimeError(
            f"Se esperaban {FULL_CANDIDATE_COUNT} candidatos, "
            f"se generaron {len(candidates)}."
        )
    return candidates


def build_xgb(candidate: XGBCandidate) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=candidate.n_estimators,
        max_depth=candidate.max_depth,
        learning_rate=candidate.learning_rate,
        min_child_weight=candidate.min_child_weight,
        subsample=candidate.subsample,
        colsample_bytree=candidate.colsample_bytree,
        reg_lambda=candidate.reg_lambda,
        gamma=candidate.gamma,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_bin=256,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def xgb_scores(
    classifier: XGBClassifier,
    x_values: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(
        classifier.predict_proba(x_values)[:, 1],
        dtype=float,
    )
    if scores.shape != (len(x_values),) or not np.isfinite(scores).all():
        raise RuntimeError("XGBoost produjo probabilidades invalidas.")
    return scores


def evaluate_candidates_oof(
    data: common.DataView,
    candidates: list[XGBCandidate],
) -> tuple[list[XGBEvaluation], pd.DataFrame]:
    oof_scores = {
        candidate.key: np.full(len(data.x_train), np.nan, dtype=float)
        for candidate in candidates
    }
    variance_rows = []

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        print(f"  Fold {fold}: ajustando PCA64 ponderada...")
        validation_mask = data.train_metadata["fold"].to_numpy() == fold
        training_mask = ~validation_mask
        training_metadata = data.train_metadata.loc[
            training_mask
        ].reset_index(drop=True)
        projection, x_training_pca = raw_pca.fit_weighted_projection(
            data.x_train[training_mask],
            training_metadata,
            PCA_COMPONENTS,
        )
        x_fold_validation_pca = projection.transform(
            data.x_train[validation_mask]
        )
        variance_rows.append(
            {
                "fold": fold,
                "pca_components": PCA_COMPONENTS,
                "cumulative_explained_variance_percent": float(
                    np.sum(projection.explained_variance_ratio) * 100.0
                ),
            }
        )

        y_training = training_metadata["stage2_target"].to_numpy(dtype=int)
        sample_weights = common.compute_training_weights(
            training_metadata
        )
        for candidate_index, candidate in enumerate(candidates, start=1):
            classifier = build_xgb(candidate)
            classifier.fit(
                x_training_pca,
                y_training,
                sample_weight=sample_weights,
            )
            oof_scores[candidate.key][validation_mask] = xgb_scores(
                classifier,
                x_fold_validation_pca,
            )
            print(
                f"    XGB {candidate_index:02d}/{len(candidates):02d}: "
                f"{candidate.key}"
            )

    evaluations = []
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
            XGBEvaluation(
                candidate=candidate,
                threshold=threshold,
                metrics=metrics,
                oof_recordings=recordings,
                fold_metrics=pd.DataFrame(fold_rows),
            )
        )
    return evaluations, pd.DataFrame(variance_rows)


def selection_key(evaluation: XGBEvaluation) -> tuple[float, ...]:
    return (
        float(evaluation.metrics["macro_f1"]),
        float(evaluation.metrics["balanced_accuracy"]),
        float(evaluation.metrics["roc_auc"]),
        -float(evaluation.candidate.theoretical_max_nodes),
    )


def save_oof_results(
    evaluations: list[XGBEvaluation],
    best: XGBEvaluation,
    variance_by_fold: pd.DataFrame,
    result_dir: Path,
    elapsed_seconds: float,
) -> None:
    rows = []
    for evaluation in evaluations:
        candidate = evaluation.candidate
        rows.append(
            {
                "candidate_key": candidate.key,
                **asdict(candidate),
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
    variance_by_fold.to_csv(
        result_dir / "pca64_explained_variance_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )


def booster_complexity(classifier: XGBClassifier) -> dict[str, float | int]:
    booster = classifier.get_booster()
    trees = booster.trees_to_dataframe()
    leaf_mask = trees["Feature"].astype(str) == "Leaf"
    tree_count = int(trees["Tree"].nunique())
    total_nodes = len(trees)
    total_leaves = int(leaf_mask.sum())
    return {
        "xgb_tree_count": tree_count,
        "xgb_total_nodes": int(total_nodes),
        "xgb_total_leaves": total_leaves,
        "xgb_total_split_nodes": int(total_nodes - total_leaves),
        "xgb_mean_nodes_per_tree": float(total_nodes / tree_count),
    }


def train(
    data: common.DataView,
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

    candidates = xgb_candidates(quick)
    print("\n" + "=" * 78)
    print("CV RAW 4096 -> PCA64 PONDERADA -> XGBOOST")
    print("=" * 78)
    print(f"Candidatos XGBoost: {len(candidates)}")
    start_time = time.perf_counter()
    evaluations, variance_by_fold = evaluate_candidates_oof(
        data,
        candidates,
    )
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
        variance_by_fold,
        result_dir,
        elapsed_seconds,
    )
    oof_summary = {
        "dataset": "train_oof",
        "experiment": "raw_pca64_xgb_event",
        "candidate_key": best.candidate.key,
        "threshold": best.threshold,
        "model_size_kb_joblib": np.nan,
        "xgb_native_size_kb": np.nan,
        "pca_float_count": np.nan,
        "estimated_pca_float32_kb": np.nan,
        **best.metrics,
    }
    if quick:
        pd.DataFrame([oof_summary]).to_csv(
            result_dir / "metrics_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\nPrueba rapida XGBoost completada.")
        print(f"Mejor OOF: {best.candidate.key}")
        print(f"Macro-F1 OOF: {best.metrics['macro_f1']:.4f}")
        print("VALIDATION no se ha evaluado.")
        print(f"Resultados: {result_dir}")
        return

    print("\nAjustando PCA64 y XGBoost finales con todo TRAIN...")
    final_projection, x_train_pca = raw_pca.fit_weighted_projection(
        data.x_train,
        data.train_metadata,
        PCA_COMPONENTS,
    )
    x_validation_pca = final_projection.transform(data.x_validation)
    classifier = build_xgb(best.candidate)
    classifier.fit(
        x_train_pca,
        data.train_metadata["stage2_target"].to_numpy(dtype=int),
        sample_weight=common.compute_training_weights(
            data.train_metadata
        ),
    )
    validation_sample_scores = xgb_scores(classifier, x_validation_pca)
    validation_recordings = common.aggregate_scores_by_recording(
        data.validation_metadata,
        validation_sample_scores,
    )
    validation_scores = validation_recordings["score"].to_numpy(float)
    validation_recordings["y_pred"] = (
        validation_scores >= best.threshold
    ).astype(int)
    validation_recordings["candidate_key"] = best.candidate.key
    validation_metrics = common.binary_metrics(
        validation_recordings["y_true"].to_numpy(dtype=int),
        validation_scores,
        best.threshold,
    )

    projection_package = {
        "weighted_mean": final_projection.weighted_mean,
        "components": final_projection.components,
        "component_mean": final_projection.component_mean,
        "component_scale": final_projection.component_scale,
        "singular_values": final_projection.singular_values,
        "explained_variance_ratio": (
            final_projection.explained_variance_ratio
        ),
    }
    complexity = booster_complexity(classifier)
    model_package = {
        "projection": projection_package,
        "classifier": classifier,
        "preset": preset,
        "experiment": "raw_pca64_xgb_event",
        "candidate_key": best.candidate.key,
        "threshold": best.threshold,
        "label_mapping": common.LABEL_TO_NAME,
        "recording_aggregation": "mean_event_probability",
        "raw_input_shape": (64, 64),
        "flatten_order": "C_filter_then_time",
        "pca_components": PCA_COMPONENTS,
        "pca_weighting": "each_recording_equal_via_1_over_event_count",
        "classifier_weighting": (
            "class_balanced_and_1_over_events_per_recording"
        ),
        "smote_used": False,
        "scale_pos_weight_used": False,
        "early_stopping_used": False,
        "unknown_rejection_calibrated": False,
        "random_state": RANDOM_STATE,
        **complexity,
    }
    model_path = model_dir / "raw_pca64_xgb_event_model.joblib"
    native_model_path = model_dir / "raw_pca64_xgb_booster.ubj"
    joblib.dump(model_package, model_path, compress=3)
    classifier.save_model(native_model_path)
    model_size_kb = model_path.stat().st_size / 1024.0
    native_size_kb = native_model_path.stat().st_size / 1024.0
    pca_float_count = int(
        final_projection.weighted_mean.size
        + final_projection.components.size
        + final_projection.component_mean.size
        + final_projection.component_scale.size
    )
    deployment = {
        "model_size_kb_joblib": model_size_kb,
        "xgb_native_size_kb": native_size_kb,
        "pca_float_count": pca_float_count,
        "estimated_pca_float32_kb": pca_float_count * 4.0 / 1024.0,
        **complexity,
    }
    oof_summary.update(deployment)
    validation_summary = {
        "dataset": "validation",
        "experiment": "raw_pca64_xgb_event",
        "candidate_key": best.candidate.key,
        "threshold": best.threshold,
        **deployment,
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
                "pca_components": PCA_COMPONENTS,
                "search_strategy": "deterministic_parameter_sampler",
                "candidate_count": len(candidates),
                "class_balance": "manual_sample_weight",
                "smote_used": False,
                "scale_pos_weight_used": False,
                "early_stopping_used": False,
                "validation_used_for_hyperparameter_selection": False,
                "test_processed": False,
                **asdict(best.candidate),
                **complexity,
            }
        ]
    ).to_csv(
        result_dir / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    graph_path = graph_dir / "validation_raw_pca64_xgb_event.png"
    common.create_validation_graph(
        validation_recordings,
        best.threshold,
        "raw_pca64_xgb_event",
        best.candidate,
        graph_path,
    )
    print("\n" + "=" * 78)
    print("RESULTADO PCA64 + XGBOOST")
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
        f"XGBoost: {complexity['xgb_tree_count']} arboles, "
        f"{complexity['xgb_total_nodes']} nodos"
    )
    print(f"Modelo joblib: {model_size_kb:.2f} KB")
    print(f"Booster nativo: {native_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print(f"Grafica: {graph_path}")
    print("TEST permanece reservado.")


def print_check(data: common.DataView, quick: bool) -> None:
    train_recordings = data.train_metadata.drop_duplicates("original_uuid")
    validation_recordings = data.validation_metadata.drop_duplicates(
        "original_uuid"
    )
    print("=" * 78)
    print("CHECK — PCA64 + XGBOOST")
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
    print(f"PCA fija: {PCA_COMPONENTS} componentes")
    print(f"Candidatos XGBoost: {len(xgb_candidates(quick))}")
    print("Sin SMOTE, scale_pos_weight ni early stopping.")
    print("TEST no sera leido ni procesado.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena XGBoost ligero sobre PCA64 de cocleogramas."
    )
    parser.add_argument(
        "--action",
        choices=["check", "train"],
        default="check",
        help="check valida datos; train ejecuta CV y, salvo quick, validation.",
    )
    parser.add_argument(
        "--preset",
        choices=["paper64"],
        default="paper64",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Prueba tres XGBoost mediante OOF sin evaluar validation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 — COCLEOGRAMA RAW + PCA64 + XGBOOST")
    print("=" * 78)
    print(f"Preset: {args.preset}")
    print(f"Accion: {args.action}")
    print("TEST no sera leido ni procesado.")
    data = raw_pca.load_raw_data(args.preset)
    print_check(data, args.quick)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(data, args.preset, args.quick)


if __name__ == "__main__":
    main()
