"""Busca modelos compactos para Stage 1 con los splits audio-quality.

Familias disponibles:

* ``lr``: StandardScaler + regresion logistica L1/L2.
* ``rf``: Random Forest con pocos arboles y profundidad limitada.

Todos los hiperparametros se seleccionan exclusivamente con predicciones OOF
de TRAIN y umbral fijo 0.5. Validation y TEST solo se evaluan despues de
congelar el ganador de cada familia.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_stage1_cough_no_cough_rf_audio_quality import (
    ROOT,
    THRESHOLD,
    classification_metrics,
    load_split,
    quality_diagnostics,
    save_graphs,
    save_predictions,
)


RANDOM_STATE = 42
MACRO_F1_TOLERANCE = 0.005
RECALL_TOLERANCE = 0.005

LR_C_VALUES = [
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
]
LR_PENALTIES = ["l1", "l2"]
LR_CLASS_WEIGHTS: list[str | None] = [None, "balanced"]

RF_N_ESTIMATORS = [25, 50, 75, 100]
RF_MAX_DEPTHS = [6, 8, 10, 12]
RF_MIN_SAMPLES_LEAF = [2, 5]
RF_FIXED_PARAMS = {
    "min_samples_split": 5,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busqueda OOF de LR o Random Forest compacto para Stage 1"
    )
    parser.add_argument("--family", choices=["lr", "rf"], required=True)
    parser.add_argument(
        "--split-mode",
        choices=["audio_quality", "random"],
        default="audio_quality",
        help="Usa los splits con calidad forzada o los splits aleatorios actuales.",
    )
    parser.add_argument(
        "--test-quality", choices=["poor", "ok"], default="poor"
    )
    return parser.parse_args()


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def candidate_grid(family: str) -> list[dict[str, Any]]:
    if family == "lr":
        return [
            {
                "C": C,
                "penalty": penalty,
                "class_weight": class_weight,
            }
            for C in LR_C_VALUES
            for penalty in LR_PENALTIES
            for class_weight in LR_CLASS_WEIGHTS
        ]
    return [
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        }
        for n_estimators in RF_N_ESTIMATORS
        for max_depth in RF_MAX_DEPTHS
        for min_samples_leaf in RF_MIN_SAMPLES_LEAF
    ]


def candidate_name(family: str, params: dict[str, Any]) -> str:
    if family == "lr":
        weight = params["class_weight"] or "none"
        return (
            f"lr__C{format_number(params['C'])}__{params['penalty']}"
            f"__weight_{weight}"
        )
    return (
        f"rf__trees{params['n_estimators']}__depth{params['max_depth']}"
        f"__leaf{params['min_samples_leaf']}"
    )


def make_model(family: str, params: dict[str, Any]) -> Any:
    if family == "lr":
        classifier = LogisticRegression(
            C=float(params["C"]),
            penalty=str(params["penalty"]),
            solver="liblinear",
            class_weight=params["class_weight"],
            max_iter=5_000,
            random_state=RANDOM_STATE,
        )
        return Pipeline(
            [("scaler", StandardScaler()), ("classifier", classifier)]
        )
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        **RF_FIXED_PARAMS,
    )


def model_complexity(family: str, model: Any) -> dict[str, int]:
    if family == "lr":
        classifier = model.named_steps["classifier"]
        scaler = model.named_steps["scaler"]
        return {
            "trainable_parameters": int(
                classifier.coef_.size + classifier.intercept_.size
            ),
            "deployment_float_values": int(
                classifier.coef_.size
                + classifier.intercept_.size
                + scaler.mean_.size
                + scaler.scale_.size
            ),
            "tree_nodes_total": 0,
        }
    return {
        "trainable_parameters": 0,
        "deployment_float_values": 0,
        "tree_nodes_total": int(
            sum(tree.tree_.node_count for tree in model.estimators_)
        ),
    }


def run_candidate_cv(
    family: str,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    name = candidate_name(family, params)
    oof_probability = np.zeros(len(y), dtype=np.float64)
    oof_prediction = np.zeros(len(y), dtype=np.int8)
    fold_rows: list[dict[str, Any]] = []
    nodes_by_fold: list[int] = []
    total_fit_seconds = 0.0

    for fold in sorted(np.unique(folds)):
        inner_validation = folds == fold
        inner_train = ~inner_validation
        model = make_model(family, params)
        started = time.perf_counter()
        model.fit(X[inner_train], y[inner_train])
        fit_seconds = time.perf_counter() - started
        total_fit_seconds += fit_seconds
        probability = model.predict_proba(X[inner_validation])[:, 1]
        prediction = (probability >= THRESHOLD).astype(np.int8)
        oof_probability[inner_validation] = probability
        oof_prediction[inner_validation] = prediction
        complexity = model_complexity(family, model)
        nodes_by_fold.append(complexity["tree_nodes_total"])
        fold_rows.append(
            {
                "candidate": name,
                "fold": int(fold),
                "n_train": int(inner_train.sum()),
                "n_validation": int(inner_validation.sum()),
                "fit_seconds": fit_seconds,
                **classification_metrics(
                    y[inner_validation], prediction, probability
                ),
                **complexity,
            }
        )

    metrics = classification_metrics(y, oof_prediction, oof_probability)
    row: dict[str, Any] = {
        "candidate": name,
        "family": family,
        **params,
        "threshold": THRESHOLD,
        **metrics,
        "cv_fit_seconds": total_fit_seconds,
        "mean_tree_nodes_cv": float(np.mean(nodes_by_fold)),
    }
    return row, oof_prediction, oof_probability, fold_rows


def baseline_oof_metrics(
    split_mode: str,
    test_quality: str,
) -> tuple[float, float]:
    experiment_dir = (
        "mfcc117_rf"
        if split_mode == "random"
        else f"mfcc117_rf_audio_quality_{test_quality}"
    )
    path = (
        ROOT
        / "results_stage1_cough_no_cough"
        / experiment_dir
        / "full"
        / "metrics_summary.csv"
    )
    if not path.is_file():
        return 0.0, 0.0
    metrics = pd.read_csv(path)
    row = metrics[metrics["dataset"] == "train_oof"]
    if row.empty:
        return 0.0, 0.0
    return float(row.iloc[0]["macro_f1"]), float(row.iloc[0]["recall_cough"])


def select_winner(
    family: str,
    candidate_results: pd.DataFrame,
    split_mode: str,
    test_quality: str,
) -> tuple[pd.Series, str]:
    if family == "lr":
        ordered = candidate_results.sort_values(
            ["macro_f1", "recall_cough", "roc_auc"],
            ascending=[False, False, False],
        )
        return ordered.iloc[0], "maximum OOF macro-F1; ties by recall and AUC"

    baseline_macro_f1, baseline_recall = baseline_oof_metrics(
        split_mode, test_quality
    )
    macro_floor = baseline_macro_f1 - MACRO_F1_TOLERANCE
    recall_floor = baseline_recall - RECALL_TOLERANCE
    eligible = candidate_results[
        (candidate_results["macro_f1"] >= macro_floor)
        & (candidate_results["recall_cough"] >= recall_floor)
    ]
    if not eligible.empty:
        ordered = eligible.sort_values(
            ["mean_tree_nodes_cv", "n_estimators", "macro_f1", "recall_cough"],
            ascending=[True, True, False, False],
        )
        rule = (
            "smallest forest within 0.005 OOF macro-F1 and recall of the "
            f"300-tree baseline (floors={macro_floor:.6f}/{recall_floor:.6f})"
        )
        return ordered.iloc[0], rule

    ordered = candidate_results.sort_values(
        ["macro_f1", "recall_cough", "mean_tree_nodes_cv"],
        ascending=[False, False, True],
    )
    return ordered.iloc[0], (
        "fallback: no compact RF met both baseline tolerances; maximum OOF macro-F1"
    )


def benchmark_inference(model: Any, X: np.ndarray) -> dict[str, float | int]:
    sample_count = min(len(X), 1_000)
    batch = X[:sample_count]
    model.predict_proba(batch[: min(32, sample_count)])

    batch_repeats = 30
    started = time.perf_counter()
    for _ in range(batch_repeats):
        model.predict_proba(batch)
    batch_elapsed = time.perf_counter() - started

    single_repeats = 500
    single = batch[:1]
    started = time.perf_counter()
    for _ in range(single_repeats):
        model.predict_proba(single)
    single_elapsed = time.perf_counter() - started
    return {
        "benchmark_samples": sample_count,
        "desktop_batch_ms_per_sample": (
            1_000 * batch_elapsed / (batch_repeats * sample_count)
        ),
        "desktop_single_sample_ms": 1_000 * single_elapsed / single_repeats,
    }


def main() -> None:
    args = parse_args()
    family = args.family
    if args.split_mode == "random":
        features_dir = ROOT / "features_extracted_stage1_random" / "mfcc117"
        experiment_name = (
            f"mfcc117_{'lr' if family == 'lr' else 'rf_compact'}_random"
        )
    else:
        features_dir = (
            ROOT
            / f"features_extracted_stage1_audio_quality_{args.test_quality}"
            / "mfcc117"
        )
        experiment_name = (
            f"mfcc117_{'lr' if family == 'lr' else 'rf_compact'}"
            f"_audio_quality_{args.test_quality}"
        )
    if not features_dir.is_dir():
        extraction_command = (
            "python .\\feature_extraction_stage1_mfcc_audio_quality.py "
            f"--split-mode {args.split_mode}"
        )
        if args.split_mode == "audio_quality":
            extraction_command += f" --test-quality {args.test_quality}"
        raise FileNotFoundError(
            f"No existe {features_dir}. Ejecuta primero:\n{extraction_command}"
        )
    results_dir = ROOT / "results_stage1_cough_no_cough" / experiment_name / "full"
    graphs_dir = (
        ROOT / "graphs_results_stage1_cough_no_cough" / experiment_name / "full"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, manifest_train = load_split(features_dir, "train")
    X_validation, y_validation, manifest_validation = load_split(
        features_dir, "validation"
    )
    X_test, y_test, manifest_test = load_split(features_dir, "test")
    folds = np.load(features_dir / "folds_train.npy").astype(int)
    if len(folds) != len(y_train):
        raise ValueError("folds_train no esta alineado con TRAIN")
    if not np.array_equal(folds, manifest_train["fold"].to_numpy(dtype=int)):
        raise ValueError("Los folds no coinciden con el manifiesto")
    fold_counts = (
        pd.DataFrame(
            {
                "uuid": manifest_train["original_uuid"].astype(str),
                "fold": folds,
            }
        )
        .groupby("uuid")["fold"]
        .nunique()
    )
    if (fold_counts > 1).any():
        raise ValueError("Una grabacion aparece en varios folds")

    candidates = candidate_grid(family)
    print("=" * 78)
    print(f"STAGE 1 - BUSQUEDA COMPACTA - {family.upper()}")
    print("=" * 78)
    print(f"Candidatos: {len(candidates)}")
    print(f"Modo de split: {args.split_mode}")
    print(f"TRAIN: {X_train.shape}; no_tos/tos={np.bincount(y_train).tolist()}")
    print("Seleccion exclusivamente OOF de TRAIN; umbral fijo 0.5.")

    rows: list[dict[str, Any]] = []
    oof_by_candidate: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    folds_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for index, params in enumerate(candidates, start=1):
        row, prediction, probability, fold_rows = run_candidate_cv(
            family, params, X_train, y_train, folds
        )
        rows.append(row)
        oof_by_candidate[row["candidate"]] = (prediction, probability)
        folds_by_candidate[row["candidate"]] = fold_rows
        print(
            f"[{index:02d}/{len(candidates):02d}] {row['candidate']} | "
            f"macro-F1={row['macro_f1']:.4f} | recall={row['recall_cough']:.4f} | "
            f"AUC={row['roc_auc']:.4f}"
        )

    candidate_results = pd.DataFrame(rows)
    winner, selection_rule = select_winner(
        family, candidate_results, args.split_mode, args.test_quality
    )
    winner_name = str(winner["candidate"])
    winner_params = {
        key: winner[key]
        for key in (
            ["C", "penalty", "class_weight"]
            if family == "lr"
            else ["n_estimators", "max_depth", "min_samples_leaf"]
        )
    }
    if family == "lr" and pd.isna(winner_params["class_weight"]):
        winner_params["class_weight"] = None
    if family == "rf":
        for key in ["n_estimators", "max_depth", "min_samples_leaf"]:
            winner_params[key] = int(winner_params[key])

    candidate_results["selected"] = candidate_results["candidate"] == winner_name
    candidate_results.to_csv(results_dir / "candidate_cv_results.csv", index=False)
    pd.DataFrame(folds_by_candidate[winner_name]).to_csv(
        results_dir / "best_cv_fold_metrics.csv", index=False
    )
    oof_prediction, oof_probability = oof_by_candidate[winner_name]
    oof_output = manifest_train.copy()
    oof_output["y_true"] = y_train
    oof_output["y_pred"] = oof_prediction
    oof_output["probability_cough"] = oof_probability
    oof_output["threshold"] = THRESHOLD
    oof_output.to_csv(results_dir / "best_oof_predictions.csv", index=False)

    print("\nGanador OOF:", winner_name)
    print("Regla:", selection_rule)
    print(
        f"Macro-F1={winner['macro_f1']:.4f} | recall={winner['recall_cough']:.4f} | "
        f"AUC={winner['roc_auc']:.4f}"
    )
    print("\nAjustando el ganador exclusivamente con todo TRAIN...")
    final_model = make_model(family, winner_params)
    started = time.perf_counter()
    final_model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    metrics_rows: list[dict[str, Any]] = [
        {
            "dataset": "train_oof",
            "threshold": THRESHOLD,
            **classification_metrics(y_train, oof_prediction, oof_probability),
        }
    ]
    test_prediction: np.ndarray | None = None
    test_probability: np.ndarray | None = None
    for split, X, y, manifest in [
        ("train_fit", X_train, y_train, manifest_train),
        ("validation", X_validation, y_validation, manifest_validation),
        ("test", X_test, y_test, manifest_test),
    ]:
        probability = final_model.predict_proba(X)[:, 1]
        prediction = (probability >= THRESHOLD).astype(np.int8)
        metrics = classification_metrics(y, prediction, probability)
        metrics_rows.append(
            {"dataset": split, "threshold": THRESHOLD, **metrics}
        )
        if split != "train_fit":
            save_predictions(
                results_dir, split, manifest, y, prediction, probability
            )
            save_graphs(graphs_dir, split, y, prediction, probability)
        if split == "test":
            test_prediction = prediction
            test_probability = probability
        print(
            f"{split:>10}: F1={metrics['f1_cough']:.4f} | "
            f"macro-F1={metrics['macro_f1']:.4f} | "
            f"recall={metrics['recall_cough']:.4f} | AUC={metrics['roc_auc']:.4f}"
        )

    assert test_prediction is not None and test_probability is not None
    quality_recall, quality_binary = quality_diagnostics(
        manifest_test, y_test, test_prediction, test_probability
    )
    quality_recall.to_csv(results_dir / "test_cough_recall_by_quality.csv", index=False)
    quality_binary.to_csv(results_dir / "test_binary_metrics_by_quality.csv", index=False)
    metrics_rows.extend(quality_binary.to_dict(orient="records"))
    pd.DataFrame(metrics_rows).to_csv(results_dir / "metrics_summary.csv", index=False)

    model_path = results_dir / f"stage1_{family}_compact.joblib"
    joblib.dump(final_model, model_path)
    model_size_kb = model_path.stat().st_size / 1024
    complexity = model_complexity(family, final_model)
    benchmark = benchmark_inference(final_model, X_validation)
    configuration: dict[str, Any] = {
        "experiment": experiment_name,
        "family": family,
        "winner": winner_name,
        "selection_rule": selection_rule,
        "feature_source": str(features_dir),
        "split_mode": args.split_mode,
        "target_mapping": "0=no_cough; 1=dry|wet|unknown",
        "test_quality_forced": (
            args.test_quality
            if args.split_mode == "audio_quality"
            else "not_applicable"
        ),
        "n_features": X_train.shape[1],
        "threshold": THRESHOLD,
        "threshold_selection": "fixed; no threshold tuning",
        "fit_final_seconds": fit_seconds,
        "model_size_kb": model_size_kb,
        **winner_params,
        **complexity,
        **benchmark,
        "benchmark_note": "desktop Python reference; not a mobile latency measurement",
    }
    pd.DataFrame(configuration.items(), columns=["parameter", "value"]).to_csv(
        results_dir / "experiment_configuration.csv", index=False
    )

    print("\n" + "=" * 78)
    print(f"RESULTADO STAGE 1 COMPACTO - {family.upper()}")
    print("=" * 78)
    print(f"Ganador: {winner_name}")
    print(f"Modelo: {model_size_kb:.2f} KB")
    print(
        f"Inferencia escritorio: {benchmark['desktop_single_sample_ms']:.4f} ms/muestra"
    )
    for row in quality_recall.to_dict(orient="records"):
        print(
            f"Recall TEST {row['cough_quality']}: {row['recall_cough']:.4f}"
        )
    print(f"Resultados: {results_dir}")

    try:
        from build_stage1_experiments_excel import build_workbook

        excel_path = build_workbook()
        print(f"Excel actualizado: {excel_path}")
    except Exception as exc:
        print(f"AVISO: no se pudo actualizar el Excel automaticamente: {exc}")


if __name__ == "__main__":
    main()
