"""Busqueda OOF de LR para Stage 1 con toses positivas de FSD50K.

La busqueda se divide en dos fases:

Fase A
    StandardScaler + LogisticRegression variando C, penalizacion y pesos de
    clase. El scaler y la LR se ajustan exclusivamente con el TRAIN interno de
    cada fold.

Fase B
    Toma las tres mejores representaciones de la fase A y prueba pesos
    especificos para las nuevas toses FSD50K. El peso solo se aplica a esas
    muestras dentro del TRAIN interno de cada fold.

Para cada candidato se selecciona tambien un umbral usando exclusivamente sus
predicciones OOF. El candidato debe mantener unos suelos respecto al baseline
fijo y, entre los candidatos elegibles, maximiza la balanced accuracy dentro
de FSD50K. Asi se evita que las 121 nuevas toses queden ocultas por las miles
de muestras del resto del dataset.

Validation se consulta una vez despues de congelar candidato y umbral. TEST no
se lee hasta ejecutar explicitamente ``--action test``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_stage1_cough_no_cough_rf_audio_quality import (
    ROOT,
    classification_metrics,
    load_split,
    quality_diagnostics,
)
from train_stage1_mfcc_lr_fsd50k_coughs import (
    FEATURES_DIR,
    assert_split_groups_are_disjoint,
    benchmark_inference,
    model_complexity,
    parse_bool_series,
    validate_feature_configuration,
    validate_fold_isolation,
    validate_manifest,
)


RANDOM_STATE = 42
BASELINE_CANDIDATE = "lr__C0p3__l1__weight_none"
BASELINE_THRESHOLD = 0.5

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
NEW_FSD50K_WEIGHTS = [1.0, 1.5, 2.0, 3.0, 4.0]
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)

MACRO_F1_TOLERANCE = 0.01
SPECIFICITY_TOLERANCE = 0.01
ORIGINAL_COUGH_RECALL_TOLERANCE = 0.02

EXPERIMENT_NAME = "mfcc117_lr_fsd50k_coughs_search_random"
RESULTS_DIR = ROOT / "results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
GRAPHS_DIR = (
    ROOT / "graphs_results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
)
MODEL_PATH = RESULTS_DIR / "stage1_lr_fsd50k_coughs_search.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busqueda OOF de LR Stage 1 con nuevas toses FSD50K"
    )
    parser.add_argument(
        "--action",
        choices=["train", "test"],
        default="train",
        help=(
            "train ejecuta la busqueda, ajusta TRAIN y evalua VALIDATION; "
            "test evalua el modelo ya congelado"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar resultados derivados de la misma accion",
    )
    return parser.parse_args()


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def candidate_name(params: dict[str, Any], source_weight: float) -> str:
    class_weight = params["class_weight"] or "none"
    return (
        f"lr__C{format_number(float(params['C']))}"
        f"__{params['penalty']}__weight_{class_weight}"
        f"__fsdposw{format_number(source_weight)}"
    )


def phase_a_name(params: dict[str, Any]) -> str:
    class_weight = params["class_weight"] or "none"
    return (
        f"lr__C{format_number(float(params['C']))}"
        f"__{params['penalty']}__weight_{class_weight}"
    )


def make_model(params: dict[str, Any]) -> Pipeline:
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


def phase_a_grid() -> list[dict[str, Any]]:
    return [
        {"C": C, "penalty": penalty, "class_weight": class_weight}
        for C in LR_C_VALUES
        for penalty in LR_PENALTIES
        for class_weight in LR_CLASS_WEIGHTS
    ]


def run_oof_candidate(
    params: dict[str, Any],
    source_weight: float,
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    is_new_fsd50k_cough: np.ndarray,
) -> tuple[np.ndarray, float, list[int]]:
    probability = np.zeros(len(y), dtype=np.float64)
    total_fit_seconds = 0.0
    iterations_by_fold: list[int] = []

    for fold in sorted(np.unique(folds)):
        inner_validation = folds == fold
        inner_train = ~inner_validation
        sample_weight = np.ones(int(inner_train.sum()), dtype=np.float64)
        new_in_inner_train = is_new_fsd50k_cough[inner_train]
        sample_weight[new_in_inner_train] = source_weight

        model = make_model(params)
        started = time.perf_counter()
        model.fit(
            X[inner_train],
            y[inner_train],
            classifier__sample_weight=sample_weight,
        )
        total_fit_seconds += time.perf_counter() - started
        probability[inner_validation] = model.predict_proba(
            X[inner_validation]
        )[:, 1]
        iterations_by_fold.append(
            int(np.max(model.named_steps["classifier"].n_iter_))
        )

    if not np.isfinite(probability).all():
        raise ValueError("Las probabilidades OOF contienen NaN o infinitos")
    return probability, total_fit_seconds, iterations_by_fold


def threshold_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    manifest: pd.DataFrame,
    is_new_fsd50k_cough: np.ndarray,
) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(np.int8)
    global_metrics = classification_metrics(y, prediction, probability)

    fsd_mask = manifest["dataset_origin"].astype(str).eq("FSD50K").to_numpy()
    if np.unique(y[fsd_mask]).size != 2:
        raise ValueError("El subconjunto FSD50K OOF no contiene ambas clases")
    fsd_metrics = classification_metrics(
        y[fsd_mask], prediction[fsd_mask], probability[fsd_mask]
    )

    original_cough = (y == 1) & ~is_new_fsd50k_cough
    new_fsd50k_cough = (y == 1) & is_new_fsd50k_cough
    if not original_cough.any() or not new_fsd50k_cough.any():
        raise ValueError("Falta alguna de las fuentes positivas en TRAIN")

    return {
        "threshold": float(threshold),
        "global_macro_f1": float(global_metrics["macro_f1"]),
        "global_balanced_accuracy": float(global_metrics["balanced_accuracy"]),
        "global_precision_cough": float(global_metrics["precision_cough"]),
        "global_recall_cough": float(global_metrics["recall_cough"]),
        "global_specificity_no_cough": float(
            global_metrics["specificity_no_cough"]
        ),
        "global_roc_auc": float(global_metrics["roc_auc"]),
        "original_cough_recall": float(np.mean(prediction[original_cough] == 1)),
        "new_fsd50k_cough_recall": float(
            np.mean(prediction[new_fsd50k_cough] == 1)
        ),
        "fsd50k_balanced_accuracy": float(fsd_metrics["balanced_accuracy"]),
        "fsd50k_macro_f1": float(fsd_metrics["macro_f1"]),
        "fsd50k_precision_cough": float(fsd_metrics["precision_cough"]),
        "fsd50k_recall_cough": float(fsd_metrics["recall_cough"]),
        "fsd50k_specificity_no_cough": float(
            fsd_metrics["specificity_no_cough"]
        ),
        "fsd50k_roc_auc": float(fsd_metrics["roc_auc"]),
        "fsd50k_false_positives": int(fsd_metrics["fp_no_cough_as_cough"]),
        "fsd50k_false_negatives": int(fsd_metrics["fn_cough_as_no_cough"]),
    }


def add_constraint_columns(
    row: dict[str, Any],
    floors: dict[str, float],
) -> dict[str, Any]:
    checks = {
        "passes_macro_f1_floor": (
            row["global_macro_f1"] >= floors["global_macro_f1"]
        ),
        "passes_specificity_floor": (
            row["global_specificity_no_cough"]
            >= floors["global_specificity_no_cough"]
        ),
        "passes_original_recall_floor": (
            row["original_cough_recall"] >= floors["original_cough_recall"]
        ),
    }
    row.update(checks)
    row["constraints_passed"] = int(sum(bool(value) for value in checks.values()))
    row["eligible"] = bool(all(checks.values()))
    row["threshold_distance_from_0p5"] = abs(float(row["threshold"]) - 0.5)
    return row


def threshold_search(
    phase: str,
    name: str,
    params: dict[str, Any],
    source_weight: float,
    y: np.ndarray,
    probability: np.ndarray,
    manifest: pd.DataFrame,
    is_new_fsd50k_cough: np.ndarray,
    floors: dict[str, float],
    fit_seconds: float,
    iterations_by_fold: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    threshold_rows: list[dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        row = threshold_metrics(
            y=y,
            probability=probability,
            threshold=float(threshold),
            manifest=manifest,
            is_new_fsd50k_cough=is_new_fsd50k_cough,
        )
        row.update(
            {
                "phase": phase,
                "candidate": name,
                "C": float(params["C"]),
                "penalty": str(params["penalty"]),
                "class_weight": params["class_weight"] or "none",
                "new_fsd50k_cough_weight": float(source_weight),
            }
        )
        threshold_rows.append(add_constraint_columns(row, floors))

    frame = pd.DataFrame(threshold_rows)
    eligible = frame.loc[frame["eligible"]].copy()
    ranking_frame = eligible if not eligible.empty else frame
    ordered = ranking_frame.sort_values(
        [
            "constraints_passed",
            "fsd50k_balanced_accuracy",
            "new_fsd50k_cough_recall",
            "global_macro_f1",
            "threshold_distance_from_0p5",
        ],
        ascending=[False, False, False, False, True],
    )
    best = ordered.iloc[0].to_dict()
    best.update(
        {
            "cv_fit_seconds": fit_seconds,
            "mean_solver_iterations": float(np.mean(iterations_by_fold)),
            "max_solver_iterations": int(np.max(iterations_by_fold)),
        }
    )
    return best, threshold_rows


def rank_candidates(frame: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    eligible = frame.loc[frame["eligible"]].copy()
    # Para el ganador final se restringe a candidatos elegibles. Para pasar
    # tres configuraciones a fase B se completa, si hiciera falta, con las que
    # satisfacen mas restricciones; de ese modo la fase siempre prueba tres
    # representaciones diferentes.
    ranking_frame = (
        eligible if n is None and not eligible.empty else frame.copy()
    )
    ordered = ranking_frame.sort_values(
        [
            "eligible",
            "constraints_passed",
            "fsd50k_balanced_accuracy",
            "new_fsd50k_cough_recall",
            "global_macro_f1",
            "threshold_distance_from_0p5",
        ],
        ascending=[False, False, False, False, False, True],
    )
    return ordered if n is None else ordered.head(n)


def fold_metrics_for_winner(
    name: str,
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    folds: np.ndarray,
    manifest: pd.DataFrame,
    is_new_fsd50k_cough: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prediction = (probability >= threshold).astype(np.int8)
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        metrics = threshold_metrics(
            y=y[mask],
            probability=probability[mask],
            threshold=threshold,
            manifest=manifest.loc[mask].reset_index(drop=True),
            is_new_fsd50k_cough=is_new_fsd50k_cough[mask],
        )
        global_metrics = classification_metrics(
            y[mask], prediction[mask], probability[mask]
        )
        rows.append(
            {
                "candidate": name,
                "fold": int(fold),
                "n_validation": int(mask.sum()),
                "threshold": threshold,
                **global_metrics,
                "original_cough_recall": metrics["original_cough_recall"],
                "new_fsd50k_cough_recall": metrics[
                    "new_fsd50k_cough_recall"
                ],
                "fsd50k_balanced_accuracy": metrics[
                    "fsd50k_balanced_accuracy"
                ],
                "fsd50k_specificity_no_cough": metrics[
                    "fsd50k_specificity_no_cough"
                ],
            }
        )
    return pd.DataFrame(rows)


def source_diagnostics(
    split: str,
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = manifest["dataset_origin"].astype(str)
    is_new = parse_bool_series(
        manifest["is_new_fsd50k_cough"], "is_new_fsd50k_cough"
    ).to_numpy()
    origin_rows: list[dict[str, Any]] = []
    for origin in sorted(origins.unique()):
        mask = origins.eq(origin).to_numpy()
        origin_rows.append(
            {
                "dataset": f"{split}_{origin.lower()}",
                "dataset_origin": origin,
                "threshold": threshold,
                **classification_metrics(
                    y_true[mask], y_pred[mask], y_prob[mask]
                ),
            }
        )

    recall_rows: list[dict[str, Any]] = []
    for source, mask in {
        "original_cough": (y_true == 1) & ~is_new,
        "new_fsd50k_cough": (y_true == 1) & is_new,
    }.items():
        if not mask.any():
            continue
        recall_rows.append(
            {
                "split": split,
                "cough_source": source,
                "threshold": threshold,
                "n_cough_segments": int(mask.sum()),
                "n_cough_recordings": int(
                    manifest.loc[mask, "original_uuid"].astype(str).nunique()
                ),
                "recall_cough": float(np.mean(y_pred[mask] == 1)),
                "false_negative_rate": float(np.mean(y_pred[mask] == 0)),
                "mean_probability_cough": float(np.mean(y_prob[mask])),
                "median_probability_cough": float(np.median(y_prob[mask])),
            }
        )
    return pd.DataFrame(origin_rows), pd.DataFrame(recall_rows)


def save_predictions(
    split: str,
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> None:
    output = manifest.copy()
    output["y_true"] = y_true
    output["y_pred"] = y_pred
    output["probability_cough"] = y_prob
    output["threshold"] = threshold
    output.to_csv(RESULTS_DIR / f"{split}_predictions.csv", index=False)


def save_graphs(
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    image = ax.imshow(cm, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(cm[row, column]), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No tos", "Tos"],
        yticklabels=["No tos", "Tos"],
        xlabel="Prediccion",
        ylabel="Etiqueta real",
        title=f"Stage 1 LR FSD50K search - {split} (t={threshold:.2f})",
    )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / f"confusion_matrix_{split}.png", dpi=180)
    plt.close(fig)

    if np.unique(y_true).size == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5.6, 4.7))
        ax.plot(fpr, tpr, label=f"LR (AUC={auc:.4f})")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set(
            xlabel="False positive rate",
            ylabel="Recall tos",
            title=f"ROC Stage 1 LR FSD50K search - {split}",
        )
        ax.legend(loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(GRAPHS_DIR / f"roc_curve_{split}.png", dpi=180)
        plt.close(fig)


def update_excel() -> None:
    try:
        from build_stage1_experiments_excel import build_workbook

        excel_path = build_workbook()
        print(f"Excel actualizado: {excel_path}")
    except Exception as exc:
        print(f"AVISO: no se pudo actualizar el Excel automaticamente: {exc}")


def train(overwrite: bool) -> None:
    if MODEL_PATH.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {MODEL_PATH}. Usa --overwrite para repetir la busqueda."
        )

    X_train, y_train, manifest_train = load_split(FEATURES_DIR, "train")
    X_validation, y_validation, manifest_validation = load_split(
        FEATURES_DIR, "validation"
    )
    is_new_train_series = validate_manifest(manifest_train, y_train, "train")
    is_new_validation = validate_manifest(
        manifest_validation, y_validation, "validation"
    )
    is_new_train = is_new_train_series.to_numpy()
    folds = np.load(FEATURES_DIR / "folds_train.npy").astype(int)
    validate_fold_isolation(manifest_train, folds)
    assert_split_groups_are_disjoint(
        {"train": manifest_train, "validation": manifest_validation}
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("STAGE 1 - BUSQUEDA LR + NUEVAS TOSES FSD50K")
    print("=" * 78)
    print(f"TRAIN: {X_train.shape}; no_tos/tos={np.bincount(y_train).tolist()}")
    print(
        f"Nuevas toses FSD50K TRAIN/VALIDATION: "
        f"{int(is_new_train.sum())}/{int(is_new_validation.sum())}"
    )
    print("Seleccion de configuracion y umbral: solo OOF de TRAIN.")
    print("TEST no sera leido ni evaluado.")

    phase_a_specs = phase_a_grid()
    phase_a_runs: dict[str, dict[str, Any]] = {}
    print("\n" + "=" * 78)
    print("FASE A - C, PENALIZACION Y CLASS_WEIGHT")
    print("=" * 78)
    for index, params in enumerate(phase_a_specs, start=1):
        name = phase_a_name(params)
        probability, fit_seconds, iterations = run_oof_candidate(
            params=params,
            source_weight=1.0,
            X=X_train,
            y=y_train,
            folds=folds,
            is_new_fsd50k_cough=is_new_train,
        )
        phase_a_runs[name] = {
            "params": params,
            "probability": probability,
            "fit_seconds": fit_seconds,
            "iterations": iterations,
        }
        at_half = threshold_metrics(
            y_train,
            probability,
            BASELINE_THRESHOLD,
            manifest_train,
            is_new_train,
        )
        print(
            f"[{index:02d}/{len(phase_a_specs):02d}] {name} | "
            f"macro-F1@0.5={at_half['global_macro_f1']:.4f} | "
            f"recall FSD+@0.5={at_half['new_fsd50k_cough_recall']:.4f} | "
            f"FSD bal-acc@0.5={at_half['fsd50k_balanced_accuracy']:.4f}"
        )

    if BASELINE_CANDIDATE not in phase_a_runs:
        raise RuntimeError("No se encontro el candidato baseline en la fase A")
    baseline_run = phase_a_runs[BASELINE_CANDIDATE]
    baseline_metrics = threshold_metrics(
        y_train,
        baseline_run["probability"],
        BASELINE_THRESHOLD,
        manifest_train,
        is_new_train,
    )
    floors = {
        "global_macro_f1": (
            baseline_metrics["global_macro_f1"] - MACRO_F1_TOLERANCE
        ),
        "global_specificity_no_cough": (
            baseline_metrics["global_specificity_no_cough"]
            - SPECIFICITY_TOLERANCE
        ),
        "original_cough_recall": (
            baseline_metrics["original_cough_recall"]
            - ORIGINAL_COUGH_RECALL_TOLERANCE
        ),
    }
    print("\nSuelos OOF derivados del baseline fijo @0.5:")
    print(f"  Macro-F1 global >= {floors['global_macro_f1']:.4f}")
    print(
        "  Especificidad global >= "
        f"{floors['global_specificity_no_cough']:.4f}"
    )
    print(
        f"  Recall tos original >= {floors['original_cough_recall']:.4f}"
    )

    phase_a_best_rows: list[dict[str, Any]] = []
    phase_a_threshold_rows: list[dict[str, Any]] = []
    for name, run in phase_a_runs.items():
        best, threshold_rows = threshold_search(
            phase="A",
            name=name,
            params=run["params"],
            source_weight=1.0,
            y=y_train,
            probability=run["probability"],
            manifest=manifest_train,
            is_new_fsd50k_cough=is_new_train,
            floors=floors,
            fit_seconds=run["fit_seconds"],
            iterations_by_fold=run["iterations"],
        )
        phase_a_best_rows.append(best)
        phase_a_threshold_rows.extend(threshold_rows)

    phase_a_results = pd.DataFrame(phase_a_best_rows)
    top_phase_a = rank_candidates(phase_a_results, n=3)
    top_names = top_phase_a["candidate"].astype(str).tolist()
    phase_a_results["selected_for_phase_b"] = phase_a_results[
        "candidate"
    ].isin(top_names)
    phase_a_results.to_csv(
        RESULTS_DIR / "phase_a_candidate_cv_results.csv", index=False
    )
    pd.DataFrame(phase_a_threshold_rows).to_csv(
        RESULTS_DIR / "phase_a_threshold_results.csv", index=False
    )
    print("\nTres configuraciones que pasan a fase B:")
    for row in top_phase_a.to_dict(orient="records"):
        print(
            f"  {row['candidate']} | t={row['threshold']:.2f} | "
            f"FSD bal-acc={row['fsd50k_balanced_accuracy']:.4f} | "
            f"recall FSD+={row['new_fsd50k_cough_recall']:.4f} | "
            f"macro-F1={row['global_macro_f1']:.4f}"
        )

    print("\n" + "=" * 78)
    print("FASE B - PESO ESPECIFICO DE LAS NUEVAS TOSES FSD50K")
    print("=" * 78)
    phase_b_runs: dict[str, dict[str, Any]] = {}
    phase_b_total = len(top_names) * len(NEW_FSD50K_WEIGHTS)
    run_index = 0
    for phase_a_candidate in top_names:
        params = phase_a_runs[phase_a_candidate]["params"]
        for source_weight in NEW_FSD50K_WEIGHTS:
            run_index += 1
            name = candidate_name(params, source_weight)
            probability, fit_seconds, iterations = run_oof_candidate(
                params=params,
                source_weight=source_weight,
                X=X_train,
                y=y_train,
                folds=folds,
                is_new_fsd50k_cough=is_new_train,
            )
            phase_b_runs[name] = {
                "params": params,
                "source_weight": source_weight,
                "probability": probability,
                "fit_seconds": fit_seconds,
                "iterations": iterations,
            }
            at_half = threshold_metrics(
                y_train,
                probability,
                BASELINE_THRESHOLD,
                manifest_train,
                is_new_train,
            )
            print(
                f"[{run_index:02d}/{phase_b_total:02d}] {name} | "
                f"macro-F1@0.5={at_half['global_macro_f1']:.4f} | "
                f"recall FSD+@0.5={at_half['new_fsd50k_cough_recall']:.4f} | "
                f"FSD bal-acc@0.5={at_half['fsd50k_balanced_accuracy']:.4f}"
            )

    phase_b_best_rows: list[dict[str, Any]] = []
    phase_b_threshold_rows: list[dict[str, Any]] = []
    for name, run in phase_b_runs.items():
        best, threshold_rows = threshold_search(
            phase="B",
            name=name,
            params=run["params"],
            source_weight=run["source_weight"],
            y=y_train,
            probability=run["probability"],
            manifest=manifest_train,
            is_new_fsd50k_cough=is_new_train,
            floors=floors,
            fit_seconds=run["fit_seconds"],
            iterations_by_fold=run["iterations"],
        )
        phase_b_best_rows.append(best)
        phase_b_threshold_rows.extend(threshold_rows)

    phase_b_results = pd.DataFrame(phase_b_best_rows)
    winner = rank_candidates(phase_b_results).iloc[0]
    winner_name = str(winner["candidate"])
    winner_run = phase_b_runs[winner_name]
    winner_params = winner_run["params"]
    winner_source_weight = float(winner_run["source_weight"])
    winner_threshold = float(winner["threshold"])
    winner_probability = winner_run["probability"]
    winner_prediction = (winner_probability >= winner_threshold).astype(np.int8)
    phase_b_results["selected"] = phase_b_results["candidate"].eq(winner_name)
    phase_b_results.to_csv(
        RESULTS_DIR / "phase_b_candidate_cv_results.csv", index=False
    )
    pd.DataFrame(phase_b_threshold_rows).to_csv(
        RESULTS_DIR / "phase_b_threshold_results.csv", index=False
    )
    combined_candidates = pd.concat(
        [phase_a_results, phase_b_results], ignore_index=True, sort=False
    )
    combined_candidates.to_csv(
        RESULTS_DIR / "candidate_cv_results.csv", index=False
    )

    winner_fold_metrics = fold_metrics_for_winner(
        name=winner_name,
        y=y_train,
        probability=winner_probability,
        threshold=winner_threshold,
        folds=folds,
        manifest=manifest_train,
        is_new_fsd50k_cough=is_new_train,
    )
    winner_fold_metrics.to_csv(
        RESULTS_DIR / "best_cv_fold_metrics.csv", index=False
    )
    oof_output = manifest_train.copy()
    oof_output["y_true"] = y_train
    oof_output["y_pred"] = winner_prediction
    oof_output["probability_cough"] = winner_probability
    oof_output["threshold"] = winner_threshold
    oof_output.to_csv(RESULTS_DIR / "best_oof_predictions.csv", index=False)
    oof_origin, oof_source = source_diagnostics(
        "train_oof",
        manifest_train,
        y_train,
        winner_prediction,
        winner_probability,
        winner_threshold,
    )
    oof_origin.to_csv(RESULTS_DIR / "train_oof_metrics_by_origin.csv", index=False)
    oof_source.to_csv(
        RESULTS_DIR / "train_oof_cough_recall_by_source.csv", index=False
    )

    print("\nGanador OOF:")
    print(f"  {winner_name}")
    print(f"  Umbral: {winner_threshold:.2f}")
    print(
        f"  Macro-F1={winner['global_macro_f1']:.4f} | "
        f"especificidad={winner['global_specificity_no_cough']:.4f} | "
        f"recall original={winner['original_cough_recall']:.4f}"
    )
    print(
        f"  FSD bal-acc={winner['fsd50k_balanced_accuracy']:.4f} | "
        f"recall nuevas FSD50K={winner['new_fsd50k_cough_recall']:.4f} | "
        f"especificidad FSD={winner['fsd50k_specificity_no_cough']:.4f}"
    )

    print("\nAjustando LR final exclusivamente con todo TRAIN...")
    final_model = make_model(winner_params)
    final_sample_weight = np.ones(len(y_train), dtype=np.float64)
    final_sample_weight[is_new_train] = winner_source_weight
    started = time.perf_counter()
    final_model.fit(
        X_train,
        y_train,
        classifier__sample_weight=final_sample_weight,
    )
    fit_final_seconds = time.perf_counter() - started

    oof_global_metrics = classification_metrics(
        y_train, winner_prediction, winner_probability
    )
    metrics_rows: list[dict[str, Any]] = [
        {
            "dataset": "train_oof",
            "threshold": winner_threshold,
            **oof_global_metrics,
        }
    ]
    for split, X, y, manifest in [
        ("train_fit", X_train, y_train, manifest_train),
        ("validation", X_validation, y_validation, manifest_validation),
    ]:
        probability = final_model.predict_proba(X)[:, 1]
        prediction = (probability >= winner_threshold).astype(np.int8)
        metrics = classification_metrics(y, prediction, probability)
        metrics_rows.append(
            {"dataset": split, "threshold": winner_threshold, **metrics}
        )
        print(
            f"{split:>10}: F1 tos={metrics['f1_cough']:.4f} | "
            f"macro-F1={metrics['macro_f1']:.4f} | "
            f"recall tos={metrics['recall_cough']:.4f} | "
            f"especificidad={metrics['specificity_no_cough']:.4f} | "
            f"AUC={metrics['roc_auc']:.4f}"
        )
        if split == "validation":
            save_predictions(
                split, manifest, y, prediction, probability, winner_threshold
            )
            save_graphs(
                split, y, prediction, probability, winner_threshold
            )
            origin_metrics, source_recall = source_diagnostics(
                split,
                manifest,
                y,
                prediction,
                probability,
                winner_threshold,
            )
            origin_metrics.to_csv(
                RESULTS_DIR / "validation_metrics_by_origin.csv", index=False
            )
            source_recall.to_csv(
                RESULTS_DIR / "validation_cough_recall_by_source.csv", index=False
            )
            for row in source_recall.to_dict(orient="records"):
                print(
                    f"  VALIDATION {row['cough_source']}: "
                    f"recall={row['recall_cough']:.4f} "
                    f"(n={row['n_cough_segments']})"
                )

    pd.DataFrame(metrics_rows).to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    joblib.dump(final_model, MODEL_PATH)
    model_size_kb = MODEL_PATH.stat().st_size / 1024
    benchmark = benchmark_inference(final_model, X_validation)
    complexity = model_complexity(final_model)
    configuration: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "family": "lr",
        "winner": winner_name,
        "selection_rule": (
            "OOF-only constraints from fixed baseline; maximize FSD50K "
            "balanced accuracy, ties by new-cough recall and global macro-F1"
        ),
        "feature_source": str(FEATURES_DIR),
        "split_protocol": "random grouped by CoughVID UUID and FSD50K uploader",
        "n_phase_a_candidates": len(phase_a_specs),
        "n_phase_b_candidates": len(phase_b_runs),
        "threshold_grid": "0.05..0.95 step 0.01",
        "baseline_candidate": BASELINE_CANDIDATE,
        "baseline_threshold": BASELINE_THRESHOLD,
        "macro_f1_floor": floors["global_macro_f1"],
        "specificity_floor": floors["global_specificity_no_cough"],
        "original_cough_recall_floor": floors["original_cough_recall"],
        "C": float(winner_params["C"]),
        "penalty": winner_params["penalty"],
        "class_weight": winner_params["class_weight"] or "none",
        "new_fsd50k_cough_weight": winner_source_weight,
        "threshold": winner_threshold,
        "threshold_selection": "OOF TRAIN only; validation and test not used",
        "new_fsd50k_coughs_train": int(is_new_train.sum()),
        "new_fsd50k_coughs_validation": int(is_new_validation.sum()),
        "stage2_use_of_new_fsd50k_coughs": False,
        "test_evaluated": False,
        "n_features": X_train.shape[1],
        "normalization": "StandardScaler inside each fold; final fitted on TRAIN",
        "fit_final_seconds": fit_final_seconds,
        "model_size_kb": model_size_kb,
        **complexity,
        **benchmark,
        "benchmark_note": "desktop Python reference; not mobile latency",
    }
    pd.DataFrame(configuration.items(), columns=["parameter", "value"]).to_csv(
        RESULTS_DIR / "experiment_configuration.csv", index=False
    )

    print("\n" + "=" * 78)
    print("RESULTADO STAGE 1 - BUSQUEDA LR + TOSES FSD50K")
    print("=" * 78)
    print(f"Ganador: {winner_name}")
    print(f"Umbral OOF congelado: {winner_threshold:.2f}")
    print(f"Modelo: {model_size_kb:.2f} KB")
    print(
        f"Inferencia escritorio: {benchmark['desktop_single_sample_ms']:.4f} ms/muestra"
    )
    print(f"Resultados: {RESULTS_DIR}")
    print("TEST permanece reservado. Para evaluarlo: --action test")
    update_excel()


def evaluate_test(overwrite: bool) -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No existe {MODEL_PATH}. Ejecuta primero --action train."
        )
    metrics_path = RESULTS_DIR / "metrics_summary.csv"
    config_path = RESULTS_DIR / "experiment_configuration.csv"
    if not metrics_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Faltan resultados del entrenamiento")

    metrics = pd.read_csv(metrics_path)
    if (metrics["dataset"] == "test").any() and not overwrite:
        raise FileExistsError(
            "TEST ya fue evaluado. Usa --overwrite solo para reproducirlo."
        )
    configuration = pd.read_csv(config_path)
    config_map = dict(zip(configuration["parameter"], configuration["value"]))
    threshold = float(config_map["threshold"])

    X_test, y_test, manifest_test = load_split(FEATURES_DIR, "test")
    is_new_test = validate_manifest(manifest_test, y_test, "test")
    model: Pipeline = joblib.load(MODEL_PATH)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(np.int8)
    test_metrics = classification_metrics(y_test, prediction, probability)

    metrics = metrics.loc[
        ~metrics["dataset"].astype(str).str.startswith("test")
    ].copy()
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame(
                [
                    {
                        "dataset": "test",
                        "threshold": threshold,
                        **test_metrics,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    save_predictions(
        "test", manifest_test, y_test, prediction, probability, threshold
    )
    save_graphs("test", y_test, prediction, probability, threshold)
    origin_metrics, source_recall = source_diagnostics(
        "test",
        manifest_test,
        y_test,
        prediction,
        probability,
        threshold,
    )
    origin_metrics.to_csv(RESULTS_DIR / "test_metrics_by_origin.csv", index=False)
    source_recall.to_csv(
        RESULTS_DIR / "test_cough_recall_by_source.csv", index=False
    )

    quality_recall, quality_binary = quality_diagnostics(
        manifest_test, y_test, prediction, probability
    )
    if not quality_binary.empty:
        quality_binary["threshold"] = threshold
        metrics = pd.concat([metrics, quality_binary], ignore_index=True, sort=False)
    quality_recall.to_csv(
        RESULTS_DIR / "test_cough_recall_by_quality.csv", index=False
    )
    quality_binary.to_csv(
        RESULTS_DIR / "test_binary_metrics_by_quality.csv", index=False
    )
    metrics.to_csv(metrics_path, index=False)

    config_map["test_evaluated"] = True
    config_map["new_fsd50k_coughs_test"] = int(is_new_test.sum())
    pd.DataFrame(config_map.items(), columns=["parameter", "value"]).to_csv(
        config_path, index=False
    )

    print("=" * 78)
    print("EVALUACION FINAL TEST - LR SEARCH + TOSES FSD50K")
    print("=" * 78)
    print(
        f"TEST: F1 tos={test_metrics['f1_cough']:.4f} | "
        f"macro-F1={test_metrics['macro_f1']:.4f} | "
        f"recall tos={test_metrics['recall_cough']:.4f} | "
        f"especificidad={test_metrics['specificity_no_cough']:.4f} | "
        f"AUC={test_metrics['roc_auc']:.4f}"
    )
    for row in source_recall.to_dict(orient="records"):
        print(
            f"  TEST {row['cough_source']}: recall={row['recall_cough']:.4f} "
            f"(n={row['n_cough_segments']})"
        )
    print(f"Resultados: {RESULTS_DIR}")
    update_excel()


def main() -> None:
    args = parse_args()
    if not FEATURES_DIR.is_dir():
        raise FileNotFoundError(
            f"No existe {FEATURES_DIR}. Extrae primero los MFCC117."
        )
    validate_feature_configuration()
    if args.action == "train":
        train(overwrite=args.overwrite)
    else:
        evaluate_test(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
