"""Entrena la LR compacta de Stage 1 con las nuevas toses FSD50K.

El experimento mantiene congelada la configuracion ganadora de los ensayos
MFCC117 aleatorios anteriores::

    StandardScaler -> LogisticRegression(
        C=0.3, penalty="l1", solver="liblinear", class_weight=None
    )

No se realiza una nueva busqueda de hiperparametros ni de umbral. Las
predicciones OOF de TRAIN se utilizan exclusivamente para evaluar la
configuracion fija. Validation se consulta despues de ajustar el modelo final
con TRAIN y TEST permanece separado hasta ejecutar explicitamente
``--action test``.

Esto permite medir de forma especifica si incorporar toses reales de FSD50K
mejora la generalizacion fuera de CoughVID sin contaminar Stage 2.
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
    THRESHOLD,
    classification_metrics,
    load_split,
    quality_diagnostics,
)


RANDOM_STATE = 42
MODEL_NAME = "lr__C0p3__l1__weight_none"
LR_PARAMS = {
    "C": 0.3,
    "penalty": "l1",
    "solver": "liblinear",
    "class_weight": None,
    "max_iter": 5_000,
    "random_state": RANDOM_STATE,
}

FEATURES_DIR = (
    ROOT / "features_extracted_stage1_fsd50k_coughs_random" / "mfcc117"
)
EXPERIMENT_NAME = "mfcc117_lr_fsd50k_coughs_augmented_random"
RESULTS_DIR = ROOT / "results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
GRAPHS_DIR = (
    ROOT / "graphs_results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
)
MODEL_PATH = RESULTS_DIR / "stage1_lr_fsd50k_coughs_augmented.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 MFCC117 + LR fija con nuevas toses FSD50K"
    )
    parser.add_argument(
        "--action",
        choices=["train", "test"],
        default="train",
        help=(
            "train calcula OOF, ajusta TRAIN y evalua VALIDATION; test evalua "
            "el modelo ya congelado sobre TEST"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar resultados de la misma accion",
    )
    return parser.parse_args()


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**LR_PARAMS)),
        ]
    )


def parse_bool_series(series: pd.Series, column: str) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        examples = sorted(normalized.loc[invalid].unique().tolist())[:5]
        raise ValueError(f"Valores booleanos invalidos en {column}: {examples}")
    return normalized.map(mapping).astype(bool)


def validate_feature_configuration() -> None:
    path = FEATURES_DIR / "feature_configuration.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    configuration = pd.read_csv(path)
    if not {"parameter", "value"}.issubset(configuration.columns):
        raise ValueError(f"Configuracion de features invalida: {path}")
    values = dict(zip(configuration["parameter"], configuration["value"]))
    if str(values.get("n_features")) not in {"117", "117.0"}:
        raise ValueError("La extraccion no contiene 117 features")
    if str(values.get("sample_rate")) not in {"16000", "16000.0"}:
        raise ValueError("La extraccion no usa sample rate de 16 kHz")
    if str(values.get("near_silence_policy")) != "reject":
        raise ValueError(
            "Para comparar con los experimentos historicos, regenera las "
            "features con --near-silence-policy reject"
        )


def validate_manifest(
    manifest: pd.DataFrame,
    y: np.ndarray,
    split: str,
) -> pd.Series:
    required = {
        "dataset_origin",
        "fold",
        "is_new_fsd50k_cough",
        "original_uuid",
        "split_group",
        "stage1_target",
        "stage2_eligible",
        "uuid_segmento",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el manifiesto {split}: {sorted(missing)}")
    if manifest["uuid_segmento"].astype(str).duplicated().any():
        raise ValueError(f"Hay uuid_segmento duplicados en {split}")

    is_new = parse_bool_series(
        manifest["is_new_fsd50k_cough"], "is_new_fsd50k_cough"
    )
    stage2_eligible = parse_bool_series(
        manifest["stage2_eligible"], "stage2_eligible"
    )
    invalid_new = is_new & (
        manifest["dataset_origin"].astype(str).ne("FSD50K")
        | pd.Series(y, index=manifest.index).ne(1)
        | stage2_eligible
    )
    if invalid_new.any():
        bad_id = manifest.loc[invalid_new, "original_uuid"].iloc[0]
        raise ValueError(f"Nueva tos FSD50K mal configurada: {bad_id}")
    return is_new


def validate_fold_isolation(manifest: pd.DataFrame, folds: np.ndarray) -> None:
    if len(folds) != len(manifest):
        raise ValueError("folds_train.npy no esta alineado con TRAIN")
    manifest_folds = manifest["fold"].to_numpy(dtype=int)
    if not np.array_equal(folds, manifest_folds):
        raise ValueError("Los folds no coinciden con el manifiesto de TRAIN")
    if set(folds.tolist()) != {0, 1, 2, 3, 4}:
        raise ValueError("TRAIN no contiene exactamente los folds 0..4")
    per_group_fold = (
        pd.DataFrame(
            {
                "split_group": manifest["split_group"].astype(str),
                "fold": folds,
            }
        )
        .groupby("split_group")["fold"]
        .nunique()
    )
    if (per_group_fold > 1).any():
        bad_group = per_group_fold.loc[per_group_fold > 1].index[0]
        raise ValueError(f"Fuga entre folds para el grupo {bad_group}")


def assert_split_groups_are_disjoint(manifests: dict[str, pd.DataFrame]) -> None:
    group_sets = {
        split: set(manifest["split_group"].astype(str))
        for split, manifest in manifests.items()
    }
    names = list(group_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = group_sets[left].intersection(group_sets[right])
            if overlap:
                raise ValueError(
                    f"Fuga de grupos entre {left} y {right}: {next(iter(overlap))}"
                )


def save_predictions(
    split: str,
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    output = manifest.copy()
    output["y_true"] = y_true
    output["y_pred"] = y_pred
    output["probability_cough"] = y_prob
    output["threshold"] = THRESHOLD
    output.to_csv(RESULTS_DIR / f"{split}_predictions.csv", index=False)


def save_graphs(
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
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
        title=f"Stage 1 LR + toses FSD50K - {split}",
    )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / f"confusion_matrix_{split}.png", dpi=180)
    plt.close(fig)

    if np.unique(y_true).size == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5.5, 4.6))
        ax.plot(fpr, tpr, label=f"LR (AUC={auc:.4f})")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set(
            xlabel="False positive rate",
            ylabel="Recall tos",
            title=f"ROC Stage 1 LR + toses FSD50K - {split}",
        )
        ax.legend(loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(GRAPHS_DIR / f"roc_curve_{split}.png", dpi=180)
        plt.close(fig)


def source_diagnostics(
    split: str,
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa rendimiento por corpus y recall de las fuentes positivas."""
    origins = manifest["dataset_origin"].astype(str)
    is_new = parse_bool_series(
        manifest["is_new_fsd50k_cough"], "is_new_fsd50k_cough"
    ).to_numpy()

    origin_rows: list[dict[str, Any]] = []
    for origin in sorted(origins.unique()):
        mask = origins.eq(origin).to_numpy()
        if not mask.any():
            continue
        origin_rows.append(
            {
                "dataset": f"{split}_{origin.lower()}",
                "dataset_origin": origin,
                "threshold": THRESHOLD,
                **classification_metrics(
                    y_true[mask], y_pred[mask], y_prob[mask]
                ),
            }
        )

    cough_sources = {
        "original_cough": (y_true == 1) & ~is_new,
        "new_fsd50k_cough": (y_true == 1) & is_new,
    }
    recall_rows: list[dict[str, Any]] = []
    for source, mask in cough_sources.items():
        if not mask.any():
            continue
        recall_rows.append(
            {
                "split": split,
                "cough_source": source,
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


def model_complexity(model: Pipeline) -> dict[str, int]:
    classifier = model.named_steps["classifier"]
    scaler = model.named_steps["scaler"]
    return {
        "trainable_parameters": int(
            classifier.coef_.size + classifier.intercept_.size
        ),
        "nonzero_coefficients": int(np.count_nonzero(classifier.coef_)),
        "deployment_float_values": int(
            classifier.coef_.size
            + classifier.intercept_.size
            + scaler.mean_.size
            + scaler.scale_.size
        ),
    }


def benchmark_inference(model: Pipeline, X: np.ndarray) -> dict[str, float | int]:
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
            f"Ya existe {MODEL_PATH}. Usa --overwrite para regenerar el experimento."
        )

    X_train, y_train, manifest_train = load_split(FEATURES_DIR, "train")
    X_validation, y_validation, manifest_validation = load_split(
        FEATURES_DIR, "validation"
    )
    is_new_train = validate_manifest(manifest_train, y_train, "train")
    is_new_validation = validate_manifest(
        manifest_validation, y_validation, "validation"
    )
    folds = np.load(FEATURES_DIR / "folds_train.npy").astype(int)
    validate_fold_isolation(manifest_train, folds)
    assert_split_groups_are_disjoint(
        {"train": manifest_train, "validation": manifest_validation}
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("STAGE 1 - MFCC117 + LR FIJA + NUEVAS TOSES FSD50K")
    print("=" * 78)
    print(f"Configuracion congelada: {MODEL_NAME}")
    print(f"Umbral fijo: {THRESHOLD}")
    print("No se realiza busqueda de hiperparametros ni de umbral.")
    print("TEST no sera leido ni evaluado durante esta accion.")
    print(
        f"TRAIN: {X_train.shape}; no_tos/tos={np.bincount(y_train).tolist()}; "
        f"nuevas_toses_FSD50K={int(is_new_train.sum())}"
    )
    print(
        f"VALIDATION: {X_validation.shape}; "
        f"no_tos/tos={np.bincount(y_validation).tolist()}; "
        f"nuevas_toses_FSD50K={int(is_new_validation.sum())}"
    )

    oof_prediction = np.zeros(len(y_train), dtype=np.int8)
    oof_probability = np.zeros(len(y_train), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    print("\nCV de TRAIN con StandardScaler ajustado dentro de cada fold:")
    for fold in sorted(np.unique(folds)):
        inner_validation = folds == fold
        inner_train = ~inner_validation
        model = make_model()
        started = time.perf_counter()
        model.fit(X_train[inner_train], y_train[inner_train])
        fit_seconds = time.perf_counter() - started
        probability = model.predict_proba(X_train[inner_validation])[:, 1]
        prediction = (probability >= THRESHOLD).astype(np.int8)
        oof_probability[inner_validation] = probability
        oof_prediction[inner_validation] = prediction
        metrics = classification_metrics(
            y_train[inner_validation], prediction, probability
        )
        fold_new_mask = inner_validation & is_new_train.to_numpy()
        fold_rows.append(
            {
                "candidate": MODEL_NAME,
                "fold": int(fold),
                "n_train": int(inner_train.sum()),
                "n_validation": int(inner_validation.sum()),
                "n_new_fsd50k_cough_validation": int(fold_new_mask.sum()),
                "recall_new_fsd50k_cough": (
                    float(np.mean(oof_prediction[fold_new_mask] == 1))
                    if fold_new_mask.any()
                    else np.nan
                ),
                "fit_seconds": fit_seconds,
                **metrics,
            }
        )
        print(
            f"  Fold {fold}: macro-F1={metrics['macro_f1']:.4f} | "
            f"recall tos={metrics['recall_cough']:.4f} | "
            f"recall nuevas FSD50K="
            f"{fold_rows[-1]['recall_new_fsd50k_cough']:.4f}"
        )

    oof_metrics = classification_metrics(y_train, oof_prediction, oof_probability)
    oof_origin, oof_source = source_diagnostics(
        "train_oof",
        manifest_train,
        y_train,
        oof_prediction,
        oof_probability,
    )
    pd.DataFrame(fold_rows).to_csv(
        RESULTS_DIR / "best_cv_fold_metrics.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "cv_fold_metrics.csv", index=False)
    oof_output = manifest_train.copy()
    oof_output["y_true"] = y_train
    oof_output["y_pred"] = oof_prediction
    oof_output["probability_cough"] = oof_probability
    oof_output["threshold"] = THRESHOLD
    oof_output.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    oof_origin.to_csv(RESULTS_DIR / "train_oof_metrics_by_origin.csv", index=False)
    oof_source.to_csv(
        RESULTS_DIR / "train_oof_cough_recall_by_source.csv", index=False
    )

    print(
        f"OOF global: macro-F1={oof_metrics['macro_f1']:.4f} | "
        f"recall tos={oof_metrics['recall_cough']:.4f} | "
        f"AUC={oof_metrics['roc_auc']:.4f}"
    )
    for row in oof_source.to_dict(orient="records"):
        print(
            f"  OOF {row['cough_source']}: recall={row['recall_cough']:.4f} "
            f"(n={row['n_cough_segments']})"
        )

    print("\nAjustando la LR final exclusivamente con todo TRAIN...")
    final_model = make_model()
    started = time.perf_counter()
    final_model.fit(X_train, y_train)
    fit_final_seconds = time.perf_counter() - started

    metrics_rows: list[dict[str, Any]] = [
        {"dataset": "train_oof", "threshold": THRESHOLD, **oof_metrics}
    ]
    for split, X, y, manifest in [
        ("train_fit", X_train, y_train, manifest_train),
        ("validation", X_validation, y_validation, manifest_validation),
    ]:
        probability = final_model.predict_proba(X)[:, 1]
        prediction = (probability >= THRESHOLD).astype(np.int8)
        metrics = classification_metrics(y, prediction, probability)
        metrics_rows.append(
            {"dataset": split, "threshold": THRESHOLD, **metrics}
        )
        print(
            f"{split:>10}: F1 tos={metrics['f1_cough']:.4f} | "
            f"macro-F1={metrics['macro_f1']:.4f} | "
            f"recall tos={metrics['recall_cough']:.4f} | "
            f"AUC={metrics['roc_auc']:.4f}"
        )
        if split == "validation":
            save_predictions(split, manifest, y, prediction, probability)
            save_graphs(split, y, prediction, probability)
            origin_metrics, source_recall = source_diagnostics(
                split, manifest, y, prediction, probability
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
        "winner": MODEL_NAME,
        "selection_rule": (
            "configuration frozen from previous random-split OOF search; "
            "no hyperparameter selection in this experiment"
        ),
        "feature_source": str(FEATURES_DIR),
        "split_protocol": "random grouped by CoughVID UUID and FSD50K uploader",
        "target_mapping": "0=no_cough; 1=cough",
        "new_fsd50k_coughs_train": int(is_new_train.sum()),
        "new_fsd50k_coughs_validation": int(is_new_validation.sum()),
        "stage2_use_of_new_fsd50k_coughs": False,
        "n_features": X_train.shape[1],
        "normalization": "StandardScaler inside each fold; final fitted on TRAIN",
        "threshold": THRESHOLD,
        "threshold_selection": "fixed; no threshold tuning",
        "test_evaluated": False,
        **LR_PARAMS,
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
    print("RESULTADO STAGE 1 - LR + NUEVAS TOSES FSD50K")
    print("=" * 78)
    print(f"Configuracion fija: {MODEL_NAME}")
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
            f"No existe el modelo congelado {MODEL_PATH}. Ejecuta antes --action train."
        )
    metrics_path = RESULTS_DIR / "metrics_summary.csv"
    config_path = RESULTS_DIR / "experiment_configuration.csv"
    if not metrics_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Faltan resultados o configuracion del entrenamiento")

    metrics = pd.read_csv(metrics_path)
    if (metrics["dataset"] == "test").any() and not overwrite:
        raise FileExistsError(
            "TEST ya fue evaluado. Usa --overwrite solo si necesitas reproducirlo."
        )

    X_test, y_test, manifest_test = load_split(FEATURES_DIR, "test")
    is_new_test = validate_manifest(manifest_test, y_test, "test")
    model: Pipeline = joblib.load(MODEL_PATH)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= THRESHOLD).astype(np.int8)
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
                        "threshold": THRESHOLD,
                        **test_metrics,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    save_predictions("test", manifest_test, y_test, prediction, probability)
    save_graphs("test", y_test, prediction, probability)
    origin_metrics, source_recall = source_diagnostics(
        "test", manifest_test, y_test, prediction, probability
    )
    origin_metrics.to_csv(RESULTS_DIR / "test_metrics_by_origin.csv", index=False)
    source_recall.to_csv(
        RESULTS_DIR / "test_cough_recall_by_source.csv", index=False
    )

    quality_recall, quality_binary = quality_diagnostics(
        manifest_test, y_test, prediction, probability
    )
    quality_recall.to_csv(
        RESULTS_DIR / "test_cough_recall_by_quality.csv", index=False
    )
    quality_binary.to_csv(
        RESULTS_DIR / "test_binary_metrics_by_quality.csv", index=False
    )
    if not quality_binary.empty:
        metrics = pd.concat([metrics, quality_binary], ignore_index=True, sort=False)
    metrics.to_csv(metrics_path, index=False)

    configuration = pd.read_csv(config_path)
    config_map = dict(zip(configuration["parameter"], configuration["value"]))
    config_map["test_evaluated"] = True
    config_map["new_fsd50k_coughs_test"] = int(is_new_test.sum())
    pd.DataFrame(config_map.items(), columns=["parameter", "value"]).to_csv(
        config_path, index=False
    )

    print("=" * 78)
    print("EVALUACION FINAL TEST - STAGE 1 LR + NUEVAS TOSES FSD50K")
    print("=" * 78)
    print(
        f"TEST: F1 tos={test_metrics['f1_cough']:.4f} | "
        f"macro-F1={test_metrics['macro_f1']:.4f} | "
        f"recall tos={test_metrics['recall_cough']:.4f} | "
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
            f"No existe {FEATURES_DIR}. Ejecuta primero:\n"
            "python .\\feature_extraction_stage1_mfcc_fsd50k_coughs.py "
            "--action extract --near-silence-policy reject"
        )
    validate_feature_configuration()
    if args.action == "train":
        train(overwrite=args.overwrite)
    else:
        evaluate_test(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
