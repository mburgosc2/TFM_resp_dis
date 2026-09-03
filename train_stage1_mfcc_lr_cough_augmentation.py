"""Entrena Stage 1 MFCC117 + LR con dos variantes por tos de TRAIN.

La configuracion del clasificador permanece congelada para aislar el efecto
de la augmentation de audio::

    StandardScaler -> LogisticRegression(
        C=0.3, penalty="l1", solver="liblinear", class_weight=None
    )

En cada fold se incorporan solo variantes cuyos padres pertenecen al TRAIN
interno. El fold OOF contiene exclusivamente muestras originales. VALIDATION
y TEST tampoco contienen augmentation. El umbral permanece fijo en 0.5.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import train_stage1_mfcc_lr_fsd50k_coughs as base


ROOT = Path(__file__).resolve().parent
ORIGINAL_FEATURES_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_cough_segments_random"
    / "mfcc117"
)
AUGMENTED_FEATURES_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_cough_segments_audio_aug_random"
    / "mfcc117"
)
EXPERIMENT_NAME = "mfcc117_lr_fsd50k_cough_segments_audio_aug_random"
RESULTS_DIR = (
    ROOT / "results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
)
GRAPHS_DIR = (
    ROOT / "graphs_results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
)
MODEL_PATH = RESULTS_DIR / "stage1_lr_cough_segments_audio_aug.joblib"
BASELINE_RESULTS_DIR = (
    ROOT
    / "results_stage1_cough_no_cough"
    / "mfcc117_lr_fsd50k_cough_segments_random"
    / "full"
)
AUGMENTATION_EXTRACTION_SCRIPT = (
    "feature_extraction_stage1_mfcc_cough_augmentation.py"
)
AUGMENTATION_DISPLAY_NAME = "AUDIO AUGMENTATION DE TOSES"
AUGMENTATION_PROTOCOL = (
    "two distinct random transformations per cough among noise at 10/20 dB "
    "SNR and pitch shift at +/-1.5 semitones"
)
EXPECTED_VARIANTS_PER_COUGH = 2

# Las utilidades heredadas escriben usando estas variables globales.
base.FEATURES_DIR = ORIGINAL_FEATURES_DIR
base.EXPERIMENT_NAME = EXPERIMENT_NAME
base.RESULTS_DIR = RESULTS_DIR
base.GRAPHS_DIR = GRAPHS_DIR
base.MODEL_PATH = MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 MFCC117 + LR fija con augmentation solo de toses"
    )
    parser.add_argument(
        "--action",
        choices=("train", "test"),
        default="train",
        help=(
            "train calcula OOF y VALIDATION; test evalua despues el modelo "
            "congelado sobre TEST."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_augmented_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    paths = {
        "X": AUGMENTED_FEATURES_DIR / "X_train_augmented.npy",
        "y": AUGMENTED_FEATURES_DIR / "y_train_augmented.npy",
        "folds": AUGMENTED_FEATURES_DIR / "folds_train_augmented.npy",
        "manifest": (
            AUGMENTED_FEATURES_DIR / "metadata_features_train_augmented.csv"
        ),
        "config": AUGMENTED_FEATURES_DIR / "augmentation_configuration.csv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Falta {missing[0]}. Ejecuta antes:\n"
            f"py -3.10 .\\{AUGMENTATION_EXTRACTION_SCRIPT} "
            "--action extract"
        )

    X = np.load(paths["X"]).astype(np.float32, copy=False)
    y = np.load(paths["y"]).astype(np.int8, copy=False)
    folds = np.load(paths["folds"]).astype(int, copy=False)
    manifest = pd.read_csv(
        paths["manifest"],
        dtype={
            "augmentation_id": str,
            "parent_uuid_segmento": str,
            "parent_original_uuid": str,
            "parent_split_group": str,
        },
    )
    return X, y, folds, manifest


def validate_augmented_features(
    X_augmented: np.ndarray,
    y_augmented: np.ndarray,
    folds_augmented: np.ndarray,
    augmented_manifest: pd.DataFrame,
    original_manifest: pd.DataFrame,
    original_y: np.ndarray,
    original_folds: np.ndarray,
) -> dict[str, Any]:
    lengths = {
        len(X_augmented),
        len(y_augmented),
        len(folds_augmented),
        len(augmented_manifest),
    }
    if len(lengths) != 1:
        raise ValueError("Features, etiquetas, folds y manifiesto no estan alineados.")
    if X_augmented.ndim != 2 or X_augmented.shape[1] != 117:
        raise ValueError(f"Shape aumentada invalida: {X_augmented.shape}")
    if not np.isfinite(X_augmented).all():
        raise ValueError("Las features aumentadas contienen valores no finitos.")
    if not np.all(y_augmented == 1):
        raise ValueError("La augmentation debe contener exclusivamente toses.")
    if set(folds_augmented.tolist()) != {0, 1, 2, 3, 4}:
        raise ValueError("Las variantes no contienen los folds 0..4.")
    if augmented_manifest["augmentation_id"].duplicated().any():
        raise ValueError("Hay augmentation_id duplicados.")

    required = {
        "augmentation_id",
        "noise_donor_fold",
        "parent_fold",
        "parent_split_group",
        "parent_uuid_segmento",
        "stage1_target",
        "transformation",
    }
    missing = required - set(augmented_manifest.columns)
    if missing:
        raise ValueError(f"Faltan columnas aumentadas: {sorted(missing)}")

    parent_table = pd.DataFrame(
        {
            "parent_uuid_segmento": original_manifest["uuid_segmento"].astype(str),
            "original_target": original_y,
            "original_fold": original_folds,
            "original_split_group": original_manifest["split_group"].astype(str),
        }
    )
    merged = augmented_manifest.merge(
        parent_table,
        on="parent_uuid_segmento",
        how="left",
        validate="many_to_one",
    )
    if merged["original_target"].isna().any():
        raise ValueError("Una variante no encuentra su audio padre.")
    if not (merged["original_target"].to_numpy(dtype=int) == 1).all():
        raise ValueError("Una variante procede de una muestra no positiva.")
    if not (
        merged["parent_fold"].to_numpy(dtype=int)
        == merged["original_fold"].to_numpy(dtype=int)
    ).all():
        raise ValueError("Una variante no conserva el fold del padre.")
    if not np.array_equal(
        folds_augmented,
        merged["original_fold"].to_numpy(dtype=int),
    ):
        raise ValueError("folds_train_augmented.npy no coincide con los padres.")
    if not (
        merged["parent_split_group"].astype(str)
        == merged["original_split_group"].astype(str)
    ).all():
        raise ValueError("Una variante no conserva el split_group del padre.")

    per_parent = merged.groupby("parent_uuid_segmento").size()
    original_positive_ids = set(
        original_manifest.loc[original_y == 1, "uuid_segmento"].astype(str)
    )
    if set(per_parent.index) != original_positive_ids:
        raise ValueError("No todas las toses originales tienen variantes.")
    if not (per_parent == EXPECTED_VARIANTS_PER_COUGH).all():
        raise ValueError(
            "Cada tos debe tener exactamente "
            f"{EXPECTED_VARIANTS_PER_COUGH} variante(s)."
        )

    noise_mask = merged["transformation"].str.startswith("noise")
    noise_donor_folds = pd.to_numeric(
        merged.loc[noise_mask, "noise_donor_fold"], errors="raise"
    ).to_numpy(dtype=int)
    if not (
        noise_donor_folds
        == merged.loc[noise_mask, "original_fold"].to_numpy(dtype=int)
    ).all():
        raise ValueError("Un donante de ruido pertenece a otro fold.")

    return {
        "n_augmented": len(merged),
        "n_augmented_parents": int(per_parent.size),
        "transformation_counts": merged["transformation"].value_counts().to_dict(),
    }


def compare_with_baseline(metrics: pd.DataFrame) -> None:
    baseline_path = BASELINE_RESULTS_DIR / "metrics_summary.csv"
    if not baseline_path.is_file():
        return
    baseline = pd.read_csv(baseline_path)
    selected_datasets = {"train_oof", "validation"}
    rows: list[dict[str, Any]] = []
    for dataset in sorted(selected_datasets):
        current_row = metrics.loc[metrics["dataset"] == dataset]
        baseline_row = baseline.loc[baseline["dataset"] == dataset]
        if current_row.empty or baseline_row.empty:
            continue
        current = current_row.iloc[0]
        previous = baseline_row.iloc[0]
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "precision_cough",
            "recall_cough",
            "specificity_no_cough",
            "f1_cough",
            "macro_f1",
            "roc_auc",
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "baseline_no_augmentation": float(previous[metric]),
                    "audio_augmentation": float(current[metric]),
                    "difference": float(current[metric] - previous[metric]),
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / "comparison_to_no_augmentation.csv", index=False
        )


def train(overwrite: bool) -> None:
    if MODEL_PATH.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {MODEL_PATH}. Usa --overwrite para regenerar."
        )

    base.validate_feature_configuration()
    X_train, y_train, manifest_train = base.load_split(
        ORIGINAL_FEATURES_DIR, "train"
    )
    X_validation, y_validation, manifest_validation = base.load_split(
        ORIGINAL_FEATURES_DIR, "validation"
    )
    is_new_train = base.validate_manifest(manifest_train, y_train, "train")
    is_new_validation = base.validate_manifest(
        manifest_validation, y_validation, "validation"
    )
    folds = np.load(ORIGINAL_FEATURES_DIR / "folds_train.npy").astype(int)
    base.validate_fold_isolation(manifest_train, folds)
    base.assert_split_groups_are_disjoint(
        {"train": manifest_train, "validation": manifest_validation}
    )

    X_augmented, y_augmented, folds_augmented, augmented_manifest = (
        load_augmented_features()
    )
    augmentation = validate_augmented_features(
        X_augmented,
        y_augmented,
        folds_augmented,
        augmented_manifest,
        manifest_train,
        y_train,
        folds,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"STAGE 1 - MFCC117 + LR + {AUGMENTATION_DISPLAY_NAME}")
    print("=" * 78)
    print(f"Configuracion LR congelada: {base.MODEL_NAME}")
    print(f"Umbral fijo: {base.THRESHOLD}")
    print("No se buscan hiperparametros ni umbral.")
    print(
        f"TRAIN original: {X_train.shape}; "
        f"no_tos/tos={np.bincount(y_train).tolist()}"
    )
    print(f"TRAIN aumentado adicional: {X_augmented.shape}; solo tos")
    print(
        "TRAIN final: "
        f"{len(X_train) + len(X_augmented)}; "
        f"no_tos/tos=[{int(np.sum(y_train == 0))}, "
        f"{int(np.sum(y_train == 1) + len(y_augmented))}]"
    )
    print(f"Transformaciones: {augmentation['transformation_counts']}")
    print("OOF, VALIDATION y TEST contienen solo audios originales.")
    print("TEST no se leera durante --action train.")

    oof_prediction = np.zeros(len(y_train), dtype=np.int8)
    oof_probability = np.zeros(len(y_train), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []

    print("\nCV agrupada de TRAIN:")
    for fold in sorted(np.unique(folds)):
        original_validation_mask = folds == fold
        original_train_mask = ~original_validation_mask
        augmented_train_mask = folds_augmented != fold

        X_fit = np.vstack(
            [X_train[original_train_mask], X_augmented[augmented_train_mask]]
        )
        y_fit = np.concatenate(
            [y_train[original_train_mask], y_augmented[augmented_train_mask]]
        )
        model = base.make_model()
        started = time.perf_counter()
        model.fit(X_fit, y_fit)
        fit_seconds = time.perf_counter() - started

        probability = model.predict_proba(
            X_train[original_validation_mask]
        )[:, 1]
        prediction = (probability >= base.THRESHOLD).astype(np.int8)
        oof_probability[original_validation_mask] = probability
        oof_prediction[original_validation_mask] = prediction
        metrics = base.classification_metrics(
            y_train[original_validation_mask], prediction, probability
        )
        fold_new_mask = original_validation_mask & is_new_train.to_numpy()
        fold_rows.append(
            {
                "candidate": base.MODEL_NAME,
                "fold": int(fold),
                "n_original_train": int(original_train_mask.sum()),
                "n_augmented_train": int(augmented_train_mask.sum()),
                "n_effective_train": int(len(X_fit)),
                "n_original_validation": int(original_validation_mask.sum()),
                "n_augmented_validation": 0,
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
            f"  Fold {fold}: original_train={original_train_mask.sum()} | "
            f"aug_train={augmented_train_mask.sum()} | "
            f"macro-F1={metrics['macro_f1']:.4f} | "
            f"recall tos={metrics['recall_cough']:.4f} | "
            f"especificidad={metrics['specificity_no_cough']:.4f}"
        )

    oof_metrics = base.classification_metrics(
        y_train, oof_prediction, oof_probability
    )
    oof_origin, oof_source = base.source_diagnostics(
        "train_oof",
        manifest_train,
        y_train,
        oof_prediction,
        oof_probability,
    )
    pd.DataFrame(fold_rows).to_csv(
        RESULTS_DIR / "best_cv_fold_metrics.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        RESULTS_DIR / "cv_fold_metrics.csv", index=False
    )
    oof_output = manifest_train.copy()
    oof_output["y_true"] = y_train
    oof_output["y_pred"] = oof_prediction
    oof_output["probability_cough"] = oof_probability
    oof_output["threshold"] = base.THRESHOLD
    oof_output["evaluation_role"] = "original_oof_only"
    oof_output.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    oof_origin.to_csv(
        RESULTS_DIR / "train_oof_metrics_by_origin.csv", index=False
    )
    oof_source.to_csv(
        RESULTS_DIR / "train_oof_cough_recall_by_source.csv", index=False
    )

    print(
        f"OOF original: macro-F1={oof_metrics['macro_f1']:.4f} | "
        f"recall tos={oof_metrics['recall_cough']:.4f} | "
        f"especificidad={oof_metrics['specificity_no_cough']:.4f} | "
        f"AUC={oof_metrics['roc_auc']:.4f}"
    )

    print("\nAjustando LR final con TRAIN original + aumentado...")
    X_final = np.vstack([X_train, X_augmented])
    y_final = np.concatenate([y_train, y_augmented])
    final_model = base.make_model()
    started = time.perf_counter()
    final_model.fit(X_final, y_final)
    fit_final_seconds = time.perf_counter() - started

    metrics_rows: list[dict[str, Any]] = [
        {"dataset": "train_oof", "threshold": base.THRESHOLD, **oof_metrics}
    ]
    for split, X, y, manifest in (
        ("train_fit_original", X_train, y_train, manifest_train),
        ("validation", X_validation, y_validation, manifest_validation),
    ):
        probability = final_model.predict_proba(X)[:, 1]
        prediction = (probability >= base.THRESHOLD).astype(np.int8)
        metrics = base.classification_metrics(y, prediction, probability)
        metrics_rows.append(
            {"dataset": split, "threshold": base.THRESHOLD, **metrics}
        )
        print(
            f"{split:>18}: macro-F1={metrics['macro_f1']:.4f} | "
            f"recall tos={metrics['recall_cough']:.4f} | "
            f"especificidad={metrics['specificity_no_cough']:.4f} | "
            f"AUC={metrics['roc_auc']:.4f}"
        )
        if split == "validation":
            base.save_predictions(split, manifest, y, prediction, probability)
            base.save_graphs(split, y, prediction, probability)
            origin_metrics, source_recall = base.source_diagnostics(
                split, manifest, y, prediction, probability
            )
            origin_metrics.to_csv(
                RESULTS_DIR / "validation_metrics_by_origin.csv", index=False
            )
            source_recall.to_csv(
                RESULTS_DIR / "validation_cough_recall_by_source.csv", index=False
            )

    metrics_table = pd.DataFrame(metrics_rows)
    metrics_table.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    compare_with_baseline(metrics_table)
    joblib.dump(final_model, MODEL_PATH)
    model_size_kb = MODEL_PATH.stat().st_size / 1024.0
    benchmark = base.benchmark_inference(final_model, X_validation)
    complexity = base.model_complexity(final_model)

    configuration: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "family": "lr",
        "winner": base.MODEL_NAME,
        "selection_rule": (
            "classifier and threshold frozen; experiment isolates audio "
            "augmentation effect"
        ),
        "original_feature_source": str(ORIGINAL_FEATURES_DIR),
        "augmented_feature_source": str(AUGMENTED_FEATURES_DIR),
        "target_mapping": "0=no_cough; 1=cough",
        "n_features": int(X_train.shape[1]),
        "augmentation_scope": "TRAIN cough class only",
        "augmentation_protocol": AUGMENTATION_PROTOCOL,
        "n_original_train": len(X_train),
        "n_original_no_cough_train": int(np.sum(y_train == 0)),
        "n_original_cough_train": int(np.sum(y_train == 1)),
        "n_augmented_cough_train": len(X_augmented),
        "n_final_fit": len(X_final),
        "n_final_no_cough": int(np.sum(y_final == 0)),
        "n_final_cough": int(np.sum(y_final == 1)),
        "variants_per_cough": EXPECTED_VARIANTS_PER_COUGH,
        "transformation_counts": str(augmentation["transformation_counts"]),
        "oof_policy": "original samples only; variants of held-out parents excluded",
        "validation_augmentation": False,
        "test_augmentation": False,
        "full_weight_per_augmented_sample": True,
        "normalization": "StandardScaler fitted inside each fold",
        "threshold": base.THRESHOLD,
        "threshold_selection": "fixed; no threshold tuning",
        "test_evaluated": False,
        **base.LR_PARAMS,
        "fit_final_seconds": fit_final_seconds,
        "model_size_kb": model_size_kb,
        **complexity,
        **benchmark,
        "new_fsd50k_coughs_train_original": int(is_new_train.sum()),
        "new_fsd50k_coughs_validation": int(is_new_validation.sum()),
        "external_pilot_used_for_selection": False,
    }
    pd.DataFrame(
        configuration.items(), columns=["parameter", "value"]
    ).to_csv(RESULTS_DIR / "experiment_configuration.csv", index=False)

    print("\n" + "=" * 78)
    print(f"RESULTADO STAGE 1 - LR + {AUGMENTATION_DISPLAY_NAME}")
    print("=" * 78)
    print(f"Modelo: {model_size_kb:.2f} KB")
    print(f"Resultados: {RESULTS_DIR}")
    print("TEST permanece reservado. Para evaluarlo: --action test")


def main() -> None:
    args = parse_args()
    if args.action == "train":
        train(overwrite=args.overwrite)
    else:
        base.validate_feature_configuration()
        # La funcion heredada usa las rutas globales configuradas arriba y
        # evalua solo el TEST original, nunca las variantes sinteticas.
        base.evaluate_test(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
