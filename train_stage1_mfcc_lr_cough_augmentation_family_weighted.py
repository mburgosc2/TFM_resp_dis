"""Stage 1 MFCC117 + LR con augmentation ponderada por familia de tos.

Este experimento reutiliza exactamente las dos variantes de audio ya extraidas
para cada tos de TRAIN, pero evita que una tos aumentada aporte tres veces mas
que una muestra negativa. Los pesos son::

    negativo original = 1.00
    tos original      = 0.50
    variante 1        = 0.25
    variante 2        = 0.25

Por tanto, cada familia positiva conserva un peso total de 1.00. Los pesos se
aplican tanto al StandardScaler como a LogisticRegression dentro de cada fold.
La configuracion de la LR y el umbral 0.5 permanecen congelados para aislar el
efecto de esta politica. OOF, VALIDATION y TEST contienen solo audios originales.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import train_stage1_mfcc_lr_cough_augmentation as augmentation_base
import train_stage1_mfcc_lr_fsd50k_coughs as base


ROOT = Path(__file__).resolve().parent
ORIGINAL_FEATURES_DIR = augmentation_base.ORIGINAL_FEATURES_DIR
AUGMENTED_FEATURES_DIR = augmentation_base.AUGMENTED_FEATURES_DIR

EXPERIMENT_NAME = (
    "mfcc117_lr_fsd50k_cough_segments_audio_aug_family_weighted_random"
)
RESULTS_DIR = (
    ROOT / "results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
)
GRAPHS_DIR = (
    ROOT / "graphs_results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
)
MODEL_PATH = RESULTS_DIR / "stage1_lr_cough_segments_audio_aug_family_weighted.joblib"

NO_COUGH_ORIGINAL_WEIGHT = 1.00
COUGH_ORIGINAL_WEIGHT = 0.50
COUGH_AUGMENTED_WEIGHT = 0.25
EXPECTED_VARIANTS_PER_COUGH = 2
AUGMENTATION_DISPLAY_NAME = "AUGMENTATION PONDERADA POR FAMILIA"
AUGMENTATION_PROTOCOL = (
    "two distinct random transformations per cough among noise at 10/20 dB "
    "SNR and pitch shift at +/-1.5 semitones"
)
WEIGHT_DESCRIPTION = (
    "negativo original=1.00; tos original=0.50; "
    "cada variante=0.25; total por familia de tos=1.00"
)

REFERENCE_EXPERIMENTS = {
    "no_augmentation": (
        ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_random"
        / "full"
    ),
    "full_weight_augmentation": augmentation_base.RESULTS_DIR,
}

# Las utilidades heredadas consultan estas rutas globales.
base.FEATURES_DIR = ORIGINAL_FEATURES_DIR
base.EXPERIMENT_NAME = EXPERIMENT_NAME
base.RESULTS_DIR = RESULTS_DIR
base.GRAPHS_DIR = GRAPHS_DIR
base.MODEL_PATH = MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 MFCC117 + LR fija con augmentation ponderada por familia"
        )
    )
    parser.add_argument(
        "--action",
        choices=("train", "test"),
        default="train",
        help=(
            "train calcula OOF y VALIDATION; test evalua despues el modelo "
            "congelado sobre TEST"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar resultados ya existentes de la misma accion",
    )
    return parser.parse_args()


def original_sample_weights(y: np.ndarray) -> np.ndarray:
    """Peso 1 para negativos y 0.5 para toses originales."""
    return np.where(
        y == 0,
        NO_COUGH_ORIGINAL_WEIGHT,
        COUGH_ORIGINAL_WEIGHT,
    ).astype(np.float64)


def fit_weighted_model(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
):
    """Ajusta scaler y LR con los mismos pesos y devuelve el pipeline."""
    if len(X) != len(y) or len(y) != len(sample_weight):
        raise ValueError("X, y y sample_weight no estan alineados.")
    if not np.isfinite(sample_weight).all() or np.any(sample_weight <= 0):
        raise ValueError("Todos los pesos deben ser positivos y finitos.")

    model = base.make_model()
    model.fit(
        X,
        y,
        scaler__sample_weight=sample_weight,
        classifier__sample_weight=sample_weight,
    )
    return model


def validate_family_weights(
    y_original: np.ndarray,
    original_mask: np.ndarray,
    augmented_mask: np.ndarray,
) -> dict[str, float | int]:
    """Comprueba que la contribucion positiva equivale a una por familia."""
    n_negative = int(np.sum(original_mask & (y_original == 0)))
    n_positive = int(np.sum(original_mask & (y_original == 1)))
    n_augmented = int(np.sum(augmented_mask))
    expected_augmented = EXPECTED_VARIANTS_PER_COUGH * n_positive
    if n_augmented != expected_augmented:
        raise ValueError(
            "El numero de variantes no coincide con "
            f"{EXPECTED_VARIANTS_PER_COUGH} por tos del train "
            f"interno: {n_augmented} frente a {expected_augmented}."
        )

    effective_negative = n_negative * NO_COUGH_ORIGINAL_WEIGHT
    effective_positive = (
        n_positive * COUGH_ORIGINAL_WEIGHT
        + n_augmented * COUGH_AUGMENTED_WEIGHT
    )
    if not np.isclose(effective_positive, float(n_positive)):
        raise ValueError("El peso total de las familias positivas no es 1.0.")
    return {
        "n_original_negative_train": n_negative,
        "n_original_positive_train": n_positive,
        "n_augmented_positive_train": n_augmented,
        "effective_negative_weight": float(effective_negative),
        "effective_positive_weight": float(effective_positive),
        "effective_total_weight": float(effective_negative + effective_positive),
    }


def compare_with_references(metrics: pd.DataFrame) -> None:
    """Guarda diferencias frente al baseline y a la augmentation con peso 1."""
    rows: list[dict[str, Any]] = []
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision_cough",
        "recall_cough",
        "specificity_no_cough",
        "f1_cough",
        "macro_f1",
        "roc_auc",
    )
    for reference_name, reference_dir in REFERENCE_EXPERIMENTS.items():
        reference_path = reference_dir / "metrics_summary.csv"
        if not reference_path.is_file():
            continue
        reference = pd.read_csv(reference_path)
        for dataset in ("train_oof", "validation"):
            current_row = metrics.loc[metrics["dataset"] == dataset]
            reference_row = reference.loc[reference["dataset"] == dataset]
            if current_row.empty or reference_row.empty:
                continue
            current = current_row.iloc[0]
            previous = reference_row.iloc[0]
            for metric in metric_names:
                rows.append(
                    {
                        "reference_experiment": reference_name,
                        "dataset": dataset,
                        "metric": metric,
                        "reference_value": float(previous[metric]),
                        "family_weighted_value": float(current[metric]),
                        "difference": float(current[metric] - previous[metric]),
                    }
                )
    if rows:
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / "comparison_to_reference_experiments.csv",
            index=False,
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
        augmentation_base.load_augmented_features()
    )
    augmentation = augmentation_base.validate_augmented_features(
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
    print(f"TRAIN aumentado fisico: {X_augmented.shape}; solo tos")
    print(f"Pesos: {WEIGHT_DESCRIPTION}")
    print(f"Transformaciones: {augmentation['transformation_counts']}")
    print("Los pesos se aplican a StandardScaler y LogisticRegression.")
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
        weight_diagnostics = validate_family_weights(
            y_train,
            original_train_mask,
            augmented_train_mask,
        )

        X_fit = np.vstack(
            [X_train[original_train_mask], X_augmented[augmented_train_mask]]
        )
        y_fit = np.concatenate(
            [y_train[original_train_mask], y_augmented[augmented_train_mask]]
        )
        fit_weights = np.concatenate(
            [
                original_sample_weights(y_train[original_train_mask]),
                np.full(
                    int(augmented_train_mask.sum()),
                    COUGH_AUGMENTED_WEIGHT,
                    dtype=np.float64,
                ),
            ]
        )

        started = time.perf_counter()
        model = fit_weighted_model(X_fit, y_fit, fit_weights)
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
                "weight_policy": "family_normalized_1p0",
                "fold": int(fold),
                "n_physical_train": int(len(X_fit)),
                "n_original_validation": int(original_validation_mask.sum()),
                "n_augmented_validation": 0,
                "n_new_fsd50k_cough_validation": int(fold_new_mask.sum()),
                "recall_new_fsd50k_cough": (
                    float(np.mean(oof_prediction[fold_new_mask] == 1))
                    if fold_new_mask.any()
                    else np.nan
                ),
                "fit_seconds": fit_seconds,
                **weight_diagnostics,
                **metrics,
            }
        )
        print(
            f"  Fold {fold}: fisicas={len(X_fit)} | "
            f"peso efectivo no_tos/tos="
            f"{weight_diagnostics['effective_negative_weight']:.0f}/"
            f"{weight_diagnostics['effective_positive_weight']:.0f} | "
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
    for row in oof_source.to_dict(orient="records"):
        print(
            f"  OOF {row['cough_source']}: recall={row['recall_cough']:.4f} "
            f"(n={row['n_cough_segments']})"
        )

    print("\nAjustando LR final con pesos normalizados por familia...")
    final_original_mask = np.ones(len(y_train), dtype=bool)
    final_augmented_mask = np.ones(len(y_augmented), dtype=bool)
    final_weight_diagnostics = validate_family_weights(
        y_train,
        final_original_mask,
        final_augmented_mask,
    )
    X_final = np.vstack([X_train, X_augmented])
    y_final = np.concatenate([y_train, y_augmented])
    final_weights = np.concatenate(
        [
            original_sample_weights(y_train),
            np.full(
                len(y_augmented),
                COUGH_AUGMENTED_WEIGHT,
                dtype=np.float64,
            ),
        ]
    )
    started = time.perf_counter()
    final_model = fit_weighted_model(X_final, y_final, final_weights)
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
            for row in source_recall.to_dict(orient="records"):
                print(
                    f"  VALIDATION {row['cough_source']}: "
                    f"recall={row['recall_cough']:.4f} "
                    f"(n={row['n_cough_segments']})"
                )

    metrics_table = pd.DataFrame(metrics_rows)
    metrics_table.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    compare_with_references(metrics_table)
    joblib.dump(final_model, MODEL_PATH)
    model_size_kb = MODEL_PATH.stat().st_size / 1024.0
    benchmark = base.benchmark_inference(final_model, X_validation)
    complexity = base.model_complexity(final_model)

    configuration: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "family": "lr",
        "winner": base.MODEL_NAME,
        "selection_rule": (
            "classifier, threshold and augmentation frozen; experiment isolates "
            "family-normalized sample weighting"
        ),
        "original_feature_source": str(ORIGINAL_FEATURES_DIR),
        "augmented_feature_source": str(AUGMENTED_FEATURES_DIR),
        "target_mapping": "0=no_cough; 1=cough",
        "n_features": int(X_train.shape[1]),
        "augmentation_scope": "TRAIN cough class only",
        "augmentation_protocol": AUGMENTATION_PROTOCOL,
        "weight_policy": "family_normalized",
        "no_cough_original_weight": NO_COUGH_ORIGINAL_WEIGHT,
        "cough_original_weight": COUGH_ORIGINAL_WEIGHT,
        "cough_augmented_weight_each": COUGH_AUGMENTED_WEIGHT,
        "cough_family_total_weight": (
            COUGH_ORIGINAL_WEIGHT
            + EXPECTED_VARIANTS_PER_COUGH * COUGH_AUGMENTED_WEIGHT
        ),
        "sample_weight_applied_to": "StandardScaler and LogisticRegression",
        "n_original_train": len(X_train),
        "n_original_no_cough_train": int(np.sum(y_train == 0)),
        "n_original_cough_train": int(np.sum(y_train == 1)),
        "n_augmented_cough_train": len(X_augmented),
        "n_physical_final_fit": len(X_final),
        "effective_no_cough_weight_final": final_weight_diagnostics[
            "effective_negative_weight"
        ],
        "effective_cough_weight_final": final_weight_diagnostics[
            "effective_positive_weight"
        ],
        "effective_total_weight_final": final_weight_diagnostics[
            "effective_total_weight"
        ],
        "variants_per_cough": EXPECTED_VARIANTS_PER_COUGH,
        "transformation_counts": str(augmentation["transformation_counts"]),
        "oof_policy": "original samples only; variants of held-out parents excluded",
        "validation_augmentation": False,
        "test_augmentation": False,
        "normalization": "weighted StandardScaler fitted inside each fold",
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
    print(
        "Peso efectivo final no_tos/tos: "
        f"{final_weight_diagnostics['effective_negative_weight']:.0f}/"
        f"{final_weight_diagnostics['effective_positive_weight']:.0f}"
    )
    print(f"Modelo: {model_size_kb:.2f} KB")
    print(f"Resultados: {RESULTS_DIR}")
    print("TEST permanece reservado. Para evaluarlo: --action test")
    base.update_excel()


def main() -> None:
    args = parse_args()
    if args.action == "train":
        train(overwrite=args.overwrite)
    else:
        base.validate_feature_configuration()
        base.evaluate_test(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
