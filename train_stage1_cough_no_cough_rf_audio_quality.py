"""Repite Stage 1 entrenando sin la calidad de tos reservada para TEST.

El objetivo principal sigue siendo tos/no-tos. Ademas de las metricas globales,
el script informa del recall de las toses good, ok y poor dentro de TEST. Asi se
puede comprobar si la degradacion de calidad afecta a la sensibilidad sin
confundirla con la composicion mixta del conjunto completo.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
THRESHOLD = 0.5
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 MFCC117+RF entrenado con toses de mejor calidad"
    )
    parser.add_argument(
        "--test-quality",
        choices=["poor", "ok"],
        default="poor",
        help="Calidad que fue forzada a TEST al construir los splits.",
    )
    return parser.parse_args()


def make_model(n_jobs: int = -1) -> Pipeline:
    params = dict(RF_PARAMS)
    params["n_jobs"] = n_jobs
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(**params)),
        ]
    )


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = np.nan
    if np.unique(y_true).size == 2:
        auc = float(roc_auc_score(y_true, y_prob))
    return {
        "n_samples": len(y_true),
        "n_no_cough": int(np.sum(y_true == 0)),
        "n_cough": int(np.sum(y_true == 1)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_cough": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall_cough": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity_no_cough": recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "f1_cough": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": auc,
        "tn_no_cough_correct": int(tn),
        "fp_no_cough_as_cough": int(fp),
        "fn_cough_as_no_cough": int(fn),
        "tp_cough_correct": int(tp),
    }


def load_split(
    features_dir: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    npy_split = "val" if split == "validation" else split
    X = np.load(features_dir / f"X_{npy_split}.npy")
    y = np.load(features_dir / f"y_{npy_split}.npy").astype(np.int8)
    manifest = pd.read_csv(features_dir / f"metadata_features_{split}.csv")
    if X.ndim != 2 or X.shape[1] != 117:
        raise ValueError(f"X_{npy_split} inesperado: {X.shape}")
    if not (len(X) == len(y) == len(manifest)):
        raise ValueError(f"Datos desalineados en {split}")
    if not np.array_equal(
        y, manifest["stage1_target"].to_numpy(dtype=np.int8)
    ):
        raise ValueError(f"Las etiquetas y el manifiesto no coinciden en {split}")
    if not np.isfinite(X).all():
        raise ValueError(f"X_{npy_split} contiene NaN o infinitos")
    return X, y, manifest


def save_predictions(
    output_dir: Path,
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
    output.to_csv(output_dir / f"{split}_predictions.csv", index=False)


def save_graphs(
    graphs_dir: Path,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    image = ax.imshow(cm, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No tos", "Tos"],
        yticklabels=["No tos", "Tos"],
        xlabel="Prediccion",
        ylabel="Etiqueta real",
        title=f"Stage 1 RF audio-quality - {split}",
    )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(graphs_dir / f"confusion_matrix_{split}.png", dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.plot(fpr, tpr, label=f"RF (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(
        xlabel="False positive rate",
        ylabel="Recall tos",
        title=f"ROC - {split}",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(graphs_dir / f"roc_curve_{split}.png", dpi=180)
    plt.close(fig)


def quality_diagnostics(
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mide sensibilidad por calidad y subconjuntos binarios comparables."""
    quality = manifest["quality"].fillna("missing").astype(str).str.lower()
    cough_rows: list[dict[str, float | int | str]] = []
    binary_rows: list[dict[str, float | int | str]] = []
    no_cough_mask = y_true == 0

    for level in ["good", "ok", "poor"]:
        cough_mask = (y_true == 1) & (quality == level)
        n_cough = int(cough_mask.sum())
        if n_cough == 0:
            continue
        cough_rows.append(
            {
                "cough_quality": level,
                "n_cough_segments": n_cough,
                "n_cough_recordings": int(
                    manifest.loc[cough_mask, "original_uuid"].astype(str).nunique()
                ),
                "recall_cough": float(np.mean(y_pred[cough_mask] == 1)),
                "false_negative_rate": float(np.mean(y_pred[cough_mask] == 0)),
                "mean_probability_cough": float(np.mean(y_prob[cough_mask])),
                "median_probability_cough": float(np.median(y_prob[cough_mask])),
            }
        )

        # Se reutilizan exactamente los mismos no-tos para que la diferencia
        # entre filas dependa de la calidad de las toses, no de los controles.
        subset = no_cough_mask | cough_mask
        binary_rows.append(
            {
                "dataset": f"test_{level}_cough_vs_shared_no_cough",
                "cough_quality": level,
                "threshold": THRESHOLD,
                **classification_metrics(
                    y_true[subset], y_pred[subset], y_prob[subset]
                ),
            }
        )

    return pd.DataFrame(cough_rows), pd.DataFrame(binary_rows)


def main() -> None:
    args = parse_args()
    features_dir = (
        ROOT
        / f"features_extracted_stage1_audio_quality_{args.test_quality}"
        / "mfcc117"
    )
    results_dir = (
        ROOT
        / "results_stage1_cough_no_cough"
        / f"mfcc117_rf_audio_quality_{args.test_quality}"
        / "full"
    )
    graphs_dir = (
        ROOT
        / "graphs_results_stage1_cough_no_cough"
        / f"mfcc117_rf_audio_quality_{args.test_quality}"
        / "full"
    )
    if not features_dir.is_dir():
        raise FileNotFoundError(
            f"No existe {features_dir}. Ejecuta primero:\n"
            "python .\\feature_extraction_stage1_mfcc_audio_quality.py "
            f"--test-quality {args.test_quality}"
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STAGE 1 - MFCC117 + RF - TRAIN MEJOR / TEST PEOR")
    print("=" * 78)
    print(f"Calidad de tos forzada a TEST: {args.test_quality}")
    print("Umbral fijo: 0.5")
    print("TEST solo se usa tras cerrar el RF; no selecciona hiperparametros.")

    X_train, y_train, manifest_train = load_split(features_dir, "train")
    X_validation, y_validation, manifest_validation = load_split(
        features_dir, "validation"
    )
    X_test, y_test, manifest_test = load_split(features_dir, "test")
    folds = np.load(features_dir / "folds_train.npy").astype(int)
    if len(folds) != len(y_train):
        raise ValueError("folds_train no esta alineado con TRAIN")
    if not np.array_equal(
        folds, manifest_train["fold"].to_numpy(dtype=int)
    ):
        raise ValueError("Los folds no coinciden con el manifiesto")
    folds_per_recording = (
        pd.DataFrame(
            {
                "original_uuid": manifest_train["original_uuid"].astype(str),
                "fold": folds,
            }
        )
        .groupby("original_uuid")["fold"]
        .nunique()
    )
    if (folds_per_recording > 1).any():
        raise ValueError("Fuga entre folds: una grabacion aparece en varios folds")

    print(f"TRAIN:      {X_train.shape}; no_tos/tos={np.bincount(y_train).tolist()}")
    print(
        f"VALIDATION: {X_validation.shape}; "
        f"no_tos/tos={np.bincount(y_validation).tolist()}"
    )
    print(f"TEST:       {X_test.shape}; no_tos/tos={np.bincount(y_test).tolist()}")

    oof_pred = np.zeros(len(y_train), dtype=np.int8)
    oof_prob = np.zeros(len(y_train), dtype=np.float64)
    fold_rows: list[dict[str, float | int]] = []
    print("\nCV de TRAIN con scaler y RF ajustados dentro de cada fold:")
    for fold in sorted(np.unique(folds)):
        inner_validation = folds == fold
        inner_train = ~inner_validation
        model = make_model()
        started = time.perf_counter()
        model.fit(X_train[inner_train], y_train[inner_train])
        elapsed = time.perf_counter() - started
        probability = model.predict_proba(X_train[inner_validation])[:, 1]
        prediction = (probability >= THRESHOLD).astype(np.int8)
        oof_prob[inner_validation] = probability
        oof_pred[inner_validation] = prediction
        row = {
            "fold": int(fold),
            "n_train": int(inner_train.sum()),
            "n_validation": int(inner_validation.sum()),
            "fit_seconds": elapsed,
            **classification_metrics(
                y_train[inner_validation], prediction, probability
            ),
        }
        fold_rows.append(row)
        print(
            f"  Fold {fold}: F1 tos={row['f1_cough']:.4f} | "
            f"macro-F1={row['macro_f1']:.4f} | recall tos={row['recall_cough']:.4f}"
        )

    metrics_rows: list[dict[str, float | int | str]] = [
        {
            "dataset": "train_oof",
            "threshold": THRESHOLD,
            **classification_metrics(y_train, oof_pred, oof_prob),
        }
    ]
    pd.DataFrame(fold_rows).to_csv(results_dir / "cv_fold_metrics.csv", index=False)
    oof_output = manifest_train.copy()
    oof_output["y_true"] = y_train
    oof_output["y_pred"] = oof_pred
    oof_output["probability_cough"] = oof_prob
    oof_output["threshold"] = THRESHOLD
    oof_output.to_csv(results_dir / "oof_predictions.csv", index=False)

    print("\nAjustando el RF final exclusivamente con TRAIN...")
    final_model = make_model()
    started = time.perf_counter()
    final_model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    test_prediction: np.ndarray | None = None
    test_probability: np.ndarray | None = None
    for split, X, y, manifest in [
        ("train_fit", X_train, y_train, manifest_train),
        ("validation", X_validation, y_validation, manifest_validation),
        ("test", X_test, y_test, manifest_test),
    ]:
        probability = final_model.predict_proba(X)[:, 1]
        prediction = (probability >= THRESHOLD).astype(np.int8)
        row = {
            "dataset": split,
            "threshold": THRESHOLD,
            **classification_metrics(y, prediction, probability),
        }
        metrics_rows.append(row)
        if split != "train_fit":
            save_predictions(
                results_dir, split, manifest, y, prediction, probability
            )
            save_graphs(graphs_dir, split, y, prediction, probability)
        if split == "test":
            test_prediction = prediction
            test_probability = probability
        print(
            f"{split:>10}: F1 tos={row['f1_cough']:.4f} | "
            f"macro-F1={row['macro_f1']:.4f} | recall tos={row['recall_cough']:.4f} | "
            f"AUC={row['roc_auc']:.4f}"
        )

    assert test_prediction is not None and test_probability is not None
    quality_recall, quality_binary = quality_diagnostics(
        manifest_test,
        y_test,
        test_prediction,
        test_probability,
    )
    quality_recall.to_csv(results_dir / "test_cough_recall_by_quality.csv", index=False)
    quality_binary.to_csv(results_dir / "test_binary_metrics_by_quality.csv", index=False)
    metrics_rows.extend(quality_binary.to_dict(orient="records"))
    pd.DataFrame(metrics_rows).to_csv(results_dir / "metrics_summary.csv", index=False)

    print("\nRecall de las toses de TEST por calidad:")
    if quality_recall.empty:
        print("  No hay grupos de calidad disponibles.")
    else:
        for row in quality_recall.to_dict(orient="records"):
            print(
                f"  {row['cough_quality']:>4}: recall={row['recall_cough']:.4f} | "
                f"grabaciones={row['n_cough_recordings']} | "
                f"segmentos={row['n_cough_segments']}"
            )

    model_path = results_dir / "stage1_mfcc117_rf_audio_quality.joblib"
    joblib.dump(final_model, model_path)
    rf = final_model.named_steps["classifier"]
    model_size_kb = model_path.stat().st_size / 1024
    configuration = {
        "experiment": "stage1_cough_no_cough_mfcc117_rf_audio_quality",
        "feature_source": str(features_dir),
        "protocol": (
            f"all cough quality={args.test_quality} forced to TEST; "
            "remaining TEST slots sampled from the other recordings"
        ),
        "test_quality_forced": args.test_quality,
        "target_mapping": "0=no_cough; 1=dry|wet|unknown",
        "n_features": X_train.shape[1],
        "normalization": "StandardScaler inside each fold; final fitted on TRAIN",
        "threshold": THRESHOLD,
        "threshold_selection": "fixed; no threshold tuning",
        **RF_PARAMS,
        "fit_final_seconds": fit_seconds,
        "tree_nodes_total": sum(tree.tree_.node_count for tree in rf.estimators_),
        "model_size_kb": model_size_kb,
        "test_note": (
            "TEST is mixed; degradation is interpreted through the quality-specific "
            "recall and binary subsets with shared no-cough controls"
        ),
    }
    pd.DataFrame(configuration.items(), columns=["parameter", "value"]).to_csv(
        results_dir / "experiment_configuration.csv", index=False
    )

    print("\n" + "=" * 78)
    print("RESULTADOS STAGE 1 AUDIO-QUALITY GUARDADOS")
    print("=" * 78)
    print(f"Resultados: {results_dir}")
    print(f"Graficas:   {graphs_dir}")
    print(f"Modelo:     {model_size_kb:.2f} KB")

    try:
        from build_stage1_experiments_excel import build_workbook

        excel_path = build_workbook()
        print(f"Excel actualizado: {excel_path}")
    except Exception as exc:
        print(f"AVISO: no se pudo actualizar el Excel automaticamente: {exc}")


if __name__ == "__main__":
    main()
