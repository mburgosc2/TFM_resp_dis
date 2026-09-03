"""Evaluacion final de TEST del WST recording + LR ya congelado.

No reentrena ni selecciona nada. Carga el pipeline StandardScaler + PCA128 +
LR guardado, aplica el umbral OOF congelado y genera predicciones, metricas y
una figura final de TEST.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRESET = "paper_q8_q1_t500_full"
MODEL_ROOT = SCRIPT_DIR / "models_stage2_dry_wet_wst_recording_lr_two_phase_search"
RESULT_ROOT = SCRIPT_DIR / "results_stage2_dry_wet_wst_recording_lr_two_phase_search"
FEATURE_ROOT = SCRIPT_DIR / "features_extracted_stage2_dry_wet_wavelet_scattering"
GRAPH_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wst_recording_lr_two_phase_search"
)
EXPECTED_EXPERIMENT = "wst_recording_lr_two_phase_search"
EXPECTED_CANDIDATE = (
    "logistic_regression__C0.001__pca128__no_whiten__gold1__wetcost1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_paths(preset: str) -> dict[str, Path]:
    feature_dir = FEATURE_ROOT / preset
    return {
        "model": MODEL_ROOT / preset / "full" / "logistic_regression_model.joblib",
        "x_test": feature_dir / "X_events_test.npy",
        "y_test": feature_dir / "y_events_test.npy",
        "folds_test": feature_dir / "folds_events_test.npy",
        "metadata_test": feature_dir / "metadata_events_features_test.csv",
        "feature_layout": feature_dir / "wavelet_scattering_feature_layout.csv",
        "train_metadata": feature_dir / "metadata_events_features_train.csv",
        "validation_metadata": feature_dir / "metadata_events_features_validation.csv",
    }


def load_frozen_package(model_path: Path, preset: str) -> dict[str, Any]:
    package = joblib.load(model_path)
    required = {
        "pipeline",
        "experiment",
        "preset",
        "candidate_key",
        "C",
        "pca_components",
        "pca_whiten",
        "gold_multiplier",
        "wet_cost",
        "threshold",
        "recording_feature_names",
    }
    missing = required - set(package)
    if missing:
        raise ValueError(f"Faltan campos en el modelo: {sorted(missing)}")
    if package["experiment"] != EXPECTED_EXPERIMENT:
        raise ValueError(f"Experimento inesperado: {package['experiment']}")
    if package["preset"] != preset:
        raise ValueError(f"Preset del modelo inesperado: {package['preset']}")
    if package["candidate_key"] != EXPECTED_CANDIDATE:
        raise ValueError(
            "El modelo no es el ganador congelado esperado: "
            f"{package['candidate_key']}"
        )
    frozen_values = {
        "C": (float(package["C"]), 0.001),
        "pca_components": (int(package["pca_components"]), 128),
        "pca_whiten": (bool(package["pca_whiten"]), False),
        "gold_multiplier": (float(package["gold_multiplier"]), 1.0),
        "wet_cost": (float(package["wet_cost"]), 1.0),
    }
    for name, (actual, expected) in frozen_values.items():
        if actual != expected:
            raise ValueError(f"{name} no coincide: {actual} frente a {expected}.")
    threshold = float(package["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Umbral congelado invalido: {threshold}")
    return package


def load_test_recording_view(
    paths: dict[str, Path], package: dict[str, Any]
) -> tuple[np.ndarray, pd.DataFrame]:
    x_events = np.load(paths["x_test"])
    y_events = np.load(paths["y_test"])
    folds = np.load(paths["folds_test"])
    metadata = pd.read_csv(paths["metadata_test"])
    feature_layout = pd.read_csv(paths["feature_layout"])
    common.validate_arrays_and_metadata(
        "event", "test", x_events, y_events, folds, metadata
    )
    if set(folds.astype(int)) != {-1}:
        raise ValueError("TEST debe conservar fold=-1.")
    if len(feature_layout) != x_events.shape[1]:
        raise ValueError("El layout WST no coincide con X_events_test.")
    if set(feature_layout["scattering_order"].astype(int)) != {0, 1, 2}:
        raise ValueError("TEST no contiene exactamente WST de orden 0, 1 y 2.")

    for other_split, path_key in (
        ("train", "train_metadata"),
        ("validation", "validation_metadata"),
    ):
        other = pd.read_csv(paths[path_key], usecols=["original_uuid"])
        overlap = set(metadata["original_uuid"].astype(str)) & set(
            other["original_uuid"].astype(str)
        )
        if overlap:
            raise ValueError(
                f"TEST comparte UUID con {other_split}: {sorted(overlap)[:10]}"
            )

    pooling = next(
        spec for spec in recording.POOLING_SPECS if spec.key == "mean_std_max"
    )
    event_feature_names = feature_layout["feature_name"].astype(str).tolist()
    x_recording, recording_metadata = recording.aggregate_split(
        x_events,
        metadata,
        event_feature_names,
        pooling,
        "test",
    )
    expected_names = recording.recording_feature_names(
        event_feature_names, pooling
    )
    frozen_names = [str(name) for name in package["recording_feature_names"]]
    if expected_names != frozen_names:
        raise ValueError(
            "Los nombres/orden de las features de TEST no coinciden con el modelo."
        )
    if x_recording.shape[1] != len(frozen_names):
        raise ValueError("La dimension recording de TEST no coincide con el modelo.")
    return x_recording, recording_metadata


def build_predictions(
    package: dict[str, Any],
    x_recording: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    scores = common.model_scores(package["pipeline"], x_recording)
    predictions = common.aggregate_scores_by_recording(metadata, scores)
    threshold = float(package["threshold"])
    predictions["y_pred_oof_threshold"] = (
        predictions["score"].to_numpy(float) >= threshold
    ).astype(int)
    predictions["y_pred_native_threshold"] = (
        predictions["score"].to_numpy(float) >= 0.5
    ).astype(int)
    predictions["predicted_label"] = predictions[
        "y_pred_oof_threshold"
    ].map({0: "dry", 1: "wet"})
    predictions["candidate_key"] = package["candidate_key"]
    predictions["threshold_source"] = "frozen_train_oof"
    return predictions


def subgroup_rows(
    predictions: pd.DataFrame, threshold: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = [("all", predictions)]
    groups.extend(
        (str(name), group)
        for name, group in predictions.groupby(
            "cough_type_consensus", sort=True, dropna=False
        )
    )
    for name, group in groups:
        labels = set(group["y_true"].astype(int))
        row: dict[str, Any] = {
            "dataset": "test",
            "subgroup": name,
            "recording_count": len(group),
            "dry_count": int((group["y_true"] == 0).sum()),
            "wet_count": int((group["y_true"] == 1).sum()),
            "threshold": threshold,
        }
        if labels == {0, 1}:
            row.update(
                common.binary_metrics(
                    group["y_true"].to_numpy(int),
                    group["score"].to_numpy(float),
                    threshold,
                )
            )
        rows.append(row)
    return rows


def create_test_graph(
    predictions: pd.DataFrame,
    threshold: float,
    candidate_key: str,
    output_path: Path,
) -> None:
    y_true = predictions["y_true"].to_numpy(int)
    scores = predictions["score"].to_numpy(float)
    y_pred = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    denominators = matrix.sum(axis=1, keepdims=True)
    row_percent = np.divide(
        matrix * 100.0,
        denominators,
        out=np.zeros_like(matrix, dtype=float),
        where=denominators != 0,
    )

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[0].text(
                column,
                row,
                f"{matrix[row, column]}\n({row_percent[row, column]:.1f}%)",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axes[0].set_xticks([0, 1], ["Dry", "Wet"])
    axes[0].set_yticks([0, 1], ["Dry", "Wet"])
    axes[0].set_xlabel("Prediccion")
    axes[0].set_ylabel("Etiqueta real")
    axes[0].set_title("Matriz de confusion")
    figure.colorbar(image, ax=axes[0], fraction=0.046)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    axes[1].plot(
        false_positive_rate, true_positive_rate, label=f"AUC={roc_auc:.3f}"
    )
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    average_precision = average_precision_score(y_true, scores)
    prevalence = float(np.mean(y_true == 1))
    axes[2].plot(recall, precision, label=f"AP wet={average_precision:.3f}")
    axes[2].axhline(
        prevalence,
        linestyle="--",
        color="grey",
        label=f"Baseline={prevalence:.3f}",
    )
    axes[2].set_xlabel("Recall wet")
    axes[2].set_ylabel("Precision wet")
    axes[2].set_title("Precision-recall")
    axes[2].legend()
    axes[2].grid(alpha=0.2)

    figure.suptitle(
        f"TEST final | WST recording + LR | threshold={threshold:.4f}\n"
        f"{candidate_key}",
        fontsize=12,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate(preset: str, overwrite: bool) -> None:
    paths = required_paths(preset)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan archivos requeridos:\n" + "\n".join(missing))

    result_dir = RESULT_ROOT / preset / "full"
    graph_dir = GRAPH_ROOT / preset / "full"
    test_predictions_path = result_dir / "test_predictions.csv"
    test_metrics_path = result_dir / "test_metrics_by_consensus_subgroup.csv"
    test_audit_path = result_dir / "test_evaluation_configuration.csv"
    graph_path = graph_dir / "test_wst_lr_two_phase.png"
    outputs = [
        test_predictions_path,
        test_metrics_path,
        test_audit_path,
        graph_path,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "TEST ya parece haber sido evaluado. Para repetir exactamente la "
            "misma evaluacion usa --overwrite. Archivos:\n"
            + "\n".join(str(path) for path in existing)
        )

    package = load_frozen_package(paths["model"], preset)
    x_recording, metadata = load_test_recording_view(paths, package)
    predictions = build_predictions(package, x_recording, metadata)
    threshold = float(package["threshold"])
    frozen_metrics = common.binary_metrics(
        predictions["y_true"].to_numpy(int),
        predictions["score"].to_numpy(float),
        threshold,
    )
    native_metrics = common.binary_metrics(
        predictions["y_true"].to_numpy(int),
        predictions["score"].to_numpy(float),
        0.5,
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(test_predictions_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(subgroup_rows(predictions, threshold)).to_csv(
        test_metrics_path, index=False, encoding="utf-8-sig"
    )

    metrics_summary_path = result_dir / "metrics_summary.csv"
    if not metrics_summary_path.is_file():
        raise FileNotFoundError(f"No existe el resumen previo: {metrics_summary_path}")
    summary = pd.read_csv(metrics_summary_path)
    summary = summary.loc[summary["dataset"].astype(str) != "test"].copy()
    test_row = {
        "dataset": "test",
        "candidate_key": package["candidate_key"],
        "threshold": threshold,
        "minimum_wet_recall_oof": package.get("minimum_wet_recall_oof", np.nan),
        "model_size_kb_joblib": paths["model"].stat().st_size / 1024.0,
        "learned_inference_float_count": np.nan,
        **frozen_metrics,
    }
    summary = pd.concat([summary, pd.DataFrame([test_row])], ignore_index=True)
    summary.to_csv(metrics_summary_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "dataset": "test",
                "model_path": str(paths["model"]),
                "model_sha256": sha256_file(paths["model"]),
                "experiment": package["experiment"],
                "candidate_key": package["candidate_key"],
                "threshold": threshold,
                "threshold_source": "train_oof_constrained_frozen",
                "pca_components": package["pca_components"],
                "pca_whiten": package["pca_whiten"],
                "gold_multiplier": package["gold_multiplier"],
                "wet_cost": package["wet_cost"],
                "recording_count": len(predictions),
                "event_count": int(metadata["event_count"].sum()),
                "recording_feature_count": x_recording.shape[1],
                "test_used_for_selection": False,
                "model_refitted_on_test": False,
                "threshold_refitted_on_test": False,
            }
        ]
    ).to_csv(test_audit_path, index=False, encoding="utf-8-sig")
    create_test_graph(
        predictions, threshold, package["candidate_key"], graph_path
    )

    print("\n" + "=" * 78)
    print("EVALUACION FINAL TEST - WST RECORDING + LR CONGELADO")
    print("=" * 78)
    print(f"Modelo: {package['candidate_key']}")
    print(f"Umbral OOF congelado: {threshold:.6f}")
    print(f"Grabaciones TEST dry/wet: {(predictions.y_true == 0).sum()} / {(predictions.y_true == 1).sum()}")
    print(f"Macro-F1 TEST: {frozen_metrics['macro_f1']:.4f}")
    print(f"Balanced accuracy TEST: {frozen_metrics['balanced_accuracy']:.4f}")
    print(
        "Recalls TEST dry/wet: "
        f"{frozen_metrics['dry_recall']:.4f} / {frozen_metrics['wet_recall']:.4f}"
    )
    print(f"AUC TEST: {frozen_metrics['roc_auc']:.4f}")
    print(f"Macro-F1 TEST @0.5 (solo referencia): {native_metrics['macro_f1']:.4f}")
    print(f"Resultados: {result_dir}")
    print(f"Grafica TEST: {graph_path}")
    print("TEST no ha modificado el modelo ni el umbral.")


def check(preset: str) -> None:
    paths = required_paths(preset)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan archivos requeridos:\n" + "\n".join(missing))
    package = load_frozen_package(paths["model"], preset)
    x_recording, metadata = load_test_recording_view(paths, package)
    print("=" * 78)
    print("CHECK - EVALUACION FINAL TEST WST RECORDING + LR")
    print("=" * 78)
    print(f"Modelo congelado: {package['candidate_key']}")
    print(f"Umbral congelado: {float(package['threshold']):.6f}")
    print(f"TEST: {len(metadata)} grabaciones; X={x_recording.shape}")
    print("No se ha calculado ninguna metrica ni escrito ningun resultado.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evalua TEST con el WST recording + LR congelado."
    )
    parser.add_argument("--action", choices=("check", "test"), default="check")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action == "check":
        check(args.preset)
        return
    evaluate(args.preset, args.overwrite)


if __name__ == "__main__":
    main()
