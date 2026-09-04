"""Evaluación final y congelada de modelos Stage 2 sobre TEST.

Este script NO entrena, NO selecciona hiperparámetros y NO ajusta umbrales.
Reutiliza exactamente:

* el MFCC117 + PCA128 + LR guardado;
* la tiny CNN Log-Mel guardada;
* el WST + PCA128 + LR usado por la late fusion;
* alpha=0.35 y umbral=0.574947... seleccionados previamente con OOF.

La acción ``extract`` crea únicamente las features MFCC y Log-Mel de TEST.
WST de TEST ya fue extraído para la evaluación final anterior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import feature_extraction_stage2_cochleograms as audio_events
import feature_extraction_stage2_logmel as logmel_features
import feature_extraction_stage2_mfcc as mfcc_features
import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models_stage2"

MFCC_FEATURE_DIR = (
    ROOT / "features_extracted_stage2_dry_wet_mfcc" / "mfcc117"
)
LOGMEL_FEATURE_DIR = (
    ROOT
    / "features_extracted_stage2_dry_wet_logmel"
    / "logmel64_win32_hop16"
)
WST_FEATURE_DIR = (
    ROOT
    / "features_extracted_stage2_dry_wet_wavelet_scattering"
    / "paper_q8_q1_t500_full"
)

MFCC_MODEL_PATH = (
    MODEL_ROOT
    / "models_stage2_dry_wet_mfcc_recording_linear"
    / "mfcc117"
    / "full"
    / "logistic_regression_model.joblib"
)
LOGMEL_MODEL_PATH = (
    MODEL_ROOT
    / "models_stage2_dry_wet_logmel_tiny_cnn"
    / "logmel64_win32_hop16"
    / "full"
    / "logmel_tiny_cnn.keras"
)
WST_FUSION_MODEL_PATH = (
    MODEL_ROOT
    / "models_stage2_dry_wet_wavelet_scattering_recording_linear"
    / "paper_q8_q1_t500_full"
    / "full"
    / "logistic_regression_model.joblib"
)

MFCC_RESULT_DIR = (
    ROOT
    / "results_stage2_dry_wet_mfcc_recording_linear"
    / "mfcc117"
    / "full"
    / "logistic_regression"
)
LOGMEL_RESULT_DIR = (
    ROOT
    / "results_stage2_dry_wet_logmel_tiny_cnn"
    / "logmel64_win32_hop16"
    / "full"
)
FUSION_RESULT_DIR = ROOT / "results_stage2_late_fusion_wst_logmel_cnn" / "full"
WST_FINAL_RESULT_DIR = (
    ROOT
    / "results_stage2_dry_wet_wst_recording_lr_two_phase_search"
    / "paper_q8_q1_t500_full"
    / "full"
)
COMPARISON_PATH = FUSION_RESULT_DIR / "stage2_test_model_comparison.csv"

MFCC_GRAPH = (
    ROOT
    / "graphs_results_stage2_dry_wet_random"
    / "training_mfcc_recording_linear"
    / "mfcc117"
    / "full"
    / "test_logistic_regression_oof_threshold.png"
)
LOGMEL_GRAPH = (
    ROOT
    / "graphs_results_stage2_dry_wet_random"
    / "training_logmel_tiny_cnn"
    / "logmel64_win32_hop16"
    / "full"
    / "test_logmel_tiny_cnn.png"
)
FUSION_GRAPH = (
    ROOT
    / "graphs_results_stage2_dry_wet_random"
    / "late_fusion_wst_logmel_cnn"
    / "full"
    / "test_max_macro_f1.png"
)

FUSION_POLICY = "max_macro_f1"


def load_event_manifest(split_name: str) -> pd.DataFrame:
    """Carga también TEST, excluido deliberadamente del extractor original."""

    filenames = {
        "train": "metadata_events_train_stage2_dry_wet.csv",
        "validation": "metadata_events_validation_stage2_dry_wet.csv",
        "test": "metadata_events_test_stage2_dry_wet.csv",
    }
    if split_name not in filenames:
        raise ValueError(f"Split desconocido: {split_name}")
    # Reutiliza todas las validaciones del cargador original sin modificar
    # permanentemente su script ni volver a segmentar los audios.
    audio_events.EVENT_FILES.setdefault(split_name, filenames[split_name])
    return audio_events.load_event_manifest(split_name)


def required_extraction_files() -> dict[str, tuple[Path, ...]]:
    return {
        "mfcc": (
            MFCC_FEATURE_DIR / "X_events_test.npy",
            MFCC_FEATURE_DIR / "metadata_events_features_test.csv",
            MFCC_FEATURE_DIR / "X_test.npy",
            MFCC_FEATURE_DIR / "metadata_recordings_features_test.csv",
        ),
        "logmel": (
            LOGMEL_FEATURE_DIR / "X_events_test.npy",
            LOGMEL_FEATURE_DIR / "metadata_events_features_test.csv",
        ),
        "wst": (
            WST_FEATURE_DIR / "X_events_test.npy",
            WST_FEATURE_DIR / "metadata_events_features_test.csv",
            WST_FEATURE_DIR / "wavelet_scattering_feature_layout.csv",
        ),
    }


def ensure_frozen_inputs() -> None:
    required = (
        MFCC_MODEL_PATH,
        LOGMEL_MODEL_PATH,
        WST_FUSION_MODEL_PATH,
        MFCC_RESULT_DIR / "metrics_summary.csv",
        LOGMEL_RESULT_DIR / "metrics_summary.csv",
        FUSION_RESULT_DIR / "selected_fusion_configurations.csv",
        WST_FINAL_RESULT_DIR / "metrics_summary.csv",
        MFCC_FEATURE_DIR / "mfcc_recording_feature_layout.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan entradas congeladas:\n" + "\n".join(missing))


def validate_test_manifest(test_manifest: pd.DataFrame) -> None:
    train = load_event_manifest("train")
    validation = load_event_manifest("validation")
    test_uuids = set(test_manifest["original_uuid"].astype(str))
    overlap_train = test_uuids & set(train["original_uuid"].astype(str))
    overlap_validation = test_uuids & set(validation["original_uuid"].astype(str))
    if overlap_train or overlap_validation:
        raise ValueError(
            "TEST comparte UUID con TRAIN/VALIDATION: "
            f"train={len(overlap_train)}, validation={len(overlap_validation)}"
        )
    if set(test_manifest["fold"].astype(int)) != {-1}:
        raise ValueError("TEST debe tener fold=-1.")


def extraction_complete(paths: tuple[Path, ...]) -> bool:
    existing = [path.is_file() for path in paths]
    if any(existing) and not all(existing):
        missing = [str(path) for path, found in zip(paths, existing) if not found]
        raise RuntimeError(
            "La extracción de TEST está incompleta. Usa --overwrite. Faltan:\n"
            + "\n".join(missing)
        )
    return all(existing)


def extract_test_features(test_manifest: pd.DataFrame, overwrite: bool) -> None:
    paths = required_extraction_files()

    if overwrite or not extraction_complete(paths["mfcc"]):
        print("Extrayendo MFCC117 de TEST...")
        names = mfcc_features.event_feature_names(mfcc_features.CONFIG)
        matrix, metadata = mfcc_features.extract_split(
            test_manifest, "test", mfcc_features.CONFIG
        )
        recording_matrix, recording_metadata, recording_names = (
            mfcc_features.save_split("test", matrix, metadata, names)
        )
        saved_names = pd.read_csv(
            MFCC_FEATURE_DIR / "mfcc_recording_feature_layout.csv"
        )["feature_name"].astype(str).tolist()
        if recording_names != saved_names:
            raise RuntimeError("El layout MFCC de TEST no coincide con TRAIN.")
        pd.DataFrame(
            [
                {
                    "split": "test",
                    "event_count": len(metadata),
                    "recording_count": len(recording_metadata),
                    "event_feature_count": matrix.shape[1],
                    "recording_feature_count": recording_matrix.shape[1],
                    "configuration_reused": "mfcc117 frozen",
                    "training_repeated": False,
                }
            ]
        ).to_csv(
            MFCC_FEATURE_DIR / "mfcc_test_extraction_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        print("MFCC TEST ya existe; no se vuelve a extraer.")

    if overwrite or not extraction_complete(paths["logmel"]):
        print("Extrayendo Log-Mel de TEST...")
        matrix, metadata = logmel_features.extract_split(
            test_manifest, "test", logmel_features.CONFIG
        )
        logmel_features.save_split("test", matrix, metadata)
        pd.DataFrame(
            [
                {
                    "split": "test",
                    "event_count": len(metadata),
                    "recording_count": metadata["original_uuid"].nunique(),
                    "tensor_shape": "x".join(map(str, matrix.shape[1:])),
                    "configuration_reused": "logmel64_win32_hop16 frozen",
                    "training_repeated": False,
                }
            ]
        ).to_csv(
            LOGMEL_FEATURE_DIR / "logmel_test_extraction_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        print("Log-Mel TEST ya existe; no se vuelve a extraer.")


def validate_prediction_frame(predictions: pd.DataFrame, name: str) -> None:
    if predictions["original_uuid"].duplicated().any():
        raise ValueError(f"{name}: UUID duplicados.")
    if set(predictions["y_true"].astype(int)) != {0, 1}:
        raise ValueError(f"{name}: TEST no contiene ambas clases.")
    scores = predictions["score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError(f"{name}: probabilidades inválidas.")


def predict_mfcc() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    package = joblib.load(MFCC_MODEL_PATH)
    matrix = np.load(MFCC_FEATURE_DIR / "X_test.npy")
    metadata = pd.read_csv(MFCC_FEATURE_DIR / "metadata_recordings_features_test.csv")
    layout = pd.read_csv(MFCC_FEATURE_DIR / "mfcc_recording_feature_layout.csv")
    names = layout["feature_name"].astype(str).tolist()
    if names != list(package["recording_feature_names"]):
        raise ValueError("Los nombres MFCC no coinciden con los del modelo congelado.")
    if matrix.shape != (len(metadata), len(names)):
        raise ValueError(f"Forma MFCC TEST inesperada: {matrix.shape}.")

    scores = package["pipeline"].predict_proba(matrix)[:, 1]
    threshold = float(package["threshold"])
    predictions = metadata[
        ["original_uuid", "cough_type", "cough_type_consensus", "event_count"]
    ].copy()
    predictions["y_true"] = metadata["stage2_target"].to_numpy(dtype=int)
    predictions["score"] = scores
    predictions["threshold"] = threshold
    predictions["y_pred"] = (scores >= threshold).astype(int)
    predictions["candidate_key"] = package["candidate_key"]
    validate_prediction_frame(predictions, "MFCC")
    metrics = common.binary_metrics(predictions["y_true"], scores, threshold)
    return predictions, metrics, package


def load_logmel_test() -> tuple[np.ndarray, pd.DataFrame]:
    matrix = np.load(LOGMEL_FEATURE_DIR / "X_events_test.npy", mmap_mode="r")
    metadata = pd.read_csv(LOGMEL_FEATURE_DIR / "metadata_events_features_test.csv")
    expected = (len(metadata), *logmel_features.CONFIG.tensor_shape)
    if matrix.shape != expected:
        raise ValueError(f"Forma Log-Mel TEST inesperada: {matrix.shape}; {expected}.")
    return matrix, metadata


def load_wst_test() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    events = np.load(WST_FEATURE_DIR / "X_events_test.npy")
    metadata = pd.read_csv(WST_FEATURE_DIR / "metadata_events_features_test.csv")
    layout = pd.read_csv(WST_FEATURE_DIR / "wavelet_scattering_feature_layout.csv")
    names = layout["feature_name"].astype(str).tolist()
    if events.shape != (len(metadata), len(names)):
        raise ValueError(f"Forma WST TEST inesperada: {events.shape}.")
    return events, metadata, names


def predict_fusion_components() -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    # TensorFlow se importa aquí para que --action check no cargue la red.
    import tensorflow as tf

    logmel_matrix, logmel_metadata = load_logmel_test()
    cnn = tf.keras.models.load_model(LOGMEL_MODEL_PATH, compile=False)
    cnn_event_scores = cnn.predict(logmel_matrix, batch_size=64, verbose=0).reshape(-1)
    cnn_recordings = common.aggregate_scores_by_recording(
        logmel_metadata, cnn_event_scores
    ).rename(columns={"score": "cnn_probability"})

    wst_events, wst_metadata, wst_event_names = load_wst_test()
    pooling = next(
        spec for spec in recording.POOLING_SPECS if spec.key == "mean_std_max"
    )
    wst_matrix, wst_recording_metadata = recording.aggregate_split(
        wst_events, wst_metadata, wst_event_names, pooling, "test"
    )
    wst_names = recording.recording_feature_names(wst_event_names, pooling)
    wst_package = joblib.load(WST_FUSION_MODEL_PATH)
    if wst_names != list(wst_package["recording_feature_names"]):
        raise ValueError("Los nombres WST no coinciden con el modelo de la fusión.")
    wst_scores = wst_package["pipeline"].predict_proba(wst_matrix)[:, 1]
    wst_recordings = wst_recording_metadata[
        ["original_uuid", "stage2_target", "cough_type", "cough_type_consensus"]
    ].copy()
    wst_recordings["wst_probability"] = wst_scores

    aligned = wst_recordings.merge(
        cnn_recordings[
            [
                "original_uuid",
                "y_true",
                "cough_type",
                "cough_type_consensus",
                "cnn_probability",
            ]
        ],
        on="original_uuid",
        how="inner",
        suffixes=("", "__cnn"),
        validate="one_to_one",
    )
    if len(aligned) != len(wst_recordings) or len(aligned) != len(cnn_recordings):
        raise ValueError("MFCC/WST/Log-Mel no contienen los mismos UUID de TEST.")
    if not np.array_equal(aligned["stage2_target"], aligned["y_true"]):
        raise ValueError("WST y Log-Mel no coinciden en las etiquetas TEST.")

    selection = pd.read_csv(FUSION_RESULT_DIR / "selected_fusion_configurations.csv")
    selected = selection[selection["policy"].astype(str) == FUSION_POLICY]
    if len(selected) != 1:
        raise ValueError("No se encuentra una única configuración late fusion congelada.")
    selected_row = selected.iloc[0].to_dict()
    alpha = float(selected_row["alpha_wst"])
    threshold = float(selected_row["threshold_oof"])
    scores = (
        alpha * aligned["wst_probability"].to_numpy(dtype=float)
        + (1.0 - alpha) * aligned["cnn_probability"].to_numpy(dtype=float)
    )

    predictions = aligned[
        ["original_uuid", "cough_type", "cough_type_consensus"]
    ].copy()
    predictions["y_true"] = aligned["stage2_target"].to_numpy(dtype=int)
    predictions["wst_probability"] = aligned["wst_probability"]
    predictions["cnn_probability"] = aligned["cnn_probability"]
    predictions["score"] = scores
    predictions["alpha_wst"] = alpha
    predictions["weight_cnn"] = 1.0 - alpha
    predictions["threshold"] = threshold
    predictions["y_pred"] = (scores >= threshold).astype(int)
    predictions["candidate_key"] = selected_row["candidate_key"]
    validate_prediction_frame(predictions, "late fusion")

    y_true = predictions["y_true"].to_numpy(dtype=int)
    fusion_metrics = common.binary_metrics(y_true, scores, threshold)
    wst_metrics = common.binary_metrics(
        y_true, predictions["wst_probability"].to_numpy(dtype=float),
        float(wst_package["threshold"]),
    )
    logmel_summary = pd.read_csv(LOGMEL_RESULT_DIR / "metrics_summary.csv")
    logmel_validation = logmel_summary[
        (logmel_summary["split"].astype(str) == "validation")
        & (logmel_summary["threshold_policy"].astype(str) == "oof_tuned_frozen")
    ]
    if len(logmel_validation) != 1:
        raise ValueError("No se encuentra el umbral OOF congelado de Log-Mel.")
    cnn_threshold = float(logmel_validation.iloc[0]["threshold"])
    cnn_metrics = common.binary_metrics(
        y_true, predictions["cnn_probability"].to_numpy(dtype=float), cnn_threshold
    )
    selected_row["threshold"] = threshold
    selected_row["cnn_threshold"] = cnn_threshold
    selected_row["wst_threshold"] = float(wst_package["threshold"])
    return predictions, fusion_metrics, wst_metrics, cnn_metrics, selected_row


def create_test_graph(
    predictions: pd.DataFrame,
    threshold: float,
    title: str,
    output_path: Path,
) -> None:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    y_pred = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    percentages = matrix / matrix.sum(axis=1, keepdims=True) * 100.0

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[0].text(
                column,
                row,
                f"{matrix[row, column]}\n({percentages[row, column]:.1f}%)",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axes[0].set_xticks([0, 1], ["Dry", "Wet"])
    axes[0].set_yticks([0, 1], ["Dry", "Wet"])
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Etiqueta real")
    axes[0].set_title("Matriz de confusión")
    figure.colorbar(image, ax=axes[0], fraction=0.046)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    axes[1].plot(false_positive_rate, true_positive_rate, linewidth=2)
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC TEST")
    axes[1].grid(alpha=0.25)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    axes[2].plot(recall, precision, linewidth=2)
    axes[2].set(xlabel="Recall wet", ylabel="Precision wet", title="Precision-Recall TEST")
    axes[2].grid(alpha=0.25)

    figure.suptitle(f"{title} — TEST — umbral OOF={threshold:.4f}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def replace_metric_row(
    path: Path,
    new_row: dict[str, Any],
    split_column: str,
    split_value: str,
    extra_filter: tuple[str, str] | None = None,
) -> None:
    data = pd.read_csv(path)
    keep = data[split_column].astype(str) != split_value
    if extra_filter is not None:
        column, value = extra_filter
        keep |= data[column].astype(str) != value
    data = data[keep].copy()
    row = {column: new_row.get(column, np.nan) for column in data.columns}
    data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def comparison_row(
    experiment_id: str,
    experiment: str,
    threshold: float,
    metrics: dict[str, Any],
    model_size_kb: float,
) -> dict[str, Any]:
    return {
        "ID": experiment_id,
        "experiment": experiment,
        "dataset": "test",
        "threshold_policy": "oof_selected_frozen",
        "threshold": threshold,
        "n_test": int(
            metrics["tn_dry_correct"]
            + metrics["fp_dry_as_wet"]
            + metrics["fn_wet_as_dry"]
            + metrics["tp_wet_correct"]
        ),
        "n_dry": int(metrics["tn_dry_correct"] + metrics["fp_dry_as_wet"]),
        "n_wet": int(metrics["fn_wet_as_dry"] + metrics["tp_wet_correct"]),
        **metrics,
        "model_size_kb": model_size_kb,
    }


def evaluate_test() -> None:
    ensure_frozen_inputs()
    paths = required_extraction_files()
    missing = [str(path) for group in paths.values() for path in group if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan features de TEST. Ejecuta primero --action extract:\n"
            + "\n".join(missing)
        )

    mfcc_predictions, mfcc_metrics, mfcc_package = predict_mfcc()
    (
        fusion_predictions,
        fusion_metrics,
        fusion_wst_metrics,
        cnn_metrics,
        fusion_config,
    ) = predict_fusion_components()

    if set(mfcc_predictions["original_uuid"]) != set(
        fusion_predictions["original_uuid"]
    ):
        raise ValueError("MFCC y late fusion no contienen los mismos UUID TEST.")

    MFCC_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mfcc_predictions.to_csv(
        MFCC_RESULT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    existing_mfcc = pd.read_csv(MFCC_RESULT_DIR / "metrics_summary.csv")
    mfcc_reference = existing_mfcc[
        (existing_mfcc["dataset"].astype(str) == "validation")
        & (existing_mfcc["threshold_policy"].astype(str) == "oof_tuned_frozen")
    ].iloc[0].to_dict()
    mfcc_row = {
        **mfcc_reference,
        "dataset": "test",
        "threshold_policy": "oof_tuned_frozen",
        "threshold": float(mfcc_package["threshold"]),
        **mfcc_metrics,
    }
    replace_metric_row(
        MFCC_RESULT_DIR / "metrics_summary.csv", mfcc_row, "dataset", "test"
    )

    LOGMEL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logmel_predictions = fusion_predictions[
        ["original_uuid", "cough_type", "cough_type_consensus", "y_true"]
    ].copy()
    logmel_predictions["score"] = fusion_predictions["cnn_probability"]
    logmel_predictions["threshold"] = float(fusion_config["cnn_threshold"])
    logmel_predictions["y_pred"] = (
        logmel_predictions["score"] >= logmel_predictions["threshold"]
    ).astype(int)
    logmel_predictions.to_csv(
        LOGMEL_RESULT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    existing_logmel = pd.read_csv(LOGMEL_RESULT_DIR / "metrics_summary.csv")
    logmel_reference = existing_logmel[
        (existing_logmel["split"].astype(str) == "validation")
        & (existing_logmel["threshold_policy"].astype(str) == "oof_tuned_frozen")
    ].iloc[0].to_dict()
    logmel_row = {
        **logmel_reference,
        "split": "test",
        "threshold_policy": "oof_tuned_frozen",
        "threshold": float(fusion_config["cnn_threshold"]),
        **cnn_metrics,
    }
    replace_metric_row(
        LOGMEL_RESULT_DIR / "metrics_summary.csv", logmel_row, "split", "test"
    )

    FUSION_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fusion_predictions.to_csv(
        FUSION_RESULT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    fusion_metrics_path = FUSION_RESULT_DIR / "metrics_summary.csv"
    fusion_row = {
        "dataset": "test",
        "policy": FUSION_POLICY,
        "candidate_key": fusion_config["candidate_key"],
        "alpha_wst": float(fusion_config["alpha_wst"]),
        "weight_cnn": float(fusion_config["weight_cnn"]),
        "threshold_policy": "oof_selected_frozen",
        "threshold": float(fusion_config["threshold"]),
        "combined_model_size_kb": float(fusion_config["combined_model_size_kb"]),
        **fusion_metrics,
    }
    replace_metric_row(
        fusion_metrics_path,
        fusion_row,
        "dataset",
        "test",
        extra_filter=("policy", FUSION_POLICY),
    )

    create_test_graph(
        mfcc_predictions,
        float(mfcc_package["threshold"]),
        "MFCC117 + PCA128 + LR",
        MFCC_GRAPH,
    )
    create_test_graph(
        logmel_predictions,
        float(fusion_config["cnn_threshold"]),
        "Log-Mel + tiny CNN",
        LOGMEL_GRAPH,
    )
    create_test_graph(
        fusion_predictions,
        float(fusion_config["threshold"]),
        "Late fusion WST-LR + Log-Mel CNN",
        FUSION_GRAPH,
    )

    final_wst = pd.read_csv(WST_FINAL_RESULT_DIR / "metrics_summary.csv")
    final_wst_row = final_wst[final_wst["dataset"].astype(str) == "test"].iloc[0]
    comparison = pd.DataFrame(
        [
            comparison_row(
                "M01",
                "MFCC117 + PCA128 + LR",
                float(mfcc_package["threshold"]),
                mfcc_metrics,
                MFCC_MODEL_PATH.stat().st_size / 1024.0,
            ),
            comparison_row(
                "L01",
                "Log-Mel + tiny CNN",
                float(fusion_config["cnn_threshold"]),
                cnn_metrics,
                LOGMEL_MODEL_PATH.stat().st_size / 1024.0,
            ),
            comparison_row(
                "W15",
                "WST recording + PCA128 + LR",
                float(final_wst_row["threshold"]),
                {
                    key: final_wst_row[key]
                    for key in (
                        "accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                        "dry_precision",
                        "wet_precision",
                        "dry_recall",
                        "wet_recall",
                        "roc_auc",
                        "average_precision_wet",
                        "tn_dry_correct",
                        "fp_dry_as_wet",
                        "fn_wet_as_dry",
                        "tp_wet_correct",
                    )
                },
                float(final_wst_row["model_size_kb_joblib"]),
            ),
            comparison_row(
                "F05",
                "Late fusion WST-LR + Log-Mel CNN",
                float(fusion_config["threshold"]),
                fusion_metrics,
                float(fusion_config["combined_model_size_kb"]),
            ),
        ]
    ).sort_values("macro_f1", ascending=False)
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "evaluation": "frozen_stage2_test_comparison",
                "training_repeated": False,
                "hyperparameters_selected_on_test": False,
                "thresholds_selected_on_test": False,
                "late_fusion_policy": FUSION_POLICY,
                "late_fusion_alpha_wst": fusion_config["alpha_wst"],
                "late_fusion_weight_cnn": fusion_config["weight_cnn"],
                "test_recordings": len(mfcc_predictions),
                "test_dry": int((mfcc_predictions["y_true"] == 0).sum()),
                "test_wet": int((mfcc_predictions["y_true"] == 1).sum()),
            }
        ]
    ).to_csv(
        FUSION_RESULT_DIR / "test_evaluation_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 78)
    print("COMPARACIÓN FINAL STAGE 2 — TEST")
    print("=" * 78)
    for _, row in comparison.iterrows():
        print(
            f"{row['ID']} {row['experiment']}: "
            f"macro-F1={row['macro_f1']:.4f} | "
            f"bal-acc={row['balanced_accuracy']:.4f} | "
            f"recall dry/wet={row['dry_recall']:.4f}/{row['wet_recall']:.4f} | "
            f"AUC={row['roc_auc']:.4f}"
        )
    print(f"\nComparación: {COMPARISON_PATH}")
    print("No se ha reentrenado ni seleccionado nada con TEST.")


def print_check(test_manifest: pd.DataFrame) -> None:
    ensure_frozen_inputs()
    print("=" * 78)
    print("CHECK — EVALUACIÓN STAGE 2 TEST CON MODELOS CONGELADOS")
    print("=" * 78)
    print(
        f"Eventos TEST: {len(test_manifest)} | "
        f"grabaciones: {test_manifest['original_uuid'].nunique()}"
    )
    counts = (
        test_manifest.drop_duplicates("original_uuid")["stage2_target"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )
    print(f"Grabaciones dry/wet: {int(counts[0])}/{int(counts[1])}")
    for name, paths in required_extraction_files().items():
        complete = all(path.is_file() for path in paths)
        print(f"Features TEST {name}: {'listas' if complete else 'pendientes'}")
    print("Modelos y parámetros: congelados antes de TEST.")
    print("No se realizará búsqueda de hiperparámetros ni de umbral.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evalúa MFCC y late fusion congelados en Stage 2 TEST."
    )
    parser.add_argument(
        "--action", choices=["check", "extract", "evaluate", "all"], default="check"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reextrae las features MFCC y Log-Mel de TEST aunque ya existan.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    test_manifest = load_event_manifest("test")
    validate_test_manifest(test_manifest)
    print_check(test_manifest)
    if args.action == "check":
        return
    if args.action in {"extract", "all"}:
        extract_test_features(test_manifest, args.overwrite)
    if args.action in {"evaluate", "all"}:
        evaluate_test()


if __name__ == "__main__":
    main()
