"""Entrena Logistic Regression con MFCC agregados por grabacion.

Consume los 117 MFCC por evento generados por
``feature_extraction_stage2_mfcc.py`` y aplica mean/std/max entre los eventos
de cada original_uuid, obteniendo 351 variables por grabacion. StandardScaler,
PCA, C, pesos y umbral se ajustan usando exclusivamente los folds de TRAIN.
TEST no se lee ni se procesa.

El script configura el motor lineal utilizado por WST para que ambos
experimentos produzcan predicciones OOF directamente comparables.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_linear as linear
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording


ROOT = Path(__file__).resolve().parent
FEATURES_ROOT = ROOT / "features_extracted_stage2_dry_wet_mfcc"
PRESET = "mfcc117"
EXPECTED_EVENT_FEATURES = 117
EXPECTED_RECORDING_FEATURES = 351


def required_paths(input_dir: Path) -> dict[str, Path]:
    return {
        "x_train": input_dir / "X_events_train.npy",
        "y_train": input_dir / "y_events_train.npy",
        "folds_train": input_dir / "folds_events_train.npy",
        "metadata_train": input_dir / "metadata_events_features_train.csv",
        "x_validation": input_dir / "X_events_validation.npy",
        "y_validation": input_dir / "y_events_validation.npy",
        "folds_validation": input_dir / "folds_events_validation.npy",
        "metadata_validation": input_dir / "metadata_events_features_validation.csv",
        "layout": input_dir / "mfcc_event_feature_layout.csv",
        "configuration": input_dir / "mfcc_configuration.csv",
    }


def load_mfcc_data(preset: str) -> tuple[common.DataView, pd.DataFrame]:
    if preset != PRESET:
        raise ValueError(f"Preset MFCC desconocido: {preset}.")
    input_dir = FEATURES_ROOT / preset
    paths = required_paths(input_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan features MFCC. Ejecuta primero la extraccion:\n"
            + "\n".join(missing)
        )

    x_train = np.load(paths["x_train"])
    y_train = np.load(paths["y_train"])
    folds_train = np.load(paths["folds_train"])
    metadata_train = pd.read_csv(paths["metadata_train"])
    x_validation = np.load(paths["x_validation"])
    y_validation = np.load(paths["y_validation"])
    folds_validation = np.load(paths["folds_validation"])
    metadata_validation = pd.read_csv(paths["metadata_validation"])
    layout = pd.read_csv(paths["layout"])
    configuration = pd.read_csv(paths["configuration"])

    common.validate_arrays_and_metadata(
        "event", "train", x_train, y_train, folds_train, metadata_train
    )
    common.validate_arrays_and_metadata(
        "event",
        "validation",
        x_validation,
        y_validation,
        folds_validation,
        metadata_validation,
    )
    if x_train.shape[1] != EXPECTED_EVENT_FEATURES:
        raise ValueError(f"Se esperaban 117 MFCC/evento: {x_train.shape}.")
    if x_validation.shape[1] != EXPECTED_EVENT_FEATURES:
        raise ValueError("TRAIN y VALIDATION tienen distinta dimension MFCC.")
    if len(layout) != EXPECTED_EVENT_FEATURES:
        raise ValueError("El layout MFCC no contiene 117 variables.")
    if layout["feature_index"].tolist() != list(range(EXPECTED_EVENT_FEATURES)):
        raise ValueError("feature_index MFCC no es consecutivo.")
    if len(configuration) != 1:
        raise ValueError("La configuracion MFCC debe contener una fila.")
    config = configuration.iloc[0]
    expected_config = {
        "event_feature_count": EXPECTED_EVENT_FEATURES,
        "recording_feature_count": EXPECTED_RECORDING_FEATURES,
        "frame_pooling": "mean|std|max",
        "recording_pooling": "mean_std_max",
        "test_processed": False,
    }
    observed_config = {
        "event_feature_count": int(config["event_feature_count"]),
        "recording_feature_count": int(config["recording_feature_count"]),
        "frame_pooling": str(config["frame_pooling"]),
        "recording_pooling": str(config["recording_pooling"]),
        "test_processed": bool(config["test_processed"]),
    }
    if observed_config != expected_config:
        raise ValueError(
            f"Configuracion MFCC inesperada: {observed_config}."
        )

    train_uuids = set(metadata_train["original_uuid"].astype(str))
    validation_uuids = set(metadata_validation["original_uuid"].astype(str))
    overlap = train_uuids & validation_uuids
    if overlap:
        raise ValueError(
            "TRAIN y VALIDATION MFCC comparten UUID: "
            f"{sorted(overlap)[:10]}"
        )

    return (
        common.DataView(
            experiment="mfcc117_event",
            x_train=np.asarray(x_train, dtype=np.float32),
            train_metadata=metadata_train,
            x_validation=np.asarray(x_validation, dtype=np.float32),
            validation_metadata=metadata_validation,
            feature_names=layout["feature_name"].astype(str).tolist(),
        ),
        configuration,
    )


def build_recording_data(event_data: common.DataView) -> common.DataView:
    data = recording.build_recording_view(event_data, linear.RECORDING_POOLING)
    if data.x_train.shape[1] != EXPECTED_RECORDING_FEATURES:
        raise ValueError(
            f"Se esperaban 351 features MFCC/grabacion: {data.x_train.shape}."
        )
    return data


linear.RESULTS_ROOT = ROOT / "results_stage2_dry_wet_mfcc_recording_linear"
linear.MODELS_ROOT = ROOT / "models_stage2_dry_wet_mfcc_recording_linear"
linear.GRAPHS_ROOT = (
    ROOT
    / "graphs_results_stage2_dry_wet_random"
    / "training_mfcc_recording_linear"
)
linear.PRESETS = (PRESET,)
linear.DEFAULT_PRESET = PRESET
linear.EXPECTED_RECORDING_FEATURE_COUNT = EXPECTED_RECORDING_FEATURES
linear.EXPERIMENT_KEY = "mfcc117_recording_logistic_regression"
linear.DISPLAY_TITLE = "STAGE 2 — MFCC RECORDING — REGRESION LOGISTICA"
linear.CV_TITLE = "CV — MFCC RECORDING + REGRESION LOGISTICA"
linear.RESULT_TITLE = "RESULTADO MFCC RECORDING + REGRESION LOGISTICA"
linear.CHECK_TITLE = "CHECK — MFCC RECORDING + REGRESION LOGISTICA"
linear.PARSER_DESCRIPTION = "MFCC recording mean+std+max con Logistic Regression."
linear.FEATURE_LOADER = load_mfcc_data
linear.build_recording_data = build_recording_data
linear.FULL_PCA_OPTIONS = (None, 32, 64, 128)
linear.QUICK_PCA_OPTIONS = (None, 32, 64)
linear.FULL_C_VALUES = {
    "logistic_regression": (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
}
linear.QUICK_C_VALUES = {
    "logistic_regression": (0.001, 0.01, 0.1),
}


if __name__ == "__main__":
    linear.main()
