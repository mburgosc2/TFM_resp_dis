"""Prepara TEST para la evaluacion final del WST recording + LR congelado.

Este script es deliberadamente independiente de los scripts originales de
TRAIN/VALIDATION. Aplica a TEST la misma segmentacion y la misma extraccion
Wavelet Scattering, sin ajustar modelos, PCA, hiperparametros ni umbrales.

Acciones:

* ``check``: valida metadatos, configuraciones y separacion por UUID.
* ``segment``: genera el manifest de eventos de TEST.
* ``extract``: extrae los 644 coeficientes WST por evento de TEST.
* ``all``: ejecuta segmentacion y extraccion en ese orden.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import feature_extraction_stage2_cochleograms as coch_base
import feature_extraction_stage2_wavelet_scattering as wst
import segment_stage2_dry_wet_events as segmentation


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_METADATA_NAME = "metadata_test_stage2_dry_wet.csv"
TEST_EVENT_NAME = "metadata_events_test_stage2_dry_wet.csv"
DEFAULT_PRESET = "paper_q8_q1_t500_full"

SEGMENT_OUTPUTS = (
    "metadata_events_test_stage2_dry_wet.csv",
    "metadata_segment_summary_test_stage2_dry_wet.csv",
    "metadata_original_summary_test_stage2_dry_wet.csv",
    "metadata_errors_test_stage2_dry_wet.csv",
)
FEATURE_OUTPUTS = (
    "X_events_test.npy",
    "y_events_test.npy",
    "folds_events_test.npy",
    "metadata_events_features_test.csv",
)


def register_test_files() -> None:
    """Registra TEST solo en este proceso, sin editar los scripts base."""

    segmentation.INPUT_FILES["test"] = TEST_METADATA_NAME
    coch_base.EVENT_FILES["test"] = TEST_EVENT_NAME


def load_split_metadata() -> dict[str, pd.DataFrame]:
    frames = {
        split: segmentation.load_stage2_metadata(split)
        for split in ("train", "validation", "test")
    }
    validate_pairwise_disjoint(
        {
            split: frame["original_uuid"].astype(str)
            for split, frame in frames.items()
        }
    )
    return frames


def validate_pairwise_disjoint(split_uuids: dict[str, pd.Series]) -> None:
    names = list(split_uuids)
    uuid_sets = {name: set(values) for name, values in split_uuids.items()}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = uuid_sets[left_name] & uuid_sets[right_name]
            if overlap:
                raise ValueError(
                    f"{left_name} y {right_name} comparten original_uuid: "
                    f"{sorted(overlap)[:10]}"
                )


def refuse_existing(paths: list[Path], overwrite: bool, stage: str) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            f"Ya existen salidas de {stage}:\n{formatted}\n"
            "Usa --overwrite solo si deseas repetir exactamente el proceso."
        )


def validate_segmentation_configuration() -> None:
    path = segmentation.OUTPUT_DIR / "segmentation_configuration_stage2.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "No existe la configuracion usada para TRAIN/VALIDATION: "
            f"{path}"
        )
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError("La configuracion de segmentacion debe tener una fila.")
    row = frame.iloc[0]
    expected = {
        "segmentation_config": segmentation.SEGMENTATION_CONFIG.name,
        "segmentation_method": segmentation.FINAL_SEGMENTATION_METHOD,
        "enforce_non_overlap": True,
        **asdict(segmentation.SEGMENTATION_CONFIG),
    }
    for key, expected_value in expected.items():
        if key not in row.index:
            raise ValueError(f"Falta {key} en {path.name}.")
        actual = row[key]
        if isinstance(expected_value, bool):
            normalized = str(actual).strip().lower() in {"true", "1", "yes"}
            if normalized != expected_value:
                raise ValueError(f"Configuracion distinta en {key}: {actual}.")
        elif isinstance(expected_value, (int, float)):
            if not np.isclose(float(actual), float(expected_value)):
                raise ValueError(f"Configuracion distinta en {key}: {actual}.")
        elif str(actual) != str(expected_value):
            raise ValueError(f"Configuracion distinta en {key}: {actual}.")


def segment_test(frames: dict[str, pd.DataFrame], overwrite: bool) -> None:
    validate_segmentation_configuration()
    output_paths = [segmentation.OUTPUT_DIR / name for name in SEGMENT_OUTPUTS]
    refuse_existing(output_paths, overwrite, "segmentacion de TEST")

    print("\n" + "=" * 78)
    print("SEGMENTACION FINAL DE TEST - STAGE 2")
    print("=" * 78)
    print(f"Configuracion congelada: {segmentation.SEGMENTATION_CONFIG.name}")
    print(f"Metodo congelado: {segmentation.FINAL_SEGMENTATION_METHOD}")
    segmentation.process_split("test", frames["test"])


def load_event_manifests() -> dict[str, pd.DataFrame]:
    manifests = {
        split: coch_base.load_event_manifest(split)
        for split in ("train", "validation", "test")
    }
    validate_pairwise_disjoint(
        {
            split: frame["original_uuid"].astype(str)
            for split, frame in manifests.items()
        }
    )
    return manifests


def validate_wst_training_configuration(
    config: wst.ScatteringConfig,
    layout: dict[str, object],
    feature_layout: pd.DataFrame,
) -> Path:
    output_dir = wst.OUTPUT_ROOT / config.name
    configuration_path = output_dir / "wavelet_scattering_configuration.csv"
    feature_layout_path = output_dir / "wavelet_scattering_feature_layout.csv"
    if not configuration_path.is_file() or not feature_layout_path.is_file():
        raise FileNotFoundError(
            "Faltan la configuracion o el layout WST de TRAIN/VALIDATION en "
            f"{output_dir}."
        )

    stored_configuration = pd.read_csv(configuration_path)
    if len(stored_configuration) != 1:
        raise ValueError("La configuracion WST guardada debe tener una fila.")
    row = stored_configuration.iloc[0]
    expected_values = {
        "name": config.name,
        "sample_rate": config.sample_rate,
        "target_samples": config.target_samples,
        "invariance_samples": config.invariance_samples,
        "J": config.j,
        "Q": str(config.q),
        "max_order": config.max_order,
        "normalize_waveform_peak": config.normalize_waveform_peak,
        "temporal_aggregation": "mean_over_scattering_time_positions",
        "dimensionality_reduction": "none",
        "feature_dimension": int(layout["feature_dimension"]),
    }
    for key, expected in expected_values.items():
        if key not in row.index:
            raise ValueError(f"Falta {key} en {configuration_path.name}.")
        actual = row[key]
        if isinstance(expected, bool):
            normalized = str(actual).strip().lower() in {"true", "1", "yes"}
            if normalized != expected:
                raise ValueError(f"Configuracion WST distinta en {key}: {actual}.")
        elif isinstance(expected, (int, float)):
            if not np.isclose(float(actual), float(expected)):
                raise ValueError(f"Configuracion WST distinta en {key}: {actual}.")
        elif str(actual) != str(expected):
            raise ValueError(f"Configuracion WST distinta en {key}: {actual}.")

    stored_layout = pd.read_csv(feature_layout_path)
    comparison_columns = (
        "feature_index",
        "feature_name",
        "scattering_order",
        "path_index",
        "temporal_pooling",
    )
    if len(stored_layout) != len(feature_layout):
        raise ValueError("El numero de caminos WST no coincide con TRAIN.")
    for column in comparison_columns:
        if not stored_layout[column].astype(str).equals(
            feature_layout[column].astype(str)
        ):
            raise ValueError(f"El layout WST de TEST difiere en {column}.")
    return output_dir


def extract_test(preset: str, batch_size: int, overwrite: bool) -> None:
    manifests = load_event_manifests()
    config = wst.PRESETS[preset]
    scattering = wst.build_scattering(config)
    path_count, _, _, layout = wst.infer_layout(scattering, config)
    feature_layout = wst.feature_layout_dataframe(scattering, path_count)
    output_dir = validate_wst_training_configuration(
        config, layout, feature_layout
    )
    output_paths = [output_dir / name for name in FEATURE_OUTPUTS]
    refuse_existing(output_paths, overwrite, "extraccion WST de TEST")

    print("\n" + "=" * 78)
    print("EXTRACCION WST FINAL DE TEST - STAGE 2")
    print("=" * 78)
    print(f"Preset congelado: {config.name}")
    print(f"Eventos TEST: {len(manifests['test'])}")
    print(f"Features por evento: {layout['feature_dimension']}")
    shape = wst.extract_and_save_split(
        "test",
        manifests["test"],
        config,
        scattering,
        int(layout["feature_dimension"]),
        batch_size,
        output_dir,
    )
    pd.DataFrame(
        [
            {
                "split": "test",
                "preset": config.name,
                "event_count": shape[0],
                "feature_dimension": shape[1],
                "segmentation_config": segmentation.SEGMENTATION_CONFIG.name,
                "segmentation_method": segmentation.FINAL_SEGMENTATION_METHOD,
                "same_configuration_as_train_validation": True,
                "model_or_threshold_fitted": False,
            }
        ]
    ).to_csv(
        output_dir / "wavelet_scattering_test_extraction_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Matriz TEST guardada: {shape}")
    print(f"Resultados: {output_dir}")


def print_check(frames: dict[str, pd.DataFrame], preset: str) -> None:
    validate_segmentation_configuration()
    config = wst.PRESETS[preset]
    scattering = wst.build_scattering(config)
    path_count, _, _, layout = wst.infer_layout(scattering, config)
    feature_layout = wst.feature_layout_dataframe(scattering, path_count)
    validate_wst_training_configuration(config, layout, feature_layout)

    print("=" * 78)
    print("CHECK - PREPARACION FINAL DE TEST WST + LR")
    print("=" * 78)
    for split, frame in frames.items():
        print(
            f"{split.upper()}: {frame['original_uuid'].nunique()} grabaciones | "
            f"dry/wet={frame['cough_type'].value_counts().to_dict()}"
        )
    print(f"Segmentacion: {segmentation.SEGMENTATION_CONFIG.name}")
    print(f"WST: {config.name}; {layout['feature_dimension']} features por evento")
    print("UUID disjuntos y configuraciones compatibles.")
    print("No se ajustara ningun modelo, PCA, peso ni umbral.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepara TEST para WST recording + LR congelado."
    )
    parser.add_argument(
        "--action",
        choices=("check", "segment", "extract", "all"),
        default="check",
    )
    parser.add_argument(
        "--preset", choices=sorted(wst.PRESETS), default=DEFAULT_PRESET
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite repetir TEST sobrescribiendo solo sus salidas.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size debe ser mayor que cero.")
    register_test_files()
    frames = load_split_metadata()
    print_check(frames, args.preset)
    if args.action == "check":
        print("Comprobacion completada; TEST no se ha transformado.")
        return
    if args.action in {"segment", "all"}:
        segment_test(frames, args.overwrite)
    if args.action in {"extract", "all"}:
        extract_test(args.preset, args.batch_size, args.overwrite)


if __name__ == "__main__":
    main()
