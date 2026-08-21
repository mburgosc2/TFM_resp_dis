"""Guarda el cocleograma gammatone completo de cada evento de Stage 2.

Este es un experimento independiente del descriptor por bloques. Reutiliza
exactamente la segmentacion final y el mismo calculo gammatone ``paper64``,
pero conserva los 64 x 64 valores (4096) para aplicar PCA posteriormente
dentro de cada fold de entrenamiento.

Solo procesa TRAIN y VALIDATION. TEST no se lee ni se procesa.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import feature_extraction_stage2_cochleograms as base


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = (
    SCRIPT_DIR / "features_extracted_stage2_dry_wet_cochleograms_raw"
)
SPLITS = ("train", "validation")


def raw_feature_names(config: base.CochleogramConfig) -> list[str]:
    return [
        f"coch_filter_{filter_index:03d}_frame_{frame_index:03d}"
        for filter_index in range(config.n_filters)
        for frame_index in range(config.n_time_frames)
    ]


def event_metadata_row(
    row: pd.Series,
    feature_row: int,
    auxiliary_metadata: dict[str, float],
) -> dict[str, object]:
    return {
        "feature_row": feature_row,
        "event_id": row["event_id"],
        "event_index": int(row["event_index"]),
        "original_uuid": row["original_uuid"],
        "uuid_segmento": row["uuid_segmento"],
        "cough_type": row["cough_type"],
        "cough_type_consensus": row["cough_type_consensus"],
        "stage2_target": int(row["stage2_target"]),
        "fold": int(row["fold"]),
        "split": row["split"],
        "event_duration": float(row["event_duration"]),
        "window_observed_duration": float(
            row["window_observed_duration"]
        ),
        "window_total_padding": float(row["window_total_padding"]),
        "non_overlap_adjusted": bool(row["non_overlap_adjusted"]),
        "fallback_no_event": bool(row["fallback_no_event"]),
        **auxiliary_metadata,
    }


def extract_and_save_split(
    split_name: str,
    manifest: pd.DataFrame,
    config: base.CochleogramConfig,
    filter_coefficients: list[tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> tuple[int, int]:
    feature_dimension = config.n_filters * config.n_time_frames
    matrix = np.empty(
        (len(manifest), feature_dimension),
        dtype=np.float32,
    )
    metadata_rows = []
    errors = []

    for output_row, (_, row) in enumerate(
        tqdm(
            manifest.iterrows(),
            total=len(manifest),
            desc=f"Cocleogramas raw {split_name}",
        )
    ):
        try:
            waveform, observed = base.load_event_waveform(row, config)
            cochleogram, _ = base.compute_cochleogram(
                waveform,
                filter_coefficients,
                config,
            )
            flattened = cochleogram.reshape(-1, order="C")
            if flattened.shape != (feature_dimension,):
                raise RuntimeError(
                    f"Forma raw inesperada: {flattened.shape}"
                )
            if not np.isfinite(flattened).all():
                raise RuntimeError("El vector raw contiene NaN o infinito.")

            matrix[output_row] = flattened
            auxiliary_metadata = base.compute_auxiliary_event_metadata(
                observed,
                row,
                config,
            )
            metadata_rows.append(
                event_metadata_row(
                    row,
                    output_row,
                    auxiliary_metadata,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "manifest_row": output_row,
                    "event_id": row["event_id"],
                    "original_uuid": row["original_uuid"],
                    "error": str(exc),
                }
            )

    if errors:
        error_path = output_dir / f"errors_{split_name}.csv"
        pd.DataFrame(errors).to_csv(
            error_path,
            index=False,
            encoding="utf-8-sig",
        )
        raise RuntimeError(
            f"Fallaron {len(errors)} eventos de {split_name}. "
            f"Revisa {error_path}. No se han guardado matrices parciales."
        )

    metadata = pd.DataFrame(metadata_rows)
    event_counts = metadata.groupby("original_uuid")["event_id"].transform(
        "count"
    )
    metadata["recording_weight"] = 1.0 / event_counts.astype(float)

    if matrix.shape != (len(metadata), feature_dimension):
        raise RuntimeError("La matriz raw y sus metadatos no coinciden.")
    if not np.allclose(
        metadata.groupby("original_uuid")["recording_weight"].sum(),
        1.0,
        atol=1e-9,
    ):
        raise RuntimeError("Los pesos por grabacion no suman uno.")

    np.save(output_dir / f"X_events_{split_name}.npy", matrix)
    np.save(
        output_dir / f"y_events_{split_name}.npy",
        metadata["stage2_target"].to_numpy(np.int64),
    )
    np.save(
        output_dir / f"folds_events_{split_name}.npy",
        metadata["fold"].to_numpy(np.int32),
    )
    metadata.to_csv(
        output_dir / f"metadata_events_features_{split_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return matrix.shape


def save_configuration(
    config: base.CochleogramConfig,
    center_frequencies: np.ndarray,
    output_dir: Path,
) -> None:
    feature_names = raw_feature_names(config)
    configuration = {
        **asdict(config),
        "target_samples": config.target_samples,
        "frame_length": config.frame_length,
        "hop_length": config.hop_length,
        "frame_overlap_fraction": 0.5,
        "dynamic_range_db": base.DYNAMIC_RANGE_DB,
        "raw_cochleogram_shape": (
            f"{config.n_filters}x{config.n_time_frames}"
        ),
        "raw_feature_dimension": len(feature_names),
        "flatten_order": "C_filter_then_time",
        "stored_dtype": "float32",
        "normalization": "relative_db_clipped_minus80_0_scaled_0_1",
        "source_experiment": "paper64_full_cochleogram_before_blocks",
    }
    pd.DataFrame([configuration]).to_csv(
        output_dir / "raw_cochleogram_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names)),
            "feature_name": feature_names,
        }
    ).to_csv(
        output_dir / "raw_feature_names.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "filter_index": np.arange(config.n_filters),
            "center_frequency_hz": center_frequencies,
        }
    ).to_csv(
        output_dir / "gammatone_center_frequencies.csv",
        index=False,
        encoding="utf-8-sig",
    )


def print_check(
    config: base.CochleogramConfig,
    manifests: dict[str, pd.DataFrame],
) -> None:
    dimension = config.n_filters * config.n_time_frames
    print("=" * 76)
    print("CHECK — COCLEOGRAMAS COMPLETOS PARA PCA")
    print("=" * 76)
    print(f"Preset: {config.name}")
    print(
        f"Cada evento: {config.n_filters} x {config.n_time_frames} "
        f"= {dimension} valores float32"
    )
    for split_name, manifest in manifests.items():
        estimated_mb = len(manifest) * dimension * 4 / (1024**2)
        print(
            f"{split_name}: {len(manifest)} eventos, "
            f"matriz estimada {estimated_mb:.2f} MiB"
        )
    print("Se reutilizan exactamente los eventos finales sin solapamiento.")
    print("TEST no sera leido ni procesado.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrae cocleogramas 64x64 completos para PCA."
    )
    parser.add_argument(
        "--action",
        choices=["check", "extract"],
        default="check",
        help="check valida manifests; extract calcula train/validation.",
    )
    parser.add_argument(
        "--preset",
        choices=["paper64"],
        default="paper64",
        help="Este experimento replica el frente gammatone 64x64.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = base.PRESETS[args.preset]
    manifests = {
        split_name: base.load_event_manifest(split_name)
        for split_name in SPLITS
    }
    base.validate_train_validation_disjoint(
        manifests["train"],
        manifests["validation"],
    )
    print_check(config, manifests)

    if args.action == "check":
        print("Comprobacion completada. No se extrajeron features.")
        return

    output_dir = OUTPUT_ROOT / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    center_frequencies, filter_coefficients = (
        base.build_gammatone_filterbank(config)
    )
    save_configuration(config, center_frequencies, output_dir)

    for split_name in SPLITS:
        shape = extract_and_save_split(
            split_name,
            manifests[split_name],
            config,
            filter_coefficients,
            output_dir,
        )
        print(f"{split_name}: matriz guardada {shape}")

    print("=" * 76)
    print("EXTRACCION RAW COMPLETADA")
    print("=" * 76)
    print(f"Resultados: {output_dir}")
    print("TEST permanece reservado.")


if __name__ == "__main__":
    main()
