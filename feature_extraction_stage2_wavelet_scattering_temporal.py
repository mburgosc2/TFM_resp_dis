"""Extrae Wavelet Scattering temporal para Stage 2 dry/wet.

Reutiliza exactamente la configuracion y los eventos del extractor WST
original, pero NO promedia el eje temporal. Cada evento se guarda con la forma
nativa ``(644 caminos, 5 posiciones temporales)`` para permitir que una red
ligera aprenda la evolucion temporal de los coeficientes.

Los resultados se escriben en una raiz nueva y no sobrescriben las features
WST de 644 valores ya utilizadas por RF, LR, SVM y MLP. Solo procesa TRAIN y
VALIDATION. TEST no se lee ni se procesa.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from kymatio.numpy import Scattering1D
from tqdm import tqdm

import feature_extraction_stage2_cochleograms as base
import feature_extraction_stage2_wavelet_scattering as pooled


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = (
    SCRIPT_DIR
    / "features_extracted_stage2_dry_wet_wavelet_scattering_temporal"
)
POOLED_BASELINE_ROOT = (
    SCRIPT_DIR / "features_extracted_stage2_dry_wet_wavelet_scattering"
)
SPLITS = ("train", "validation")
PRESETS = pooled.PRESETS


def temporal_path_layout_dataframe(
    scattering: Scattering1D,
    path_count: int,
) -> pd.DataFrame:
    """Una fila por camino; el eje temporal permanece separado."""

    meta = scattering.meta()
    rows: list[dict[str, object]] = []
    for path_index in range(path_count):
        order = int(meta["order"][path_index])
        rows.append(
            {
                "path_index": path_index,
                "path_name": f"wst_order{order}_path{path_index:04d}",
                "scattering_order": order,
                "temporal_pooling": "none",
                "j_1": pooled.meta_value(meta["j"], path_index, 0),
                "j_2": pooled.meta_value(meta["j"], path_index, 1),
                "xi_1": pooled.meta_value(meta["xi"], path_index, 0),
                "xi_2": pooled.meta_value(meta["xi"], path_index, 1),
            }
        )
    return pd.DataFrame(rows)


def temporal_position_layout_dataframe(time_count: int) -> pd.DataFrame:
    """Documenta el indice del eje temporal sin inventar centros en segundos."""

    return pd.DataFrame(
        [
            {
                "time_position_index": position,
                "axis_in_saved_tensor": 2,
                "interpretation": "downsampled_scattering_time_position",
                "exact_center_seconds_available": False,
            }
            for position in range(time_count)
        ]
    )


def process_batch(
    scattering: Scattering1D,
    waveforms: list[np.ndarray],
    path_count: int,
    time_count: int,
) -> tuple[np.ndarray, float]:
    tensor = np.stack(waveforms).astype(np.float32, copy=False)
    started = time.perf_counter()
    coefficients = scattering(tensor)
    elapsed = time.perf_counter() - started
    coefficients = np.asarray(coefficients, dtype=np.float32)
    expected_shape = (len(waveforms), path_count, time_count)
    if coefficients.shape != expected_shape:
        raise RuntimeError(
            f"Forma WST temporal inesperada: {coefficients.shape}; "
            f"esperada {expected_shape}."
        )
    if not np.isfinite(coefficients).all():
        raise RuntimeError("Wavelet Scattering temporal produjo NaN o Inf.")
    return coefficients, elapsed


def validate_against_pooled_baseline(
    split_name: str,
    temporal_tensor: np.ndarray,
    config: pooled.ScatteringConfig,
) -> dict[str, object]:
    """Verifica que mean(time) reproduce el experimento WST anterior."""

    baseline_path = (
        POOLED_BASELINE_ROOT / config.name / f"X_events_{split_name}.npy"
    )
    comparison: dict[str, object] = {
        "split": split_name,
        "baseline_path": str(baseline_path),
        "baseline_available": baseline_path.is_file(),
        "shape_matches": np.nan,
        "allclose": np.nan,
        "max_absolute_difference": np.nan,
        "mean_absolute_difference": np.nan,
        "rtol": 1e-5,
        "atol": 1e-7,
    }
    if not baseline_path.is_file():
        return comparison

    baseline = np.load(baseline_path, mmap_mode="r")
    temporal_mean = temporal_tensor.mean(axis=2, dtype=np.float32)
    comparison["shape_matches"] = baseline.shape == temporal_mean.shape
    if baseline.shape != temporal_mean.shape:
        comparison["allclose"] = False
        raise RuntimeError(
            f"La media temporal {temporal_mean.shape} no coincide con el "
            f"baseline WST {baseline.shape} de {split_name}."
        )

    differences = np.abs(
        np.asarray(baseline, dtype=np.float32) - temporal_mean
    )
    comparison["max_absolute_difference"] = float(np.max(differences))
    comparison["mean_absolute_difference"] = float(np.mean(differences))
    equivalent = bool(
        np.allclose(
            np.asarray(baseline),
            temporal_mean,
            rtol=float(comparison["rtol"]),
            atol=float(comparison["atol"]),
        )
    )
    comparison["allclose"] = equivalent
    if not equivalent:
        raise RuntimeError(
            "La media de las features temporales no reproduce el baseline "
            f"WST existente para {split_name}."
        )
    return comparison


def extract_and_save_split(
    split_name: str,
    manifest: pd.DataFrame,
    config: pooled.ScatteringConfig,
    scattering: Scattering1D,
    path_count: int,
    time_count: int,
    batch_size: int,
    output_dir: Path,
) -> tuple[tuple[int, ...], dict[str, object]]:
    tensor = np.empty(
        (len(manifest), path_count, time_count),
        dtype=np.float32,
    )
    metadata_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    pending_waveforms: list[np.ndarray] = []
    pending_rows: list[tuple[int, pd.Series, float]] = []

    def flush_batch() -> None:
        if not pending_waveforms:
            return
        coefficients, elapsed = process_batch(
            scattering,
            pending_waveforms,
            path_count,
            time_count,
        )
        seconds_per_event = elapsed / len(pending_waveforms)
        for batch_index, (output_row, row, peak) in enumerate(pending_rows):
            tensor[output_row] = coefficients[batch_index]
            metadata_rows.append(
                pooled.event_metadata_row(
                    row,
                    output_row,
                    peak,
                    seconds_per_event,
                )
            )
        pending_waveforms.clear()
        pending_rows.clear()

    for output_row, (_, row) in enumerate(
        tqdm(
            manifest.iterrows(),
            total=len(manifest),
            desc=f"Cargando WST temporal {split_name}",
        )
    ):
        try:
            waveform, _ = base.load_event_waveform(
                row,
                base.PRESETS["paper64"],
            )
            waveform, peak = pooled.normalize_waveform(
                waveform,
                config.normalize_waveform_peak,
            )
            pending_waveforms.append(waveform)
            pending_rows.append((output_row, row, peak))
            if len(pending_waveforms) >= batch_size:
                flush_batch()
        except Exception as exc:
            errors.append(
                {
                    "manifest_row": output_row,
                    "event_id": row.get("event_id", ""),
                    "original_uuid": row.get("original_uuid", ""),
                    "error": str(exc),
                }
            )
    flush_batch()

    if errors:
        error_path = output_dir / f"errors_{split_name}.csv"
        pd.DataFrame(errors).to_csv(
            error_path,
            index=False,
            encoding="utf-8-sig",
        )
        raise RuntimeError(
            f"Fallaron {len(errors)} eventos de {split_name}. "
            f"Revisa {error_path}. No se guardaron matrices parciales."
        )

    metadata = (
        pd.DataFrame(metadata_rows)
        .sort_values("feature_row")
        .reset_index(drop=True)
    )
    event_counts = metadata.groupby("original_uuid")["event_id"].transform(
        "count"
    )
    metadata["recording_weight"] = 1.0 / event_counts.astype(float)
    metadata["wst_path_count"] = path_count
    metadata["wst_time_position_count"] = time_count
    metadata["stored_axis_order"] = "event,path,time"

    if metadata["feature_row"].tolist() != list(range(len(metadata))):
        raise RuntimeError("Los metadatos WST temporales no quedaron alineados.")
    expected_shape = (len(metadata), path_count, time_count)
    if tensor.shape != expected_shape:
        raise RuntimeError(
            f"Tensor y metadatos no coinciden: {tensor.shape} != {expected_shape}."
        )
    if not np.allclose(
        metadata.groupby("original_uuid")["recording_weight"].sum(),
        1.0,
        atol=1e-9,
    ):
        raise RuntimeError("Los pesos por grabacion no suman uno.")

    comparison = validate_against_pooled_baseline(
        split_name,
        tensor,
        config,
    )
    np.save(output_dir / f"X_events_{split_name}.npy", tensor)
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
    return tensor.shape, comparison


def save_configuration(
    config: pooled.ScatteringConfig,
    layout: dict[str, object],
    path_layout: pd.DataFrame,
    time_layout: pd.DataFrame,
    output_dir: Path,
    batch_size: int,
) -> None:
    configuration = {
        **asdict(config),
        "target_samples": config.target_samples,
        "invariance_samples": config.invariance_samples,
        "J": config.j,
        "Q": str(config.q),
        "backend": "kymatio.numpy",
        "batch_size": batch_size,
        "coefficient_transform": "none_raw_nonnegative_scattering",
        "temporal_aggregation": "none",
        "dimensionality_reduction": "none",
        "stored_axis_order": "event,path,time",
        "cnn_axis_order_recommended": "event,time,path",
        "cnn_transpose": "np.transpose(X, (0, 2, 1))",
        "stored_dtype": "float32",
        "test_processed": False,
        **layout,
    }
    configuration["feature_dimension"] = int(layout["raw_scattering_value_count"])
    configuration["tensor_shape_per_event"] = (
        f"{layout['path_count']}x{layout['time_position_count']}"
    )
    pd.DataFrame([configuration]).to_csv(
        output_dir / "wavelet_scattering_temporal_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    path_layout.to_csv(
        output_dir / "wavelet_scattering_temporal_path_layout.csv",
        index=False,
        encoding="utf-8-sig",
    )
    time_layout.to_csv(
        output_dir / "wavelet_scattering_temporal_position_layout.csv",
        index=False,
        encoding="utf-8-sig",
    )


def print_check(
    config: pooled.ScatteringConfig,
    manifests: dict[str, pd.DataFrame],
    layout: dict[str, object],
    batch_size: int,
) -> None:
    path_count = int(layout["path_count"])
    time_count = int(layout["time_position_count"])
    values_per_event = path_count * time_count
    print("=" * 78)
    print("CHECK - WST TEMPORAL SIN POOLING - STAGE 2")
    print("=" * 78)
    print(f"Preset WST reutilizado: {config.name}")
    print(f"Audio: {config.target_samples} muestras a {config.sample_rate} Hz")
    print(f"Q={config.q}; J={config.j}; T={config.invariance_samples}")
    print(
        f"Caminos orden 0/1/2: {layout['order_0_paths']} / "
        f"{layout['order_1_paths']} / {layout['order_2_paths']}"
    )
    print(f"Tensor guardado por evento: ({path_count}, {time_count})")
    print(f"Valores conservados por evento: {values_per_event}")
    print("No se aplica media temporal, PCA ni seleccion de features.")
    print("Para Conv1D se transpondra en memoria a (evento, tiempo, camino).")
    print(
        "Las 5 posiciones son frames temporales internos de scattering; "
        "no se interpretan como cinco ventanas independientes de 0,5 s."
    )
    print(f"Backend: kymatio.numpy; batch_size={batch_size}")
    for split_name, manifest in manifests.items():
        estimated_mb = len(manifest) * values_per_event * 4 / (1024**2)
        print(
            f"{split_name}: {len(manifest)} eventos; "
            f"tensor estimado {estimated_mb:.2f} MiB"
        )
    print("TEST no sera leido ni procesado.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrae WST temporal 644x5 por evento para Stage 2."
    )
    parser.add_argument(
        "--action",
        choices=["check", "extract"],
        default="check",
        help="check valida; extract procesa TRAIN y VALIDATION.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="paper_q8_q1_t500_full",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size debe ser mayor que cero.")

    config = PRESETS[args.preset]
    manifests = {
        split_name: base.load_event_manifest(split_name)
        for split_name in SPLITS
    }
    base.validate_train_validation_disjoint(
        manifests["train"],
        manifests["validation"],
    )
    scattering = pooled.build_scattering(config)
    path_count, time_count, _, layout = pooled.infer_layout(
        scattering,
        config,
    )
    path_layout = temporal_path_layout_dataframe(scattering, path_count)
    time_layout = temporal_position_layout_dataframe(time_count)
    print_check(config, manifests, layout, args.batch_size)

    if args.action == "check":
        print("Comprobacion completada. No se extrajeron features.")
        return

    output_dir = OUTPUT_ROOT / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_configuration(
        config,
        layout,
        path_layout,
        time_layout,
        output_dir,
        args.batch_size,
    )
    comparison_rows = []
    for split_name in SPLITS:
        shape, comparison = extract_and_save_split(
            split_name,
            manifests[split_name],
            config,
            scattering,
            path_count,
            time_count,
            args.batch_size,
            output_dir,
        )
        comparison_rows.append(comparison)
        print(f"{split_name}: tensor guardado {shape}")
        if comparison["baseline_available"]:
            print(
                "  Equivalencia con WST promediado: "
                f"allclose={comparison['allclose']}; "
                f"max_diff={comparison['max_absolute_difference']:.3e}"
            )
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "temporal_mean_equivalence_with_pooled_wst.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 78)
    print("EXTRACCION WST TEMPORAL COMPLETADA")
    print("=" * 78)
    print(f"Resultados: {output_dir}")
    print("El extractor WST original no se ha modificado.")
    print("TEST permanece reservado.")


if __name__ == "__main__":
    main()
