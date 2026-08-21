"""Extrae Wavelet Scattering completo para Stage 2 dry/wet.

Este experimento reutiliza exactamente los eventos finales no solapados de
1,5 s. Aplica scattering 1-D con dos bancos de filtros (Q=(8, 1)), conserva
los caminos de orden 0, 1 y 2 y promedia sus posiciones temporales. El
resultado contiene un valor por camino, sin PCA ni seleccion de variables.

Solo procesa TRAIN y VALIDATION. TEST no se lee ni se procesa.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from kymatio.numpy import Scattering1D
from tqdm import tqdm

import feature_extraction_stage2_cochleograms as base


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "features_extracted_stage2_dry_wet_wavelet_scattering"
SPLITS = ("train", "validation")


@dataclass(frozen=True)
class ScatteringConfig:
    name: str = "paper_q8_q1_t500_full"
    sample_rate: int = 16_000
    target_duration_seconds: float = 1.5
    invariance_scale_seconds: float = 0.5
    first_order_wavelets_per_octave: int = 8
    second_order_wavelets_per_octave: int = 1
    max_order: int = 2
    normalize_waveform_peak: bool = True

    @property
    def target_samples(self) -> int:
        return int(round(self.sample_rate * self.target_duration_seconds))

    @property
    def invariance_samples(self) -> int:
        return int(round(self.sample_rate * self.invariance_scale_seconds))

    @property
    def j(self) -> int:
        # 2**13=8192 muestras, la potencia de dos mas cercana a 0,5 s
        # para 16 kHz. T conserva la escala solicitada exacta de 8000.
        return int(round(math.log2(self.invariance_samples)))

    @property
    def q(self) -> tuple[int, int]:
        return (
            self.first_order_wavelets_per_octave,
            self.second_order_wavelets_per_octave,
        )


PRESETS = {
    "paper_q8_q1_t500_full": ScatteringConfig(),
}


def build_scattering(
    config: ScatteringConfig,
) -> Scattering1D:
    return Scattering1D(
        J=config.j,
        shape=config.target_samples,
        Q=config.q,
        T=config.invariance_samples,
        max_order=config.max_order,
        out_type="array",
    )


def normalize_waveform(
    waveform: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(waveform)))
    if enabled and peak > 0.0:
        waveform = waveform / peak
    return np.asarray(waveform, dtype=np.float32), peak


def infer_layout(
    scattering: Scattering1D,
    config: ScatteringConfig,
) -> tuple[int, int, np.ndarray, dict[str, object]]:
    probe = np.zeros(
        (1, config.target_samples),
        dtype=np.float32,
    )
    output = scattering(probe)
    if output.ndim != 3 or output.shape[0] != 1:
        raise RuntimeError(f"Forma scattering inesperada: {tuple(output.shape)}")

    path_count = int(output.shape[1])
    time_count = int(output.shape[2])
    meta = scattering.meta()
    orders = np.asarray(meta["order"], dtype=np.int32)
    if orders.shape != (path_count,):
        raise RuntimeError("Los metadatos de caminos no coinciden con la salida.")
    expected_orders = set(range(config.max_order + 1))
    if set(orders.tolist()) != expected_orders:
        raise RuntimeError(f"Ordenes scattering inesperados: {set(orders.tolist())}")

    layout = {
        "path_count": path_count,
        "time_position_count": time_count,
        "raw_scattering_value_count": path_count * time_count,
        "feature_dimension": path_count,
        "order_0_paths": int(np.sum(orders == 0)),
        "order_1_paths": int(np.sum(orders == 1)),
        "order_2_paths": int(np.sum(orders == 2)),
    }
    return path_count, time_count, orders, layout


def feature_layout_dataframe(
    scattering: Scattering1D,
    path_count: int,
) -> pd.DataFrame:
    meta = scattering.meta()
    rows: list[dict[str, object]] = []
    for path_index in range(path_count):
        order = int(meta["order"][path_index])
        rows.append(
            {
                "feature_index": path_index,
                "feature_name": f"wst_order{order}_path{path_index:04d}_mean",
                "scattering_order": order,
                "path_index": path_index,
                "temporal_pooling": "mean",
                "j_1": meta_value(meta["j"], path_index, 0),
                "j_2": meta_value(meta["j"], path_index, 1),
                "xi_1": meta_value(meta["xi"], path_index, 0),
                "xi_2": meta_value(meta["xi"], path_index, 1),
            }
        )
    return pd.DataFrame(rows)


def meta_value(values: object, row: int, column: int) -> float:
    array = np.asarray(values)
    value = float(array[row, column])
    return value if np.isfinite(value) else np.nan


def event_metadata_row(
    row: pd.Series,
    feature_row: int,
    waveform_peak: float,
    extraction_seconds: float,
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
        "window_observed_duration": float(row["window_observed_duration"]),
        "window_total_padding": float(row["window_total_padding"]),
        "non_overlap_adjusted": bool(row["non_overlap_adjusted"]),
        "fallback_no_event": bool(row["fallback_no_event"]),
        "waveform_peak_before_normalization": waveform_peak,
        "scattering_extraction_seconds": extraction_seconds,
    }


def process_batch(
    scattering: Scattering1D,
    waveforms: list[np.ndarray],
    feature_dimension: int,
) -> tuple[np.ndarray, float]:
    tensor = np.stack(waveforms).astype(np.float32, copy=False)
    start = time.perf_counter()
    coefficients = scattering(tensor)
    elapsed = time.perf_counter() - start
    coefficients = np.asarray(coefficients, dtype=np.float32)
    if coefficients.ndim != 3 or coefficients.shape[0] != len(waveforms):
        raise RuntimeError(
            f"Forma scattering inesperada: {coefficients.shape}"
        )
    pooled = coefficients.mean(axis=-1, dtype=np.float32)
    if pooled.shape != (len(waveforms), feature_dimension):
        raise RuntimeError(f"Forma tras media temporal inesperada: {pooled.shape}")
    if not np.isfinite(pooled).all():
        raise RuntimeError("Wavelet scattering produjo NaN o infinito.")
    return pooled, elapsed


def extract_and_save_split(
    split_name: str,
    manifest: pd.DataFrame,
    config: ScatteringConfig,
    scattering: Scattering1D,
    feature_dimension: int,
    batch_size: int,
    output_dir: Path,
) -> tuple[int, int]:
    matrix = np.empty((len(manifest), feature_dimension), dtype=np.float32)
    metadata_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    pending_waveforms: list[np.ndarray] = []
    pending_rows: list[tuple[int, pd.Series, float]] = []

    def flush_batch() -> None:
        if not pending_waveforms:
            return
        features, elapsed = process_batch(
            scattering,
            pending_waveforms,
            feature_dimension,
        )
        seconds_per_event = elapsed / len(pending_waveforms)
        for batch_index, (output_row, row, peak) in enumerate(pending_rows):
            matrix[output_row] = features[batch_index]
            metadata_rows.append(
                event_metadata_row(row, output_row, peak, seconds_per_event)
            )
        pending_waveforms.clear()
        pending_rows.clear()

    for output_row, (_, row) in enumerate(
        tqdm(
            manifest.iterrows(),
            total=len(manifest),
            desc=f"Cargando WST {split_name}",
        )
    ):
        try:
            waveform, _ = base.load_event_waveform(row, base.PRESETS["paper64"])
            waveform, peak = normalize_waveform(
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
        pd.DataFrame(errors).to_csv(error_path, index=False, encoding="utf-8-sig")
        raise RuntimeError(
            f"Fallaron {len(errors)} eventos de {split_name}. Revisa {error_path}. "
            "No se han guardado matrices parciales."
        )

    metadata = pd.DataFrame(metadata_rows).sort_values("feature_row").reset_index(drop=True)
    event_counts = metadata.groupby("original_uuid")["event_id"].transform("count")
    metadata["recording_weight"] = 1.0 / event_counts.astype(float)

    if metadata["feature_row"].tolist() != list(range(len(metadata))):
        raise RuntimeError("Los metadatos WST no quedaron alineados.")
    if matrix.shape != (len(metadata), feature_dimension):
        raise RuntimeError("La matriz WST y sus metadatos no coinciden.")
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
    config: ScatteringConfig,
    layout: dict[str, object],
    feature_layout: pd.DataFrame,
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
        "temporal_aggregation": "mean_over_scattering_time_positions",
        "dimensionality_reduction": "none",
        "feature_layout": "one_temporal_mean_per_scattering_path",
        "stored_dtype": "float32",
        "test_processed": False,
        **layout,
    }
    pd.DataFrame([configuration]).to_csv(
        output_dir / "wavelet_scattering_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_layout.to_csv(
        output_dir / "wavelet_scattering_feature_layout.csv",
        index=False,
        encoding="utf-8-sig",
    )


def print_check(
    config: ScatteringConfig,
    manifests: dict[str, pd.DataFrame],
    layout: dict[str, object],
    batch_size: int,
) -> None:
    dimension = int(layout["feature_dimension"])
    print("=" * 78)
    print("CHECK - WAVELET SCATTERING COMPLETO STAGE 2")
    print("=" * 78)
    print(f"Preset: {config.name}")
    print(f"Audio: {config.target_samples} muestras a {config.sample_rate} Hz")
    print(
        "Bancos: 8 wavelets/octava (orden 1), "
        "1 wavelet/octava (orden 2)"
    )
    print(
        f"Invariancia: {config.invariance_samples} muestras "
        f"({config.invariance_scale_seconds:.3f} s), J={config.j}"
    )
    print(
        f"Caminos por orden 0/1/2: {layout['order_0_paths']} / "
        f"{layout['order_1_paths']} / {layout['order_2_paths']}"
    )
    print(
        f"Salida interna: {layout['path_count']} caminos x "
        f"{layout['time_position_count']} posiciones temporales"
    )
    print(
        f"Salida guardada: media de las posiciones para cada camino = "
        f"{dimension} features por evento"
    )
    print("Sin PCA ni seleccion de features; se conservan todos los caminos.")
    print(f"Backend estable: kymatio.numpy; batch_size={batch_size}")
    for split_name, manifest in manifests.items():
        estimated_mb = len(manifest) * dimension * 4 / (1024**2)
        print(
            f"{split_name}: {len(manifest)} eventos; "
            f"matriz estimada {estimated_mb:.2f} MiB"
        )
    print("TEST no sera leido ni procesado.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrae Wavelet Scattering completo por evento dry/wet."
    )
    parser.add_argument(
        "--action",
        choices=["check", "extract"],
        default="check",
        help="check valida configuracion; extract procesa train/validation.",
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
    scattering = build_scattering(config)
    path_count, _, _, layout = infer_layout(
        scattering,
        config,
    )
    feature_layout = feature_layout_dataframe(
        scattering,
        path_count,
    )
    print_check(config, manifests, layout, args.batch_size)

    if args.action == "check":
        print("Comprobacion completada. No se extrajeron features.")
        return

    output_dir = OUTPUT_ROOT / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_configuration(
        config,
        layout,
        feature_layout,
        output_dir,
        args.batch_size,
    )
    for split_name in SPLITS:
        shape = extract_and_save_split(
            split_name,
            manifests[split_name],
            config,
            scattering,
            int(layout["feature_dimension"]),
            args.batch_size,
            output_dir,
        )
        print(f"{split_name}: matriz guardada {shape}")

    print("=" * 78)
    print("EXTRACCION WAVELET SCATTERING COMPLETADA")
    print("=" * 78)
    print(f"Resultados: {output_dir}")
    print("TEST permanece reservado.")


if __name__ == "__main__":
    main()
