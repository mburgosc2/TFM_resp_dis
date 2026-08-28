"""Extrae WST reutilizable por ventana para Stage 1 cough/no-cough.

El calculo caro se realiza una sola vez y se guarda antes del pooling por
grabacion. Cada segmento se divide en ventanas de 1,5 s. Si queda una cola se
toma una ventana completa alineada al final y se registra un peso igual a la
fraccion de audio nuevo. Los audios cortos usan padding y conservan su
fraccion valida. Antes de normalizar se guardan peak, RMS y crest factor.

La salida principal ``X_windows_raw_<split>.npy`` contiene una fila por
ventana y los paths WST raw de orden 0 y 1, despues de su promedio
temporal. ``metadata_windows_<split>.csv`` permite construir posteriormente
weighted mean/std y max sin ponderar por ``original_uuid``. ``wst_paths.csv``
permite comparar S1 frente a S0+S1 sin recalcular la WST.

No se aplica log-WST, pooling recording-level, scaler, PCA ni clasificador.
TRAIN y VALIDATION son los splits predeterminados; TEST debe solicitarse con
``--splits test``.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from kymatio.numpy import Scattering1D
from tqdm import tqdm

import feature_extraction_stage1_mfcc_fsd50k_coughs as metadata_base


ROOT = Path(__file__).resolve().parent
PRESET_NAME = "paper_q8_t500_window_raw_o01"
OUTPUT_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_coughs_random"
    / PRESET_NAME
)
ALL_SPLITS = ("train", "validation", "test")
DEFAULT_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class ScatteringConfig:
    sample_rate: int = 16_000
    window_seconds: float = 1.5
    invariance_seconds: float = 0.5
    first_order_wavelets_per_octave: int = 8
    max_order: int = 1
    normalize_each_window_peak: bool = True
    silence_threshold: float = metadata_base.SILENCE_AUDIT_THRESHOLD
    amplitude_epsilon: float = 1e-8

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.window_seconds))

    @property
    def invariance_samples(self) -> int:
        return int(round(self.sample_rate * self.invariance_seconds))

    @property
    def j(self) -> int:
        return int(round(math.log2(self.invariance_samples)))

    @property
    def q(self) -> int:
        return self.first_order_wavelets_per_octave


CONFIG = ScatteringConfig()


@dataclass(frozen=True)
class WindowSlice:
    """Ventana y trazabilidad respecto al segmento del que procede."""

    samples: np.ndarray
    start_sample: int
    valid_samples: int
    new_content_samples: int
    padding_left_samples: int
    padding_right_samples: int
    overlap_previous_samples: int
    is_end_aligned: bool

    @property
    def total_padding_samples(self) -> int:
        return self.padding_left_samples + self.padding_right_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WST raw por ventana para Stage 1 cough/no-cough"
    )
    parser.add_argument(
        "--action",
        choices=["check", "extract"],
        default="check",
        help="check valida y estima el trabajo; extract calcula WST",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALL_SPLITS,
        default=list(DEFAULT_SPLITS),
        help=(
            "Splits que se comprobaran o extraeran. Por defecto TRAIN y "
            "VALIDATION; TEST debe solicitarse explicitamente."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--near-silence-policy",
        choices=["reject", "keep"],
        default="reject",
        help="reject reproduce las siete exclusiones del experimento MFCC",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar solo los splits solicitados",
    )
    return parser.parse_args()


def unique_splits(values: list[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(values))
    if not requested:
        raise ValueError("Debe solicitarse al menos un split")
    return requested


def build_scattering(config: ScatteringConfig) -> Scattering1D:
    return Scattering1D(
        J=config.j,
        shape=config.window_samples,
        Q=config.q,
        T=config.invariance_samples,
        max_order=config.max_order,
        out_type="array",
    )


def infer_layout(
    scattering: Scattering1D,
    config: ScatteringConfig,
) -> tuple[int, int, np.ndarray]:
    probe = np.zeros((1, config.window_samples), dtype=np.float32)
    output = np.asarray(scattering(probe))
    if output.ndim != 3 or output.shape[0] != 1:
        raise RuntimeError(f"Forma WST inesperada: {output.shape}")
    path_count = int(output.shape[1])
    time_positions = int(output.shape[2])
    orders = np.asarray(scattering.meta()["order"], dtype=np.int8)
    if orders.shape != (path_count,):
        raise RuntimeError("Los metadatos WST no coinciden con los caminos")
    if set(orders.tolist()) != {0, 1}:
        raise RuntimeError(f"Ordenes WST inesperados: {set(orders.tolist())}")
    return path_count, time_positions, orders


def python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def wst_paths_dataframe(scattering: Scattering1D) -> pd.DataFrame:
    """Crea una fila por path con los metadatos disponibles en Kymatio."""
    meta = scattering.meta()
    path_count = len(meta["order"])
    rows: list[dict[str, object]] = []
    for path_index in range(path_count):
        row: dict[str, object] = {
            "path_index": path_index,
            "scattering_order": int(meta["order"][path_index]),
        }
        for key, values in meta.items():
            if key == "order":
                continue
            try:
                item = values[path_index]
            except (IndexError, TypeError):
                continue
            array = np.asarray(item)
            if array.ndim == 0:
                row[key] = python_scalar(array.item())
            else:
                for position, value in enumerate(array.reshape(-1), start=1):
                    row[f"{key}_{position}"] = python_scalar(value)
        rows.append(row)
    return pd.DataFrame(rows)


def split_into_windows(
    signal: np.ndarray,
    window_samples: int,
) -> list[WindowSlice]:
    """Crea ventanas completas y una ultima ventana alineada al final."""
    signal = np.asarray(signal, dtype=np.float32)
    sample_count = int(signal.size)
    if sample_count <= 0:
        raise ValueError("empty_audio")
    if window_samples <= 0:
        raise ValueError("window_samples debe ser positivo")

    if sample_count <= window_samples:
        padding = window_samples - sample_count
        padding_left = padding // 2
        padding_right = padding - padding_left
        padded = np.pad(
            signal, (padding_left, padding_right), mode="constant"
        ).astype(np.float32, copy=False)
        return [
            WindowSlice(
                samples=padded,
                start_sample=0,
                valid_samples=sample_count,
                new_content_samples=sample_count,
                padding_left_samples=padding_left,
                padding_right_samples=padding_right,
                overlap_previous_samples=0,
                is_end_aligned=False,
            )
        ]

    windows: list[WindowSlice] = []
    regular_starts = list(
        range(0, sample_count - window_samples + 1, window_samples)
    )
    for start in regular_starts:
        windows.append(
            WindowSlice(
                samples=np.asarray(
                    signal[start : start + window_samples], dtype=np.float32
                ),
                start_sample=start,
                valid_samples=window_samples,
                new_content_samples=window_samples,
                padding_left_samples=0,
                padding_right_samples=0,
                overlap_previous_samples=0,
                is_end_aligned=False,
            )
        )

    remainder = sample_count % window_samples
    if remainder:
        final_start = sample_count - window_samples
        previous_end = regular_starts[-1] + window_samples
        new_content = sample_count - previous_end
        if new_content != remainder or final_start == regular_starts[-1]:
            raise RuntimeError("Inconsistencia al crear la ventana final")
        windows.append(
            WindowSlice(
                samples=np.asarray(signal[final_start:sample_count], dtype=np.float32),
                start_sample=final_start,
                valid_samples=window_samples,
                new_content_samples=new_content,
                padding_left_samples=0,
                padding_right_samples=0,
                overlap_previous_samples=window_samples - new_content,
                is_end_aligned=True,
            )
        )

    if any(window.samples.shape != (window_samples,) for window in windows):
        raise RuntimeError("Se creo una ventana con longitud inesperada")
    return windows


def load_segment(row: pd.Series, config: ScatteringConfig) -> np.ndarray:
    duration = float(row["end_time"]) - float(row["start_time"])
    signal, _ = librosa.load(
        Path(str(row["audio_path"])),
        sr=config.sample_rate,
        mono=True,
        offset=float(row["start_time"]),
        duration=duration,
        dtype=np.float32,
    )
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        raise ValueError("empty_audio")
    if not np.isfinite(signal).all():
        raise ValueError("non_finite_audio_samples")
    return signal


def amplitude_statistics(
    real_samples: np.ndarray,
    epsilon: float,
) -> dict[str, float]:
    """Calcula amplitud antes del padding y de la normalizacion por pico."""
    real_samples = np.asarray(real_samples, dtype=np.float32)
    if real_samples.size == 0:
        raise ValueError("No hay muestras validas para medir amplitud")
    peak = float(np.max(np.abs(real_samples)))
    rms = float(np.sqrt(np.mean(np.square(real_samples, dtype=np.float64))))
    return {
        "peak_amplitude": peak,
        "rms_amplitude": rms,
        "log10_peak": float(np.log10(peak + epsilon)),
        "log10_rms": float(np.log10(rms + epsilon)),
        "log10_crest_factor": float(
            np.log10((peak + epsilon) / (rms + epsilon))
        ),
    }


def prepare_window(
    window: WindowSlice,
    original_signal: np.ndarray,
    config: ScatteringConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Mide amplitud original y normaliza la copia enviada a WST."""
    real_samples = original_signal[
        window.start_sample : window.start_sample + window.valid_samples
    ]
    amplitude = amplitude_statistics(real_samples, config.amplitude_epsilon)
    near_silence = amplitude["peak_amplitude"] < config.silence_threshold
    normalized = np.asarray(window.samples, dtype=np.float32).copy()
    peak_normalized = False
    if (
        config.normalize_each_window_peak
        and not near_silence
        and amplitude["peak_amplitude"] > 0.0
    ):
        normalized /= float(amplitude["peak_amplitude"])
        peak_normalized = True
    if not np.isfinite(normalized).all():
        raise RuntimeError("La normalizacion produjo NaN o infinito")
    return normalized, {
        **amplitude,
        "near_silence_window": bool(near_silence),
        "peak_normalized_for_wst": bool(peak_normalized),
    }


def scattering_windows(
    windows: list[np.ndarray],
    scattering: Scattering1D,
    path_count: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    outputs: list[np.ndarray] = []
    elapsed = 0.0
    for start in range(0, len(windows), batch_size):
        tensor = np.stack(windows[start : start + batch_size]).astype(
            np.float32, copy=False
        )
        began = time.perf_counter()
        coefficients = np.asarray(scattering(tensor), dtype=np.float32)
        elapsed += time.perf_counter() - began
        if coefficients.ndim != 3 or coefficients.shape[1] != path_count:
            raise RuntimeError(f"Forma WST inesperada: {coefficients.shape}")
        pooled_time = coefficients.mean(axis=-1, dtype=np.float32)
        if not np.isfinite(pooled_time).all():
            raise RuntimeError("WST produjo NaN o infinito")
        outputs.append(pooled_time)
    return np.concatenate(outputs, axis=0), elapsed


def constant_value(group: pd.DataFrame, column: str) -> object:
    values = group[column].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(
            f"{column} no es constante para {group['original_uuid'].iloc[0]}: "
            f"{values[:5]}"
        )
    return group[column].iloc[0]


def group_constants(group: pd.DataFrame) -> dict[str, object]:
    return {
        "original_uuid": str(constant_value(group, "original_uuid")),
        "split_group": str(constant_value(group, "split_group")),
        "split": str(constant_value(group, "split")),
        "fold": int(constant_value(group, "fold")),
        "stage1_target": int(constant_value(group, "stage1_target")),
        "dataset_origin": str(constant_value(group, "dataset_origin")),
        "record_source": str(constant_value(group, "record_source")),
        "is_new_fsd50k_cough": bool(
            metadata_base.parse_bool_series(
                pd.Series([constant_value(group, "is_new_fsd50k_cough")]),
                "is_new_fsd50k_cough",
            ).iloc[0]
        ),
        "stage2_eligible": bool(
            metadata_base.parse_bool_series(
                pd.Series([constant_value(group, "stage2_eligible")]),
                "stage2_eligible",
            ).iloc[0]
        ),
    }


def window_metadata_row(
    constants: dict[str, object],
    row: pd.Series,
    source_row: int,
    window: WindowSlice,
    amplitude: dict[str, object],
    feature_row: int,
    window_index_recording: int,
    window_index_segment: int,
    config: ScatteringConfig,
) -> dict[str, object]:
    start_seconds_segment = window.start_sample / config.sample_rate
    valid_seconds = window.valid_samples / config.sample_rate
    return {
        "feature_row": feature_row,
        **constants,
        "source_row": source_row,
        "uuid_segmento": str(row["uuid_segmento"]),
        "audio_path": str(row["audio_path"]),
        "window_index_recording": window_index_recording,
        "window_index_segment": window_index_segment,
        "window_start_seconds_segment": start_seconds_segment,
        "window_end_seconds_segment": start_seconds_segment + valid_seconds,
        "window_start_seconds_audio": float(row["start_time"]) + start_seconds_segment,
        "window_end_seconds_audio": (
            float(row["start_time"]) + start_seconds_segment + valid_seconds
        ),
        "valid_samples": window.valid_samples,
        "valid_fraction": window.valid_samples / config.window_samples,
        "new_content_samples": window.new_content_samples,
        "new_content_seconds": window.new_content_samples / config.sample_rate,
        "window_weight": window.new_content_samples / config.window_samples,
        "is_end_aligned": window.is_end_aligned,
        "overlap_previous_samples": window.overlap_previous_samples,
        "overlap_previous_seconds": (
            window.overlap_previous_samples / config.sample_rate
        ),
        "padding_left_samples": window.padding_left_samples,
        "padding_right_samples": window.padding_right_samples,
        "padding_seconds": window.total_padding_samples / config.sample_rate,
        **amplitude,
    }


def recording_metadata_row(
    group: pd.DataFrame,
    constants: dict[str, object],
    recording_row: int,
    first_feature_row: int,
    window_rows: list[dict[str, object]],
    excluded_segments: int,
) -> dict[str, object]:
    first = group.iloc[0]
    weights = np.asarray(
        [float(row["window_weight"]) for row in window_rows], dtype=float
    )
    return {
        "recording_row": recording_row,
        **constants,
        "quality": first.get("quality", ""),
        "uploader": first.get("uploader", ""),
        "input_segment_count": int(len(group)),
        "excluded_segment_count": excluded_segments,
        "included_segment_count": int(len(group) - excluded_segments),
        "uuid_segmentos": "|".join(group["uuid_segmento"].astype(str)),
        "observed_duration_seconds": float(
            (group["end_time"].astype(float) - group["start_time"].astype(float)).sum()
        ),
        "first_feature_row": first_feature_row,
        "last_feature_row": first_feature_row + len(window_rows) - 1,
        "window_count": len(window_rows),
        "effective_window_count": float(weights.sum()),
        "last_window_weight": float(weights[-1]),
        "end_aligned_window_count": int(
            sum(bool(row["is_end_aligned"]) for row in window_rows)
        ),
        "padded_window_count": int(
            sum(float(row["padding_seconds"]) > 0.0 for row in window_rows)
        ),
        "near_silence_window_count": int(
            sum(bool(row["near_silence_window"]) for row in window_rows)
        ),
    }


def extract_split(
    split: str,
    metadata: pd.DataFrame,
    config: ScatteringConfig,
    scattering: Scattering1D,
    path_count: int,
    batch_size: int,
    near_silence_policy: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_blocks: list[np.ndarray] = []
    pending_windows: list[np.ndarray] = []
    window_rows_all: list[dict[str, object]] = []
    recording_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []
    feature_row_offset = 0

    def flush_complete_batches(force: bool = False) -> None:
        """Calcula batches compartidos por varias grabaciones."""
        nonlocal pending_windows
        if force:
            process_count = len(pending_windows)
        else:
            process_count = (len(pending_windows) // batch_size) * batch_size
        if process_count == 0:
            return
        block, _ = scattering_windows(
            pending_windows[:process_count],
            scattering,
            path_count,
            batch_size,
        )
        feature_blocks.append(block)
        pending_windows = pending_windows[process_count:]

    grouped = metadata.groupby("original_uuid", sort=False)
    for original_uuid, group in tqdm(
        grouped,
        total=metadata["original_uuid"].nunique(),
        desc=f"WST ventanas {split}",
    ):
        normalized_windows: list[np.ndarray] = []
        local_window_rows: list[dict[str, object]] = []
        excluded_segments = 0
        try:
            constants = group_constants(group)
            window_index_recording = 0
            for source_row_value, row in group.iterrows():
                source_row = int(source_row_value)
                signal = load_segment(row, config)
                peak = float(np.max(np.abs(signal)))
                if (
                    peak < config.silence_threshold
                    and near_silence_policy == "reject"
                ):
                    excluded_segments += 1
                    issue_rows.append(
                        {
                            "issue_kind": "configured_exclusion",
                            "split": split,
                            "source_row": source_row,
                            "original_uuid": str(original_uuid),
                            "uuid_segmento": str(row["uuid_segmento"]),
                            "audio_path": str(row["audio_path"]),
                            "peak_amplitude": peak,
                            "error": "near_silence_segment",
                        }
                    )
                    continue
                segment_windows = split_into_windows(signal, config.window_samples)
                for window_index_segment, window in enumerate(segment_windows):
                    normalized, amplitude = prepare_window(window, signal, config)
                    normalized_windows.append(normalized)
                    local_window_rows.append(
                        window_metadata_row(
                            constants=constants,
                            row=row,
                            source_row=source_row,
                            window=window,
                            amplitude=amplitude,
                            feature_row=feature_row_offset + len(local_window_rows),
                            window_index_recording=window_index_recording,
                            window_index_segment=window_index_segment,
                            config=config,
                        )
                    )
                    window_index_recording += 1

            if not normalized_windows:
                if excluded_segments == len(group):
                    continue
                raise RuntimeError("La grabacion no produjo ventanas")

            window_rows_all.extend(local_window_rows)
            pending_windows.extend(normalized_windows)
            recording_rows.append(
                recording_metadata_row(
                    group=group,
                    constants=constants,
                    recording_row=len(recording_rows),
                    first_feature_row=feature_row_offset,
                    window_rows=local_window_rows,
                    excluded_segments=excluded_segments,
                )
            )
            feature_row_offset += len(local_window_rows)
            flush_complete_batches()
        except Exception as exc:
            issue_rows.append(
                {
                    "issue_kind": "extraction_error",
                    "split": split,
                    "source_row": "",
                    "original_uuid": str(original_uuid),
                    "uuid_segmento": "",
                    "audio_path": "",
                    "peak_amplitude": np.nan,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    flush_complete_batches(force=True)
    X_windows = (
        np.concatenate(feature_blocks, axis=0).astype(np.float32, copy=False)
        if feature_blocks
        else np.empty((0, path_count), dtype=np.float32)
    )
    windows_manifest = pd.DataFrame(window_rows_all)
    recordings_manifest = pd.DataFrame(recording_rows)
    issues = pd.DataFrame(issue_rows)
    if len(X_windows) != len(windows_manifest):
        raise RuntimeError(f"Salida de ventanas desalineada en {split}")
    if len(windows_manifest):
        expected_rows = np.arange(len(windows_manifest))
        observed_rows = windows_manifest["feature_row"].to_numpy(int)
        if not np.array_equal(expected_rows, observed_rows):
            raise RuntimeError(f"feature_row desalineado en {split}")
        if not np.isfinite(X_windows).all():
            raise RuntimeError(f"La matriz WST de {split} contiene NaN o infinito")
        weights = windows_manifest["window_weight"].to_numpy(float)
        if not np.all((weights > 0.0) & (weights <= 1.0)):
            raise RuntimeError(f"Hay pesos de ventana invalidos en {split}")
    return X_windows, windows_manifest, recordings_manifest, issues


def split_output_files(split: str) -> list[Path]:
    return [
        OUTPUT_DIR / f"X_windows_raw_{split}.npy",
        OUTPUT_DIR / f"metadata_windows_{split}.csv",
        OUTPUT_DIR / f"metadata_recordings_{split}.csv",
        OUTPUT_DIR / f"wst_extraction_issues_{split}.csv",
        OUTPUT_DIR / f"wst_extraction_summary_{split}.csv",
    ]


def summary_row(
    split: str,
    X_windows: np.ndarray,
    windows: pd.DataFrame,
    recordings: pd.DataFrame,
    issues: pd.DataFrame,
    elapsed_seconds: float,
) -> dict[str, object]:
    counts = np.bincount(
        recordings["stage1_target"].to_numpy(np.int8), minlength=2
    )
    issue_kinds = issues.get("issue_kind", pd.Series(dtype=str))
    return {
        "split": split,
        "recordings_extracted": len(recordings),
        "no_cough_recordings": int(counts[0]),
        "cough_recordings": int(counts[1]),
        "new_fsd50k_cough_recordings": int(
            recordings["is_new_fsd50k_cough"].astype(bool).sum()
        ),
        "windows_extracted": len(X_windows),
        "effective_window_count": float(windows["window_weight"].sum()),
        "end_aligned_windows": int(windows["is_end_aligned"].astype(bool).sum()),
        "padded_windows": int((windows["padding_seconds"] > 0.0).sum()),
        "near_silence_windows": int(
            windows["near_silence_window"].astype(bool).sum()
        ),
        "configured_exclusions": int(
            (issue_kinds == "configured_exclusion").sum()
        ),
        "extraction_errors": int((issue_kinds == "extraction_error").sum()),
        "elapsed_seconds": elapsed_seconds,
    }


def rebuild_combined_csv(filename: str, template: str) -> None:
    frames: list[pd.DataFrame] = []
    for split in ALL_SPLITS:
        path = OUTPUT_DIR / template.format(split=split)
        if path.is_file():
            frame = pd.read_csv(path, low_memory=False)
            if not frame.empty:
                frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            OUTPUT_DIR / filename,
            index=False,
            encoding="utf-8-sig",
        )


def save_common_outputs(
    config: ScatteringConfig,
    path_count: int,
    time_positions: int,
    orders: np.ndarray,
    paths: pd.DataFrame,
    near_silence_policy: str,
) -> None:
    paths.to_csv(
        OUTPUT_DIR / "wst_paths.csv", index=False, encoding="utf-8-sig"
    )
    configuration = {
        "experiment": "stage1_wst_window_raw_fsd50k_coughs_random",
        "preset": PRESET_NAME,
        **asdict(config),
        "window_samples": config.window_samples,
        "regular_window_hop_samples": config.window_samples,
        "final_window_policy": "complete_window_aligned_to_segment_end",
        "final_window_weight": "new_content_samples/window_samples",
        "short_window_policy": "symmetric_zero_padding",
        "short_window_weight": "valid_samples/window_samples",
        "invariance_samples": config.invariance_samples,
        "J": config.j,
        "Q": str(config.q),
        "backend": "kymatio.numpy",
        "batching_scope": "shared_across_recordings_within_each_split",
        "path_count": path_count,
        "time_positions_per_path": time_positions,
        "order_0_paths": int(np.sum(orders == 0)),
        "order_1_paths": int(np.sum(orders == 1)),
        "within_window_pooling": "arithmetic mean over WST temporal positions",
        "stored_representation": "raw WST per window after temporal mean",
        "recording_pooling_applied": False,
        "log_transform_applied": False,
        "amplitude_measured_before_peak_normalization": True,
        "near_silence_policy": near_silence_policy,
        "scaler_or_pca_fitted": False,
        "test_default": "not extracted; requires --splits test",
    }
    pd.DataFrame(configuration.items(), columns=["parameter", "value"]).to_csv(
        OUTPUT_DIR / "wst_extraction_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_split_outputs(
    split: str,
    result: tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    elapsed_seconds: float,
) -> dict[str, object]:
    X_windows, windows, recordings, issues = result
    np.save(OUTPUT_DIR / f"X_windows_raw_{split}.npy", X_windows)
    windows.to_csv(
        OUTPUT_DIR / f"metadata_windows_{split}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    recordings.to_csv(
        OUTPUT_DIR / f"metadata_recordings_{split}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if issues.empty:
        issues = pd.DataFrame(
            columns=[
                "issue_kind",
                "split",
                "source_row",
                "original_uuid",
                "uuid_segmento",
                "audio_path",
                "peak_amplitude",
                "error",
            ]
        )
    issues.to_csv(
        OUTPUT_DIR / f"wst_extraction_issues_{split}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = summary_row(
        split, X_windows, windows, recordings, issues, elapsed_seconds
    )
    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / f"wst_extraction_summary_{split}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def print_check(
    splits: dict[str, pd.DataFrame],
    requested_splits: tuple[str, ...],
    config: ScatteringConfig,
    path_count: int,
    time_positions: int,
    orders: np.ndarray,
) -> None:
    print("=" * 78)
    print("CHECK - STAGE 1 WST RAW POR VENTANA")
    print("=" * 78)
    print(f"Preset: {PRESET_NAME}")
    print(
        f"Ventana: {config.window_seconds:.1f} s; hop regular "
        f"{config.window_seconds:.1f} s"
    )
    print("Cola: ventana completa al final, ponderada por contenido nuevo")
    print("Audio corto: padding simetrico y valid_fraction registrada")
    print(
        f"WST: Q={config.q}, J={config.j}, T={config.invariance_samples}, "
        f"orden maximo={config.max_order}"
    )
    print(
        f"Caminos 0/1: {np.sum(orders == 0)} / {np.sum(orders == 1)}; "
        f"posiciones temporales={time_positions}"
    )
    print(
        f"Salida maestra: N_ventanas x {path_count}; sin log ni pooling recording"
    )
    print(f"Splits solicitados: {', '.join(requested_splits)}")
    for split in requested_splits:
        metadata = splits[split]
        durations = (
            metadata["end_time"].astype(float)
            - metadata["start_time"].astype(float)
        )
        estimated_windows = int(
            np.ceil(durations.to_numpy() / config.window_seconds).sum()
        )
        estimated_mib = estimated_windows * path_count * 4 / (1024**2)
        print(
            f"{split:>10}: {metadata['original_uuid'].nunique()} grabaciones, "
            f"{len(metadata)} segmentos, aproximadamente {estimated_windows} "
            f"ventanas ({estimated_mib:.1f} MiB)"
        )
    if "test" not in requested_splits:
        print("TEST no sera procesado. Se extraera solo cuando se solicite.")


def validate_existing_outputs(
    requested_splits: tuple[str, ...], overwrite: bool
) -> None:
    existing = [
        path
        for split in requested_splits
        for path in split_output_files(split)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Ya existe {existing[0]}. Usa --overwrite para reemplazar los "
            "splits solicitados."
        )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size debe ser mayor que cero")
    requested_splits = unique_splits(args.splits)
    # La funcion base valida integridad, etiquetas, folds, rutas y ausencia de
    # grupos compartidos. Solo los audios solicitados se cargan mas adelante.
    splits = metadata_base.load_and_validate_metadata()
    scattering = build_scattering(CONFIG)
    path_count, time_positions, orders = infer_layout(scattering, CONFIG)
    paths = wst_paths_dataframe(scattering)
    print_check(
        splits,
        requested_splits,
        CONFIG,
        path_count,
        time_positions,
        orders,
    )

    if args.action == "check":
        print("Comprobacion completada. No se ha escrito ningun archivo.")
        return

    validate_existing_outputs(requested_splits, args.overwrite)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_common_outputs(
        CONFIG,
        path_count,
        time_positions,
        orders,
        paths,
        args.near_silence_policy,
    )

    total_started = time.perf_counter()
    summaries: list[dict[str, object]] = []
    for split in requested_splits:
        split_started = time.perf_counter()
        result = extract_split(
            split=split,
            metadata=splits[split],
            config=CONFIG,
            scattering=scattering,
            path_count=path_count,
            batch_size=args.batch_size,
            near_silence_policy=args.near_silence_policy,
        )
        errors = result[3]
        real_errors = (
            int(errors["issue_kind"].eq("extraction_error").sum())
            if not errors.empty
            else 0
        )
        if real_errors:
            errors.to_csv(
                OUTPUT_DIR / f"wst_extraction_issues_{split}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            raise RuntimeError(
                f"Fallaron {real_errors} grabaciones de {split}; no se guardara "
                "una matriz parcial. Revisa el CSV de errores."
            )
        summaries.append(
            save_split_outputs(
                split,
                result,
                time.perf_counter() - split_started,
            )
        )

    rebuild_combined_csv(
        "wst_extraction_summary.csv", "wst_extraction_summary_{split}.csv"
    )
    rebuild_combined_csv(
        "wst_extraction_issues.csv", "wst_extraction_issues_{split}.csv"
    )

    elapsed = time.perf_counter() - total_started
    print("\n" + "=" * 78)
    print("EXTRACCION STAGE 1 WST POR VENTANA COMPLETADA")
    print("=" * 78)
    for summary in summaries:
        print(
            f"{summary['split']:>10}: ventanas={summary['windows_extracted']}; "
            f"grabaciones={summary['recordings_extracted']}; "
            f"no_tos/tos={summary['no_cough_recordings']}/"
            f"{summary['cough_recordings']}; "
            f"exclusiones={summary['configured_exclusions']}"
        )
    print(f"Tiempo total de esta ejecucion: {elapsed:.2f} s")
    print(f"Resultados: {OUTPUT_DIR}")
    print("No se ha aplicado log, pooling recording, scaler, PCA ni modelo.")
    if "test" not in requested_splits:
        print("TEST permanece sin extraer.")


if __name__ == "__main__":
    main()
