"""Extrae características gammatone para Stage 2 dry/wet.

El extractor consume los manifests finales de eventos, reconstruye cada
entrada de 1,5 s con el padding indicado y calcula un cocleograma real. No
procesa test y no guarda WAV ni matrices 64x64 completas.

Se guardan dos vistas:
1. Características por evento, para entrenar un clasificador por tos y
   promediar probabilidades por grabación.
2. Características agregadas por original_uuid, para entrenar directamente
   un clasificador por grabación.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
EVENTS_DIR = SCRIPT_DIR / "metadata_audio_events_stage2_dry_wet"
OUTPUT_ROOT = (
    SCRIPT_DIR / "features_extracted_stage2_dry_wet_cochleograms"
)
GRAPH_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "cochleogram_audit"
)

EVENT_FILES = {
    "train": "metadata_events_train_stage2_dry_wet.csv",
    "validation": "metadata_events_validation_stage2_dry_wet.csv",
}

EXPECTED_SEGMENTATION_CONFIG = (
    "top15_margin6_smooth30_gap500_min80_win1500"
)
EXPECTED_SEGMENTATION_METHOD = (
    "energy_hysteresis_bounded_merge_nonoverlap_v3"
)

RANDOM_STATE = 42
DYNAMIC_RANGE_DB = 80.0


@dataclass(frozen=True)
class CochleogramConfig:
    name: str
    sample_rate: int = 16_000
    target_duration_seconds: float = 1.5
    n_filters: int = 64
    n_time_frames: int = 64
    minimum_frequency_hz: float = 80.0
    maximum_frequency_hz: float = 7_500.0
    frequency_blocks: int = 8
    time_blocks: int = 8

    @property
    def target_samples(self) -> int:
        return int(
            round(self.sample_rate * self.target_duration_seconds)
        )

    @property
    def frame_length(self) -> int:
        # Produce 64 frames con un 50 % de solapamiento para 24.000
        # muestras: frame=738 y hop=369.
        approximate = int(
            round(2 * self.target_samples / (self.n_time_frames + 1))
        )
        return approximate - (approximate % 2)

    @property
    def hop_length(self) -> int:
        return self.frame_length // 2


PRESETS = {
    # Réplica ligera del frente auditivo del paper: 64 filtros y 64 frames.
    # La diferencia documentada es que conservamos 1,5 s, no 1 s.
    "paper64": CochleogramConfig(name="paper64", n_filters=64),
    # Ablación posterior para reducir aproximadamente a la mitad el coste
    # del banco de filtros manteniendo el mismo vector de bloques.
    "compact32": CochleogramConfig(name="compact32", n_filters=32),
}


REQUIRED_EVENT_COLUMNS = {
    "event_id",
    "event_index",
    "original_uuid",
    "uuid_segmento",
    "cough_type",
    "cough_type_consensus",
    "stage2_target",
    "fold",
    "split",
    "source_audio_path",
    "window_start",
    "window_end",
    "window_padding_left",
    "window_padding_right",
    "window_observed_duration",
    "window_total_padding",
    "target_window_duration",
    "event_duration",
    "non_overlap_adjusted",
    "fallback_no_event",
    "segmentation_config",
    "segmentation_method",
    "enforce_non_overlap",
}


def parse_boolean_column(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    invalid = sorted(set(normalized.unique()) - set(mapping))
    if invalid:
        raise ValueError(
            f"Valores booleanos inválidos en {column_name}: {invalid}"
        )
    return normalized.map(mapping).astype(bool)


def load_event_manifest(split_name: str) -> pd.DataFrame:
    path = EVENTS_DIR / EVENT_FILES[split_name]
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra el manifest: {path}")

    df = pd.read_csv(
        path,
        dtype={
            "event_id": str,
            "original_uuid": str,
            "uuid_segmento": str,
            "source_audio_path": str,
        },
    )
    missing = REQUIRED_EVENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en {path.name}: {sorted(missing)}"
        )
    if df.empty:
        raise ValueError(f"El manifest {path.name} está vacío.")
    if df["event_id"].duplicated().any():
        raise ValueError(f"Hay event_id duplicados en {path.name}.")

    df = df.copy()
    for column in (
        "non_overlap_adjusted",
        "fallback_no_event",
        "enforce_non_overlap",
    ):
        df[column] = parse_boolean_column(df[column], column)

    numeric_columns = [
        "event_index",
        "stage2_target",
        "fold",
        "window_start",
        "window_end",
        "window_padding_left",
        "window_padding_right",
        "window_observed_duration",
        "window_total_padding",
        "target_window_duration",
        "event_duration",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if set(df["split"].astype(str).unique()) != {split_name}:
        raise ValueError(f"La columna split no coincide con {split_name}.")
    if set(df["cough_type"].unique()) != {"dry", "wet"}:
        raise ValueError(f"{split_name} no contiene exactamente dry/wet.")
    expected_target = df["cough_type"].map({"dry": 0, "wet": 1})
    if not np.array_equal(
        df["stage2_target"].astype(int).to_numpy(),
        expected_target.astype(int).to_numpy(),
    ):
        raise ValueError("stage2_target no coincide con cough_type.")
    if set(df["segmentation_config"].astype(str).unique()) != {
        EXPECTED_SEGMENTATION_CONFIG
    }:
        raise ValueError("El manifest no usa la configuración final.")
    if set(df["segmentation_method"].astype(str).unique()) != {
        EXPECTED_SEGMENTATION_METHOD
    }:
        raise ValueError("El manifest no usa el método final.")
    if not df["enforce_non_overlap"].all():
        raise ValueError("El manifest contiene eventos con solape habilitado.")

    target_duration = df["target_window_duration"].to_numpy(float)
    reconstructed = (
        df["window_observed_duration"].to_numpy(float)
        + df["window_total_padding"].to_numpy(float)
    )
    if not np.allclose(reconstructed, target_duration, atol=1e-6):
        raise ValueError("Audio observado + padding no reconstruye 1,5 s.")

    for uuid_segmento, group in df.groupby("uuid_segmento"):
        ordered = group.sort_values("window_start")
        starts = ordered["window_start"].to_numpy(float)
        ends = ordered["window_end"].to_numpy(float)
        if np.any(ends[:-1] > starts[1:] + 1e-9):
            raise ValueError(
                f"Persisten ventanas solapadas en {uuid_segmento}."
            )

    consistency_columns = [
        "cough_type",
        "cough_type_consensus",
        "stage2_target",
        "fold",
        "split",
    ]
    inconsistent = (
        df.groupby("original_uuid")[consistency_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if inconsistent.any():
        raise ValueError(
            "Metadatos inconsistentes dentro de original_uuid: "
            f"{inconsistent[inconsistent].index[:10].tolist()}"
        )

    return df.sort_values(
        ["fold", "stage2_target", "original_uuid", "event_index"]
    ).reset_index(drop=True)


def hz_to_erb(frequency_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.00437 * frequency_hz)


def erb_to_hz(erb_value: np.ndarray) -> np.ndarray:
    return (10.0 ** (erb_value / 21.4) - 1.0) / 0.00437


def erb_spaced_frequencies(config: CochleogramConfig) -> np.ndarray:
    erb_minimum = hz_to_erb(
        np.asarray(config.minimum_frequency_hz, dtype=np.float64)
    )
    erb_maximum = hz_to_erb(
        np.asarray(config.maximum_frequency_hz, dtype=np.float64)
    )
    erb_values = np.linspace(
        float(erb_minimum),
        float(erb_maximum),
        config.n_filters,
    )
    return erb_to_hz(erb_values).astype(np.float64)


def build_gammatone_filterbank(
    config: CochleogramConfig,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    center_frequencies = erb_spaced_frequencies(config)
    coefficients = []
    for frequency in center_frequencies:
        numerator, denominator = signal.gammatone(
            float(frequency),
            "iir",
            fs=config.sample_rate,
        )
        coefficients.append((numerator, denominator))
    return center_frequencies, coefficients


def load_event_waveform(
    row: pd.Series,
    config: CochleogramConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve waveform de 1,5 s y su porción real sin padding."""

    audio_path = Path(str(row["source_audio_path"]))
    if not audio_path.is_file():
        raise FileNotFoundError(f"No existe el audio: {audio_path}")

    left_padding = int(
        round(float(row["window_padding_left"]) * config.sample_rate)
    )
    right_padding = int(
        round(float(row["window_padding_right"]) * config.sample_rate)
    )
    observed_target = (
        config.target_samples - left_padding - right_padding
    )
    if observed_target <= 0:
        raise ValueError(
            f"Padding inválido para {row['event_id']}: "
            f"left={left_padding}, right={right_padding}"
        )

    duration = float(row["window_end"]) - float(row["window_start"])
    observed, _ = librosa.load(
        audio_path,
        sr=config.sample_rate,
        mono=True,
        offset=float(row["window_start"]),
        duration=duration,
    )
    observed = np.asarray(observed, dtype=np.float32)

    if observed.size > observed_target:
        observed = observed[:observed_target]
    elif observed.size < observed_target:
        observed = np.pad(
            observed,
            (0, observed_target - observed.size),
            mode="constant",
        )

    waveform = np.pad(
        observed,
        (left_padding, right_padding),
        mode="constant",
    ).astype(np.float32)
    if waveform.size != config.target_samples:
        raise RuntimeError(
            f"Longitud final inválida para {row['event_id']}: "
            f"{waveform.size}"
        )
    return waveform, observed


def compute_cochleogram(
    waveform: np.ndarray,
    filter_coefficients: list[tuple[np.ndarray, np.ndarray]],
    config: CochleogramConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula energía gammatone [filtros, frames] y valores relativos dB."""

    energy_rows = []
    for numerator, denominator in filter_coefficients:
        filtered = signal.lfilter(
            numerator,
            denominator,
            waveform,
        )
        frames = librosa.util.frame(
            filtered,
            frame_length=config.frame_length,
            hop_length=config.hop_length,
        )
        energy_rows.append(np.mean(frames * frames, axis=0))

    energy = np.asarray(energy_rows, dtype=np.float64)
    if energy.shape != (config.n_filters, config.n_time_frames):
        raise RuntimeError(
            "Dimensión de cocleograma inesperada: "
            f"{energy.shape}; esperada "
            f"({config.n_filters}, {config.n_time_frames})."
        )
    if not np.isfinite(energy).all():
        raise RuntimeError("El cocleograma contiene NaN o infinito.")

    reference = float(np.max(energy))
    if reference <= np.finfo(np.float64).tiny:
        raise ValueError("Evento completamente silencioso.")

    relative_power = np.maximum(
        energy / reference,
        10.0 ** (-DYNAMIC_RANGE_DB / 10.0),
    )
    cochleogram_db = 10.0 * np.log10(relative_power)
    cochleogram_db = np.clip(
        cochleogram_db,
        -DYNAMIC_RANGE_DB,
        0.0,
    )
    cochleogram_scaled = (
        cochleogram_db + DYNAMIC_RANGE_DB
    ) / DYNAMIC_RANGE_DB
    return cochleogram_scaled.astype(np.float32), cochleogram_db


def extract_block_features(
    cochleogram: np.ndarray,
    config: CochleogramConfig,
) -> np.ndarray:
    if config.n_filters % config.frequency_blocks != 0:
        raise ValueError("n_filters no es divisible por frequency_blocks.")
    if config.n_time_frames % config.time_blocks != 0:
        raise ValueError("n_time_frames no es divisible por time_blocks.")

    filters_per_block = config.n_filters // config.frequency_blocks
    frames_per_block = config.n_time_frames // config.time_blocks
    features = []

    for frequency_block in range(config.frequency_blocks):
        frequency_start = frequency_block * filters_per_block
        frequency_end = frequency_start + filters_per_block
        for time_block in range(config.time_blocks):
            time_start = time_block * frames_per_block
            time_end = time_start + frames_per_block
            block = cochleogram[
                frequency_start:frequency_end,
                time_start:time_end,
            ]
            features.extend(
                [float(np.mean(block)), float(np.std(block))]
            )

    return np.asarray(features, dtype=np.float32)


def compute_auxiliary_event_metadata(
    observed: np.ndarray,
    row: pd.Series,
    config: CochleogramConfig,
) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(observed.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(observed)))
    epsilon = 1e-12
    rms_dbfs = 20.0 * np.log10(max(rms, epsilon))
    peak_dbfs = 20.0 * np.log10(max(peak, epsilon))
    crest_factor_db = peak_dbfs - rms_dbfs
    padding_fraction = float(row["window_total_padding"]) / (
        config.target_duration_seconds
    )
    return {
        "waveform_rms_dbfs": rms_dbfs,
        "waveform_peak_dbfs": peak_dbfs,
        "waveform_crest_factor_db": crest_factor_db,
        "padding_fraction": padding_fraction,
    }


def event_feature_names(config: CochleogramConfig) -> list[str]:
    names = []
    for frequency_block in range(config.frequency_blocks):
        for time_block in range(config.time_blocks):
            prefix = f"coch_f{frequency_block}_t{time_block}"
            names.extend([f"{prefix}_mean", f"{prefix}_std"])
    return names


def extract_event_features(
    manifest: pd.DataFrame,
    split_name: str,
    config: CochleogramConfig,
    filter_coefficients: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, pd.DataFrame]:
    features = []
    metadata_rows = []
    errors = []

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc=f"Cocleogramas {split_name}",
    ):
        try:
            waveform, observed = load_event_waveform(row, config)
            cochleogram, _ = compute_cochleogram(
                waveform,
                filter_coefficients,
                config,
            )
            block_features = extract_block_features(
                cochleogram,
                config,
            )
            auxiliary_metadata = compute_auxiliary_event_metadata(
                observed,
                row,
                config,
            )
            event_features = block_features

            if not np.isfinite(event_features).all():
                raise RuntimeError("Vector con NaN o infinito.")

            feature_row = len(features)
            features.append(event_features)
            metadata_rows.append(
                {
                    "feature_row": feature_row,
                    "event_id": row["event_id"],
                    "event_index": int(row["event_index"]),
                    "original_uuid": row["original_uuid"],
                    "uuid_segmento": row["uuid_segmento"],
                    "cough_type": row["cough_type"],
                    "cough_type_consensus": (
                        row["cough_type_consensus"]
                    ),
                    "stage2_target": int(row["stage2_target"]),
                    "fold": int(row["fold"]),
                    "split": row["split"],
                    "event_duration": float(row["event_duration"]),
                    "window_observed_duration": float(
                        row["window_observed_duration"]
                    ),
                    "window_total_padding": float(
                        row["window_total_padding"]
                    ),
                    "non_overlap_adjusted": bool(
                        row["non_overlap_adjusted"]
                    ),
                    "fallback_no_event": bool(
                        row["fallback_no_event"]
                    ),
                    **auxiliary_metadata,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "event_id": row["event_id"],
                    "original_uuid": row["original_uuid"],
                    "error": str(exc),
                }
            )

    errors_df = pd.DataFrame(
        errors,
        columns=["event_id", "original_uuid", "error"],
    )
    if not errors_df.empty:
        error_path = OUTPUT_ROOT / config.name / (
            f"errors_{split_name}.csv"
        )
        error_path.parent.mkdir(parents=True, exist_ok=True)
        errors_df.to_csv(error_path, index=False, encoding="utf-8-sig")
        raise RuntimeError(
            f"Fallaron {len(errors_df)} eventos de {split_name}. "
            f"Revisa {error_path}."
        )

    matrix = np.asarray(features, dtype=np.float32)
    metadata = pd.DataFrame(metadata_rows)
    expected_dimension = len(event_feature_names(config))
    if matrix.shape != (len(manifest), expected_dimension):
        raise RuntimeError(
            f"Matriz de eventos inválida: {matrix.shape}; "
            f"esperada ({len(manifest)}, {expected_dimension})."
        )

    event_counts = metadata.groupby("original_uuid")["event_id"].transform(
        "count"
    )
    metadata["recording_weight"] = 1.0 / event_counts.astype(float)
    return matrix, metadata


def aggregate_recording_features(
    event_matrix: np.ndarray,
    event_metadata: pd.DataFrame,
    config: CochleogramConfig,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    pooled_rows = []
    metadata_rows = []

    for original_uuid, group in event_metadata.groupby(
        "original_uuid",
        sort=False,
    ):
        indices = group["feature_row"].to_numpy(dtype=int)
        values = event_matrix[indices]
        feature_mean = np.mean(values, axis=0)
        feature_std = np.std(values, axis=0)
        feature_maximum = np.max(values, axis=0)
        pooled_rows.append(
            np.concatenate(
                [
                    feature_mean,
                    feature_std,
                    feature_maximum,
                ]
            ).astype(np.float32)
        )
        metadata_rows.append(
            {
                "feature_row": len(metadata_rows),
                "original_uuid": original_uuid,
                "cough_type": group["cough_type"].iloc[0],
                "cough_type_consensus": (
                    group["cough_type_consensus"].iloc[0]
                ),
                "stage2_target": int(group["stage2_target"].iloc[0]),
                "fold": int(group["fold"].iloc[0]),
                "split": group["split"].iloc[0],
                "event_count": len(group),
                "event_duration_mean": group["event_duration"].mean(),
                "event_duration_std": group["event_duration"].std(ddof=0),
                "padding_fraction_mean": group["padding_fraction"].mean(),
                "padding_fraction_max": group["padding_fraction"].max(),
                "waveform_rms_dbfs_mean": group["waveform_rms_dbfs"].mean(),
            }
        )

    event_names = event_feature_names(config)
    recording_names = (
        [f"event_mean__{name}" for name in event_names]
        + [f"event_std__{name}" for name in event_names]
        + [f"event_max__{name}" for name in event_names]
    )
    matrix = np.asarray(pooled_rows, dtype=np.float32)
    metadata = pd.DataFrame(metadata_rows)
    if matrix.shape[1] != len(recording_names):
        raise RuntimeError("Dimensión agregada y nombres no coinciden.")
    return matrix, metadata, recording_names


def save_split_features(
    split_name: str,
    event_matrix: np.ndarray,
    event_metadata: pd.DataFrame,
    recording_matrix: np.ndarray,
    recording_metadata: pd.DataFrame,
    output_dir: Path,
) -> None:
    np.save(output_dir / f"X_events_{split_name}.npy", event_matrix)
    np.save(
        output_dir / f"y_events_{split_name}.npy",
        event_metadata["stage2_target"].to_numpy(np.int64),
    )
    np.save(
        output_dir / f"folds_events_{split_name}.npy",
        event_metadata["fold"].to_numpy(np.int32),
    )
    event_metadata.to_csv(
        output_dir / f"metadata_events_features_{split_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.save(output_dir / f"X_{split_name}.npy", recording_matrix)
    np.save(
        output_dir / f"y_{split_name}.npy",
        recording_metadata["stage2_target"].to_numpy(np.int64),
    )
    np.save(
        output_dir / f"folds_{split_name}.npy",
        recording_metadata["fold"].to_numpy(np.int32),
    )
    recording_metadata.to_csv(
        output_dir / f"metadata_recordings_features_{split_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def select_audit_events(manifest: pd.DataFrame) -> pd.DataFrame:
    selected = []
    group_specs = [
        ("dry", "gold_expert"),
        ("dry", "weak_expert"),
        ("wet", "gold_expert"),
        ("wet", "weak_expert"),
    ]
    for group_index, (cough_type, consensus) in enumerate(group_specs):
        candidates = manifest[
            (manifest["cough_type"] == cough_type)
            & (manifest["cough_type_consensus"] == consensus)
        ]
        if candidates.empty:
            raise ValueError(
                f"No hay eventos para {cough_type}/{consensus}."
            )
        chosen = candidates.sample(
            n=min(2, len(candidates)),
            random_state=RANDOM_STATE + group_index,
        )
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True)


def create_cochleogram_audit_graph(
    train_manifest: pd.DataFrame,
    config: CochleogramConfig,
    filter_coefficients: list[tuple[np.ndarray, np.ndarray]],
    center_frequencies: np.ndarray,
) -> Path:
    examples = select_audit_events(train_manifest)
    figure, axes = plt.subplots(
        nrows=len(examples),
        ncols=2,
        figsize=(16, 3.2 * len(examples)),
        squeeze=False,
    )

    for row_index, (_, row) in enumerate(examples.iterrows()):
        waveform, _ = load_event_waveform(row, config)
        cochleogram, cochleogram_db = compute_cochleogram(
            waveform,
            filter_coefficients,
            config,
        )
        time_axis = np.arange(waveform.size) / config.sample_rate
        axes[row_index, 0].plot(time_axis, waveform, linewidth=0.65)
        axes[row_index, 0].set_xlim(0.0, config.target_duration_seconds)
        axes[row_index, 0].set_ylabel("Amplitude")
        axes[row_index, 0].set_title(
            f"{row['cough_type']} | {row['cough_type_consensus']} | "
            f"{row['event_id']}"
        )
        axes[row_index, 0].grid(alpha=0.2)

        image = axes[row_index, 1].imshow(
            cochleogram_db,
            origin="lower",
            aspect="auto",
            extent=[
                0.0,
                config.target_duration_seconds,
                float(center_frequencies[0]),
                float(center_frequencies[-1]),
            ],
            cmap="magma",
            vmin=-DYNAMIC_RANGE_DB,
            vmax=0.0,
        )
        axes[row_index, 1].set_ylabel("Center frequency (Hz)")
        axes[row_index, 1].set_title(
            f"Gammatone cochleogram {cochleogram.shape}"
        )
        figure.colorbar(image, ax=axes[row_index, 1], label="Relative dB")

    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")

    figure.suptitle(
        f"Stage 2 cochleogram audit — {config.name}",
        fontsize=15,
        y=1.002,
    )
    figure.tight_layout()
    graph_dir = GRAPH_ROOT / config.name
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "cochleogram_examples_train.png"
    figure.savefig(graph_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return graph_path


def save_configuration(
    config: CochleogramConfig,
    center_frequencies: np.ndarray,
    output_dir: Path,
) -> None:
    configuration = {
        **asdict(config),
        "target_samples": config.target_samples,
        "frame_length": config.frame_length,
        "hop_length": config.hop_length,
        "frame_overlap_fraction": 0.5,
        "dynamic_range_db": DYNAMIC_RANGE_DB,
        "event_feature_dimension": len(event_feature_names(config)),
        "paper_signal_duration_seconds": 1.0,
        "implementation_note": (
            "Paper-like 64x64 gammatone energy; this experiment uses "
            "the audited 1.5-second event window."
        ),
    }
    pd.DataFrame([configuration]).to_csv(
        output_dir / "cochleogram_configuration.csv",
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


def validate_train_validation_disjoint(
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
) -> None:
    overlap = set(train_manifest["original_uuid"]) & set(
        validation_manifest["original_uuid"]
    )
    if overlap:
        raise ValueError(
            "Train y validation comparten original_uuid: "
            f"{sorted(overlap)[:10]}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrae cocleogramas gammatone para Stage 2 dry/wet."
    )
    parser.add_argument(
        "--action",
        choices=["audit", "extract", "all"],
        default="audit",
        help=(
            "audit: genera ejemplos; extract: procesa train/validation; "
            "all: ejecuta ambas acciones."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="paper64",
        help="paper64 replica 64 filtros; compact32 es la ablación móvil.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PRESETS[args.preset]
    output_dir = OUTPUT_ROOT / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = load_event_manifest("train")
    center_frequencies, filter_coefficients = (
        build_gammatone_filterbank(config)
    )
    save_configuration(config, center_frequencies, output_dir)

    print("=" * 72)
    print("COCHLEOGRAMAS GAMMATONE — STAGE 2 DRY/WET")
    print("=" * 72)
    print(f"Preset: {config.name}")
    print(
        f"Forma por evento: {config.n_filters} x "
        f"{config.n_time_frames}"
    )
    print(
        f"Vector compacto por evento: "
        f"{len(event_feature_names(config))} características"
    )
    print("TEST no será leído ni procesado.")

    if args.action in {"audit", "all"}:
        graph_path = create_cochleogram_audit_graph(
            train_manifest,
            config,
            filter_coefficients,
            center_frequencies,
        )
        print(f"Gráfica de auditoría: {graph_path}")

    if args.action in {"extract", "all"}:
        validation_manifest = load_event_manifest("validation")
        validate_train_validation_disjoint(
            train_manifest,
            validation_manifest,
        )

        event_names = event_feature_names(config)
        pd.DataFrame(
            {
                "feature_index": np.arange(len(event_names)),
                "feature_name": event_names,
            }
        ).to_csv(
            output_dir / "event_feature_names.csv",
            index=False,
            encoding="utf-8-sig",
        )

        for split_name, manifest in (
            ("train", train_manifest),
            ("validation", validation_manifest),
        ):
            event_matrix, event_metadata = extract_event_features(
                manifest,
                split_name,
                config,
                filter_coefficients,
            )
            (
                recording_matrix,
                recording_metadata,
                recording_names,
            ) = aggregate_recording_features(
                event_matrix,
                event_metadata,
                config,
            )
            save_split_features(
                split_name,
                event_matrix,
                event_metadata,
                recording_matrix,
                recording_metadata,
                output_dir,
            )
            print(
                f"{split_name}: eventos {event_matrix.shape}; "
                f"grabaciones {recording_matrix.shape}"
            )

        pd.DataFrame(
            {
                "feature_index": np.arange(len(recording_names)),
                "feature_name": recording_names,
            }
        ).to_csv(
            output_dir / "recording_feature_names.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Características guardadas en: {output_dir}")


if __name__ == "__main__":
    main()
