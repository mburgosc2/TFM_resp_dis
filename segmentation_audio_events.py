"""Deteccion y analisis de eventos acusticos dentro de los splits de audio.

Este script no clasifica todavia los eventos como tos/no-tos. Su funcion es
proponer regiones acusticamente activas y construir un manifiesto con offsets
temporales. Todos los eventos heredan el split y el fold de su grabacion, por
lo que no se vuelven a repartir datos en esta fase.

Modos de uso
------------
1. Comparar parametros exclusivamente sobre TRAIN, sin escribir archivos:

   python segmentation_audio_events.py --action analyze --split train

2. Extraer manifiestos una vez congelados los parametros:

   python segmentation_audio_events.py --action extract --split all

3. Crear la muestra cerrada de 100 audios de TRAIN para auditoria:

   python segmentation_audio_events.py --action audit --split train

La deteccion por energia solo genera candidatos. Para determinar si un
candidato es realmente una tos se necesitara la etapa 1 y una auditoria manual
de una muestra de eventos.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# =============================================================================
# RUTAS Y CONSTANTES
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent

SPLITS_DIR = SCRIPT_DIR / "metadata_splits_multi_experiment_random"
OUTPUT_DIR = SCRIPT_DIR / "metadata_audio_events_experiment_random"
AUDIT_DIR = SCRIPT_DIR / "auditory_sample_audio_events"

PATH_AUDIOS_COUGHVID = REPOSITORY_ROOT / "DATA"
PATH_AUDIOS_FSD50K = (
    REPOSITORY_ROOT
    / "FSD50K_DATA"
    / "FSD50K.dev_audio"
    / "FSD50K_negative_class_dataset"
)

SPLIT_FILES = {
    "train": "metadata_train_multiclass_4c.csv",
    "val": "metadata_val_multiclass_4c.csv",
    "test": "metadata_test_multiclass_4c.csv",
}

SAMPLE_RATE = 16_000
FRAME_LENGTH = 512       # 32 ms a 16 kHz
HOP_LENGTH = 160         # 10 ms a 16 kHz
RANDOM_STATE = 42

AUDIT_EXPECTED_SUBGROUPS = {
    "dry_gold": 8,
    "dry_weak": 16,
    "wet_gold": 6,
    "wet_weak": 18,
    "unknown_gold": 2,
    "unknown_weak": 3,
    "unknown_ambiguous": 3,
    "coughvid_no_cough_low_amplitude": 22,
    "fsd_respiratory_low_amplitude": 11,
    "fsd_non_respiratory_short": 5,
    "fsd_low_amplitude": 3,
    "fsd_long": 3,
}

AUDIT_EXPECTED_CATEGORIES = {
    "dry": 24,
    "wet": 24,
    "unknown": 8,
    "coughvid_no_cough": 22,
    "fsd_no_cough": 22,
}


# =============================================================================
# CONFIGURACION DEL DETECTOR
# =============================================================================
@dataclass(frozen=True)
class SegmentationConfig:
    """Parametros del detector de actividad basado en energia RMS."""

    top_db: float = 25.0
    snr_margin_db: float = 3.0
    hysteresis_db: float = 3.0
    smoothing_ms: float = 30.0
    merge_gap_ms: float = 200.0
    min_event_ms: float = 80.0
    max_event_ms: float = 2_000.0
    output_window_ms: float = 1_500.0

    @property
    def name(self) -> str:
        return (
            f"top{self.top_db:g}_margin{self.snr_margin_db:g}_"
            f"smooth{self.smoothing_ms:g}_gap{self.merge_gap_ms:g}_"
            f"min{self.min_event_ms:g}_win{self.output_window_ms:g}"
        )


@dataclass
class EnvelopeData:
    """Envolvente calculada una sola vez y reutilizable en el grid."""

    rms_db: np.ndarray
    frame_times: np.ndarray
    segment_duration: float
    segment_start: float
    peak_amplitude: float


# =============================================================================
# CARGA DE METADATOS Y AUDIO
# =============================================================================
def load_split_metadata(split_name: str) -> pd.DataFrame:
    csv_path = SPLITS_DIR / SPLIT_FILES[split_name]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el split {split_name}: {csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        dtype={
            "uuid": str,
            "original_uuid": str,
            "uuid_segmento": str,
        },
    )

    required_columns = {
        "uuid_segmento",
        "original_uuid",
        "dataset_origin",
        "start_time",
        "end_time",
        "split",
        "fold",
        "stage1_target",
        "stage2_eligible",
        "stage2_gold_eval",
        "stage2_reject_challenge",
        "cough_type",
        "cough_type_label",
        "cough_type_consensus",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas en {csv_path.name}: {sorted(missing_columns)}"
        )

    if df["uuid_segmento"].duplicated().any():
        raise ValueError(
            f"Hay uuid_segmento duplicados en el split {split_name}."
        )

    return df


def resolve_audio_path(row: pd.Series) -> Path:
    original_uuid = str(row["original_uuid"])

    if row["dataset_origin"] == "FSD50K":
        return PATH_AUDIOS_FSD50K / f"{original_uuid}.wav"

    return PATH_AUDIOS_COUGHVID / f"{original_uuid}.wav"


def load_audio_segment(row: pd.Series) -> tuple[np.ndarray, int, float]:
    audio_path = resolve_audio_path(row)

    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))

    start_time = float(row["start_time"])
    end_time = float(row["end_time"])
    requested_duration = end_time - start_time

    if requested_duration <= 0:
        raise ValueError(
            f"Intervalo invalido para {row['uuid_segmento']}: "
            f"{start_time} - {end_time}"
        )

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        offset=start_time,
        duration=requested_duration,
    )

    if y.size == 0:
        raise ValueError("Audio vacio")

    actual_duration = len(y) / float(sr)
    return y.astype(np.float32, copy=False), sr, actual_duration


# =============================================================================
# ENVOLVENTE Y DETECCION DE EVENTOS
# =============================================================================
def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.size <= 1:
        return values

    window_size = min(window_size, values.size)
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(values, kernel, mode="same")


def compute_envelope(
    y: np.ndarray,
    sr: int,
    segment_start: float,
    smoothing_ms: float,
) -> EnvelopeData:
    """Calcula RMS suavizado y lo expresa en dB relativos al pico."""

    peak_amplitude = float(np.max(np.abs(y))) if y.size else 0.0
    actual_duration = len(y) / float(sr)

    if len(y) < FRAME_LENGTH:
        y = np.pad(y, (0, FRAME_LENGTH - len(y)), mode="constant")

    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        center=False,
    )[0]

    frame_step_ms = 1_000.0 * HOP_LENGTH / sr
    smoothing_frames = max(1, int(round(smoothing_ms / frame_step_ms)))
    rms_smoothed = moving_average(rms, smoothing_frames)

    if rms_smoothed.size == 0 or np.max(rms_smoothed) <= 1e-10:
        rms_db = np.empty(0, dtype=np.float32)
        frame_times = np.empty(0, dtype=np.float32)
    else:
        rms_db = librosa.amplitude_to_db(
            rms_smoothed,
            ref=np.max,
            top_db=80.0,
        ).astype(np.float32)

        frame_times = (
            np.arange(len(rms_db), dtype=np.float32) * HOP_LENGTH
            + FRAME_LENGTH / 2.0
        ) / sr

    return EnvelopeData(
        rms_db=rms_db,
        frame_times=frame_times,
        segment_duration=actual_duration,
        segment_start=float(segment_start),
        peak_amplitude=peak_amplitude,
    )


def find_hysteresis_runs(
    rms_db: np.ndarray,
    on_threshold_db: float,
    off_threshold_db: float,
) -> list[tuple[int, int]]:
    """Devuelve intervalos de frames usando umbrales de entrada/salida."""

    intervals: list[tuple[int, int]] = []
    active = False
    start_idx = 0

    for idx, value in enumerate(rms_db):
        if not active and value >= on_threshold_db:
            active = True
            start_idx = idx
        elif active and value < off_threshold_db:
            intervals.append((start_idx, max(start_idx, idx - 1)))
            active = False

    if active:
        intervals.append((start_idx, len(rms_db) - 1))

    return intervals


def merge_close_runs(
    intervals: list[tuple[int, int]],
    merge_gap_ms: float,
    sr: int = SAMPLE_RATE,
) -> list[tuple[int, int]]:
    if not intervals:
        return []

    max_gap_frames = int(
        round((merge_gap_ms / 1_000.0) * sr / HOP_LENGTH)
    )

    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        gap_frames = current_start - previous_end - 1

        if gap_frames <= max_gap_frames:
            merged[-1] = (previous_start, current_end)
        else:
            merged.append((current_start, current_end))

    return merged


def fixed_window_around_peak(
    peak_time: float,
    segment_duration: float,
    output_window_ms: float,
) -> tuple[float, float, float, float]:
    """Crea una ventana fija y devuelve tambien padding izq./der."""

    window_duration = output_window_ms / 1_000.0

    if segment_duration >= window_duration:
        window_start = peak_time - window_duration / 2.0
        window_start = min(
            max(window_start, 0.0),
            segment_duration - window_duration,
        )
        window_end = window_start + window_duration
        return window_start, window_end, 0.0, 0.0

    padding_total = window_duration - segment_duration
    padding_left = padding_total / 2.0
    padding_right = padding_total - padding_left
    return 0.0, segment_duration, padding_left, padding_right


def detect_events(
    envelope: EnvelopeData,
    config: SegmentationConfig,
) -> list[dict]:
    """Detecta candidatos y devuelve tiempos absolutos y relativos."""

    if envelope.rms_db.size == 0 or envelope.peak_amplitude < 1e-6:
        return []

    noise_floor_db = float(np.median(envelope.rms_db))

    # El umbral debe estar como maximo top_db por debajo del pico, pero
    # tambien al menos snr_margin_db por encima de la mediana de energia.
    on_threshold_db = max(
        -float(config.top_db),
        noise_floor_db + float(config.snr_margin_db),
    )
    on_threshold_db = min(on_threshold_db, -1.0)
    off_threshold_db = on_threshold_db - float(config.hysteresis_db)

    intervals = find_hysteresis_runs(
        envelope.rms_db,
        on_threshold_db=on_threshold_db,
        off_threshold_db=off_threshold_db,
    )
    intervals = merge_close_runs(
        intervals,
        merge_gap_ms=config.merge_gap_ms,
    )

    min_event_seconds = config.min_event_ms / 1_000.0
    max_event_seconds = config.max_event_ms / 1_000.0
    frame_half_width = FRAME_LENGTH / (2.0 * SAMPLE_RATE)

    events: list[dict] = []

    for start_idx, end_idx in intervals:
        event_start = max(
            0.0,
            float(envelope.frame_times[start_idx]) - frame_half_width,
        )
        event_end = min(
            envelope.segment_duration,
            float(envelope.frame_times[end_idx]) + frame_half_width,
        )
        event_duration = event_end - event_start

        if event_duration < min_event_seconds:
            continue

        local_values = envelope.rms_db[start_idx : end_idx + 1]
        peak_idx = start_idx + int(np.argmax(local_values))
        peak_time = float(envelope.frame_times[peak_idx])

        (
            window_start,
            window_end,
            padding_left,
            padding_right,
        ) = fixed_window_around_peak(
            peak_time=peak_time,
            segment_duration=envelope.segment_duration,
            output_window_ms=config.output_window_ms,
        )

        events.append(
            {
                "event_start_relative": event_start,
                "event_end_relative": event_end,
                "event_duration": event_duration,
                "event_peak_relative": peak_time,
                "window_start_relative": window_start,
                "window_end_relative": window_end,
                "window_padding_left": padding_left,
                "window_padding_right": padding_right,
                "event_start": envelope.segment_start + event_start,
                "event_end": envelope.segment_start + event_end,
                "event_peak": envelope.segment_start + peak_time,
                "window_start": envelope.segment_start + window_start,
                "window_end": envelope.segment_start + window_end,
                "event_too_long": event_duration > max_event_seconds,
                "noise_floor_db": noise_floor_db,
                "on_threshold_db": on_threshold_db,
                "off_threshold_db": off_threshold_db,
            }
        )

    return events


# =============================================================================
# MUESTREO Y PRECALCULO
# =============================================================================
def stratified_sample(
    df: pd.DataFrame,
    max_rows: int | None,
) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or max_rows >= len(df):
        return df.reset_index(drop=True)

    if max_rows < df["split_stratum"].nunique():
        raise ValueError(
            "max_rows debe ser al menos igual al numero de estratos."
        )

    _, sampled = train_test_split(
        df,
        test_size=int(max_rows),
        random_state=RANDOM_STATE,
        stratify=df["split_stratum"],
    )
    return sampled.reset_index(drop=True)


def precompute_envelopes(
    df: pd.DataFrame,
    smoothing_ms: float,
) -> tuple[list[tuple[pd.Series, EnvelopeData]], list[dict]]:
    prepared: list[tuple[pd.Series, EnvelopeData]] = []
    errors: list[dict] = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Calculando envolventes",
    ):
        try:
            y, sr, _ = load_audio_segment(row)
            envelope = compute_envelope(
                y=y,
                sr=sr,
                segment_start=float(row["start_time"]),
                smoothing_ms=smoothing_ms,
            )
            prepared.append((row, envelope))
        except Exception as exc:
            errors.append(
                {
                    "uuid_segmento": row["uuid_segmento"],
                    "original_uuid": row["original_uuid"],
                    "error": str(exc),
                }
            )

    return prepared, errors


# =============================================================================
# ANALISIS DE PARAMETROS
# =============================================================================
def parse_float_list(raw_value: str) -> list[float]:
    return [
        float(item.strip())
        for item in raw_value.split(",")
        if item.strip()
    ]


def build_analysis_grid(args: argparse.Namespace) -> list[SegmentationConfig]:
    configs: list[SegmentationConfig] = []

    for top_db in parse_float_list(args.grid_top_db):
        for merge_gap_ms in parse_float_list(args.grid_merge_gap_ms):
            for min_event_ms in parse_float_list(args.grid_min_event_ms):
                configs.append(
                    SegmentationConfig(
                        top_db=top_db,
                        snr_margin_db=args.snr_margin_db,
                        hysteresis_db=args.hysteresis_db,
                        smoothing_ms=args.smoothing_ms,
                        merge_gap_ms=merge_gap_ms,
                        min_event_ms=min_event_ms,
                        max_event_ms=args.max_event_ms,
                        output_window_ms=args.output_window_ms,
                    )
                )

    return configs


def summarize_configuration(
    prepared: list[tuple[pd.Series, EnvelopeData]],
    config: SegmentationConfig,
) -> dict:
    per_segment_rows: list[dict] = []
    event_durations: list[float] = []
    too_long_count = 0

    for row, envelope in prepared:
        events = detect_events(envelope, config)
        stage1_target = int(row["stage1_target"])

        per_segment_rows.append(
            {
                "stage1_target": stage1_target,
                "event_count": len(events),
                "segment_duration": envelope.segment_duration,
            }
        )

        for event in events:
            event_durations.append(float(event["event_duration"]))
            too_long_count += int(event["event_too_long"])

    segment_stats = pd.DataFrame(per_segment_rows)
    cough_stats = segment_stats[segment_stats["stage1_target"] == 1]
    no_cough_stats = segment_stats[segment_stats["stage1_target"] == 0]

    def coverage(group: pd.DataFrame) -> float:
        if group.empty:
            return float("nan")
        return float((group["event_count"] > 0).mean())

    def mean_events(group: pd.DataFrame) -> float:
        if group.empty:
            return float("nan")
        return float(group["event_count"].mean())

    def events_per_minute(group: pd.DataFrame) -> float:
        if group.empty or group["segment_duration"].sum() <= 0:
            return float("nan")
        return float(
            group["event_count"].sum()
            / group["segment_duration"].sum()
            * 60.0
        )

    durations = np.asarray(event_durations, dtype=np.float32)
    total_events = int(segment_stats["event_count"].sum())

    return {
        "config": config.name,
        "top_db": config.top_db,
        "merge_gap_ms": config.merge_gap_ms,
        "min_event_ms": config.min_event_ms,
        "segments": len(segment_stats),
        "total_events": total_events,
        "cough_coverage": coverage(cough_stats),
        "no_cough_activity_coverage": coverage(no_cough_stats),
        "mean_events_cough": mean_events(cough_stats),
        "mean_events_no_cough": mean_events(no_cough_stats),
        "events_minute_cough": events_per_minute(cough_stats),
        "events_minute_no_cough": events_per_minute(no_cough_stats),
        "median_event_duration": (
            float(np.median(durations)) if durations.size else float("nan")
        ),
        "p95_event_duration": (
            float(np.percentile(durations, 95))
            if durations.size
            else float("nan")
        ),
        "too_long_fraction": (
            too_long_count / total_events if total_events else 0.0
        ),
    }


def run_analysis(args: argparse.Namespace) -> None:
    if args.split != "train":
        raise ValueError(
            "La seleccion de parametros debe realizarse solo con --split train."
        )

    df = load_split_metadata("train")
    max_rows = 600 if args.max_rows is None else args.max_rows
    df = stratified_sample(df, max_rows=max_rows)

    configs = build_analysis_grid(args)
    smoothing_values = {config.smoothing_ms for config in configs}
    if len(smoothing_values) != 1:
        raise ValueError(
            "El grid debe compartir smoothing_ms para reutilizar envolventes."
        )

    print(f"Analizando {len(df)} segmentos de TRAIN.")
    print(f"Configuraciones candidatas: {len(configs)}")

    prepared, errors = precompute_envelopes(
        df,
        smoothing_ms=configs[0].smoothing_ms,
    )

    if errors:
        print(f"Aviso: {len(errors)} segmentos no pudieron cargarse.")

    results = pd.DataFrame(
        [
            summarize_configuration(prepared, config)
            for config in tqdm(configs, desc="Evaluando configuraciones")
        ]
    )

    # Prioridad: no perder grabaciones de tos. Entre configuraciones con
    # cobertura similar, se prefiere la que propone menos eventos no-cough.
    results = results.sort_values(
        by=[
            "cough_coverage",
            "events_minute_no_cough",
            "too_long_fraction",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    display_columns = [
        "config",
        "total_events",
        "cough_coverage",
        "no_cough_activity_coverage",
        "mean_events_cough",
        "mean_events_no_cough",
        "events_minute_cough",
        "events_minute_no_cough",
        "median_event_duration",
        "p95_event_duration",
        "too_long_fraction",
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.float_format",
        "{:.4f}".format,
    ):
        print("\n" + "=" * 100)
        print("COMPARACION DE PARAMETROS — SOLO TRAIN")
        print("=" * 100)
        print(results[display_columns].to_string(index=False))

    print("\nInterpretacion:")
    print("- cough_coverage debe ser alta: indica grabaciones de tos con >=1 candidato.")
    print("- eventos no-cough no son automaticamente falsos positivos; son candidatos")
    print("  que posteriormente debe rechazar la etapa 1.")
    print("- una tasa excesiva de candidatos aumenta el coste movil.")
    print("- la configuracion final requiere auditoria manual de eventos.")


# =============================================================================
# EXTRACCION DE MANIFIESTOS
# =============================================================================
def create_event_manifest(
    df: pd.DataFrame,
    prepared: list[tuple[pd.Series, EnvelopeData]],
    config: SegmentationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict] = []
    summary_rows: list[dict] = []

    for row, envelope in tqdm(prepared, desc="Generando manifiesto"):
        events = detect_events(envelope, config)

        summary_rows.append(
            {
                "uuid_segmento": row["uuid_segmento"],
                "original_uuid": row["original_uuid"],
                "split": row["split"],
                "fold": row["fold"],
                "stage1_target": row["stage1_target"],
                "cough_type": row["cough_type"],
                "cough_type_consensus": row["cough_type_consensus"],
                "segment_duration": envelope.segment_duration,
                "peak_amplitude": envelope.peak_amplitude,
                "event_count": len(events),
            }
        )

        for event_idx, event in enumerate(events):
            event_row = row.to_dict()
            event_row.update(event)
            event_row["event_index"] = event_idx
            event_row["event_id"] = (
                f"{row['uuid_segmento']}_event_{event_idx:03d}"
            )
            event_row["segmentation_config"] = config.name
            event_rows.append(event_row)

    return pd.DataFrame(event_rows), pd.DataFrame(summary_rows)


def run_extraction_for_split(
    split_name: str,
    args: argparse.Namespace,
) -> None:
    df = load_split_metadata(split_name)
    df = stratified_sample(df, max_rows=args.max_rows)

    config = SegmentationConfig(
        top_db=args.top_db,
        snr_margin_db=args.snr_margin_db,
        hysteresis_db=args.hysteresis_db,
        smoothing_ms=args.smoothing_ms,
        merge_gap_ms=args.merge_gap_ms,
        min_event_ms=args.min_event_ms,
        max_event_ms=args.max_event_ms,
        output_window_ms=args.output_window_ms,
    )

    print("\n" + "=" * 80)
    print(f"EXTRAYENDO EVENTOS DE {split_name.upper()}")
    print(f"Configuracion: {config.name}")
    print("=" * 80)

    prepared, errors = precompute_envelopes(
        df,
        smoothing_ms=config.smoothing_ms,
    )

    event_manifest, segment_summary = create_event_manifest(
        df=df,
        prepared=prepared,
        config=config,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / f"metadata_events_{split_name}.csv"
    summary_path = OUTPUT_DIR / f"segmentation_summary_{split_name}.csv"
    errors_path = OUTPUT_DIR / f"segmentation_errors_{split_name}.csv"

    event_manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    segment_summary.to_csv(summary_path, index=False, encoding="utf-8")
    pd.DataFrame(errors).to_csv(errors_path, index=False, encoding="utf-8")

    print(f"Segmentos procesados: {len(prepared)}")
    print(f"Eventos generados: {len(event_manifest)}")
    print(f"Errores de lectura: {len(errors)}")
    print(f"Manifiesto: {manifest_path}")
    print(f"Resumen: {summary_path}")


def run_extraction(args: argparse.Namespace) -> None:
    split_names: Iterable[str]
    if args.split == "all":
        split_names = ("train", "val", "test")
    else:
        split_names = (args.split,)

    for split_name in split_names:
        run_extraction_for_split(split_name, args)


# =============================================================================
# MUESTRA CERRADA DE 100 AUDIOS PARA AUDITORIA
# =============================================================================
def build_original_audio_table(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce TRAIN a una fila por audio original, sin duplicar segmentos."""

    audit_required = {"type_noise", "duration"}
    missing_columns = audit_required - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Faltan columnas necesarias para la auditoria: "
            f"{sorted(missing_columns)}"
        )

    working = df.copy()
    working["start_time"] = pd.to_numeric(
        working["start_time"], errors="coerce"
    )
    working["end_time"] = pd.to_numeric(
        working["end_time"], errors="coerce"
    )
    working["duration"] = pd.to_numeric(
        working["duration"], errors="coerce"
    )

    ordered = working.sort_values(
        ["original_uuid", "start_time", "uuid_segmento"]
    )
    originals = ordered.drop_duplicates(
        subset="original_uuid", keep="first"
    ).copy()

    grouped = working.groupby("original_uuid", sort=False)
    max_end = grouped["end_time"].max()
    metadata_duration = grouped["duration"].max()
    segment_count = grouped["uuid_segmento"].nunique()

    originals["original_duration_sec"] = originals["original_uuid"].map(
        pd.concat(
            [max_end.rename("end"), metadata_duration.rename("metadata")],
            axis=1,
        ).max(axis=1)
    )
    originals["train_segment_count"] = originals["original_uuid"].map(
        segment_count
    )
    originals["is_respiratory_sound"] = originals["type_noise"].fillna(
        ""
    ).str.contains("Respiratory_sounds", case=False, regex=False)
    originals["original_audio_path"] = originals.apply(
        lambda row: str(resolve_audio_path(row)), axis=1
    )
    originals["audio_exists"] = originals["original_audio_path"].map(
        lambda path: Path(path).is_file()
    )

    return originals.reset_index(drop=True)


def measure_peak_amplitude(
    audio_path: Path,
    peak_cache: dict[str, float],
) -> float:
    """Calcula el pico del WAV completo en mono sin cargarlo entero en RAM."""

    cache_key = str(audio_path.resolve())
    if cache_key in peak_cache:
        return peak_cache[cache_key]

    peak_amplitude = 0.0
    with sf.SoundFile(str(audio_path)) as audio_file:
        for block in audio_file.blocks(
            blocksize=65_536,
            dtype="float32",
            always_2d=True,
        ):
            mono_block = block.mean(axis=1)
            if mono_block.size:
                peak_amplitude = max(
                    peak_amplitude,
                    float(np.max(np.abs(mono_block))),
                )

    peak_cache[cache_key] = peak_amplitude
    return peak_amplitude


def add_peak_amplitudes(
    candidates: pd.DataFrame,
    peak_cache: dict[str, float],
    description: str,
) -> tuple[pd.DataFrame, list[dict]]:
    measured_rows: list[pd.Series] = []
    errors: list[dict] = []

    for _, row in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc=description,
    ):
        try:
            measured = row.copy()
            measured["peak_amplitude"] = measure_peak_amplitude(
                Path(row["original_audio_path"]), peak_cache
            )
            measured_rows.append(measured)
        except Exception as exc:
            errors.append(
                {
                    "original_uuid": row["original_uuid"],
                    "original_audio_path": row["original_audio_path"],
                    "error": str(exc),
                }
            )

    if not measured_rows:
        return pd.DataFrame(columns=list(candidates.columns) + [
            "peak_amplitude"
        ]), errors

    return pd.DataFrame(measured_rows).reset_index(drop=True), errors


def require_exact_candidates(
    candidates: pd.DataFrame,
    amount: int,
    subgroup: str,
) -> None:
    if len(candidates) < amount:
        raise ValueError(
            f"No hay suficientes audios para {subgroup}: "
            f"se necesitan {amount} y solo hay {len(candidates)}."
        )


def tag_audit_group(
    selected: pd.DataFrame,
    category: str,
    subgroup: str,
    reason: str,
) -> pd.DataFrame:
    tagged = selected.copy()
    tagged["audit_category"] = category
    tagged["audit_subgroup"] = subgroup
    tagged["selection_reason"] = reason
    return tagged


def select_random_audios(
    candidates: pd.DataFrame,
    amount: int,
    category: str,
    subgroup: str,
    reason: str,
    random_offset: int,
) -> pd.DataFrame:
    require_exact_candidates(candidates, amount, subgroup)
    selected = candidates.sample(
        n=amount,
        replace=False,
        random_state=RANDOM_STATE + random_offset,
    )
    return tag_audit_group(selected, category, subgroup, reason)


def select_lowest_amplitude_audios(
    candidates: pd.DataFrame,
    amount: int,
    category: str,
    subgroup: str,
    reason: str,
    peak_cache: dict[str, float],
) -> tuple[pd.DataFrame, list[dict]]:
    measured, errors = add_peak_amplitudes(
        candidates,
        peak_cache=peak_cache,
        description=f"Midiendo amplitud: {subgroup}",
    )
    require_exact_candidates(measured, amount, subgroup)
    selected = measured.sort_values(
        ["peak_amplitude", "original_uuid"], ascending=[True, True]
    ).head(amount)
    return (
        tag_audit_group(selected, category, subgroup, reason),
        errors,
    )


def select_audit_sample(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """Selecciona exactamente los 100 originales acordados para auditar."""

    originals = build_original_audio_table(train_df)
    originals = originals[originals["audio_exists"]].copy()
    peak_cache: dict[str, float] = {}
    scan_errors: list[dict] = []
    selected_groups: list[pd.DataFrame] = []

    def available(mask: pd.Series) -> pd.DataFrame:
        selected_ids = {
            str(value)
            for group in selected_groups
            for value in group["original_uuid"].tolist()
        }
        return originals[
            mask & ~originals["original_uuid"].astype(str).isin(selected_ids)
        ].copy()

    random_specs = [
        (
            (originals["cough_type"] == "dry")
            & (originals["cough_type_consensus"] == "gold_expert"),
            8,
            "dry",
            "dry_gold",
            "Tos dry con consenso gold_expert",
        ),
        (
            (originals["cough_type"] == "dry")
            & (originals["cough_type_consensus"] == "weak_expert"),
            16,
            "dry",
            "dry_weak",
            "Tos dry con consenso weak_expert",
        ),
        (
            (originals["cough_type"] == "wet")
            & (originals["cough_type_consensus"] == "gold_expert"),
            6,
            "wet",
            "wet_gold",
            "Tos wet con consenso gold_expert",
        ),
        (
            (originals["cough_type"] == "wet")
            & (originals["cough_type_consensus"] == "weak_expert"),
            18,
            "wet",
            "wet_weak",
            "Tos wet con consenso weak_expert",
        ),
        (
            (originals["cough_type"] == "unknown")
            & (originals["cough_type_consensus"] == "gold_expert"),
            2,
            "unknown",
            "unknown_gold",
            "Tos unknown con consenso gold_expert",
        ),
        (
            (originals["cough_type"] == "unknown")
            & (originals["cough_type_consensus"] == "weak_expert"),
            3,
            "unknown",
            "unknown_weak",
            "Tos unknown con consenso weak_expert",
        ),
        (
            (originals["cough_type"] == "unknown")
            & (originals["cough_type_consensus"] == "ambiguous_expert"),
            3,
            "unknown",
            "unknown_ambiguous",
            "Tos unknown con consenso ambiguous_expert",
        ),
    ]

    for random_offset, spec in enumerate(random_specs):
        mask, amount, category, subgroup, reason = spec
        selected_groups.append(
            select_random_audios(
                available(mask),
                amount=amount,
                category=category,
                subgroup=subgroup,
                reason=reason,
                random_offset=random_offset,
            )
        )

    coughvid_no_cough_mask = (
        (originals["dataset_origin"] == "COUGHVID")
        & (pd.to_numeric(originals["stage1_target"]) == 0)
    )
    selected, errors = select_lowest_amplitude_audios(
        available(coughvid_no_cough_mask),
        amount=22,
        category="coughvid_no_cough",
        subgroup="coughvid_no_cough_low_amplitude",
        reason="No tos CoughVID entre las 22 amplitudes mas bajas",
        peak_cache=peak_cache,
    )
    selected_groups.append(selected)
    scan_errors.extend(errors)

    fsd_no_cough_mask = (
        (originals["dataset_origin"] == "FSD50K")
        & (pd.to_numeric(originals["stage1_target"]) == 0)
    )
    fsd_respiratory_mask = (
        fsd_no_cough_mask & originals["is_respiratory_sound"]
    )
    selected, errors = select_lowest_amplitude_audios(
        available(fsd_respiratory_mask),
        amount=11,
        category="fsd_no_cough",
        subgroup="fsd_respiratory_low_amplitude",
        reason=(
            "No tos FSD50K con Respiratory_sounds entre las 11 "
            "amplitudes mas bajas"
        ),
        peak_cache=peak_cache,
    )
    selected_groups.append(selected)
    scan_errors.extend(errors)

    fsd_non_respiratory_mask = (
        fsd_no_cough_mask & ~originals["is_respiratory_sound"]
    )
    fsd_short_mask = (
        fsd_non_respiratory_mask
        & (originals["original_duration_sec"] < 1.0)
    )
    selected_groups.append(
        select_random_audios(
            available(fsd_short_mask),
            amount=5,
            category="fsd_no_cough",
            subgroup="fsd_non_respiratory_short",
            reason="No tos FSD50K no respiratorio y menor de 1 segundo",
            random_offset=len(random_specs),
        )
    )

    selected, errors = select_lowest_amplitude_audios(
        available(fsd_no_cough_mask),
        amount=3,
        category="fsd_no_cough",
        subgroup="fsd_low_amplitude",
        reason=(
            "No tos FSD50K entre las 3 amplitudes mas bajas restantes"
        ),
        peak_cache=peak_cache,
    )
    selected_groups.append(selected)
    scan_errors.extend(errors)

    long_candidates = available(fsd_no_cough_mask).sort_values(
        ["original_duration_sec", "original_uuid"],
        ascending=[False, True],
    )
    require_exact_candidates(long_candidates, 3, "fsd_long")
    selected_groups.append(
        tag_audit_group(
            long_candidates.head(3),
            category="fsd_no_cough",
            subgroup="fsd_long",
            reason=(
                "No tos FSD50K entre los 3 audios mas largos restantes; "
                "no se calcula el numero de eventos para seleccionarlos"
            ),
        )
    )

    selection = pd.concat(selected_groups, ignore_index=True)

    if len(selection) != 100:
        raise AssertionError(
            f"La muestra debe contener 100 audios y contiene {len(selection)}."
        )
    if selection["original_uuid"].duplicated().any():
        duplicates = selection.loc[
            selection["original_uuid"].duplicated(keep=False),
            "original_uuid",
        ].tolist()
        raise AssertionError(
            f"La muestra contiene originales duplicados: {duplicates}"
        )

    subgroup_counts = selection["audit_subgroup"].value_counts().to_dict()
    category_counts = selection["audit_category"].value_counts().to_dict()
    if subgroup_counts != AUDIT_EXPECTED_SUBGROUPS:
        raise AssertionError(
            "Las cuotas de subgrupos no coinciden: "
            f"esperado={AUDIT_EXPECTED_SUBGROUPS}, real={subgroup_counts}"
        )
    if category_counts != AUDIT_EXPECTED_CATEGORIES:
        raise AssertionError(
            "Las cuotas principales no coinciden: "
            f"esperado={AUDIT_EXPECTED_CATEGORIES}, real={category_counts}"
        )

    measured_selection, errors = add_peak_amplitudes(
        selection,
        peak_cache=peak_cache,
        description="Completando amplitudes de la muestra",
    )
    scan_errors.extend(errors)
    if len(measured_selection) != 100:
        failed_ids = sorted(
            set(selection["original_uuid"])
            - set(measured_selection["original_uuid"])
        )
        raise RuntimeError(
            "No se pudieron medir todos los audios seleccionados: "
            f"{failed_ids}"
        )

    selection = measured_selection.reset_index(drop=True)
    selection.insert(
        0,
        "audit_id",
        [f"AUD{idx:03d}" for idx in range(1, len(selection) + 1)],
    )
    selection["peak_amplitude_dbfs"] = np.where(
        selection["peak_amplitude"] > 0,
        20.0 * np.log10(selection["peak_amplitude"]),
        -np.inf,
    )

    preferred_columns = [
        "audit_id",
        "audit_category",
        "audit_subgroup",
        "selection_reason",
        "original_uuid",
        "dataset_origin",
        "cough_type",
        "cough_type_consensus",
        "type_noise",
        "stage1_target",
        "original_duration_sec",
        "train_segment_count",
        "peak_amplitude",
        "peak_amplitude_dbfs",
        "original_audio_path",
    ]
    remaining_columns = [
        column
        for column in selection.columns
        if column not in preferred_columns
    ]
    return selection[preferred_columns + remaining_columns], scan_errors


def extract_fixed_event_window(
    y: np.ndarray,
    sr: int,
    event: dict,
    output_window_ms: float,
) -> np.ndarray:
    """Extrae y rellena una ventana con exactamente la longitud pedida."""

    target_samples = int(round(output_window_ms / 1_000.0 * sr))
    start_sample = int(round(float(event["window_start_relative"]) * sr))
    end_sample = int(round(float(event["window_end_relative"]) * sr))
    start_sample = min(max(start_sample, 0), len(y))
    end_sample = min(max(end_sample, start_sample), len(y))
    window = y[start_sample:end_sample]

    if len(window) > target_samples:
        return window[:target_samples].astype(np.float32, copy=False)

    missing_samples = target_samples - len(window)
    requested_left = int(
        round(float(event["window_padding_left"]) * sr)
    )
    padding_left = min(max(requested_left, 0), missing_samples)
    padding_right = missing_samples - padding_left

    return np.pad(
        window,
        (padding_left, padding_right),
        mode="constant",
    ).astype(np.float32, copy=False)


def run_audit(args: argparse.Namespace) -> None:
    """Selecciona 100 originales de TRAIN y crea los WAV para escucharlos."""

    if args.split != "train":
        raise ValueError("La auditoria solo puede construirse desde TRAIN.")

    train_df = load_split_metadata("train")
    config = SegmentationConfig(
        top_db=args.top_db,
        snr_margin_db=args.snr_margin_db,
        hysteresis_db=args.hysteresis_db,
        smoothing_ms=args.smoothing_ms,
        merge_gap_ms=args.merge_gap_ms,
        min_event_ms=args.min_event_ms,
        max_event_ms=args.max_event_ms,
        output_window_ms=args.output_window_ms,
    )

    print("Seleccionando exactamente 100 audios originales de TRAIN...")
    selection, selection_errors = select_audit_sample(train_df)

    originals_dir = AUDIT_DIR / "originals"
    segmented_dir = AUDIT_DIR / "segmented" / config.name
    originals_dir.mkdir(parents=True, exist_ok=True)
    segmented_dir.mkdir(parents=True, exist_ok=True)

    selection_path = AUDIT_DIR / "audit_selection_100.csv"
    selection.to_csv(selection_path, index=False, encoding="utf-8-sig")

    event_rows: list[dict] = []
    summary_rows: list[dict] = []
    generation_errors: list[dict] = list(selection_errors)

    for _, selected_audio in tqdm(
        selection.iterrows(),
        total=len(selection),
        desc="Generando audios de auditoria",
    ):
        audit_id = str(selected_audio["audit_id"])
        original_uuid = str(selected_audio["original_uuid"])
        source_path = Path(selected_audio["original_audio_path"])
        original_filename = f"{audit_id}__{original_uuid}.wav"
        copied_original_path = originals_dir / original_filename

        try:
            shutil.copy2(source_path, copied_original_path)
        except Exception as exc:
            generation_errors.append(
                {
                    "original_uuid": original_uuid,
                    "original_audio_path": str(source_path),
                    "error": f"No se pudo copiar el original: {exc}",
                }
            )
            continue

        audio_segment_rows = train_df[
            train_df["original_uuid"].astype(str) == original_uuid
        ].sort_values(["start_time", "uuid_segmento"])
        original_event_index = 0

        for _, segment_row in audio_segment_rows.iterrows():
            try:
                y, sr, _ = load_audio_segment(segment_row)
                envelope = compute_envelope(
                    y=y,
                    sr=sr,
                    segment_start=float(segment_row["start_time"]),
                    smoothing_ms=config.smoothing_ms,
                )
                events = detect_events(envelope, config)

                summary_rows.append(
                    {
                        "audit_id": audit_id,
                        "audit_category": selected_audio["audit_category"],
                        "audit_subgroup": selected_audio["audit_subgroup"],
                        "original_uuid": original_uuid,
                        "uuid_segmento": segment_row["uuid_segmento"],
                        "segment_start": segment_row["start_time"],
                        "segment_end": segment_row["end_time"],
                        "segment_duration": envelope.segment_duration,
                        "event_count": len(events),
                        "segmentation_config": config.name,
                    }
                )

                event_audio_dir = segmented_dir / (
                    f"{audit_id}__{original_uuid}"
                )
                event_audio_dir.mkdir(parents=True, exist_ok=True)

                for segment_event_index, event in enumerate(events):
                    event_id = (
                        f"{audit_id}__{original_uuid}__"
                        f"event_{original_event_index:03d}"
                    )
                    event_audio = extract_fixed_event_window(
                        y=y,
                        sr=sr,
                        event=event,
                        output_window_ms=config.output_window_ms,
                    )
                    event_path = event_audio_dir / f"{event_id}.wav"
                    sf.write(
                        str(event_path),
                        event_audio,
                        sr,
                        subtype="PCM_16",
                    )

                    event_row = {
                        "event_id": event_id,
                        "audit_id": audit_id,
                        "audit_category": selected_audio["audit_category"],
                        "audit_subgroup": selected_audio["audit_subgroup"],
                        "original_uuid": original_uuid,
                        "uuid_segmento": segment_row["uuid_segmento"],
                        "segment_event_index": segment_event_index,
                        "original_event_index": original_event_index,
                        "original_audio_copy": str(
                            copied_original_path.relative_to(AUDIT_DIR)
                        ),
                        "event_audio": str(event_path.relative_to(AUDIT_DIR)),
                        "segmentation_config": config.name,
                    }
                    event_row.update(event)
                    event_rows.append(event_row)
                    original_event_index += 1
            except Exception as exc:
                generation_errors.append(
                    {
                        "original_uuid": original_uuid,
                        "uuid_segmento": segment_row["uuid_segmento"],
                        "original_audio_path": str(source_path),
                        "error": str(exc),
                    }
                )

    event_manifest = pd.DataFrame(event_rows)
    segment_summary = pd.DataFrame(summary_rows)
    errors_df = pd.DataFrame(
        generation_errors,
        columns=[
            "original_uuid",
            "uuid_segmento",
            "original_audio_path",
            "error",
        ],
    )

    manifest_path = AUDIT_DIR / f"audit_events__{config.name}.csv"
    summary_path = AUDIT_DIR / f"audit_summary__{config.name}.csv"
    errors_path = AUDIT_DIR / f"audit_errors__{config.name}.csv"
    review_path = AUDIT_DIR / f"audit_review__{config.name}.csv"

    event_manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    segment_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    errors_df.to_csv(errors_path, index=False, encoding="utf-8-sig")

    review_columns = [
        "event_id",
        "audit_id",
        "audit_category",
        "audit_subgroup",
        "original_uuid",
        "original_audio_copy",
        "event_audio",
    ]
    if event_manifest.empty:
        review = pd.DataFrame(columns=review_columns)
    else:
        review = event_manifest[review_columns].copy()
    review["relevant_event_missed"] = ""
    review["event_split"] = ""
    review["events_merged"] = ""
    review["only_noise"] = ""
    review["window_truncated"] = ""
    review["acceptable"] = ""
    review["notes"] = ""
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    print("\nMuestra de auditoria creada.")
    print(f"Audios originales seleccionados: {len(selection)}")
    print(f"Eventos WAV generados: {len(event_manifest)}")
    print(f"Errores: {len(errors_df)}")
    print(f"Configuracion: {config.name}")
    print(f"Carpeta: {AUDIT_DIR}")
    print(f"Seleccion: {selection_path}")
    print(f"Plantilla de revision: {review_path}")


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza parametros, genera manifiestos o crea la auditoria."
        )
    )

    parser.add_argument(
        "--action",
        choices=["analyze", "extract", "audit"],
        default="analyze",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="train",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Limita filas de forma estratificada. En analyze el default "
            "es 600; en extract, None procesa el split completo."
        ),
    )

    parser.add_argument("--top-db", type=float, default=25.0)
    parser.add_argument("--snr-margin-db", type=float, default=3.0)
    parser.add_argument("--hysteresis-db", type=float, default=3.0)
    parser.add_argument("--smoothing-ms", type=float, default=30.0)
    parser.add_argument("--merge-gap-ms", type=float, default=200.0)
    parser.add_argument("--min-event-ms", type=float, default=80.0)
    parser.add_argument("--max-event-ms", type=float, default=2_000.0)
    parser.add_argument("--output-window-ms", type=float, default=1_500.0)

    parser.add_argument(
        "--grid-top-db",
        default="20,25,30",
        help="Valores separados por coma para analyze.",
    )
    parser.add_argument(
        "--grid-merge-gap-ms",
        default="150,250",
        help="Valores separados por coma para analyze.",
    )
    parser.add_argument(
        "--grid-min-event-ms",
        default="100,150",
        help="Valores separados por coma para analyze.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.action == "analyze":
        run_analysis(args)
    elif args.action == "extract":
        run_extraction(args)
    else:
        run_audit(args)


if __name__ == "__main__":
    main()
