"""Genera manifests de eventos para la clasificación dry/wet.

No crea nuevos splits, no procesa test y no convierte los eventos en
muestras independientes. Los eventos se agruparán posteriormente por
original_uuid durante la extracción de cochleogramas.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from segmentation_audio_events import (
    FRAME_LENGTH,
    HOP_LENGTH,
    SAMPLE_RATE,
    SegmentationConfig,
    compute_envelope,
    find_hysteresis_runs,
    fixed_window_around_peak,
    load_audio_segment,
    resolve_audio_path,
)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import argparse

# =============================================================================
# RUTAS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

METADATA_DIR = (
    SCRIPT_DIR / "metadata_stage2_dry_wet_experiment_random"
)

OUTPUT_DIR = (
    SCRIPT_DIR / "metadata_audio_events_stage2_dry_wet"
)

INPUT_FILES = {
    "train": "metadata_train_stage2_dry_wet.csv",
    "validation": "metadata_validation_stage2_dry_wet.csv",
}

GRAPHS_DIR = (
    SCRIPT_DIR / "graphs_results_stage2_dry_wet_random"
)

N_GRAPH_EXAMPLES = 5
GRAPH_RANDOM_STATE = 42
N_AUDIT_EXAMPLES_PER_GROUP = 5
N_LOW_AMPLITUDE_PER_GROUP = 2
AUDIT_SILENCE_BETWEEN_EVENTS_MS = 250.0

# Colores alternos para que los solapamientos entre ventanas se distingan.
WINDOW_FACE_COLORS = ("#2a9d8f", "#8e5ea2")
WINDOW_EDGE_COLORS = ("#1b6f64", "#5f3b73")
COMPARISON_SEGMENTATION_METHOD = (
    "energy_hysteresis_bounded_merge_v2"
)
FINAL_SEGMENTATION_METHOD = (
    "energy_hysteresis_bounded_merge_nonoverlap_v3"
)

# =============================================================================
# CONFIGURACIONES ELEGIDAS Y CANDIDATAS DE LA AUDITORÍA
# =============================================================================
BASELINE_SEGMENTATION_CONFIG = SegmentationConfig(
    top_db=25.0,
    snr_margin_db=3.0,
    hysteresis_db=3.0,
    smoothing_ms=30.0,
    merge_gap_ms=200.0,
    min_event_ms=80.0,
    max_event_ms=2_000.0,
    output_window_ms=1_500.0,
)

# Configuración final elegida mediante auditoría visual y auditiva:
# sensibilidad strict y merge_gap_ms=500.
SEGMENTATION_CONFIG = SegmentationConfig(
    top_db=15.0,
    snr_margin_db=6.0,
    hysteresis_db=3.0,
    smoothing_ms=30.0,
    merge_gap_ms=500.0,
    min_event_ms=80.0,
    max_event_ms=2_000.0,
    output_window_ms=1_500.0,
)

# Candidatos para elegir la sensibilidad del detector. Se mantiene
# merge_gap_ms constante para no confundir el efecto del umbral con el
# efecto de fusionar eventos cercanos.
SENSITIVITY_CONFIGS = {
    "baseline": BASELINE_SEGMENTATION_CONFIG,
    "moderate": SegmentationConfig(
        top_db=20.0,
        snr_margin_db=5.0,
        hysteresis_db=3.0,
        smoothing_ms=30.0,
        merge_gap_ms=200.0,
        min_event_ms=80.0,
        max_event_ms=2_000.0,
        output_window_ms=1_500.0,
    ),
    "strict": SegmentationConfig(
        top_db=15.0,
        snr_margin_db=6.0,
        hysteresis_db=3.0,
        smoothing_ms=30.0,
        merge_gap_ms=200.0,
        min_event_ms=80.0,
        max_event_ms=2_000.0,
        output_window_ms=1_500.0,
    ),
}

OVERLAP_GAPS_MS = (200.0, 350.0, 500.0)


REQUIRED_COLUMNS = {
    "uuid_segmento",
    "original_uuid",
    "dataset_origin",
    "cough_type",
    "cough_type_consensus",
    "stage2_eligible",
    "stage2_target",
    "fold",
    "split",
    "start_time",
    "end_time",
}


# =============================================================================
# VALIDACIÓN DE METADATOS
# =============================================================================
def parse_boolean_column(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """Convierte valores bool/string/0-1 en booleanos reales."""

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

    invalid_values = sorted(
        set(normalized.unique()) - set(mapping)
    )

    if invalid_values:
        raise ValueError(
            f"Valores no reconocidos en {column_name}: "
            f"{invalid_values}"
        )

    return normalized.map(mapping).astype(bool)


def load_stage2_metadata(
    split_name: str,
) -> pd.DataFrame:
    """Carga y valida una vista dry/wet ya derivada."""

    csv_path = METADATA_DIR / INPUT_FILES[split_name]

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No se encuentra el metadata de {split_name}: "
            f"{csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        dtype={
            "uuid": str,
            "original_uuid": str,
            "uuid_segmento": str,
        },
    )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en {csv_path.name}: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            f"El metadata dry/wet de {split_name} está vacío."
        )

    if df["uuid_segmento"].duplicated().any():
        duplicated = df.loc[
            df["uuid_segmento"].duplicated(keep=False),
            "uuid_segmento",
        ].head(20).tolist()

        raise ValueError(
            f"uuid_segmento duplicados: {duplicated}"
        )

    df = df.copy()

    df["stage2_eligible"] = parse_boolean_column(
        df["stage2_eligible"],
        "stage2_eligible",
    )

    if not df["stage2_eligible"].all():
        raise ValueError(
            f"El metadata {split_name} contiene filas no elegibles."
        )

    if set(df["cough_type"].unique()) != {"dry", "wet"}:
        raise ValueError(
            f"{split_name} no contiene exactamente dry/wet: "
            f"{sorted(df['cough_type'].unique())}"
        )

    df["stage2_target"] = pd.to_numeric(
        df["stage2_target"],
        errors="raise",
    ).astype(int)

    if set(df["stage2_target"].unique()) != {0, 1}:
        raise ValueError(
            f"stage2_target inválido en {split_name}."
        )

    expected_target = df["cough_type"].map(
        {"dry": 0, "wet": 1}
    ).astype(int)

    if not df["stage2_target"].equals(expected_target):
        raise ValueError(
            f"stage2_target no coincide con dry/wet en {split_name}."
        )

    df["fold"] = pd.to_numeric(
        df["fold"],
        errors="raise",
    ).astype(int)

    df["start_time"] = pd.to_numeric(
        df["start_time"],
        errors="raise",
    )

    df["end_time"] = pd.to_numeric(
        df["end_time"],
        errors="raise",
    )

    if (df["end_time"] <= df["start_time"]).any():
        raise ValueError(
            f"Hay intervalos inválidos en {split_name}."
        )

    expected_split_value = split_name

    if set(df["split"].astype(str).unique()) != {
        expected_split_value
    }:
        raise ValueError(
            f"La columna split no coincide con {split_name}."
        )

    if split_name == "train":
        if set(df["fold"].unique()) != {0, 1, 2, 3, 4}:
            raise ValueError(
                "Train no contiene exactamente los folds 0-4."
            )
    elif not (df["fold"] == -1).all():
        raise ValueError(
            "Validation debe tener fold=-1."
        )

    # Todos los segmentos de un original deben compartir etiqueta y fold.
    consistency_columns = [
        "cough_type",
        "stage2_target",
        "cough_type_consensus",
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
        bad_uuids = inconsistent[
            inconsistent
        ].index[:20].tolist()

        raise ValueError(
            "Originales con metadatos inconsistentes: "
            f"{bad_uuids}"
        )

    return df.sort_values(
        ["fold", "stage2_target", "original_uuid", "start_time"]
    ).reset_index(drop=True)


def validate_no_overlap(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    """Comprueba que train y validation no comparten originales."""

    train_uuids = set(
        train_df["original_uuid"].astype(str)
    )
    validation_uuids = set(
        validation_df["original_uuid"].astype(str)
    )

    overlap = train_uuids & validation_uuids

    if overlap:
        raise ValueError(
            "Hay original_uuid compartidos entre train y "
            f"validation: {sorted(overlap)[:20]}"
        )


# =============================================================================
# DETECTOR STAGE 2 CON FUSION LIMITADA POR LA VENTANA DE SALIDA
# =============================================================================
def frame_interval_times(
    envelope,
    start_index: int,
    end_index: int,
) -> tuple[float, float]:
    """Convierte un intervalo de frames en tiempos relativos."""

    frame_half_width = FRAME_LENGTH / (2.0 * SAMPLE_RATE)
    event_start = max(
        0.0,
        float(envelope.frame_times[start_index]) - frame_half_width,
    )
    event_end = min(
        envelope.segment_duration,
        float(envelope.frame_times[end_index]) + frame_half_width,
    )
    return event_start, event_end


def merge_close_runs_bounded(
    intervals: list[tuple[int, int]],
    envelope,
    config: SegmentationConfig,
) -> list[tuple[int, int, int]]:
    """Fusiona runs cercanos solo si el resultado cabe en la ventana."""

    if not intervals:
        return []

    maximum_gap_frames = int(
        round(
            (config.merge_gap_ms / 1_000.0)
            * SAMPLE_RATE
            / HOP_LENGTH
        )
    )
    maximum_merged_duration = (
        config.output_window_ms / 1_000.0
    )
    merged: list[tuple[int, int, int]] = [
        (intervals[0][0], intervals[0][1], 1)
    ]

    for current_start, current_end in intervals[1:]:
        previous_start, previous_end, run_count = merged[-1]
        gap_frames = current_start - previous_end - 1
        proposed_start, proposed_end = frame_interval_times(
            envelope,
            previous_start,
            current_end,
        )
        proposed_duration = proposed_end - proposed_start

        if (
            gap_frames <= maximum_gap_frames
            and proposed_duration
            <= maximum_merged_duration + 1e-9
        ):
            merged[-1] = (
                previous_start,
                current_end,
                run_count + 1,
            )
        else:
            merged.append((current_start, current_end, 1))

    return merged


def fixed_window_covering_event(
    event_start: float,
    event_end: float,
    peak_time: float,
    segment_duration: float,
    output_window_ms: float,
) -> tuple[float, float, float, float]:
    """Crea una ventana fija que contiene todo el evento si cabe."""

    window_duration = output_window_ms / 1_000.0

    if (
        segment_duration < window_duration
        or event_end - event_start > window_duration + 1e-9
    ):
        return fixed_window_around_peak(
            peak_time=peak_time,
            segment_duration=segment_duration,
            output_window_ms=output_window_ms,
        )

    centered_start = peak_time - window_duration / 2.0
    minimum_start = max(0.0, event_end - window_duration)
    maximum_start = min(
        event_start,
        segment_duration - window_duration,
    )
    window_start = min(
        max(centered_start, minimum_start),
        maximum_start,
    )
    window_end = window_start + window_duration
    return window_start, window_end, 0.0, 0.0


def remove_window_overlaps(
    events: list[dict],
    target_window_ms: float,
) -> list[dict]:
    """Reparte el contexto entre eventos sin reutilizar audio real.

    La frontera se sitúa en el silencio entre dos regiones activas. Las
    ventanas resultantes pueden contener menos de 1,5 s de audio, pero el
    tiempo retirado queda registrado como padding para rellenarlo con ceros
    durante la extracción de características.
    """

    if not events:
        return []

    target_duration = target_window_ms / 1_000.0
    adjusted_events = [dict(event) for event in events]
    adjusted_events.sort(key=lambda event: event["event_peak_relative"])

    for event in adjusted_events:
        event["window_start_before_overlap"] = event[
            "window_start_relative"
        ]
        event["window_end_before_overlap"] = event[
            "window_end_relative"
        ]
        event["overlap_trimmed_left"] = 0.0
        event["overlap_trimmed_right"] = 0.0
        event["non_overlap_adjusted"] = False
        event["target_window_duration"] = target_duration

    for left_event, right_event in zip(
        adjusted_events,
        adjusted_events[1:],
    ):
        left_window_end = float(
            left_event["window_end_relative"]
        )
        right_window_start = float(
            right_event["window_start_relative"]
        )

        if left_window_end <= right_window_start + 1e-12:
            continue

        overlap_start = right_window_start
        overlap_end = left_window_end
        preferred_boundary = (
            float(left_event["event_end_relative"])
            + float(right_event["event_start_relative"])
        ) / 2.0
        boundary = min(
            max(preferred_boundary, overlap_start),
            overlap_end,
        )

        left_trim = max(0.0, left_window_end - boundary)
        right_trim = max(0.0, boundary - right_window_start)

        left_event["window_end_relative"] = boundary
        left_event["window_end"] = (
            float(left_event["event_start"])
            - float(left_event["event_start_relative"])
            + boundary
        )
        left_event["window_padding_right"] = (
            float(left_event["window_padding_right"])
            + left_trim
        )
        left_event["overlap_trimmed_right"] += left_trim
        left_event["non_overlap_adjusted"] = True

        right_event["window_start_relative"] = boundary
        right_event["window_start"] = (
            float(right_event["event_start"])
            - float(right_event["event_start_relative"])
            + boundary
        )
        right_event["window_padding_left"] = (
            float(right_event["window_padding_left"])
            + right_trim
        )
        right_event["overlap_trimmed_left"] += right_trim
        right_event["non_overlap_adjusted"] = True

    for event in adjusted_events:
        observed_duration = (
            float(event["window_end_relative"])
            - float(event["window_start_relative"])
        )
        event["window_observed_duration"] = observed_duration
        event["window_total_padding"] = (
            float(event["window_padding_left"])
            + float(event["window_padding_right"])
        )

        if observed_duration <= 0.0:
            raise RuntimeError(
                "La eliminación de solapamiento creó una ventana vacía."
            )

        reconstructed_duration = (
            observed_duration + event["window_total_padding"]
        )
        if not np.isclose(
            reconstructed_duration,
            target_duration,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Duración observada + padding no coincide con la "
                f"ventana objetivo: {reconstructed_duration:.9f} s."
            )

    for left_event, right_event in zip(
        adjusted_events,
        adjusted_events[1:],
    ):
        if (
            float(left_event["window_end_relative"])
            > float(right_event["window_start_relative"]) + 1e-9
        ):
            raise RuntimeError(
                "Persistió un solapamiento después del ajuste."
            )

    return adjusted_events


def validate_non_overlapping_manifest(
    event_manifest: pd.DataFrame,
    split_name: str,
) -> None:
    """Impide guardar un manifest final con audio reutilizado."""

    if event_manifest.empty:
        return

    overlap_errors: list[str] = []

    for uuid_segmento, group in event_manifest.groupby(
        "uuid_segmento"
    ):
        ordered = group.sort_values("window_start")
        previous_end = None

        for _, event in ordered.iterrows():
            window_start = float(event["window_start"])
            window_end = float(event["window_end"])

            if window_end <= window_start:
                overlap_errors.append(
                    f"{uuid_segmento}: ventana vacía o invertida"
                )
                break

            if (
                previous_end is not None
                and previous_end > window_start + 1e-9
            ):
                overlap_errors.append(
                    f"{uuid_segmento}: {previous_end:.9f} > "
                    f"{window_start:.9f}"
                )
                break

            previous_end = window_end

    observed = pd.to_numeric(
        event_manifest["window_observed_duration"],
        errors="raise",
    )
    padding = pd.to_numeric(
        event_manifest["window_total_padding"],
        errors="raise",
    )
    target = pd.to_numeric(
        event_manifest["target_window_duration"],
        errors="raise",
    )
    invalid_duration = ~np.isclose(
        observed + padding,
        target,
        atol=1e-6,
    )

    if overlap_errors or invalid_duration.any():
        raise RuntimeError(
            f"Manifest final inválido en {split_name}. "
            f"Solapamientos: {overlap_errors[:10]}; "
            "ventanas con duración/padding inválidos: "
            f"{int(invalid_duration.sum())}."
        )


def detect_stage2_events(
    envelope,
    config: SegmentationConfig,
    enforce_non_overlap: bool = False,
) -> list[dict]:
    """Detecta eventos sin fusionar una region mayor que la salida."""

    if envelope.rms_db.size == 0 or envelope.peak_amplitude < 1e-6:
        return []

    noise_floor_db = float(np.median(envelope.rms_db))
    on_threshold_db = max(
        -float(config.top_db),
        noise_floor_db + float(config.snr_margin_db),
    )
    on_threshold_db = min(on_threshold_db, -1.0)
    off_threshold_db = (
        on_threshold_db - float(config.hysteresis_db)
    )
    intervals = find_hysteresis_runs(
        envelope.rms_db,
        on_threshold_db=on_threshold_db,
        off_threshold_db=off_threshold_db,
    )
    intervals = merge_close_runs_bounded(
        intervals=intervals,
        envelope=envelope,
        config=config,
    )

    minimum_event_duration = config.min_event_ms / 1_000.0
    maximum_event_duration = config.max_event_ms / 1_000.0
    output_window_duration = config.output_window_ms / 1_000.0
    events: list[dict] = []

    for start_index, end_index, merged_run_count in intervals:
        event_start, event_end = frame_interval_times(
            envelope,
            start_index,
            end_index,
        )
        event_duration = event_end - event_start

        if event_duration < minimum_event_duration:
            continue

        local_values = envelope.rms_db[
            start_index : end_index + 1
        ]
        peak_index = start_index + int(np.argmax(local_values))
        peak_time = float(envelope.frame_times[peak_index])
        (
            window_start,
            window_end,
            padding_left,
            padding_right,
        ) = fixed_window_covering_event(
            event_start=event_start,
            event_end=event_end,
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
                "event_too_long": (
                    event_duration > maximum_event_duration
                ),
                "event_exceeds_output_window": (
                    event_duration > output_window_duration + 1e-9
                ),
                "merged_active_run_count": merged_run_count,
                "merge_span_limit_ms": config.output_window_ms,
                "noise_floor_db": noise_floor_db,
                "on_threshold_db": on_threshold_db,
                "off_threshold_db": off_threshold_db,
            }
        )

    if enforce_non_overlap:
        return remove_window_overlaps(
            events=events,
            target_window_ms=config.output_window_ms,
        )

    for event in events:
        event["window_start_before_overlap"] = event[
            "window_start_relative"
        ]
        event["window_end_before_overlap"] = event[
            "window_end_relative"
        ]
        event["overlap_trimmed_left"] = 0.0
        event["overlap_trimmed_right"] = 0.0
        event["non_overlap_adjusted"] = False
        event["target_window_duration"] = (
            config.output_window_ms / 1_000.0
        )
        event["window_observed_duration"] = (
            float(event["window_end_relative"])
            - float(event["window_start_relative"])
        )
        event["window_total_padding"] = (
            float(event["window_padding_left"])
            + float(event["window_padding_right"])
        )

    return events


# =============================================================================
# AUDITORIA COMPARATIVA DE PARAMETROS (SOLO TRAIN)
# =============================================================================
def build_comparison_configs(
    comparison_phase: str,
    overlap_base: str,
) -> dict[str, SegmentationConfig]:
    """Construye un barrido controlado de un unico tipo de parametro."""

    if comparison_phase == "sensitivity":
        return dict(SENSITIVITY_CONFIGS)

    if comparison_phase == "final":
        return {"final": SEGMENTATION_CONFIG}

    if overlap_base not in SENSITIVITY_CONFIGS:
        raise ValueError(
            f"Configuracion base no reconocida: {overlap_base}. "
            f"Opciones: {sorted(SENSITIVITY_CONFIGS)}"
        )

    base_config = SENSITIVITY_CONFIGS[overlap_base]

    return {
        f"gap_{int(gap_ms)}": replace(
            base_config,
            merge_gap_ms=gap_ms,
        )
        for gap_ms in OVERLAP_GAPS_MS
    }


def get_original_train_candidates(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """Obtiene una fila por original y anade su amplitud maxima."""

    candidates = (
        train_df.sort_values(
            ["original_uuid", "start_time", "uuid_segmento"]
        )
        .drop_duplicates("original_uuid", keep="first")
        .reset_index(drop=True)
        .copy()
    )

    # La amplitud depende del audio, no de los parámetros del detector.
    # Reutilizar cualquier resumen completo evita releer todos los WAV.
    summary_path = (
        OUTPUT_DIR
        / "metadata_segment_summary_train_stage2_dry_wet.csv"
    )

    if summary_path.is_file():
        summary = pd.read_csv(
            summary_path,
            dtype={"uuid_segmento": str},
        )

        required = {
            "uuid_segmento",
            "peak_amplitude",
        }

        if required.issubset(summary.columns):
            amplitude_table = summary[
                ["uuid_segmento", "peak_amplitude"]
            ].copy()
            amplitude_table["peak_amplitude"] = pd.to_numeric(
                amplitude_table["peak_amplitude"],
                errors="coerce",
            )

            candidates = candidates.merge(
                amplitude_table,
                on="uuid_segmento",
                how="left",
                validate="one_to_one",
            )

            if candidates["peak_amplitude"].notna().all():
                print(
                    "Amplitudes cargadas del resumen existente."
                )
                return candidates

            candidates = candidates.drop(
                columns="peak_amplitude"
            )

    print(
        "No hay un resumen baseline compatible. "
        "Calculando amplitudes de train..."
    )

    peak_amplitudes: list[float] = []

    for _, row in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc="Calculando amplitudes",
    ):
        y, _, _ = load_audio_segment(row)
        peak_amplitudes.append(
            float(np.max(np.abs(y))) if y.size else 0.0
        )

    candidates["peak_amplitude"] = peak_amplitudes
    return candidates


def select_parameter_audit_examples(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """Selecciona 20 originales: cinco por clase y consenso."""

    candidates = get_original_train_candidates(train_df)
    group_specs = [
        ("dry", "gold_expert"),
        ("dry", "weak_expert"),
        ("wet", "gold_expert"),
        ("wet", "weak_expert"),
    ]
    selected_groups: list[pd.DataFrame] = []

    for group_index, (cough_type, consensus) in enumerate(
        group_specs
    ):
        group = candidates[
            (candidates["cough_type"] == cough_type)
            & (
                candidates["cough_type_consensus"]
                == consensus
            )
        ].copy()

        if len(group) < N_AUDIT_EXAMPLES_PER_GROUP:
            raise ValueError(
                "No hay suficientes audios para auditar "
                f"{cough_type}/{consensus}: {len(group)}."
            )

        group = group.sort_values(
            ["peak_amplitude", "original_uuid"]
        )
        low_amplitude = group.head(
            N_LOW_AMPLITUDE_PER_GROUP
        ).copy()
        low_amplitude["audit_selection_reason"] = (
            "low_amplitude"
        )

        remaining = group.drop(index=low_amplitude.index)
        random_count = (
            N_AUDIT_EXAMPLES_PER_GROUP
            - N_LOW_AMPLITUDE_PER_GROUP
        )
        random_examples = remaining.sample(
            n=random_count,
            replace=False,
            random_state=GRAPH_RANDOM_STATE + group_index,
        ).copy()
        random_examples["audit_selection_reason"] = (
            "random"
        )

        selected = pd.concat(
            [low_amplitude, random_examples],
            ignore_index=True,
        )
        selected["audit_group"] = (
            f"{cough_type}_{consensus.replace('_expert', '')}"
        )
        selected_groups.append(selected)

    examples = pd.concat(
        selected_groups,
        ignore_index=True,
    )
    examples.insert(
        0,
        "audit_id",
        [f"PAR{i:03d}" for i in range(1, len(examples) + 1)],
    )

    if examples["original_uuid"].duplicated().any():
        raise RuntimeError(
            "La auditoria de parametros contiene originales duplicados."
        )

    return examples


def detect_audit_events(
    examples: pd.DataFrame,
    configurations: dict[str, SegmentationConfig],
    enforce_non_overlap: bool,
    segmentation_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica todas las configuraciones a los mismos 20 audios."""

    event_rows: list[dict] = []
    summary_rows: list[dict] = []

    for _, row in tqdm(
        examples.iterrows(),
        total=len(examples),
        desc="Comparando configuraciones",
    ):
        y, sr, _ = load_audio_segment(row)
        envelopes = {}

        for config_key, config in configurations.items():
            if config.smoothing_ms not in envelopes:
                envelopes[config.smoothing_ms] = compute_envelope(
                    y=y,
                    sr=sr,
                    segment_start=float(row["start_time"]),
                    smoothing_ms=config.smoothing_ms,
                )

            envelope = envelopes[config.smoothing_ms]
            detected_events = detect_stage2_events(
                envelope,
                config,
                enforce_non_overlap=enforce_non_overlap,
            )
            fallback_used = len(detected_events) == 0

            if fallback_used:
                output_events = [
                    create_fallback_event(envelope, config)
                ]
            else:
                output_events = []
                for event in detected_events:
                    event_copy = dict(event)
                    event_copy["detected_by_energy"] = True
                    event_copy["fallback_no_event"] = False
                    output_events.append(event_copy)

            summary_rows.append(
                {
                    "audit_id": row["audit_id"],
                    "audit_group": row["audit_group"],
                    "original_uuid": row["original_uuid"],
                    "uuid_segmento": row["uuid_segmento"],
                    "cough_type": row["cough_type"],
                    "cough_type_consensus": (
                        row["cough_type_consensus"]
                    ),
                    "peak_amplitude": row["peak_amplitude"],
                    "config_key": config_key,
                    "segmentation_config": config.name,
                    "segmentation_method": segmentation_method,
                    "enforce_non_overlap": enforce_non_overlap,
                    "detected_event_count": len(detected_events),
                    "output_event_count": len(output_events),
                    "fallback_used": fallback_used,
                }
            )

            for event_index, event in enumerate(output_events):
                event_row = {
                    "audit_id": row["audit_id"],
                    "audit_group": row["audit_group"],
                    "original_uuid": row["original_uuid"],
                    "uuid_segmento": row["uuid_segmento"],
                    "cough_type": row["cough_type"],
                    "cough_type_consensus": (
                        row["cough_type_consensus"]
                    ),
                    "config_key": config_key,
                    "segmentation_config": config.name,
                    "segmentation_method": segmentation_method,
                    "enforce_non_overlap": enforce_non_overlap,
                    "event_index": event_index,
                    "event_id": (
                        f"{row['uuid_segmento']}__{config_key}"
                        f"__event_{event_index:03d}"
                    ),
                    **event,
                }
                event_rows.append(event_row)

    return pd.DataFrame(event_rows), pd.DataFrame(summary_rows)


def calculate_adjacent_overlap_metrics(
    events: pd.DataFrame,
) -> dict[str, float | int]:
    """Resume el solapamiento de ventanas consecutivas por segmento."""

    adjacent_pairs = 0
    overlapping_pairs = 0
    overlap_50_pairs = 0
    overlap_75_pairs = 0
    segments_any_overlap = 0
    segments_overlap_50 = 0

    for _, group in events.groupby("uuid_segmento"):
        ordered = group.sort_values("window_start")
        records = ordered.to_dict("records")
        has_overlap = False
        has_overlap_50 = False

        for first, second in zip(records, records[1:]):
            adjacent_pairs += 1
            intersection = max(
                0.0,
                min(
                    float(first["window_end"]),
                    float(second["window_end"]),
                )
                - max(
                    float(first["window_start"]),
                    float(second["window_start"]),
                ),
            )

            if intersection <= 0.0:
                continue

            overlapping_pairs += 1
            has_overlap = True
            shorter_length = min(
                float(first["window_end"])
                - float(first["window_start"]),
                float(second["window_end"])
                - float(second["window_start"]),
            )
            overlap_ratio = (
                intersection / shorter_length
                if shorter_length > 0.0
                else 0.0
            )

            if overlap_ratio >= 0.50:
                overlap_50_pairs += 1
                has_overlap_50 = True
            if overlap_ratio >= 0.75:
                overlap_75_pairs += 1

        segments_any_overlap += int(has_overlap)
        segments_overlap_50 += int(has_overlap_50)

    segment_count = events["uuid_segmento"].nunique()

    def percentage(numerator: int, denominator: int) -> float:
        return (
            100.0 * numerator / denominator
            if denominator
            else 0.0
        )

    return {
        "adjacent_pair_count": adjacent_pairs,
        "overlapping_pair_count": overlapping_pairs,
        "overlapping_pair_pct": percentage(
            overlapping_pairs, adjacent_pairs
        ),
        "overlap_50_pair_count": overlap_50_pairs,
        "overlap_50_pair_pct": percentage(
            overlap_50_pairs, adjacent_pairs
        ),
        "overlap_75_pair_count": overlap_75_pairs,
        "overlap_75_pair_pct": percentage(
            overlap_75_pairs, adjacent_pairs
        ),
        "segments_any_overlap": segments_any_overlap,
        "segments_any_overlap_pct": percentage(
            segments_any_overlap, segment_count
        ),
        "segments_overlap_50": segments_overlap_50,
        "segments_overlap_50_pct": percentage(
            segments_overlap_50, segment_count
        ),
    }


def summarize_parameter_audit(
    events: pd.DataFrame,
    segment_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula metricas globales y por clase/consenso."""

    def summarize_group(
        summary_group: pd.DataFrame,
        event_group: pd.DataFrame,
    ) -> dict:
        event_counts = summary_group["output_event_count"]
        return {
            "audio_count": len(summary_group),
            "event_count": int(event_counts.sum()),
            "merged_event_count": int(
                (event_group["merged_active_run_count"] > 1).sum()
            ),
            "event_exceeds_output_window_count": int(
                event_group["event_exceeds_output_window"].sum()
            ),
            "non_overlap_adjusted_event_count": int(
                event_group["non_overlap_adjusted"].sum()
            ),
            "events_per_audio_mean": float(event_counts.mean()),
            "events_per_audio_median": float(event_counts.median()),
            "events_per_audio_p95": float(
                event_counts.quantile(0.95)
            ),
            "fallback_count": int(
                summary_group["fallback_used"].sum()
            ),
            "fallback_pct": float(
                100.0 * summary_group["fallback_used"].mean()
            ),
            **calculate_adjacent_overlap_metrics(event_group),
        }

    overall_rows: list[dict] = []
    group_rows: list[dict] = []

    for config_key, summary_group in segment_summary.groupby(
        "config_key", sort=False
    ):
        event_group = events[events["config_key"] == config_key]
        overall_rows.append(
            {
                "config_key": config_key,
                "segmentation_config": summary_group[
                    "segmentation_config"
                ].iloc[0],
                **summarize_group(summary_group, event_group),
            }
        )

        for audit_group, subgroup in summary_group.groupby(
            "audit_group", sort=False
        ):
            subgroup_events = event_group[
                event_group["audit_group"] == audit_group
            ]
            group_rows.append(
                {
                    "config_key": config_key,
                    "segmentation_config": subgroup[
                        "segmentation_config"
                    ].iloc[0],
                    "audit_group": audit_group,
                    **summarize_group(subgroup, subgroup_events),
                }
            )

    return pd.DataFrame(overall_rows), pd.DataFrame(group_rows)


def create_parameter_comparison_graphs(
    examples: pd.DataFrame,
    events: pd.DataFrame,
    configurations: dict[str, SegmentationConfig],
    comparison_name: str,
) -> list[Path]:
    """Crea cuatro figuras 5x3 con los mismos audios por columna."""

    graph_dir = (
        GRAPHS_DIR / "parameter_audit" / comparison_name
    )
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_paths: list[Path] = []
    config_items = list(configurations.items())
    batch_count = int(
        np.ceil(len(examples) / N_GRAPH_EXAMPLES)
    )

    for graph_batch in range(batch_count):
        start = graph_batch * N_GRAPH_EXAMPLES
        end = start + N_GRAPH_EXAMPLES
        batch_examples = examples.iloc[start:end]

        figure, axes = plt.subplots(
            nrows=len(batch_examples),
            ncols=len(config_items),
            figsize=(7 * len(config_items), 3.5 * len(batch_examples)),
            sharex=False,
            squeeze=False,
        )

        for row_index, (_, row) in enumerate(
            batch_examples.iterrows()
        ):
            y, sr, _ = load_audio_segment(row)
            segment_start = float(row["start_time"])
            time_axis = (
                segment_start
                + np.arange(len(y), dtype=np.float32) / sr
            )
            maximum_amplitude = float(np.max(np.abs(y)))
            if maximum_amplitude < 1e-8:
                maximum_amplitude = 1.0

            for column_index, (config_key, config) in enumerate(
                config_items
            ):
                axis = axes[row_index, column_index]
                segment_events = events[
                    (events["config_key"] == config_key)
                    & (
                        events["uuid_segmento"].astype(str)
                        == str(row["uuid_segmento"])
                    )
                ].sort_values("event_index")

                axis.plot(
                    time_axis,
                    y,
                    color="#264653",
                    linewidth=0.60,
                    alpha=0.90,
                )
                axis.set_ylim(
                    -1.20 * maximum_amplitude,
                    1.20 * maximum_amplitude,
                )

                for _, event in segment_events.iterrows():
                    event_index = int(event["event_index"])
                    color_index = event_index % len(
                        WINDOW_FACE_COLORS
                    )
                    axis.axvspan(
                        float(event["window_start"]),
                        float(event["window_end"]),
                        facecolor=WINDOW_FACE_COLORS[color_index],
                        edgecolor=WINDOW_EDGE_COLORS[color_index],
                        linewidth=1.5,
                        alpha=0.28,
                        zorder=1,
                    )

                    if bool(event["detected_by_energy"]):
                        axis.axvspan(
                            float(event["event_start"]),
                            float(event["event_end"]),
                            facecolor="#f4a261",
                            edgecolor="#d97724",
                            linewidth=1.0,
                            alpha=0.35,
                            zorder=2,
                        )

                    event_peak = float(event["event_peak"])
                    axis.axvline(
                        event_peak,
                        color="#d62828",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.95,
                        zorder=3,
                    )
                    axis.text(
                        event_peak,
                        0.92 * maximum_amplitude,
                        f"W{event_index}",
                        color="#a41313",
                        fontsize=7,
                        rotation=90,
                        horizontalalignment="right",
                        verticalalignment="top",
                    )

                fallback_used = bool(
                    segment_events["fallback_no_event"].any()
                )
                title = (
                    f"{row['audit_id']} | {row['audit_group']} | "
                    f"eventos={len(segment_events)}"
                )
                if fallback_used:
                    title += " | FALLBACK"

                axis.set_title(title, fontsize=9, loc="left")
                axis.set_xlabel("Time (s)")
                axis.grid(True, alpha=0.20, linewidth=0.5)

                if column_index == 0:
                    axis.set_ylabel(
                        f"{str(row['original_uuid'])[:12]}...\nAmplitude"
                    )

                if row_index == 0:
                    axis.text(
                        0.5,
                        1.22,
                        (
                            f"{config_key}: top={config.top_db:g}, "
                            f"margin={config.snr_margin_db:g}, "
                            f"gap={config.merge_gap_ms:g} ms"
                        ),
                        transform=axis.transAxes,
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        fontsize=11,
                        fontweight="bold",
                    )

        legend_elements = [
            Line2D(
                [0], [0], color="#264653", linewidth=1.2,
                label="Señal de audio",
            ),
            Patch(
                facecolor="#f4a261", edgecolor="#d97724",
                alpha=0.35, label="Evento activo detectado",
            ),
            Patch(
                facecolor=WINDOW_FACE_COLORS[0],
                edgecolor=WINDOW_EDGE_COLORS[0],
                alpha=0.28,
                label="Ventana final par (W0, W2...)",
            ),
            Patch(
                facecolor=WINDOW_FACE_COLORS[1],
                edgecolor=WINDOW_EDGE_COLORS[1],
                alpha=0.28,
                label="Ventana final impar (W1, W3...)",
            ),
            Line2D(
                [0], [0], color="#d62828", linestyle="--",
                linewidth=1.2, label="Pico de energía",
            ),
        ]
        figure.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.5, 0.995),
        )
        figure.suptitle(
            f"Stage 2 parameter comparison: {comparison_name}",
            fontsize=15,
            y=1.015,
        )
        figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])

        graph_path = graph_dir / (
            f"segmentation_comparison_{comparison_name}"
            f"__batch_{graph_batch:03d}.png"
        )
        figure.savefig(
            graph_path,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)
        graph_paths.append(graph_path)

    return graph_paths


def export_parameter_audit_audio(
    examples: pd.DataFrame,
    events: pd.DataFrame,
    configurations: dict[str, SegmentationConfig],
    comparison_name: str,
) -> Path:
    """Exporta originales y secuencias auditables de ventanas por config."""

    audio_dir = (
        OUTPUT_DIR
        / "parameter_audit"
        / comparison_name
        / "audio"
    )
    originals_dir = audio_dir / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)

    for config_key in configurations:
        (audio_dir / config_key).mkdir(
            parents=True,
            exist_ok=True,
        )

    for _, row in tqdm(
        examples.iterrows(),
        total=len(examples),
        desc="Exportando audios de auditoria",
    ):
        y, sr, _ = load_audio_segment(row)
        audit_id = str(row["audit_id"])
        original_uuid = str(row["original_uuid"])
        file_stem = f"{audit_id}__{original_uuid}"

        # Es el segmento de entrada completo y conserva su amplitud real.
        sf.write(
            originals_dir / f"{file_stem}__original.wav",
            y,
            sr,
            subtype="PCM_16",
        )

        silence_samples = int(
            round(sr * AUDIT_SILENCE_BETWEEN_EVENTS_MS / 1_000.0)
        )
        silence = np.zeros(silence_samples, dtype=np.float32)
        segment_start = float(row["start_time"])

        for config_key in configurations:
            segment_events = events[
                (events["config_key"] == config_key)
                & (
                    events["uuid_segmento"].astype(str)
                    == str(row["uuid_segmento"])
                )
            ].sort_values("event_index")
            clips: list[np.ndarray] = []

            for _, event in segment_events.iterrows():
                relative_start = (
                    float(event["window_start"]) - segment_start
                )
                relative_end = (
                    float(event["window_end"]) - segment_start
                )
                start_sample = max(
                    0,
                    int(round(relative_start * sr)),
                )
                end_sample = min(
                    len(y),
                    int(round(relative_end * sr)),
                )

                if end_sample > start_sample:
                    clips.append(
                        np.asarray(
                            y[start_sample:end_sample],
                            dtype=np.float32,
                        )
                    )

            if not clips:
                raise RuntimeError(
                    "No hay ventanas de audio para "
                    f"{audit_id}/{config_key}."
                )

            sequence_parts: list[np.ndarray] = []
            for clip_index, clip in enumerate(clips):
                if clip_index > 0:
                    sequence_parts.append(silence)
                sequence_parts.append(clip)

            audition_sequence = np.concatenate(sequence_parts)
            sf.write(
                (
                    audio_dir
                    / config_key
                    / f"{file_stem}__selected_windows.wav"
                ),
                audition_sequence,
                sr,
                subtype="PCM_16",
            )

    return audio_dir


def save_parameter_audit_results(
    examples: pd.DataFrame,
    events: pd.DataFrame,
    segment_summary: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    group_metrics: pd.DataFrame,
    configurations: dict[str, SegmentationConfig],
    comparison_name: str,
    segmentation_method: str,
    enforce_non_overlap: bool,
) -> Path:
    """Guarda la auditoria sin sobrescribir los manifests principales."""

    audit_dir = (
        OUTPUT_DIR / "parameter_audit" / comparison_name
    )
    audit_dir.mkdir(parents=True, exist_ok=True)

    examples.to_csv(
        audit_dir / "audit_selection_20.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall_metrics.to_csv(
        audit_dir / "audit_metrics_by_config.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_metrics.to_csv(
        audit_dir / "audit_metrics_by_config_and_group.csv",
        index=False,
        encoding="utf-8-sig",
    )

    configuration_rows = []
    for config_key, config in configurations.items():
        configuration_rows.append(
            {
                "config_key": config_key,
                "segmentation_config": config.name,
                "segmentation_method": segmentation_method,
                "enforce_non_overlap": enforce_non_overlap,
                **asdict(config),
            }
        )

        config_dir = audit_dir / config_key
        config_dir.mkdir(parents=True, exist_ok=True)
        events[events["config_key"] == config_key].to_csv(
            config_dir / "audit_events_20.csv",
            index=False,
            encoding="utf-8-sig",
        )
        segment_summary[
            segment_summary["config_key"] == config_key
        ].to_csv(
            config_dir / "audit_segment_summary_20.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(configuration_rows).to_csv(
        audit_dir / "audit_configurations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return audit_dir


def run_parameter_audit(
    train_df: pd.DataFrame,
    comparison_phase: str,
    overlap_base: str,
) -> None:
    """Ejecuta el barrido solamente sobre 20 audios de train."""

    configurations = build_comparison_configs(
        comparison_phase=comparison_phase,
        overlap_base=overlap_base,
    )
    if comparison_phase == "sensitivity":
        comparison_name = "sensitivity"
    elif comparison_phase == "overlap":
        comparison_name = f"overlap_{overlap_base}"
    else:
        comparison_name = "final_nonoverlap"

    enforce_non_overlap = comparison_phase == "final"
    segmentation_method = (
        FINAL_SEGMENTATION_METHOD
        if enforce_non_overlap
        else COMPARISON_SEGMENTATION_METHOD
    )

    print("=" * 72)
    print("AUDITORIA DE PARAMETROS STAGE 2 DRY/WET")
    print("=" * 72)
    print(f"Fase: {comparison_name}")
    print("Se leeran solamente 20 audios de TRAIN.")
    print("VALIDATION y TEST no seran leidos ni procesados.")

    examples = select_parameter_audit_examples(train_df)
    events, segment_summary = detect_audit_events(
        examples=examples,
        configurations=configurations,
        enforce_non_overlap=enforce_non_overlap,
        segmentation_method=segmentation_method,
    )
    overall_metrics, group_metrics = summarize_parameter_audit(
        events=events,
        segment_summary=segment_summary,
    )
    audit_dir = save_parameter_audit_results(
        examples=examples,
        events=events,
        segment_summary=segment_summary,
        overall_metrics=overall_metrics,
        group_metrics=group_metrics,
        configurations=configurations,
        comparison_name=comparison_name,
        segmentation_method=segmentation_method,
        enforce_non_overlap=enforce_non_overlap,
    )
    graph_paths = create_parameter_comparison_graphs(
        examples=examples,
        events=events,
        configurations=configurations,
        comparison_name=comparison_name,
    )
    audio_dir = export_parameter_audit_audio(
        examples=examples,
        events=events,
        configurations=configurations,
        comparison_name=comparison_name,
    )

    display_columns = [
        "config_key",
        "event_count",
        "events_per_audio_mean",
        "fallback_count",
        "event_exceeds_output_window_count",
        "non_overlap_adjusted_event_count",
        "overlapping_pair_pct",
        "overlap_50_pair_pct",
    ]
    print("\nResumen global de los 20 audios:")
    print(
        overall_metrics[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )
    print(f"\nResultados de auditoria: {audit_dir}")
    print("Graficas comparativas:")
    for graph_path in graph_paths:
        print(f"  {graph_path}")
    print(f"Audios para escucha comparativa: {audio_dir}")


# =============================================================================
# FALLBACK PARA SEGMENTOS SIN EVENTOS
# =============================================================================
def create_fallback_event(
    envelope,
    config: SegmentationConfig,
) -> dict:
    """Crea una ventana alrededor del máximo si no hubo detecciones.

    Este evento no se considera una tos confirmada. Solo evita eliminar
    completamente una grabación del dataset.
    """

    if envelope.rms_db.size > 0:
        peak_index = int(np.argmax(envelope.rms_db))
        peak_time = float(
            envelope.frame_times[peak_index]
        )
        noise_floor_db = float(
            np.median(envelope.rms_db)
        )
    else:
        peak_time = envelope.segment_duration / 2.0
        noise_floor_db = float("nan")

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

    event_duration = window_end - window_start

    return {
        "event_start_relative": window_start,
        "event_end_relative": window_end,
        "event_duration": event_duration,
        "event_peak_relative": peak_time,
        "window_start_relative": window_start,
        "window_end_relative": window_end,
        "window_padding_left": padding_left,
        "window_padding_right": padding_right,
        "window_start_before_overlap": window_start,
        "window_end_before_overlap": window_end,
        "overlap_trimmed_left": 0.0,
        "overlap_trimmed_right": 0.0,
        "non_overlap_adjusted": False,
        "target_window_duration": (
            config.output_window_ms / 1_000.0
        ),
        "window_observed_duration": window_end - window_start,
        "window_total_padding": padding_left + padding_right,
        "event_start": (
            envelope.segment_start + window_start
        ),
        "event_end": (
            envelope.segment_start + window_end
        ),
        "event_peak": (
            envelope.segment_start + peak_time
        ),
        "window_start": (
            envelope.segment_start + window_start
        ),
        "window_end": (
            envelope.segment_start + window_end
        ),
        "event_too_long": False,
        "event_exceeds_output_window": False,
        "merged_active_run_count": 0,
        "merge_span_limit_ms": config.output_window_ms,
        "noise_floor_db": noise_floor_db,
        "on_threshold_db": float("nan"),
        "off_threshold_db": float("nan"),
        "detected_by_energy": False,
        "fallback_no_event": True,
    }


# =============================================================================
# GENERACIÓN DEL MANIFEST
# =============================================================================
def process_split(
    split_name: str,
    df: pd.DataFrame,
) -> None:
    """Detecta eventos y guarda manifests, sin crear WAV."""

    event_rows: list[dict] = []
    segment_summary_rows: list[dict] = []
    errors: list[dict] = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Segmentando {split_name}",
    ):
        try:
            y, sr, _ = load_audio_segment(row)

            envelope = compute_envelope(
                y=y,
                sr=sr,
                segment_start=float(row["start_time"]),
                smoothing_ms=SEGMENTATION_CONFIG.smoothing_ms,
            )

            detected_events = detect_stage2_events(
                envelope,
                SEGMENTATION_CONFIG,
                enforce_non_overlap=True,
            )

            fallback_used = len(detected_events) == 0

            if fallback_used:
                output_events = [
                    create_fallback_event(
                        envelope,
                        SEGMENTATION_CONFIG,
                    )
                ]
            else:
                output_events = []

                for event in detected_events:
                    event_copy = dict(event)
                    event_copy["detected_by_energy"] = True
                    event_copy["fallback_no_event"] = False
                    output_events.append(event_copy)

            segment_summary_rows.append(
                {
                    "uuid_segmento": row["uuid_segmento"],
                    "original_uuid": row["original_uuid"],
                    "split": row["split"],
                    "fold": row["fold"],
                    "cough_type": row["cough_type"],
                    "stage2_target": row["stage2_target"],
                    "cough_type_consensus": (
                        row["cough_type_consensus"]
                    ),
                    "segment_duration": (
                        envelope.segment_duration
                    ),
                    "peak_amplitude": (
                        envelope.peak_amplitude
                    ),
                    "detected_event_count": (
                        len(detected_events)
                    ),
                    "output_event_count": (
                        len(output_events)
                    ),
                    "fallback_used": fallback_used,
                    "segmentation_config": (
                        SEGMENTATION_CONFIG.name
                    ),
                    "segmentation_method": FINAL_SEGMENTATION_METHOD,
                    "enforce_non_overlap": True,
                }
            )

            for event_index, event in enumerate(
                output_events
            ):
                event_id = (
                    f"{row['uuid_segmento']}"
                    f"__event_{event_index:03d}"
                )

                event_row = row.to_dict()
                event_row.update(event)

                event_row["event_id"] = event_id
                event_row["event_index"] = event_index
                event_row["source_audio_path"] = str(
                    resolve_audio_path(row)
                )
                event_row["segmentation_config"] = (
                    SEGMENTATION_CONFIG.name
                )
                event_row["segmentation_method"] = (
                    FINAL_SEGMENTATION_METHOD
                )
                event_row["enforce_non_overlap"] = True

                event_rows.append(event_row)

        except Exception as exc:
            errors.append(
                {
                    "split": split_name,
                    "uuid_segmento": row["uuid_segmento"],
                    "original_uuid": row["original_uuid"],
                    "error": str(exc),
                }
            )

    event_manifest = pd.DataFrame(event_rows)
    segment_summary = pd.DataFrame(
        segment_summary_rows
    )

    validate_non_overlapping_manifest(
        event_manifest=event_manifest,
        split_name=split_name,
    )

    if segment_summary.empty:
        original_summary = pd.DataFrame()
    else:
        original_summary = (
            segment_summary.groupby(
                "original_uuid",
                as_index=False,
            )
            .agg(
                split=("split", "first"),
                fold=("fold", "first"),
                cough_type=("cough_type", "first"),
                stage2_target=(
                    "stage2_target",
                    "first",
                ),
                cough_type_consensus=(
                    "cough_type_consensus",
                    "first",
                ),
                segment_count=(
                    "uuid_segmento",
                    "nunique",
                ),
                detected_event_count=(
                    "detected_event_count",
                    "sum",
                ),
                output_event_count=(
                    "output_event_count",
                    "sum",
                ),
                fallback_used=(
                    "fallback_used",
                    "max",
                ),
                total_segment_duration=(
                    "segment_duration",
                    "sum",
                ),
            )
        )

    errors_df = pd.DataFrame(
        errors,
        columns=[
            "split",
            "uuid_segmento",
            "original_uuid",
            "error",
        ],
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    events_path = (
        OUTPUT_DIR
        / f"metadata_events_{split_name}_stage2_dry_wet.csv"
    )

    segment_summary_path = (
        OUTPUT_DIR
        / f"metadata_segment_summary_{split_name}_stage2_dry_wet.csv"
    )

    original_summary_path = (
        OUTPUT_DIR
        / f"metadata_original_summary_{split_name}_stage2_dry_wet.csv"
    )

    errors_path = (
        OUTPUT_DIR
        / f"metadata_errors_{split_name}_stage2_dry_wet.csv"
    )

    event_manifest.to_csv(
        events_path,
        index=False,
        encoding="utf-8-sig",
    )

    segment_summary.to_csv(
        segment_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    original_summary.to_csv(
        original_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    errors_df.to_csv(
        errors_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print(f"RESULTADO {split_name.upper()}")
    print("=" * 72)
    print(f"Segmentos de entrada: {len(df)}")
    print(
        "Originales de entrada: "
        f"{df['original_uuid'].nunique()}"
    )
    print(f"Eventos detectados: {len(event_manifest)}")
    print(
        "Originales con fallback: "
        f"{int(original_summary['fallback_used'].sum())}"
    )
    print(f"Errores: {len(errors_df)}")
    print(f"Manifest: {events_path}")

    # No permitimos continuar silenciosamente con audios perdidos.
    if not errors_df.empty:
        raise RuntimeError(
            f"Se produjeron {len(errors_df)} errores en "
            f"{split_name}. Revisa {errors_path}."
        )

    expected_originals = df[
        "original_uuid"
    ].nunique()

    generated_originals = original_summary[
        "original_uuid"
    ].nunique()

    if generated_originals != expected_originals:
        raise RuntimeError(
            f"Se esperaban {expected_originals} originales "
            f"en {split_name}, pero se generaron "
            f"{generated_originals}."
        )

    if (original_summary["output_event_count"] < 1).any():
        raise RuntimeError(
            f"Hay originales de {split_name} sin eventos "
            "de salida incluso después del fallback."
        )
    
    return (
    event_manifest,
    segment_summary,
    original_summary,
)

# =============================================================================
# VISUALIZACIÓN DE EJEMPLOS DE TRAIN
# =============================================================================
def select_train_graph_examples(
    train_df: pd.DataFrame,
    graph_batch: int,
) -> pd.DataFrame:
    """Selecciona bloques no solapados de 3 dry y 2 wet."""

    if graph_batch < 0:
        raise ValueError(
            "graph_batch debe ser mayor o igual que cero."
        )

    # Una fila representativa por original_uuid.
    original_candidates = (
        train_df.sort_values(
            ["original_uuid", "start_time", "uuid_segmento"]
        )
        .drop_duplicates(
            subset="original_uuid",
            keep="first",
        )
        .reset_index(drop=True)
    )

    dry_candidates = original_candidates[
        original_candidates["cough_type"] == "dry"
    ].copy()

    wet_candidates = original_candidates[
        original_candidates["cough_type"] == "wet"
    ].copy()

    # Crear un único orden aleatorio reproducible.
    dry_order = dry_candidates.sample(
        frac=1.0,
        replace=False,
        random_state=GRAPH_RANDOM_STATE,
    ).reset_index(drop=True)

    wet_order = wet_candidates.sample(
        frac=1.0,
        replace=False,
        random_state=GRAPH_RANDOM_STATE,
    ).reset_index(drop=True)

    # Cada batch consume 3 dry y 2 wet.
    dry_start = graph_batch * 3
    dry_end = dry_start + 3

    wet_start = graph_batch * 2
    wet_end = wet_start + 2

    if dry_end > len(dry_order):
        raise ValueError(
            f"No hay suficientes dry para graph_batch={graph_batch}. "
            f"Se necesitan índices hasta {dry_end - 1} y solamente "
            f"hay {len(dry_order)} audios."
        )

    if wet_end > len(wet_order):
        raise ValueError(
            f"No hay suficientes wet para graph_batch={graph_batch}. "
            f"Se necesitan índices hasta {wet_end - 1} y solamente "
            f"hay {len(wet_order)} audios."
        )

    dry_examples = dry_order.iloc[
        dry_start:dry_end
    ].reset_index(drop=True)

    wet_examples = wet_order.iloc[
        wet_start:wet_end
    ].reset_index(drop=True)

    # Orden: dry, wet, dry, wet, dry.
    selected_rows = [
        dry_examples.iloc[0],
        wet_examples.iloc[0],
        dry_examples.iloc[1],
        wet_examples.iloc[1],
        dry_examples.iloc[2],
    ]

    examples = pd.DataFrame(
        selected_rows
    ).reset_index(drop=True)

    if examples["original_uuid"].duplicated().any():
        raise RuntimeError(
            "El bloque gráfico contiene originales duplicados."
        )

    return examples

def create_train_segmentation_graph(
    train_df: pd.DataFrame,
    train_event_manifest: pd.DataFrame,
    graph_batch: int,
) -> Path:
    """Genera un subplot 5x1 con eventos, ventanas y picos."""

    if train_event_manifest.empty:
        raise ValueError(
            "El manifest de eventos de train está vacío."
        )

    examples = select_train_graph_examples(
        train_df=train_df,
        graph_batch=graph_batch,
    )

    GRAPHS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        nrows=N_GRAPH_EXAMPLES,
        ncols=1,
        figsize=(16, 18),
        sharex=False,
    )

    for axis, (_, row) in zip(
        axes,
        examples.iterrows(),
    ):
        y, sr, _ = load_audio_segment(row)

        segment_start = float(row["start_time"])

        time_axis = (
            segment_start
            + np.arange(len(y), dtype=np.float32) / sr
        )

        segment_events = train_event_manifest[
            train_event_manifest["uuid_segmento"].astype(str)
            == str(row["uuid_segmento"])
        ].sort_values("event_index")

        if segment_events.empty:
            raise RuntimeError(
                "No se encontraron eventos para el ejemplo "
                f"{row['uuid_segmento']}."
            )

        # Dibujar primero la forma de onda.
        axis.plot(
            time_axis,
            y,
            color="#264653",
            linewidth=0.65,
            alpha=0.90,
        )

        maximum_amplitude = float(
            np.max(np.abs(y))
        )

        if maximum_amplitude < 1e-8:
            maximum_amplitude = 1.0

        axis.set_ylim(
            -1.20 * maximum_amplitude,
            1.20 * maximum_amplitude,
        )

        # Dibujar todas las regiones del segmento.
        for _, event in segment_events.iterrows():
            window_start = float(event["window_start"])
            window_end = float(event["window_end"])
            event_peak = float(event["event_peak"])
            event_index = int(event["event_index"])
            color_index = event_index % len(
                WINDOW_FACE_COLORS
            )

            detected_by_energy = bool(
                event["detected_by_energy"]
            )

            # Ventana final de 1,5 segundos.
            axis.axvspan(
                window_start,
                window_end,
                facecolor=WINDOW_FACE_COLORS[color_index],
                edgecolor=WINDOW_EDGE_COLORS[color_index],
                linewidth=1.5,
                alpha=0.28,
                zorder=1,
            )

            # Intervalo activo detectado antes de crear
            # la ventana fija.
            if detected_by_energy:
                axis.axvspan(
                    float(event["event_start"]),
                    float(event["event_end"]),
                    facecolor="#f4a261",
                    edgecolor="#d97724",
                    linewidth=1.2,
                    alpha=0.35,
                    zorder=2,
                )

            # Pico energético del evento.
            axis.axvline(
                event_peak,
                color="#d62828",
                linestyle="--",
                linewidth=1.15,
                alpha=0.95,
                zorder=3,
            )

            axis.text(
                event_peak,
                0.92 * maximum_amplitude,
                f"W{event_index}",
                color="#a41313",
                fontsize=8,
                rotation=90,
                horizontalalignment="right",
                verticalalignment="top",
            )

        fallback_used = bool(
            segment_events["fallback_no_event"].any()
        )

        title = (
            f"ID: {row['original_uuid']} | "
            f"segmento: {row['uuid_segmento']} | "
            f"clase: {row['cough_type']} | "
            f"consenso: {row['cough_type_consensus']} | "
            f"eventos: {len(segment_events)}"
        )

        if fallback_used:
            title += " | FALLBACK"

        axis.set_title(
            title,
            fontsize=10,
            loc="left",
        )

        axis.set_xlabel(
            "Time (s)"
        )

        axis.set_ylabel(
            "Amplitude"
        )

        axis.grid(
            True,
            alpha=0.20,
            linewidth=0.5,
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="#264653",
            linewidth=1.2,
            label="Señal de audio",
        ),
        Patch(
            facecolor="#f4a261",
            edgecolor="#d97724",
            alpha=0.35,
            label="Evento activo detectado",
        ),
        Patch(
            facecolor=WINDOW_FACE_COLORS[0],
            edgecolor=WINDOW_EDGE_COLORS[0],
            alpha=0.28,
            label="Ventana final par (W0, W2...)",
        ),
        Patch(
            facecolor=WINDOW_FACE_COLORS[1],
            edgecolor=WINDOW_EDGE_COLORS[1],
            alpha=0.28,
            label="Ventana final impar (W1, W3...)",
        ),
        Line2D(
            [0],
            [0],
            color="#d62828",
            linestyle="--",
            linewidth=1.2,
            label="Pico de energía",
        ),
    ]

    figure.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.995),
    )

    figure.suptitle(
        "Segmentation samples for Stage 2 dry/wet",
        fontsize=15,
        y=1.015,
    )

    figure.tight_layout(
        rect=[0.0, 0.0, 1.0, 0.975]
    )

    graph_path = (
        GRAPHS_DIR
        / (
            "segmentation_examples_train__"
            f"{SEGMENTATION_CONFIG.name}"
            f"__batch_{graph_batch:03d}.png"
        )
    )

    figure.savefig(
        graph_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"\nGráfica de ejemplos guardada en: {graph_path}")

    return graph_path

def load_existing_train_event_manifest() -> pd.DataFrame:
    """Carga los eventos ya generados sin repetir la segmentación."""

    manifest_path = (
        OUTPUT_DIR
        / "metadata_events_train_stage2_dry_wet.csv"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "No existe el manifest de train. Ejecuta primero:\n"
            "python segment_stage2_dry_wet_events.py "
            "--action segment"
        )

    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "uuid": str,
            "original_uuid": str,
            "uuid_segmento": str,
            "event_id": str,
        },
    )

    required_columns = {
        "event_id",
        "event_index",
        "original_uuid",
        "uuid_segmento",
        "event_start",
        "event_end",
        "event_peak",
        "window_start",
        "window_end",
        "detected_by_energy",
        "fallback_no_event",
        "segmentation_config",
    }

    missing_columns = (
        required_columns - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas en el manifest existente: "
            f"{sorted(missing_columns)}"
        )

    configurations = set(
        manifest["segmentation_config"]
        .astype(str)
        .unique()
    )

    if configurations != {SEGMENTATION_CONFIG.name}:
        raise ValueError(
            "El manifest fue generado con una configuración "
            "diferente.\n"
            f"Manifest: {configurations}\n"
            f"Script actual: {SEGMENTATION_CONFIG.name}"
        )

    for column in (
        "detected_by_energy",
        "fallback_no_event",
    ):
        manifest[column] = parse_boolean_column(
            manifest[column],
            column,
        )

    numeric_columns = [
        "event_index",
        "event_start",
        "event_end",
        "event_peak",
        "window_start",
        "window_end",
    ]

    for column in numeric_columns:
        manifest[column] = pd.to_numeric(
            manifest[column],
            errors="raise",
        )

    return manifest

# =============================================================================
# EJECUCIÓN
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera eventos stage 2 y/o gráficas de ejemplos."
        )
    )

    parser.add_argument(
        "--action",
        choices=[
            "all",
            "segment",
            "graphs",
            "compare",
        ],
        default="all",
        help=(
            "all: segmenta y genera gráfica; "
            "segment: solo genera manifests; "
            "graphs: usa el manifest existente; "
            "compare: audita configuraciones sobre 20 audios train."
        ),
    )

    parser.add_argument(
        "--graph-batch",
        type=int,
        default=0,
        help=(
            "Bloque de cinco ejemplos: 0, 1, 2... "
            "Cada bloque contiene 3 dry y 2 wet."
        ),
    )

    parser.add_argument(
        "--comparison-phase",
        choices=["sensitivity", "overlap", "final"],
        default="sensitivity",
        help=(
            "sensitivity: compara top_db/margen con gap=200; "
            "overlap: compara gaps 200/350/500; "
            "final: audita strict+gap500 sin solapamientos."
        ),
    )

    parser.add_argument(
        "--overlap-base",
        choices=sorted(SENSITIVITY_CONFIGS),
        default="moderate",
        help=(
            "Configuración de sensibilidad usada como base cuando "
            "--comparison-phase overlap."
        ),
    )

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    train_df = load_stage2_metadata("train")

    if args.action == "compare":
        run_parameter_audit(
            train_df=train_df,
            comparison_phase=args.comparison_phase,
            overlap_base=args.overlap_base,
        )
        return

    train_event_manifest = None

    if args.action in {"all", "segment"}:
        validation_df = load_stage2_metadata(
            "validation"
        )

        validate_no_overlap(
            train_df,
            validation_df,
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        config_row = {
            "segmentation_config": (
                SEGMENTATION_CONFIG.name
            ),
            "segmentation_method": FINAL_SEGMENTATION_METHOD,
            "enforce_non_overlap": True,
            **asdict(SEGMENTATION_CONFIG),
        }

        pd.DataFrame([config_row]).to_csv(
            (
                OUTPUT_DIR
                / "segmentation_configuration_stage2.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        print("=" * 72)
        print("SEGMENTACIÓN DE EVENTOS PARA STAGE 2 DRY/WET")
        print("=" * 72)
        print(
            f"Configuración: {SEGMENTATION_CONFIG.name}"
        )
        print(
            "Se procesarán solamente TRAIN y VALIDATION."
        )
        print("TEST no será leído ni procesado.")

        (
            train_event_manifest,
            train_segment_summary,
            train_original_summary,
        ) = process_split(
            "train",
            train_df,
        )

        (
            validation_event_manifest,
            validation_segment_summary,
            validation_original_summary,
        ) = process_split(
            "validation",
            validation_df,
        )

        print("\n" + "=" * 72)
        print("SEGMENTACIÓN STAGE 2 COMPLETADA")
        print("=" * 72)
        print(
            f"Resultados guardados en: {OUTPUT_DIR}"
        )

    if args.action == "graphs":
        print(
            "Cargando el manifest existente de train. "
            "No se repetirá la segmentación."
        )

        train_event_manifest = (
            load_existing_train_event_manifest()
        )

    if args.action in {"all", "graphs"}:
        create_train_segmentation_graph(
            train_df=train_df,
            train_event_manifest=train_event_manifest,
            graph_batch=args.graph_batch,
        )

if __name__ == "__main__":
    main()
