"""Extrae MFCC de los eventos finales de Stage 2 dry/wet.

Cada evento se reconstruye como la misma ventana de 1,5 s utilizada por
los experimentos WST y cocleograma. Se calculan 13 MFCC, 13 delta y 13
delta-delta. Cada una de esas 39 trayectorias se resume mediante media,
desviacion estandar y maximo, produciendo 117 variables por evento.

Tambien se guarda una vista por grabacion con mean/std/max entre eventos:
351 variables y una sola etiqueta por original_uuid. Solo se procesan TRAIN
y VALIDATION; TEST no se lee.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

import feature_extraction_stage2_cochleograms as audio_events
import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "features_extracted_stage2_dry_wet_mfcc"
PRESET_NAME = "mfcc117"
OUTPUT_DIR = OUTPUT_ROOT / PRESET_NAME


@dataclass(frozen=True)
class MFCCConfig:
    name: str = PRESET_NAME
    sample_rate: int = 16_000
    target_duration_seconds: float = 1.5
    n_mfcc: int = 13
    n_fft: int = 2_048
    win_length: int = 2_048
    hop_length: int = 512
    n_mels: int = 128
    fmin_hz: float = 0.0
    fmax_hz: float = 8_000.0
    delta_width: int = 9

    @property
    def target_samples(self) -> int:
        return int(round(self.sample_rate * self.target_duration_seconds))


CONFIG = MFCCConfig()
RECORDING_POOLING = next(
    spec for spec in recording.POOLING_SPECS if spec.key == "mean_std_max"
)


def audio_config(config: MFCCConfig) -> audio_events.CochleogramConfig:
    """Configuracion minima compatible con el cargador comun de eventos."""

    return audio_events.CochleogramConfig(
        name=config.name,
        sample_rate=config.sample_rate,
        target_duration_seconds=config.target_duration_seconds,
    )


def trajectory_names(config: MFCCConfig) -> list[str]:
    names = []
    for family in ("mfcc", "delta", "delta2"):
        names.extend(
            f"{family}_{coefficient:02d}"
            for coefficient in range(1, config.n_mfcc + 1)
        )
    return names


def event_feature_names(config: MFCCConfig) -> list[str]:
    trajectories = trajectory_names(config)
    return [
        f"frame_{statistic}__{name}"
        for statistic in ("mean", "std", "max")
        for name in trajectories
    ]


def extract_mfcc_vector(
    waveform: np.ndarray,
    config: MFCCConfig,
) -> np.ndarray:
    if waveform.shape != (config.target_samples,):
        raise ValueError(f"Longitud de evento inesperada: {waveform.shape}.")
    if not np.isfinite(waveform).all():
        raise ValueError("Waveform con NaN o infinito.")
    if float(np.max(np.abs(waveform))) <= np.finfo(np.float32).tiny:
        raise ValueError("Evento completamente silencioso.")

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        fmin=config.fmin_hz,
        fmax=config.fmax_hz,
        center=True,
    )
    delta = librosa.feature.delta(
        mfcc,
        width=config.delta_width,
        order=1,
        mode="interp",
    )
    delta2 = librosa.feature.delta(
        mfcc,
        width=config.delta_width,
        order=2,
        mode="interp",
    )
    trajectories = np.vstack((mfcc, delta, delta2))
    expected_rows = config.n_mfcc * 3
    if trajectories.shape[0] != expected_rows:
        raise RuntimeError(f"Forma MFCC inesperada: {trajectories.shape}.")

    vector = np.concatenate(
        (
            np.mean(trajectories, axis=1),
            np.std(trajectories, axis=1, ddof=0),
            np.max(trajectories, axis=1),
        )
    ).astype(np.float32)
    expected_dimension = expected_rows * 3
    if vector.shape != (expected_dimension,) or not np.isfinite(vector).all():
        raise RuntimeError("Vector MFCC invalido.")
    return vector


def metadata_row(row: pd.Series, feature_row: int) -> dict[str, object]:
    return {
        "feature_row": feature_row,
        "event_id": str(row["event_id"]),
        "event_index": int(row["event_index"]),
        "original_uuid": str(row["original_uuid"]),
        "uuid_segmento": str(row["uuid_segmento"]),
        "cough_type": str(row["cough_type"]),
        "cough_type_consensus": str(row["cough_type_consensus"]),
        "stage2_target": int(row["stage2_target"]),
        "fold": int(row["fold"]),
        "split": str(row["split"]),
        "event_duration": float(row["event_duration"]),
        "window_observed_duration": float(row["window_observed_duration"]),
        "window_total_padding": float(row["window_total_padding"]),
        "non_overlap_adjusted": bool(row["non_overlap_adjusted"]),
        "fallback_no_event": bool(row["fallback_no_event"]),
    }


def extract_split(
    manifest: pd.DataFrame,
    split_name: str,
    config: MFCCConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    vectors: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    loader_config = audio_config(config)

    for _, row in tqdm(
        manifest.iterrows(), total=len(manifest), desc=f"MFCC {split_name}"
    ):
        try:
            waveform, _ = audio_events.load_event_waveform(row, loader_config)
            vector = extract_mfcc_vector(waveform, config)
            vectors.append(vector)
            rows.append(metadata_row(row, len(rows)))
        except Exception as exc:
            errors.append(
                {
                    "event_id": str(row["event_id"]),
                    "original_uuid": str(row["original_uuid"]),
                    "error": str(exc),
                }
            )

    if errors:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        error_path = OUTPUT_DIR / f"errors_{split_name}.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False, encoding="utf-8-sig")
        raise RuntimeError(
            f"Fallaron {len(errors)} eventos de {split_name}. Revisa {error_path}."
        )

    matrix = np.asarray(vectors, dtype=np.float32)
    metadata = pd.DataFrame(rows)
    expected = len(event_feature_names(config))
    if matrix.shape != (len(manifest), expected):
        raise RuntimeError(
            f"Matriz MFCC {split_name} inesperada: {matrix.shape}; "
            f"esperada ({len(manifest)}, {expected})."
        )
    event_counts = metadata.groupby("original_uuid")["event_id"].transform("count")
    metadata["recording_weight"] = 1.0 / event_counts.astype(float)
    common.validate_arrays_and_metadata(
        "event",
        split_name,
        matrix,
        metadata["stage2_target"].to_numpy(np.int64),
        metadata["fold"].to_numpy(np.int32),
        metadata,
    )
    return matrix, metadata


def save_split(
    split_name: str,
    event_matrix: np.ndarray,
    event_metadata: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    recording_matrix, recording_metadata = recording.aggregate_split(
        event_matrix,
        event_metadata,
        feature_names,
        RECORDING_POOLING,
        split_name,
    )
    recording_names = recording.recording_feature_names(
        feature_names, RECORDING_POOLING
    )
    audio_events.save_split_features(
        split_name,
        event_matrix,
        event_metadata,
        recording_matrix,
        recording_metadata,
        OUTPUT_DIR,
    )
    return recording_matrix, recording_metadata, recording_names


def save_configuration(config: MFCCConfig, names: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feature_index": np.arange(len(names), dtype=int),
            "feature_name": names,
        }
    ).to_csv(
        OUTPUT_DIR / "mfcc_event_feature_layout.csv",
        index=False,
        encoding="utf-8-sig",
    )
    configuration = {
        **asdict(config),
        "event_feature_count": len(names),
        "frame_pooling": "mean|std|max",
        "recording_pooling": RECORDING_POOLING.key,
        "recording_feature_count": len(names) * RECORDING_POOLING.statistic_count,
        "segmentation_config": audio_events.EXPECTED_SEGMENTATION_CONFIG,
        "segmentation_method": audio_events.EXPECTED_SEGMENTATION_METHOD,
        "test_processed": False,
    }
    pd.DataFrame([configuration]).to_csv(
        OUTPUT_DIR / "mfcc_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )


def check_manifests(
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    config: MFCCConfig,
) -> None:
    audio_events.validate_train_validation_disjoint(
        train_manifest, validation_manifest
    )
    names = event_feature_names(config)
    if len(names) != 117 or len(set(names)) != 117:
        raise RuntimeError("Se esperaban 117 nombres MFCC unicos.")
    examples = pd.concat(
        [
            train_manifest[train_manifest["stage2_target"] == target].head(1)
            for target in (0, 1)
        ],
        ignore_index=True,
    )
    for _, row in examples.iterrows():
        waveform, _ = audio_events.load_event_waveform(row, audio_config(config))
        vector = extract_mfcc_vector(waveform, config)
        if vector.shape != (117,):
            raise RuntimeError("El check MFCC no produjo 117 variables.")

    print("=" * 78)
    print("CHECK — EXTRACCION MFCC STAGE 2")
    print("=" * 78)
    print(
        f"Eventos TRAIN/VALIDATION: {len(train_manifest)} / "
        f"{len(validation_manifest)}"
    )
    print(
        "Grabaciones TRAIN/VALIDATION: "
        f"{train_manifest['original_uuid'].nunique()} / "
        f"{validation_manifest['original_uuid'].nunique()}"
    )
    print("Salida: 117 features/evento y 351 features/grabacion.")
    print("Ventanas: 1,5 s con el padding del manifest final.")
    print("TRAIN y VALIDATION no comparten UUID. TEST no se ha leido.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrae MFCC de los eventos Stage 2 dry/wet."
    )
    parser.add_argument("--action", choices=["check", "extract"], default="check")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 — EXTRACCION MFCC POR EVENTO Y GRABACION")
    print("=" * 78)
    print(f"Preset: {CONFIG.name} | Accion: {args.action}")
    print("TEST no sera leido ni procesado.")

    train_manifest = audio_events.load_event_manifest("train")
    validation_manifest = audio_events.load_event_manifest("validation")
    check_manifests(train_manifest, validation_manifest, CONFIG)
    if args.action == "check":
        print("Comprobacion completada. No se extrajeron features.")
        return

    names = event_feature_names(CONFIG)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_configuration(CONFIG, names)
    recording_names: list[str] | None = None
    for split_name, manifest in (
        ("train", train_manifest),
        ("validation", validation_manifest),
    ):
        event_matrix, event_metadata = extract_split(manifest, split_name, CONFIG)
        recording_matrix, _, current_recording_names = save_split(
            split_name, event_matrix, event_metadata, names
        )
        if recording_names is None:
            recording_names = current_recording_names
        elif recording_names != current_recording_names:
            raise RuntimeError("Los nombres recording cambian entre splits.")
        print(
            f"{split_name}: eventos {event_matrix.shape}; "
            f"grabaciones {recording_matrix.shape}"
        )

    pd.DataFrame(
        {
            "feature_index": np.arange(len(recording_names), dtype=int),
            "feature_name": recording_names,
        }
    ).to_csv(
        OUTPUT_DIR / "mfcc_recording_feature_layout.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Features guardadas en: {OUTPUT_DIR}")
    print("TEST permanece reservado.")


if __name__ == "__main__":
    main()
