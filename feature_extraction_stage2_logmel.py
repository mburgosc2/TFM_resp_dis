"""Extrae espectrogramas Log-Mel para la tiny CNN de Stage 2 dry/wet.

Consume exclusivamente los manifests finales de TRAIN y VALIDATION. Cada
evento se reconstruye con la misma ventana auditada de 1,5 s y el padding
registrado por la segmentacion. TEST no se lee ni se procesa.

La salida de cada evento es un tensor [64 bandas Mel, 92 frames, 1 canal]
en float32. La potencia Log-Mel relativa se recorta a 80 dB y se escala a
[0, 1]. No se ajustan parametros estadisticos con los datos.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import feature_extraction_stage2_cochleograms as audio_events


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "features_extracted_stage2_dry_wet_logmel"
GRAPH_ROOT = (
    ROOT
    / "graphs_results_stage2_dry_wet_random"
    / "logmel_audit"
)
PRESET_NAME = "logmel64_win32_hop16"
OUTPUT_DIR = OUTPUT_ROOT / PRESET_NAME

RANDOM_STATE = 42


@dataclass(frozen=True)
class LogMelConfig:
    name: str = PRESET_NAME
    sample_rate: int = 16_000
    target_duration_seconds: float = 1.5
    n_mels: int = 64
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 256
    fmin_hz: float = 50.0
    fmax_hz: float = 7_500.0
    power: float = 2.0
    top_db: float = 80.0
    center: bool = False

    @property
    def target_samples(self) -> int:
        return int(round(self.sample_rate * self.target_duration_seconds))

    @property
    def n_frames(self) -> int:
        if self.center:
            raise ValueError("Este experimento exige center=False.")
        return 1 + (self.target_samples - self.n_fft) // self.hop_length

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        return (self.n_mels, self.n_frames, 1)


CONFIG = LogMelConfig()


def loader_config(config: LogMelConfig) -> audio_events.CochleogramConfig:
    """Crea la configuracion minima para el cargador comun de eventos."""

    return audio_events.CochleogramConfig(
        name=config.name,
        sample_rate=config.sample_rate,
        target_duration_seconds=config.target_duration_seconds,
    )


def compute_logmel(
    waveform: np.ndarray,
    config: LogMelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve Log-Mel escalado [0,1] y su version relativa en dB."""

    if waveform.shape != (config.target_samples,):
        raise ValueError(
            f"Longitud de evento inesperada: {waveform.shape}; "
            f"esperada ({config.target_samples},)."
        )
    if not np.isfinite(waveform).all():
        raise ValueError("Waveform con NaN o infinito.")
    if float(np.max(np.abs(waveform))) <= np.finfo(np.float32).tiny:
        raise ValueError("Evento completamente silencioso.")

    mel_power = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window="hann",
        center=config.center,
        power=config.power,
        n_mels=config.n_mels,
        fmin=config.fmin_hz,
        fmax=config.fmax_hz,
        htk=False,
        norm="slaney",
    )
    expected_2d = (config.n_mels, config.n_frames)
    if mel_power.shape != expected_2d:
        raise RuntimeError(
            f"Forma Log-Mel inesperada: {mel_power.shape}; "
            f"esperada {expected_2d}."
        )
    if not np.isfinite(mel_power).all():
        raise RuntimeError("El Mel espectrograma contiene NaN o infinito.")

    reference = float(np.max(mel_power))
    if reference <= np.finfo(np.float32).tiny:
        raise ValueError("Evento sin potencia espectral util.")

    logmel_db = librosa.power_to_db(
        mel_power,
        ref=reference,
        top_db=config.top_db,
    )
    logmel_db = np.clip(logmel_db, -config.top_db, 0.0)
    scaled = (logmel_db + config.top_db) / config.top_db
    tensor = scaled[..., np.newaxis].astype(np.float32)

    if tensor.shape != config.tensor_shape:
        raise RuntimeError(f"Tensor Log-Mel invalido: {tensor.shape}.")
    if not np.isfinite(tensor).all():
        raise RuntimeError("El tensor Log-Mel contiene NaN o infinito.")
    if float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
        raise RuntimeError("El tensor Log-Mel no esta dentro de [0, 1].")

    return tensor, logmel_db.astype(np.float32)


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
    config: LogMelConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    tensors: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    audio_config = loader_config(config)

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc=f"Log-Mel {split_name}",
    ):
        try:
            waveform, _ = audio_events.load_event_waveform(row, audio_config)
            tensor, _ = compute_logmel(waveform, config)
            tensors.append(tensor)
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
        pd.DataFrame(errors).to_csv(
            error_path,
            index=False,
            encoding="utf-8-sig",
        )
        raise RuntimeError(
            f"Fallaron {len(errors)} eventos de {split_name}. "
            f"Revisa {error_path}."
        )

    matrix = np.asarray(tensors, dtype=np.float32)
    metadata = pd.DataFrame(rows)
    expected_shape = (len(manifest), *config.tensor_shape)
    if matrix.shape != expected_shape:
        raise RuntimeError(
            f"Matriz Log-Mel inesperada: {matrix.shape}; "
            f"esperada {expected_shape}."
        )
    if metadata["event_id"].duplicated().any():
        raise RuntimeError("Se generaron event_id duplicados.")

    event_counts = metadata.groupby("original_uuid")["event_id"].transform(
        "count"
    )
    metadata["recording_weight"] = 1.0 / event_counts.astype(float)
    return matrix, metadata


def save_split(
    split_name: str,
    matrix: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / f"X_events_{split_name}.npy", matrix)
    np.save(
        OUTPUT_DIR / f"y_events_{split_name}.npy",
        metadata["stage2_target"].to_numpy(np.int64),
    )
    np.save(
        OUTPUT_DIR / f"folds_events_{split_name}.npy",
        metadata["fold"].to_numpy(np.int32),
    )
    metadata.to_csv(
        OUTPUT_DIR / f"metadata_events_features_{split_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def select_audit_events(manifest: pd.DataFrame) -> pd.DataFrame:
    selected = []
    groups = [
        ("dry", "gold_expert"),
        ("dry", "weak_expert"),
        ("wet", "gold_expert"),
        ("wet", "weak_expert"),
    ]
    for group_index, (cough_type, consensus) in enumerate(groups):
        candidates = manifest[
            (manifest["cough_type"] == cough_type)
            & (manifest["cough_type_consensus"] == consensus)
        ]
        if candidates.empty:
            raise ValueError(f"No hay ejemplos {cough_type}/{consensus}.")
        selected.append(
            candidates.sample(
                n=min(2, len(candidates)),
                random_state=RANDOM_STATE + group_index,
            )
        )
    return pd.concat(selected, ignore_index=True)


def create_audit_graph(
    train_manifest: pd.DataFrame,
    config: LogMelConfig,
) -> Path:
    examples = select_audit_events(train_manifest)
    figure, axes = plt.subplots(
        len(examples),
        2,
        figsize=(16, 3.0 * len(examples)),
        squeeze=False,
    )
    audio_config = loader_config(config)

    for row_index, (_, row) in enumerate(examples.iterrows()):
        waveform, _ = audio_events.load_event_waveform(row, audio_config)
        _, logmel_db = compute_logmel(waveform, config)
        time_axis = np.arange(config.target_samples) / config.sample_rate

        axes[row_index, 0].plot(time_axis, waveform, linewidth=0.65)
        axes[row_index, 0].set_xlim(0.0, config.target_duration_seconds)
        axes[row_index, 0].set_ylabel("Amplitud")
        axes[row_index, 0].set_title(
            f"{row['cough_type']} | {row['cough_type_consensus']} | "
            f"{row['event_id']}"
        )
        axes[row_index, 0].grid(alpha=0.2)

        image = axes[row_index, 1].imshow(
            logmel_db,
            origin="lower",
            aspect="auto",
            extent=[
                0.0,
                config.target_duration_seconds,
                0,
                config.n_mels - 1,
            ],
            cmap="magma",
            vmin=-config.top_db,
            vmax=0.0,
        )
        axes[row_index, 1].set_ylabel("Banda Mel (grave -> aguda)")
        axes[row_index, 1].set_title(
            f"Log-Mel {config.n_mels} x {config.n_frames}"
        )
        figure.colorbar(image, ax=axes[row_index, 1], label="dB relativos")

    for axis in axes[-1]:
        axis.set_xlabel("Tiempo (s)")

    figure.suptitle(
        f"Auditoria Log-Mel Stage 2 - {config.name}",
        fontsize=15,
        y=1.002,
    )
    figure.tight_layout()
    graph_dir = GRAPH_ROOT / config.name
    graph_dir.mkdir(parents=True, exist_ok=True)
    output_path = graph_dir / "logmel_examples_train.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def validate_manifests(
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    config: LogMelConfig,
) -> None:
    audio_events.validate_train_validation_disjoint(
        train_manifest,
        validation_manifest,
    )
    if config.target_samples != 24_000:
        raise RuntimeError("Se esperaban exactamente 24.000 muestras.")
    if config.tensor_shape != (64, 92, 1):
        raise RuntimeError(
            f"Forma Log-Mel no esperada: {config.tensor_shape}."
        )
    examples = pd.concat(
        [
            train_manifest[train_manifest["stage2_target"] == target].head(1)
            for target in (0, 1)
        ],
        ignore_index=True,
    )
    for _, row in examples.iterrows():
        waveform, _ = audio_events.load_event_waveform(
            row,
            loader_config(config),
        )
        tensor, _ = compute_logmel(waveform, config)
        if tensor.shape != config.tensor_shape:
            raise RuntimeError("El check no produjo la forma esperada.")

    print("=" * 78)
    print("CHECK - EXTRACCION LOG-MEL STAGE 2")
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
    print(f"Tensor por evento: {config.tensor_shape} float32.")
    print("Escala fija: potencia relativa -80..0 dB -> 0..1.")
    print("TRAIN y VALIDATION no comparten UUID. TEST no se ha leido.")


def save_configuration(config: LogMelConfig) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configuration = {
        **asdict(config),
        "target_samples": config.target_samples,
        "n_frames": config.n_frames,
        "tensor_shape": "x".join(map(str, config.tensor_shape)),
        "scaling": "per_event_relative_power_db_clipped_then_0_1",
        "segmentation_config": audio_events.EXPECTED_SEGMENTATION_CONFIG,
        "segmentation_method": audio_events.EXPECTED_SEGMENTATION_METHOD,
        "test_processed": False,
    }
    pd.DataFrame([configuration]).to_csv(
        OUTPUT_DIR / "logmel_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrae Log-Mel de eventos Stage 2 dry/wet."
    )
    parser.add_argument(
        "--action",
        choices=["check", "audit", "extract", "all"],
        default="check",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - EXTRACCION LOG-MEL PARA TINY CNN")
    print("=" * 78)
    print(f"Preset: {CONFIG.name} | Accion: {args.action}")
    print("TEST no sera leido ni procesado.")

    train_manifest = audio_events.load_event_manifest("train")
    validation_manifest = audio_events.load_event_manifest("validation")
    validate_manifests(train_manifest, validation_manifest, CONFIG)

    if args.action == "check":
        print("Comprobacion completada. No se guardaron features.")
        return

    save_configuration(CONFIG)
    if args.action in {"audit", "all"}:
        graph_path = create_audit_graph(train_manifest, CONFIG)
        print(f"Grafica de auditoria: {graph_path}")

    if args.action in {"extract", "all"}:
        for split_name, manifest in (
            ("train", train_manifest),
            ("validation", validation_manifest),
        ):
            matrix, metadata = extract_split(manifest, split_name, CONFIG)
            save_split(split_name, matrix, metadata)
            size_mb = matrix.nbytes / (1024.0**2)
            print(
                f"{split_name}: {matrix.shape}, {size_mb:.2f} MB; "
                f"grabaciones={metadata['original_uuid'].nunique()}"
            )
        print(f"Features guardadas en: {OUTPUT_DIR}")
        print("TEST permanece reservado.")


if __name__ == "__main__":
    main()
