"""Data augmentation controlado de grabaciones gold para Stage 2 dry/wet.

Este experimento NO recalcula las WST originales. Selecciona exclusivamente
de TRAIN las seis grabaciones wet gold y seis controles dry gold emparejados
por fold y nivel de acuerdo experto (3/4 o 4/4). Para cada grabacion genera
cuatro variantes suaves y extrae WST solo de esas variantes:

* tempo_slow: velocidad 0.90 (sin alterar deliberadamente el tono)
* tempo_fast: velocidad 1.10
* pitch_down: -0.75 semitonos
* pitch_up: +0.75 semitonos

Todas las ventanas/eventos de una grabacion reciben la misma transformacion.
Las filas aumentadas conservan parent_original_uuid y el fold del padre. Esto
permite que el futuro script de entrenamiento las introduzca solo en el train
interno de cada fold, nunca en su validacion OOF. VALIDATION y TEST no se leen.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import feature_extraction_stage2_cochleograms as audio_base
import feature_extraction_stage2_wavelet_scattering as wst_base


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERT_VOTES_PATH = (
    SCRIPT_DIR
    / "metadata_stage2_expert_vote_analysis"
    / "stage2_expert_votes_detailed.csv"
)
ORIGINAL_WST_DIR = (
    SCRIPT_DIR
    / "features_extracted_stage2_dry_wet_wavelet_scattering"
    / "paper_q8_q1_t500_full"
)
OUTPUT_ROOT = (
    SCRIPT_DIR / "features_extracted_stage2_dry_wet_wst_gold_augmentation"
)
AUDIT_AUDIO_ROOT = SCRIPT_DIR / "auditory_stage2_wst_gold_augmentation"
GRAPH_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "wst_gold_augmentation"
)

RANDOM_STATE = 42
EXPECTED_WET_GOLD_RECORDINGS = 6
EXPECTED_DRY_CONTROLS = 6


@dataclass(frozen=True)
class AugmentationSpec:
    name: str
    transform: str
    value: float
    unit: str


AUGMENTATIONS = (
    AugmentationSpec("tempo_slow", "time_stretch", 0.90, "rate"),
    AugmentationSpec("tempo_fast", "time_stretch", 1.10, "rate"),
    AugmentationSpec("pitch_down", "pitch_shift", -0.75, "semitones"),
    AugmentationSpec("pitch_up", "pitch_shift", 0.75, "semitones"),
)


def load_expert_votes() -> pd.DataFrame:
    if not EXPERT_VOTES_PATH.is_file():
        raise FileNotFoundError(
            f"No se encuentra el analisis de votos: {EXPERT_VOTES_PATH}"
        )
    votes = pd.read_csv(
        EXPERT_VOTES_PATH,
        dtype={"original_uuid": str, "split": str},
    )
    required = {
        "original_uuid",
        "cough_type",
        "cough_type_consensus",
        "split",
        "fold",
        "expert_evaluator_count",
        "dry_vote_count",
        "wet_vote_count",
        "unknown_vote_count",
        "maximum_agreement_count",
        "vote_pattern",
    }
    missing = required - set(votes.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en {EXPERT_VOTES_PATH.name}: {sorted(missing)}"
        )
    if votes["original_uuid"].duplicated().any():
        raise ValueError("El analisis de expertos contiene UUID duplicados.")
    for column in (
        "fold",
        "expert_evaluator_count",
        "dry_vote_count",
        "wet_vote_count",
        "unknown_vote_count",
        "maximum_agreement_count",
    ):
        votes[column] = pd.to_numeric(votes[column], errors="raise").astype(int)
    return votes


def recording_table(
    train_events: pd.DataFrame,
    votes: pd.DataFrame,
) -> pd.DataFrame:
    consistency_columns = [
        "cough_type",
        "cough_type_consensus",
        "stage2_target",
        "fold",
        "split",
    ]
    grouped = train_events.groupby("original_uuid", sort=True)
    for column in consistency_columns:
        if (grouped[column].nunique(dropna=False) != 1).any():
            raise ValueError(
                f"La columna {column} no es constante dentro de alguna grabacion."
            )

    recordings = grouped.first().reset_index()[
        ["original_uuid", *consistency_columns]
    ]
    recordings["event_count"] = recordings["original_uuid"].map(
        grouped.size()
    )
    vote_columns = [
        "original_uuid",
        "expert_evaluator_count",
        "dry_vote_count",
        "wet_vote_count",
        "unknown_vote_count",
        "maximum_agreement_count",
        "vote_pattern",
    ]
    recordings = recordings.merge(
        votes.loc[votes["split"].eq("train"), vote_columns],
        on="original_uuid",
        how="left",
        validate="one_to_one",
    )
    if recordings["maximum_agreement_count"].isna().any():
        missing = recordings.loc[
            recordings["maximum_agreement_count"].isna(), "original_uuid"
        ].tolist()
        raise ValueError(
            "Faltan votos expertos para grabaciones TRAIN: " + str(missing[:5])
        )
    return recordings


def select_parent_recordings(recordings: pd.DataFrame) -> pd.DataFrame:
    """Empareja cada wet gold con una dry gold del mismo fold y acuerdo."""

    gold = recordings.loc[
        recordings["cough_type_consensus"].eq("gold_expert")
        & recordings["cough_type"].isin(["dry", "wet"])
    ].copy()
    wet = gold.loc[gold["cough_type"].eq("wet")].sort_values(
        ["fold", "maximum_agreement_count", "original_uuid"]
    )
    if len(wet) != EXPECTED_WET_GOLD_RECORDINGS:
        raise ValueError(
            f"Se esperaban {EXPECTED_WET_GOLD_RECORDINGS} wet gold TRAIN y "
            f"se encontraron {len(wet)}."
        )

    dry_pool = gold.loc[gold["cough_type"].eq("dry")].sort_values(
        ["fold", "maximum_agreement_count", "original_uuid"]
    )
    rng = np.random.default_rng(RANDOM_STATE)
    used_dry: set[str] = set()
    rows: list[dict[str, object]] = []

    for pair_number, (_, wet_row) in enumerate(wet.iterrows(), start=1):
        exact = dry_pool.loc[
            dry_pool["fold"].eq(wet_row["fold"])
            & dry_pool["maximum_agreement_count"].eq(
                wet_row["maximum_agreement_count"]
            )
            & ~dry_pool["original_uuid"].isin(used_dry)
        ]
        if exact.empty:
            raise ValueError(
                "No existe dry gold de control para "
                f"fold={wet_row['fold']} y acuerdo="
                f"{wet_row['maximum_agreement_count']}/4."
            )
        dry_row = exact.iloc[int(rng.integers(0, len(exact)))]
        used_dry.add(str(dry_row["original_uuid"]))
        pair_id = f"pair_{pair_number:02d}"

        for role, selected in (("wet_gold", wet_row), ("dry_gold_control", dry_row)):
            item = selected.to_dict()
            item.update(
                {
                    "pair_id": pair_id,
                    "selection_role": role,
                    "matched_wet_uuid": str(wet_row["original_uuid"]),
                    "selection_seed": RANDOM_STATE,
                }
            )
            rows.append(item)

    selection = pd.DataFrame(rows).sort_values(
        ["pair_id", "selection_role"], ascending=[True, False]
    ).reset_index(drop=True)
    if selection["original_uuid"].duplicated().any():
        raise RuntimeError("La seleccion contiene grabaciones repetidas.")
    if selection["selection_role"].value_counts().to_dict() != {
        "wet_gold": EXPECTED_WET_GOLD_RECORDINGS,
        "dry_gold_control": EXPECTED_DRY_CONTROLS,
    }:
        raise RuntimeError("La seleccion final no contiene 6 wet y 6 dry.")
    if not (selection["split"] == "train").all():
        raise RuntimeError("La seleccion contiene grabaciones fuera de TRAIN.")
    return selection


def force_target_length(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size > target_samples:
        excess = waveform.size - target_samples
        left = excess // 2
        waveform = waveform[left : left + target_samples]
    elif waveform.size < target_samples:
        missing = target_samples - waveform.size
        left = missing // 2
        waveform = np.pad(
            waveform,
            (left, missing - left),
            mode="constant",
        )
    if waveform.size != target_samples:
        raise RuntimeError("No se pudo imponer la longitud de 1,5 s.")
    if not np.isfinite(waveform).all():
        raise RuntimeError("La transformacion genero NaN o infinito.")
    return waveform.astype(np.float32, copy=False)


def augment_waveform(
    waveform: np.ndarray,
    spec: AugmentationSpec,
    config: wst_base.ScatteringConfig,
) -> np.ndarray:
    if spec.transform == "time_stretch":
        augmented = librosa.effects.time_stretch(
            y=np.asarray(waveform, dtype=np.float32),
            rate=spec.value,
        )
    elif spec.transform == "pitch_shift":
        augmented = librosa.effects.pitch_shift(
            y=np.asarray(waveform, dtype=np.float32),
            sr=config.sample_rate,
            n_steps=spec.value,
            bins_per_octave=12,
        )
    else:
        raise ValueError(f"Transformacion desconocida: {spec.transform}")
    return force_target_length(augmented, config.target_samples)


def normalized_for_wst(
    waveform: np.ndarray,
    config: wst_base.ScatteringConfig,
) -> tuple[np.ndarray, float]:
    waveform = force_target_length(waveform, config.target_samples)
    return wst_base.normalize_waveform(
        waveform,
        config.normalize_waveform_peak,
    )


def representative_events(
    train_events: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    selected = train_events.loc[
        train_events["original_uuid"].isin(selection["original_uuid"])
    ].copy()
    # La mayor duracion activa es el ejemplo mas informativo; los desempates
    # se resuelven por el indice original del evento.
    selected = selected.sort_values(
        ["original_uuid", "event_duration", "event_index"],
        ascending=[True, False, True],
    )
    representatives = selected.groupby("original_uuid", sort=False).head(1)
    return representatives.merge(
        selection[
            [
                "original_uuid",
                "pair_id",
                "selection_role",
                "maximum_agreement_count",
                "vote_pattern",
            ]
        ],
        on="original_uuid",
        how="left",
        validate="one_to_one",
    ).sort_values(["pair_id", "selection_role"], ascending=[True, False])


def log_spectrogram(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    magnitude = np.abs(
        librosa.stft(
            waveform,
            n_fft=512,
            hop_length=128,
            win_length=512,
            center=True,
        )
    )
    return librosa.amplitude_to_db(magnitude, ref=np.max, top_db=80.0)


def save_audit(
    train_events: pd.DataFrame,
    selection: pd.DataFrame,
    config: wst_base.ScatteringConfig,
) -> pd.DataFrame:
    AUDIT_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    graph_dir = GRAPH_ROOT / config.name
    graph_dir.mkdir(parents=True, exist_ok=True)
    representatives = representative_events(train_events, selection)
    audit_rows: list[dict[str, object]] = []

    for _, row in tqdm(
        representatives.iterrows(),
        total=len(representatives),
        desc="Generando auditoria augmentation",
    ):
        original, _ = audio_base.load_event_waveform(
            row, audio_base.PRESETS["paper64"]
        )
        original, original_peak = normalized_for_wst(original, config)
        variants: list[tuple[str, np.ndarray, str]] = [
            ("original", original, "sin transformacion")
        ]
        for spec in AUGMENTATIONS:
            transformed = augment_waveform(original, spec, config)
            transformed, _ = normalized_for_wst(transformed, config)
            variants.append(
                (spec.name, transformed, f"{spec.value:g} {spec.unit}")
            )

        short_uuid = str(row["original_uuid"])[:8]
        stem = (
            f"{row['pair_id']}__{row['cough_type']}__{short_uuid}"
            f"__{row['event_id']}"
        )
        fig, axes = plt.subplots(
            len(variants),
            2,
            figsize=(14, 12),
            constrained_layout=True,
        )
        times = np.arange(config.target_samples) / config.sample_rate
        colors = ["#333333", "#2b8cbe", "#e34a33", "#31a354", "#756bb1"]
        for index, (variant_name, waveform, parameter) in enumerate(variants):
            wav_name = f"{stem}__{variant_name}.wav"
            wav_path = AUDIT_AUDIO_ROOT / wav_name
            sf.write(wav_path, waveform, config.sample_rate, subtype="PCM_16")

            axes[index, 0].plot(times, waveform, color=colors[index], linewidth=0.65)
            axes[index, 0].set_xlim(0.0, config.target_duration_seconds)
            axes[index, 0].set_ylim(-1.05, 1.05)
            axes[index, 0].set_ylabel(variant_name)
            axes[index, 0].grid(alpha=0.15)
            if index == len(variants) - 1:
                axes[index, 0].set_xlabel("Tiempo (s)")

            spectrum = log_spectrogram(waveform, config.sample_rate)
            axes[index, 1].imshow(
                spectrum,
                origin="lower",
                aspect="auto",
                extent=[0.0, config.target_duration_seconds, 0.0, config.sample_rate / 2],
                cmap="magma",
                vmin=-80.0,
                vmax=0.0,
            )
            axes[index, 1].set_ylim(0.0, 8000.0)
            axes[index, 1].set_ylabel("Hz")
            axes[index, 1].set_title(f"{variant_name}: {parameter}", fontsize=9)
            if index == len(variants) - 1:
                axes[index, 1].set_xlabel("Tiempo (s)")

            audit_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "selection_role": row["selection_role"],
                    "cough_type": row["cough_type"],
                    "parent_original_uuid": row["original_uuid"],
                    "representative_event_id": row["event_id"],
                    "fold": int(row["fold"]),
                    "maximum_agreement_count": int(
                        row["maximum_agreement_count"]
                    ),
                    "vote_pattern": row["vote_pattern"],
                    "augmentation_type": variant_name,
                    "augmentation_parameter": parameter,
                    "wav_path": str(wav_path),
                    "original_peak_before_normalization": original_peak,
                }
            )

        fig.suptitle(
            f"{row['pair_id']} | {row['cough_type']} {row['selection_role']} | "
            f"fold {int(row['fold'])} | acuerdo "
            f"{int(row['maximum_agreement_count'])}/4\n"
            f"UUID {row['original_uuid']} | evento {row['event_id']}",
            fontsize=12,
        )
        figure_path = graph_dir / f"{stem}.png"
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(
        AUDIT_AUDIO_ROOT / "augmentation_audit_index.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return audit


def augmented_metadata_row(
    source_row: pd.Series,
    selection_row: pd.Series,
    spec: AugmentationSpec,
    feature_row: int,
    waveform_peak: float,
    extraction_seconds: float,
) -> dict[str, object]:
    parent_uuid = str(source_row["original_uuid"])
    augmented_uuid = f"{parent_uuid}__aug_{spec.name}"
    return {
        "feature_row": feature_row,
        "event_id": f"{source_row['event_id']}__aug_{spec.name}",
        "event_index": int(source_row["event_index"]),
        "original_uuid": augmented_uuid,
        "parent_original_uuid": parent_uuid,
        "parent_event_id": source_row["event_id"],
        "uuid_segmento": f"{source_row['uuid_segmento']}__aug_{spec.name}",
        "cough_type": source_row["cough_type"],
        "cough_type_consensus": source_row["cough_type_consensus"],
        "stage2_target": int(source_row["stage2_target"]),
        "fold": int(source_row["fold"]),
        "split": "train",
        "pair_id": selection_row["pair_id"],
        "selection_role": selection_row["selection_role"],
        "expert_evaluator_count": int(selection_row["expert_evaluator_count"]),
        "dry_vote_count": int(selection_row["dry_vote_count"]),
        "wet_vote_count": int(selection_row["wet_vote_count"]),
        "unknown_vote_count": int(selection_row["unknown_vote_count"]),
        "maximum_agreement_count": int(
            selection_row["maximum_agreement_count"]
        ),
        "vote_pattern": selection_row["vote_pattern"],
        "is_augmented": True,
        "augmentation_type": spec.name,
        "augmentation_transform": spec.transform,
        "augmentation_value": spec.value,
        "augmentation_unit": spec.unit,
        "event_duration": float(source_row["event_duration"]),
        "window_observed_duration": float(
            source_row["window_observed_duration"]
        ),
        "window_total_padding": float(source_row["window_total_padding"]),
        "waveform_peak_before_normalization": waveform_peak,
        "scattering_extraction_seconds": extraction_seconds,
    }


def extract_augmented_wst(
    train_events: pd.DataFrame,
    selection: pd.DataFrame,
    config: wst_base.ScatteringConfig,
    batch_size: int,
    output_dir: Path,
) -> tuple[tuple[int, int], pd.DataFrame]:
    selected_events = train_events.loc[
        train_events["original_uuid"].isin(selection["original_uuid"])
    ].sort_values(["original_uuid", "event_index"])
    selection_lookup = selection.set_index("original_uuid")
    expected_rows = len(selected_events) * len(AUGMENTATIONS)

    scattering = wst_base.build_scattering(config)
    path_count, _, _, layout = wst_base.infer_layout(scattering, config)
    matrix = np.empty((expected_rows, path_count), dtype=np.float32)
    metadata_rows: list[dict[str, object]] = []
    pending_waveforms: list[np.ndarray] = []
    pending_info: list[
        tuple[int, pd.Series, pd.Series, AugmentationSpec, float]
    ] = []

    def flush_batch() -> None:
        if not pending_waveforms:
            return
        features, elapsed = wst_base.process_batch(
            scattering, pending_waveforms, path_count
        )
        seconds_per_event = elapsed / len(pending_waveforms)
        for batch_index, info in enumerate(pending_info):
            output_row, source_row, selection_row, spec, peak = info
            matrix[output_row] = features[batch_index]
            metadata_rows.append(
                augmented_metadata_row(
                    source_row,
                    selection_row,
                    spec,
                    output_row,
                    peak,
                    seconds_per_event,
                )
            )
        pending_waveforms.clear()
        pending_info.clear()

    output_row = 0
    for _, source_row in tqdm(
        selected_events.iterrows(),
        total=len(selected_events),
        desc="Extrayendo WST aumentadas",
    ):
        waveform, _ = audio_base.load_event_waveform(
            source_row, audio_base.PRESETS["paper64"]
        )
        selection_row = selection_lookup.loc[str(source_row["original_uuid"])]
        for spec in AUGMENTATIONS:
            augmented = augment_waveform(waveform, spec, config)
            augmented, peak = normalized_for_wst(augmented, config)
            pending_waveforms.append(augmented)
            pending_info.append(
                (output_row, source_row, selection_row, spec, peak)
            )
            output_row += 1
            if len(pending_waveforms) >= batch_size:
                flush_batch()
    flush_batch()

    metadata = pd.DataFrame(metadata_rows).sort_values("feature_row").reset_index(
        drop=True
    )
    validate_augmented_output(
        matrix,
        metadata,
        selected_events,
        selection,
        train_events,
    )
    counts = metadata.groupby("original_uuid")["event_id"].transform("count")
    metadata["recording_weight"] = 1.0 / counts.astype(float)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_events_train_augmented.npy", matrix)
    np.save(
        output_dir / "y_events_train_augmented.npy",
        metadata["stage2_target"].to_numpy(np.int64),
    )
    np.save(
        output_dir / "folds_events_train_augmented.npy",
        metadata["fold"].to_numpy(np.int32),
    )
    metadata.to_csv(
        output_dir / "metadata_events_features_train_augmented.csv",
        index=False,
        encoding="utf-8-sig",
    )
    wst_base.feature_layout_dataframe(scattering, path_count).to_csv(
        output_dir / "wavelet_scattering_feature_layout.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_configuration(config, layout, batch_size, selected_events, output_dir)
    return matrix.shape, metadata


def validate_augmented_output(
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    selected_events: pd.DataFrame,
    selection: pd.DataFrame,
    all_train_events: pd.DataFrame,
) -> None:
    expected = len(selected_events) * len(AUGMENTATIONS)
    if matrix.shape[0] != expected or len(metadata) != expected:
        raise RuntimeError("Numero inesperado de eventos aumentados.")
    if not np.isfinite(matrix).all():
        raise RuntimeError("La matriz WST aumentada contiene NaN o infinito.")
    if metadata["event_id"].duplicated().any():
        raise RuntimeError("Existen event_id aumentados duplicados.")
    if set(metadata["split"]) != {"train"}:
        raise RuntimeError("Se introdujeron filas ajenas a TRAIN.")
    if set(metadata["parent_original_uuid"]) != set(selection["original_uuid"]):
        raise RuntimeError("Los padres aumentados no coinciden con la seleccion.")
    if set(metadata["augmentation_type"]) != {
        spec.name for spec in AUGMENTATIONS
    }:
        raise RuntimeError("Falta alguna de las cuatro transformaciones.")
    if set(metadata["original_uuid"]) & set(all_train_events["original_uuid"]):
        raise RuntimeError("Un UUID aumentado colisiona con un UUID original.")

    parent_fold = selection.set_index("original_uuid")["fold"].astype(int)
    inherited_fold = metadata["parent_original_uuid"].map(parent_fold).astype(int)
    if not np.array_equal(inherited_fold.to_numpy(), metadata["fold"].to_numpy()):
        raise RuntimeError("Alguna variante no heredo el fold de su padre.")

    source_counts = selected_events.groupby("original_uuid").size()
    augmented_counts = metadata.groupby(
        ["parent_original_uuid", "augmentation_type"]
    ).size()
    for (parent_uuid, _), count in augmented_counts.items():
        if int(count) != int(source_counts.loc[parent_uuid]):
            raise RuntimeError(
                f"Numero de eventos incoherente para {parent_uuid}."
            )


def save_configuration(
    config: wst_base.ScatteringConfig,
    layout: dict[str, object],
    batch_size: int,
    selected_events: pd.DataFrame,
    output_dir: Path,
) -> None:
    rows = []
    for spec in AUGMENTATIONS:
        rows.append(
            {
                **asdict(config),
                **layout,
                "augmentation_name": spec.name,
                "augmentation_transform": spec.transform,
                "augmentation_value": spec.value,
                "augmentation_unit": spec.unit,
                "selection_seed": RANDOM_STATE,
                "wet_gold_parent_count": EXPECTED_WET_GOLD_RECORDINGS,
                "dry_gold_control_parent_count": EXPECTED_DRY_CONTROLS,
                "selected_original_event_count": len(selected_events),
                "batch_size": batch_size,
                "temporal_aggregation": "mean_over_scattering_time_positions",
                "normalization": "peak_after_augmentation",
                "time_stretch_length_policy": "center_crop_or_symmetric_zero_pad",
                "validation_processed": False,
                "test_processed": False,
                "original_wst_recomputed": False,
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "augmentation_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )


def validate_original_layout(config: wst_base.ScatteringConfig) -> None:
    original_layout_path = (
        ORIGINAL_WST_DIR / "wavelet_scattering_feature_layout.csv"
    )
    if not original_layout_path.is_file():
        raise FileNotFoundError(
            "No se encuentra el layout WST original con el que deben ser "
            f"compatibles las variantes: {original_layout_path}"
        )
    original = pd.read_csv(original_layout_path)
    scattering = wst_base.build_scattering(config)
    path_count, _, _, _ = wst_base.infer_layout(scattering, config)
    current = wst_base.feature_layout_dataframe(scattering, path_count)
    columns = ["feature_index", "feature_name", "scattering_order", "path_index"]
    if not original[columns].equals(current[columns]):
        raise RuntimeError(
            "El layout WST actual no coincide con las features originales."
        )


def print_check(
    selection: pd.DataFrame,
    train_events: pd.DataFrame,
    config: wst_base.ScatteringConfig,
    batch_size: int,
) -> None:
    selected_events = train_events.loc[
        train_events["original_uuid"].isin(selection["original_uuid"])
    ]
    event_counts = selected_events.groupby("cough_type").size().to_dict()
    augmented_counts = {
        key: value * len(AUGMENTATIONS) for key, value in event_counts.items()
    }
    print("=" * 78)
    print("CHECK - WST GOLD DATA AUGMENTATION STAGE 2")
    print("=" * 78)
    print("Fuente: exclusivamente TRAIN; VALIDATION y TEST no se leen.")
    print("Padres: 6 wet gold + 6 dry gold emparejados por fold y acuerdo.")
    print("Variantes: tempo 0.97/1.03 y pitch -0.5/+0.5 semitonos.")
    print(f"Eventos originales seleccionados dry/wet: {event_counts}")
    print(f"Eventos WST nuevos estimados dry/wet: {augmented_counts}")
    print(
        f"Longitud: {config.target_samples} muestras; batch_size={batch_size}; "
        "WST original no se recalcula."
    )
    print("\nSeleccion:")
    print(
        selection[
            [
                "pair_id",
                "selection_role",
                "original_uuid",
                "fold",
                "maximum_agreement_count",
                "event_count",
            ]
        ].to_string(index=False)
    )
    print(
        "\nRegla posterior obligatoria: cada variante solo puede entrar en el "
        "train interno cuando su padre no sea parte del fold OOF evaluado."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera auditoria y WST incrementales de augmentation gold."
    )
    parser.add_argument(
        "--action",
        choices=["check", "audit", "extract", "all"],
        default="check",
        help=(
            "check no escribe; audit crea WAV/graficas; extract crea solo las "
            "WST nuevas; all ejecuta audit y extract."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(wst_base.PRESETS),
        default="paper_q8_q1_t500_full",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size debe ser mayor que cero.")

    config = wst_base.PRESETS[args.preset]
    train_events = audio_base.load_event_manifest("train")
    votes = load_expert_votes()
    recordings = recording_table(train_events, votes)
    selection = select_parent_recordings(recordings)
    validate_original_layout(config)
    print_check(selection, train_events, config, args.batch_size)

    if args.action == "check":
        print("\nCheck completado. No se ha escrito ningun archivo.")
        return

    output_dir = OUTPUT_ROOT / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(
        output_dir / "selected_gold_parent_recordings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.action in {"audit", "all"}:
        audit = save_audit(train_events, selection, config)
        print(
            f"\nAuditoria: {len(audit)} WAV (12 originales representativos + "
            f"48 variantes) en {AUDIT_AUDIO_ROOT}"
        )
        print(f"Graficas: {GRAPH_ROOT / config.name}")

    if args.action in {"extract", "all"}:
        start = time.perf_counter()
        shape, metadata = extract_augmented_wst(
            train_events,
            selection,
            config,
            args.batch_size,
            output_dir,
        )
        elapsed = time.perf_counter() - start
        print("\n" + "=" * 78)
        print("WST GOLD AUGMENTADAS EXTRAIDAS")
        print("=" * 78)
        print(f"X_events_train_augmented: {shape} float32")
        print(
            "Grabaciones aumentadas dry/wet: "
            f"{metadata.groupby('cough_type')['original_uuid'].nunique().to_dict()}"
        )
        print(f"Tiempo total: {elapsed:.2f} s")
        print(f"Resultados: {output_dir}")
        print("WST originales reutilizables; VALIDATION y TEST intactos.")


if __name__ == "__main__":
    main()
