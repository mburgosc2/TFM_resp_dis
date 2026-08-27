"""Crea splits de Stage 1 incorporando las toses FSD50K auditadas.

Esta es una version paralela de ``splits_analysis_multiclass_4c.py``.
No modifica los splits multiclase ni los metadatos usados por Stage 2.

Cambios metodologicos principales
---------------------------------
* El objetivo es binario: no_tos=0, tos=1.
* Se incorporan solo las toses FSD50K aprobadas manualmente.
* Todos los clips FSD50K de un mismo uploader permanecen en el mismo split
  y, si estan en TRAIN, en el mismo fold interno.
* CoughVID se agrupa por UUID de grabacion.
* Se preservan los cuatro estratos origen x clase en train/validation/test.
* Los positivos FSD50K nuevos conservan el audio completo, porque su etiqueta
  es debil a nivel de clip y no se dispone del instante exacto de la tos.
* La accion predeterminada es ``audit`` y no escribe ningun split.

Uso
---
Auditar la particion propuesta::

    python splits_analysis_stage1_fsd50k_coughs.py --action audit

Guardar los splits tras revisar el resumen::

    python splits_analysis_stage1_fsd50k_coughs.py --action write
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent

BASE_METADATA_PATH = ROOT / "combined_metadata_multiclass_4c.csv"
APPROVED_COUGH_PATH = (
    ROOT
    / "metadata_fsd50k_cough_stage1_safe"
    / "fsd50k_cough_stage1_approved_metadata.csv"
)
REJECTED_COUGH_PATH = (
    ROOT
    / "metadata_fsd50k_cough_stage1_safe"
    / "fsd50k_cough_stage1_rejected_manual.csv"
)
ATTRIBUTION_PATH = (
    ROOT
    / "metadata_fsd50k_cough_stage1_safe"
    / "fsd50k_stage1_attribution_manifest.csv"
)

COUGHVID_AUDIO_DIR = WORKSPACE / "DATA"
FSD50K_NEGATIVE_AUDIO_DIR = (
    WORKSPACE
    / "FSD50K_DATA"
    / "FSD50K.dev_audio"
    / "FSD50K_negative_class_dataset"
)
FSD50K_COUGH_AUDIO_DIR = (
    WORKSPACE
    / "FSD50K_DATA"
    / "FSD50K.dev_audio"
    / "FSD50K_cough_stage1_only"
)

OUTPUT_DIR = ROOT / "metadata_splits_stage1_fsd50k_coughs_random"
TRAIN_PATH = OUTPUT_DIR / "metadata_train_stage1.csv"
VALIDATION_PATH = OUTPUT_DIR / "metadata_validation_stage1.csv"
TEST_PATH = OUTPUT_DIR / "metadata_test_stage1.csv"
RECORDINGS_PATH = OUTPUT_DIR / "metadata_recordings_stage1_all.csv"
SUMMARY_PATH = OUTPUT_DIR / "metadata_stage1_split_summary.csv"
GROUPS_PATH = OUTPUT_DIR / "metadata_stage1_group_assignments.csv"
CONFIGURATION_PATH = OUTPUT_DIR / "metadata_stage1_split_configuration.csv"

DEFAULT_SEED = 42
# Elegida entre 42..141 usando exclusivamente el equilibrio de los cuatro
# estratos de TRAIN. No se consultaron metricas, validation ni test.
DEFAULT_INNER_SEED = 94
OUTER_MICROFOLDS = 20
TEST_MICROFOLDS = 3
VALIDATION_MICROFOLDS = 3
INNER_FOLDS = 5
COUGHVID_POSITIVE_WINDOW_SECONDS = 10.0
NEGATIVE_WINDOW_SECONDS = 10.0
NEGATIVE_MAX_SECONDS = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Splits binarios de Stage 1 con toses FSD50K auditadas y "
            "agrupacion por uploader."
        )
    )
    parser.add_argument(
        "--action",
        choices=["audit", "write"],
        default="audit",
        help="audit no escribe resultados; write guarda los CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semilla usada para construir los microfolds externos.",
    )
    parser.add_argument(
        "--inner-seed",
        type=int,
        default=DEFAULT_INNER_SEED,
        help=(
            "Semilla de los cinco folds internos. El valor predeterminado "
            "se eligio solo por equilibrio de estratos dentro de TRAIN."
        ),
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontro el archivo requerido: {path}")


def bool_series(values: pd.Series) -> pd.Series:
    """Convierte booleanos leidos desde CSV de forma estricta."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.casefold()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        raise ValueError(
            "Valores booleanos no reconocidos: "
            + ", ".join(sorted(normalized[invalid].unique()))
        )
    return normalized.eq("true")


def load_original_metadata() -> pd.DataFrame:
    require_file(BASE_METADATA_PATH)
    metadata = pd.read_csv(BASE_METADATA_PATH, dtype={"uuid": str})

    required = {
        "uuid",
        "dataset_origin",
        "label",
        "quality",
        "cough_type",
        "cough_type_label",
        "cough_type_name",
        "cough_type_consensus",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(
            "Faltan columnas en el dataset original: "
            + ", ".join(sorted(missing))
        )
    if metadata["uuid"].duplicated().any():
        raise ValueError("El dataset original contiene UUID duplicados.")
    if not metadata["dataset_origin"].isin({"COUGHVID", "FSD50K"}).all():
        raise ValueError("Hay dataset_origin no reconocido en el dataset original.")

    metadata["label_multiclass_original"] = metadata["label"]
    metadata["stage1_target"] = (
        pd.to_numeric(metadata["cough_type_label"], errors="raise") > 0
    ).astype(int)
    metadata["label"] = metadata["stage1_target"]
    metadata["stage1_inclusion"] = True
    metadata["stage1_exclusion_reason"] = ""
    metadata["record_source"] = "original_combined_metadata"
    metadata["is_new_fsd50k_cough"] = False

    metadata["stage2_eligible"] = (
        metadata["cough_type"].isin(["dry", "wet"])
        & metadata["cough_type_consensus"].isin(
            ["gold_expert", "weak_expert"]
        )
    )
    metadata["stage2_gold_eval"] = (
        metadata["cough_type"].isin(["dry", "wet"])
        & metadata["cough_type_consensus"].eq("gold_expert")
    )
    metadata["stage2_reject_challenge"] = (
        metadata["cough_type"].eq("unknown")
        | metadata["cough_type_consensus"].eq("ambiguous_expert")
    )
    return metadata


def load_approved_fsd50k_coughs() -> pd.DataFrame:
    require_file(APPROVED_COUGH_PATH)
    approved = pd.read_csv(
        APPROVED_COUGH_PATH,
        dtype={"uuid": str, "fname": str, "original_uuid": str},
    )

    required = {
        "uuid",
        "fname",
        "dataset_origin",
        "label",
        "stage1_target",
        "stage1_inclusion",
        "manual_review_decision",
        "uploader",
        "cough_type_label",
        "stage2_eligible",
        "audio_path",
    }
    missing = required.difference(approved.columns)
    if missing:
        raise ValueError(
            "Faltan columnas en las toses FSD50K aprobadas: "
            + ", ".join(sorted(missing))
        )
    if approved["uuid"].duplicated().any():
        raise ValueError("Las toses FSD50K aprobadas contienen UUID duplicados.")

    approved["stage1_inclusion"] = bool_series(
        approved["stage1_inclusion"]
    )
    approved["stage2_eligible"] = bool_series(approved["stage2_eligible"])

    if not approved["stage1_inclusion"].all():
        raise ValueError("El CSV aprobado contiene muestras no incluidas.")
    if not pd.to_numeric(approved["label"], errors="raise").eq(1).all():
        raise ValueError("No todas las toses aprobadas tienen label=1.")
    if not pd.to_numeric(
        approved["stage1_target"], errors="raise"
    ).eq(1).all():
        raise ValueError("No todas las toses aprobadas tienen stage1_target=1.")
    if not approved["manual_review_decision"].eq("SI").all():
        raise ValueError("El CSV aprobado contiene una decision distinta de SI.")
    if approved["stage2_eligible"].any():
        raise ValueError("Una tos FSD50K aprobada esta habilitada para Stage 2.")
    if not approved["dataset_origin"].eq("FSD50K").all():
        raise ValueError("Una tos nueva no tiene dataset_origin=FSD50K.")

    approved["label_multiclass_original"] = approved["cough_type_label"]
    approved["label"] = 1
    approved["stage1_target"] = 1
    approved["stage1_inclusion"] = True
    approved["stage1_exclusion_reason"] = ""
    approved["record_source"] = "fsd50k_cough_manual_approved"
    approved["is_new_fsd50k_cough"] = True
    approved["stage2_eligible"] = False
    approved["stage2_gold_eval"] = False
    approved["stage2_reject_challenge"] = False
    approved["quality"] = "not_available"
    return approved


def add_fsd50k_attribution(metadata: pd.DataFrame) -> pd.DataFrame:
    require_file(ATTRIBUTION_PATH)
    attribution = pd.read_csv(ATTRIBUTION_PATH, dtype={"fname": str})
    if attribution["fname"].duplicated().any():
        raise ValueError("El manifiesto de atribucion contiene IDs duplicados.")

    required = {"fname", "uploader", "license", "license_id"}
    missing = required.difference(attribution.columns)
    if missing:
        raise ValueError(
            "Faltan columnas en el manifiesto de atribucion: "
            + ", ".join(sorted(missing))
        )

    lookup = attribution.set_index("fname")
    result = metadata.copy()
    fsd_mask = result["dataset_origin"].eq("FSD50K")
    fsd_ids = result.loc[fsd_mask, "uuid"].astype(str)

    result.loc[fsd_mask, "uploader"] = fsd_ids.map(lookup["uploader"])
    result.loc[fsd_mask, "license"] = fsd_ids.map(lookup["license"])
    result.loc[fsd_mask, "license_id"] = fsd_ids.map(lookup["license_id"])

    missing_uploader = result.loc[fsd_mask, "uploader"].isna()
    if missing_uploader.any():
        missing_ids = result.loc[fsd_mask].loc[missing_uploader, "uuid"]
        raise ValueError(
            "Hay audios FSD50K sin uploader: "
            + ", ".join(missing_ids.head(20))
        )
    if result.loc[fsd_mask, "uploader"].astype(str).str.strip().eq("").any():
        raise ValueError("Hay audios FSD50K con uploader vacio.")
    return result


def combine_recordings() -> pd.DataFrame:
    original = load_original_metadata()
    approved = load_approved_fsd50k_coughs()

    overlap = set(original["uuid"]) & set(approved["uuid"])
    if overlap:
        raise ValueError(
            "Las toses nuevas ya existen en el dataset original: "
            + ", ".join(sorted(overlap)[:20])
        )

    all_columns = sorted(set(original.columns) | set(approved.columns))
    recordings = pd.concat(
        [
            original.reindex(columns=all_columns),
            approved.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )
    recordings["uuid"] = recordings["uuid"].astype(str)
    if recordings["uuid"].duplicated().any():
        raise ValueError("El dataset Stage 1 combinado contiene UUID duplicados.")

    recordings = add_fsd50k_attribution(recordings)
    recordings["quality"] = recordings["quality"].fillna("not_available")
    recordings["stage1_target"] = pd.to_numeric(
        recordings["stage1_target"], errors="raise"
    ).astype(int)
    recordings["label"] = recordings["stage1_target"]
    recordings["is_new_fsd50k_cough"] = bool_series(
        recordings["is_new_fsd50k_cough"]
    )

    fsd_mask = recordings["dataset_origin"].eq("FSD50K")
    uploader_key = (
        recordings["uploader"].fillna("").astype(str).str.strip().str.casefold()
    )
    recordings["split_group"] = np.where(
        fsd_mask,
        "FSD50K_UPLOADER::" + uploader_key,
        "COUGHVID_UUID::" + recordings["uuid"],
    )
    recordings["split_stratum"] = (
        recordings["dataset_origin"]
        + "__"
        + recordings["stage1_target"].astype(str)
    )

    recordings["audio_path"] = recordings.apply(audio_path_for_row, axis=1)
    validate_rejected_not_included(recordings)
    add_audio_information(recordings)
    return recordings


def audio_path_for_row(row: pd.Series) -> str:
    uuid = str(row["uuid"])
    if bool(row["is_new_fsd50k_cough"]):
        path = FSD50K_COUGH_AUDIO_DIR / f"{uuid}.wav"
    elif row["dataset_origin"] == "FSD50K":
        path = FSD50K_NEGATIVE_AUDIO_DIR / f"{uuid}.wav"
    else:
        path = COUGHVID_AUDIO_DIR / f"{uuid}.wav"
    return str(path.resolve())


def validate_rejected_not_included(recordings: pd.DataFrame) -> None:
    if not REJECTED_COUGH_PATH.is_file():
        return
    rejected = pd.read_csv(REJECTED_COUGH_PATH, dtype={"fname": str})
    overlap = set(rejected["fname"]) & set(recordings["uuid"])
    if overlap:
        raise ValueError(
            "Audios rechazados manualmente entraron en Stage 1: "
            + ", ".join(sorted(overlap)[:20])
        )


def add_audio_information(recordings: pd.DataFrame) -> None:
    durations: list[float] = []
    sample_rates: list[int] = []
    channels: list[int] = []
    errors: list[str] = []

    for row in tqdm(
        recordings.itertuples(index=False),
        total=len(recordings),
        desc="Leyendo cabeceras de audio",
    ):
        path = Path(row.audio_path)
        if not path.is_file():
            errors.append(f"{row.uuid}: audio_not_found: {path}")
            durations.append(np.nan)
            sample_rates.append(-1)
            channels.append(-1)
            continue
        try:
            info = sf.info(path)
            if info.duration <= 0 or info.frames <= 0:
                raise ValueError("duracion o numero de frames no valido")
            durations.append(float(info.duration))
            sample_rates.append(int(info.samplerate))
            channels.append(int(info.channels))
        except Exception as exc:
            errors.append(f"{row.uuid}: {type(exc).__name__}: {exc}")
            durations.append(np.nan)
            sample_rates.append(-1)
            channels.append(-1)

    recordings["duration"] = durations
    recordings["audio_sample_rate_original"] = sample_rates
    recordings["audio_channels_original"] = channels

    if errors:
        raise ValueError(
            f"Hay {len(errors)} audios ausentes o no legibles:\n"
            + "\n".join(errors[:30])
        )


def subset_score(
    folds: tuple[int, ...],
    counts_by_microfold: pd.DataFrame,
    target_fraction: float,
) -> float:
    selected = counts_by_microfold.loc[list(folds)].sum(axis=0).to_numpy(float)
    totals = counts_by_microfold.sum(axis=0).to_numpy(float)
    target = target_fraction * totals
    relative_stratum_error = np.mean(
        np.abs(selected - target) / np.maximum(target, 1.0)
    )
    selected_total = float(selected.sum())
    target_total = float(target.sum())
    relative_total_error = abs(selected_total - target_total) / target_total
    return float(0.8 * relative_stratum_error + 0.2 * relative_total_error)


def best_microfold_subset(
    available_folds: Iterable[int],
    subset_size: int,
    counts_by_microfold: pd.DataFrame,
    target_fraction: float,
) -> tuple[int, ...]:
    candidates = itertools.combinations(sorted(available_folds), subset_size)
    return min(
        candidates,
        key=lambda folds: (
            subset_score(folds, counts_by_microfold, target_fraction),
            folds,
        ),
    )


def assign_outer_splits(
    recordings: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, tuple[int, ...], tuple[int, ...]]:
    result = recordings.reset_index(drop=True).copy()
    stratum_counts = result["split_stratum"].value_counts()
    if stratum_counts.min() < OUTER_MICROFOLDS:
        raise ValueError(
            "No hay suficientes grabaciones por estrato para crear "
            f"{OUTER_MICROFOLDS} microfolds:\n{stratum_counts}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=OUTER_MICROFOLDS,
        shuffle=True,
        random_state=seed,
    )
    result["outer_microfold"] = -1
    for fold, (_, held_out_indices) in enumerate(
        splitter.split(
            X=result,
            y=result["split_stratum"],
            groups=result["split_group"],
        )
    ):
        result.loc[held_out_indices, "outer_microfold"] = fold

    if (result["outer_microfold"] < 0).any():
        raise ValueError("Hay grabaciones sin microfold externo.")

    counts = pd.crosstab(
        result["outer_microfold"],
        result["split_stratum"],
    ).reindex(index=range(OUTER_MICROFOLDS), fill_value=0)

    test_folds = best_microfold_subset(
        available_folds=range(OUTER_MICROFOLDS),
        subset_size=TEST_MICROFOLDS,
        counts_by_microfold=counts,
        target_fraction=0.15,
    )
    remaining = [
        fold for fold in range(OUTER_MICROFOLDS) if fold not in test_folds
    ]
    validation_folds = best_microfold_subset(
        available_folds=remaining,
        subset_size=VALIDATION_MICROFOLDS,
        counts_by_microfold=counts,
        target_fraction=0.15,
    )

    result["split"] = np.select(
        [
            result["outer_microfold"].isin(test_folds),
            result["outer_microfold"].isin(validation_folds),
        ],
        ["test", "validation"],
        default="train",
    )
    return result, test_folds, validation_folds


def assign_inner_folds(recordings: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = recordings.copy()
    result["fold"] = -1
    train_indices = result.index[result["split"].eq("train")]
    train = result.loc[train_indices].reset_index()

    stratum_counts = train["split_stratum"].value_counts()
    if stratum_counts.min() < INNER_FOLDS:
        raise ValueError(
            "No hay suficientes grabaciones de train por estrato para "
            f"{INNER_FOLDS} folds:\n{stratum_counts}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    for fold, (_, held_out_positions) in enumerate(
        splitter.split(
            X=train,
            y=train["split_stratum"],
            groups=train["split_group"],
        )
    ):
        original_indices = train.loc[held_out_positions, "index"]
        result.loc[original_indices, "fold"] = fold

    if (result.loc[train_indices, "fold"] < 0).any():
        raise ValueError("Hay grabaciones de TRAIN sin fold interno.")
    if not result.loc[~result.index.isin(train_indices), "fold"].eq(-1).all():
        raise ValueError("Validation o test recibieron un fold interno.")
    return result


def segment_recordings(recordings: pd.DataFrame) -> pd.DataFrame:
    segments: list[dict[str, object]] = []

    for row in recordings.itertuples(index=False):
        row_dict = row._asdict()
        uuid = str(row.uuid)
        duration = float(row.duration)
        target = int(row.stage1_target)

        if target == 1:
            if bool(row.is_new_fsd50k_cough):
                intervals = [(0.0, duration)]
                policy = "full_clip_new_fsd50k_cough_weak_label"
            else:
                intervals = [
                    (0.0, min(duration, COUGHVID_POSITIVE_WINDOW_SECONDS))
                ]
                policy = "original_positive_first_10s"
        elif duration <= NEGATIVE_WINDOW_SECONDS:
            intervals = [(0.0, duration)]
            policy = "negative_full_clip_up_to_10s"
        else:
            usable_duration = min(duration, NEGATIVE_MAX_SECONDS)
            number_of_segments = int(usable_duration // NEGATIVE_WINDOW_SECONDS)
            intervals = [
                (
                    index * NEGATIVE_WINDOW_SECONDS,
                    (index + 1) * NEGATIVE_WINDOW_SECONDS,
                )
                for index in range(number_of_segments)
            ]
            policy = "negative_nonoverlap_10s_max_20s"

        if not intervals:
            raise ValueError(f"No se pudo segmentar la grabacion {uuid}.")

        for segment_index, (start_time, end_time) in enumerate(intervals):
            segment = row_dict.copy()
            segment["original_uuid"] = uuid
            segment["uuid_segmento"] = f"{uuid}_seg_{segment_index}"
            segment["start_time"] = float(start_time)
            segment["end_time"] = float(end_time)
            segment["segment_duration"] = float(end_time - start_time)
            segment["segmentation_policy"] = policy
            segments.append(segment)

    segmented = pd.DataFrame(segments)
    if segmented["uuid_segmento"].duplicated().any():
        raise ValueError("Se generaron uuid_segmento duplicados.")
    if (segmented["segment_duration"] <= 0).any():
        raise ValueError("Se generaron segmentos de duracion no positiva.")
    return segmented


def validate_partitions(recordings: pd.DataFrame, segments: pd.DataFrame) -> None:
    if recordings["uuid"].duplicated().any():
        raise ValueError("Una grabacion aparece mas de una vez antes de segmentar.")

    groups_per_split = recordings.groupby("split_group")["split"].nunique()
    if (groups_per_split > 1).any():
        raise ValueError("Un grupo aparece en mas de un split externo.")

    fsd = recordings[recordings["dataset_origin"].eq("FSD50K")]
    uploader_splits = fsd.groupby("uploader")["split"].nunique()
    if (uploader_splits > 1).any():
        raise ValueError("Un uploader FSD50K aparece en mas de un split.")

    train = recordings[recordings["split"].eq("train")]
    train_group_folds = train.groupby("split_group")["fold"].nunique()
    if (train_group_folds > 1).any():
        raise ValueError("Un grupo de TRAIN aparece en varios folds internos.")
    if set(train["fold"].astype(int).unique()) != set(range(INNER_FOLDS)):
        raise ValueError("TRAIN no contiene exactamente los folds 0..4.")
    if not recordings.loc[
        ~recordings["split"].eq("train"), "fold"
    ].eq(-1).all():
        raise ValueError("Validation o test tienen folds internos.")

    segment_split_counts = segments.groupby("original_uuid")["split"].nunique()
    if (segment_split_counts > 1).any():
        raise ValueError("Los segmentos de una grabacion cruzan splits.")
    segment_fold_counts = segments[
        segments["split"].eq("train")
    ].groupby("original_uuid")["fold"].nunique()
    if (segment_fold_counts > 1).any():
        raise ValueError("Los segmentos de una grabacion cruzan folds.")

    cross_table = pd.crosstab(
        recordings["split_stratum"],
        recordings["split"],
    )
    expected_splits = {"train", "validation", "test"}
    if set(cross_table.columns) != expected_splits:
        raise ValueError("No todos los splits contienen todos los estratos.")

    split_proportions = recordings["split"].value_counts(normalize=True)
    targets = {"train": 0.70, "validation": 0.15, "test": 0.15}
    for split, target in targets.items():
        if abs(float(split_proportions[split]) - target) > 0.05:
            raise ValueError(
                f"La proporcion de {split} se aleja demasiado del objetivo: "
                f"{split_proportions[split]:.4f} frente a {target:.4f}."
            )

    new_fsd = recordings[recordings["is_new_fsd50k_cough"]]
    if len(new_fsd) == 0:
        raise ValueError("No se incorporo ninguna tos FSD50K nueva.")
    if not new_fsd["stage1_target"].eq(1).all():
        raise ValueError("Una tos FSD50K nueva no tiene target=1.")
    if bool_series(new_fsd["stage2_eligible"]).any():
        raise ValueError("Una tos FSD50K nueva quedo habilitada en Stage 2.")


def build_group_assignments(recordings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, subset in recordings.groupby("split_group", sort=True):
        rows.append(
            {
                "split_group": group,
                "dataset_origin": "|".join(
                    sorted(subset["dataset_origin"].unique())
                ),
                "uploader": "|".join(
                    sorted(
                        value
                        for value in subset["uploader"].dropna().astype(str).unique()
                        if value.strip()
                    )
                ),
                "split": subset["split"].iloc[0],
                "fold": (
                    int(subset["fold"].iloc[0])
                    if subset["split"].iloc[0] == "train"
                    else -1
                ),
                "recording_count": len(subset),
                "no_cough_count": int(subset["stage1_target"].eq(0).sum()),
                "cough_count": int(subset["stage1_target"].eq(1).sum()),
                "new_fsd50k_cough_count": int(
                    subset["is_new_fsd50k_cough"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    recordings: pd.DataFrame,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    recording_summary = (
        recordings.groupby(
            ["split", "dataset_origin", "stage1_target", "is_new_fsd50k_cough"],
            dropna=False,
        )
        .agg(
            recording_count=("uuid", "size"),
            unique_groups=("split_group", "nunique"),
            recording_duration_seconds=("duration", "sum"),
        )
        .reset_index()
    )
    segment_summary = (
        segments.groupby(
            ["split", "dataset_origin", "stage1_target", "is_new_fsd50k_cough"],
            dropna=False,
        )
        .agg(
            segment_count=("uuid_segmento", "size"),
            segment_duration_seconds=("segment_duration", "sum"),
        )
        .reset_index()
    )
    return recording_summary.merge(
        segment_summary,
        on=["split", "dataset_origin", "stage1_target", "is_new_fsd50k_cough"],
        how="outer",
        validate="one_to_one",
    ).sort_values(
        ["split", "dataset_origin", "stage1_target", "is_new_fsd50k_cough"]
    )


def print_summary(recordings: pd.DataFrame, segments: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("SPLIT STAGE 1 CON TOSES FSD50K AUDITADAS")
    print("=" * 78)
    print(f"Grabaciones totales: {len(recordings)}")
    print(f"Segmentos totales:   {len(segments)}")
    print(
        "Toses FSD50K nuevas: "
        f"{int(recordings['is_new_fsd50k_cough'].sum())}"
    )
    print(f"Grupos totales:      {recordings['split_group'].nunique()}")

    print("\nGrabaciones por origen, clase y split:")
    print(
        pd.crosstab(
            [recordings["dataset_origin"], recordings["stage1_target"]],
            recordings["split"],
            margins=True,
        )
    )

    print("\nToses FSD50K nuevas por split:")
    print(
        recordings[recordings["is_new_fsd50k_cough"]]
        ["split"]
        .value_counts()
        .reindex(["train", "validation", "test"], fill_value=0)
    )

    print("\nProporcion global por split:")
    print(
        recordings["split"]
        .value_counts(normalize=True)
        .reindex(["train", "validation", "test"])
    )

    print("\nGrabaciones de TRAIN por estrato y fold interno:")
    train = recordings[recordings["split"].eq("train")]
    print(pd.crosstab(train["fold"], train["split_stratum"], margins=True))

    print("\nPoliticas de segmentacion:")
    print(segments["segmentation_policy"].value_counts())


def save_outputs(
    recordings: pd.DataFrame,
    segments: pd.DataFrame,
    summary: pd.DataFrame,
    groups: pd.DataFrame,
    seed: int,
    inner_seed: int,
    test_microfolds: tuple[int, ...],
    validation_microfolds: tuple[int, ...],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = segments[segments["split"].eq("train")].copy()
    validation = segments[segments["split"].eq("validation")].copy()
    test = segments[segments["split"].eq("test")].copy()

    train.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    validation.to_csv(VALIDATION_PATH, index=False, encoding="utf-8-sig")
    test.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")
    recordings.to_csv(RECORDINGS_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    groups.to_csv(GROUPS_PATH, index=False, encoding="utf-8-sig")

    configuration = pd.DataFrame(
        [
            {"parameter": "experiment", "value": "stage1_fsd50k_coughs_random"},
            {"parameter": "seed", "value": seed},
            {"parameter": "inner_seed", "value": inner_seed},
            {
                "parameter": "inner_seed_selection",
                "value": (
                    "minimum grouped stratum imbalance on TRAIN among "
                    "seeds 42..141; no model metrics"
                ),
            },
            {"parameter": "outer_microfolds", "value": OUTER_MICROFOLDS},
            {
                "parameter": "test_microfolds",
                "value": "|".join(map(str, test_microfolds)),
            },
            {
                "parameter": "validation_microfolds",
                "value": "|".join(map(str, validation_microfolds)),
            },
            {"parameter": "inner_folds", "value": INNER_FOLDS},
            {
                "parameter": "outer_stratification",
                "value": "dataset_origin__stage1_target",
            },
            {
                "parameter": "grouping_fsd50k",
                "value": "uploader_casefolded",
            },
            {"parameter": "grouping_coughvid", "value": "uuid"},
            {
                "parameter": "target_recording_ratios",
                "value": "train=0.70;validation=0.15;test=0.15",
            },
            {
                "parameter": "new_fsd50k_positive_segmentation",
                "value": "full_clip",
            },
            {
                "parameter": "stage2_usage",
                "value": "forbidden_for_new_fsd50k_coughs",
            },
        ]
    )
    configuration.to_csv(
        CONFIGURATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()
    recordings = combine_recordings()
    recordings, test_microfolds, validation_microfolds = assign_outer_splits(
        recordings,
        seed=args.seed,
    )
    recordings = assign_inner_folds(recordings, seed=args.inner_seed)
    segments = segment_recordings(recordings)
    validate_partitions(recordings, segments)

    summary = build_summary(recordings, segments)
    groups = build_group_assignments(recordings)
    print_summary(recordings, segments)
    print(f"\nMicrofolds de TEST:       {test_microfolds}")
    print(f"Microfolds de VALIDATION: {validation_microfolds}")
    print(f"Semilla de folds internos: {args.inner_seed}")

    if args.action == "audit":
        print("\nAUDITORIA COMPLETADA: no se ha escrito ningun split.")
        print(
            "Si el resumen es correcto, ejecuta:\n"
            "python .\\splits_analysis_stage1_fsd50k_coughs.py --action write"
        )
        return

    save_outputs(
        recordings=recordings,
        segments=segments,
        summary=summary,
        groups=groups,
        seed=args.seed,
        inner_seed=args.inner_seed,
        test_microfolds=test_microfolds,
        validation_microfolds=validation_microfolds,
    )
    print("\n" + "=" * 78)
    print("SPLITS STAGE 1 GUARDADOS")
    print("=" * 78)
    print(f"TRAIN:      {TRAIN_PATH}")
    print(f"VALIDATION: {VALIDATION_PATH}")
    print(f"TEST:       {TEST_PATH}")
    print(f"Resumen:    {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
