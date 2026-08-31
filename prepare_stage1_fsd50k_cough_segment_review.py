"""Segmenta y audita temporalmente las toses FSD50K de Stage 1.

Este pipeline conserva las particiones y folds ya asignados por
``splits_analysis_stage1_fsd50k_coughs.py`` y reemplaza solamente las filas
correspondientes a las 172 toses FSD50K revisadas inicialmente.

Politica temporal acordada
--------------------------
* duracion <= 10 s: se conserva el clip completo y su etiqueta positiva;
* 10 s < duracion <= 20 s: [0, 10] y [10, duracion];
* duracion > 20 s: se descarta la cola posterior a 20 s y se crean
  [0, 10] y [10, 20].

Los segmentos procedentes de clips de mas de 10 s deben revisarse de forma
independiente. La columna ``manual_decision`` acepta ``SI``, ``NO`` o
``AMBIGUO``. Los ambiguos se excluyen; los ``NO`` pasan a ser negativos de
Stage 1 y los ``SI`` permanecen como positivos. Ningun segmento cambia el
split, fold o grupo de su grabacion original.

Uso
---
1. Crear los WAV y la plantilla de revision::

    python prepare_stage1_fsd50k_cough_segment_review.py --action prepare

2. Editar ``manual_decision`` en el CSV indicado por el programa.

3. Construir los nuevos metadatos tras completar todas las decisiones::

    python prepare_stage1_fsd50k_cough_segment_review.py --action finalize

4. Auditar en cualquier momento los metadatos finales::

    python prepare_stage1_fsd50k_cough_segment_review.py --action audit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
BASE_METADATA_DIR = ROOT / "metadata_splits_stage1_fsd50k_coughs_random"
OUTPUT_METADATA_DIR = (
    ROOT / "metadata_splits_stage1_fsd50k_cough_segments_random"
)
REVIEW_DIR = ROOT / "metadata_fsd50k_cough_stage1_segment_review"
REVIEW_AUDIO_DIR = ROOT / "auditory_stage1_fsd50k_cough_segment_review"

PLAN_PATH = REVIEW_DIR / "fsd50k_cough_segment_plan.csv"
REVIEW_PATH = REVIEW_DIR / "fsd50k_cough_segment_manual_review.csv"
RESOLVED_PATH = REVIEW_DIR / "fsd50k_cough_segment_review_resolved.csv"
REVIEW_SUMMARY_PATH = REVIEW_DIR / "fsd50k_cough_segment_review_summary.csv"

METADATA_FILES = {
    "train": "metadata_train_stage1.csv",
    "validation": "metadata_validation_stage1.csv",
    "test": "metadata_test_stage1.csv",
}
WINDOW_SECONDS = 10.0
MAX_SOURCE_SECONDS = 20.0
REVIEW_SAMPLE_RATE = 16_000
TIME_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segmenta en ventanas de hasta 10 s las toses FSD50K y aplica "
            "la revision manual sin cambiar splits ni folds."
        )
    )
    parser.add_argument(
        "--action",
        choices=["prepare", "finalize", "audit"],
        default="prepare",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Permite regenerar la plantilla o los metadatos. PRECAUCION: "
            "prepare reemplaza las decisiones existentes."
        ),
    )
    return parser.parse_args()


def parse_bool(series: pd.Series, column: str) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    normalized = series.astype(str).str.strip().str.casefold()
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        examples = sorted(normalized.loc[invalid].unique())[:5]
        raise ValueError(f"Booleanos invalidos en {column}: {examples}")
    return normalized.map(mapping).astype(bool)


def load_base_splits() -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    seen_segment_ids: set[str] = set()
    groups_by_split: dict[str, set[str]] = {}

    for split, filename in METADATA_FILES.items():
        path = BASE_METADATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False, dtype={"uuid": str})
        required = {
            "audio_path",
            "duration",
            "fold",
            "is_new_fsd50k_cough",
            "original_uuid",
            "split",
            "split_group",
            "stage1_target",
            "uuid_segmento",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")
        if set(frame["split"].astype(str)) != {split}:
            raise ValueError(f"La columna split no coincide en {path}")

        frame["is_new_fsd50k_cough"] = parse_bool(
            frame["is_new_fsd50k_cough"], "is_new_fsd50k_cough"
        )
        frame["stage1_target"] = pd.to_numeric(
            frame["stage1_target"], errors="raise"
        ).astype(np.int8)
        frame["duration"] = pd.to_numeric(
            frame["duration"], errors="raise"
        ).astype(float)
        frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)

        new_rows = frame.loc[frame["is_new_fsd50k_cough"]]
        if new_rows["original_uuid"].astype(str).duplicated().any():
            raise ValueError(
                f"Una tos FSD50K nueva ya esta segmentada en el split {split}"
            )
        if not new_rows["stage1_target"].eq(1).all():
            raise ValueError("Una tos FSD50K de origen no tiene target=1")

        segment_ids = set(frame["uuid_segmento"].astype(str))
        overlap = seen_segment_ids.intersection(segment_ids)
        if overlap:
            raise ValueError(
                f"UUID de segmento compartido entre splits: {next(iter(overlap))}"
            )
        seen_segment_ids.update(segment_ids)

        groups = set(frame["split_group"].astype(str))
        for previous_split, previous_groups in groups_by_split.items():
            overlap = groups.intersection(previous_groups)
            if overlap:
                raise ValueError(
                    f"Grupo compartido entre {previous_split} y {split}: "
                    f"{next(iter(overlap))}"
                )
        groups_by_split[split] = groups
        splits[split] = frame

    total_new = sum(int(f["is_new_fsd50k_cough"].sum()) for f in splits.values())
    if total_new != 172:
        raise ValueError(
            f"Se esperaban 172 toses FSD50K aprobadas y se encontraron {total_new}"
        )
    return splits


def build_plan(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    review_order = 0
    for split in ("train", "validation", "test"):
        source = splits[split].loc[
            splits[split]["is_new_fsd50k_cough"]
        ].copy()
        source = source.sort_values("original_uuid")
        for _, row in source.iterrows():
            uuid = str(row["original_uuid"])
            duration = float(row["duration"])
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError(f"Duracion invalida para {uuid}: {duration}")

            needs_review = duration > WINDOW_SECONDS + TIME_TOLERANCE
            if needs_review:
                intervals = [
                    (0.0, WINDOW_SECONDS),
                    (WINDOW_SECONDS, min(duration, MAX_SOURCE_SECONDS)),
                ]
                policy = "fsd50k_cough_two_nonoverlap_10s_max20s"
            else:
                intervals = [(0.0, duration)]
                policy = "fsd50k_cough_full_clip_up_to_10s"

            for segment_index, (start, end) in enumerate(intervals):
                if end <= start:
                    raise ValueError(f"Intervalo vacio para {uuid}: {start}-{end}")
                segment_id = f"{uuid}_fsdcough_seg_{segment_index}"
                if needs_review:
                    review_order += 1
                    review_audio_filename = (
                        f"{review_order:03d}_{segment_id}.wav"
                    )
                else:
                    review_audio_filename = ""
                rows.append(
                    {
                        "segment_id": segment_id,
                        "original_uuid": uuid,
                        "split": split,
                        "fold": int(row["fold"]),
                        "split_group": str(row["split_group"]),
                        "audio_path": str(row["audio_path"]),
                        "source_duration_seconds": duration,
                        "start_time": start,
                        "end_time": end,
                        "segment_duration": end - start,
                        "discarded_tail_seconds": max(
                            0.0, duration - MAX_SOURCE_SECONDS
                        ),
                        "segmentation_policy": policy,
                        "requires_manual_review": needs_review,
                        "review_order": review_order if needs_review else "",
                        "review_audio_filename": review_audio_filename,
                        "automatic_decision": "" if needs_review else "SI",
                    }
                )

    plan = pd.DataFrame(rows)
    if plan["segment_id"].duplicated().any():
        raise ValueError("El plan contiene IDs de segmento duplicados")
    if len(plan) != 206:
        raise ValueError(f"Se esperaban 206 segmentos planificados y hay {len(plan)}")
    if int(plan["requires_manual_review"].sum()) != 68:
        raise ValueError("Se esperaban 68 segmentos para revision manual")
    if plan.loc[plan["requires_manual_review"], "original_uuid"].nunique() != 34:
        raise ValueError("Se esperaban 34 grabaciones largas")
    if (plan["segment_duration"] > WINDOW_SECONDS + TIME_TOLERANCE).any():
        raise ValueError("El plan contiene un segmento superior a 10 s")
    if (plan["end_time"] > MAX_SOURCE_SECONDS + TIME_TOLERANCE).any():
        raise ValueError("El plan conserva audio posterior a 20 s")
    return plan


def export_review_audio(plan: pd.DataFrame) -> None:
    REVIEW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    review = plan.loc[plan["requires_manual_review"]]
    for row in tqdm(
        review.itertuples(index=False),
        total=len(review),
        desc="Exportando segmentos de revision",
    ):
        audio_path = Path(row.audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        signal, _ = librosa.load(
            audio_path,
            sr=REVIEW_SAMPLE_RATE,
            mono=True,
            offset=float(row.start_time),
            duration=float(row.segment_duration),
        )
        if signal.size == 0 or not np.isfinite(signal).all():
            raise ValueError(f"Audio de revision invalido: {row.segment_id}")
        output_path = REVIEW_AUDIO_DIR / row.review_audio_filename
        sf.write(output_path, signal, REVIEW_SAMPLE_RATE, subtype="PCM_16")


def ensure_prepare_can_write(overwrite: bool) -> None:
    existing = [path for path in (PLAN_PATH, REVIEW_PATH) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Ya existe una plantilla de revision. No se reemplazara para evitar "
            f"perder decisiones: {existing}. Usa --overwrite solo si procede."
        )
    if overwrite and PLAN_PATH.is_file():
        previous = pd.read_csv(PLAN_PATH)
        if "review_audio_filename" in previous.columns:
            review_root = REVIEW_AUDIO_DIR.resolve()
            for value in previous["review_audio_filename"].dropna():
                filename = str(value).strip()
                if not filename:
                    continue
                path = (REVIEW_AUDIO_DIR / Path(filename).name).resolve()
                if path.parent != review_root:
                    raise ValueError(f"Ruta de auditoria no segura: {path}")
                if path.is_file():
                    path.unlink()


def prepare(overwrite: bool) -> None:
    ensure_prepare_can_write(overwrite)
    splits = load_base_splits()
    plan = build_plan(splits)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    export_review_audio(plan)
    plan.to_csv(PLAN_PATH, index=False, encoding="utf-8-sig")

    review_columns = [
        "review_order",
        "segment_id",
        "original_uuid",
        "start_time",
        "end_time",
        "segment_duration",
        "source_duration_seconds",
        "discarded_tail_seconds",
        "review_audio_filename",
    ]
    review = plan.loc[plan["requires_manual_review"], review_columns].copy()
    review["manual_decision"] = ""
    review["notes"] = ""
    review.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")

    print("=" * 78)
    print("REVISION DE SEGMENTOS FSD50K PREPARADA")
    print("=" * 78)
    print("Grabaciones <=10 s conservadas automaticamente: 138")
    print("Grabaciones >10 s que deben revisarse:          34")
    print("WAV generados para revision:                    68")
    print(f"Audios:    {REVIEW_AUDIO_DIR}")
    print(f"Plantilla: {REVIEW_PATH}")
    print("Rellena manual_decision con SI, NO o AMBIGUO en las 68 filas.")


def normalize_decision(value: object) -> str:
    text = str(value).strip().casefold()
    mapping = {
        "si": "SI",
        "sí": "SI",
        "yes": "SI",
        "1": "SI",
        "tos": "SI",
        "cough": "SI",
        "no": "NO",
        "0": "NO",
        "no_tos": "NO",
        "no-cough": "NO",
        "ambiguous": "AMBIGUO",
        "ambiguo": "AMBIGUO",
        "ambigua": "AMBIGUO",
        "x": "AMBIGUO",
    }
    return mapping.get(text, "")


def resolve_review(plan: pd.DataFrame) -> pd.DataFrame:
    if not REVIEW_PATH.is_file():
        raise FileNotFoundError(
            f"No existe {REVIEW_PATH}. Ejecuta primero --action prepare."
        )
    review = pd.read_csv(REVIEW_PATH, dtype={"segment_id": str})
    required = {"segment_id", "manual_decision", "notes"}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"Faltan columnas en la revision: {sorted(missing)}")
    if review["segment_id"].duplicated().any():
        raise ValueError("La revision contiene segment_id duplicados")

    expected_ids = set(
        plan.loc[plan["requires_manual_review"], "segment_id"].astype(str)
    )
    actual_ids = set(review["segment_id"].astype(str))
    if actual_ids != expected_ids:
        missing_ids = sorted(expected_ids - actual_ids)[:10]
        extra_ids = sorted(actual_ids - expected_ids)[:10]
        raise ValueError(
            f"La plantilla fue alterada: faltan={missing_ids}; sobran={extra_ids}"
        )

    review["resolved_decision"] = review["manual_decision"].map(
        normalize_decision
    )
    pending = review["resolved_decision"].eq("")
    if pending.any():
        examples = review.loc[pending, "segment_id"].head(10).tolist()
        raise ValueError(
            f"Quedan {int(pending.sum())} decisiones sin resolver o invalidas: "
            f"{examples}. Usa SI, NO o AMBIGUO."
        )

    resolved = plan.copy()
    decision_lookup = review.set_index("segment_id")["resolved_decision"]
    notes_lookup = review.set_index("segment_id")["notes"].fillna("")
    resolved["segment_review_decision"] = np.where(
        resolved["requires_manual_review"],
        resolved["segment_id"].map(decision_lookup),
        "SI",
    )
    resolved["segment_review_notes"] = np.where(
        resolved["requires_manual_review"],
        resolved["segment_id"].map(notes_lookup),
        "automatic_positive_from_approved_clip_up_to_10s",
    )
    resolved["segment_review_source"] = np.where(
        resolved["requires_manual_review"],
        "manual_segment_audit",
        "automatic_original_clip_audit",
    )
    resolved["stage1_inclusion"] = resolved[
        "segment_review_decision"
    ].ne("AMBIGUO")
    resolved["stage1_target"] = resolved["segment_review_decision"].map(
        {"SI": 1, "NO": 0, "AMBIGUO": np.nan}
    )
    return resolved


def apply_resolved_segments(
    splits: dict[str, pd.DataFrame], resolved: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for split, base in splits.items():
        source_rows = base.loc[base["is_new_fsd50k_cough"]].copy()
        source_lookup = source_rows.set_index(
            source_rows["original_uuid"].astype(str), drop=False
        )
        retained = base.loc[~base["is_new_fsd50k_cough"]].copy()
        retained["is_fsd50k_cough_source_recording"] = False
        retained["segment_review_decision"] = "not_applicable"
        retained["segment_review_source"] = "not_applicable"
        retained["segment_review_notes"] = ""
        retained["source_duration_before_segment_review"] = np.nan
        retained["discarded_tail_seconds"] = 0.0

        derived_rows: list[pd.Series] = []
        split_plan = resolved.loc[
            resolved["split"].eq(split) & resolved["stage1_inclusion"]
        ]
        for plan_row in split_plan.itertuples(index=False):
            if plan_row.original_uuid not in source_lookup.index:
                raise ValueError(
                    f"No se encontro la grabacion fuente {plan_row.original_uuid}"
                )
            row = source_lookup.loc[plan_row.original_uuid].copy()
            if isinstance(row, pd.DataFrame):
                raise ValueError(
                    f"Grabacion fuente duplicada: {plan_row.original_uuid}"
                )
            target = int(plan_row.stage1_target)
            row["uuid_segmento"] = plan_row.segment_id
            row["start_time"] = float(plan_row.start_time)
            row["end_time"] = float(plan_row.end_time)
            row["segment_duration"] = float(plan_row.segment_duration)
            row["segmentation_policy"] = plan_row.segmentation_policy
            row["stage1_target"] = target
            row["label"] = target
            row["split_stratum"] = f"FSD50K__{target}"
            row["is_new_fsd50k_cough"] = target == 1
            row["is_fsd50k_cough_source_recording"] = True
            row["segment_review_decision"] = plan_row.segment_review_decision
            row["segment_review_source"] = plan_row.segment_review_source
            row["segment_review_notes"] = plan_row.segment_review_notes
            row["manual_review_decision"] = plan_row.segment_review_decision
            row["manual_review_has_cough"] = target == 1
            row["manual_review_completed"] = True
            row["source_duration_before_segment_review"] = float(
                plan_row.source_duration_seconds
            )
            row["discarded_tail_seconds"] = float(
                plan_row.discarded_tail_seconds
            )
            row["record_source"] = (
                "fsd50k_cough_recording_segment_manual_cough"
                if target == 1
                else "fsd50k_cough_recording_segment_manual_no_cough"
            )
            row["stage1_inclusion"] = True
            row["stage1_exclusion_reason"] = ""
            row["stage2_eligible"] = False
            row["stage2_gold_eval"] = False
            row["stage2_reject_challenge"] = False
            row["stage2_exclusion_reason"] = (
                "fsd50k_has_no_dry_wet_annotation"
            )
            if target == 0:
                row["cough_type"] = "no_cough"
                row["cough_type_name"] = "no_cough"
                row["cough_type_label"] = 0
                row["type_noise"] = "manual_no_cough_segment"
            derived_rows.append(row)

        derived = pd.DataFrame(derived_rows)
        combined = pd.concat([retained, derived], ignore_index=True, sort=False)
        combined = combined.sort_values(
            ["split_group", "original_uuid", "start_time", "uuid_segmento"]
        ).reset_index(drop=True)
        output[split] = combined
    return output


def validate_final_splits(splits: dict[str, pd.DataFrame]) -> None:
    all_ids: set[str] = set()
    groups_by_split: dict[str, set[str]] = {}
    recordings_by_split: dict[str, set[str]] = {}

    for split, frame in splits.items():
        if frame.empty:
            raise ValueError(f"El split {split} esta vacio")
        if set(frame["split"].astype(str)) != {split}:
            raise ValueError(f"Declaracion split incorrecta en {split}")
        ids = set(frame["uuid_segmento"].astype(str))
        if len(ids) != len(frame):
            raise ValueError(f"Segmentos duplicados dentro de {split}")
        overlap = all_ids.intersection(ids)
        if overlap:
            raise ValueError(f"Segmento compartido: {next(iter(overlap))}")
        all_ids.update(ids)

        groups = set(frame["split_group"].astype(str))
        recordings = set(frame["original_uuid"].astype(str))
        for previous in groups_by_split:
            group_overlap = groups.intersection(groups_by_split[previous])
            recording_overlap = recordings.intersection(
                recordings_by_split[previous]
            )
            if group_overlap:
                raise ValueError(
                    f"Fuga de grupo entre {previous} y {split}: "
                    f"{next(iter(group_overlap))}"
                )
            if recording_overlap:
                raise ValueError(
                    f"Fuga de original_uuid entre {previous} y {split}: "
                    f"{next(iter(recording_overlap))}"
                )
        groups_by_split[split] = groups
        recordings_by_split[split] = recordings

        folds_per_group = frame.groupby("split_group")["fold"].nunique()
        if (folds_per_group > 1).any():
            raise ValueError(f"Un grupo cruza folds dentro de {split}")
        expected_folds = {0, 1, 2, 3, 4} if split == "train" else {-1}
        if set(frame["fold"].astype(int)) != expected_folds:
            raise ValueError(f"Folds inesperados en {split}")

        fsd_reviewed = frame.loc[
            parse_bool(
                frame["is_fsd50k_cough_source_recording"],
                "is_fsd50k_cough_source_recording",
            )
        ]
        durations = fsd_reviewed["end_time"].astype(float) - fsd_reviewed[
            "start_time"
        ].astype(float)
        if (durations > WINDOW_SECONDS + TIME_TOLERANCE).any():
            raise ValueError("Quedo un segmento FSD50K revisado superior a 10 s")
        if (fsd_reviewed["end_time"].astype(float) > 20 + TIME_TOLERANCE).any():
            raise ValueError("Quedo audio FSD50K posterior a 20 s")


def save_final_outputs(
    splits: dict[str, pd.DataFrame], resolved: pd.DataFrame, overwrite: bool
) -> None:
    paths = [OUTPUT_METADATA_DIR / name for name in METADATA_FILES.values()]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Ya existen metadatos finales: {existing}. Usa --overwrite si procede."
        )
    OUTPUT_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, filename in METADATA_FILES.items():
        splits[split].to_csv(
            OUTPUT_METADATA_DIR / filename, index=False, encoding="utf-8-sig"
        )

    included = resolved.loc[resolved["stage1_inclusion"]].copy()
    excluded = resolved.loc[~resolved["stage1_inclusion"]].copy()
    resolved.to_csv(RESOLVED_PATH, index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, object]] = []
    for split, frame in splits.items():
        for (origin, target), subset in frame.groupby(
            ["dataset_origin", "stage1_target"], dropna=False
        ):
            summary_rows.append(
                {
                    "split": split,
                    "dataset_origin": origin,
                    "stage1_target": int(target),
                    "segment_count": len(subset),
                    "recording_count": subset["original_uuid"].astype(str).nunique(),
                    "unique_groups": subset["split_group"].astype(str).nunique(),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_METADATA_DIR / "metadata_stage1_split_summary.csv", index=False
    )

    all_segments = pd.concat(splits.values(), ignore_index=True, sort=False)
    group_summary = (
        all_segments.groupby(
            ["split_group", "dataset_origin", "uploader", "split", "fold"],
            dropna=False,
        )
        .agg(
            recording_count=("original_uuid", "nunique"),
            segment_count=("uuid_segmento", "size"),
            no_cough_count=("stage1_target", lambda x: int((x == 0).sum())),
            cough_count=("stage1_target", lambda x: int((x == 1).sum())),
        )
        .reset_index()
    )
    group_summary.to_csv(
        OUTPUT_METADATA_DIR / "metadata_stage1_group_assignments.csv", index=False
    )

    review_summary = pd.DataFrame(
        [
            {"metric": "approved_source_recordings", "value": 172},
            {"metric": "source_recordings_up_to_10s", "value": 138},
            {"metric": "source_recordings_over_10s", "value": 34},
            {"metric": "planned_segments", "value": len(resolved)},
            {"metric": "manually_reviewed_segments", "value": 68},
            {
                "metric": "included_cough_segments",
                "value": int(included["stage1_target"].eq(1).sum()),
            },
            {
                "metric": "included_no_cough_segments",
                "value": int(included["stage1_target"].eq(0).sum()),
            },
            {"metric": "excluded_ambiguous_segments", "value": len(excluded)},
            {
                "metric": "source_recordings_trimmed_after_20s",
                "value": int(resolved["discarded_tail_seconds"].gt(0).groupby(
                    resolved["original_uuid"]
                ).any().sum()),
            },
        ]
    )
    review_summary.to_csv(REVIEW_SUMMARY_PATH, index=False)

    configuration = {
        "experiment": "stage1_fsd50k_cough_segments_random",
        "base_metadata_dir": str(BASE_METADATA_DIR),
        "split_assignment": "preserved from base experiment",
        "fold_assignment": "preserved from base experiment",
        "grouping": "CoughVID UUID; FSD50K uploader",
        "short_cough_clip_policy": "full clip when duration <=10s",
        "long_cough_clip_policy": "two consecutive segments; max 10s each",
        "maximum_retained_source_time_seconds": MAX_SOURCE_SECONDS,
        "tail_after_20s": "discarded",
        "long_segment_labels": "manual SI/NO/AMBIGUO review",
        "ambiguous_policy": "exclude",
        "stage2_use": "forbidden for all new FSD50K cough-source segments",
    }
    pd.DataFrame(configuration.items(), columns=["parameter", "value"]).to_csv(
        OUTPUT_METADATA_DIR / "metadata_stage1_split_configuration.csv",
        index=False,
    )


def finalize(overwrite: bool) -> None:
    splits = load_base_splits()
    plan = build_plan(splits)
    if PLAN_PATH.is_file():
        stored_plan = pd.read_csv(PLAN_PATH, dtype={"segment_id": str})
        if set(stored_plan["segment_id"]) != set(plan["segment_id"]):
            raise ValueError("El plan guardado no coincide con los splits fuente")
    resolved = resolve_review(plan)
    final_splits = apply_resolved_segments(splits, resolved)
    validate_final_splits(final_splits)
    save_final_outputs(final_splits, resolved, overwrite)

    counts = resolved.loc[resolved["stage1_inclusion"], "stage1_target"].value_counts()
    print("=" * 78)
    print("SEGMENTACION FSD50K FINALIZADA")
    print("=" * 78)
    print(f"Segmentos positivos FSD50K incluidos: {int(counts.get(1, 0))}")
    print(f"Segmentos negativos FSD50K incluidos: {int(counts.get(0, 0))}")
    print(
        "Segmentos ambiguos excluidos:        "
        f"{int((~resolved['stage1_inclusion']).sum())}"
    )
    print(f"Metadatos finales: {OUTPUT_METADATA_DIR}")
    print("Los splits, folds y grupos originales se han conservado.")


def audit() -> None:
    if not OUTPUT_METADATA_DIR.is_dir():
        raise FileNotFoundError(
            f"No existe {OUTPUT_METADATA_DIR}. Ejecuta antes --action finalize."
        )
    splits: dict[str, pd.DataFrame] = {}
    for split, filename in METADATA_FILES.items():
        path = OUTPUT_METADATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        splits[split] = pd.read_csv(path, low_memory=False)
    validate_final_splits(splits)

    print("=" * 78)
    print("AUDITORIA DE SPLITS FSD50K SEGMENTADOS")
    print("=" * 78)
    for split, frame in splits.items():
        source = parse_bool(
            frame["is_fsd50k_cough_source_recording"],
            "is_fsd50k_cough_source_recording",
        )
        subset = frame.loc[source]
        counts = subset["stage1_target"].astype(int).value_counts()
        print(
            f"{split:>10}: total={len(frame)} | segmentos fuente cough FSD50K="
            f"{len(subset)} | no_tos/tos="
            f"{[int(counts.get(0, 0)), int(counts.get(1, 0))]}"
        )
    print("Sin fugas entre splits/folds; duracion maxima revisada <=10 s.")


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        prepare(args.overwrite)
    elif args.action == "finalize":
        finalize(args.overwrite)
    else:
        audit()


if __name__ == "__main__":
    main()
