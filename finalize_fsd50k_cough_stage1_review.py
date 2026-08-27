"""Finaliza la auditoria manual de las toses FSD50K para Stage 1.

Este script no modifica el TXT rellenado por la revisora ni mueve audios.
Genera un nuevo CSV de metadatos con unicamente los audios que contienen
tos segun la auditoria manual.

Convencion acordada para ``fsd50k_cough_manual_review.txt``:

* respuesta vacia: SI contiene tos;
* ``no`` en la columna de respuesta o en ``notes``: NO contiene tos.

Los audios rechazados no se convierten en negativos. Se conservan en un
CSV separado, sin etiqueta final de Stage 1, para evitar introducir falsos
negativos si alguna tos fuera tenue o discutible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent

METADATA_DIR = ROOT / "metadata_fsd50k_cough_stage1_safe"
SELECTED_PATH = METADATA_DIR / "fsd50k_cough_selected_stage1.csv"
REVIEW_PATH = METADATA_DIR / "fsd50k_cough_manual_review.txt"
EXTRACTION_PATH = METADATA_DIR / "fsd50k_cough_extraction_manifest.csv"
RATINGS_PATH = (
    WORKSPACE
    / "FSD-50k_METADATA"
    / "FSD50K.metadata"
    / "pp_pnp_ratings_FSD50K.json"
)

AUDIO_DIR = (
    WORKSPACE
    / "FSD50K_DATA"
    / "FSD50K.dev_audio"
    / "FSD50K_cough_stage1_only"
)

RESOLVED_PATH = (
    METADATA_DIR / "fsd50k_cough_stage1_manual_review_resolved.csv"
)
APPROVED_PATH = METADATA_DIR / "fsd50k_cough_stage1_approved_metadata.csv"
REJECTED_PATH = METADATA_DIR / "fsd50k_cough_stage1_rejected_manual.csv"
APPROVED_IDS_PATH = METADATA_DIR / "ids_fsd50k_cough_stage1_approved.txt"
SUMMARY_PATH = METADATA_DIR / "fsd50k_cough_stage1_manual_review_summary.csv"
RATING_SUMMARY_PATH = (
    METADATA_DIR / "fsd50k_cough_stage1_manual_review_by_rating.csv"
)

COUGH_MID = "/m/01b_21"
EXPECTED_REVIEWED_COUNT = 239
YES_VALUES = {"si", "sí", "yes"}
NO_VALUES = {"no"}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontro el archivo requerido: {path}")


def normalize(value: object) -> str:
    return str(value).strip().casefold()


def load_review() -> pd.DataFrame:
    require_file(REVIEW_PATH)
    review = pd.read_csv(
        REVIEW_PATH,
        sep="\t",
        comment="#",
        dtype=str,
        keep_default_na=False,
    )

    required = {"review_order", "filename", "has_cough_si_no", "notes"}
    missing = required.difference(review.columns)
    if missing:
        raise ValueError(
            "Faltan columnas en la plantilla de revision: "
            + ", ".join(sorted(missing))
        )
    if len(review) != EXPECTED_REVIEWED_COUNT:
        raise ValueError(
            f"La revision contiene {len(review)} filas; se esperaban "
            f"{EXPECTED_REVIEWED_COUNT}."
        )
    if review["review_order"].duplicated().any():
        raise ValueError("Hay numeros de orden duplicados en la revision.")
    if review["filename"].duplicated().any():
        raise ValueError("Hay nombres de audio duplicados en la revision.")

    review["fname"] = review["filename"].str.strip().str.removesuffix(".wav")
    invalid_filename = review[
        review["fname"].eq("")
        | review["fname"].str.contains(r"[\\/]", regex=True)
    ]
    if not invalid_filename.empty:
        raise ValueError("La revision contiene nombres de audio no validos.")

    decisions: list[str] = []
    decision_sources: list[str] = []

    for row in review.itertuples(index=False):
        answer = normalize(row.has_cough_si_no)
        notes = normalize(row.notes)

        if answer in YES_VALUES:
            if notes in NO_VALUES:
                raise ValueError(
                    f"Respuesta contradictoria para {row.filename}: SI/NO."
                )
            decisions.append("SI")
            decision_sources.append("explicit_yes")
        elif answer in NO_VALUES:
            decisions.append("NO")
            decision_sources.append("explicit_no_response_column")
        elif answer == "" and notes in NO_VALUES:
            decisions.append("NO")
            decision_sources.append("explicit_no_notes_column")
        elif answer == "" and notes == "":
            decisions.append("SI")
            decision_sources.append("blank_means_yes_by_declared_protocol")
        else:
            raise ValueError(
                f"Decision no reconocida para {row.filename}: "
                f"respuesta={row.has_cough_si_no!r}, notas={row.notes!r}"
            )

    review["manual_review_decision"] = decisions
    review["manual_review_decision_source"] = decision_sources
    review["manual_review_has_cough"] = review[
        "manual_review_decision"
    ].eq("SI")
    review["manual_review_completed"] = True
    review["review_order"] = review["review_order"].astype(int)
    return review


def load_candidates_and_audio_info() -> pd.DataFrame:
    require_file(SELECTED_PATH)
    require_file(EXTRACTION_PATH)

    selected = pd.read_csv(SELECTED_PATH, dtype={"fname": str})
    extraction = pd.read_csv(EXTRACTION_PATH, dtype={"fname": str})

    if selected["fname"].duplicated().any():
        raise ValueError("Hay IDs duplicados en los metadatos seleccionados.")
    if extraction["fname"].duplicated().any():
        raise ValueError("Hay IDs duplicados en el manifiesto de extraccion.")

    audio_columns = [
        "fname",
        "file_size_bytes",
        "sha256",
        "sample_rate",
        "channels",
        "subtype",
        "frames",
        "duration_seconds",
        "extraction_status",
    ]
    missing_audio_columns = set(audio_columns).difference(extraction.columns)
    if missing_audio_columns:
        raise ValueError(
            "Faltan columnas en el manifiesto de extraccion: "
            + ", ".join(sorted(missing_audio_columns))
        )

    return selected.merge(
        extraction[audio_columns],
        on="fname",
        how="left",
        validate="one_to_one",
    )


def add_cough_ratings(metadata: pd.DataFrame) -> pd.DataFrame:
    require_file(RATINGS_PATH)
    with RATINGS_PATH.open("r", encoding="utf-8") as file_handle:
        ratings = json.load(file_handle)

    result = metadata.copy()
    rating_values = result["fname"].map(
        lambda fname: ratings.get(str(fname), {}).get(COUGH_MID, [])
    )
    result["cough_rating_values"] = rating_values.map(
        lambda values: "|".join(str(value) for value in values)
    )
    result["cough_rating_n_pp"] = rating_values.map(
        lambda values: values.count(1.0)
    )
    result["cough_rating_n_pnp"] = rating_values.map(
        lambda values: values.count(0.5)
    )
    result["cough_rating_n_uncertain"] = rating_values.map(
        lambda values: values.count(0)
    )
    result["cough_rating_n_not_present"] = rating_values.map(
        lambda values: values.count(-1)
    )

    def rating_group(values: list[float]) -> str:
        if not values:
            return "missing"
        n_pp = values.count(1.0)
        n_positive = n_pp + values.count(0.5)
        if n_pp >= 2:
            return "pp_agreement"
        if n_positive >= 2:
            return "positive_agreement_without_two_pp"
        return "single_positive"

    result["cough_rating_group"] = rating_values.map(rating_group)
    return result


def build_final_metadata() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review = load_review()
    candidates = load_candidates_and_audio_info()

    reviewed_ids = set(review["fname"])
    candidate_ids = set(candidates["fname"])
    if reviewed_ids != candidate_ids:
        missing_review = sorted(candidate_ids - reviewed_ids)
        unexpected_review = sorted(reviewed_ids - candidate_ids)
        raise ValueError(
            "Los IDs revisados no coinciden con la seleccion por licencia. "
            f"Sin revisar={missing_review[:20]}; "
            f"inesperados={unexpected_review[:20]}"
        )

    resolved = candidates.merge(
        review[
            [
                "fname",
                "review_order",
                "has_cough_si_no",
                "notes",
                "manual_review_decision",
                "manual_review_decision_source",
                "manual_review_has_cough",
                "manual_review_completed",
            ]
        ],
        on="fname",
        how="inner",
        validate="one_to_one",
    )
    resolved = add_cough_ratings(resolved)

    # Conservamos el split oficial de FSD50K como procedencia, no como el
    # split experimental que se creara en el siguiente paso.
    resolved = resolved.rename(columns={"split": "fsd50k_official_split"})

    resolved["uuid"] = resolved["fname"]
    resolved["original_uuid"] = resolved["fname"]
    resolved["uuid_segmento"] = resolved["fname"] + "_seg_0"
    resolved["cough_detected"] = 1
    resolved["dataset_origin"] = "FSD50K"
    resolved["stage1_inclusion"] = resolved["manual_review_has_cough"]
    resolved["stage1_exclusion_reason"] = resolved[
        "manual_review_has_cough"
    ].map({True: "", False: "manual_review_no_audible_cough"})

    # Solo las filas aprobadas reciben una etiqueta final. De este modo, el
    # CSV de rechazados no puede emplearse accidentalmente como negativo.
    approved_mask = resolved["manual_review_has_cough"]
    resolved["label"] = pd.Series(pd.NA, index=resolved.index, dtype="Int64")
    resolved["stage1_target"] = pd.Series(
        pd.NA,
        index=resolved.index,
        dtype="Int64",
    )
    resolved.loc[approved_mask, "label"] = 1
    resolved.loc[approved_mask, "stage1_target"] = 1

    resolved["cough_type"] = "unknown"
    resolved["cough_type_label"] = 3
    resolved["cough_type_name"] = "unknown"
    resolved["cough_type_consensus"] = "not_applicable_fsd50k"
    resolved["stage2_eligible"] = False
    resolved["stage2_exclusion_reason"] = (
        "fsd50k_has_no_dry_wet_annotation"
    )
    resolved["type_noise"] = "cough"
    resolved["start_time"] = 0.0
    resolved["end_time"] = resolved["duration_seconds"]
    resolved["duration"] = resolved["duration_seconds"]
    resolved["audio_path"] = resolved["fname"].map(
        lambda fname: str((AUDIO_DIR / f"{fname}.wav").resolve())
    )
    resolved["audio_relative_path"] = resolved["fname"].map(
        lambda fname: (
            "FSD50K_DATA/FSD50K.dev_audio/"
            f"FSD50K_cough_stage1_only/{fname}.wav"
        )
    )

    missing_audio = resolved[
        ~resolved["audio_path"].map(lambda value: Path(value).is_file())
    ]
    if not missing_audio.empty:
        raise FileNotFoundError(
            "Faltan WAV revisados: "
            + ", ".join(missing_audio["fname"].head(20))
        )
    if resolved["stage2_eligible"].any():
        raise ValueError("Una muestra FSD50K quedo habilitada para Stage 2.")

    leading_columns = [
        "uuid",
        "original_uuid",
        "uuid_segmento",
        "fname",
        "dataset_origin",
        "label",
        "stage1_target",
        "stage1_inclusion",
        "stage1_exclusion_reason",
        "manual_review_decision",
        "manual_review_has_cough",
        "manual_review_decision_source",
        "manual_review_completed",
        "review_order",
        "cough_detected",
        "cough_type",
        "cough_type_label",
        "cough_type_name",
        "cough_type_consensus",
        "stage2_eligible",
        "stage2_exclusion_reason",
        "type_noise",
        "fsd50k_official_split",
        "audio_path",
        "audio_relative_path",
        "duration",
        "start_time",
        "end_time",
    ]
    remaining_columns = [
        column for column in resolved.columns if column not in leading_columns
    ]
    resolved = resolved[leading_columns + remaining_columns].sort_values(
        "review_order"
    )

    approved = resolved[resolved["stage1_inclusion"]].copy()
    rejected = resolved[~resolved["stage1_inclusion"]].copy()

    if not approved["label"].eq(1).all():
        raise ValueError("No todas las muestras aprobadas tienen label=1.")
    if approved["label"].isna().any():
        raise ValueError("Hay muestras aprobadas sin etiqueta final.")
    if rejected["label"].notna().any():
        raise ValueError("Una muestra rechazada conserva una etiqueta final.")

    return resolved, approved, rejected


def save_outputs(
    resolved: pd.DataFrame,
    approved: pd.DataFrame,
    rejected: pd.DataFrame,
) -> None:
    resolved.to_csv(RESOLVED_PATH, index=False, encoding="utf-8-sig")
    approved.to_csv(APPROVED_PATH, index=False, encoding="utf-8-sig")
    rejected.to_csv(REJECTED_PATH, index=False, encoding="utf-8-sig")
    APPROVED_IDS_PATH.write_text(
        "".join(f"{fname}\n" for fname in approved["fname"]),
        encoding="utf-8",
    )

    summary = pd.DataFrame(
        [
            {"metric": "reviewed_total", "value": len(resolved)},
            {"metric": "approved_cough_label_1", "value": len(approved)},
            {"metric": "rejected_no_audible_cough", "value": len(rejected)},
            {
                "metric": "approved_fsd50k_official_train",
                "value": int(
                    approved["fsd50k_official_split"].eq("train").sum()
                ),
            },
            {
                "metric": "approved_fsd50k_official_val",
                "value": int(
                    approved["fsd50k_official_split"].eq("val").sum()
                ),
            },
            {
                "metric": "approved_unique_uploaders",
                "value": int(approved["uploader"].nunique()),
            },
            {
                "metric": "approved_stage2_eligible",
                "value": int(approved["stage2_eligible"].sum()),
            },
            {
                "metric": "approved_missing_audio",
                "value": int(
                    (~approved["audio_path"].map(lambda value: Path(value).is_file())).sum()
                ),
            },
        ]
    )
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    rating_summary = (
        resolved.groupby(
            ["cough_rating_group", "manual_review_decision"],
            dropna=False,
        )
        .size()
        .rename("recording_count")
        .reset_index()
        .sort_values(["cough_rating_group", "manual_review_decision"])
    )
    rating_summary.to_csv(
        RATING_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    resolved, approved, rejected = build_final_metadata()
    save_outputs(resolved, approved, rejected)

    print("=" * 78)
    print("REVISION MANUAL FSD50K COUGH FINALIZADA")
    print("=" * 78)
    print(f"Audios revisados:             {len(resolved)}")
    print(f"Aprobados con label=1:        {len(approved)}")
    print(f"Rechazados, fuera de training: {len(rejected)}")
    print(f"Uploaders entre aprobados:    {approved['uploader'].nunique()}")
    print(f"Stage 2 elegibles:            {int(approved['stage2_eligible'].sum())}")
    print(f"Metadatos aprobados: {APPROVED_PATH}")
    print(f"Auditoria completa:  {RESOLVED_PATH}")
    print("El TXT original no se ha modificado.")


if __name__ == "__main__":
    main()
