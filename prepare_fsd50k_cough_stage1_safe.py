"""Audita y extrae de forma segura las toses de FSD50K para Stage 1.

Principios del script
---------------------
* No realiza ninguna descarga de Internet.
* Solo admite mediante lista blanca exacta CC0 1.0 y CC BY 3.0.
* La seleccion se obtiene de la etiqueta exacta ``Cough``.
* La accion predeterminada es ``audit`` y no toca ningun WAV.
* ``extract`` usa las seis partes locales del ZIP multivolumen y no las borra.
* Nunca sobrescribe un audio existente.
* Los positivos FSD50K se etiquetan exclusivamente para Stage 1 y quedan
  excluidos explicitamente de Stage 2 porque no tienen etiqueta dry/wet.

Uso recomendado
---------------
1. Solo auditoria::

       python prepare_fsd50k_cough_stage1_safe.py --action audit

2. Extraccion, verificando antes el MD5 de los 18 GB de archivos ZIP::

       python prepare_fsd50k_cough_stage1_safe.py \
           --action extract \
           --verify-archive-md5

El segundo comando puede tardar varios minutos incluso antes de extraer, ya
que calcula el MD5 de las seis partes completas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent

GROUND_TRUTH_PATH = (
    WORKSPACE_DIR
    / "FSD-50k_METADATA"
    / "FSD50K.ground_truth"
    / "dev.csv"
)
CLIP_METADATA_PATH = (
    WORKSPACE_DIR
    / "FSD-50k_METADATA"
    / "FSD50K.metadata"
    / "dev_clips_info_FSD50K.json"
)

FSD50K_DATA_DIR = WORKSPACE_DIR / "FSD50K_DATA"
ARCHIVE_PATH = FSD50K_DATA_DIR / "FSD50K.dev_audio.zip"
AUDIO_SEARCH_ROOT = FSD50K_DATA_DIR / "FSD50K.dev_audio"
NEGATIVE_AUDIO_DIR = AUDIO_SEARCH_ROOT / "FSD50K_negative_class_dataset"
COUGH_AUDIO_DIR = AUDIO_SEARCH_ROOT / "FSD50K_cough_stage1_only"

OUTPUT_DIR = SCRIPT_DIR / "metadata_fsd50k_cough_stage1_safe"

LICENSE_AUDIT_PATH = OUTPUT_DIR / "fsd50k_cough_license_audit.csv"
SELECTED_METADATA_PATH = OUTPUT_DIR / "fsd50k_cough_selected_stage1.csv"
REJECTED_METADATA_PATH = OUTPUT_DIR / "fsd50k_cough_rejected_license.csv"
SELECTED_IDS_PATH = OUTPUT_DIR / "ids_fsd50k_cough_stage1_selected.txt"
ATTRIBUTION_MANIFEST_PATH = OUTPUT_DIR / "fsd50k_stage1_attribution_manifest.csv"
SUMMARY_PATH = OUTPUT_DIR / "fsd50k_cough_stage1_summary.csv"
EXTRACTION_MANIFEST_PATH = OUTPUT_DIR / "fsd50k_cough_extraction_manifest.csv"
MANUAL_REVIEW_PATH = OUTPUT_DIR / "fsd50k_cough_manual_review.txt"


# Lista blanca deliberadamente estrecha. Cualquier valor no incluido se
# rechaza, aunque parezca otra licencia Creative Commons permisiva.
ALLOWED_LICENSES = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "https://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
    "https://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
}

EXPECTED_COUGH_HIERARCHY = {
    "Cough",
    "Human_voice",
    "Respiratory_sounds",
}

# Tamano y MD5 publicados por FSD50K en Zenodo (record 4060432).
ARCHIVE_PARTS = {
    "FSD50K.dev_audio.z01": {
        "size": 3_221_225_472,
        "md5": "faa7cf4cc076fc34a44a479a5ed862a3",
    },
    "FSD50K.dev_audio.z02": {
        "size": 3_221_225_472,
        "md5": "8f9b66153e68571164fb1315d00bc7bc",
    },
    "FSD50K.dev_audio.z03": {
        "size": 3_221_225_472,
        "md5": "1196ef47d267a993d30fa98af54b7159",
    },
    "FSD50K.dev_audio.z04": {
        "size": 3_221_225_472,
        "md5": "d088ac4e11ba53daf9f7574c11cccac9",
    },
    "FSD50K.dev_audio.z05": {
        "size": 3_221_225_472,
        "md5": "81356521aa159accd3c35de22da28c7f",
    },
    "FSD50K.dev_audio.zip": {
        "size": 2_306_663_327,
        "md5": "c480d119b8f7a7e32fdb58f3ea4d6c5a",
    },
}

EXPECTED_WAV_SAMPLE_RATE = 44_100
EXPECTED_WAV_CHANNELS = 1
EXPECTED_WAV_SUBTYPE = "PCM_16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audita o extrae toses FSD50K con licencias CC0/CC-BY para "
            "Stage 1. No descarga ni borra archivos."
        )
    )
    parser.add_argument(
        "--action",
        choices=["audit", "extract"],
        default="audit",
        help="audit solo genera CSV; extract extrae los WAV aprobados.",
    )
    parser.add_argument(
        "--verify-archive-md5",
        action="store_true",
        help=(
            "Calcula y compara el MD5 oficial de las seis partes antes de "
            "extraer. Lee aproximadamente 18 GB."
        ),
    )
    parser.add_argument(
        "--seven-zip",
        type=Path,
        default=None,
        help="Ruta opcional a 7z.exe si no se encuentra automaticamente.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontro {description}: {path}")


def label_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        token.strip()
        for token in str(value).split(",")
        if token.strip()
    }


def joined_tokens(tokens: Iterable[str]) -> str:
    return "|".join(sorted(tokens))


def load_fsd50k_metadata() -> pd.DataFrame:
    require_file(GROUND_TRUTH_PATH, "el ground truth de FSD50K")
    require_file(CLIP_METADATA_PATH, "los metadatos de clips de FSD50K")

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH,
        dtype={"fname": str},
    )

    with CLIP_METADATA_PATH.open("r", encoding="utf-8") as file_handle:
        metadata_json = json.load(file_handle)

    clip_metadata = pd.DataFrame.from_dict(metadata_json, orient="index")
    clip_metadata.index = clip_metadata.index.astype(str)
    clip_metadata.index.name = "fname"
    clip_metadata = clip_metadata.reset_index()

    required_ground_truth = {"fname", "labels", "mids", "split"}
    required_clip_metadata = {"fname", "title", "uploader", "license"}

    missing_ground_truth = required_ground_truth.difference(
        ground_truth.columns
    )
    missing_clip_metadata = required_clip_metadata.difference(
        clip_metadata.columns
    )

    if missing_ground_truth:
        raise ValueError(
            "Faltan columnas en dev.csv: "
            + ", ".join(sorted(missing_ground_truth))
        )
    if missing_clip_metadata:
        raise ValueError(
            "Faltan columnas en dev_clips_info: "
            + ", ".join(sorted(missing_clip_metadata))
        )

    merged = ground_truth.merge(
        clip_metadata[["fname", "title", "uploader", "license"]],
        on="fname",
        how="left",
        validate="one_to_one",
    )

    if merged["license"].isna().any():
        missing_ids = merged.loc[merged["license"].isna(), "fname"].tolist()
        raise ValueError(
            "Hay audios del ground truth sin licencia en los metadatos: "
            f"{missing_ids[:20]}"
        )

    merged["label_tokens_set"] = merged["labels"].map(label_tokens)
    return merged


def build_audio_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    if not root.is_dir():
        return dict(index)

    for audio_path in root.rglob("*.wav"):
        if audio_path.is_file():
            index[audio_path.stem].append(audio_path.resolve())

    return dict(index)


def path_status(paths: list[Path]) -> str:
    if not paths:
        return "missing"
    if len(paths) == 1:
        return "already_present"
    return "duplicate_paths"


def attribution_text(row: pd.Series) -> str:
    title = str(row.get("title", "")).strip() or "Untitled"
    uploader = str(row.get("uploader", "")).strip() or "Unknown uploader"
    fname = str(row["fname"])
    license_id = str(row["license_id"])
    license_url = str(row["license"])
    return (
        f'"{title}" by {uploader}; Freesound ID {fname}; '
        f"{license_id} ({license_url}); accessed through FSD50K."
    )


def build_cough_audit(
    metadata: pd.DataFrame,
    audio_index: dict[str, list[Path]],
) -> pd.DataFrame:
    cough = metadata[
        metadata["label_tokens_set"].map(lambda tokens: "Cough" in tokens)
    ].copy()

    cough["license"] = cough["license"].astype(str).str.strip()
    cough["license_id"] = cough["license"].map(ALLOWED_LICENSES)
    cough["license_allowed"] = cough["license_id"].notna()
    cough["license_rejection_reason"] = cough["license_allowed"].map(
        {True: "", False: "license_not_in_exact_allowlist"}
    )

    cough["additional_non_hierarchical_labels"] = cough[
        "label_tokens_set"
    ].map(
        lambda tokens: joined_tokens(tokens - EXPECTED_COUGH_HIERARCHY)
    )
    cough["has_additional_non_hierarchical_labels"] = cough[
        "additional_non_hierarchical_labels"
    ].ne("")
    cough["audio_profile"] = cough[
        "has_additional_non_hierarchical_labels"
    ].map(
        {
            False: "cough_with_expected_hierarchy_only",
            True: "cough_with_additional_labels_review_required",
        }
    )

    paths_by_id = cough["fname"].map(
        lambda fname: audio_index.get(str(fname), [])
    )
    cough["local_audio_status"] = paths_by_id.map(path_status)
    cough["local_paths"] = paths_by_id.map(
        lambda paths: "|".join(str(path) for path in paths)
    )
    cough["conflict_in_negative_directory"] = paths_by_id.map(
        lambda paths: any(NEGATIVE_AUDIO_DIR.resolve() in path.parents for path in paths)
    )

    cough["selection_status"] = cough["license_allowed"].map(
        {True: "selected_license_safe", False: "rejected_license"}
    )
    cough["manual_review_status"] = cough[
        "has_additional_non_hierarchical_labels"
    ].map(
        {
            False: "recommended_random_audit",
            True: "mandatory_review_before_training",
        }
    )

    # Contrato de etiquetas: positivo en Stage 1, nunca dry/wet en Stage 2.
    cough["dataset_origin"] = "FSD50K"
    cough["label"] = 1
    cough["stage1_target"] = 1
    cough["cough_type"] = "unknown"
    cough["cough_type_label"] = 3
    cough["cough_type_consensus"] = "not_applicable_fsd50k"
    cough["stage2_eligible"] = False
    cough["stage2_exclusion_reason"] = "fsd50k_has_no_dry_wet_annotation"
    cough["source_url"] = cough["fname"].map(
        lambda fname: f"https://freesound.org/s/{fname}/"
    )
    cough["planned_transformations"] = (
        "source WAV preserved; model preprocessing may resample to 16 kHz "
        "mono and extract features"
    )
    cough["attribution_text"] = cough.apply(attribution_text, axis=1)
    cough["labels_exact"] = cough["label_tokens_set"].map(joined_tokens)

    return cough.drop(columns=["label_tokens_set"]).sort_values(
        ["license_allowed", "split", "fname"],
        ascending=[False, True, True],
    )


def validate_existing_negative_audio(
    metadata: pd.DataFrame,
    audio_index: dict[str, list[Path]],
) -> pd.DataFrame:
    negative_paths = sorted(NEGATIVE_AUDIO_DIR.glob("*.wav"))
    negative_ids = {path.stem for path in negative_paths}

    negative = metadata[metadata["fname"].isin(negative_ids)].copy()
    if len(negative) != len(negative_ids):
        missing_metadata = sorted(negative_ids - set(negative["fname"]))
        raise ValueError(
            "Hay negativos locales sin metadatos FSD50K: "
            f"{missing_metadata[:20]}"
        )

    negative["license"] = negative["license"].astype(str).str.strip()
    negative["license_id"] = negative["license"].map(ALLOWED_LICENSES)

    invalid_license = negative[negative["license_id"].isna()]
    if not invalid_license.empty:
        raise ValueError(
            "Hay negativos locales con licencia fuera de la lista blanca:\n"
            + invalid_license[["fname", "license"]]
            .head(20)
            .to_string(index=False)
        )

    cough_conflicts = negative[
        negative["label_tokens_set"].map(lambda tokens: "Cough" in tokens)
    ]
    if not cough_conflicts.empty:
        raise ValueError(
            "Hay audios Cough dentro de la carpeta negativa:\n"
            + cough_conflicts[["fname", "labels"]]
            .head(20)
            .to_string(index=False)
        )

    negative["labels_exact"] = negative["label_tokens_set"].map(joined_tokens)
    negative["dataset_origin"] = "FSD50K"
    negative["label"] = 0
    negative["stage1_target"] = 0
    negative["cough_type"] = "no_cough"
    negative["cough_type_label"] = 0
    negative["cough_type_consensus"] = "not_applicable_fsd50k"
    negative["stage2_eligible"] = False
    negative["stage2_exclusion_reason"] = "fsd50k_negative_stage1_only"
    negative["source_url"] = negative["fname"].map(
        lambda fname: f"https://freesound.org/s/{fname}/"
    )
    negative["local_audio_status"] = "already_present"
    negative["local_paths"] = negative["fname"].map(
        lambda fname: "|".join(
            str(path) for path in audio_index.get(str(fname), [])
        )
    )
    negative["planned_transformations"] = (
        "source WAV preserved; model preprocessing may resample to 16 kHz "
        "mono and extract features"
    )
    negative["attribution_text"] = negative.apply(attribution_text, axis=1)

    return negative.drop(columns=["label_tokens_set"]).sort_values("fname")


def attribution_columns() -> list[str]:
    return [
        "fname",
        "dataset_origin",
        "stage1_target",
        "cough_type",
        "stage2_eligible",
        "stage2_exclusion_reason",
        "labels",
        "labels_exact",
        "mids",
        "split",
        "title",
        "uploader",
        "license_id",
        "license",
        "source_url",
        "planned_transformations",
        "attribution_text",
        "local_audio_status",
        "local_paths",
    ]


def create_manual_review_template(selected: pd.DataFrame) -> None:
    """Crea una plantilla ciega y no sobrescribe revisiones existentes."""
    if MANUAL_REVIEW_PATH.exists():
        return

    review_order = selected[["fname"]].sample(
        frac=1.0,
        random_state=20260827,
    ).reset_index(drop=True)

    lines = [
        "# AUDITORIA MANUAL DE TOSES FSD50K PARA STAGE 1",
        "# Escucha cada WAV y escribe SI o NO en la tercera columna.",
        "# Si tienes dudas, deja la respuesta vacia y explicalo en notas.",
        "# No cambies el numero de orden ni el nombre del archivo.",
        "review_order\tfilename\thas_cough_si_no\tnotes",
    ]
    lines.extend(
        f"{index:03d}\t{fname}.wav\t\t"
        for index, fname in enumerate(review_order["fname"], start=1)
    )
    MANUAL_REVIEW_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def save_reports(
    cough_audit: pd.DataFrame,
    negative_manifest: pd.DataFrame,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = cough_audit[cough_audit["license_allowed"]].copy()
    rejected = cough_audit[~cough_audit["license_allowed"]].copy()

    if selected["fname"].duplicated().any():
        raise ValueError("Hay IDs duplicados en las toses seleccionadas.")
    if selected["conflict_in_negative_directory"].any():
        conflicts = selected.loc[
            selected["conflict_in_negative_directory"],
            ["fname", "local_paths"],
        ]
        raise ValueError(
            "Una tos seleccionada ya aparece en la carpeta negativa:\n"
            + conflicts.to_string(index=False)
        )

    cough_audit.to_csv(LICENSE_AUDIT_PATH, index=False, encoding="utf-8-sig")
    selected.to_csv(SELECTED_METADATA_PATH, index=False, encoding="utf-8-sig")
    create_manual_review_template(selected)
    rejected.to_csv(REJECTED_METADATA_PATH, index=False, encoding="utf-8-sig")
    SELECTED_IDS_PATH.write_text(
        "".join(f"{fname}\n" for fname in selected["fname"]),
        encoding="utf-8",
    )

    selected_attribution = selected[attribution_columns()].copy()
    attribution = pd.concat(
        [negative_manifest[attribution_columns()], selected_attribution],
        ignore_index=True,
    ).sort_values(["stage1_target", "fname"])

    if attribution["fname"].duplicated().any():
        duplicates = attribution.loc[
            attribution["fname"].duplicated(keep=False),
            ["fname", "stage1_target", "local_paths"],
        ]
        raise ValueError(
            "El manifiesto de atribucion contiene IDs duplicados:\n"
            + duplicates.head(20).to_string(index=False)
        )

    attribution.to_csv(
        ATTRIBUTION_MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        [
            {"metric": "cough_candidates_total", "value": len(cough_audit)},
            {"metric": "cough_selected_license_safe", "value": len(selected)},
            {"metric": "cough_rejected_license", "value": len(rejected)},
            {
                "metric": "cough_selected_cc_by_3_0",
                "value": int((selected["license_id"] == "CC-BY-3.0").sum()),
            },
            {
                "metric": "cough_selected_cc0_1_0",
                "value": int((selected["license_id"] == "CC0-1.0").sum()),
            },
            {
                "metric": "cough_selected_train",
                "value": int((selected["split"] == "train").sum()),
            },
            {
                "metric": "cough_selected_validation",
                "value": int((selected["split"] == "val").sum()),
            },
            {
                "metric": "cough_additional_labels_manual_review",
                "value": int(
                    selected["has_additional_non_hierarchical_labels"].sum()
                ),
            },
            {
                "metric": "cough_already_present",
                "value": int(
                    (selected["local_audio_status"] == "already_present").sum()
                ),
            },
            {
                "metric": "cough_missing_to_extract",
                "value": int(
                    (selected["local_audio_status"] == "missing").sum()
                ),
            },
            {
                "metric": "existing_negative_audio_license_safe",
                "value": len(negative_manifest),
            },
            {
                "metric": "stage2_eligible_fsd50k_cough",
                "value": int(selected["stage2_eligible"].sum()),
            },
        ]
    )
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    return selected


def run_audit() -> pd.DataFrame:
    metadata = load_fsd50k_metadata()
    audio_index = build_audio_index(AUDIO_SEARCH_ROOT)
    cough_audit = build_cough_audit(metadata, audio_index)
    negative_manifest = validate_existing_negative_audio(metadata, audio_index)
    selected = save_reports(cough_audit, negative_manifest)

    print("=" * 78)
    print("AUDITORIA FSD50K COUGH PARA STAGE 1")
    print("=" * 78)
    print(f"Candidatas con etiqueta exacta Cough: {len(cough_audit)}")
    print(f"Seleccionadas por licencia:         {len(selected)}")
    print(f"Rechazadas por licencia:            {len(cough_audit) - len(selected)}")
    print(
        "  CC BY 3.0:                      "
        f"{int((selected['license_id'] == 'CC-BY-3.0').sum())}"
    )
    print(
        "  CC0 1.0:                        "
        f"{int((selected['license_id'] == 'CC0-1.0').sum())}"
    )
    print(
        "Con etiquetas adicionales:        "
        f"{int(selected['has_additional_non_hierarchical_labels'].sum())}"
    )
    print(
        "Ya presentes / por extraer:       "
        f"{int((selected['local_audio_status'] == 'already_present').sum())} / "
        f"{int((selected['local_audio_status'] == 'missing').sum())}"
    )
    print(f"Negativos existentes auditados:     {len(negative_manifest)}")
    print(f"Stage 2 elegibles:                   {int(selected['stage2_eligible'].sum())}")
    print(f"Informes: {OUTPUT_DIR}")
    print(f"Revision manual: {MANUAL_REVIEW_PATH}")
    print("No se ha extraido, descargado ni borrado ningun audio.")

    return selected


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - se usa para integridad, no seguridad.
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive_parts(verify_md5: bool) -> None:
    for filename, expected in ARCHIVE_PARTS.items():
        path = FSD50K_DATA_DIR / filename
        require_file(path, f"la parte del archivo {filename}")

        observed_size = path.stat().st_size
        if observed_size != expected["size"]:
            raise ValueError(
                f"Tamano incorrecto para {filename}: {observed_size}; "
                f"esperado: {expected['size']}"
            )

        print(f"OK tamano: {filename} ({observed_size:,} bytes)")

        if verify_md5:
            print(f"Calculando MD5 de {filename}...")
            observed_md5 = md5_file(path)
            if observed_md5 != expected["md5"]:
                raise ValueError(
                    f"MD5 incorrecto para {filename}: {observed_md5}; "
                    f"esperado: {expected['md5']}"
                )
            print(f"OK MD5:    {filename}")


def find_seven_zip(explicit_path: Path | None) -> Path:
    candidates: list[Path] = []

    if explicit_path is not None:
        candidates.append(explicit_path)

    discovered = shutil.which("7z") or shutil.which("7zz") or shutil.which("7za")
    if discovered:
        candidates.append(Path(discovered))

    candidates.append(Path(r"C:\Program Files\7-Zip\7z.exe"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "No se encontro 7-Zip. Instala 7-Zip o indica su ruta con "
        "--seven-zip."
    )


def validate_wav(path: Path) -> dict[str, object]:
    try:
        info = sf.info(path)
    except Exception as exc:
        raise ValueError(f"WAV no legible {path}: {exc}") from exc

    if info.frames <= 0 or info.duration <= 0:
        raise ValueError(f"WAV vacio o sin duracion valida: {path}")
    if info.samplerate != EXPECTED_WAV_SAMPLE_RATE:
        raise ValueError(
            f"Sample rate inesperado en {path}: {info.samplerate}; "
            f"esperado: {EXPECTED_WAV_SAMPLE_RATE}"
        )
    if info.channels != EXPECTED_WAV_CHANNELS:
        raise ValueError(
            f"Numero de canales inesperado en {path}: {info.channels}; "
            f"esperado: {EXPECTED_WAV_CHANNELS}"
        )
    if info.subtype != EXPECTED_WAV_SUBTYPE:
        raise ValueError(
            f"Subtipo WAV inesperado en {path}: {info.subtype}; "
            f"esperado: {EXPECTED_WAV_SUBTYPE}"
        )

    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "subtype": str(info.subtype),
        "frames": int(info.frames),
        "duration_seconds": float(info.duration),
    }


def extract_selected_audio(
    selected: pd.DataFrame,
    seven_zip_path: Path,
) -> None:
    duplicate_rows = selected[selected["local_audio_status"] == "duplicate_paths"]
    if not duplicate_rows.empty:
        raise ValueError(
            "No se puede extraer mientras existan IDs en varias rutas:\n"
            + duplicate_rows[["fname", "local_paths"]].to_string(index=False)
        )

    missing = selected[selected["local_audio_status"] == "missing"].copy()
    if missing.empty:
        print("Todos los audios seleccionados ya existen. No se extrae nada.")
        return

    COUGH_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="fsd50k_cough_staging_",
        dir=FSD50K_DATA_DIR,
    ) as temporary_directory:
        staging_dir = Path(temporary_directory)
        list_path = staging_dir / "archive_members.txt"
        archive_members = [
            f"FSD50K.dev_audio\\{fname}.wav"
            for fname in missing["fname"]
        ]
        list_path.write_text(
            "\n".join(archive_members) + "\n",
            encoding="utf-8",
        )

        command = [
            str(seven_zip_path),
            "x",
            str(ARCHIVE_PATH),
            f"-o{staging_dir}",
            "-y",
            "-aoa",
            f"@{list_path}",
        ]

        print(f"Extrayendo {len(missing)} WAV desde el archivo local...")
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "7-Zip no pudo completar la extraccion.\n"
                f"STDOUT:\n{completed.stdout[-4000:]}\n"
                f"STDERR:\n{completed.stderr[-4000:]}"
            )

        validation_rows: list[dict[str, object]] = []
        staged_paths: dict[str, Path] = {}

        # Primero se validan todos los archivos. No se mueve ninguno si falla
        # cualquier elemento del lote.
        for fname in missing["fname"]:
            staged_path = staging_dir / "FSD50K.dev_audio" / f"{fname}.wav"
            if not staged_path.is_file():
                raise FileNotFoundError(
                    f"7-Zip no extrajo el miembro esperado: {staged_path}"
                )

            audio_info = validate_wav(staged_path)
            staged_paths[str(fname)] = staged_path
            validation_rows.append(
                {
                    "fname": str(fname),
                    "source_archive_member": (
                        f"FSD50K.dev_audio\\{fname}.wav"
                    ),
                    "destination_path": str(
                        (COUGH_AUDIO_DIR / f"{fname}.wav").resolve()
                    ),
                    "file_size_bytes": staged_path.stat().st_size,
                    "sha256": sha256_file(staged_path),
                    **audio_info,
                    "extraction_status": "validated_before_move",
                }
            )

        for row in validation_rows:
            fname = str(row["fname"])
            destination = COUGH_AUDIO_DIR / f"{fname}.wav"
            if destination.exists():
                raise FileExistsError(
                    "El destino aparecio durante la extraccion; no se "
                    f"sobrescribe: {destination}"
                )
            shutil.move(str(staged_paths[fname]), str(destination))
            row["extraction_status"] = "extracted_and_validated"

        pd.DataFrame(validation_rows).to_csv(
            EXTRACTION_MANIFEST_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    # Regenera los informes para registrar las rutas que ahora existen.
    updated_selected = run_audit()
    still_missing = updated_selected[
        updated_selected["local_audio_status"] == "missing"
    ]
    if not still_missing.empty:
        raise RuntimeError(
            "La extraccion termino, pero siguen faltando audios: "
            + ", ".join(still_missing["fname"].head(20))
        )

    print("=" * 78)
    print("EXTRACCION SEGURA COMPLETADA")
    print("=" * 78)
    print(f"Audios nuevos extraidos: {len(validation_rows)}")
    print(f"Carpeta: {COUGH_AUDIO_DIR}")
    print(f"Manifiesto: {EXTRACTION_MANIFEST_PATH}")
    print("No se ha descargado ni borrado ningun archivo ZIP.")


def main() -> None:
    args = parse_args()
    selected = run_audit()

    if args.action == "audit":
        return

    validate_archive_parts(verify_md5=args.verify_archive_md5)
    seven_zip_path = find_seven_zip(args.seven_zip)
    print(f"7-Zip: {seven_zip_path}")
    extract_selected_audio(selected, seven_zip_path)


if __name__ == "__main__":
    main()
