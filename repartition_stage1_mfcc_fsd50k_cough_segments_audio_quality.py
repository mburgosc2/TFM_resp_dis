"""Reorganiza los MFCC117 existentes para el split de calidad ``poor``.

Los audios, segmentos y parametros MFCC son identicos a los del experimento
aleatorio definitivo. Por ello no se recalculan las features: cada fila se
identifica mediante ``uuid_segmento`` y se mueve al nuevo split/fold. El
script comprueba etiquetas, exclusiones por silencio y ausencia de fugas.

Uso::

    py -3.10 .\repartition_stage1_mfcc_fsd50k_cough_segments_audio_quality.py --action check
    py -3.10 .\repartition_stage1_mfcc_fsd50k_cough_segments_audio_quality.py --action write
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_FEATURES_DIR = (
    ROOT / "features_extracted_stage1_fsd50k_cough_segments_random" / "mfcc117"
)
TARGET_METADATA_DIR = (
    ROOT / "metadata_splits_stage1_fsd50k_cough_segments_audio_quality_poor"
)
OUTPUT_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_cough_segments_audio_quality_poor"
    / "mfcc117"
)
SPLIT_FILES = {
    "train": "metadata_train_stage1.csv",
    "validation": "metadata_validation_stage1.csv",
    "test": "metadata_test_stage1.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reparticiona MFCC117 para el experimento quality poor"
    )
    parser.add_argument(
        "--action", choices=["check", "write"], default="check"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar la reparticion de features existente",
    )
    return parser.parse_args()


def load_source_features() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    manifests: list[pd.DataFrame] = []
    for split in SPLIT_FILES:
        npy_name = "val" if split == "validation" else split
        X = np.load(SOURCE_FEATURES_DIR / f"X_{npy_name}.npy")
        y = np.load(SOURCE_FEATURES_DIR / f"y_{npy_name}.npy").astype(np.int8)
        manifest = pd.read_csv(
            SOURCE_FEATURES_DIR / f"metadata_features_{split}.csv",
            low_memory=False,
            dtype={"uuid_segmento": str, "original_uuid": str},
        )
        if X.ndim != 2 or X.shape[1] != 117:
            raise ValueError(f"Features inesperadas en {split}: {X.shape}")
        if not (len(X) == len(y) == len(manifest)):
            raise ValueError(f"Datos fuente desalineados en {split}")
        if not np.array_equal(
            y, manifest["stage1_target"].to_numpy(dtype=np.int8)
        ):
            raise ValueError(f"Etiquetas fuente desalineadas en {split}")
        manifest["source_feature_split"] = split
        manifest["source_feature_position"] = np.arange(len(manifest))
        arrays.append(X)
        labels.append(y)
        manifests.append(manifest)

    X_all = np.concatenate(arrays, axis=0)
    y_all = np.concatenate(labels, axis=0)
    metadata_all = pd.concat(manifests, ignore_index=True, sort=False)
    ids = metadata_all["uuid_segmento"].astype(str)
    if ids.duplicated().any():
        duplicate = ids.loc[ids.duplicated()].iloc[0]
        raise ValueError(f"Feature duplicada para {duplicate}")
    if not np.isfinite(X_all).all():
        raise ValueError("Las features fuente contienen NaN o infinitos")
    return X_all, y_all, metadata_all


def load_target_metadata() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    seen: set[str] = set()
    group_sets: dict[str, set[str]] = {}
    for split, filename in SPLIT_FILES.items():
        path = TARGET_METADATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"No existe {path}. Ejecuta primero el script de splits con "
                "--action write."
            )
        frame = pd.read_csv(
            path,
            low_memory=False,
            dtype={"uuid_segmento": str, "original_uuid": str},
        )
        frame = frame.reset_index(drop=True)
        if not frame["split"].astype(str).eq(split).all():
            raise ValueError(f"La columna split no coincide en {path}")
        ids = set(frame["uuid_segmento"].astype(str))
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(
                f"Segmento compartido entre splits: {next(iter(overlap))}"
            )
        seen.update(ids)
        groups = set(frame["split_group"].astype(str))
        for previous_split, previous_groups in group_sets.items():
            group_overlap = groups.intersection(previous_groups)
            if group_overlap:
                raise ValueError(
                    f"Grupo compartido entre {previous_split} y {split}: "
                    f"{next(iter(group_overlap))}"
                )
        group_sets[split] = groups
        result[split] = frame
    return result


def load_exclusions() -> pd.DataFrame:
    path = SOURCE_FEATURES_DIR / "feature_extraction_exclusions.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(
        path,
        low_memory=False,
        dtype={"uuid_segmento": str, "original_uuid": str},
    )


def build_repartition() -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    pd.DataFrame,
    pd.DataFrame,
]:
    X_all, y_all, source_manifest = load_source_features()
    target = load_target_metadata()
    exclusions = load_exclusions()

    source_ids = source_manifest["uuid_segmento"].astype(str)
    source_lookup = pd.Series(np.arange(len(source_ids)), index=source_ids)
    target_all = pd.concat(target.values(), ignore_index=True, sort=False)
    target_ids = set(target_all["uuid_segmento"].astype(str))
    feature_ids = set(source_ids)
    exclusion_ids = (
        set(exclusions["uuid_segmento"].astype(str))
        if not exclusions.empty
        else set()
    )
    if feature_ids - target_ids:
        raise ValueError("Hay features sin segmento en los nuevos metadatos")
    missing_features = target_ids - feature_ids
    if missing_features != exclusion_ids:
        unexpected_missing = missing_features - exclusion_ids
        unexpected_exclusion = exclusion_ids - missing_features
        raise ValueError(
            "Las features ausentes no coinciden con las exclusiones por silencio. "
            f"Ausencias inesperadas={sorted(unexpected_missing)[:5]}; "
            f"exclusiones inesperadas={sorted(unexpected_exclusion)[:5]}"
        )

    audit_columns = [
        "uuid_segmento",
        "loaded_samples",
        "loaded_duration_seconds",
        "peak_amplitude",
        "near_silence",
        "source_feature_split",
        "source_feature_position",
    ]
    source_audit = source_manifest[audit_columns].copy()
    packages: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    summary_rows: list[dict[str, object]] = []

    for split, raw_metadata in target.items():
        raw = raw_metadata.copy()
        raw["source_row"] = np.arange(len(raw))
        accepted = raw["uuid_segmento"].astype(str).isin(feature_ids)
        selected = raw.loc[accepted].copy().reset_index(drop=True)
        selected_ids = selected["uuid_segmento"].astype(str)
        source_positions = source_lookup.loc[selected_ids].to_numpy(dtype=int)
        X = X_all[source_positions].astype(np.float32, copy=False)
        y = selected["stage1_target"].to_numpy(dtype=np.int8)
        if not np.array_equal(y, y_all[source_positions]):
            raise ValueError(f"Cambio de etiqueta detectado en {split}")

        selected = selected.merge(
            source_audit,
            on="uuid_segmento",
            how="left",
            validate="one_to_one",
        )
        if selected["loaded_samples"].isna().any():
            raise ValueError(f"Falta la auditoria de extraccion en {split}")
        selected.insert(0, "feature_row", np.arange(len(selected)))
        packages[split] = (X, y, selected)
        summary_rows.append(
            {
                "split": split,
                "input_segments": len(raw),
                "extracted_segments": len(selected),
                "excluded_near_silence": int((~accepted).sum()),
                "error_count": 0,
                "no_cough_segments": int((y == 0).sum()),
                "cough_segments": int((y == 1).sum()),
                "new_fsd50k_cough_segments": int(
                    selected["is_new_fsd50k_cough"]
                    .astype(str)
                    .str.casefold()
                    .isin(["true", "1"])
                    .sum()
                ),
                "poor_cough_segments": int(
                    (
                        selected["stage1_target"].eq(1)
                        & selected["dataset_origin"].eq("COUGHVID")
                        & selected["quality"].astype(str).str.casefold().eq("poor")
                    ).sum()
                ),
            }
        )

    exclusion_output = exclusions.copy()
    if not exclusion_output.empty:
        assignment_parts: list[pd.DataFrame] = []
        for split, frame in target.items():
            assignment = frame[["uuid_segmento"]].copy()
            assignment["new_split"] = split
            assignment["new_source_row"] = np.arange(len(frame))
            assignment_parts.append(assignment)
        assignment_all = pd.concat(assignment_parts, ignore_index=True)
        exclusion_output = exclusion_output.drop(
            columns=["split", "source_row"], errors="ignore"
        ).merge(
            assignment_all,
            on="uuid_segmento",
            how="left",
            validate="one_to_one",
        )
        if exclusion_output["new_split"].isna().any():
            raise ValueError("Una exclusion no aparece en los nuevos metadatos")
        exclusion_output = exclusion_output.rename(
            columns={"new_split": "split", "new_source_row": "source_row"}
        )

    return packages, pd.DataFrame(summary_rows), exclusion_output


def validate_packages(
    packages: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]
) -> None:
    group_sets: dict[str, set[str]] = {}
    for split, (X, y, manifest) in packages.items():
        if not (len(X) == len(y) == len(manifest)):
            raise ValueError(f"Reparticion desalineada en {split}")
        if X.shape[1] != 117 or not np.isfinite(X).all():
            raise ValueError(f"Features invalidas en {split}: {X.shape}")
        groups = set(manifest["split_group"].astype(str))
        for previous_split, previous_groups in group_sets.items():
            overlap = groups.intersection(previous_groups)
            if overlap:
                raise ValueError(
                    f"Fuga de grupos entre {previous_split} y {split}: "
                    f"{next(iter(overlap))}"
                )
        group_sets[split] = groups

    train_manifest = packages["train"][2]
    train_folds = train_manifest["fold"].to_numpy(dtype=int)
    if set(train_folds) != {0, 1, 2, 3, 4}:
        raise ValueError("TRAIN no contiene los folds 0..4")
    if (
        train_manifest.groupby("split_group")["fold"].nunique().gt(1).any()
    ):
        raise ValueError("Un grupo de TRAIN cruza folds")

    for split in ["validation", "test"]:
        if not packages[split][2]["fold"].astype(int).eq(-1).all():
            raise ValueError(f"{split} contiene folds internos")
    test = packages["test"][2]
    positives = test[test["stage1_target"].eq(1)]
    if not (
        positives["dataset_origin"].eq("COUGHVID")
        & positives["quality"].astype(str).str.casefold().eq("poor")
    ).all():
        raise ValueError("TEST contiene positivos distintos de COUGHVID poor")


def print_report(
    packages: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    summary: pd.DataFrame,
) -> None:
    print("=" * 78)
    print("MFCC117 REPARTICIONADOS - AUDIO QUALITY POOR")
    print("=" * 78)
    print(f"Features fuente: {SOURCE_FEATURES_DIR}")
    print("No se ha decodificado audio ni recalculado ningun MFCC.")
    for row in summary.to_dict(orient="records"):
        split = str(row["split"])
        X, y, _ = packages[split]
        print(
            f"{split:>10}: X={X.shape}; no_tos/tos="
            f"{np.bincount(y, minlength=2).tolist()}; "
            f"poor={row['poor_cough_segments']}; "
            f"FSD50K_tos={row['new_fsd50k_cough_segments']}; "
            f"silencios_excluidos={row['excluded_near_silence']}"
        )


def write_outputs(
    packages: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    summary: pd.DataFrame,
    exclusions: pd.DataFrame,
    overwrite: bool,
) -> None:
    sentinel = OUTPUT_DIR / "feature_configuration.csv"
    if sentinel.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {OUTPUT_DIR}. Usa --overwrite para regenerarlo."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, (X, y, manifest) in packages.items():
        npy_name = "val" if split == "validation" else split
        np.save(OUTPUT_DIR / f"X_{npy_name}.npy", X)
        np.save(OUTPUT_DIR / f"y_{npy_name}.npy", y)
        manifest.to_csv(
            OUTPUT_DIR / f"metadata_features_{split}.csv", index=False
        )
    np.save(
        OUTPUT_DIR / "folds_train.npy",
        packages["train"][2]["fold"].to_numpy(dtype=np.int8),
    )
    summary.to_csv(OUTPUT_DIR / "feature_extraction_summary.csv", index=False)
    exclusions.to_csv(
        OUTPUT_DIR / "feature_extraction_exclusions.csv", index=False
    )

    error_path = SOURCE_FEATURES_DIR / "feature_extraction_errors.csv"
    errors = pd.read_csv(error_path) if error_path.is_file() else pd.DataFrame()
    errors.to_csv(OUTPUT_DIR / "feature_extraction_errors.csv", index=False)

    source_configuration = pd.read_csv(
        SOURCE_FEATURES_DIR / "feature_configuration.csv"
    )
    configuration = dict(
        zip(source_configuration["parameter"], source_configuration["value"])
    )
    configuration.update(
        {
            "experiment": (
                "stage1_mfcc117_fsd50k_cough_segments_audio_quality_poor"
            ),
            "metadata_dir": str(TARGET_METADATA_DIR),
            "output_dir": str(OUTPUT_DIR),
            "source_feature_dir": str(SOURCE_FEATURES_DIR),
            "feature_reuse": (
                "exact MFCC117 rows remapped by uuid_segmento; no recomputation"
            ),
            "split_protocol": (
                "COUGHVID poor coughs forced to TEST; grouped negatives held out"
            ),
            "test_usage": (
                "feature repartition only; no model selection or test metrics"
            ),
        }
    )
    pd.DataFrame(
        configuration.items(), columns=["parameter", "value"]
    ).to_csv(OUTPUT_DIR / "feature_configuration.csv", index=False)
    print(f"\nFeatures reparticionadas: {OUTPUT_DIR}")


def main() -> None:
    args = parse_args()
    packages, summary, exclusions = build_repartition()
    validate_packages(packages)
    print_report(packages, summary)
    if args.action == "write":
        write_outputs(
            packages, summary, exclusions, overwrite=args.overwrite
        )
    else:
        print("\nCHECK completado: no se ha escrito ningun archivo.")


if __name__ == "__main__":
    main()
