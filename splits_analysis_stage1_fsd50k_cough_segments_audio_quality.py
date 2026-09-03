"""Construye el split Stage 1 de robustez frente a calidad ``poor``.

El punto de partida son los metadatos definitivos del experimento aleatorio,
despues de segmentar y revisar manualmente las toses FSD50K. Se cambian solo
las asignaciones de split y fold; no se modifica ningun audio ni etiqueta.

Protocolo
---------
* todas las toses COUGHVID con ``quality=poor`` se reservan para TEST;
* TEST no contiene otras muestras positivas;
* las toses FSD50K (calidad no disponible) quedan en TRAIN/VALIDATION;
* se reserva para TEST una muestra agrupada de negativos;
* TRAIN/VALIDATION se separan por grupo y TRAIN recibe cinco folds agrupados;
* un UUID COUGHVID o uploader FSD50K nunca cruza splits ni folds.

Uso::

    py -3.10 .\splits_analysis_stage1_fsd50k_cough_segments_audio_quality.py --action check
    py -3.10 .\splits_analysis_stage1_fsd50k_cough_segments_audio_quality.py --action write
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "metadata_splits_stage1_fsd50k_cough_segments_random"
OUTPUT_DIR = (
    ROOT / "metadata_splits_stage1_fsd50k_cough_segments_audio_quality_poor"
)

SPLIT_FILES = {
    "train": "metadata_train_stage1.csv",
    "validation": "metadata_validation_stage1.csv",
    "test": "metadata_test_stage1.csv",
}
RANDOM_STATE = 42
TEST_NEGATIVE_FRACTION = 0.15
VALIDATION_FRACTION_OF_COMPLETE_DATASET = 0.15
TEST_NEGATIVE_MICROFOLDS = 20
DEVELOPMENT_MICROFOLDS = 17
INNER_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea el split agrupado de Stage 1 con toses COUGHVID poor "
            "reservadas para TEST."
        )
    )
    parser.add_argument(
        "--action",
        choices=["check", "write"],
        default="check",
        help="check audita sin escribir; write guarda los nuevos metadatos",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar el directorio de salida ya generado",
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


def load_source_metadata() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_split, filename in SPLIT_FILES.items():
        path = SOURCE_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False, dtype={"uuid": str})
        if not frame["split"].astype(str).eq(source_split).all():
            raise ValueError(f"La columna split no coincide en {path}")
        frame["source_random_split"] = source_split
        frame["source_random_fold"] = pd.to_numeric(
            frame["fold"], errors="raise"
        ).astype(int)
        frames.append(frame)

    metadata = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "dataset_origin",
        "fold",
        "is_new_fsd50k_cough",
        "original_uuid",
        "quality",
        "split",
        "split_group",
        "stage1_target",
        "stage2_eligible",
        "uuid_segmento",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Faltan columnas en los metadatos: {sorted(missing)}")

    metadata["stage1_target"] = pd.to_numeric(
        metadata["stage1_target"], errors="raise"
    ).astype(np.int8)
    metadata["is_new_fsd50k_cough"] = parse_bool(
        metadata["is_new_fsd50k_cough"], "is_new_fsd50k_cough"
    )
    metadata["stage2_eligible"] = parse_bool(
        metadata["stage2_eligible"], "stage2_eligible"
    )
    metadata["quality"] = (
        metadata["quality"].fillna("not_available").astype(str).str.casefold()
    )
    metadata["dataset_origin"] = metadata["dataset_origin"].astype(str)
    metadata["split_group"] = metadata["split_group"].astype(str)
    metadata["uuid_segmento"] = metadata["uuid_segmento"].astype(str)

    if metadata["uuid_segmento"].duplicated().any():
        duplicate = metadata.loc[
            metadata["uuid_segmento"].duplicated(), "uuid_segmento"
        ].iloc[0]
        raise ValueError(f"uuid_segmento duplicado: {duplicate}")
    if not metadata["stage1_target"].isin([0, 1]).all():
        raise ValueError("stage1_target contiene valores distintos de 0/1")

    new_fsd = metadata["is_new_fsd50k_cough"]
    invalid_new = new_fsd & (
        metadata["dataset_origin"].ne("FSD50K")
        | metadata["stage1_target"].ne(1)
        | metadata["stage2_eligible"]
    )
    if invalid_new.any():
        bad = metadata.loc[invalid_new, "uuid_segmento"].iloc[0]
        raise ValueError(f"Tos FSD50K nueva mal configurada: {bad}")

    return metadata.sort_values(
        ["split_group", "original_uuid", "start_time", "uuid_segmento"],
        kind="stable",
    ).reset_index(drop=True)


def detailed_stratum(frame: pd.DataFrame) -> pd.Series:
    target = frame["stage1_target"].astype(str)
    origin = frame["dataset_origin"].astype(str)
    quality = frame["quality"].astype(str)
    return np.where(
        frame["stage1_target"].eq(1) & origin.eq("COUGHVID"),
        origin + "__" + target + "__" + quality,
        origin + "__" + target,
    )


def assign_microfolds(
    frame: pd.DataFrame,
    strata: pd.Series,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    counts = pd.Series(strata).value_counts()
    if counts.min() < n_splits:
        raise ValueError(
            f"No hay suficientes filas por estrato para {n_splits} "
            f"microfolds:\n{counts}"
        )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds = np.full(len(frame), -1, dtype=int)
    for fold, (_, held_out) in enumerate(
        splitter.split(
            X=np.zeros(len(frame)),
            y=np.asarray(strata),
            groups=frame["split_group"].to_numpy(),
        )
    ):
        folds[held_out] = fold
    if np.any(folds < 0):
        raise ValueError("Hay filas sin microfold asignado")
    return folds


def choose_microfold_subset(
    frame: pd.DataFrame,
    microfolds: np.ndarray,
    strata: pd.Series,
    target_counts: pd.Series,
    candidate_sizes: tuple[int, ...],
) -> tuple[int, ...]:
    work = pd.DataFrame(
        {
            "microfold": microfolds,
            "stratum": np.asarray(strata),
        }
    )
    counts = pd.crosstab(work["microfold"], work["stratum"])
    counts = counts.reindex(
        index=sorted(np.unique(microfolds)),
        columns=sorted(set(counts.columns).union(target_counts.index)),
        fill_value=0,
    )
    target = target_counts.reindex(counts.columns, fill_value=0).to_numpy(float)
    target_total = float(target.sum())

    def score(selection: tuple[int, ...]) -> tuple[float, float, tuple[int, ...]]:
        selected = counts.loc[list(selection)].sum(axis=0).to_numpy(float)
        stratum_error = float(
            np.mean(np.abs(selected - target) / np.maximum(target, 1.0))
        )
        total_error = abs(float(selected.sum()) - target_total) / max(
            target_total, 1.0
        )
        return 0.8 * stratum_error + 0.2 * total_error, total_error, selection

    candidates = (
        selection
        for size in candidate_sizes
        for selection in itertools.combinations(counts.index.tolist(), size)
    )
    return min(candidates, key=score)


def assign_splits(metadata: pd.DataFrame) -> pd.DataFrame:
    result = metadata.copy()
    poor_positive = (
        result["dataset_origin"].eq("COUGHVID")
        & result["stage1_target"].eq(1)
        & result["quality"].eq("poor")
    )
    if int(poor_positive.sum()) != 270:
        raise ValueError(
            "Se esperaban 270 toses COUGHVID poor y se encontraron "
            f"{int(poor_positive.sum())}"
        )

    forced_test_groups = set(result.loc[poor_positive, "split_group"])
    other_positive = result["stage1_target"].eq(1) & ~poor_positive
    forced_development_groups = set(result.loc[other_positive, "split_group"])
    overlap = forced_test_groups.intersection(forced_development_groups)
    if overlap:
        raise ValueError(
            "Un grupo contiene positivos poor y positivos de desarrollo: "
            f"{next(iter(overlap))}"
        )

    negative_only = result[
        result["stage1_target"].eq(0)
        & ~result["split_group"].isin(forced_test_groups)
        & ~result["split_group"].isin(forced_development_groups)
    ].copy()
    negative_strata = negative_only["dataset_origin"].astype(str) + "__0"
    negative_microfolds = assign_microfolds(
        negative_only,
        negative_strata,
        TEST_NEGATIVE_MICROFOLDS,
        RANDOM_STATE,
    )
    all_negative = result[result["stage1_target"].eq(0)]
    negative_targets = (
        (all_negative["dataset_origin"].astype(str) + "__0")
        .value_counts()
        .mul(TEST_NEGATIVE_FRACTION)
    )
    expected_fraction_of_eligible = (
        TEST_NEGATIVE_FRACTION * len(all_negative) / len(negative_only)
    )
    expected_size = int(
        round(TEST_NEGATIVE_MICROFOLDS * expected_fraction_of_eligible)
    )
    candidate_sizes = tuple(
        size
        for size in range(max(1, expected_size - 1), expected_size + 2)
        if size < TEST_NEGATIVE_MICROFOLDS
    )
    test_negative_folds = choose_microfold_subset(
        negative_only,
        negative_microfolds,
        negative_strata,
        negative_targets,
        candidate_sizes,
    )
    test_negative_groups = set(
        negative_only.loc[
            np.isin(negative_microfolds, test_negative_folds), "split_group"
        ]
    )

    result["split"] = "development"
    result.loc[result["split_group"].isin(forced_test_groups), "split"] = "test"
    result.loc[result["split_group"].isin(test_negative_groups), "split"] = "test"
    result["quality_split_role"] = "development"
    result.loc[poor_positive, "quality_split_role"] = "forced_poor_cough_test"
    result.loc[
        result["split_group"].isin(test_negative_groups), "quality_split_role"
    ] = "held_out_negative_test"

    development = result[result["split"].eq("development")].copy()
    development_strata = pd.Series(
        detailed_stratum(development), index=development.index
    )
    development_microfolds = assign_microfolds(
        development,
        development_strata,
        DEVELOPMENT_MICROFOLDS,
        RANDOM_STATE + 1,
    )
    validation_fraction_in_development = (
        VALIDATION_FRACTION_OF_COMPLETE_DATASET / (1.0 - TEST_NEGATIVE_FRACTION)
    )
    validation_targets = development_strata.value_counts().mul(
        validation_fraction_in_development
    )
    validation_folds = choose_microfold_subset(
        development,
        development_microfolds,
        development_strata,
        validation_targets,
        (3,),
    )
    validation_groups = set(
        development.loc[
            np.isin(development_microfolds, validation_folds), "split_group"
        ]
    )
    result.loc[result["split"].eq("development"), "split"] = "train"
    result.loc[result["split_group"].isin(validation_groups), "split"] = (
        "validation"
    )

    result["fold"] = -1
    train = result[result["split"].eq("train")].copy()
    train_strata = pd.Series(detailed_stratum(train), index=train.index)
    inner_folds = assign_microfolds(
        train,
        train_strata,
        INNER_FOLDS,
        RANDOM_STATE + 2,
    )
    result.loc[train.index, "fold"] = inner_folds
    result["fold"] = result["fold"].astype(int)
    result["outer_microfold"] = -1
    return result


def validate_result(metadata: pd.DataFrame) -> None:
    group_splits = metadata.groupby("split_group")["split"].nunique()
    if (group_splits > 1).any():
        bad = group_splits.loc[group_splits > 1].index[0]
        raise ValueError(f"Fuga de grupo entre splits: {bad}")

    train = metadata[metadata["split"].eq("train")]
    group_folds = train.groupby("split_group")["fold"].nunique()
    if (group_folds > 1).any():
        bad = group_folds.loc[group_folds > 1].index[0]
        raise ValueError(f"Fuga de grupo entre folds: {bad}")
    if set(train["fold"].unique()) != set(range(INNER_FOLDS)):
        raise ValueError("TRAIN no contiene exactamente los folds 0..4")
    if not metadata.loc[~metadata["split"].eq("train"), "fold"].eq(-1).all():
        raise ValueError("VALIDATION o TEST recibieron folds internos")

    poor_positive = (
        metadata["dataset_origin"].eq("COUGHVID")
        & metadata["stage1_target"].eq(1)
        & metadata["quality"].eq("poor")
    )
    if not metadata.loc[poor_positive, "split"].eq("test").all():
        raise ValueError("Alguna tos COUGHVID poor no esta en TEST")
    if (poor_positive & ~metadata["split"].eq("test")).any():
        raise ValueError("Una tos poor entro en desarrollo")

    test_positive = metadata["split"].eq("test") & metadata["stage1_target"].eq(1)
    expected_test_positive = (
        metadata["dataset_origin"].eq("COUGHVID")
        & metadata["quality"].eq("poor")
    )
    if not expected_test_positive.loc[test_positive].all():
        raise ValueError("TEST contiene una tos que no es COUGHVID poor")
    if (
        metadata["split"].eq("test") & metadata["is_new_fsd50k_cough"]
    ).any():
        raise ValueError("Una tos FSD50K con calidad desconocida entro en TEST")
    if not set(metadata["split"].unique()) == {"train", "validation", "test"}:
        raise ValueError("Falta alguno de los tres splits")


def make_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    return (
        metadata.groupby(
            ["split", "dataset_origin", "stage1_target", "quality"],
            dropna=False,
        )
        .agg(
            n_segments=("uuid_segmento", "size"),
            n_recordings=("original_uuid", "nunique"),
            n_groups=("split_group", "nunique"),
        )
        .reset_index()
        .sort_values(["split", "dataset_origin", "stage1_target", "quality"])
    )


def print_report(metadata: pd.DataFrame) -> None:
    print("=" * 78)
    print("SPLIT STAGE 1 - ROBUSTEZ AUDIO QUALITY POOR")
    print("=" * 78)
    print("Los positivos de TEST son exclusivamente toses COUGHVID poor.")
    print("Las toses FSD50K (quality no disponible) quedan en desarrollo.")
    for split in ["train", "validation", "test"]:
        subset = metadata[metadata["split"].eq(split)]
        counts = np.bincount(
            subset["stage1_target"].to_numpy(dtype=int), minlength=2
        ).tolist()
        poor = int(
            (
                subset["stage1_target"].eq(1)
                & subset["dataset_origin"].eq("COUGHVID")
                & subset["quality"].eq("poor")
            ).sum()
        )
        fsd_cough = int(subset["is_new_fsd50k_cough"].sum())
        print(
            f"{split:>10}: segmentos={len(subset)}; no_tos/tos={counts}; "
            f"tos_poor={poor}; tos_FSD50K={fsd_cough}; "
            f"grupos={subset['split_group'].nunique()}"
        )
    print("\nDistribucion detallada:")
    print(make_summary(metadata).to_string(index=False))


def write_outputs(metadata: pd.DataFrame, overwrite: bool) -> None:
    expected_paths = [OUTPUT_DIR / filename for filename in SPLIT_FILES.values()]
    if any(path.exists() for path in expected_paths) and not overwrite:
        raise FileExistsError(
            f"Ya existen metadatos en {OUTPUT_DIR}. Usa --overwrite para regenerarlos."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, filename in SPLIT_FILES.items():
        subset = metadata[metadata["split"].eq(split)].copy()
        subset = subset.sort_values(
            ["split_group", "original_uuid", "start_time", "uuid_segmento"],
            kind="stable",
        ).reset_index(drop=True)
        subset.to_csv(OUTPUT_DIR / filename, index=False)

    make_summary(metadata).to_csv(
        OUTPUT_DIR / "metadata_stage1_split_summary.csv", index=False
    )
    (
        metadata.groupby("split_group", as_index=False)
        .agg(
            split=("split", "first"),
            fold=("fold", "first"),
            dataset_origin=("dataset_origin", "first"),
            uploader=("uploader", "first"),
            n_segments=("uuid_segmento", "size"),
            n_positive_segments=("stage1_target", "sum"),
            quality_split_role=("quality_split_role", "first"),
        )
        .sort_values(["split", "fold", "split_group"])
        .to_csv(
            OUTPUT_DIR / "metadata_stage1_group_assignments.csv", index=False
        )
    )
    configuration = {
        "experiment": "stage1_fsd50k_cough_segments_audio_quality_poor",
        "source_metadata_dir": str(SOURCE_DIR),
        "output_metadata_dir": str(OUTPUT_DIR),
        "random_state": RANDOM_STATE,
        "forced_test_positive_rule": (
            "stage1_target=1 AND dataset_origin=COUGHVID AND quality=poor"
        ),
        "test_positive_policy": "only forced COUGHVID poor coughs",
        "fsd50k_positive_policy": (
            "quality not available; restricted to TRAIN or VALIDATION"
        ),
        "test_negative_target_fraction": TEST_NEGATIVE_FRACTION,
        "validation_target_fraction_complete_dataset": (
            VALIDATION_FRACTION_OF_COMPLETE_DATASET
        ),
        "grouping": "CoughVID UUID; FSD50K uploader",
        "inner_folds": INNER_FOLDS,
        "hyperparameter_or_threshold_selection_with_test": False,
    }
    pd.DataFrame(
        configuration.items(), columns=["parameter", "value"]
    ).to_csv(OUTPUT_DIR / "metadata_stage1_split_configuration.csv", index=False)
    print(f"\nMetadatos guardados: {OUTPUT_DIR}")


def main() -> None:
    args = parse_args()
    metadata = assign_splits(load_source_metadata())
    validate_result(metadata)
    print_report(metadata)
    if args.action == "write":
        write_outputs(metadata, overwrite=args.overwrite)
    else:
        print("\nCHECK completado: no se ha escrito ningun archivo.")


if __name__ == "__main__":
    main()
