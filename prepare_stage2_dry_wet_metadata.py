"""Prepara vistas de metadatos para la clasificacion dry/wet.

Este script NO crea nuevos splits y NO cambia los folds. Parte de los splits
maestros ya congelados y genera:

* Una vista dry/wet (gold_expert + weak_expert) por split.
* Una vista separada unknown/ambiguous para el rechazo futuro.
* Un resumen de recuentos para auditar el resultado.

La unidad final de clasificacion sera ``original_uuid``. En esta fase se
conservan todas las filas/segmentos del original para que la segmentacion de
actividad posterior pueda procesarlos sin perder informacion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS_DIR = SCRIPT_DIR / "metadata_splits_multi_experiment_random"
OUTPUT_DIR = SCRIPT_DIR / "metadata_stage2_dry_wet_experiment_random"

SPLIT_FILES = {
    "train": "metadata_train_multiclass_4c.csv",
    "validation": "metadata_val_multiclass_4c.csv",
    "test": "metadata_test_multiclass_4c.csv",
}

DRY_WET_LABELS = {"dry": 0, "wet": 1}
ELIGIBLE_CONSENSUS = {"gold_expert", "weak_expert"}
EXPECTED_TRAIN_FOLDS = {0, 1, 2, 3, 4}

REQUIRED_COLUMNS = {
    "uuid_segmento",
    "original_uuid",
    "dataset_origin",
    "cough_type",
    "cough_type_label",
    "cough_type_consensus",
    "stage2_eligible",
    "stage2_gold_eval",
    "stage2_reject_challenge",
    "fold",
    "split",
    "start_time",
    "end_time",
}


def parse_boolean_column(series: pd.Series, column_name: str) -> pd.Series:
    """Convierte bool/string/0-1 a bool y rechaza valores desconocidos."""

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
    invalid = sorted(set(normalized.unique()) - set(mapping))
    if invalid:
        raise ValueError(
            f"Valores booleanos no reconocidos en {column_name}: {invalid}"
        )
    return normalized.map(mapping).astype(bool)


def load_and_validate_master_split(
    split_name: str,
    filename: str,
) -> pd.DataFrame:
    csv_path = SPLITS_DIR / filename
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encuentra el split: {csv_path}")

    df = pd.read_csv(
        csv_path,
        dtype={
            "uuid": str,
            "original_uuid": str,
            "uuid_segmento": str,
        },
    )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en {filename}: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"El split {split_name} esta vacio.")

    if df["uuid_segmento"].duplicated().any():
        duplicated = df.loc[
            df["uuid_segmento"].duplicated(keep=False),
            "uuid_segmento",
        ].head(20).tolist()
        raise ValueError(
            f"uuid_segmento duplicados en {split_name}: {duplicated}"
        )

    actual_split_values = set(df["split"].astype(str).unique())
    if actual_split_values != {split_name}:
        raise ValueError(
            f"La columna split de {filename} contiene "
            f"{actual_split_values}, se esperaba {{{split_name!r}}}."
        )

    df = df.copy()
    for column in (
        "stage2_eligible",
        "stage2_gold_eval",
        "stage2_reject_challenge",
    ):
        df[column] = parse_boolean_column(df[column], column)

    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
    df["start_time"] = pd.to_numeric(
        df["start_time"], errors="raise"
    )
    df["end_time"] = pd.to_numeric(df["end_time"], errors="raise")

    invalid_intervals = df["end_time"] <= df["start_time"]
    if invalid_intervals.any():
        examples = df.loc[
            invalid_intervals,
            ["uuid_segmento", "start_time", "end_time"],
        ].head(20)
        raise ValueError(
            "Se encontraron intervalos de audio invalidos:\n"
            f"{examples.to_string(index=False)}"
        )

    expected_eligible = (
        df["cough_type"].isin(DRY_WET_LABELS)
        & df["cough_type_consensus"].isin(ELIGIBLE_CONSENSUS)
    )
    mismatch_eligible = df["stage2_eligible"] != expected_eligible
    if mismatch_eligible.any():
        examples = df.loc[
            mismatch_eligible,
            [
                "uuid_segmento",
                "cough_type",
                "cough_type_consensus",
                "stage2_eligible",
            ],
        ].head(20)
        raise ValueError(
            "stage2_eligible no coincide con las reglas dry/wet:\n"
            f"{examples.to_string(index=False)}"
        )

    expected_gold = (
        df["cough_type"].isin(DRY_WET_LABELS)
        & (df["cough_type_consensus"] == "gold_expert")
    )
    if (df["stage2_gold_eval"] != expected_gold).any():
        raise ValueError(
            f"stage2_gold_eval no coincide con las reglas en {filename}."
        )

    expected_reject = (
        (df["cough_type"] == "unknown")
        | (df["cough_type_consensus"] == "ambiguous_expert")
    )
    if (df["stage2_reject_challenge"] != expected_reject).any():
        raise ValueError(
            "stage2_reject_challenge no coincide con las reglas en "
            f"{filename}."
        )

    if split_name == "train":
        actual_folds = set(df["fold"].unique())
        if actual_folds != EXPECTED_TRAIN_FOLDS:
            raise ValueError(
                "Los folds de train no son 0-4: "
                f"encontrados={sorted(actual_folds)}"
            )
    elif not (df["fold"] == -1).all():
        raise ValueError(
            f"Todas las filas de {split_name} deben tener fold=-1."
        )

    return df


def validate_original_consistency(
    df: pd.DataFrame,
    split_name: str,
) -> None:
    """Comprueba que los segmentos de un original heredan lo mismo."""

    columns = [
        "dataset_origin",
        "cough_type",
        "cough_type_consensus",
        "fold",
        "split",
    ]
    inconsistent = (
        df.groupby("original_uuid")[columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if inconsistent.any():
        examples = inconsistent[inconsistent].index[:20].tolist()
        raise ValueError(
            f"Originales inconsistentes en {split_name}: {examples}"
        )


def build_dry_wet_view(
    master_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    dry_wet = master_df[master_df["stage2_eligible"]].copy()
    if dry_wet.empty:
        raise ValueError(f"No hay muestras dry/wet en {split_name}.")

    invalid_types = set(dry_wet["cough_type"].unique()) - set(
        DRY_WET_LABELS
    )
    if invalid_types:
        raise ValueError(
            f"Clases no permitidas en {split_name}: {invalid_types}"
        )

    invalid_consensus = set(
        dry_wet["cough_type_consensus"].unique()
    ) - ELIGIBLE_CONSENSUS
    if invalid_consensus:
        raise ValueError(
            f"Consensos no permitidos en {split_name}: "
            f"{invalid_consensus}"
        )

    dry_wet["stage2_target"] = dry_wet["cough_type"].map(
        DRY_WET_LABELS
    ).astype(int)
    validate_original_consistency(dry_wet, split_name)

    labels_per_original = (
        dry_wet.groupby("original_uuid")["stage2_target"].nunique()
    )
    if (labels_per_original != 1).any():
        examples = labels_per_original[labels_per_original != 1]
        raise ValueError(
            "Hay originales con mas de una etiqueta dry/wet:\n"
            f"{examples.head(20).to_string()}"
        )

    if set(dry_wet["stage2_target"].unique()) != {0, 1}:
        raise ValueError(
            f"El split {split_name} no contiene ambas clases dry/wet."
        )

    return dry_wet.sort_values(
        ["fold", "stage2_target", "original_uuid", "start_time"]
    ).reset_index(drop=True)


def build_reject_challenge_view(master_df: pd.DataFrame) -> pd.DataFrame:
    challenge = master_df[
        master_df["stage2_reject_challenge"]
    ].copy()
    return challenge.sort_values(
        ["fold", "original_uuid", "start_time"]
    ).reset_index(drop=True)


def validate_no_split_overlap(
    dry_wet_views: dict[str, pd.DataFrame],
) -> None:
    uuid_sets = {
        split_name: set(df["original_uuid"].astype(str))
        for split_name, df in dry_wet_views.items()
    }
    split_names = list(uuid_sets)
    for index, first_name in enumerate(split_names):
        for second_name in split_names[index + 1 :]:
            overlap = uuid_sets[first_name] & uuid_sets[second_name]
            if overlap:
                raise ValueError(
                    f"Solapamiento {first_name}/{second_name}: "
                    f"{sorted(overlap)[:20]}"
                )


def build_summary(
    dry_wet_views: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    summary_rows: list[dict] = []
    for split_name, df in dry_wet_views.items():
        grouped = df.groupby(
            ["cough_type", "cough_type_consensus"],
            dropna=False,
        )
        for (cough_type, consensus), group in grouped:
            summary_rows.append(
                {
                    "split": split_name,
                    "cough_type": cough_type,
                    "cough_type_consensus": consensus,
                    "rows": len(group),
                    "original_uuids": group["original_uuid"].nunique(),
                }
            )
    return pd.DataFrame(summary_rows).sort_values(
        ["split", "cough_type", "cough_type_consensus"]
    ).reset_index(drop=True)


def print_view_summary(split_name: str, df: pd.DataFrame) -> None:
    print("\n" + "-" * 72)
    print(f"STAGE 2 DRY/WET - {split_name.upper()}")
    print("-" * 72)
    print(f"Filas/segmentos: {len(df)}")
    print(f"Originales: {df['original_uuid'].nunique()}")
    print("\nPor clase:")
    print(df.groupby("cough_type")["original_uuid"].nunique())
    print("\nPor clase y consenso:")
    print(
        pd.crosstab(
            df["cough_type_consensus"],
            df["cough_type"],
        )
    )
    if split_name == "train":
        print("\nOriginales por fold y clase:")
        print(
            df.groupby(["fold", "cough_type"])["original_uuid"]
            .nunique()
            .unstack(fill_value=0)
        )


def main() -> None:
    master_splits: dict[str, pd.DataFrame] = {}
    dry_wet_views: dict[str, pd.DataFrame] = {}
    challenge_views: dict[str, pd.DataFrame] = {}

    for split_name, filename in SPLIT_FILES.items():
        master = load_and_validate_master_split(split_name, filename)
        master_splits[split_name] = master
        dry_wet_views[split_name] = build_dry_wet_view(
            master, split_name
        )
        challenge_views[split_name] = build_reject_challenge_view(master)

    validate_no_split_overlap(dry_wet_views)

    for split_name, df in dry_wet_views.items():
        print_view_summary(split_name, df)

    summary = build_summary(dry_wet_views)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, df in dry_wet_views.items():
        output_path = (
            OUTPUT_DIR / f"metadata_{split_name}_stage2_dry_wet.csv"
        )
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    for split_name, df in challenge_views.items():
        output_path = (
            OUTPUT_DIR
            / f"metadata_{split_name}_stage2_reject_challenge.csv"
        )
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary_path = OUTPUT_DIR / "metadata_stage2_dry_wet_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("METADATOS STAGE 2 GENERADOS SIN MODIFICAR LOS SPLITS")
    print("=" * 72)
    print(f"Carpeta de salida: {OUTPUT_DIR}")
    print(f"Resumen: {summary_path}")
    print("Test se ha preparado, pero no debe usarse en los experimentos.")


if __name__ == "__main__":
    main()
