import os
from collections import Counter

import pandas as pd


# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
PATH_METADATA_BASE = (
    r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM"
    r"\DATA\metadata_compiled.csv"
)

PATH_OUTPUT_COUGHVID_FILTRADO = (
    r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM"
    r"\TFM_resp_dis\coughvid_metadata_filt.csv"
)

QUALITY_COLS = [f"quality_{i}" for i in range(1, 5)]
COUGH_TYPE_COLS = [f"cough_type_{i}" for i in range(1, 5)]

VALID_COUGH_TYPES = {"dry", "wet", "unknown"}


# =============================================================================
# 2. FUNCIONES AUXILIARES
# =============================================================================
def normalize_cough_type(value):
    """
    Normaliza una anotación de tipo de tos.

    Returns
    -------
    str | None
        'dry', 'wet', 'unknown' o None si no hay anotación.

    Raises
    ------
    ValueError
        Si aparece una etiqueta diferente de las esperadas.
    """
    if pd.isna(value):
        return None

    normalized_value = str(value).strip().lower()

    if normalized_value in {"", "nan", "none"}:
        return None

    if normalized_value not in VALID_COUGH_TYPES:
        raise ValueError(
            f"Etiqueta de cough_type no reconocida: {value!r}. "
            f"Valores permitidos: {sorted(VALID_COUGH_TYPES)}"
        )

    return normalized_value


def resolve_cough_type_consensus(row):
    """
    Determina la etiqueta final de tipo de tos utilizando cough_type_1..4.

    Reglas
    ------
    1. Si al menos 3 expertos coinciden:
       - Se utiliza esa etiqueta.
       - consensus = 'gold_expert'.

    2. Si existe un ganador único con 1 o 2 votos:
       - Se utiliza la etiqueta más frecuente.
       - consensus = 'weak_expert'.

    3. Si existe empate entre las etiquetas más frecuentes:
       - cough_type = 'unknown'.
       - consensus = 'ambiguous_expert'.

    4. Si no existe ninguna anotación de tipo:
       - Si quality == 'no_cough', el tipo de tos no aplica:
         cough_type = 'no_cough' y consensus = 'not_applicable'.
       - En cualquier otro caso:
         cough_type = 'unknown' y consensus = 'ambiguous_expert'.
    """
    votes = []

    for column in COUGH_TYPE_COLS:
        vote = normalize_cough_type(row[column])

        if vote is not None:
            votes.append(vote)

    # -------------------------------------------------------------------------
    # No existe ninguna anotación de tipo de tos
    # -------------------------------------------------------------------------
    if len(votes) == 0:
        quality = row.get("quality", pd.NA)

        if not pd.isna(quality):
            quality = str(quality).strip().lower()

        # En un audio no-cough, cough_type no es aplicable.
        if quality == "no_cough":
            return pd.Series(
                {
                    "cough_type": "no_cough",
                    "cough_type_consensus": "not_applicable",
                }
            )

        # Hay actividad/tos, pero ningún experto indicó el tipo.
        return pd.Series(
            {
                "cough_type": "unknown",
                "cough_type_consensus": "ambiguous_expert",
            }
        )

    # -------------------------------------------------------------------------
    # Calcular la moda de los votos
    # -------------------------------------------------------------------------
    vote_counts = Counter(votes)
    maximum_agreement = max(vote_counts.values())

    most_common_labels = [
        label
        for label, count in vote_counts.items()
        if count == maximum_agreement
    ]

    # -------------------------------------------------------------------------
    # Empate: no hay una etiqueta ganadora única
    #
    # Ejemplos:
    #   dry, wet                       -> empate 1-1
    #   dry, dry, wet, wet             -> empate 2-2
    #   dry, wet, unknown              -> empate 1-1-1
    # -------------------------------------------------------------------------
    if len(most_common_labels) > 1:
        return pd.Series(
            {
                "cough_type": "unknown",
                "cough_type_consensus": "ambiguous_expert",
            }
        )

    # Existe una moda única
    final_cough_type = most_common_labels[0]

    # Al menos 3 expertos coinciden
    if maximum_agreement >= 3:
        consensus = "gold_expert"

    # Uno o dos votos para la etiqueta ganadora
    else:
        consensus = "weak_expert"

    return pd.Series(
        {
            "cough_type": final_cough_type,
            "cough_type_consensus": consensus,
        }
    )


def collapse_first_non_null(df, base_variable):
    """
    Mantiene el comportamiento anterior para el resto de variables de expertos:
    toma el primer valor no nulo entre var_1, var_2, var_3 y var_4.

    Esta función no se utiliza para cough_type, ya que cough_type se resuelve
    mediante consenso.
    """
    expert_columns = [
        f"{base_variable}_{i}"
        for i in range(1, 5)
    ]

    existing_columns = [
        column
        for column in expert_columns
        if column in df.columns
    ]

    if len(existing_columns) == 0:
        return df

    df[base_variable] = (
        df[existing_columns]
        .bfill(axis=1)
        .iloc[:, 0]
    )

    df.drop(columns=existing_columns, inplace=True)

    return df


# =============================================================================
# 3. CARGAR METADATOS
# =============================================================================
print("=" * 80)
print(" FILTRADO DE COUGHVID CON CONSENSO DE EXPERTOS")
print("=" * 80)

if not os.path.exists(PATH_METADATA_BASE):
    raise FileNotFoundError(
        f"No se encuentra el archivo de metadatos:\n{PATH_METADATA_BASE}"
    )

df = pd.read_csv(PATH_METADATA_BASE)

print(f"Registros originales: {len(df):,}")


# =============================================================================
# 4. VALIDAR COLUMNAS NECESARIAS
# =============================================================================
required_columns = ["uuid"] + QUALITY_COLS + COUGH_TYPE_COLS

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Faltan columnas necesarias en metadata_compiled.csv: "
        f"{missing_columns}"
    )


# =============================================================================
# 5. FILTRAR FILAS CON ALGUNA ANOTACIÓN EXPERTA
# =============================================================================
expert_filter_columns = QUALITY_COLS + COUGH_TYPE_COLS

has_expert_annotation = (
    df[expert_filter_columns]
    .notna()
    .any(axis=1)
)

df_expert = df.loc[has_expert_annotation].copy()

print(
    "Registros con al menos una anotación experta: "
    f"{len(df_expert):,}"
)


# =============================================================================
# 6. COLAPSAR EL RESTO DE VARIABLES
# =============================================================================
# Se mantiene el comportamiento anterior para estas variables.
# cough_type se excluye porque se calculará mediante consenso.
base_expert_variables = [
    "quality",
    "dyspnea",
    "wheezing",
    "stridor",
    "choking",
    "congestion",
    "nothing",
    "diagnosis",
    "severity",
]

for variable in base_expert_variables:
    df_expert = collapse_first_non_null(
        df=df_expert,
        base_variable=variable,
    )


# =============================================================================
# 7. CALCULAR CONSENSO DE COUGH_TYPE
# =============================================================================
consensus_result = df_expert.apply(
    resolve_cough_type_consensus,
    axis=1,
)

df_expert["cough_type"] = consensus_result["cough_type"]
df_expert["cough_type_consensus"] = consensus_result[
    "cough_type_consensus"
]

# Las columnas individuales ya han sido utilizadas y no se necesitan
# en el CSV filtrado final.
df_expert.drop(
    columns=COUGH_TYPE_COLS,
    inplace=True,
)


# =============================================================================
# 8. SELECCIONAR COLUMNAS FINALES
# =============================================================================
final_columns = [
    "uuid",
    "datetime",
    "cough_detected",
    "SNR",
    "latitude",
    "longitude",
    "age",
    "gender",
    "respiratory_condition",
    "fever_muscle_pain",
    "status",
    "quality",
    "cough_type",
    "cough_type_consensus",
    "dyspnea",
    "wheezing",
    "stridor",
    "choking",
    "congestion",
    "nothing",
    "diagnosis",
    "severity",
]

missing_final_columns = [
    column
    for column in final_columns
    if column not in df_expert.columns
]

if missing_final_columns:
    raise ValueError(
        "No se pueden generar los metadatos porque faltan "
        f"columnas finales: {missing_final_columns}"
    )

df_coughvid_final = df_expert[final_columns].copy()


# =============================================================================
# 9. VALIDACIONES
# =============================================================================
allowed_consensus_values = {
    "gold_expert",
    "weak_expert",
    "ambiguous_expert",
    "not_applicable",
}

unexpected_consensus = set(
    df_coughvid_final["cough_type_consensus"]
    .dropna()
    .unique()
) - allowed_consensus_values

if unexpected_consensus:
    raise ValueError(
        "Se han generado valores de consenso inesperados: "
        f"{unexpected_consensus}"
    )

# Un empate siempre debe producir unknown.
ambiguous_mask = (
    df_coughvid_final["cough_type_consensus"]
    == "ambiguous_expert"
)

invalid_ambiguous = (
    df_coughvid_final.loc[ambiguous_mask, "cough_type"]
    != "unknown"
).any()

if invalid_ambiguous:
    raise ValueError(
        "Existen muestras ambiguous_expert cuya etiqueta no es unknown."
    )

# Los registros not_applicable deben representar audios no-cough.
not_applicable_mask = (
    df_coughvid_final["cough_type_consensus"]
    == "not_applicable"
)

invalid_not_applicable = (
    (
        df_coughvid_final.loc[
            not_applicable_mask,
            "quality",
        ].astype(str).str.strip().str.lower()
        != "no_cough"
    )
    |
    (
        df_coughvid_final.loc[
            not_applicable_mask,
            "cough_type",
        ].astype(str).str.strip().str.lower()
        != "no_cough"
    )
).any()

if invalid_not_applicable:
    raise ValueError(
        "Existen registros not_applicable que no tienen "
        "quality=no_cough y cough_type=no_cough."
    )


# =============================================================================
# 10. MOSTRAR RESUMEN
# =============================================================================
print("\n" + "=" * 80)
print(" DISTRIBUCIÓN DE COUGH_TYPE")
print("=" * 80)

print(
    df_coughvid_final["cough_type"]
    .fillna("not_applicable")
    .value_counts(dropna=False)
)

print("\n" + "=" * 80)
print(" DISTRIBUCIÓN DEL CONSENSO")
print("=" * 80)

print(
    df_coughvid_final["cough_type_consensus"]
    .value_counts(dropna=False)
)

print("\n" + "=" * 80)
print(" TABLA COUGH_TYPE × CONSENSO")
print("=" * 80)

print(
    pd.crosstab(
        df_coughvid_final["cough_type_consensus"],
        df_coughvid_final["cough_type"].fillna("not_applicable"),
        margins=True,
    )
)


# =============================================================================
# 11. GUARDAR CSV
# =============================================================================
os.makedirs(
    os.path.dirname(PATH_OUTPUT_COUGHVID_FILTRADO),
    exist_ok=True,
)

df_coughvid_final.to_csv(
    PATH_OUTPUT_COUGHVID_FILTRADO,
    index=False,
    encoding="utf-8",
)

print("\n" + "=" * 80)
print(" PROCESO COMPLETADO")
print("=" * 80)

print(
    f"Metadatos guardados en:\n{PATH_OUTPUT_COUGHVID_FILTRADO}"
)
print(f"Dimensiones finales: {df_coughvid_final.shape}")