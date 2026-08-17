import os
import pandas as pd
import numpy as np

# =====================================================================
# 📂 FASE 1: RUTA DE ARCHIVOS
# =====================================================================
PATH_INPUT_CSV = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\combined_metadata_datasets.csv"
PATH_OUTPUT_CSV = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\combined_metadata_multiclass_4c.csv"

print("=" * 80)
print(" GENERANDO CSV DE METADATOS MULTICLASE (4 CLASES: 0, 1, 2, 3)")
print("=" * 80)

# Cargar dataset original
df = pd.read_csv(PATH_INPUT_CSV)
print(f"✓ Dataset original cargado: {len(df):,} filas.")

# =====================================================================
# 🏷️ FASE 2: MAPEO DE ETIQUETAS A VALORES NUMÉRICOS (0, 1, 2, 3)
# =====================================================================

COUGH_TYPE_TO_LABEL = {
    "no_cough": 0,
    "dry": 1,
    "wet": 2,
    "unknown": 3,
}

LABEL_TO_NAME = {
    0: "no_cough",
    1: "dry",
    2: "wet",
    3: "unknown",
}

# Normalizar texto
df["cough_type"] = (
    df["cough_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# No permitir que un error termine silenciosamente en clase 0
unexpected_cough_types = (
    set(df["cough_type"].unique())
    - set(COUGH_TYPE_TO_LABEL)
)

if unexpected_cough_types:
    raise ValueError(
        "Se encontraron valores inesperados en cough_type: "
        f"{unexpected_cough_types}"
    )

df["cough_type_label"] = (
    df["cough_type"]
    .map(COUGH_TYPE_TO_LABEL)
    .astype(int)
)

df["cough_type_name"] = (
    df["cough_type_label"]
    .map(LABEL_TO_NAME)
)

# =====================================================================
# 📊 FASE 3: VERIFICACIÓN Y GUARDADO
# =====================================================================
print("\n📊 Resumen de la nueva distribución de clases:")
counts = df['cough_type_label'].value_counts().sort_index()

# Clase 0 debe coincidir exactamente con label binario 0
invalid_binary_mapping = (
    ((df["cough_type_label"] == 0) & (df["label"] != 0))
    |
    ((df["cough_type_label"] > 0) & (df["label"] != 1))
)

if invalid_binary_mapping.any():
    invalid_rows = df.loc[
        invalid_binary_mapping,
        [
            "uuid",
            "cough_type",
            "cough_type_label",
            "label",
            "dataset_origin",
        ],
    ]

    raise ValueError(
        "Hay incoherencias entre label binario y cough_type_label:\n"
        f"{invalid_rows.head(20)}"
    )

if df["uuid"].duplicated().any():
    raise ValueError("Existen UUID duplicados en el dataset combinado.")

if df["cough_type_label"].isna().any():
    raise ValueError("Existen etiquetas numéricas sin asignar.")

# Guardar el nuevo CSV duplicado y enriquecido
df.to_csv(PATH_OUTPUT_CSV, index=False)

print("\nDistribución por tipo de consenso:")
print(
    pd.crosstab(
        df["cough_type_consensus"],
        df["cough_type_name"],
        margins=True,
    )
)

print("\n" + "=" * 80)
print(f"✅ Nuevo archivo de metadatos guardado exitosamente en:\n {PATH_OUTPUT_CSV}")
print("=" * 80)

