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

def map_cough_type(val):
    # Si es NaN / NULL / None -> No es tos (Clase 0)
    if pd.isna(val):
        return 0
    
    # Limpieza básica de texto
    val_str = str(val).lower().strip()
    
    if val_str == 'dry':
        return 1
    elif val_str == 'wet':
        return 2
    elif val_str == 'unknown':
        return 3
    else:
        # En caso de que exista alguna cadena rara no identificada, la mandamos a 0 o 3
        return 0

# Crear la nueva columna con el código numérico
df['cough_type_label'] = df['cough_type'].apply(map_cough_type)

# Crear también una columna de texto descriptiva para mayor claridad visual
label_map_text = {0: 'no_cough', 1: 'dry', 2: 'wet', 3: 'unknown'}
df['cough_type_name'] = df['cough_type_label'].map(label_map_text)

# =====================================================================
# 📊 FASE 3: VERIFICACIÓN Y GUARDADO
# =====================================================================
print("\n📊 Resumen de la nueva distribución de clases:")
counts = df['cough_type_label'].value_counts().sort_index()

for label, count in counts.items():
    name = label_map_text[label]
    pct = (count / len(df)) * 100
    print(f"   • Clase {label} ({name:<8}): {count:>6,} muestras ({pct:.2f}%)")

# Guardar el nuevo CSV duplicado y enriquecido
df.to_csv(PATH_OUTPUT_CSV, index=False)

print("\n" + "=" * 80)
print(f"✅ Nuevo archivo de metadatos guardado exitosamente en:\n {PATH_OUTPUT_CSV}")
print("=" * 80)