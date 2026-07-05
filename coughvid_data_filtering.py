import os
import pandas as pd

# 1. Configuración de rutas (Ajusta si tus carpetas reales cambian)
PATH_METADATA_BASE = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\DATA\metadata_compiled.csv"
PATH_OUTPUT_COUGHVID_FILTRADO = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\coughvid_metadata_filt.csv"

print(" Cargando y filtrando dataset original de COUGHVID...")

# Asegurar que existe la carpeta de salida
os.makedirs(os.path.dirname(PATH_OUTPUT_COUGHVID_FILTRADO), exist_ok=True)

# Cargar el dataset base
df = pd.read_csv(PATH_METADATA_BASE)

# Definir columnas de expertos
quality_cols = [f'quality_{i}' for i in range(1, 5)]
cough_type_cols = [f'cough_type_{i}' for i in range(1, 5)]
expert_cols = quality_cols + cough_type_cols

# Quedarnos solo con las filas que tienen al menos una etiqueta de experto
df_expert = df[df[expert_cols].notna().any(axis=1)].copy()
print(f"✅ Filas con validación experta encontradas: {len(df_expert)}")

# Variables base de expertos que se van a unificar
base_expert_vars = [
    'quality', 'cough_type', 'dyspnea', 'wheezing', 'stridor', 
    'choking', 'congestion', 'nothing', 'diagnosis', 'severity'
]

# Iterar sobre cada variable base para colapsar sus versiones _1, _2, _3, _4
for var in base_expert_vars:
    expert_cols_for_var = [f"{var}_1", f"{var}_2", f"{var}_3", f"{var}_4"]
    existing_cols = [c for c in expert_cols_for_var if c in df_expert.columns]
    
    if len(existing_cols) > 0:
        # Colapsar tomando el primer valor NO NULO (backfill en eje horizontal)
        df_expert[var] = df_expert[existing_cols].bfill(axis=1).iloc[:, 0]
        # Eliminar las columnas originales duplicadas (_1.._4)
        df_expert.drop(columns=existing_cols, inplace=True)

# Columnas definitivas que requieres mantener de COUGHVID
columnas_finales_coughvid = [
    'uuid', 'datetime', 'cough_detected', 'SNR', 'latitude', 'longitude', 
    'age', 'gender', 'respiratory_condition', 'fever_muscle_pain', 'status', 
    'quality', 'cough_type', 'dyspnea', 'wheezing', 'stridor', 
    'choking', 'congestion', 'nothing', 'diagnosis', 'severity'
]

# Filtrar el dataframe para quedarnos única y exclusivamente con estas columnas
df_coughvid_final = df_expert[columnas_finales_coughvid].copy()

# Guardar el CSV resultante
df_coughvid_final.to_csv(PATH_OUTPUT_COUGHVID_FILTRADO, index=False)
print(f" Archivo intermedio guardado en:\n➡️ {PATH_OUTPUT_COUGHVID_FILTRADO} ({df_coughvid_final.shape[0]} filas)")