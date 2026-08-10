import os
import pandas as pd
import numpy as np

# 1. Configuración de rutas reales
PATH_COUGHVID_FILTRADO = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\coughvid_metadata_filt.csv"
PATH_FSD50K_DEV_CSV = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\FSD-50k_METADATA\FSD50K.ground_truth\dev.csv"
PATH_CSV_COMBINED_METADATA = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\combined_metadata_datasets.csv"

# IDs específicos de FSD50K que ya tienes filtrados en tu disco (Clase 0)
PATH_TXT_IDS_FSD50K = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\ids_fsd50k_class_0.txt"

print("Iniciando la unificación y construcción del dataset metadatos combinados...")

# 2. Cargar los datasets origen
df_coughvid = pd.read_csv(PATH_COUGHVID_FILTRADO)
df_fsd50k_raw = pd.read_csv(PATH_FSD50K_DEV_CSV)

# Cargar los 7,471 IDs válidos de FSD50K que tienes en local
with open(PATH_TXT_IDS_FSD50K, 'r') as f:
    ids_validos_fsd50k = set(line.strip() for line in f.readlines())

# Filtrar el dataframe de FSD50K usando esos IDs
df_fsd50k_filtrado = df_fsd50k_raw[df_fsd50k_raw['fname'].astype(str).isin(ids_validos_fsd50k)].copy()
print(f" Audios cargados de COUGHVID: {len(df_coughvid)}")
print(f" Audios filtrados de FSD50K (Clase 0): {len(df_fsd50k_filtrado)}")

# -------------------------------------------------------------
# PROCESAR COUGHVID
# -------------------------------------------------------------
df_coughvid['dataset_origin'] = 'COUGHVID'

# Definir label (1 para toses, 0 para no_cough)
# Toses reales: quality en 'good', 'ok', 'poor'
df_coughvid['label'] = np.where(df_coughvid['quality'].isin(['good', 'ok', 'poor']), 1, 0)

# Definir type_noise según tu regla
df_coughvid['type_noise'] = np.where(df_coughvid['quality'] == 'no_cough', 'no_cough', 'cough')

# -------------------------------------------------------------
# PROCESAR FSD50K (Para acoplarlo a la estructura común)
# -------------------------------------------------------------
df_fsd50k_comun = pd.DataFrame()

# Mapeo directo de variables
df_fsd50k_comun['uuid'] = df_fsd50k_filtrado['fname'].astype(str)
df_fsd50k_comun['type_noise'] = df_fsd50k_filtrado['labels'] # Inyecta las etiquetas originales (guitarras, música...)
df_fsd50k_comun['quality'] = 'no_cough'
df_fsd50k_comun['label'] = 0  # FSD50K es 100% clase negativa
df_fsd50k_comun['dataset_origin'] = 'FSD50K'

# Añadir el resto de columnas de COUGHVID que FSD50K no posee (rellenándolas con NaN automáticamente)
columnas_coughvid = [
    'datetime', 'cough_detected', 'SNR', 'latitude', 'longitude', 'age', 'gender', 
    'respiratory_condition', 'fever_muscle_pain', 'status', 'cough_type', 'dyspnea', 
    'wheezing', 'stridor', 'choking', 'congestion', 'nothing', 'diagnosis', 'severity'
]

for col in columnas_coughvid:
    df_fsd50k_comun[col] = np.nan

# -------------------------------------------------------------
# CONCATENAR AMBOS DATASETS
# -------------------------------------------------------------
# Asegurarnos de que el orden de las columnas sea idéntico para concatenar limpio
columnas_totales_ordenadas = [
    'uuid', 'datetime', 'cough_detected', 'SNR', 'latitude', 'longitude', 'age', 'gender', 
    'respiratory_condition', 'fever_muscle_pain', 'status', 'quality', 'cough_type', 
    'dyspnea', 'wheezing', 'stridor', 'choking', 'congestion', 'nothing', 'diagnosis', 'severity',
    'label', 'dataset_origin', 'type_noise'
]

df_coughvid = df_coughvid[columnas_totales_ordenadas]
df_fsd50k_comun = df_fsd50k_comun[columnas_totales_ordenadas]

# Combinación final
df_final_combined = pd.concat([df_coughvid, df_fsd50k_comun], ignore_index=True)

# Guardar el CSV final unificado
df_final_combined.to_csv(PATH_CSV_COMBINED_METADATA, index=False)

print("\n🚀 ¡PROCESO COMPLETADO CON ÉXITO!")
print(f"📊 Filas finales de la clase positiva (label=1): {len(df_final_combined[df_final_combined['label']==1])}")
print(f"📊 Filas finales de la clase negativa (label=0): {len(df_final_combined[df_final_combined['label']==0])}")
print(f"💾 Dataset maestro unificado guardado en:\n➡️ {PATH_CSV_COMBINED_METADATA}")