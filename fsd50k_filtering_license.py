import pandas as pd
import json
import os

# Rutas locales donde has descomprimido los zips pequeños
PATH_METADATA = 'C:\\Users\\Usuario\\Desktop\\UNI MARINA\\master\\TFM\\FSD-50k_METADATA\\FSD50K.metadata\\dev_clips_info_FSD50K.json'
PATH_GROUND_TRUTH = 'C:\\Users\\Usuario\\Desktop\\UNI MARINA\\master\\TFM\\FSD-50k_METADATA\\FSD50K.ground_truth\\dev.csv'

# 1. Cargar el JSON de licencias mapeándolo por el ID del clip (Freesound ID)
with open(PATH_METADATA, 'r') as f:
    metadata_json = json.load(f)

# Convertimos el JSON a DataFrame (la clave es el ID del audio)
df_licenses = pd.DataFrame.from_dict(metadata_json, orient='index')
df_licenses.index = df_licenses.index.astype(int)
df_licenses.index.name = 'fname'

# 2. Cargar el CSV de etiquetas (Ground Truth)
df_gt = pd.read_csv(PATH_GROUND_TRUTH)

# 3. Unir ambos DataFrames usando el ID del archivo ('fname')
df_fsd = df_gt.merge(df_licenses['license'], on='fname', how='inner')

# 4. FILTRADO CRUCIAL: Solo licencias permitidas para uso comercial
df_comercial = df_fsd[
    df_fsd['license'].str.contains('publicdomain/zero', na=False) | 
    (df_fsd['license'].str.contains('licenses/by/', na=False) & ~df_fsd['license'].str.contains('-nc', na=False))
].copy()
      
# 5. FILTRADO POR CATEGORÍAS BIOLÓGICAS (Clase 0)
# Elegimos categorías estratégicas de tu lista que NO sean 'Cough'
clases_no_tos = [
    # Voz y respiración
    'Female_speech_and_woman_speaking', 'Male_speech_and_man_speaking', 
    'Child_speech_and_kid_speaking', 'Conversation', 'Breathing', 
    'Sneeze', 'Gasp', 'Laughter', 'Giggle', 'Chuckle_and_chortle', 
    'Screaming', 'Shout', 'Yell', 'Whispering',

    'Hands', 'Fart', 'Clapping', 'Finger_snapping',
    
    # Cocina y baño
    'Water_tap_and_faucet',  'Toilet_flush', 'Chewing_and_mastication',
    
    # Puertas, pasos y cotidianidad
    'Door', 'Slam', 'Knock', 'Cupboard_open_or_close', 'Drawer_open_or_close', 
    'Walk_and_footsteps', 'Run', 'Keys_jangling',
    
    # Mascotas y tecnología
    'Dog', 'Bark', 'Cat', 'Meow', 'Purr', 
    'Telephone', 'Ringtone', 'Alarm'
]

# Buscamos qué audios contienen estas etiquetas en su columna 'labels'
df_clase_0_candidatos = df_comercial[df_comercial['labels'].str.contains('|'.join(clases_no_tos), na=False)]
# Excluimos por si acaso algún audio tuviera doble etiqueta y contuviera tos
df_clase_0_final = df_clase_0_candidatos[~df_clase_0_candidatos['labels'].str.contains('Cough', na=False)]

print(f"✅ Filtro completado.")
print(f"Muestras candidatas comerciales para Clase 0: {len(df_clase_0_final)}")

# Guardamos la lista de IDs que necesitamos
lista_ids_descargar = df_clase_0_final['fname'].tolist()
print(f"Ejemplo de los primeros 10 IDs que tienes que buscar: {lista_ids_descargar[:10]}")



# 1. Creamos una lista vacía para meter todas las etiquetas individuales
todas_las_etiquetas = []

# 2. Recorremos tu dataframe comercial (el que ya no tiene licencias -nc)
# Separamos las etiquetas de cada fila por la coma
for lista_labels in df_comercial['labels'].dropna():
    etiquetas_fila = [label.strip() for label in lista_labels.split(',')]
    todas_las_etiquetas.extend(etiquetas_fila)

# 3. Convertimos a una Serie de Pandas y contamos las frecuencias
frecuencias_clases = pd.Series(todas_las_etiquetas).value_counts()

# 4. Configurar Pandas para que no esconda filas y pinte las 200 categorías completas
pd.set_option('display.max_rows', 250)

print("="*60)
print("📋 LISTA COMPLETA DE CATEGORÍAS DISPONIBLES (SOLO COMMERCIAL-SAFE)")
print("Muestra el nombre de la clase y cuántos audios reales tienes en 'dev'")
print("="*60)
print(frecuencias_clases)



# Filtrar todos los audios comerciales que tengan la etiqueta madre "Respiratory_sounds"
df_respiratorios = df_comercial[df_comercial['labels'].str.contains('Respiratory_sounds', na=False)]

# Extraer y contar todas las etiquetas individuales que aparecen junto a ella
etiquetas_respiratorias = []
for lista_labels in df_respiratorios['labels'].dropna():
    etiquetas_fila = [label.strip() for label in lista_labels.split(',')]
    etiquetas_respiratorias.extend(etiquetas_fila)

# Ver qué subcategorías reales y frecuencias tienen tus datos
print(pd.Series(etiquetas_respiratorias).value_counts())



# ==========================================
# 💾 EXPORTACIÓN DE IDS A ARCHIVO TXT
# ==========================================
# Definimos el nombre del archivo de salida
PATH_SALIDA_TXT = 'ids_finales_casa_clase_0.txt'

# Guardamos la columna 'fname' directamente como texto plano (un ID por línea)
df_clase_0_final['fname'].to_csv(PATH_SALIDA_TXT, index=False, header=False)

print(f"💾 Archivo '{PATH_SALIDA_TXT}' guardado con éxito.")
print(f"Contiene los {len(df_clase_0_final)} IDs listos para el script de descarga.")
print("="*60)