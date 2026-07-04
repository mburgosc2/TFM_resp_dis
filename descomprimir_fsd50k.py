import os
import zipfile
import shutil
from tqdm import tqdm

# Rutas a tus carpetas reales
CARPETA_DATA = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\FSD50K_DATA"
RUTA_ZIP_UNSPLIT = os.path.join(CARPETA_DATA, "unsplit.zip")
CARPETA_CLASE_0 = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\dataset_tfm\clase_0"
PATH_TXT_IDS = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\ids_finales_casa_clase_0.txt"

os.makedirs(CARPETA_CLASE_0, exist_ok=True)

# 1. Leer los IDs que queremos salvar de la Clase 0
with open(PATH_TXT_IDS, 'r') as f:
    ids_salvar = set(line.strip() for line in f.readlines())

print(f"🎯 IDs cargados en memoria: {len(ids_salvar)}")
print("🔓 Abriendo unsplit.zip para la extracción directa (esto puede tardar unos minutos)...")

# 2. Abrir el zip unificado y extraer SOLO lo que está en tu lista
try:
    with zipfile.ZipFile(RUTA_ZIP_UNSPLIT, 'r') as z:
        todos_los_archivos = z.namelist()
        
        # Filtrar los archivos internos (.wav) que coinciden con tus IDs
        archivos_validos = [
            f for f in todos_los_archivos 
            if os.path.basename(f).replace('.wav', '') in ids_salvar
        ]
        
        print(f"📦 Encontrados {len(archivos_validos)} audios válidos dentro del ZIP.")
        
        # Extraer quirúrgicamente uno a uno directo a dataset_tfm/clase_0
        for archivo in tqdm(archivos_validos, desc="Extrayendo archivos filtrados"):
            nombre_final = os.path.basename(archivo)
            ruta_destino_final = os.path.join(CARPETA_CLASE_0, nombre_final)
            
            # Leer el archivo dentro del zip y escribirlo fuera
            with z.open(archivo) as fuente, open(ruta_destino_final, 'wb') as destino:
                destino.write(fuente.read())
                
    print(f"\n✨ ¡CONSEGUIDO! Tus audios limpios están listos en: {CARPETA_CLASE_0}")
    print("Ya puedes borrar manualmente los archivos .zip y .z01... de la carpeta FSD50K_DATA para recuperar espacio.")

except Exception as e:
    print(f"\n❌ Error al procesar el archivo con Python: {e}")