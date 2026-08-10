import os
import shutil
from tqdm import tqdm

# 1. Configuración de tus rutas reales exactas
CARPETA_ORIGEN = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\FSD50K_DATA\FSD50K.dev_audio\FSD50K.dev_audio"
CARPETA_DESTINO = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\FSD50K_DATA\FSD50K.dev_audio\FSD50K_negative_class_dataset"
PATH_TXT_IDS = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\ids_fsd50k_class_0.txt"

# Crear la carpeta de destino si no existe
os.makedirs(CARPETA_DESTINO, exist_ok=True)

# 2. Leer los IDs que queremos conservar (Clase 0)
with open(PATH_TXT_IDS, 'r') as f:
    ids_salvar = set(line.strip() for line in f.readlines())

print(f" IDs cargados desde el TXT: {len(ids_salvar)}")
print(f" Buscando y moviendo audios seleccionados a la nueva carpeta...")

# 3. Escanear la carpeta origen y mover los que coincidan con la lista
archivos_totales = os.listdir(CARPETA_ORIGEN)
contador_movidos = 0

for archivo in tqdm(archivos_totales, desc="Filtrando Dataset"):
    # Quitar el '.wav' para comparar con el ID del TXT
    id_archivo = archivo.replace('.wav', '')
    
    if id_archivo in ids_salvar:
        ruta_origen = os.path.join(CARPETA_ORIGEN, archivo)
        ruta_destino = os.path.join(CARPETA_DESTINO, archivo)
        
        # shutil.move cambia el archivo de sitio al instante
        shutil.move(ruta_origen, ruta_destino)
        contador_movidos += 1

print(f" Se han movido {contador_movidos} audios correctamente a: {CARPETA_DESTINO}")

# 4. Limpieza opcional del resto de los 40.000 audios sobrantes
print(f"\n En la carpeta origen todavía quedan los miles de audios sobrantes que NO necesitas.")
eliminar = input("¿Quieres borrar definitivamente el resto de audios sobrantes para liberar espacio? (s/n): ")

if eliminar.lower() == 's':
    print("🧹 Borrando audios sobrantes...")
    # Volvemos a listar lo que queda en la carpeta origen (los no deseados) y los borramos
    archivos_restantes = os.listdir(CARPETA_ORIGEN)
    for archivo in tqdm(archivos_restantes, desc="Borrando sobrantes"):
        ruta_borrar = os.path.join(CARPETA_ORIGEN, archivo)
        os.remove(ruta_borrar)
        
    # Intentar borrar la carpeta origen si ya quedó vacía
    try:
        os.rmdir(CARPETA_ORIGEN)
        print(" Carpeta original vacía eliminada con éxito.")
    except Exception:
        pass
    print("Limpieza terminada")
else:
    print("No se ha borrado nada. Los audios sobrantes se mantienen en la carpeta original.")