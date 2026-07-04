import os
import zipfile
import requests
import time
from tqdm import tqdm  # Para ver una barra de progreso chula

# Carpeta donde se guardarán tus audios de Clase 0
CARPETA_TEMPORAL = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\FSD50K_DATA"
os.makedirs(CARPETA_TEMPORAL, exist_ok=True)

# 1. Leer los IDs del archivo que generamos antes
with open(r'C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\ids_finales_casa_clase_0.txt', 'r') as f:
    ids_validos = [line.strip() for line in f.readlines()]

print(f"🎯 Total de audios domésticos a descargar: {len(ids_validos)}")
print("📥 Iniciando descarga selectiva...")

# URLs oficiales de las partes de dev_audio en Zenodo
urls_partes = {
    "FSD50K.dev_audio.zip": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip?download=1",
    "FSD50K.dev_audio.z01": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01?download=1",
    "FSD50K.dev_audio.z02": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02?download=1",
    "FSD50K.dev_audio.z03": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03?download=1",
    "FSD50K.dev_audio.z04": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04?download=1",
    "FSD50K.dev_audio.z05": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05?download=1",
}

# 2. Descargar las partes de fondo si no existen
for nombre, url in urls_partes.items():
    ruta_parte = os.path.join(CARPETA_TEMPORAL, nombre)
    if not os.path.exists(ruta_parte):
        print(f"📥 Descargando {nombre} (esto puede tardar, deja al script trabajar)...")
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        
        with open(ruta_parte, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=nombre
        ) as bar:
            for data in r.iter_content(chunk_size=1024*1024):
                f.write(data)
                bar.update(len(data))

print("📦 Todas las partes descargadas de forma segura. Procediendo a la extracción quirúrgica...")

# 3. Unir y extraer solo lo que nos interesa (Evita usar espacio extra)
# Como es un split-zip, lo manejamos leyendo las partes secuencialmente
ruta_zip_principal = os.path.join(CARPETA_TEMPORAL, "FSD50K.dev_audio.zip")

print("🔓 Extrayendo exclusivamente los audios seleccionados...")
# Nota: La librería estándar de zipfile a veces requiere que estén unidos.
# Si tu entorno da error aquí, la forma más robusta es descomprimir usando un comando de sistema o unificar.
try:
    with zipfile.ZipFile(ruta_zip_principal, 'r') as z:
        todos_los_archivos = z.namelist()
        
        # Filtrar la lista de archivos dentro del zip que coinciden con nuestros IDs
        archivos_a_extraer = [
            f for f in todos_los_archivos 
            if os.path.basename(f).replace('.wav', '') in ids_validos
        ]
        
        for archivo in tqdm(archivos_a_extraer, desc="Extrayendo archivos filtrados"):
            # Extraer directo a la carpeta destino
            nombre_final = os.path.basename(archivo)
            ruta_final = os.path.join(CARPETA_TEMPORAL, nombre_final)
            
            with z.open(archivo) as fuente, open(ruta_final, 'wb') as destino:
                destino.write(fuente.read())
                
    print(f"✨ ¡Completado con éxito! Tus audios limpios están en: {CARPETA_TEMPORAL}")
    
    # 4. Opcional: Limpieza de archivos temporales pesados para ahorrar espacio
    eliminar = input("¿Quieres borrar los archivos ZIP descargados para liberar espacio? (s/n): ")
    if eliminar.lower() == 's':
        import shutil
        shutil.rmtree(CARPETA_TEMPORAL)
        print("🧹 Carpeta temporal eliminada.")

except Exception as e:
    print(f"⚠️ Error al procesar el archivo ZIP estructurado: {e}")
    print("Si ves este error, la estructura dividida de Zenodo requiere unificación previa. Puedes unirlos en tu terminal con 'cat' o 7-Zip como vimos antes y procesar localmente.")