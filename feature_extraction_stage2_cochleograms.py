import os
import sys
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
PATH_SPLITS_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_multi_experiment_random"
PATH_AUDIOS_COUGHVID = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\DATA"
PATH_OUTPUT_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_multiclass_4c"

# Parámetros del pipeline de audio
# Parámetros del pipeline de audio
SAMPLE_RATE = 16000   # Frecuencia nativa de tus audios
TOP_DB_TRIM = 25      # Umbral de energía RMS para recortar silencio/ruido
N_FILTERS = 64        # Bins del cocleograma (CQT)
GRID_ROWS = 4         # Rejilla vertical
GRID_COLS = 4         # Rejilla horizontal
# ==============================================================================
# 2. FUNCIONES DE PROCESAMIENTO
# ==============================================================================
def trim_and_concat_cough_by_energy(y: np.ndarray, sr: int = 16000, top_db: int = 25, min_samples: int = 16000) -> np.ndarray:
    """
    Recorta silencios por energía RMS. Garantiza al menos 16.000 muestras (1.0s a 16kHz)
    aplicando padding si el fragmento es muy breve para evitar errores en la CQT.
    """
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        y_active = y
    else:
        y_active = np.concatenate([y[start:end] for start, end in intervals])
    
    if len(y_active) < min_samples:
        pad_width = min_samples - len(y_active)
        y_active = np.pad(y_active, (0, pad_width), mode='constant')
        
    return y_active


def compute_cochleogram(y: np.ndarray, sr: int = 16000, n_filters: int = 64) -> np.ndarray:
    """
    Calcula el cocleograma ajustado a 16kHz (fmin en C2 = ~65Hz).
    """
    cqt = np.abs(librosa.cqt(
        y, 
        sr=sr, 
        n_bins=n_filters, 
        bins_per_octave=12, 
        fmin=librosa.note_to_hz('C2')
    ))
    cochleogram_db = librosa.amplitude_to_db(cqt, ref=np.max)
    return cochleogram_db

def extract_block_features(cochleogram: np.ndarray, grid_rows: int = 4, grid_cols: int = 4) -> np.ndarray:
    """
    Divide el cocleograma en una rejilla (grid_rows x grid_cols) y extrae media,
    desviación estándar y valor máximo de cada sub-bloque.
    """
    n_freqs, n_times = cochleogram.shape
    row_sub_size = n_freqs // grid_rows
    col_sub_size = n_times // grid_cols
    
    features = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            r_start = r * row_sub_size
            r_end = (r + 1) * row_sub_size if r < grid_rows - 1 else n_freqs
            
            c_start = c * col_sub_size
            c_end = (c + 1) * col_sub_size if c < grid_cols - 1 else n_times
            
            sub_block = cochleogram[r_start:r_end, c_start:c_end]
            
            if sub_block.size > 0:
                features.extend([np.mean(sub_block), np.std(sub_block), np.max(sub_block)])
            else:
                features.extend([0.0, 0.0, 0.0])
                
    return np.array(features, dtype=np.float32)

def process_single_cough_segment(audio_path: str, start_time: float, end_time: float) -> np.ndarray:
    """
    Carga el segmento del audio, aplica recorte de energía RMS, calcula cocleograma y extrae bloque.
    """
    duration = end_time - start_time
    # Cargar segmento específico definido en el CSV
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, offset=start_time, duration=duration if duration > 0 else None)
    
    if len(y) == 0:
        raise ValueError("Audio vacío o corrupto")
        
    # 1. Recorte y filtrado de energía RMS
    y_trimmed = trim_and_concat_cough_by_energy(y, sr, top_db=TOP_DB_TRIM)
    
    # 2. Cocleograma
    cochleo = compute_cochleogram(y_trimmed, sr, n_filters=N_FILTERS)
    
    # 3. Vector de características por rejilla (4x4x3 = 48 características)
    feat_vector = extract_block_features(cochleo, grid_rows=GRID_ROWS, grid_cols=GRID_COLS)
    
    return feat_vector

# ==============================================================================
# 3. EJECUCIÓN PRINCIPAL
# ==============================================================================
def process_split_set(csv_filename: str, set_name: str):
    csv_path = os.path.join(PATH_SPLITS_DIR, csv_filename)
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: No se encuentra {csv_path}")
        return None, None, None, None

    df = pd.read_csv(csv_path)
    
    # FILTRAR SOLO TOSES (cough_type_label: 1=Dry, 2=Wet, 3=Unknown)
    df_coughs = df[df['cough_type_label'] > 0].reset_index(drop=True)
    print(f"\nProcesando [{set_name.upper()}]: {len(df_coughs)} registros de tos encontrados...")

    X_list, y_list, folds_list = [], [], []
    valid_indices = []

    for idx, row in tqdm(df_coughs.iterrows(), total=len(df_coughs), desc=f"Extrayendo {set_name}"):
        uuid = str(row['uuid'])
        audio_path = os.path.join(PATH_AUDIOS_COUGHVID, f"{uuid}.wav")
        
        start_t = row.get('start_time', 0.0)
        end_t = row.get('end_time', 10.0)
        
        if os.path.exists(audio_path):
            try:
                features = process_single_cough_segment(audio_path, start_t, end_t)
                X_list.append(features)
                y_list.append(row['cough_type_label'])
                folds_list.append(row.get('fold', -1))
                valid_indices.append(idx)
            except Exception as e:
                print(f"\n⚠️ Error al procesar {uuid}.wav: {e}")
        else:
            print(f"\n❌ Archivo no encontrado: {audio_path}")

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    folds_arr = np.array(folds_list, dtype=np.int32)
    
    df_valid_metadata = df_coughs.iloc[valid_indices].reset_index(drop=True)

    return X_arr, y_arr, folds_arr, df_valid_metadata

def main():
    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print(" EXTRACCIÓN DE COCLEOGRAMAS PARA ETAPA 2 (TOSES)")
    print("=" * 80)

    splits = [
        ("metadata_train_multiclass_4c.csv", "train"),
        ("metadata_val_multiclass_4c.csv", "val"),
        ("metadata_test_multiclass_4c.csv", "test")
    ]

    for csv_file, split_name in splits:
        X, y, folds, df_meta = process_split_set(csv_file, split_name)
        
        if X is not None and len(X) > 0:
            # Guardar archivos numpy
            np.save(os.path.join(PATH_OUTPUT_DIR, f"X_{split_name}_cochleo.npy"), X)
            np.save(os.path.join(PATH_OUTPUT_DIR, f"y_{split_name}_cochleo.npy"), y)
            np.save(os.path.join(PATH_OUTPUT_DIR, f"folds_{split_name}_cochleo.npy"), folds)
            
            # Guardar el CSV de metadatos filtrado coincidente fila a fila
            df_meta.to_csv(os.path.join(PATH_OUTPUT_DIR, f"metadata_{split_name}_cochleo.csv"), index=False)
            
            print(f"✅ Guardado {split_name.upper()}: X shape={X.shape}, y shape={y.shape}")

    print("\n PROCESO FINALIZADO CON ÉXITO")
    print(f"Archivos guardados en: {PATH_OUTPUT_DIR}")

if __name__ == "__main__":
    main()