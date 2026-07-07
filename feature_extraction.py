import numpy as np
import librosa
import os
import pandas as pd
from tqdm import tqdm

SAMPLE_RATE = 16000  # Resample a 16kHz (estándar para audio)
N_MFCC = 13         # Número de coeficientes MFCC
MAX_DURATION = 10    # Duración máxima en segundos (para audio muy largo)

PATH_AUDIOS_COUGHVID = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\DATA"
PATH_AUDIOS_FSD50K = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\FSD50K_DATA\FSD50K.dev_audio\FSD50K_negative_class_dataset"

# ============= 2. FUNCIÓN PARA EXTRAER MFCC + DELTA + DELTA-DELTA =============
def extract_features_from_segment(audio_path, start_time, end_time, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Carga el segmento exacto de un audio, extrae MFCC, Delta, Delta-Delta 
    y colapsa el tiempo calculando Mean, Std y Max.
    Returns: Vector plano de 117 características.
    """
    try:
        # Calcular la duración exacta que le corresponde a este trozo
        duration_to_load = end_time - start_time
        
        # Cargar estrictamente el pedazo de audio definido por los metadatos
        y, sr = librosa.load(audio_path, sr=sample_rate, offset=start_time, duration=duration_to_load)
        
        # Control de seguridad: Si el audio cargado está vacío o es puro silencio absoluto
        if len(y) == 0 or np.max(np.abs(y)) < 1e-4:
            return None
            
        # 1. MFCCs estáticos (shape: 13, time_steps)
        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
        
        # 2. Delta (Velocidad) (shape: 13, time_steps)
        delta = librosa.feature.delta(mfcc)
        
        # 3. Delta-Delta (Aceleración) (shape: 13, time_steps)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Concatenamos en el eje de las características para tener un bloque de 39 x time_steps
        features_39 = np.vstack([mfcc, delta, delta2])
        
        # --- COLAPSO TEMPORAL MEDIANTE ESTADÍSTICOS ROBUSTOS ---
        # Calculamos estadísticas a lo largo del eje del tiempo (axis=1)
        mean_feat = np.mean(features_39, axis=1) # 39 features
        std_feat = np.std(features_39, axis=1)   # 39 features
        max_feat = np.max(features_39, axis=1)   # 39 features
        
        # Vector plano final de: 39 + 39 + 39 = 117 descriptores
        vector_117 = np.concatenate([mean_feat, std_feat, max_feat])
        
        return vector_117
        
    except Exception as e:
        # Silenciamos prints masivos, pasamos silenciosamente
        return None
    

# ============= 3. EXTRAER MFCC PARA CADA SPLIT =============
def extract_features_batch(df_meta, split_name):
    """
    Extrae características para todas las muestras en un split.
    
    Returns:
        X: array de features (n_samples, 117) - VARIABLE
        y: array de labels
        uuids: lista de UUIDs
    """
    X_list = []
    y_list = []
    uuids_seg_list = []
    original_uuids_list = []
    folds_list = []
    
    print(f"\n EXTRAYENDO MFCC PARA {split_name}...")
    
    for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc=split_name):
        uuid_orig = str(row['original_uuid'])
        uuid_seg = str(row['uuid_segmento'])
        label = row['label']
        start = float(row['start_time'])
        end = float(row['end_time'])
        
        # Recuperar el fold si existe (solo estará en el conjunto de Train)
        fold = row['fold'] if 'fold' in df_meta.columns else -1

        # Enrutamiento dinámico según el origen del dataset
        if row['dataset_origin'] == 'FSD50K':
            audio_path = os.path.join(PATH_AUDIOS_FSD50K, f"{uuid_orig}.wav")
        else:
            audio_path = os.path.join(PATH_AUDIOS_COUGHVID, f"{uuid_orig}.wav")
            
        # Extraer el vector de 117 descriptores
        vector_117 = extract_features_from_segment(audio_path, start_time=start, end_time=end)
        
        if vector_117 is not None:
            X_list.append(vector_117)
            y_list.append(label)
            uuids_seg_list.append(uuid_seg)
            original_uuids_list.append(uuid_orig)
            folds_list.append(fold)
            
    X_array = np.array(X_list)
    y_array = np.array(y_list)
    
    print(f"✅ {split_name} Completado: Matriz X con dimensión {X_array.shape}")
    return X_array, y_array, uuids_seg_list, original_uuids_list, folds_list


if __name__ == '__main__':
    PATH_IN_TRAIN = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_experiment_random\metadata_train.csv"
    PATH_IN_VAL = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_experiment_random\metadata_val.csv"
    PATH_IN_TEST = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_experiment_random\metadata_test.csv"

    print("Cargando ficheros de metadatos segmentados...")
    df_train = pd.read_csv(PATH_IN_TRAIN)
    df_val = pd.read_csv(PATH_IN_VAL)
    df_test = pd.read_csv(PATH_IN_TEST)

    # Extraer MFCC para cada split
    X_train, y_train, seg_train, orig_train, folds_train = extract_features_batch(df_train, "TRAIN")
    X_val, y_val, seg_val, orig_val, _ = extract_features_batch(df_val, "VALIDATION")
    X_test, y_test, seg_test, orig_test, _ = extract_features_batch(df_test, "TEST")

    PATH_OUTPUT_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_experiment_random"
    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(PATH_OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PATH_OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PATH_OUTPUT_DIR, "folds_train.npy"), np.array(folds_train))
    
    np.save(os.path.join(PATH_OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PATH_OUTPUT_DIR, "y_val.npy"), y_val)
    
    np.save(os.path.join(PATH_OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PATH_OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"\nMatrices guardadas en {PATH_OUTPUT_DIR}")