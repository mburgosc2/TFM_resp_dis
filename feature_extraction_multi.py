import numpy as np
import librosa
import os
import pandas as pd
from tqdm import tqdm

SAMPLE_RATE = 16000  # Resample a 16kHz (estándar para audio)
N_MFCC = 13         # Número de coeficientes MFCC
MAX_DURATION = 10    # Duración máxima en segundos

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
        duration_to_load = end_time - start_time
        
        y, sr = librosa.load(audio_path, sr=sample_rate, offset=start_time, duration=duration_to_load)
        
        if len(y) == 0 or np.max(np.abs(y)) < 1e-4:
            return None
            
        # 1. MFCCs estáticos (13)
        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
        # 2. Delta (13)
        delta = librosa.feature.delta(mfcc)
        # 3. Delta-Delta (13)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        features_39 = np.vstack([mfcc, delta, delta2])
        
        # Estadísticos robustos a lo largo del tiempo
        mean_feat = np.mean(features_39, axis=1) # 39 features
        std_feat = np.std(features_39, axis=1)   # 39 features
        max_feat = np.max(features_39, axis=1)   # 39 features
        
        vector_117 = np.concatenate([mean_feat, std_feat, max_feat])
        return vector_117
        
    except Exception:
        return None
    

# ============= 3. EXTRAER MFCC EN BATCH (MULTICLASE) =============
def extract_features_batch(df_meta, split_name):
    """
    Extrae características para todas las muestras en un split multiclase.
    """
    X_list = []
    y_list = []
    uuids_seg_list = []
    original_uuids_list = []
    folds_list = []
    
    print(f"\n EXTRAYENDO MFCC PARA {split_name} (MULTICLASE 4C)...")
    
    for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc=split_name):
        uuid_orig = str(row['original_uuid'])
        uuid_seg = str(row['uuid_segmento'])
        
        # ⚠️ CAMBIO CLAVE MULTICLASE: Leer cough_type_label (0, 1, 2, 3)
        label = int(row['cough_type_label'])
        
        start = float(row['start_time'])
        end = float(row['end_time'])
        
        fold = row['fold'] if 'fold' in df_meta.columns else -1

        if row['dataset_origin'] == 'FSD50K':
            audio_path = os.path.join(PATH_AUDIOS_FSD50K, f"{uuid_orig}.wav")
        else:
            audio_path = os.path.join(PATH_AUDIOS_COUGHVID, f"{uuid_orig}.wav")
            
        vector_117 = extract_features_from_segment(audio_path, start_time=start, end_time=end)
        
        if vector_117 is not None:
            X_list.append(vector_117)
            y_list.append(label)
            uuids_seg_list.append(uuid_seg)
            original_uuids_list.append(uuid_orig)
            folds_list.append(fold)
            
    X_array = np.array(X_list)
    y_array = np.array(y_list)
    
    print(f"✅ {split_name} Completado: Matriz X con dimensión {X_array.shape} | Matriz y con dimensión {y_array.shape}")
    return X_array, y_array, uuids_seg_list, original_uuids_list, folds_list


if __name__ == '__main__':
    # Rutas a los CSVs generados por splits_analysis_multiclass_4c.py
    PATH_IN_TRAIN = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_multi_experiment_random\metadata_train_multiclass_4c.csv"
    PATH_IN_VAL = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_multi_experiment_random\metadata_val_multiclass_4c.csv"
    PATH_IN_TEST = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_multi_experiment_random\metadata_test_multiclass_4c.csv"

    print("Cargando ficheros de metadatos segmentados (Multiclase 4C)...")
    df_train = pd.read_csv(PATH_IN_TRAIN)
    df_val = pd.read_csv(PATH_IN_VAL)
    df_test = pd.read_csv(PATH_IN_TEST)

    # Extraer MFCC para cada split
    X_train, y_train, seg_train, orig_train, folds_train = extract_features_batch(df_train, "TRAIN")
    X_val, y_val, seg_val, orig_val, _ = extract_features_batch(df_val, "VALIDATION")
    X_test, y_test, seg_test, orig_test, _ = extract_features_batch(df_test, "TEST")

    # Guardar en nueva carpeta dedicada
    PATH_OUTPUT_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_multiclass_4c"
    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(PATH_OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PATH_OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PATH_OUTPUT_DIR, "folds_train.npy"), np.array(folds_train))
    
    np.save(os.path.join(PATH_OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PATH_OUTPUT_DIR, "y_val.npy"), y_val)
    
    np.save(os.path.join(PATH_OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PATH_OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"\n🎉 Matrices multiclase guardadas exitosamente en:\n {PATH_OUTPUT_DIR}")