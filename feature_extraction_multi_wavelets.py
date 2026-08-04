import os
import numpy as np
import pandas as pd
import librosa
import pywt
from scipy.stats import skew, kurtosis
from tqdm import tqdm

SAMPLE_RATE = 16000 
WAVELET_NAME = 'db4'  # Daubechies 4: Excelente representación para impulsos vocales/acústicos
DECOMP_LEVEL = 5     # 5 niveles -> Genera 1 aproximación (A5) y 5 detalles (D5, D4, D3, D2, D1)

PATH_AUDIOS_COUGHVID = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\DATA"
PATH_AUDIOS_FSD50K = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\FSD50K_DATA\FSD50K.dev_audio\FSD50K_negative_class_dataset"

# ============= 1. FUNCIÓN DE EXTRACCIÓN DE WAVELETS (DWT) =============
def extract_wavelet_features_from_segment(audio_path, start_time, end_time, sample_rate=SAMPLE_RATE, wavelet=WAVELET_NAME, level=DECOMP_LEVEL):
    """
    Carga el segmento exacto de audio y calcula coeficientes DWT (db4, nivel 5).
    Extrae estadísticos de energía, entropía y distribución por cada banda.
    """
    try:
        duration_to_load = end_time - start_time
        y, sr = librosa.load(audio_path, sr=sample_rate, offset=start_time, duration=duration_to_load)
        
        if len(y) == 0 or np.max(np.abs(y)) < 1e-4:
            return None
            
        # Normalización de amplitud para consistencia energética
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        # Descomposición Wavelet Multiresolución
        # coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]
        coeffs = pywt.wavedec(y, wavelet=wavelet, level=level)
        
        # Energía total de la señal para calcular energías relativas
        total_energy = np.sum(y**2) + 1e-8
        
        band_features = []
        
        for c in coeffs:
            if len(c) == 0:
                band_features.extend([0, 0, 0, 0, 0, 0])
                continue
                
            # 1. Energía absoluta y relativa
            energy = np.sum(c**2)
            rel_energy = energy / total_energy
            
            # 2. Entropía logarítmica de la banda
            log_energy_entropy = -np.sum((c**2 + 1e-8) * np.log(c**2 + 1e-8))
            
            # 3. Estadísticos descriptivos de la banda
            mean_val = np.mean(c)
            std_val = np.std(c)
            max_val = np.max(np.abs(c))
            skew_val = skew(c)
            
            band_features.extend([rel_energy, log_energy_entropy, mean_val, std_val, max_val, skew_val])
        
        # Características temporales/globales adicionales (RMS y ZCR)
        rms_val = np.sqrt(np.mean(y**2))
        zcr_val = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Vector final de características Wavelet (6 bandas * 6 métricas + 2 globales = 38 características)
        feature_vector = np.concatenate([band_features, [rms_val, zcr_val]])
        return feature_vector
        
    except Exception:
        return None

# ============= 2. EXTRAER WAVELETS EN BATCH (MULTICLASE) =============
def extract_features_batch(df_meta, split_name):
    """
    Extrae características Wavelet para todas las muestras en un split multiclase.
    """
    X_list = []
    y_list = []
    uuids_seg_list = []
    original_uuids_list = []
    folds_list = []
    
    print(f"\n 🌊 EXTRAYENDO CARACTERÍSTICAS WAVELET ({WAVELET_NAME}) PARA {split_name}...")
    
    for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc=split_name):
        uuid_orig = str(row['original_uuid'])
        uuid_seg = str(row['uuid_segmento'])
        
        label = int(row['cough_type_label'])
        start = float(row['start_time'])
        end = float(row['end_time'])
        
        fold = row['fold'] if 'fold' in df_meta.columns else -1

        if row['dataset_origin'] == 'FSD50K':
            audio_path = os.path.join(PATH_AUDIOS_FSD50K, f"{uuid_orig}.wav")
        else:
            audio_path = os.path.join(PATH_AUDIOS_COUGHVID, f"{uuid_orig}.wav")
            
        vector_wavelet = extract_wavelet_features_from_segment(audio_path, start_time=start, end_time=end)
        
        if vector_wavelet is not None:
            X_list.append(vector_wavelet)
            y_list.append(label)
            uuids_seg_list.append(uuid_seg)
            original_uuids_list.append(uuid_orig)
            folds_list.append(fold)
            
    X_array = np.array(X_list)
    y_array = np.array(y_list)
    
    print(f"✅ {split_name} Completado: Matriz X con dimensión {X_array.shape} | Matriz y con dimensión {y_array.shape}")
    return X_array, y_array, uuids_seg_list, original_uuids_list, folds_list


if __name__ == '__main__':
    # Rutas a los CSVs generados previamente
    PATH_IN_TRAIN = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_multi_experiment_random\metadata_train_multiclass_4c.csv"
    PATH_IN_VAL = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_multi_experiment_random\metadata_val_multiclass_4c.csv"
    PATH_IN_TEST = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_multi_experiment_random\metadata_test_multiclass_4c.csv"

    print("Cargando ficheros de metadatos segmentados (Multiclase 4C)...")
    df_train = pd.read_csv(PATH_IN_TRAIN)
    df_val = pd.read_csv(PATH_IN_VAL)
    df_test = pd.read_csv(PATH_IN_TEST)

    # Extraer Wavelets para cada split
    X_train, y_train, seg_train, orig_train, folds_train = extract_features_batch(df_train, "TRAIN")
    X_val, y_val, seg_val, orig_val, _ = extract_features_batch(df_val, "VALIDATION")
    X_test, y_test, seg_test, orig_test, _ = extract_features_batch(df_test, "TEST")

    # Guardar en nueva carpeta DEDICADA A WAVELETS para no sobrescribir los MFCCs
    PATH_OUTPUT_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_wavelets_4c"
    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(PATH_OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PATH_OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PATH_OUTPUT_DIR, "folds_train.npy"), np.array(folds_train))
    
    np.save(os.path.join(PATH_OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PATH_OUTPUT_DIR, "y_val.npy"), y_val)
    
    np.save(os.path.join(PATH_OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PATH_OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"\n🎉 Matrices Wavelets multiclase guardadas exitosamente en:\n {PATH_OUTPUT_DIR}")