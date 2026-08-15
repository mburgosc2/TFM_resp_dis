import numpy as np
import librosa
import os
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGURACION
# ============================================================
SAMPLE_RATE = 16000
N_MELS = 64
N_FRAMES = 64
N_FFT = 1024
HOP_LENGTH = 256
MAX_DURATION = 10.0

PATH_AUDIOS_COUGHVID = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\DATA"
PATH_AUDIOS_FSD50K = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\FSD50K_DATA\FSD50K.dev_audio\FSD50K_negative_class_dataset"

PATH_IN_TRAIN = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_multi_experiment_random\metadata_train_multiclass_4c.csv"
PATH_IN_VAL = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_multi_experiment_random\metadata_val_multiclass_4c.csv"
PATH_IN_TEST = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_splits_multi_experiment_random\metadata_test_multiclass_4c.csv"

PATH_OUTPUT_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_multiclass_4c_cochleogram_experiment_random"


# ============================================================
# UTILIDADES
# ============================================================
def resize_time_axis(matrix_2d, target_frames=N_FRAMES):
    """
    Reescala la dimensión temporal de una matriz [n_bands, time] a [n_bands, target_frames].
    Se usa interpolación lineal por fila.
    """
    n_bands, current_frames = matrix_2d.shape

    if current_frames == target_frames:
        return matrix_2d

    x_old = np.linspace(0.0, 1.0, current_frames)
    x_new = np.linspace(0.0, 1.0, target_frames)

    resized = np.zeros((n_bands, target_frames), dtype=np.float32)
    for i in range(n_bands):
        resized[i] = np.interp(x_new, x_old, matrix_2d[i])

    return resized


def extract_cochleogram_from_segment(
    audio_path,
    start_time,
    end_time,
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS,
    n_frames=N_FRAMES,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH
):
    """
    Carga un segmento, calcula una representación tiempo-frecuencia tipo cochleagram
    usando un banco mel-log, fuerza tamaño fijo 64x64 y devuelve el vector aplanado.

    Devuelve:
        vector_flattened: np.ndarray shape (4096,)
        o None si falla la extracción.
    """
    try:
        duration_to_load = max(0.0, float(end_time) - float(start_time))

        y, sr = librosa.load(
            audio_path,
            sr=sample_rate,
            offset=float(start_time),
            duration=duration_to_load
        )

        if y is None or len(y) == 0:
            return None

        if np.max(np.abs(y)) < 1e-4:
            return None

        # Opcional: recorte de duración máxima por seguridad
        if len(y) > int(MAX_DURATION * sample_rate):
            y = y[: int(MAX_DURATION * sample_rate)]

        # Banco mel de 64 bandas como aproximación ligera de cochleagram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=50,
            fmax=sample_rate / 2,
            power=2.0
        )

        # Pasar a escala logarítmica
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Asegurar dimensión temporal fija N_FRAMES
        mel_db_fixed = resize_time_axis(mel_db, target_frames=n_frames)

        # Normalización ligera por segmento
        mean_val = np.mean(mel_db_fixed)
        std_val = np.std(mel_db_fixed)
        if std_val < 1e-8:
            std_val = 1e-8

        mel_db_fixed = (mel_db_fixed - mean_val) / std_val

        # Aplanar a vector
        vector_flat = mel_db_fixed.astype(np.float32).reshape(-1)

        return vector_flat

    except Exception:
        return None


def extract_features_batch(df_meta, split_name):
    """
    Extrae cochleagrams ligeros para todas las muestras de un split multiclase.
    Guarda:
        - X: matriz aplanada [n_samples, 4096]
        - y: etiqueta multiclase (0,1,2,3)
        - folds_train: solo para TRAIN
    """
    X_list = []
    y_list = []
    uuids_seg_list = []
    original_uuids_list = []
    folds_list = []

    print(f"\n EXTRAYENDO COCHLEOGRAMAS PARA {split_name} (MULTICLASE 4C)...")

    for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc=split_name):
        uuid_orig = str(row["original_uuid"])
        uuid_seg = str(row["uuid_segmento"])

        label = int(row["cough_type_label"])
        start = float(row["start_time"])
        end = float(row["end_time"])
        fold = row["fold"] if "fold" in df_meta.columns else -1

        if row["dataset_origin"] == "FSD50K":
            audio_path = os.path.join(PATH_AUDIOS_FSD50K, f"{uuid_orig}.wav")
        else:
            audio_path = os.path.join(PATH_AUDIOS_COUGHVID, f"{uuid_orig}.wav")

        vector_flat = extract_cochleogram_from_segment(
            audio_path=audio_path,
            start_time=start,
            end_time=end
        )

        if vector_flat is not None:
            X_list.append(vector_flat)
            y_list.append(label)
            uuids_seg_list.append(uuid_seg)
            original_uuids_list.append(uuid_orig)
            folds_list.append(fold)

    X_array = np.array(X_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.int64)

    print(f"✅ {split_name} completado: X {X_array.shape} | y {y_array.shape}")
    return X_array, y_array, uuids_seg_list, original_uuids_list, folds_list


if __name__ == "__main__":
    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

    print("Cargando ficheros de metadatos segmentados (Multiclase 4C)...")
    df_train = pd.read_csv(PATH_IN_TRAIN)
    df_val = pd.read_csv(PATH_IN_VAL)
    df_test = pd.read_csv(PATH_IN_TEST)

    X_train, y_train, seg_train, orig_train, folds_train = extract_features_batch(df_train, "TRAIN")
    X_val, y_val, seg_val, orig_val, _ = extract_features_batch(df_val, "VALIDATION")
    X_test, y_test, seg_test, orig_test, _ = extract_features_batch(df_test, "TEST")

    np.save(os.path.join(PATH_OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PATH_OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PATH_OUTPUT_DIR, "folds_train.npy"), np.array(folds_train))

    np.save(os.path.join(PATH_OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PATH_OUTPUT_DIR, "y_val.npy"), y_val)

    np.save(os.path.join(PATH_OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PATH_OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"\n🎉 Matrices cochleogram guardadas exitosamente en:\n {PATH_OUTPUT_DIR}")