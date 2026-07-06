import os
import sys
import argparse
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split

def segment_int_df(df_split, window_size=10.0, max_safe_threshold=20.0):
    """
    Aplica la segmentación (Opción B para controles, recorte para toses)
    dentro de un split ya cerrado para evitar fugas de datos.
    """
    segmented_records = []
    
    for idx, row in df_split.iterrows():
        total_time = row['duration']
        uuid_orig = str(row['uuid'])
        label = row['label']
        
        row_dict = row.to_dict()
        row_dict['original_uuid'] = uuid_orig
        
        if label == 1:
            # TOSES: Se cortan a 10s fijos (un único registro)
            row_dict['uuid_segmento'] = f"{uuid_orig}_seg_0"
            row_dict['start_time'] = 0.0
            row_dict['end_time'] = min(total_time, window_size)
            segmented_records.append(row_dict)
        else:
            # CONTROLES (No-Tos): Lógica multi-ventana de 10s
            if total_time <= window_size:
                row_dict['uuid_segmento'] = f"{uuid_orig}_seg_0"
                row_dict['start_time'] = 0.0
                row_dict['end_time'] = total_time
                segmented_records.append(row_dict)
            else:
                limite_tiempo = min(total_time, max_safe_threshold)
                num_segments = int(limite_tiempo // window_size)
                for i in range(num_segments):
                    new_segment = row_dict.copy()
                    new_segment['uuid_segmento'] = f"{uuid_orig}_seg_{i}"
                    new_segment['start_time'] = i * window_size
                    new_segment['end_time'] = (i + 1) * window_size
                    segmented_records.append(new_segment)
                    
    return pd.DataFrame(segmented_records).reset_index(drop=True)


def main():
    # 1. Configurar los argumentos de la terminal
    parser = argparse.ArgumentParser(description="Split del Dataset Maestro para el TFM")
    parser.add_argument('--mode', type=str, choices=['random', 'calidad'], default='random',
                        help="Modo de split: 'random' (puro) o 'calidad' (segregación forzada)")
    parser.add_argument('--test_quality', type=str, choices=['poor', 'ok'], default='poor',
                        help="Si el modo es 'calidad', qué calidad de tos se enviará exclusivamente a Test")
    
    args = parser.parse_args()

    # Rutas del proyecto
    PATH_METADATA_UNIFIED_CSV = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\combined_metadata_datasets.csv"
    PATH_AUDIOS_COUGHVID = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\DATA"
    PATH_AUDIOS_FSD50K = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\FSD50K_DATA\FSD50K.dev_audio\FSD50K_negative_class_dataset"

    print("=" * 80)
    print(f" FASE 1: CARGA Y AUDITORÍA EN MODO: [{args.mode.upper()}]")
    print("=" * 80)

    df_metadata_unified = pd.read_csv(PATH_METADATA_UNIFIED_CSV)
    total_filas = len(df_metadata_unified)
    uuids_unicos = df_metadata_unified['uuid'].nunique()

    print(f"Total de registros cargados: {total_filas}")
    print(f"UUIDs únicos: {uuids_unicos}")
    if total_filas != uuids_unicos:
        print(" ALERTA: Existen UUIDs duplicados en el CSV maestro.")
        sys.exit(1)
    else:
        print(" Confirmado: No hay solapamiento ni duplicados de IDs.")

    # ----------------------------------------------------------------------
    # ⏱️ FASE 2: CÁLCULO MÁX/MÍN DURACIONES DE AUDIO
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" ANALIZANDO DURACIONES REALES DE LOS ARCHIVOS (.wav)")
    print("=" * 80)

    duraciones = []
    errores_lectura = 0

    for idx, row in tqdm(df_metadata_unified.iterrows(), total=len(df_metadata_unified), desc="Leyendo cabeceras"):
        uuid = str(row['uuid'])
        origin = row['dataset_origin']
        
        if origin == 'FSD50K':
            audio_path = os.path.join(PATH_AUDIOS_FSD50K, f"{uuid}.wav")
        else:
            audio_path = os.path.join(PATH_AUDIOS_COUGHVID, f"{uuid}.wav")
            
        if os.path.exists(audio_path):
            try:
                duracion = librosa.get_duration(path=audio_path)
                duraciones.append(duracion)
            except Exception:
                errores_lectura += 1
        else:
            errores_lectura += 1

    df_metadata_unified['duration'] = duraciones
    df_metadata_unified = df_metadata_unified.dropna(subset=['duration']).reset_index(drop=True)
    total_filas = len(df_metadata_unified)


    if duraciones:
        print(f"\n📈 REPORT DE DURACIONES:")
        print(f"   - Audio más corto: {min(duraciones):.3f} segundos")
        print(f"   - Audio más largo: {max(duraciones):.3f} segundos")
        print(f"   - Duración media:  {np.mean(duraciones):.3f} segundos")
        if errores_lectura > 0:
            print(f"   -  Ficheros no encontrados o corruptos: {errores_lectura}")
    else:
        print("❌ Alerta: No se han podido mapear las duraciones físicas.")

    # ----------------------------------------------------------------------
    # ✂️ FASE 3: ESTRATEGIA DE SPLITS SEGÚN EL MODO
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FASE 3: GENERANDO PARTICIONES DE DATOS")
    print("=" * 80)

    if args.mode == 'random':
        print("🎲 Ejecutando Split Aleatorio (70% Train / 15% Val / 15% Test)...")
        # Split estratificado simple directo sobre todo el conjunto
        df_dev, df_test_base = train_test_split(
            df_metadata_unified,
            test_size=0.15,
            random_state=42,
            stratify=df_metadata_unified['label']
        )

        df_train_base, df_val_base = train_test_split(
            df_dev,
            test_size=0.1765,
            random_state=42,
            stratify=df_dev['label']
        )
    
    elif args.mode == 'audio_quality':
        calidad_objetivo = args.test_quality
        print(f" Modo Calidad: Aislando toses con calidad '{calidad_objetivo}' directo a Test...")
        
        # Condición: Es una tos (label=1) Y coincide con la calidad seleccionada
        condicion_forzada = (df_metadata_unified['label'] == 1) & (df_metadata_unified['quality'] == calidad_objetivo)
        
        df_test_forzado = df_metadata_unified[condicion_forzada].copy()
        df_resto_pool = df_metadata_unified[~condicion_forzada].copy()
        
        print(f" Toses '{calidad_objetivo}' fijadas en Test: {len(df_test_forzado)}")
        
        # Completar el test set con controles o toses restantes de forma proporcional (15% global esperado)
        target_test_size = int(0.15 * total_filas)
        resto_a_extraer = target_test_size - len(df_test_forzado)
        
        if resto_a_extraer > 0:
            test_ratio_ajustado = resto_a_extraer / len(df_resto_pool)
            df_dev, df_test_random = train_test_split(
                df_resto_pool,
                test_size=test_ratio_ajustado,
                random_state=42,
                stratify=df_resto_pool['label']
            )
            df_test_base = pd.concat([df_test_forzado, df_test_random], ignore_index=True)
        else:
            df_dev = df_resto_pool.copy()
            df_test_base = df_test_forzado.copy()

        df_train_base, df_val_base = train_test_split(
            df_dev,
            test_size=0.1765,
            random_state=42,
            stratify=df_dev['label']
        )

    print("\n" + "=" * 80)
    print(" EXPANDIENDO SEGMENTOS DE FORMA SEGURA DENTRO DE CADA CONJUNTO")
    print("=" * 80)
    
    df_train = segment_int_df(df_train_base)
    df_val = segment_int_df(df_val_base)
    df_test = segment_int_df(df_test_base)

    # ----------------------------------------------------------------------
    # 🔄 FASE 4: ASIGNACIÓN DE LOS 5 FOLDS AL CONJUNTO DE DESARROLLO
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FASE 4: DISTRIBUYENDO EN 5-FOLDS ESTRATIFICADOS")
    print("=" * 80)

    #df_train = df_dev.reset_index(drop=True)
    df_train['fold'] = -1

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['label'])):
        df_train.loc[val_idx, 'fold'] = fold_idx

    # Guardar resultados
    PATH_OUT_TRAIN = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_train.csv"
    PATH_OUT_VAL = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_val.csv"
    PATH_OUT_TEST = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\metadata_test.csv"

    df_train.to_csv(PATH_OUT_TRAIN, index=False)
    df_val.to_csv(PATH_OUT_VAL, index=False)
    df_test.to_csv(PATH_OUT_TEST, index=False)

    print(" PROCESO COMPLETADO")
    print(f"📊 Resumen TRAIN toses (label=1) vs no_toses (label=0): {len(df_train[df_train['label']==1])} vs {len(df_train[df_train['label']==0])}")
    print(f"📊 Resumen VALIDACIÓN toses (label=1) vs no_toses (label=0): {len(df_val[df_val['label']==1])} vs {len(df_val[df_val['label']==0])}")
    print(f"📊 Resumen TEST toses (label=1) vs no_toses (label=0): {len(df_test[df_test['label']==1])} vs {len(df_test[df_test['label']==0])}")
if __name__ == '__main__':
    main()



# python "c:/Users/Usuario/Desktop/UNI MARINA/master/TFM/TFM_resp_dis/splits_analysis.py" --mode random
# python "c:/Users/Usuario/Desktop/UNI MARINA/master/TFM/TFM_resp_dis/splits_analysis.py" --mode audio_quality --test_quality poor
# python "c:/Users/Usuario/Desktop/UNI MARINA/master/TFM/TFM_resp_dis/splits_analysis.py" --mode audio_quality --test_quality ok