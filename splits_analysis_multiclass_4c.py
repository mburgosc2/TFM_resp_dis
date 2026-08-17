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
        # Consideramos tos si cough_type_label > 0
        target_class = row['cough_type_label']
        
        row_dict = row.to_dict()
        row_dict['original_uuid'] = uuid_orig
        
        if target_class in [1, 2, 3]:
            # TOSES (Dry, Wet, Unknown): Se cortan a 10s fijos (un único registro)
            row_dict['uuid_segmento'] = f"{uuid_orig}_seg_0"
            row_dict['start_time'] = 0.0
            row_dict['end_time'] = min(total_time, window_size)
            segmented_records.append(row_dict)
        else:
            # CONTROLES (No-Tos / Clase 0): Lógica multi-ventana de 10s
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
    parser = argparse.ArgumentParser(description="Split Multiclase (4 Clases) del Dataset Maestro")
    parser.add_argument('--mode', type=str, choices=['random', 'audio_quality'], default='random',
                        help="Modo de split: 'random' (puro) o 'audio_quality' (segregación forzada)")
    parser.add_argument('--test_quality', type=str, choices=['poor', 'ok'], default='poor',
                        help="Si el modo es 'audio_quality', qué calidad de tos se enviará exclusivamente a Test")
    
    args = parser.parse_args()

    # Rutas del proyecto
    PATH_METADATA_UNIFIED_CSV = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\combined_metadata_multiclass_4c.csv"
    PATH_AUDIOS_COUGHVID = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\DATA"
    PATH_AUDIOS_FSD50K = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\FSD50K_DATA\FSD50K.dev_audio\FSD50K_negative_class_dataset"

    print("=" * 80)
    print(f" FASE 1: CARGA Y AUDITORÍA MULTICLASE (4 CLASES) EN MODO: [{args.mode.upper()}]")
    print("=" * 80)

    if not os.path.exists(PATH_METADATA_UNIFIED_CSV):
        print(f"❌ ERROR: No se encuentra el CSV {PATH_METADATA_UNIFIED_CSV}. Ejecuta primero create_multiclass_metadata.py")
        sys.exit(1)

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

    required_columns = {
        "uuid",
        "dataset_origin",
        "quality",
        "cough_type",
        "cough_type_label",
        "cough_type_name",
        "cough_type_consensus",
    }

    missing_columns = required_columns - set(df_metadata_unified.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(missing_columns)}"
        )

    # Etapa 1: todas las toses, independientemente del tipo o consenso
    df_metadata_unified["stage1_target"] = (
        df_metadata_unified["cough_type_label"] > 0
    ).astype(int)

    # Etapa 2: únicamente dry/wet con etiqueta gold o weak
    df_metadata_unified["stage2_eligible"] = (
        df_metadata_unified["cough_type"].isin(["dry", "wet"])
        & df_metadata_unified["cough_type_consensus"].isin(
            ["gold_expert", "weak_expert"]
        )
    )

    # Subconjunto gold para evaluación adicional
    df_metadata_unified["stage2_gold_eval"] = (
        df_metadata_unified["cough_type"].isin(["dry", "wet"])
        & (
            df_metadata_unified["cough_type_consensus"]
            == "gold_expert"
        )
    )

    # Unknown/ambiguous para evaluar posteriormente la salida inconclusa
    df_metadata_unified["stage2_reject_challenge"] = (
        (df_metadata_unified["cough_type"] == "unknown")
        | (
            df_metadata_unified["cough_type_consensus"]
            == "ambiguous_expert"
        )
    )

    # Estratificar simultáneamente por clase y calidad de etiqueta
    df_metadata_unified["split_stratum"] = (
        df_metadata_unified["cough_type_label"].astype(str)
        + "__"
        + df_metadata_unified["cough_type_consensus"].astype(str)
    )


    # ----------------------------------------------------------------------
    # ⏱️ FASE 2: CÁLCULO MÁX/MÍN DURACIONES DE AUDIO
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" ANALIZANDO DURACIONES REALES DE LOS ARCHIVOS (.wav)")
    print("=" * 80)

    duraciones = []
    errores_lectura = 0

    for _, row in tqdm(
        df_metadata_unified.iterrows(),
        total=len(df_metadata_unified),
        desc="Leyendo cabeceras",
    ):
        uuid = str(row["uuid"])
        origin = row["dataset_origin"]

        if origin == "FSD50K":
            audio_path = os.path.join(
                PATH_AUDIOS_FSD50K,
                f"{uuid}.wav",
            )
        else:
            audio_path = os.path.join(
                PATH_AUDIOS_COUGHVID,
                f"{uuid}.wav",
            )

        duracion = np.nan

        if os.path.exists(audio_path):
            try:
                duracion = librosa.get_duration(path=audio_path)

                if not np.isfinite(duracion) or duracion <= 0:
                    duracion = np.nan
                    errores_lectura += 1

            except Exception:
                errores_lectura += 1
        else:
            errores_lectura += 1

        # Siempre se añade un valor para mantener la alineación con el DataFrame
        duraciones.append(duracion)

    df_metadata_unified["duration"] = duraciones

    df_metadata_unified = (
        df_metadata_unified
        .dropna(subset=["duration"])
        .reset_index(drop=True)
    )

    total_filas = len(df_metadata_unified)

    valid_durations = df_metadata_unified["duration"]

    if not valid_durations.empty:
        print(f"\n📈 REPORT DE DURACIONES:")
        print(f"   - Audio más corto: {valid_durations.min():.3f} segundos")
        print(f"   - Audio más largo: {valid_durations.max():.3f} segundos")
        print(f"   - Duración media:  {valid_durations.mean():.3f} segundos")
        if errores_lectura > 0:
            print(f"   -  Ficheros no encontrados o corruptos: {errores_lectura}")
    else:
        print("No se han podido mapear las duraciones físicas.")

    # ----------------------------------------------------------------------
    #  FASE 3: ESTRATEGIA DE SPLITS SEGÚN EL MODO (ESTRATIFICADO MULTICLASE)
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FASE 3: GENERANDO PARTICIONES DE DATOS ESTRATIFICADAS (cough_type_label)")
    print("=" * 80)

    target_col = 'cough_type_label'

    if args.mode == 'random':
        print("🎲 Ejecutando Split Aleatorio Estratificado (70% Train / 15% Val / 15% Test)...")
        
        df_dev, df_test_base = train_test_split(
            df_metadata_unified,
            test_size=0.15,
            random_state=42,
            stratify=df_metadata_unified["split_stratum"],
        )

        df_train_base, df_val_base = train_test_split(
            df_dev,
            test_size=0.1765,
            random_state=42,
            stratify=df_dev["split_stratum"],
        )
    
    elif args.mode == 'audio_quality':
        calidad_objetivo = args.test_quality
        print(f" Modo Calidad: Aislando toses con calidad '{calidad_objetivo}' directo a Test...")
        
        condicion_forzada = (df_metadata_unified[target_col] > 0) & (df_metadata_unified['quality'] == calidad_objetivo)
        
        df_test_forzado = df_metadata_unified[condicion_forzada].copy()
        df_resto_pool = df_metadata_unified[~condicion_forzada].copy()
        
        print(f" Toses '{calidad_objetivo}' fijadas en Test: {len(df_test_forzado)}")
        
        target_test_size = int(0.15 * total_filas)
        resto_a_extraer = target_test_size - len(df_test_forzado)
        
        if resto_a_extraer > 0:
            test_ratio_ajustado = resto_a_extraer / len(df_resto_pool)
            df_dev, df_test_random = train_test_split(
                df_resto_pool,
                test_size=test_ratio_ajustado,
                random_state=42,
                stratify=df_resto_pool[target_col]
            )
            df_test_base = pd.concat([df_test_forzado, df_test_random], ignore_index=True)
        else:
            df_dev = df_resto_pool.copy()
            df_test_base = df_test_forzado.copy()

        df_train_base, df_val_base = train_test_split(
            df_dev,
            test_size=0.1765,
            random_state=42,
            stratify=df_dev[target_col]
        )

    # ----------------------------------------------------------------------
    # 🔄 FASE 4: ASIGNACIÓN DE LOS 5 FOLDS AL CONJUNTO DE DESARROLLO
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FASE 4: DISTRIBUYENDO EN 5-FOLDS ESTRATIFICADOS MULTICLASE")
    print("=" * 80)

    # Reiniciar los índices para que los índices devueltos por sklearn
    # correspondan directamente con las filas del DataFrame.
    df_train_base = df_train_base.reset_index(drop=True).copy()
    df_val_base = df_val_base.reset_index(drop=True).copy()
    df_test_base = df_test_base.reset_index(drop=True).copy()

    # Solo train utiliza folds internos.
    # Validation y test se mantienen completamente separados.
    df_train_base["fold"] = -1
    df_val_base["fold"] = -1
    df_test_base["fold"] = -1

    # Identificar cada partición explícitamente.
    df_train_base["split"] = "train"
    df_val_base["split"] = "validation"
    df_test_base["split"] = "test"

    # ----------------------------------------------------------------------
    # Seleccionar la variable utilizada para estratificar los folds
    # ----------------------------------------------------------------------
    # split_stratum combina:
    #   cough_type_label + cough_type_consensus
    #
    # Ejemplos:
    #   0__not_applicable
    #   1__gold_expert
    #   1__weak_expert
    #   2__gold_expert
    #   2__weak_expert
    #   3__ambiguous_expert
    #
    # Para utilizar 5 folds necesitamos al menos 5 grabaciones
    # de cada estrato dentro de train.
    stratum_counts_train = (
        df_train_base["split_stratum"]
        .value_counts()
    )

    print("\nDistribución de estratos en TRAIN antes de crear folds:")
    print(stratum_counts_train.sort_index())

    if stratum_counts_train.min() >= 5:
        fold_target_col = "split_stratum"

        print(
            "\nLos folds se estratificarán por clase y consenso: "
            "split_stratum."
        )
    else:
        # Fallback para experimentos especiales donde algún estrato
        # tenga menos de cinco grabaciones.
        fold_target_col = "cough_type_label"

        print(
            "\nAVISO: algún estrato clase-consenso tiene menos de "
            "5 grabaciones."
        )
        print(
            "Los folds se estratificarán únicamente por "
            "cough_type_label."
        )

    # ----------------------------------------------------------------------
    # Crear los 5 folds sobre las grabaciones originales
    # ----------------------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    for fold_idx, (_, fold_val_indices) in enumerate(
        skf.split(
            X=df_train_base,
            y=df_train_base[fold_target_col],
        )
    ):
        # fold indica en qué iteración esa grabación actuará
        # como validación interna.
        df_train_base.loc[
            fold_val_indices,
            "fold",
        ] = fold_idx

    # ----------------------------------------------------------------------
    # Validar que todas las grabaciones de train tengan fold
    # ----------------------------------------------------------------------
    if (df_train_base["fold"] < 0).any():
        uuids_without_fold = df_train_base.loc[
            df_train_base["fold"] < 0,
            "uuid",
        ].tolist()

        raise ValueError(
            "Existen grabaciones de train sin fold asignado: "
            f"{uuids_without_fold[:20]}"
        )

    expected_folds = {0, 1, 2, 3, 4}
    obtained_folds = set(
        df_train_base["fold"]
        .astype(int)
        .unique()
    )

    if obtained_folds != expected_folds:
        raise ValueError(
            "Los folds obtenidos no son los esperados. "
            f"Esperados: {expected_folds}. "
            f"Obtenidos: {obtained_folds}."
        )

    print("\nDistribución de grabaciones originales por fold:")
    print(
        df_train_base["fold"]
        .value_counts()
        .sort_index()
    )

    print("\nDistribución por clase y fold:")
    print(
        pd.crosstab(
            df_train_base["fold"],
            df_train_base["cough_type_name"],
        )
    )

    print("\nDistribución por estrato y fold:")
    print(
        pd.crosstab(
            df_train_base["fold"],
            df_train_base["split_stratum"],
        )
    )

    print(
        "\n✓ Todos los UUID de train tienen asignado exactamente "
        "un fold."
    )
    print(
        "✓ Validation y test permanecen fuera de los folds internos."
    )


    # ----------------------------------------------------------------------
    # FASE 5: SEGMENTACIÓN DENTRO DE CADA PARTICIÓN
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FASE 5: EXPANDIENDO SEGMENTOS DENTRO DE CADA CONJUNTO")
    print("=" * 80)

    # Al convertir cada fila a diccionario dentro de segment_int_df(),
    # todos los segmentos heredarán automáticamente:
    #
    #   - fold
    #   - split
    #   - cough_type_consensus
    #   - stage1_target
    #   - stage2_eligible
    #   - stage2_gold_eval
    #   - stage2_reject_challenge
    #
    df_train = segment_int_df(df_train_base)
    df_val = segment_int_df(df_val_base)
    df_test = segment_int_df(df_test_base)

    # ----------------------------------------------------------------------
    # VALIDAR QUE UNA GRABACIÓN NO APARECE EN VARIOS FOLDS
    # ----------------------------------------------------------------------
    train_uuids = set(df_train["original_uuid"].astype(str))
    val_uuids = set(df_val["original_uuid"].astype(str))
    test_uuids = set(df_test["original_uuid"].astype(str))

    overlap_train_val = train_uuids & val_uuids
    overlap_train_test = train_uuids & test_uuids
    overlap_val_test = val_uuids & test_uuids

    if overlap_train_val:
        raise ValueError(
            f"Hay {len(overlap_train_val)} UUID compartidos entre train y validation."
        )

    if overlap_train_test:
        raise ValueError(
            f"Hay {len(overlap_train_test)} UUID compartidos entre train y test."
        )

    if overlap_val_test:
        raise ValueError(
            f"Hay {len(overlap_val_test)} UUID compartidos entre validation y test."
        )


    folds_per_recording = (
        df_train
        .groupby("original_uuid")["fold"]
        .nunique()
    )

    recordings_in_multiple_folds = folds_per_recording[
        folds_per_recording > 1
    ]

    if not recordings_in_multiple_folds.empty:
        raise ValueError(
            "Hay grabaciones cuyos segmentos aparecen en varios folds:\n"
            f"{recordings_in_multiple_folds.head(20)}"
        )

    if not (df_val["fold"] == -1).all():
        raise ValueError(
            "Las muestras de validation no deben tener folds internos."
        )

    if not (df_test["fold"] == -1).all():
        raise ValueError(
            "Las muestras de test no deben tener folds internos."
        )

    print(
        "✓ Todos los segmentos de una grabación de train "
        "permanecen en el mismo fold."
    )


    # Guardar resultados con nombres dedicados multiclase
    if args.mode == "random":
        output_dir = os.path.join(
            os.path.dirname(PATH_METADATA_UNIFIED_CSV),
            "metadata_splits_multi_experiment_random",
        )
    else:
        output_dir = os.path.join(
            os.path.dirname(PATH_METADATA_UNIFIED_CSV),
            f"metadata_splits_multi_experiment_audio_quality_{args.test_quality}",
        )

    os.makedirs(output_dir, exist_ok=True)

    PATH_OUT_TRAIN = os.path.join(
        output_dir,
        "metadata_train_multiclass_4c.csv",
    )

    PATH_OUT_VAL = os.path.join(
        output_dir,
        "metadata_val_multiclass_4c.csv",
    )

    PATH_OUT_TEST = os.path.join(
        output_dir,
        "metadata_test_multiclass_4c.csv",
    )

    def print_split_summary(df_split, split_name):
        print("\n" + "-" * 80)
        print(f"RESUMEN {split_name}")
        print("-" * 80)

        print("\nGrabaciones originales:")
        print(df_split["original_uuid"].nunique())

        print("\nSegmentos por clase:")
        print(
            df_split["cough_type_name"]
            .value_counts()
        )

        print("\nSegmentos por consenso:")
        print(
            pd.crosstab(
                df_split["cough_type_consensus"],
                df_split["cough_type_name"],
            )
        )

        print("\nMuestras elegibles para etapa 2:")
        stage2_df = df_split[df_split["stage2_eligible"]]

        print(
            stage2_df["cough_type_name"]
            .value_counts()
        )

        print("\nMuestras gold de etapa 2:")
        gold_df = df_split[df_split["stage2_gold_eval"]]

        print(
            gold_df["cough_type_name"]
            .value_counts()
        )

        print("\nChallenge unknown/ambiguous:")
        print(
            int(df_split["stage2_reject_challenge"].sum())
        )


    print_split_summary(df_train, "TRAIN")
    print_split_summary(df_val, "VALIDATION")
    print_split_summary(df_test, "TEST")

    # ----------------------------------------------------------------------
    # GUARDAR LOS SPLITS
    # ----------------------------------------------------------------------
    df_train.to_csv(
        PATH_OUT_TRAIN,
        index=False,
        encoding="utf-8",
    )

    df_val.to_csv(
        PATH_OUT_VAL,
        index=False,
        encoding="utf-8",
    )

    df_test.to_csv(
        PATH_OUT_TEST,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print(" PROCESO COMPLETADO CORRECTAMENTE")
    print("=" * 80)

    print(f"Train guardado en:\n{PATH_OUT_TRAIN}")
    print(f"\nValidation guardado en:\n{PATH_OUT_VAL}")
    print(f"\nTest guardado en:\n{PATH_OUT_TEST}")

    print("\nDimensiones finales:")
    print(f"  Train:      {df_train.shape}")
    print(f"  Validation: {df_val.shape}")
    print(f"  Test:       {df_test.shape}")

if __name__ == '__main__':
    main()