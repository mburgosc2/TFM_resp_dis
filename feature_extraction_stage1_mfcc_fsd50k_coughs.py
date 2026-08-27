"""Extrae MFCC117 para Stage 1 incluyendo las nuevas toses de FSD50K.

Este script es paralelo a ``feature_extraction_stage1_mfcc_audio_quality.py``.
Consume exclusivamente los splits creados por::

    python splits_analysis_stage1_fsd50k_coughs.py --action write

La representacion se mantiene compatible con los experimentos anteriores:

* 13 MFCC;
* delta y delta-delta (39 series en total);
* media, desviacion estandar y maximo temporal (39 x 3 = 117 features).

No se ajusta ningun escalador en este paso. El escalado debe aprenderse dentro
de cada fold durante el entrenamiento para evitar data leakage.

Los audios muy suaves no se eliminan por defecto. Se conservan y quedan
marcados en los manifiestos para no perder respiraciones u otros negativos
validos de baja amplitud. Los errores reales nunca se descartan en silencio:
en modo estricto impiden guardar matrices incompletas y quedan auditados en un
CSV.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
METADATA_DIR = ROOT / "metadata_splits_stage1_fsd50k_coughs_random"
OUTPUT_DIR = ROOT / "features_extracted_stage1_fsd50k_coughs_random" / "mfcc117"

SAMPLE_RATE = 16_000
N_MFCC = 13
N_FEATURES = 117
SILENCE_AUDIT_THRESHOLD = 1e-4

METADATA_FILES = {
    "train": "metadata_train_stage1.csv",
    "validation": "metadata_validation_stage1.csv",
    "test": "metadata_test_stage1.csv",
}

REQUIRED_COLUMNS = {
    "audio_path",
    "dataset_origin",
    "end_time",
    "fold",
    "is_new_fsd50k_cough",
    "original_uuid",
    "record_source",
    "split",
    "split_group",
    "stage1_target",
    "stage2_eligible",
    "start_time",
    "uuid_segmento",
}

MANIFEST_COLUMNS = (
    "original_uuid",
    "uuid_segmento",
    "audio_path",
    "dataset_origin",
    "record_source",
    "is_new_fsd50k_cough",
    "uploader",
    "split_group",
    "split_stratum",
    "quality",
    "cough_type",
    "cough_type_name",
    "cough_type_consensus",
    "cough_type_label",
    "stage1_target",
    "stage2_eligible",
    "start_time",
    "end_time",
    "segment_duration",
    "segmentation_policy",
    "fold",
    "split",
)

CORE_OUTPUT_FILES = (
    "X_train.npy",
    "y_train.npy",
    "folds_train.npy",
    "X_val.npy",
    "y_val.npy",
    "X_test.npy",
    "y_test.npy",
)


@dataclass(frozen=True)
class FeatureDiagnostics:
    """Informacion observable durante la carga de un segmento."""

    loaded_samples: int
    loaded_duration_seconds: float
    peak_amplitude: float
    near_silence: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrae MFCC117 para Stage 1 con las toses FSD50K revisadas "
            "manualmente"
        )
    )
    parser.add_argument(
        "--action",
        choices=["audit", "extract"],
        default="audit",
        help=(
            "audit valida splits, etiquetas, grupos y rutas sin calcular MFCC; "
            "extract realiza la extraccion completa"
        ),
    )
    parser.add_argument(
        "--near-silence-policy",
        choices=["keep", "reject"],
        default="keep",
        help=(
            "keep conserva y marca los segmentos con pico < 1e-4; reject "
            "reproduce el descarte del extractor historico"
        ),
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help=(
            "Permite guardar matrices omitiendo filas con errores. Por defecto "
            "cualquier error impide guardar matrices incompletas"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar salidas derivadas de una ejecucion anterior",
    )
    return parser.parse_args()


def parse_bool_series(series: pd.Series, column: str) -> pd.Series:
    """Convierte una columna booleana sin aceptar valores ambiguos."""
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        examples = sorted(normalized.loc[invalid].unique().tolist())[:5]
        raise ValueError(f"Valores booleanos invalidos en {column}: {examples}")
    return normalized.map(mapping).astype(bool)


def load_and_validate_metadata() -> dict[str, pd.DataFrame]:
    """Carga y valida los tres splits antes de tocar los audios."""
    if not METADATA_DIR.is_dir():
        raise FileNotFoundError(
            f"No existe {METADATA_DIR}. Ejecuta primero:\n"
            "python .\\splits_analysis_stage1_fsd50k_coughs.py --action write"
        )

    splits: dict[str, pd.DataFrame] = {}
    global_segment_ids: set[str] = set()
    groups_by_split: dict[str, set[str]] = {}

    for split, filename in METADATA_FILES.items():
        path = METADATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)

        metadata = pd.read_csv(path, low_memory=False)
        missing_columns = REQUIRED_COLUMNS - set(metadata.columns)
        if missing_columns:
            raise ValueError(
                f"Faltan columnas en {split}: {sorted(missing_columns)}"
            )
        if metadata.empty:
            raise ValueError(f"El split {split} esta vacio")

        if metadata["uuid_segmento"].isna().any():
            raise ValueError(f"Hay uuid_segmento vacios en {split}")
        segment_ids = metadata["uuid_segmento"].astype(str)
        if segment_ids.duplicated().any():
            duplicated = segment_ids.loc[segment_ids.duplicated()].iloc[0]
            raise ValueError(f"uuid_segmento duplicado en {split}: {duplicated}")
        overlap = global_segment_ids.intersection(segment_ids)
        if overlap:
            raise ValueError(
                f"Hay uuid_segmento compartidos entre splits: {next(iter(overlap))}"
            )
        global_segment_ids.update(segment_ids)

        declared_splits = set(metadata["split"].astype(str).str.lower())
        if declared_splits != {split}:
            raise ValueError(
                f"La columna split de {split} contiene: {sorted(declared_splits)}"
            )

        targets = pd.to_numeric(metadata["stage1_target"], errors="coerce")
        if targets.isna().any() or not targets.isin([0, 1]).all():
            raise ValueError(f"stage1_target no es binario en {split}")
        metadata["stage1_target"] = targets.astype(np.int8)

        starts = pd.to_numeric(metadata["start_time"], errors="coerce")
        ends = pd.to_numeric(metadata["end_time"], errors="coerce")
        invalid_times = (
            starts.isna()
            | ends.isna()
            | ~np.isfinite(starts)
            | ~np.isfinite(ends)
            | (starts < 0)
            | (ends <= starts)
        )
        if invalid_times.any():
            row = int(np.flatnonzero(invalid_times.to_numpy())[0])
            raise ValueError(f"Intervalo temporal invalido en {split}, fila {row}")
        metadata["start_time"] = starts.astype(float)
        metadata["end_time"] = ends.astype(float)

        folds = pd.to_numeric(metadata["fold"], errors="coerce")
        if folds.isna().any():
            raise ValueError(f"Hay folds no numericos en {split}")
        metadata["fold"] = folds.astype(np.int8)
        expected_folds = {0, 1, 2, 3, 4} if split == "train" else {-1}
        actual_folds = set(metadata["fold"].unique().tolist())
        if actual_folds != expected_folds:
            raise ValueError(
                f"Folds inesperados en {split}: {sorted(actual_folds)}; "
                f"esperados: {sorted(expected_folds)}"
            )

        per_group_fold = metadata.groupby("split_group")["fold"].nunique()
        if (per_group_fold > 1).any():
            bad_group = str(per_group_fold.loc[per_group_fold > 1].index[0])
            raise ValueError(
                f"El grupo {bad_group} aparece en varios folds de {split}"
            )

        groups = set(metadata["split_group"].astype(str))
        for previous_split, previous_groups in groups_by_split.items():
            group_overlap = groups.intersection(previous_groups)
            if group_overlap:
                raise ValueError(
                    f"Fuga de grupos entre {previous_split} y {split}: "
                    f"{next(iter(group_overlap))}"
                )
        groups_by_split[split] = groups

        paths = metadata["audio_path"].astype(str).map(Path)
        missing_audio = ~paths.map(Path.is_file)
        if missing_audio.any():
            first_missing = paths.loc[missing_audio].iloc[0]
            raise FileNotFoundError(
                f"Faltan {int(missing_audio.sum())} audios en {split}; "
                f"primero: {first_missing}"
            )

        is_new = parse_bool_series(
            metadata["is_new_fsd50k_cough"], "is_new_fsd50k_cough"
        )
        stage2_eligible = parse_bool_series(
            metadata["stage2_eligible"], "stage2_eligible"
        )
        invalid_new = is_new & (
            metadata["dataset_origin"].astype(str).ne("FSD50K")
            | metadata["stage1_target"].ne(1)
            | stage2_eligible
        )
        if invalid_new.any():
            bad_id = metadata.loc[invalid_new, "original_uuid"].iloc[0]
            raise ValueError(
                "Una nueva tos FSD50K no cumple origin=FSD50K, target=1 y "
                f"stage2_eligible=False: {bad_id}"
            )

        metadata["is_new_fsd50k_cough"] = is_new
        metadata["stage2_eligible"] = stage2_eligible
        splits[split] = metadata

    return splits


def print_audit(splits: dict[str, pd.DataFrame]) -> None:
    print("=" * 78)
    print("AUDITORIA - STAGE 1 MFCC117 + TOS FSD50K")
    print("=" * 78)
    print(f"Metadatos: {METADATA_DIR}")
    print("TEST solo se preprocesara; no se calcularan metricas ni se ajustara nada.")

    total_rows = 0
    total_new = 0
    for split, metadata in splits.items():
        total_rows += len(metadata)
        total_new += int(metadata["is_new_fsd50k_cough"].sum())
        counts = (
            metadata.groupby(["dataset_origin", "stage1_target"])
            .size()
            .rename("segments")
            .reset_index()
        )
        duration = metadata["end_time"] - metadata["start_time"]
        print(f"\n{split.upper()}: {len(metadata)} segmentos")
        print(counts.to_string(index=False))
        print(
            "Nuevas toses FSD50K: "
            f"{int(metadata['is_new_fsd50k_cough'].sum())}; "
            f"duracion min/mediana/max: {duration.min():.3f} / "
            f"{duration.median():.3f} / {duration.max():.3f} s"
        )

    print(f"\nTotal: {total_rows} segmentos; nuevas toses FSD50K: {total_new}")
    print("Rutas, etiquetas, intervalos, folds y aislamiento de grupos: correctos.")


def extract_mfcc117(
    audio_path: Path,
    start_time: float,
    end_time: float,
    near_silence_policy: str,
) -> tuple[np.ndarray | None, FeatureDiagnostics | None, str | None]:
    """Extrae el vector MFCC117 y sus diagnosticos de amplitud."""
    duration = float(end_time) - float(start_time)
    try:
        signal, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True,
            offset=float(start_time),
            duration=duration,
        )
        if signal.size == 0:
            return None, None, "empty_audio"
        if not np.isfinite(signal).all():
            return None, None, "non_finite_audio_samples"

        peak = float(np.max(np.abs(signal)))
        near_silence = peak < SILENCE_AUDIT_THRESHOLD
        diagnostics = FeatureDiagnostics(
            loaded_samples=int(signal.size),
            loaded_duration_seconds=float(signal.size / SAMPLE_RATE),
            peak_amplitude=peak,
            near_silence=near_silence,
        )
        if near_silence and near_silence_policy == "reject":
            return None, diagnostics, "near_silence"

        mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        coefficients = np.vstack([mfcc, delta, delta2])
        vector = np.concatenate(
            [
                np.mean(coefficients, axis=1),
                np.std(coefficients, axis=1),
                np.max(coefficients, axis=1),
            ]
        ).astype(np.float32)
        if vector.shape != (N_FEATURES,) or not np.isfinite(vector).all():
            return None, diagnostics, "invalid_feature_vector"
        return vector, diagnostics, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def extract_split(
    metadata: pd.DataFrame,
    split: str,
    near_silence_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    features: list[np.ndarray] = []
    targets: list[int] = []
    folds: list[int] = []
    manifest_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []

    for source_row, row in tqdm(
        metadata.iterrows(), total=len(metadata), desc=f"MFCC {split}"
    ):
        audio_path = Path(str(row["audio_path"]))
        vector, diagnostics, error = extract_mfcc117(
            audio_path=audio_path,
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            near_silence_policy=near_silence_policy,
        )

        diagnostic_values: dict[str, object] = {}
        if diagnostics is not None:
            diagnostic_values = {
                "loaded_samples": diagnostics.loaded_samples,
                "loaded_duration_seconds": diagnostics.loaded_duration_seconds,
                "peak_amplitude": diagnostics.peak_amplitude,
                "near_silence": diagnostics.near_silence,
            }

        if error is not None:
            error_rows.append(
                {
                    "issue_kind": (
                        "configured_exclusion"
                        if error == "near_silence"
                        else "extraction_error"
                    ),
                    "split": split,
                    "source_row": int(source_row),
                    "original_uuid": str(row["original_uuid"]),
                    "uuid_segmento": str(row["uuid_segmento"]),
                    "audio_path": str(audio_path),
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "stage1_target": int(row["stage1_target"]),
                    "is_new_fsd50k_cough": bool(row["is_new_fsd50k_cough"]),
                    "error": error,
                    **diagnostic_values,
                }
            )
            continue

        if vector is None or diagnostics is None:
            raise RuntimeError("Resultado interno incoherente durante la extraccion")

        feature_row = len(features)
        target = int(row["stage1_target"])
        fold = int(row["fold"])
        features.append(vector)
        targets.append(target)
        folds.append(fold)

        manifest_row: dict[str, object] = {
            "feature_row": feature_row,
            "source_row": int(source_row),
        }
        for column in MANIFEST_COLUMNS:
            if column in row.index:
                manifest_row[column] = row[column]
        manifest_row.update(diagnostic_values)
        manifest_rows.append(manifest_row)

    if features:
        X = np.stack(features).astype(np.float32, copy=False)
    else:
        X = np.empty((0, N_FEATURES), dtype=np.float32)
    y = np.asarray(targets, dtype=np.int8)
    fold_array = np.asarray(folds, dtype=np.int8)
    manifest = pd.DataFrame(manifest_rows)
    errors = pd.DataFrame(error_rows)

    if X.shape != (len(features), N_FEATURES):
        raise ValueError(f"Matriz inesperada en {split}: {X.shape}")
    if not (len(X) == len(y) == len(fold_array) == len(manifest)):
        raise ValueError(f"Salida desalineada en {split}")
    if len(manifest) and not np.array_equal(
        manifest["stage1_target"].to_numpy(dtype=np.int8), y
    ):
        raise ValueError(f"El manifiesto de {split} no esta alineado con y")
    if split == "train" and len(fold_array):
        if set(fold_array.tolist()) != {0, 1, 2, 3, 4}:
            raise ValueError("La extraccion ha dejado TRAIN sin alguno de los folds")

    return X, y, fold_array, manifest, errors


def ensure_output_can_be_written(overwrite: bool) -> None:
    existing = [name for name in CORE_OUTPUT_FILES if (OUTPUT_DIR / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Ya existen salidas en {OUTPUT_DIR}: {existing}. "
            "Usa --overwrite solamente si quieres regenerarlas."
        )


def save_results(
    results: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame],
    ],
    near_silence_policy: str,
    elapsed_seconds: float,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    error_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []

    for split, (X, y, folds, manifest, errors) in results.items():
        npy_split = "val" if split == "validation" else split
        np.save(OUTPUT_DIR / f"X_{npy_split}.npy", X)
        np.save(OUTPUT_DIR / f"y_{npy_split}.npy", y)
        if split == "train":
            np.save(OUTPUT_DIR / "folds_train.npy", folds)
        manifest.to_csv(
            OUTPUT_DIR / f"metadata_features_{split}.csv", index=False
        )
        if errors.empty:
            actual_errors = errors
            exclusions = errors
        else:
            exclusions = errors.loc[
                errors["issue_kind"].eq("configured_exclusion")
            ].copy()
            actual_errors = errors.loc[
                errors["issue_kind"].eq("extraction_error")
            ].copy()
            if not actual_errors.empty:
                error_frames.append(actual_errors)
            if not exclusions.empty:
                exclusion_frames.append(exclusions)

        counts = np.bincount(y, minlength=2)
        summaries.append(
            {
                "split": split,
                "input_segments": len(X) + len(errors),
                "extracted_segments": len(X),
                "excluded_near_silence": len(exclusions),
                "error_count": len(actual_errors),
                "no_cough_segments": int(counts[0]),
                "cough_segments": int(counts[1]),
                "new_fsd50k_cough_segments": int(
                    manifest.get(
                        "is_new_fsd50k_cough",
                        pd.Series(False, index=manifest.index),
                    )
                    .astype(bool)
                    .sum()
                ),
                "near_silence_kept": int(
                    manifest.get(
                        "near_silence", pd.Series(False, index=manifest.index)
                    )
                    .astype(bool)
                    .sum()
                ),
            }
        )

    issue_columns = [
        "issue_kind",
        "split",
        "source_row",
        "original_uuid",
        "uuid_segmento",
        "audio_path",
        "start_time",
        "end_time",
        "stage1_target",
        "is_new_fsd50k_cough",
        "error",
        "loaded_samples",
        "loaded_duration_seconds",
        "peak_amplitude",
        "near_silence",
    ]
    all_errors = (
        pd.concat(error_frames, ignore_index=True)
        if error_frames
        else pd.DataFrame(columns=issue_columns)
    )
    all_exclusions = (
        pd.concat(exclusion_frames, ignore_index=True)
        if exclusion_frames
        else pd.DataFrame(columns=issue_columns)
    )
    all_errors.to_csv(OUTPUT_DIR / "feature_extraction_errors.csv", index=False)
    all_exclusions.to_csv(
        OUTPUT_DIR / "feature_extraction_exclusions.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "feature_extraction_summary.csv", index=False
    )

    configuration = {
        "experiment": "stage1_mfcc117_fsd50k_coughs_random",
        "metadata_dir": str(METADATA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "sample_rate": SAMPLE_RATE,
        "mono": True,
        "n_mfcc": N_MFCC,
        "coefficient_groups": "MFCC,delta,delta2",
        "statistics": "mean,std,max",
        "n_features": N_FEATURES,
        "silence_audit_threshold": SILENCE_AUDIT_THRESHOLD,
        "near_silence_policy": near_silence_policy,
        "normalization": "none; fit scaler inside each training fold",
        "target_column": "stage1_target",
        "path_column": "audio_path",
        "test_usage": "feature extraction only; no model selection or metrics",
        "elapsed_seconds": elapsed_seconds,
    }
    pd.DataFrame(configuration.items(), columns=["parameter", "value"]).to_csv(
        OUTPUT_DIR / "feature_configuration.csv", index=False
    )


def save_errors_only(
    results: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame],
    ]
) -> Path:
    """Guarda el diagnostico sin publicar matrices parciales."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    error_frames = [
        result[4].loc[result[4]["issue_kind"].eq("extraction_error")]
        for result in results.values()
        if not result[4].empty
    ]
    error_frames = [frame for frame in error_frames if not frame.empty]
    errors = pd.concat(error_frames, ignore_index=True)
    path = OUTPUT_DIR / "feature_extraction_errors.csv"
    errors.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    splits = load_and_validate_metadata()
    print_audit(splits)

    if args.action == "audit":
        print("\nAuditoria completada. No se ha creado ni modificado ningun archivo.")
        print("Para extraer: python .\\feature_extraction_stage1_mfcc_fsd50k_coughs.py --action extract")
        return

    ensure_output_can_be_written(args.overwrite)
    print("\n" + "=" * 78)
    print("EXTRACCION - STAGE 1 MFCC117 + TOS FSD50K")
    print("=" * 78)
    print(f"Salida: {OUTPUT_DIR}")
    print(f"Politica de audios muy suaves: {args.near_silence_policy}")
    print("TEST se transforma con parametros fijos; no participa en ningun ajuste.")

    started = time.perf_counter()
    results: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame],
    ] = {}
    for split, metadata in splits.items():
        results[split] = extract_split(
            metadata=metadata,
            split=split,
            near_silence_policy=args.near_silence_policy,
        )

    total_errors = sum(
        int(result[4]["issue_kind"].eq("extraction_error").sum())
        for result in results.values()
        if not result[4].empty
    )
    if total_errors and not args.allow_errors:
        error_path = save_errors_only(results)
        raise RuntimeError(
            f"Se detectaron {total_errors} errores. No se guardaron matrices "
            f"incompletas. Revisa {error_path}. Si son descartes justificados, "
            "repite con --allow-errors."
        )

    elapsed = time.perf_counter() - started
    save_results(
        results=results,
        near_silence_policy=args.near_silence_policy,
        elapsed_seconds=elapsed,
    )

    print("\n" + "=" * 78)
    print("EXTRACCION MFCC117 COMPLETADA")
    print("=" * 78)
    for split, (X, y, _, manifest, errors) in results.items():
        counts = np.bincount(y, minlength=2).tolist()
        near_silence = int(manifest["near_silence"].astype(bool).sum())
        if errors.empty:
            excluded_near_silence = 0
            actual_errors = 0
        else:
            excluded_near_silence = int(
                errors["issue_kind"].eq("configured_exclusion").sum()
            )
            actual_errors = int(errors["issue_kind"].eq("extraction_error").sum())
        print(
            f"{split:>10}: X={X.shape}; no_tos/tos={counts}; "
            f"muy_suaves_conservados={near_silence}; "
            f"muy_suaves_descartados={excluded_near_silence}; "
            f"errores={actual_errors}"
        )
    print(f"Tiempo total: {elapsed:.2f} s")
    print(f"Features y manifiestos: {OUTPUT_DIR}")
    print("No se ha ajustado ningun scaler ni modelo.")


if __name__ == "__main__":
    main()
