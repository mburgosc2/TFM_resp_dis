"""Extrae MFCC117 para el experimento Stage 1 con cambio de calidad.

El script consume los splits creados por::

    python splits_analysis_multiclass_4c.py --mode audio_quality --test_quality poor

Mantiene un manifiesto alineado con cada matriz ``.npy``. Esto permite medir
despues el rendimiento de TEST por calidad de la tos sin asumir que todas las
filas de los metadatos sobrevivieron a la extraccion de caracteristicas.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
COUGHVID_AUDIO_DIR = ROOT.parent / "DATA"
FSD50K_AUDIO_DIR = (
    ROOT.parent
    / "FSD50K_DATA"
    / "FSD50K.dev_audio"
    / "FSD50K_negative_class_dataset"
)

SAMPLE_RATE = 16_000
N_MFCC = 13
N_FEATURES = 117
SILENCE_THRESHOLD = 1e-4

METADATA_FILES = {
    "train": "metadata_train_multiclass_4c.csv",
    "validation": "metadata_val_multiclass_4c.csv",
    "test": "metadata_test_multiclass_4c.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MFCC117 de Stage 1 para splits random o audio-quality"
    )
    parser.add_argument(
        "--split-mode",
        choices=["audio_quality", "random"],
        default="audio_quality",
        help=(
            "Origen de los splits. 'audio_quality' fuerza una calidad a TEST; "
            "'random' usa las particiones aleatorias actuales."
        ),
    )
    parser.add_argument(
        "--test-quality",
        choices=["poor", "ok"],
        default="poor",
        help="Calidad de tos forzada a TEST por el script de splits.",
    )
    return parser.parse_args()


def audio_path_for(row: pd.Series) -> Path:
    original_uuid = str(row["original_uuid"])
    if row["dataset_origin"] == "FSD50K":
        return FSD50K_AUDIO_DIR / f"{original_uuid}.wav"
    return COUGHVID_AUDIO_DIR / f"{original_uuid}.wav"


def extract_mfcc117(
    audio_path: Path,
    start_time: float,
    end_time: float,
) -> tuple[np.ndarray | None, str | None]:
    """Devuelve mean/std/max de MFCC, delta y delta-delta (39 x 3)."""
    if not audio_path.is_file():
        return None, "audio_not_found"

    duration = float(end_time) - float(start_time)
    if duration <= 0:
        return None, "invalid_segment_duration"

    try:
        signal, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            offset=float(start_time),
            duration=duration,
        )
        if signal.size == 0:
            return None, "empty_audio"
        if float(np.max(np.abs(signal))) < SILENCE_THRESHOLD:
            return None, "near_silence"

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
            return None, "invalid_feature_vector"
        return vector, None
    except Exception as exc:  # El motivo queda auditado; el lote debe continuar.
        return None, f"{type(exc).__name__}: {exc}"


def validate_metadata(df: pd.DataFrame, split: str) -> None:
    required = {
        "original_uuid",
        "uuid_segmento",
        "start_time",
        "end_time",
        "dataset_origin",
        "cough_type_label",
        "quality",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {split}: {sorted(missing)}")

    if split == "train":
        if "fold" not in df.columns:
            raise ValueError("TRAIN no contiene la columna fold")
        folds_per_recording = df.groupby("original_uuid")["fold"].nunique()
        if (folds_per_recording > 1).any():
            raise ValueError(
                "Hay segmentos de una misma grabacion distribuidos entre varios folds"
            )
        if set(df["fold"].astype(int).unique()) != {0, 1, 2, 3, 4}:
            raise ValueError("TRAIN no contiene exactamente los folds 0..4")


def extract_split(
    metadata: pd.DataFrame,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    features: list[np.ndarray] = []
    targets: list[int] = []
    folds: list[int] = []
    manifest_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []

    for source_row, row in tqdm(
        metadata.iterrows(), total=len(metadata), desc=f"MFCC {split}"
    ):
        audio_path = audio_path_for(row)
        vector, error = extract_mfcc117(
            audio_path,
            float(row["start_time"]),
            float(row["end_time"]),
        )
        if error is not None:
            error_rows.append(
                {
                    "split": split,
                    "source_row": int(source_row),
                    "original_uuid": str(row["original_uuid"]),
                    "uuid_segmento": str(row["uuid_segmento"]),
                    "audio_path": str(audio_path),
                    "error": error,
                }
            )
            continue

        feature_row = len(features)
        target = int(int(row["cough_type_label"]) != 0)
        fold = int(row.get("fold", -1))
        features.append(vector)
        targets.append(target)
        folds.append(fold)

        keep_columns = [
            "original_uuid",
            "uuid_segmento",
            "dataset_origin",
            "quality",
            "cough_type",
            "cough_type_name",
            "cough_type_consensus",
            "cough_type_label",
            "start_time",
            "end_time",
            "fold",
            "split",
        ]
        manifest_row: dict[str, object] = {
            "feature_row": feature_row,
            "source_row": int(source_row),
            "stage1_target": target,
        }
        for column in keep_columns:
            if column in row.index:
                manifest_row[column] = row[column]
        manifest_rows.append(manifest_row)

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int8)
    fold_array = np.asarray(folds, dtype=np.int8)
    manifest = pd.DataFrame(manifest_rows)
    errors = pd.DataFrame(error_rows)

    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"Matriz de {split} inesperada: {X.shape}")
    if not (len(X) == len(y) == len(fold_array) == len(manifest)):
        raise ValueError(f"Salida desalineada en {split}")
    if not np.array_equal(manifest["stage1_target"].to_numpy(dtype=np.int8), y):
        raise ValueError(f"El manifiesto de {split} no esta alineado con y")

    return X, y, fold_array, manifest, errors


def main() -> None:
    args = parse_args()
    if args.split_mode == "random":
        metadata_dir = ROOT / "metadata_splits_multi_experiment_random"
        output_dir = ROOT / "features_extracted_stage1_random" / "mfcc117"
    else:
        metadata_dir = (
            ROOT / f"metadata_splits_multi_experiment_audio_quality_{args.test_quality}"
        )
        output_dir = (
            ROOT
            / f"features_extracted_stage1_audio_quality_{args.test_quality}"
            / "mfcc117"
        )

    if not metadata_dir.is_dir():
        raise FileNotFoundError(
            f"No existe {metadata_dir}. Ejecuta primero:\n"
            "python .\\splits_analysis_multiclass_4c.py "
            f"--mode {args.split_mode}"
            + (
                f" --test_quality {args.test_quality}"
                if args.split_mode == "audio_quality"
                else ""
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STAGE 1 - EXTRACCION MFCC117")
    print("=" * 78)
    print(f"Modo de split: {args.split_mode}")
    if args.split_mode == "audio_quality":
        print(f"Calidad forzada a TEST: {args.test_quality}")
    print(f"Metadatos: {metadata_dir}")
    print(f"Salida: {output_dir}")

    error_frames: list[pd.DataFrame] = []
    started = time.perf_counter()
    for split, filename in METADATA_FILES.items():
        path = metadata_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata = pd.read_csv(path)
        validate_metadata(metadata, split)
        X, y, folds, manifest, errors = extract_split(metadata, split)

        npy_split = "val" if split == "validation" else split
        np.save(output_dir / f"X_{npy_split}.npy", X)
        np.save(output_dir / f"y_{npy_split}.npy", y)
        if split == "train":
            np.save(output_dir / "folds_train.npy", folds)
        manifest.to_csv(
            output_dir / f"metadata_features_{split}.csv", index=False
        )
        if not errors.empty:
            error_frames.append(errors)

        counts = np.bincount(y, minlength=2).tolist()
        print(
            f"{split:>10}: X={X.shape}; no_tos/tos={counts}; "
            f"descartadas={len(errors)}"
        )

    all_errors = (
        pd.concat(error_frames, ignore_index=True)
        if error_frames
        else pd.DataFrame(
            columns=[
                "split",
                "source_row",
                "original_uuid",
                "uuid_segmento",
                "audio_path",
                "error",
            ]
        )
    )
    all_errors.to_csv(output_dir / "feature_extraction_errors.csv", index=False)

    configuration = {
        "experiment": f"stage1_mfcc117_{args.split_mode}",
        "split_mode": args.split_mode,
        "test_quality": (
            args.test_quality if args.split_mode == "audio_quality" else "not_applicable"
        ),
        "metadata_dir": str(metadata_dir),
        "sample_rate": SAMPLE_RATE,
        "n_mfcc": N_MFCC,
        "statistics": "mean,std,max over MFCC+delta+delta2",
        "n_features": N_FEATURES,
        "silence_threshold": SILENCE_THRESHOLD,
        "elapsed_seconds": time.perf_counter() - started,
    }
    pd.DataFrame(configuration.items(), columns=["parameter", "value"]).to_csv(
        output_dir / "feature_configuration.csv", index=False
    )

    print("\nExtraccion terminada.")
    print(f"Features y manifiestos: {output_dir}")


if __name__ == "__main__":
    main()
