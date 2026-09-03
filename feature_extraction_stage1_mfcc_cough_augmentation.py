"""Genera MFCC117 de dos variantes aleatorias por tos de TRAIN.

Las cuatro transformaciones candidatas son:

* ``noise_high``: mezcla con un negativo real de TRAIN a 10 dB SNR;
* ``noise_moderate``: mezcla con un negativo real de TRAIN a 20 dB SNR;
* ``pitch_up``: desplazamiento de +1.5 semitonos;
* ``pitch_down``: desplazamiento de -1.5 semitonos.

Cada tos original recibe exactamente dos transformaciones distintas elegidas
de forma determinista con ``random_state=42``. Las variantes heredan el fold y
el grupo del audio padre. Los donantes de ruido se toman del mismo fold que la
tos, por lo que nunca introducen informacion del fold OOF durante el
entrenamiento posterior. VALIDATION y TEST no se leen ni se aumentan.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parent
SOURCE_FEATURES_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_cough_segments_random"
    / "mfcc117"
)
SOURCE_MANIFEST_PATH = SOURCE_FEATURES_DIR / "metadata_features_train.csv"
OUTPUT_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_cough_segments_audio_aug_random"
    / "mfcc117"
)
AUDIT_DIR = ROOT / "auditory_stage1_mfcc_cough_augmentation"

SAMPLE_RATE = 16_000
N_MFCC = 13
N_FEATURES = 117
COUGH_CLASS = 1
N_VARIANTS_PER_COUGH = 2
RANDOM_STATE = 42
SILENCE_THRESHOLD = 1e-4
NOISE_HIGH_SNR_DB = 10.0
NOISE_MODERATE_SNR_DB = 20.0
PITCH_SHIFT_SEMITONES = 1.5
MAX_NOISE_DONOR_ATTEMPTS = 30

TRANSFORMATIONS = (
    "noise_high",
    "noise_moderate",
    "pitch_up",
    "pitch_down",
)
EXPERIMENT_ID = "stage1_mfcc117_cough_audio_augmentation"
SELECTION_DESCRIPTION = (
    "two distinct transformations per parent; deterministic random"
)

REQUIRED_COLUMNS = {
    "audio_path",
    "dataset_origin",
    "end_time",
    "fold",
    "original_uuid",
    "split",
    "split_group",
    "stage1_target",
    "start_time",
    "uuid_segmento",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea variantes aleatorias por tos de TRAIN y extrae MFCC117."
        )
    )
    parser.add_argument(
        "--action",
        choices=("check", "extract"),
        default="check",
    )
    parser.add_argument(
        "--audit-examples-per-transform",
        type=int,
        default=3,
        help="Numero de parejas original/aumentada guardadas para escuchar.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar una extraccion anterior.",
    )
    return parser.parse_args()


def parse_bool_series(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        raise ValueError(
            f"Booleanos no validos: {sorted(normalized[invalid].unique())[:5]}"
        )
    return normalized.map(mapping).astype(bool)


def load_source_manifest() -> pd.DataFrame:
    if not SOURCE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"No existe {SOURCE_MANIFEST_PATH}. Ejecuta primero la extraccion "
            "MFCC117 sin augmentation."
        )
    manifest = pd.read_csv(
        SOURCE_MANIFEST_PATH,
        dtype={
            "original_uuid": str,
            "uuid_segmento": str,
            "split_group": str,
        },
    )
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("El manifiesto TRAIN esta vacio.")
    if manifest["uuid_segmento"].duplicated().any():
        raise ValueError("Hay uuid_segmento duplicados en TRAIN.")
    if set(manifest["split"].astype(str)) != {"train"}:
        raise ValueError("El manifiesto fuente debe contener solo TRAIN.")

    manifest = manifest.copy()
    manifest["stage1_target"] = pd.to_numeric(
        manifest["stage1_target"], errors="raise"
    ).astype(int)
    manifest["fold"] = pd.to_numeric(
        manifest["fold"], errors="raise"
    ).astype(int)
    manifest["start_time"] = pd.to_numeric(
        manifest["start_time"], errors="raise"
    ).astype(float)
    manifest["end_time"] = pd.to_numeric(
        manifest["end_time"], errors="raise"
    ).astype(float)

    if set(manifest["stage1_target"]) != {0, 1}:
        raise ValueError("TRAIN no contiene exactamente las clases 0 y 1.")
    if set(manifest["fold"]) != {0, 1, 2, 3, 4}:
        raise ValueError("TRAIN no contiene exactamente los folds 0..4.")
    if (manifest["end_time"] <= manifest["start_time"]).any():
        raise ValueError("Hay intervalos de audio no validos.")

    missing_paths = [
        path
        for path in manifest["audio_path"].astype(str).map(Path)
        if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Falta un audio fuente: {missing_paths[0]}")

    group_folds = manifest.groupby("split_group")["fold"].nunique()
    if (group_folds > 1).any():
        raise ValueError("Un split_group aparece en varios folds.")
    return manifest


def validate_source_feature_configuration() -> None:
    path = SOURCE_FEATURES_DIR / "feature_configuration.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    values = dict(zip(table["parameter"], table["value"]))
    expected = {
        "sample_rate": "16000",
        "n_mfcc": "13",
        "n_features": "117",
        "coefficient_groups": "MFCC,delta,delta2",
        "statistics": "mean,std,max",
        "near_silence_policy": "reject",
    }
    for parameter, expected_value in expected.items():
        actual = str(values.get(parameter, ""))
        if actual.endswith(".0") and expected_value.isdigit():
            actual = actual[:-2]
        if actual != expected_value:
            raise ValueError(
                f"Configuracion incompatible: {parameter}={actual!r}; "
                f"esperado={expected_value!r}."
            )


def stable_rng(identifier: str) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{RANDOM_STATE}|{identifier}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed)


def load_row_audio(row: pd.Series) -> np.ndarray:
    duration = float(row["end_time"]) - float(row["start_time"])
    signal, _ = librosa.load(
        Path(str(row["audio_path"])),
        sr=SAMPLE_RATE,
        mono=True,
        offset=float(row["start_time"]),
        duration=duration,
        dtype=np.float32,
    )
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        raise ValueError(f"Audio vacio: {row['uuid_segmento']}")
    if not np.isfinite(signal).all():
        raise ValueError(f"Audio no finito: {row['uuid_segmento']}")
    if float(np.max(np.abs(signal))) < SILENCE_THRESHOLD:
        raise ValueError(f"Audio casi silencioso: {row['uuid_segmento']}")
    return signal


def calculate_mfcc117(signal: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    coefficients = np.vstack([mfcc, delta, delta2])
    features = np.concatenate(
        [
            np.mean(coefficients, axis=1),
            np.std(coefficients, axis=1),
            np.max(coefficients, axis=1),
        ]
    ).astype(np.float32)
    if features.shape != (N_FEATURES,) or not np.isfinite(features).all():
        raise ValueError(f"Vector MFCC117 no valido: {features.shape}")
    return features


def match_noise_length(
    noise: np.ndarray,
    target_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise.size >= target_length:
        maximum_start = noise.size - target_length
        start = int(rng.integers(0, maximum_start + 1)) if maximum_start else 0
        return noise[start : start + target_length].copy()

    repeats = int(np.ceil(target_length / max(1, noise.size)))
    tiled = np.tile(noise, repeats)
    return tiled[:target_length].copy()


def mix_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> tuple[np.ndarray, float]:
    signal_rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
    noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
    if signal_rms <= 1e-10 or noise_rms <= 1e-10:
        raise ValueError("RMS insuficiente para mezclar a una SNR definida.")

    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noise_scale = target_noise_rms / noise_rms
    augmented = signal.astype(np.float64) + noise_scale * noise.astype(np.float64)
    augmented = prevent_clipping(augmented)
    return augmented, noise_scale


def prevent_clipping(signal: np.ndarray) -> np.ndarray:
    output = np.asarray(signal, dtype=np.float64)
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.999:
        output = output * (0.999 / peak)
    return output.astype(np.float32)


def choose_noise_donor(
    candidates: pd.DataFrame,
    target_length: int,
    rng: np.random.Generator,
) -> tuple[pd.Series, np.ndarray]:
    order = rng.permutation(len(candidates))
    for candidate_index in order[:MAX_NOISE_DONOR_ATTEMPTS]:
        donor = candidates.iloc[int(candidate_index)]
        noise = load_row_audio(donor)
        if float(np.sqrt(np.mean(np.square(noise, dtype=np.float64)))) > 1e-8:
            return donor, match_noise_length(noise, target_length, rng)
    raise ValueError("No se encontro un donante de ruido valido.")


def apply_transformation(
    signal: np.ndarray,
    transformation: str,
    noise_candidates: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if transformation in {"noise_high", "noise_moderate"}:
        snr_db = (
            NOISE_HIGH_SNR_DB
            if transformation == "noise_high"
            else NOISE_MODERATE_SNR_DB
        )
        donor, noise = choose_noise_donor(
            noise_candidates,
            target_length=len(signal),
            rng=rng,
        )
        augmented, noise_scale = mix_at_snr(signal, noise, snr_db)
        return augmented, {
            "snr_db": snr_db,
            "pitch_semitones": np.nan,
            "noise_scale": noise_scale,
            "noise_donor_uuid_segmento": str(donor["uuid_segmento"]),
            "noise_donor_original_uuid": str(donor["original_uuid"]),
            "noise_donor_split_group": str(donor["split_group"]),
            "noise_donor_fold": int(donor["fold"]),
            "noise_donor_dataset_origin": str(donor["dataset_origin"]),
        }

    semitones = (
        PITCH_SHIFT_SEMITONES
        if transformation == "pitch_up"
        else -PITCH_SHIFT_SEMITONES
    )
    augmented = librosa.effects.pitch_shift(
        y=signal,
        sr=SAMPLE_RATE,
        n_steps=semitones,
    )
    augmented = prevent_clipping(augmented)
    return augmented, {
        "snr_db": np.nan,
        "pitch_semitones": semitones,
        "noise_scale": np.nan,
        "noise_donor_uuid_segmento": "",
        "noise_donor_original_uuid": "",
        "noise_donor_split_group": "",
        "noise_donor_fold": np.nan,
        "noise_donor_dataset_origin": "",
    }


def save_audit_pair(
    parent: pd.Series,
    original: np.ndarray,
    augmented: np.ndarray,
    transformation: str,
    variant_id: str,
) -> None:
    target = AUDIT_DIR / transformation
    target.mkdir(parents=True, exist_ok=True)
    safe_parent = str(parent["uuid_segmento"]).replace("/", "_")
    original_path = target / f"{safe_parent}__original.wav"
    augmented_path = target / f"{variant_id}.wav"
    if not original_path.exists():
        sf.write(original_path, original, SAMPLE_RATE, subtype="FLOAT")
    sf.write(augmented_path, augmented, SAMPLE_RATE, subtype="FLOAT")


def expected_output_files() -> tuple[Path, ...]:
    return (
        OUTPUT_DIR / "X_train_augmented.npy",
        OUTPUT_DIR / "y_train_augmented.npy",
        OUTPUT_DIR / "folds_train_augmented.npy",
        OUTPUT_DIR / "metadata_features_train_augmented.csv",
        OUTPUT_DIR / "augmentation_configuration.csv",
        OUTPUT_DIR / "augmentation_summary.csv",
    )


def ensure_writable(overwrite: bool) -> None:
    existing = [path for path in expected_output_files() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Ya existe {existing[0]}. Usa --overwrite para regenerar."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_check(manifest: pd.DataFrame) -> None:
    positives = manifest[manifest["stage1_target"] == 1]
    negatives = manifest[manifest["stage1_target"] == 0]
    expected_augmented = len(positives) * N_VARIANTS_PER_COUGH
    print("=" * 78)
    print("CHECK - STAGE 1 AUGMENTATION DE TOSES EN TRAIN")
    print("=" * 78)
    print(f"TRAIN original: {len(manifest)}")
    print(f"No tos / tos: [{len(negatives)}, {len(positives)}]")
    print(f"Variantes por tos: {N_VARIANTS_PER_COUGH}")
    print(f"Variantes esperadas: {expected_augmented}")
    print(f"TRAIN final esperado: {len(manifest) + expected_augmented}")
    print(f"Transformaciones candidatas: {', '.join(TRANSFORMATIONS)}")
    print(
        f"Ruido alto/moderado: SNR {NOISE_HIGH_SNR_DB:g}/"
        f"{NOISE_MODERATE_SNR_DB:g} dB"
    )
    if any(transformation.startswith("pitch") for transformation in TRANSFORMATIONS):
        print(f"Pitch: +/-{PITCH_SHIFT_SEMITONES:g} semitonos")
    print("VALIDATION y TEST no se leen ni se modifican.")


def extract(args: argparse.Namespace, manifest: pd.DataFrame) -> None:
    if args.audit_examples_per_transform < 0:
        raise ValueError("--audit-examples-per-transform debe ser >= 0.")
    ensure_writable(args.overwrite)

    positives = manifest[manifest["stage1_target"] == 1].reset_index(drop=True)
    negatives_by_fold = {
        fold: manifest[
            (manifest["stage1_target"] == 0) & (manifest["fold"] == fold)
        ].reset_index(drop=True)
        for fold in range(5)
    }
    for fold, candidates in negatives_by_fold.items():
        if candidates.empty:
            raise ValueError(f"No hay negativos donantes en el fold {fold}.")

    features: list[np.ndarray] = []
    targets: list[int] = []
    folds: list[int] = []
    metadata_rows: list[dict[str, Any]] = []
    transformation_counts: Counter[str] = Counter()
    audit_counts: Counter[str] = Counter()
    started = time.perf_counter()

    print(
        f"\nGenerando {N_VARIANTS_PER_COUGH} variante(s) por tos "
        "y extrayendo MFCC117..."
    )
    for parent_index, parent in positives.iterrows():
        parent_id = str(parent["uuid_segmento"])
        rng = stable_rng(parent_id)
        selected = rng.choice(
            np.asarray(TRANSFORMATIONS),
            size=N_VARIANTS_PER_COUGH,
            replace=False,
        ).tolist()
        original = load_row_audio(parent)
        parent_fold = int(parent["fold"])

        for variant_number, transformation in enumerate(selected, start=1):
            variant_id = (
                f"{parent_id}__aug{variant_number:02d}__{transformation}"
            )
            variant_rng = stable_rng(variant_id)
            augmented, diagnostics = apply_transformation(
                signal=original,
                transformation=transformation,
                noise_candidates=negatives_by_fold[parent_fold],
                rng=variant_rng,
            )
            vector = calculate_mfcc117(augmented)

            features.append(vector)
            targets.append(COUGH_CLASS)
            folds.append(parent_fold)
            transformation_counts[transformation] += 1

            metadata_rows.append(
                {
                    "feature_row": len(features) - 1,
                    "augmentation_id": variant_id,
                    "parent_feature_row": int(parent["feature_row"]),
                    "parent_uuid_segmento": parent_id,
                    "parent_original_uuid": str(parent["original_uuid"]),
                    "parent_split_group": str(parent["split_group"]),
                    "parent_fold": parent_fold,
                    "parent_dataset_origin": str(parent["dataset_origin"]),
                    "parent_audio_path": str(parent["audio_path"]),
                    "parent_start_time": float(parent["start_time"]),
                    "parent_end_time": float(parent["end_time"]),
                    "stage1_target": COUGH_CLASS,
                    "split": "train_augmentation_only",
                    "transformation": transformation,
                    "random_state": RANDOM_STATE,
                    "original_samples": int(original.size),
                    "augmented_samples": int(augmented.size),
                    "original_peak": float(np.max(np.abs(original))),
                    "augmented_peak": float(np.max(np.abs(augmented))),
                    **diagnostics,
                }
            )

            if audit_counts[transformation] < args.audit_examples_per_transform:
                save_audit_pair(
                    parent,
                    original,
                    augmented,
                    transformation,
                    variant_id,
                )
                audit_counts[transformation] += 1

        if (parent_index + 1) % 100 == 0 or parent_index + 1 == len(positives):
            print(f"  Toses procesadas: {parent_index + 1}/{len(positives)}")

    X_augmented = np.vstack(features).astype(np.float32)
    y_augmented = np.asarray(targets, dtype=np.int8)
    folds_augmented = np.asarray(folds, dtype=np.int8)
    metadata_augmented = pd.DataFrame(metadata_rows)

    expected = len(positives) * N_VARIANTS_PER_COUGH
    if X_augmented.shape != (expected, N_FEATURES):
        raise ValueError(f"Shape aumentada inesperada: {X_augmented.shape}")
    if not np.all(y_augmented == 1):
        raise ValueError("La augmentation contiene una clase distinta de tos.")
    per_parent = metadata_augmented.groupby("parent_uuid_segmento").size()
    if not (per_parent == N_VARIANTS_PER_COUGH).all():
        raise ValueError(
            "No todos los padres tienen exactamente "
            f"{N_VARIANTS_PER_COUGH} variante(s)."
        )
    if not np.array_equal(
        folds_augmented,
        metadata_augmented["parent_fold"].to_numpy(dtype=np.int8),
    ):
        raise ValueError("Los folds aumentados no coinciden con sus padres.")
    noise_rows = metadata_augmented["transformation"].str.startswith("noise")
    if not (
        metadata_augmented.loc[noise_rows, "noise_donor_fold"].to_numpy(dtype=int)
        == metadata_augmented.loc[noise_rows, "parent_fold"].to_numpy(dtype=int)
    ).all():
        raise ValueError("Un donante de ruido procede de otro fold.")

    np.save(OUTPUT_DIR / "X_train_augmented.npy", X_augmented)
    np.save(OUTPUT_DIR / "y_train_augmented.npy", y_augmented)
    np.save(OUTPUT_DIR / "folds_train_augmented.npy", folds_augmented)
    metadata_augmented.to_csv(
        OUTPUT_DIR / "metadata_features_train_augmented.csv", index=False
    )

    summary_rows = [
        {
            "transformation": transformation,
            "count": transformation_counts[transformation],
            "percentage": 100.0
            * transformation_counts[transformation]
            / len(metadata_augmented),
        }
        for transformation in TRANSFORMATIONS
    ]
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "augmentation_summary.csv", index=False
    )

    configuration = {
        "experiment": EXPERIMENT_ID,
        "source_features": str(SOURCE_FEATURES_DIR),
        "scope": "TRAIN cough class only",
        "validation_augmented": False,
        "test_augmented": False,
        "n_original_train": len(manifest),
        "n_original_cough_train": len(positives),
        "n_original_no_cough_train": int((manifest["stage1_target"] == 0).sum()),
        "n_variants_per_cough": N_VARIANTS_PER_COUGH,
        "n_augmented_cough_train": len(metadata_augmented),
        "candidate_transformations": ",".join(TRANSFORMATIONS),
        "selection": SELECTION_DESCRIPTION,
        "random_state": RANDOM_STATE,
        "noise_source": "real negative TRAIN segment from same fold as parent",
        "noise_high_snr_db": NOISE_HIGH_SNR_DB,
        "noise_moderate_snr_db": NOISE_MODERATE_SNR_DB,
        "pitch_shift_semitones": (
            PITCH_SHIFT_SEMITONES
            if any(
                transformation.startswith("pitch")
                for transformation in TRANSFORMATIONS
            )
            else np.nan
        ),
        "clipping_policy": "scale whole augmented waveform to peak 0.999 if needed",
        "sample_rate": SAMPLE_RATE,
        "n_mfcc": N_MFCC,
        "features": "MFCC,delta,delta2 pooled by mean,std,max",
        "n_features": N_FEATURES,
        "fold_policy": "variant and noise donor inherit/match parent fold",
        "elapsed_seconds": time.perf_counter() - started,
    }
    pd.DataFrame(
        configuration.items(), columns=["parameter", "value"]
    ).to_csv(OUTPUT_DIR / "augmentation_configuration.csv", index=False)

    print("\n" + "=" * 78)
    print("AUGMENTATION STAGE 1 COMPLETADA")
    print("=" * 78)
    print(f"X_train_augmented: {X_augmented.shape} {X_augmented.dtype}")
    print(f"Variantes por padre: {N_VARIANTS_PER_COUGH}")
    print(f"Distribucion: {dict(transformation_counts)}")
    print(f"Features: {OUTPUT_DIR}")
    print(f"Auditoria de audio: {AUDIT_DIR}")
    print("VALIDATION y TEST no se han leido ni modificado.")


def main() -> None:
    args = parse_args()
    validate_source_feature_configuration()
    manifest = load_source_manifest()
    print_check(manifest)
    if args.action == "extract":
        extract(args, manifest)
    else:
        print("\nCheck completado. Para extraer usa --action extract.")


if __name__ == "__main__":
    main()
