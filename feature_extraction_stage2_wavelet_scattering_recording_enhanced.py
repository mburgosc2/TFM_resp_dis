"""Extrae WST reutilizable por evento para Stage 2 dry/wet.

Este experimento paralelo parte del extractor que alimento el mejor modelo
WST recording-level + Logistic Regression. Consume exclusivamente los eventos
finales no solapados de COUGHVID y conserva los paths WST de orden 0, 1 y 2.

Cada evento ya representa una ventana objetivo de 1,5 s, por lo que este
script no vuelve a segmentar ni crea ventanas adicionales. Antes de la
normalizacion por pico guarda peak, RMS y crest factor de la porcion de audio
real. Los eventos con padding reciben un peso igual a:

    window_observed_duration / target_window_duration

La salida queda antes del pooling recording-level. Esto permite comparar
posteriormente WST raw/log, con o sin S0, y distintas estadisticas de amplitud
sin recalcular scattering. El pooling previsto es weighted mean/std y max sin
ponderar sobre todos los eventos de un ``original_uuid``.

Solo procesa TRAIN y VALIDATION. TEST no se lee ni se procesa. No usa las
variantes de gold augmentation ni ninguna muestra FSD50K.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from kymatio.numpy import Scattering1D
from tqdm import tqdm

import feature_extraction_stage2_cochleograms as audio_base
import feature_extraction_stage2_wavelet_scattering as wst_base


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = (
    SCRIPT_DIR
    / "features_extracted_stage2_dry_wet_wavelet_scattering_recording_enhanced"
)
PRESET_NAME = "paper_q8_q1_t500_window_raw_o012_weighted"
OUTPUT_DIR = OUTPUT_ROOT / PRESET_NAME
SPLITS = ("train", "validation")
AMPLITUDE_EPSILON = 1e-8
NEAR_SILENCE_AUDIT_THRESHOLD = 1e-4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extrae WST raw S0+S1+S2 por evento COUGHVID para Stage 2."
        )
    )
    parser.add_argument(
        "--action",
        choices=["check", "extract"],
        default="check",
        help="check valida sin escribir; extract procesa TRAIN/VALIDATION",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar esta extraccion derivada",
    )
    return parser


def load_and_validate_manifests() -> dict[str, pd.DataFrame]:
    manifests = {
        split: audio_base.load_event_manifest(split) for split in SPLITS
    }
    audio_base.validate_train_validation_disjoint(
        manifests["train"], manifests["validation"]
    )
    for split, manifest in manifests.items():
        if "dataset_origin" not in manifest.columns:
            raise ValueError(f"{split} no contiene dataset_origin")
        origins = set(manifest["dataset_origin"].astype(str).str.upper())
        if origins != {"COUGHVID"}:
            raise ValueError(
                f"Stage 2 debe usar solo COUGHVID; {split} contiene {origins}"
            )
        if "stage1_target" in manifest.columns:
            targets = pd.to_numeric(
                manifest["stage1_target"], errors="raise"
            ).astype(int)
            if not targets.eq(1).all():
                raise ValueError(f"{split} contiene muestras no-cough")
        if "stage2_eligible" in manifest.columns:
            eligible = audio_base.parse_boolean_column(
                manifest["stage2_eligible"], "stage2_eligible"
            )
            if not eligible.all():
                raise ValueError(f"{split} contiene muestras no elegibles")
        if manifest["source_audio_path"].astype(str).str.contains(
            "FSD50K", case=False, regex=False
        ).any():
            raise ValueError(f"{split} contiene una ruta FSD50K")
    return manifests


def python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def wst_paths_dataframe(scattering: Scattering1D) -> pd.DataFrame:
    meta = scattering.meta()
    rows: list[dict[str, object]] = []
    for path_index in range(len(meta["order"])):
        row: dict[str, object] = {
            "path_index": path_index,
            "scattering_order": int(meta["order"][path_index]),
        }
        for key, values in meta.items():
            if key == "order":
                continue
            try:
                item = values[path_index]
            except (IndexError, TypeError):
                continue
            array = np.asarray(item)
            if array.ndim == 0:
                row[key] = python_scalar(array.item())
            else:
                for position, value in enumerate(array.reshape(-1), start=1):
                    row[f"{key}_{position}"] = python_scalar(value)
        rows.append(row)
    return pd.DataFrame(rows)


def amplitude_statistics(observed: np.ndarray) -> dict[str, object]:
    observed = np.asarray(observed, dtype=np.float32)
    if observed.size == 0:
        raise ValueError("Evento sin muestras de audio observadas")
    if not np.isfinite(observed).all():
        raise ValueError("Evento con muestras NaN o infinitas")
    peak = float(np.max(np.abs(observed)))
    rms = float(np.sqrt(np.mean(np.square(observed, dtype=np.float64))))
    return {
        "peak_amplitude": peak,
        "rms_amplitude": rms,
        "log10_peak": float(np.log10(peak + AMPLITUDE_EPSILON)),
        "log10_rms": float(np.log10(rms + AMPLITUDE_EPSILON)),
        "log10_crest_factor": float(
            np.log10(
                (peak + AMPLITUDE_EPSILON)
                / (rms + AMPLITUDE_EPSILON)
            )
        ),
        "near_silence_event": bool(
            peak < NEAR_SILENCE_AUDIT_THRESHOLD
        ),
    }


def normalize_for_wst(
    waveform: np.ndarray,
    peak: float,
) -> tuple[np.ndarray, bool]:
    normalized = np.asarray(waveform, dtype=np.float32).copy()
    applied = peak > 0.0
    if applied:
        normalized /= peak
    if not np.isfinite(normalized).all():
        raise RuntimeError("La normalizacion produjo NaN o infinito")
    return normalized, applied


def recording_manifest_dataframe(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    constant_columns = (
        "cough_type",
        "cough_type_consensus",
        "stage2_target",
        "fold",
        "split",
        "dataset_origin",
    )
    for recording_row, (original_uuid, group) in enumerate(
        metadata.groupby("original_uuid", sort=False)
    ):
        for column in constant_columns:
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"{column} no es constante en {original_uuid}"
                )
        rows.append(
            {
                "recording_row": recording_row,
                "original_uuid": str(original_uuid),
                "cough_type": str(group["cough_type"].iloc[0]),
                "cough_type_consensus": str(
                    group["cough_type_consensus"].iloc[0]
                ),
                "stage2_target": int(group["stage2_target"].iloc[0]),
                "fold": int(group["fold"].iloc[0]),
                "split": str(group["split"].iloc[0]),
                "dataset_origin": str(group["dataset_origin"].iloc[0]),
                "first_feature_row": int(group["feature_row"].min()),
                "last_feature_row": int(group["feature_row"].max()),
                "event_count": len(group),
                "effective_event_count": float(group["window_weight"].sum()),
                "padded_event_count": int(
                    (group["window_total_padding"].astype(float) > 0.0).sum()
                ),
                "near_silence_event_count": int(
                    group["near_silence_event"].astype(bool).sum()
                ),
                "minimum_valid_fraction": float(
                    group["valid_fraction"].min()
                ),
                "mean_valid_fraction": float(
                    group["valid_fraction"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def extract_split(
    split: str,
    manifest: pd.DataFrame,
    config: wst_base.ScatteringConfig,
    scattering: Scattering1D,
    path_count: int,
    batch_size: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = np.empty((len(manifest), path_count), dtype=np.float32)
    diagnostics: list[dict[str, object] | None] = [None] * len(manifest)
    errors: list[dict[str, object]] = []
    pending_waveforms: list[np.ndarray] = []
    pending: list[tuple[int, dict[str, object], bool]] = []

    def flush_batch() -> None:
        if not pending_waveforms:
            return
        features, elapsed = wst_base.process_batch(
            scattering,
            pending_waveforms,
            path_count,
        )
        seconds_per_event = elapsed / len(pending_waveforms)
        for batch_row, (output_row, amplitude, normalized) in enumerate(pending):
            matrix[output_row] = features[batch_row]
            diagnostics[output_row] = {
                **amplitude,
                "peak_normalized_for_wst": normalized,
                "scattering_seconds_per_event": seconds_per_event,
            }
        pending_waveforms.clear()
        pending.clear()

    for output_row, (_, row) in enumerate(
        tqdm(
            manifest.iterrows(),
            total=len(manifest),
            desc=f"WST mejorada {split}",
        )
    ):
        try:
            waveform, observed = audio_base.load_event_waveform(
                row, audio_base.PRESETS["paper64"]
            )
            amplitude = amplitude_statistics(observed)
            normalized, normalization_applied = normalize_for_wst(
                waveform, float(amplitude["peak_amplitude"])
            )
            pending_waveforms.append(normalized)
            pending.append(
                (output_row, amplitude, normalization_applied)
            )
            if len(pending_waveforms) >= batch_size:
                flush_batch()
        except Exception as exc:
            errors.append(
                {
                    "manifest_row": output_row,
                    "split": split,
                    "event_id": row.get("event_id", ""),
                    "original_uuid": row.get("original_uuid", ""),
                    "source_audio_path": row.get("source_audio_path", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    flush_batch()

    issues = pd.DataFrame(errors)
    if errors:
        return matrix, pd.DataFrame(), pd.DataFrame(), issues
    if any(item is None for item in diagnostics):
        raise RuntimeError(f"Diagnosticos incompletos en {split}")
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"La matriz WST de {split} contiene NaN o infinito")

    metadata = manifest.reset_index(drop=True).copy()
    metadata.insert(0, "feature_row", np.arange(len(metadata), dtype=int))
    diagnostic_frame = pd.DataFrame(diagnostics)
    for column in diagnostic_frame.columns:
        metadata[column] = diagnostic_frame[column].to_numpy()

    valid_fraction = (
        metadata["window_observed_duration"].to_numpy(float)
        / metadata["target_window_duration"].to_numpy(float)
    )
    if not np.all((valid_fraction > 0.0) & (valid_fraction <= 1.0 + 1e-6)):
        raise RuntimeError(f"Fracciones validas fuera de rango en {split}")
    metadata["valid_fraction"] = np.clip(valid_fraction, 0.0, 1.0)
    metadata["window_weight"] = metadata["valid_fraction"]

    event_counts = metadata.groupby("original_uuid")["event_id"].transform(
        "count"
    )
    metadata["equal_event_recording_weight"] = 1.0 / event_counts.astype(float)
    weight_sums = metadata.groupby("original_uuid")["window_weight"].transform(
        "sum"
    )
    if np.any(weight_sums <= 0.0):
        raise RuntimeError(f"Una grabacion de {split} tiene peso total nulo")
    metadata["recording_pooling_weight"] = (
        metadata["window_weight"] / weight_sums
    )
    normalized_sums = metadata.groupby("original_uuid")[
        "recording_pooling_weight"
    ].sum()
    if not np.allclose(normalized_sums.to_numpy(float), 1.0, atol=1e-9):
        raise RuntimeError(f"Los pesos por grabacion no suman uno en {split}")

    recordings = recording_manifest_dataframe(metadata)
    if not np.array_equal(
        metadata["stage2_target"].to_numpy(np.int64),
        manifest["stage2_target"].to_numpy(np.int64),
    ):
        raise RuntimeError(f"Etiquetas desalineadas en {split}")
    return matrix, metadata, recordings, issues


def split_output_files(split: str) -> tuple[Path, ...]:
    return (
        OUTPUT_DIR / f"X_events_raw_{split}.npy",
        OUTPUT_DIR / f"y_events_{split}.npy",
        OUTPUT_DIR / f"folds_events_{split}.npy",
        OUTPUT_DIR / f"metadata_events_{split}.csv",
        OUTPUT_DIR / f"metadata_recordings_{split}.csv",
        OUTPUT_DIR / f"extraction_issues_{split}.csv",
    )


def validate_outputs_do_not_exist(overwrite: bool) -> None:
    existing = [
        path for split in SPLITS for path in split_output_files(split) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Ya existe {existing[0]}. Usa --overwrite para regenerar este preset."
        )


def save_split(
    split: str,
    result: tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> dict[str, object]:
    matrix, metadata, recordings, issues = result
    np.save(OUTPUT_DIR / f"X_events_raw_{split}.npy", matrix)
    np.save(
        OUTPUT_DIR / f"y_events_{split}.npy",
        metadata["stage2_target"].to_numpy(np.int64),
    )
    np.save(
        OUTPUT_DIR / f"folds_events_{split}.npy",
        metadata["fold"].to_numpy(np.int32),
    )
    metadata.to_csv(
        OUTPUT_DIR / f"metadata_events_{split}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    recordings.to_csv(
        OUTPUT_DIR / f"metadata_recordings_{split}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if issues.empty:
        issues = pd.DataFrame(
            columns=[
                "manifest_row",
                "split",
                "event_id",
                "original_uuid",
                "source_audio_path",
                "error",
            ]
        )
    issues.to_csv(
        OUTPUT_DIR / f"extraction_issues_{split}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    targets = recordings["stage2_target"].to_numpy(np.int8)
    counts = np.bincount(targets, minlength=2)
    return {
        "split": split,
        "events": len(metadata),
        "recordings": len(recordings),
        "dry_recordings": int(counts[0]),
        "wet_recordings": int(counts[1]),
        "padded_events": int(
            (metadata["window_total_padding"].astype(float) > 0.0).sum()
        ),
        "near_silence_events": int(
            metadata["near_silence_event"].astype(bool).sum()
        ),
        "minimum_valid_fraction": float(metadata["valid_fraction"].min()),
    }


def save_common_files(
    config: wst_base.ScatteringConfig,
    layout: dict[str, object],
    paths: pd.DataFrame,
    batch_size: int,
    summaries: list[dict[str, object]],
) -> None:
    paths.to_csv(
        OUTPUT_DIR / "wst_paths.csv", index=False, encoding="utf-8-sig"
    )
    configuration = {
        "experiment": "stage2_wst_recording_enhanced_coughvid_only",
        "preset": PRESET_NAME,
        "reference_extractor": "feature_extraction_stage2_wavelet_scattering.py",
        "reference_model": "WST recording-level + LR PCA128",
        "reference_validation_macro_f1": 0.6257,
        **asdict(config),
        "target_samples": config.target_samples,
        "invariance_samples": config.invariance_samples,
        "J": config.j,
        "Q": str(config.q),
        "backend": "kymatio.numpy",
        "batch_size": batch_size,
        "dataset_scope": "COUGHVID_only_dry_wet",
        "source_events": "final_nonoverlap_stage2_train_validation",
        "stored_representation": "raw_WST_per_event_after_temporal_mean",
        "coefficient_transform": "none",
        "recording_pooling_applied": False,
        "planned_recording_pooling": "weighted_mean_std_and_unweighted_max",
        "event_weight": "window_observed_duration/target_window_duration",
        "amplitude_measured_before_peak_normalization": True,
        "near_silence_policy": "keep_and_audit",
        "log_transform_applied": False,
        "scaler_pca_model_fitted": False,
        "test_processed": False,
        **layout,
    }
    pd.DataFrame(configuration.items(), columns=["parameter", "value"]).to_csv(
        OUTPUT_DIR / "wst_extraction_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "wst_extraction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def print_check(
    config: wst_base.ScatteringConfig,
    manifests: dict[str, pd.DataFrame],
    layout: dict[str, object],
    batch_size: int,
) -> None:
    print("=" * 78)
    print("CHECK - STAGE 2 WST RECORDING ENHANCED")
    print("=" * 78)
    print(f"Preset: {PRESET_NAME}")
    print("Referencia: WST recording-level + LR/PCA128")
    print("Datos: solo COUGHVID dry/wet; sin FSD50K ni gold augmentation")
    print(
        f"WST: Q={config.q}, J={config.j}, T={config.invariance_samples}, "
        f"orden maximo={config.max_order}"
    )
    print(
        f"Caminos 0/1/2: {layout['order_0_paths']} / "
        f"{layout['order_1_paths']} / {layout['order_2_paths']}"
    )
    print(
        f"Salida: N_eventos x {layout['feature_dimension']} raw; "
        "sin log ni pooling recording"
    )
    print(
        "Peso futuro: fraccion observada; weighted mean/std y max sin ponderar"
    )
    print(f"Batch size: {batch_size}")
    for split, manifest in manifests.items():
        recordings = manifest.drop_duplicates("original_uuid")
        recording_counts = np.bincount(
            recordings["stage2_target"].to_numpy(np.int8), minlength=2
        )
        estimated_mib = (
            len(manifest) * int(layout["feature_dimension"]) * 4 / (1024**2)
        )
        print(
            f"{split:>10}: {len(manifest)} eventos, "
            f"{len(recordings)} grabaciones dry/wet="
            f"{recording_counts[0]}/{recording_counts[1]}, "
            f"matriz={estimated_mib:.1f} MiB"
        )
    print("TEST no sera leido ni procesado.")


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size debe ser mayor que cero")

    manifests = load_and_validate_manifests()
    config = wst_base.PRESETS["paper_q8_q1_t500_full"]
    scattering = wst_base.build_scattering(config)
    path_count, _, orders, layout = wst_base.infer_layout(scattering, config)
    if set(orders.tolist()) != {0, 1, 2}:
        raise RuntimeError("La extraccion Stage 2 no contiene S0+S1+S2")
    paths = wst_paths_dataframe(scattering)
    print_check(config, manifests, layout, args.batch_size)

    if args.action == "check":
        print("Comprobacion completada. No se escribio ningun archivo.")
        return

    validate_outputs_do_not_exist(args.overwrite)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    started = time.perf_counter()
    for split in SPLITS:
        result = extract_split(
            split=split,
            manifest=manifests[split],
            config=config,
            scattering=scattering,
            path_count=path_count,
            batch_size=args.batch_size,
        )
        issues = result[3]
        if not issues.empty:
            issues.to_csv(
                OUTPUT_DIR / f"extraction_issues_{split}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            raise RuntimeError(
                f"Fallaron {len(issues)} eventos de {split}. No se guardara "
                "una matriz parcial."
            )
        summaries.append(save_split(split, result))

    save_common_files(config, layout, paths, args.batch_size, summaries)
    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print("EXTRACCION STAGE 2 WST RECORDING ENHANCED COMPLETADA")
    print("=" * 78)
    for summary in summaries:
        print(
            f"{summary['split']:>10}: eventos={summary['events']}; "
            f"grabaciones={summary['recordings']}; dry/wet="
            f"{summary['dry_recordings']}/{summary['wet_recordings']}; "
            f"padding={summary['padded_events']}"
        )
    print(f"Tiempo total: {elapsed:.2f} s")
    print(f"Resultados: {OUTPUT_DIR}")
    print("No se ha aplicado log, pooling, scaler, PCA ni clasificador.")
    print("TEST permanece reservado.")


if __name__ == "__main__":
    main()
