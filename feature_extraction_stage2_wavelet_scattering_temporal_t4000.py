"""Extrae WST temporal para Stage 2 con T=4000 y J=13 fijo.

Es una version paralela del extractor temporal T=8000. Mantiene los mismos
eventos COUGHVID, la ventana de 1,5 s, Q=(8,1), J=13, orden maximo 2 y la
normalizacion por pico. El unico cambio del scattering es T=4000 muestras
(250 ms). Los coeficientes se guardan sin pooling temporal con forma
``(evento, path, tiempo)`` para comparar posteriormente mean frente a
mean+std+max usando el mismo script de entrenamiento.

Solo procesa TRAIN y VALIDATION. TEST no se lee ni se procesa. No modifica ni
sobrescribe el preset T=8000.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from kymatio.numpy import Scattering1D

import feature_extraction_stage2_cochleograms as audio_base
import feature_extraction_stage2_wavelet_scattering as pooled
import feature_extraction_stage2_wavelet_scattering_temporal as temporal


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = (
    SCRIPT_DIR
    / "features_extracted_stage2_dry_wet_wavelet_scattering_temporal"
)
PRESET_NAME = "paper_q8_q1_t250_j13_full"
SPLITS = ("train", "validation")


@dataclass(frozen=True)
class FixedJScatteringConfig:
    name: str = PRESET_NAME
    sample_rate: int = 16_000
    target_duration_seconds: float = 1.5
    invariance_scale_seconds: float = 0.25
    first_order_wavelets_per_octave: int = 8
    second_order_wavelets_per_octave: int = 1
    max_order: int = 2
    normalize_waveform_peak: bool = True
    fixed_j: int = 13

    @property
    def target_samples(self) -> int:
        return int(round(self.sample_rate * self.target_duration_seconds))

    @property
    def invariance_samples(self) -> int:
        return int(round(self.sample_rate * self.invariance_scale_seconds))

    @property
    def j(self) -> int:
        return self.fixed_j

    @property
    def q(self) -> tuple[int, int]:
        return (
            self.first_order_wavelets_per_octave,
            self.second_order_wavelets_per_octave,
        )


CONFIG = FixedJScatteringConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extrae WST temporal Stage 2 con J=13, Q=(8,1) y T=4000."
        )
    )
    parser.add_argument(
        "--action",
        choices=("check", "extract"),
        default="check",
        help="check valida sin escribir; extract procesa TRAIN/VALIDATION.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar exclusivamente este preset T=4000.",
    )
    return parser


def build_scattering(config: FixedJScatteringConfig) -> Scattering1D:
    return Scattering1D(
        J=config.j,
        shape=config.target_samples,
        Q=config.q,
        T=config.invariance_samples,
        max_order=config.max_order,
        out_type="array",
    )


def output_files(output_dir: Path) -> list[Path]:
    shared = [
        output_dir / "wavelet_scattering_temporal_configuration.csv",
        output_dir / "wavelet_scattering_temporal_path_layout.csv",
        output_dir / "wavelet_scattering_temporal_position_layout.csv",
        output_dir / "temporal_mean_equivalence_with_pooled_wst.csv",
    ]
    split_files = []
    for split in SPLITS:
        split_files.extend(
            [
                output_dir / f"X_events_{split}.npy",
                output_dir / f"y_events_{split}.npy",
                output_dir / f"folds_events_{split}.npy",
                output_dir / f"metadata_events_features_{split}.csv",
            ]
        )
    return [*shared, *split_files]


def check_existing_outputs(output_dir: Path, overwrite: bool) -> None:
    existing = [path for path in output_files(output_dir) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Ya existe {existing[0]}. Usa --overwrite solo si deseas "
            "regenerar el preset T=4000."
        )


def print_check(
    manifests: dict[str, pd.DataFrame],
    layout: dict[str, object],
    batch_size: int,
) -> None:
    path_count = int(layout["path_count"])
    time_count = int(layout["time_position_count"])
    values_per_event = path_count * time_count
    print("=" * 78)
    print("CHECK - WST TEMPORAL T=4000, J=13 - STAGE 2")
    print("=" * 78)
    print(f"Preset nuevo: {CONFIG.name}")
    print(
        f"Audio: {CONFIG.target_samples} muestras a {CONFIG.sample_rate} Hz "
        f"({CONFIG.target_duration_seconds:.1f} s)"
    )
    print(
        f"Q={CONFIG.q}; J={CONFIG.j}; T={CONFIG.invariance_samples} "
        f"({CONFIG.invariance_scale_seconds * 1000:.0f} ms)"
    )
    print(
        "Caminos orden 0/1/2: "
        f"{layout['order_0_paths']} / {layout['order_1_paths']} / "
        f"{layout['order_2_paths']}"
    )
    print(f"Tensor por evento: ({path_count}, {time_count})")
    print(f"Valores conservados por evento: {values_per_event}")
    print("No se aplica media temporal, PCA ni seleccion de paths.")
    print(f"Backend: kymatio.numpy; batch_size={batch_size}")
    for split, manifest in manifests.items():
        estimated_mib = len(manifest) * values_per_event * 4 / (1024**2)
        print(
            f"{split}: {len(manifest)} eventos; "
            f"tensor estimado={estimated_mib:.2f} MiB"
        )
    print("TEST no sera leido ni procesado.")


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size debe ser mayor que cero.")

    manifests = {
        split: audio_base.load_event_manifest(split)
        for split in SPLITS
    }
    audio_base.validate_train_validation_disjoint(
        manifests["train"], manifests["validation"]
    )
    scattering = build_scattering(CONFIG)
    path_count, time_count, _, layout = pooled.infer_layout(
        scattering,
        CONFIG,
    )
    if path_count != 644 or time_count != 11:
        raise RuntimeError(
            "Layout T=4000 inesperado: "
            f"paths={path_count}, posiciones={time_count}."
        )
    path_layout = temporal.temporal_path_layout_dataframe(
        scattering, path_count
    )
    time_layout = temporal.temporal_position_layout_dataframe(time_count)
    print_check(manifests, layout, args.batch_size)

    if args.action == "check":
        print("Comprobacion completada. No se extrajeron features.")
        return

    output_dir = OUTPUT_ROOT / CONFIG.name
    check_existing_outputs(output_dir, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal.save_configuration(
        CONFIG,
        layout,
        path_layout,
        time_layout,
        output_dir,
        args.batch_size,
    )

    comparison_rows = []
    for split in SPLITS:
        shape, comparison = temporal.extract_and_save_split(
            split,
            manifests[split],
            CONFIG,
            scattering,
            path_count,
            time_count,
            args.batch_size,
            output_dir,
        )
        comparison_rows.append(comparison)
        print(f"{split}: tensor guardado {shape}")
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "temporal_mean_equivalence_with_pooled_wst.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 78)
    print("EXTRACCION WST TEMPORAL T=4000 COMPLETADA")
    print("=" * 78)
    print(f"Resultados: {output_dir}")
    print("El preset T=8000 y los extractores anteriores no se han modificado.")
    print("TEST permanece sin procesar en este extractor.")


if __name__ == "__main__":
    main()
