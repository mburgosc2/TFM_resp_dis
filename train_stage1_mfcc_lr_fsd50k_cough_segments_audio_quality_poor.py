"""Entrena MFCC117 + LR en el experimento de robustez ``quality=poor``.

La configuracion del modelo permanece congelada (C=0.3, L1, sin pesos de
clase y umbral 0.5). TRAIN y VALIDATION no contienen toses COUGHVID poor.
TEST contiene como positivos exclusivamente las 270 toses COUGHVID poor y no
se lee hasta solicitar explicitamente ``--action test``.

Uso::

    py -3.10 .\train_stage1_mfcc_lr_fsd50k_cough_segments_audio_quality_poor.py --action train
    py -3.10 .\train_stage1_mfcc_lr_fsd50k_cough_segments_audio_quality_poor.py --action test
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import train_stage1_mfcc_lr_fsd50k_coughs as base


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = "mfcc117_lr_fsd50k_cough_segments_audio_quality_poor"

base.FEATURES_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_cough_segments_audio_quality_poor"
    / "mfcc117"
)
base.EXPERIMENT_NAME = EXPERIMENT_NAME
base.RESULTS_DIR = (
    ROOT / "results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
)
base.GRAPHS_DIR = (
    ROOT / "graphs_results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
)
base.MODEL_PATH = (
    base.RESULTS_DIR
    / "stage1_lr_fsd50k_cough_segments_audio_quality_poor.joblib"
)
base.FEATURE_EXTRACTION_COMMAND = (
    "py -3.10 .\\splits_analysis_stage1_fsd50k_cough_segments_audio_quality.py "
    "--action write\n"
    "py -3.10 .\\repartition_stage1_mfcc_fsd50k_cough_segments_audio_quality.py "
    "--action write"
)
base.TRAIN_TITLE = (
    "STAGE 1 - MFCC117 + LR - TRAIN SIN COUGHVID POOR / TEST POOR"
)
base.RESULT_TITLE = (
    "RESULTADO STAGE 1 - LR + FSD50K - ROBUSTEZ AUDIO QUALITY POOR"
)
base.TEST_TITLE = (
    "EVALUACION FINAL TEST - STAGE 1 LR - COUGHVID QUALITY POOR"
)
base.SPLIT_PROTOCOL = (
    "all COUGHVID poor coughs forced to TEST; TEST positives are only "
    "COUGHVID poor; negative test groups held out; CoughVID grouped by UUID "
    "and FSD50K by uploader"
)
base.EXTRA_CONFIGURATION = {
    "quality_stress_test": True,
    "forced_test_cough_quality": "poor",
    "test_positive_policy": "only COUGHVID coughs with quality=poor",
    "fsd50k_cough_quality": "not_available",
    "fsd50k_cough_split_policy": "TRAIN or VALIDATION only",
    "comparison_role": "secondary robustness analysis; random split is primary",
}


def load_manifest(split: str) -> pd.DataFrame:
    path = base.FEATURES_DIR / f"metadata_features_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def poor_cough_mask(manifest: pd.DataFrame) -> pd.Series:
    return (
        manifest["stage1_target"].astype(int).eq(1)
        & manifest["dataset_origin"].astype(str).eq("COUGHVID")
        & manifest["quality"].astype(str).str.casefold().eq("poor")
    )


def validate_development_protocol() -> None:
    for split in ["train", "validation"]:
        manifest = load_manifest(split)
        if poor_cough_mask(manifest).any():
            bad = manifest.loc[poor_cough_mask(manifest), "uuid_segmento"].iloc[0]
            raise ValueError(f"Una tos poor entro en {split}: {bad}")


def validate_test_protocol() -> None:
    manifest = load_manifest("test")
    positives = manifest[manifest["stage1_target"].astype(int).eq(1)]
    if len(positives) != 270:
        raise ValueError(
            f"TEST deberia contener 270 toses poor y contiene {len(positives)}"
        )
    if not poor_cough_mask(positives).all():
        raise ValueError("TEST contiene una tos que no es COUGHVID poor")
    is_new = base.parse_bool_series(
        manifest["is_new_fsd50k_cough"], "is_new_fsd50k_cough"
    )
    if is_new.any():
        raise ValueError("Una tos FSD50K con calidad desconocida entro en TEST")


def main() -> None:
    args = base.parse_args()
    if not base.FEATURES_DIR.is_dir():
        raise FileNotFoundError(
            f"No existe {base.FEATURES_DIR}. Ejecuta primero:\n"
            f"{base.FEATURE_EXTRACTION_COMMAND}"
        )
    base.validate_feature_configuration()
    if args.action == "train":
        validate_development_protocol()
        base.train(overwrite=args.overwrite)
    else:
        validate_test_protocol()
        base.evaluate_test(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
