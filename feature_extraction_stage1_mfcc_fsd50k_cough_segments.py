"""Extrae MFCC117 usando las toses FSD50K segmentadas y reauditadas.

Es una entrada paralela al extractor original. Reutiliza exactamente la misma
implementacion MFCC117, pero lee los metadatos donde cada fragmento FSD50K de
mas de 10 s ya tiene una decision manual independiente.
"""

from __future__ import annotations

from pathlib import Path

import feature_extraction_stage1_mfcc_fsd50k_coughs as base


ROOT = Path(__file__).resolve().parent

base.METADATA_DIR = ROOT / "metadata_splits_stage1_fsd50k_cough_segments_random"
base.OUTPUT_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_cough_segments_random"
    / "mfcc117"
)
base.FEATURE_EXPERIMENT_NAME = (
    "stage1_mfcc117_fsd50k_cough_segments_random"
)
base.METADATA_PREPARATION_COMMAND = (
    "python .\\prepare_stage1_fsd50k_cough_segment_review.py "
    "--action finalize"
)


if __name__ == "__main__":
    base.main()
