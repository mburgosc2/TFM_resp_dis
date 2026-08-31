"""Repite MFCC117 + LR con las toses FSD50K segmentadas y reauditadas.

La configuracion del clasificador permanece congelada para que el cambio
respecto al experimento anterior proceda exclusivamente del tratamiento de
los clips FSD50K largos::

    StandardScaler -> LogisticRegression(
        C=0.3, penalty="l1", solver="liblinear", class_weight=None
    )

TRAIN se evalua mediante los cinco folds agrupados existentes, VALIDATION se
consulta despues de ajustar el modelo final y TEST solo se carga mediante
``--action test``.
"""

from __future__ import annotations

from pathlib import Path

import train_stage1_mfcc_lr_fsd50k_coughs as base


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = "mfcc117_lr_fsd50k_cough_segments_random"

base.FEATURES_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_cough_segments_random"
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
    base.RESULTS_DIR / "stage1_lr_fsd50k_cough_segments.joblib"
)
base.FEATURE_EXTRACTION_COMMAND = (
    "python .\\feature_extraction_stage1_mfcc_fsd50k_cough_segments.py "
    "--action extract --near-silence-policy reject"
)


if __name__ == "__main__":
    base.main()
