"""Entrena Stage 1 MFCC117 + LR con augmentation solo de ruido.

Es una ablacion paralela al experimento de ruido+pitch. Mantiene congelados
el clasificador LR, el umbral 0.5, los folds y el peso completo de cada
variante. Cada tos de TRAIN recibe una unica variante, seleccionada de forma
reproducible entre 10 y 20 dB SNR, sin modificar el tono.
"""

from pathlib import Path

import train_stage1_mfcc_lr_cough_augmentation as experiment


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = "mfcc117_lr_fsd50k_cough_segments_noise_aug_random"

experiment.AUGMENTED_FEATURES_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_cough_segments_noise_aug_random"
    / "mfcc117"
)
experiment.EXPERIMENT_NAME = EXPERIMENT_NAME
experiment.RESULTS_DIR = (
    ROOT / "results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
)
experiment.GRAPHS_DIR = (
    ROOT / "graphs_results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
)
experiment.MODEL_PATH = (
    experiment.RESULTS_DIR / "stage1_lr_cough_segments_noise_aug.joblib"
)
experiment.AUGMENTATION_EXTRACTION_SCRIPT = (
    "feature_extraction_stage1_mfcc_cough_augmentation_noise_only.py"
)
experiment.AUGMENTATION_DISPLAY_NAME = "AUGMENTATION SOLO CON RUIDO"
experiment.AUGMENTATION_PROTOCOL = (
    "one deterministic random real-negative mixture at 10 or 20 dB SNR per "
    "TRAIN cough; no pitch shift"
)
experiment.EXPECTED_VARIANTS_PER_COUGH = 1

# Las funciones compartidas de carga, graficas y TEST escriben mediante estas
# variables globales del modulo base.
experiment.base.FEATURES_DIR = experiment.ORIGINAL_FEATURES_DIR
experiment.base.EXPERIMENT_NAME = EXPERIMENT_NAME
experiment.base.RESULTS_DIR = experiment.RESULTS_DIR
experiment.base.GRAPHS_DIR = experiment.GRAPHS_DIR
experiment.base.MODEL_PATH = experiment.MODEL_PATH


if __name__ == "__main__":
    experiment.main()
