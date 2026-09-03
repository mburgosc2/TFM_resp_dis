"""Stage 1 MFCC117 + LR con ruido y peso normalizado por familia.

Reutiliza la unica variante ruidosa ya extraida para cada tos de TRAIN y
aplica los siguientes pesos tanto al StandardScaler como a la LR::

    negativo original = 1.00
    tos original      = 0.50
    variante ruidosa  = 0.50

La contribucion total de cada familia positiva es 1.00. El clasificador, el
umbral 0.5, los folds y los conjuntos de evaluacion permanecen congelados.
"""

from pathlib import Path

import train_stage1_mfcc_lr_cough_augmentation_family_weighted as experiment


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = (
    "mfcc117_lr_fsd50k_cough_segments_noise_aug_family_weighted_random"
)
NOISE_FEATURES_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_cough_segments_noise_aug_random"
    / "mfcc117"
)
FULL_WEIGHT_NOISE_RESULTS = (
    ROOT
    / "results_stage1_cough_no_cough"
    / "mfcc117_lr_fsd50k_cough_segments_noise_aug_random"
    / "full"
)

experiment.AUGMENTED_FEATURES_DIR = NOISE_FEATURES_DIR
experiment.EXPERIMENT_NAME = EXPERIMENT_NAME
experiment.RESULTS_DIR = (
    ROOT / "results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
)
experiment.GRAPHS_DIR = (
    ROOT / "graphs_results_stage1_cough_no_cough" / EXPERIMENT_NAME / "full"
)
experiment.MODEL_PATH = (
    experiment.RESULTS_DIR
    / "stage1_lr_cough_segments_noise_aug_family_weighted.joblib"
)
experiment.COUGH_ORIGINAL_WEIGHT = 0.50
experiment.COUGH_AUGMENTED_WEIGHT = 0.50
experiment.EXPECTED_VARIANTS_PER_COUGH = 1
experiment.AUGMENTATION_DISPLAY_NAME = (
    "AUGMENTATION SOLO RUIDO PONDERADA POR FAMILIA"
)
experiment.AUGMENTATION_PROTOCOL = (
    "one deterministic random real-negative mixture at 10 or 20 dB SNR per "
    "TRAIN cough; no pitch shift"
)
experiment.WEIGHT_DESCRIPTION = (
    "negativo original=1.00; tos original=0.50; variante ruidosa=0.50; "
    "total por familia de tos=1.00"
)
experiment.REFERENCE_EXPERIMENTS = {
    "no_augmentation": (
        ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_random"
        / "full"
    ),
    "full_weight_noise_augmentation": FULL_WEIGHT_NOISE_RESULTS,
}

# Las funciones del experimento de augmentation cargan mediante estas
# variables globales. Se actualizan para leer las variantes solo-ruido.
experiment.augmentation_base.AUGMENTED_FEATURES_DIR = NOISE_FEATURES_DIR
experiment.augmentation_base.EXPECTED_VARIANTS_PER_COUGH = 1
experiment.augmentation_base.AUGMENTATION_EXTRACTION_SCRIPT = (
    "feature_extraction_stage1_mfcc_cough_augmentation_noise_only.py"
)

# Las utilidades comunes escriben resultados, graficas y TEST en las rutas
# exclusivas de esta ablacion.
experiment.base.FEATURES_DIR = experiment.ORIGINAL_FEATURES_DIR
experiment.base.EXPERIMENT_NAME = EXPERIMENT_NAME
experiment.base.RESULTS_DIR = experiment.RESULTS_DIR
experiment.base.GRAPHS_DIR = experiment.GRAPHS_DIR
experiment.base.MODEL_PATH = experiment.MODEL_PATH


if __name__ == "__main__":
    experiment.main()
