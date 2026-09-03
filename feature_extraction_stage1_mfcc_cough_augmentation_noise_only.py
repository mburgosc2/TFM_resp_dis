"""Extrae MFCC117 de una variante con ruido por tos de TRAIN.

Esta ablacion reutiliza el protocolo seguro del experimento de augmentation
original, pero elimina por completo los desplazamientos de tono. Cada tos
genera una unica variante, escogida de forma pseudoaleatoria y reproducible
entre ruido mas intenso a 10 dB SNR y ruido moderado a 20 dB SNR.

Los audios negativos usados como ruido pertenecen al mismo fold que el padre.
VALIDATION y TEST no se leen ni se aumentan.
"""

from pathlib import Path

import feature_extraction_stage1_mfcc_cough_augmentation as experiment


ROOT = Path(__file__).resolve().parent

experiment.TRANSFORMATIONS = ("noise_high", "noise_moderate")
experiment.N_VARIANTS_PER_COUGH = 1
experiment.EXPERIMENT_ID = "stage1_mfcc117_cough_audio_augmentation_noise_only"
experiment.SELECTION_DESCRIPTION = (
    "one deterministic random 10 or 20 dB SNR variant per cough"
)
experiment.OUTPUT_DIR = (
    ROOT
    / "features_extracted_stage1_fsd50k_cough_segments_noise_aug_random"
    / "mfcc117"
)
experiment.AUDIT_DIR = (
    ROOT / "auditory_stage1_mfcc_cough_augmentation_noise_only"
)


if __name__ == "__main__":
    experiment.main()
