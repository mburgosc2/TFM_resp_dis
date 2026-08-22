"""Late fusion recording-level entre WST-LR y MFCC-LR.

Combina probabilidades alineadas por ``original_uuid`` mediante:

    p_fusion = alpha * p_wst + (1 - alpha) * p_mfcc

Alpha y umbral se seleccionan exclusivamente con predicciones OOF de TRAIN.
La estabilidad se comprueba mediante leave-one-existing-fold-out y la
configuracion final se aplica una sola vez a VALIDATION. TEST no se lee.

El script configura el motor comun de late fusion y guarda resultados,
graficas y configuracion en directorios independientes de los experimentos
con cocleogramas.
"""

from pathlib import Path

import train_stage2_late_fusion_wst_cochleograms as fusion


ROOT = Path(__file__).resolve().parent

fusion.EXPERIMENT_KEY = "late_fusion_wst_mfcc_recording"
fusion.DISPLAY_TITLE = "STAGE 2 — LATE FUSION WST-LR + MFCC-LR RECORDING"
fusion.CHECK_TITLE = "CHECK — LATE FUSION WST-LR + MFCC-LR RECORDING"
fusion.PARSER_DESCRIPTION = (
    "Late fusion recording-level de WST-LR y MFCC-LR."
)
fusion.VALIDATION_GRAPH_FILENAME = (
    "validation_late_fusion_wst_mfcc_oof_threshold.png"
)
fusion.SECONDARY_DISPLAY_LABEL = "MFCC"
fusion.CANDIDATE_COMPARISON_FILENAME = "mfcc_candidate_comparison.csv"
fusion.RESULT_TITLE = "RESULTADO LATE FUSION WST + MFCC"
fusion.SECONDARY_WEIGHT_LABEL = "peso MFCC"

fusion.RESULTS_DIR = ROOT / "results_stage2_late_fusion_wst_mfcc" / "full"
fusion.GRAPHS_DIR = (
    ROOT
    / "graphs_results_stage2_dry_wet_random"
    / "late_fusion_wst_mfcc"
    / "full"
)
fusion.MODELS_DIR = ROOT / "models_stage2_late_fusion_wst_mfcc" / "full"

fusion.COCH_CANDIDATES = (
    fusion.CochCandidate(
        key="mfcc117_recording_lr",
        display_name="MFCC117 mean/std/max + PCA128 + LR",
        result_dir=(
            ROOT
            / "results_stage2_dry_wet_mfcc_recording_linear"
            / "mfcc117"
            / "full"
            / "logistic_regression"
        ),
        model_path=(
            ROOT
            / "models_stage2_dry_wet_mfcc_recording_linear"
            / "mfcc117"
            / "full"
            / "logistic_regression_model.joblib"
        ),
    ),
)


if __name__ == "__main__":
    fusion.main()
