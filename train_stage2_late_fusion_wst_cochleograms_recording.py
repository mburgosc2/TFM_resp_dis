"""Late fusion de dos modelos que predicen directamente por grabacion.

Combina:

* WST mean/std/max + StandardScaler + PCA128 + Logistic Regression.
* Cocleograma mean/std/max + StandardScaler + Logistic Regression.

La formula es ``alpha * p_wst + (1 - alpha) * p_coch``. El alpha y el
umbral de decision se seleccionan exclusivamente con predicciones OOF de
TRAIN. Una sensibilidad leave-one-fold-out comprueba la estabilidad de esa
seleccion antes de aplicar la configuracion congelada a VALIDATION. TEST no
se lee ni se procesa.

Este archivo configura y reutiliza el motor comun de la fusion por evento,
evitando duplicar la logica de alineacion, busqueda, metricas y graficas.
Los resultados se guardan en directorios independientes.
"""

from pathlib import Path

import train_stage2_late_fusion_wst_cochleograms as fusion


ROOT = Path(__file__).resolve().parent

fusion.EXPERIMENT_KEY = "late_fusion_wst_cochleograms_recording"
fusion.DISPLAY_TITLE = (
    "STAGE 2 — LATE FUSION WST-LR RECORDING + COCLEOGRAMA-LR RECORDING"
)
fusion.CHECK_TITLE = (
    "CHECK — LATE FUSION WST-LR RECORDING + COCLEOGRAMA-LR RECORDING"
)
fusion.PARSER_DESCRIPTION = (
    "Late fusion de WST-LR y cocleograma-LR, ambos recording-level."
)
fusion.VALIDATION_GRAPH_FILENAME = (
    "validation_late_fusion_recording_oof_threshold.png"
)

fusion.RESULTS_DIR = (
    ROOT / "results_stage2_late_fusion_wst_cochleograms_recording" / "full"
)
fusion.GRAPHS_DIR = (
    ROOT
    / "graphs_results_stage2_dry_wet_random"
    / "late_fusion_wst_cochleograms_recording"
    / "full"
)
fusion.MODELS_DIR = (
    ROOT / "models_stage2_late_fusion_wst_cochleograms_recording" / "full"
)

fusion.COCH_CANDIDATES = (
    fusion.CochCandidate(
        key="coch_summary_recording_lr",
        display_name="Cocleograma mean/std/max + LR por grabacion",
        result_dir=(
            ROOT
            / "results_stage2_dry_wet_cochleograms"
            / "paper64"
            / "full"
            / "recording"
        ),
        model_path=(
            ROOT
            / "models_stage2_dry_wet_cochleograms"
            / "paper64"
            / "full"
            / "recording_model.joblib"
        ),
    ),
)


if __name__ == "__main__":
    fusion.main()
