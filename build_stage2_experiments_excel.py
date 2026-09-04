"""Genera un Excel resumen de los experimentos Stage 2 dry/wet.

Lee exclusivamente los CSV de resultados ya existentes. El libro se puede
regenerar después de terminar nuevos experimentos para actualizar estados y
métricas sin copiar valores manualmente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "stage2_dry_wet_experiments_summary.xlsx"


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": "C01",
        "name": "Cocleograma resumido por evento + SVM lineal",
        "family": "Cocleograma gammatone resumido",
        "unit": "Evento; media de scores por UUID",
        "representation": "Features estadísticas por bloques",
        "reduction": "Sin PCA",
        "balance": "Pesos clase + 1/eventos por UUID",
        "classifier": "SVM lineal",
        "metrics": "results_stage2_dry_wet_cochleograms/paper64/full/event/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_cochleograms/paper64/full/event/best_cv_fold_metrics.csv",
        "note": "Recall wet alto, pero macro-F1 bajo.",
    },
    {
        "id": "C02",
        "name": "Cocleograma resumido recording + LR",
        "family": "Cocleograma gammatone resumido",
        "unit": "Grabación",
        "representation": "Media+std+máximo entre eventos",
        "reduction": "Sin PCA",
        "balance": "Pesos por clase",
        "classifier": "Regresión logística",
        "metrics": "results_stage2_dry_wet_cochleograms/paper64/full/recording/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_cochleograms/paper64/full/recording/best_cv_fold_metrics.csv",
        "note": "Modelo muy pequeño; separación limitada.",
    },
    {
        "id": "C03",
        "name": "Cocleograma raw PCA64 + SVM lineal",
        "family": "Cocleograma gammatone raw",
        "unit": "Evento; media de scores por UUID",
        "representation": "4096 raw → PCA64",
        "reduction": "PCA64 ponderada",
        "balance": "Pesos clase + 1/eventos por UUID",
        "classifier": "SVM lineal",
        "metrics": "results_stage2_dry_wet_cochleograms_raw_pca/paper64/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_cochleograms_raw_pca/paper64/full/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_cochleograms_raw_pca/paper64/full/experiment_configuration.csv",
        "note": "PCA conserva mucha varianza, pero rendimiento moderado.",
    },
    {
        "id": "C04",
        "name": "Cocleograma raw PCA64 + RF",
        "family": "Cocleograma gammatone raw",
        "unit": "Evento; media de scores por UUID",
        "representation": "4096 raw → PCA64",
        "reduction": "PCA64 ponderada",
        "balance": "Pesos clase + 1/eventos por UUID",
        "classifier": "Random Forest",
        "metrics": "results_stage2_dry_wet_cochleograms_raw_pca_rf/paper64/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_cochleograms_raw_pca_rf/paper64/full/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_cochleograms_raw_pca_rf/paper64/full/experiment_configuration.csv",
        "note": "Mejora ligera frente al SVM raw PCA.",
    },
    {
        "id": "C05",
        "name": "Cocleograma raw PCA64 + XGBoost",
        "family": "Cocleograma gammatone raw",
        "unit": "Evento; media de scores por UUID",
        "representation": "4096 raw → PCA64",
        "reduction": "PCA64 ponderada",
        "balance": "Configuración cost-sensitive de entrenamiento",
        "classifier": "XGBoost",
        "metrics": "results_stage2_dry_wet_cochleograms_raw_pca_xgb/paper64/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_cochleograms_raw_pca_xgb/paper64/full/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_cochleograms_raw_pca_xgb/paper64/full/experiment_configuration.csv",
        "note": "Mejor macro-F1 entre experimentos de cocleograma raw.",
    },
    {
        "id": "W01",
        "name": "WST completo por evento + RF",
        "family": "Wavelet Scattering",
        "unit": "Evento; media de probabilidades por UUID",
        "representation": "644 caminos WST",
        "reduction": "Sin reducción",
        "balance": "Pesos clase + 1/eventos por UUID",
        "classifier": "Random Forest",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_rf/paper_q8_q1_t500_full/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_rf/paper_q8_q1_t500_full/full/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_rf/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "note": "Mejor macro-F1 de VALIDATION hasta ahora.",
    },
    {
        "id": "W02",
        "name": "WST + selección MI + RF",
        "family": "Wavelet Scattering",
        "unit": "Evento; media de probabilidades por UUID",
        "representation": "MI seleccionó 644/644",
        "reduction": "Información mutua; sin compactación final",
        "balance": "Pesos clase + 1/eventos por UUID",
        "classifier": "Random Forest",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_mi_rf/paper_q8_q1_t500_full/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_mi_rf/paper_q8_q1_t500_full/full/selected_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_mi_rf/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "note": "Resultado idéntico a W01: MI no permitió reducir features.",
    },
    {
        "id": "W03",
        "name": "WST completo + SMOTE + RF",
        "family": "Wavelet Scattering",
        "unit": "Evento; media de probabilidades por UUID",
        "representation": "644 caminos WST",
        "reduction": "Sin reducción",
        "balance": "SMOTE 0.75 k=3; sin pesos adicionales",
        "classifier": "Random Forest",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_smote_rf/paper_q8_q1_t500_full/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_smote_rf/paper_q8_q1_t500_full/full/selected_smote_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_smote_rf/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "note": "SMOTE empeoró macro-F1, AUC y recall wet.",
    },
    {
        "id": "W04",
        "name": "WST recording mean+std+max + RF",
        "family": "Wavelet Scattering",
        "unit": "Grabación",
        "representation": "1932: mean+std+max de 644 caminos",
        "reduction": "Sin PCA",
        "balance": "Pesos por clase",
        "classifier": "Random Forest",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_recording_rf/paper_q8_q1_t500_full/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_recording_rf/paper_q8_q1_t500_full/full/best_cv_fold_metrics_oof_threshold.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_recording_rf/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "note": "Modelo 0.59 MB; recall wet algo mayor, macro-F1 menor.",
    },
    {
        "id": "W05",
        "name": "WST recording StandardScaler+PCA+RF",
        "family": "Wavelet Scattering",
        "unit": "Grabación",
        "representation": "1932 mean+std+max",
        "reduction": "Control + PCA32/64/128/256",
        "balance": "Pesos por clase",
        "classifier": "Random Forest",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_recording_pca_rf/paper_q8_q1_t500_full/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_recording_pca_rf/paper_q8_q1_t500_full/full/best_cv_fold_metrics_oof_threshold.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_recording_pca_rf/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "projection": "results_stage2_dry_wet_wavelet_scattering_recording_pca_rf/paper_q8_q1_t500_full/full/projection_cv_comparison.csv",
        "note": "PCA fue descartado; no_pca produjo un RF de 0,20 MB sin mejora en validation.",
    },
    {
        "id": "W06",
        "name": "WST recording StandardScaler/PCA + LR",
        "family": "Wavelet Scattering",
        "unit": "Grabación",
        "representation": "1932 mean+std+max",
        "reduction": "Sin PCA / PCA32/64/128/256",
        "balance": "Pesos por clase",
        "classifier": "Regresión logística",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_recording_linear/paper_q8_q1_t500_full/full/logistic_regression/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_recording_linear/paper_q8_q1_t500_full/full/logistic_regression/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_recording_linear/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "note": "Mejor resultado actual: macro-F1 validation 0,6257 y recall wet 0,6049; PCA128, modelo <1 MB.",
    },
    {
        "id": "W07",
        "name": "WST recording StandardScaler/PCA + SVM lineal",
        "family": "Wavelet Scattering",
        "unit": "Grabación",
        "representation": "1932 mean+std+max",
        "reduction": "Sin PCA / PCA32/64/128/256",
        "balance": "Pesos por clase",
        "classifier": "SVM lineal",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_recording_linear/paper_q8_q1_t500_full/full/linear_svm/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_recording_linear/paper_q8_q1_t500_full/full/linear_svm/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_recording_linear/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "note": "Más compacto que LR, pero menor macro-F1 y recall wet en validation.",
    },
    {
        "id": "W08",
        "name": "WST recording PCA whiten + SVM RBF",
        "family": "Wavelet Scattering",
        "unit": "Grabación",
        "representation": "1932 mean+std+max",
        "reduction": "PCA32/64/128 con whitening",
        "balance": "Pesos por clase",
        "classifier": "SVM RBF",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_recording_rbf_svm/paper_q8_q1_t500_full/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_recording_rbf_svm/paper_q8_q1_t500_full/full/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_recording_rbf_svm/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "note": "Competitivo y compacto, pero retiene 1320 support vectors (81,3 % de TRAIN); menor recall wet que LR.",
    },
    {
        "id": "W09",
        "name": "WST recording mean+std+max + MLP ligera",
        "family": "Wavelet Scattering + deep learning ligero",
        "unit": "Grabacion",
        "representation": "1932: mean+std+max de 644 caminos",
        "reduction": "StandardScaler; sin PCA",
        "balance": "Pesos balanceados por grabacion",
        "classifier": "MLP 32-8 / 64-16",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_recording_mlp/paper_q8_q1_t500_full/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_recording_mlp/paper_q8_q1_t500_full/full/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_recording_mlp/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "note": "Compara dos MLP ligeras sobre WST agregado; scaler, arquitectura, early stopping y umbral se controlan sin usar VALIDATION.",
    },
    {
        "id": "W10",
        "name": "WST recording StandardScaler+PCA128+MLP",
        "family": "Wavelet Scattering + deep learning ligero",
        "unit": "Grabacion",
        "representation": "1932 mean+std+max -> 128 componentes",
        "reduction": "StandardScaler + PCA128 dentro de cada fold",
        "balance": "Pesos balanceados por grabacion",
        "classifier": "MLP 32-8 / 64-16",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_recording_pca_mlp/paper_q8_q1_t500_full/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_recording_pca_mlp/paper_q8_q1_t500_full/full/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_recording_pca_mlp/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "note": "Compara la no linealidad de MLP con la referencia LR usando exactamente PCA128 y una evaluacion OOF anidada.",
    },
    {
        "id": "W11",
        "name": "WST temporal 644x5 + tiny CNN + MIL",
        "family": "Wavelet Scattering temporal + deep learning ligero",
        "unit": "Grabacion como bolsa de eventos",
        "representation": "Cada evento: 5 posiciones temporales x 644 caminos WST",
        "reduction": "Conv1D 1x1 + convolucion temporal separable + pooling MIL",
        "balance": "Pesos balanceados por grabacion",
        "classifier": "Tiny CNN con mean / mean+max / gated attention",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_temporal_mil_cnn/paper_q8_q1_t500_full/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_temporal_mil_cnn/paper_q8_q1_t500_full/full/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_temporal_mil_cnn/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "note": "Conserva la temporalidad WST y aplica una sola etiqueta/perdida por grabacion; attention puede reducir el peso de eventos poco informativos.",
    },
    {
        "id": "W12",
        "name": "WST temporal + MIL attention robusto",
        "family": "Wavelet Scattering temporal + deep learning ligero",
        "unit": "Grabacion como bolsa de eventos",
        "representation": "Cada evento: 5 posiciones temporales x 644 caminos WST",
        "reduction": "Conv1D + gated attention; tres inner splits por outer fold",
        "balance": "Pesos balanceados por grabacion",
        "classifier": "Cuatro arquitecturas tiny CNN attention",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_temporal_mil_attention_robust/paper_q8_q1_t500_full/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_temporal_mil_attention_robust/paper_q8_q1_t500_full/full/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_temporal_mil_attention_robust/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "note": "Compara capacidad y regularizacion; usa la mediana de tres inner splits y registra precision, recall, F1, AUC, loss y learning rate por epoca.",
    },
    {
        "id": "W13",
        "name": "WST recording + LR + augmentation gold",
        "family": "Wavelet Scattering + data augmentation",
        "unit": "Grabacion; OOF y validation solo con originales",
        "representation": "1932 mean+std+max; 48 grabaciones gold aumentadas solo en ajuste",
        "reduction": "StandardScaler + PCA64 dentro de cada fold",
        "balance": "Pesos gold/wet; variantes excluidas junto a su padre del fold OOF",
        "classifier": "Regresion logistica",
        "metrics": "results_stage2_dry_wet_wst_recording_lr_gold_augmentation/paper_q8_q1_t500_full/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wst_recording_lr_gold_augmentation/paper_q8_q1_t500_full/full/best_cv_fold_metrics_original_only.csv",
        "config": "results_stage2_dry_wet_wst_recording_lr_gold_augmentation/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "candidates": "results_stage2_dry_wet_wst_recording_lr_gold_augmentation/paper_q8_q1_t500_full/full/phase_b_candidate_cv_results.csv",
        "note": "Ganador oficial por macro-F1 OOF: C=0,001/PCA64/gold1/wet-cost1,15. Validation macro-F1=0,5845 y recalls dry/wet=0,6754/0,5556: el augmentation no mejora W06. La alternativa con mayor balanced accuracy OOF es C=0,0003/PCA64/gold1/wet-cost1,3 (0,6579; recalls 0,7041/0,6117), pero no fue el ganador del protocolo.",
    },
    {
        "id": "W14",
        "name": "WST recording mejorado + PCA + LR",
        "family": "Wavelet Scattering",
        "unit": "Grabacion",
        "representation": (
            "S1+S2 raw; 1929 features: mean/std ponderadas por "
            "window_weight y max entre eventos"
        ),
        "reduction": "StandardScaler + PCA128 dentro de cada fold",
        "balance": "Pesos balanceados por clase a nivel de grabacion",
        "classifier": "Regresion logistica (C=0,0003)",
        "metrics": "results_stage2_dry_wet_wavelet_scattering_recording_enhanced_lr/paper_q8_q1_t500_window_raw_o012_weighted/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wavelet_scattering_recording_enhanced_lr/paper_q8_q1_t500_window_raw_o012_weighted/full/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_wavelet_scattering_recording_enhanced_lr/paper_q8_q1_t500_window_raw_o012_weighted/full/experiment_configuration.csv",
        "candidates": "results_stage2_dry_wet_wavelet_scattering_recording_enhanced_lr/paper_q8_q1_t500_window_raw_o012_weighted/full/phase_b_candidate_cv_results.csv",
        "note": (
            "Seleccion OOF entre ordenes, raw/log-WST, amplitud y PCA. "
            "Gana S1+S2 raw sin amplitud con PCA128: macro-F1 OOF=0,6289, "
            "pero validation baja a 0,5878 (recalls dry/wet=0,7351/0,4691); "
            "no mejora la referencia W06."
        ),
    },
    {
        "id": "W15",
        "name": "WST recording + LR: busqueda en dos fases",
        "family": "Wavelet Scattering",
        "unit": "Grabacion",
        "representation": "1932: mean+std+max de 644 caminos WST",
        "reduction": "StandardScaler + PCA; seleccion OOF en dos fases",
        "balance": "Multiplicador gold y coste wet seleccionados por OOF",
        "classifier": "Regresion logistica",
        "metrics": "results_stage2_dry_wet_wst_recording_lr_two_phase_search/paper_q8_q1_t500_full/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_wst_recording_lr_two_phase_search/paper_q8_q1_t500_full/full/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_wst_recording_lr_two_phase_search/paper_q8_q1_t500_full/full/experiment_configuration.csv",
        "candidates": "results_stage2_dry_wet_wst_recording_lr_two_phase_search/paper_q8_q1_t500_full/full/phase_b_candidate_cv_results.csv",
        "note": (
            "Fase A selecciona C/PCA/whitening y fase B prueba pesos gold/wet. "
            "El ganador conserva pesos neutros: C=0,001, PCA128 sin whitening. "
            "Validation macro-F1=0,6257; en la evaluacion final de TEST obtiene "
            "macro-F1=0,5942, recalls dry/wet=0,7425/0,4750 y AUC=0,6999."
        ),
    },
    {
        "id": "F01",
        "name": "Late fusion WST-LR + cocleograma event",
        "family": "Fusión WST + cocleograma",
        "unit": "Grabación",
        "representation": "Probabilidades WST y cocleograma",
        "reduction": "Media ponderada por alpha",
        "balance": "Heredado de modelos base",
        "classifier": "Late fusion RF/XGBoost",
        "metrics": "results_stage2_late_fusion_wst_cochleograms/full/metrics_summary.csv",
        "folds": "results_stage2_late_fusion_wst_cochleograms/full/meta_cv_sensitivity_by_fold.csv",
        "config": "results_stage2_late_fusion_wst_cochleograms/full/experiment_configuration.csv",
        "note": "Seleccionó RF y alpha=0,80; ligera mejora OOF que no se mantuvo en validation frente a WST-LR.",
    },
    {
        "id": "F02",
        "name": "Late fusion WST-LR + cocleograma event XGBoost",
        "family": "Fusión WST + cocleograma",
        "unit": "Grabación",
        "representation": "Probabilidades WST y XGBoost-cocleograma",
        "reduction": "Media ponderada por alpha",
        "balance": "Heredado de modelos base",
        "classifier": "Late fusion XGBoost",
        "metrics": "results_stage2_late_fusion_wst_cochleograms/full/coch_raw_pca64_xgb/metrics_summary.csv",
        "config": "results_stage2_late_fusion_wst_cochleograms/full/experiment_configuration.csv",
        "note": "Mejor fusión XGBoost por OOF: alpha=0,80. En validation alcanza macro-F1=0,6210 y recall wet=0,6049; supera a la fusión RF en validation, pero no a WST-LR solo (0,6257). No se usa validation para seleccionarlo.",
    },
    {
        "id": "F03",
        "name": "Late fusion WST-LR + cocleograma recording LR",
        "family": "Fusión WST + cocleograma",
        "unit": "Grabación",
        "representation": "Probabilidades de dos modelos recording-level",
        "reduction": "Media ponderada por alpha",
        "balance": "Heredado de modelos base",
        "classifier": "Late fusion LR/LR",
        "metrics": "results_stage2_late_fusion_wst_cochleograms_recording/full/metrics_summary.csv",
        "folds": "results_stage2_late_fusion_wst_cochleograms_recording/full/meta_cv_sensitivity_by_fold.csv",
        "config": "results_stage2_late_fusion_wst_cochleograms_recording/full/experiment_configuration.csv",
        "note": "OOF selecciona alpha WST=0,65 y cocleograma=0,35 (macro-F1=0,6311), pero la seleccion es inestable entre folds y cae en validation a macro-F1=0,5969 y recall wet=0,3951. No mejora WST-LR solo.",
    },
    {
        "id": "M01",
        "name": "MFCC117 recording + Logistic Regression",
        "family": "MFCC",
        "unit": "Grabación",
        "representation": "MFCC+delta+delta2; mean/std/max temporal y entre eventos",
        "reduction": "StandardScaler + PCA128",
        "balance": "Pesos balanceados por grabación",
        "classifier": "Logistic Regression",
        "metrics": "results_stage2_dry_wet_mfcc_recording_linear/mfcc117/full/logistic_regression/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_mfcc_recording_linear/mfcc117/full/logistic_regression/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_mfcc_recording_linear/mfcc117/full/experiment_configuration.csv",
        "note": "Ganador OOF C=10/PCA128: macro-F1 OOF=0,5764; validation=0,5717 y recall wet=0,4568. Es inferior a WST-LR, pero Pearson OOF=0,4401 y recupera 220 errores de WST, por lo que puede aportar en fusión.",
    },
    {
        "id": "F04",
        "name": "Late fusion WST-LR + MFCC-LR recording",
        "family": "Fusión WST + MFCC",
        "unit": "Grabación",
        "representation": "Probabilidades WST-LR y MFCC-LR recording-level",
        "reduction": "Media ponderada por alpha",
        "balance": "Heredado de modelos base",
        "classifier": "Late fusion LR/LR",
        "metrics": "results_stage2_late_fusion_wst_mfcc/full/metrics_summary.csv",
        "folds": "results_stage2_late_fusion_wst_mfcc/full/meta_cv_sensitivity_by_fold.csv",
        "config": "results_stage2_late_fusion_wst_mfcc/full/experiment_configuration.csv",
        "note": "OOF selecciona alpha WST=0,70 y MFCC=0,30 (macro-F1=0,6325; meta-CV=0,6208). En validation obtiene macro-F1=0,6186 y recall wet=0,4938, por debajo de WST-LR solo; la complementariedad no se aprovecha con un unico alpha global.",
    },
    {
        "id": "L01",
        "name": "Log-Mel 64x92 + tiny depthwise CNN",
        "family": "Log-Mel + deep learning ligero",
        "unit": "Evento; media de probabilidades por UUID",
        "representation": "Log-Mel 64x92x1; ventana 1,5 s",
        "reduction": "CNN separable + global average pooling",
        "balance": "Pesos clase + mismo peso total por grabación",
        "classifier": "Tiny depthwise CNN (5697 parámetros)",
        "metrics": "results_stage2_dry_wet_logmel_tiny_cnn/logmel64_win32_hop16/full/metrics_summary.csv",
        "folds": "results_stage2_dry_wet_logmel_tiny_cnn/logmel64_win32_hop16/full/best_cv_fold_metrics.csv",
        "config": "results_stage2_dry_wet_logmel_tiny_cnn/logmel64_win32_hop16/full/experiment_configuration.csv",
        "note": "Modelo de 128,57 KB: macro-F1 OOF=0,6111 y validation=0,6216; AUC validation=0,6775. Muy compacto, pero recall wet=0,5309, inferior a WST-LR.",
    },
    {
        "id": "F05",
        "name": "Late fusion WST-LR + Log-Mel tiny CNN",
        "family": "Fusión WST + Log-Mel",
        "unit": "Grabación",
        "representation": "Probabilidades WST-LR y CNN Log-Mel por UUID",
        "reduction": "Media ponderada por alpha",
        "balance": "Heredado de los modelos base",
        "classifier": "Late fusion LR/CNN",
        "metrics": "results_stage2_late_fusion_wst_logmel_cnn/full/metrics_summary.csv",
        "folds": "results_stage2_late_fusion_wst_logmel_cnn/full/meta_cv_by_fold.csv",
        "config": "results_stage2_late_fusion_wst_logmel_cnn/full/experiment_configuration.csv",
        "metric_filters": {"policy": "max_macro_f1"},
        "note": "La política max macro-F1 selecciona alpha WST/CNN=0,35/0,65: validation=0,6398 y AUC=0,6915, pero recall wet cae a 0,4815 y meta-CV a 0,6139. Con recall wet OOF>=0,55 se selecciona WST puro; la fusión no mejora el compromiso equilibrado.",
    },
]


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "dry_precision",
    "wet_precision",
    "dry_recall",
    "wet_recall",
    "roc_auc",
    "average_precision_wet",
    "tn_dry_correct",
    "fp_dry_as_wet",
    "fn_wet_as_dry",
    "tp_wet_correct",
]


WST_TEMPORAL_ABLATIONS = [
    (
        "T=8000 (500 ms)",
        "results_stage2_dry_wet_wst_temporal_statistics_lr/"
        "paper_q8_q1_t500_full/full/temporal_statistics_comparison.csv",
    ),
    (
        "T=4000 (250 ms)",
        "results_stage2_dry_wet_wst_temporal_statistics_lr/"
        "paper_q8_q1_t250_j13_full/full/temporal_statistics_comparison.csv",
    ),
]

WST_ORDER_ABLATION = (
    "results_stage2_dry_wet_wst_order_ablation_lr/"
    "paper_q8_q1_t500_full/full/order_ablation_comparison.csv"
)


def absolute(relative: str | None) -> Path | None:
    if not relative:
        return None
    return SCRIPT_DIR / relative


def filter_metrics_for_spec(
    df: pd.DataFrame,
    spec: dict[str, Any],
) -> pd.DataFrame:
    """Aplica filtros declarados sin copiar resultados manualmente."""

    result = df.copy()
    for column, expected_value in spec.get("metric_filters", {}).items():
        if column not in result.columns:
            raise ValueError(
                f"El filtro {column} no existe en las métricas de "
                f"{spec['id']}."
            )
        result = result[result[column].astype(str) == str(expected_value)]
    return result


def first_dataset(df: pd.DataFrame, dataset: str) -> pd.Series | None:
    dataset_column = "dataset" if "dataset" in df.columns else "split"
    if dataset_column not in df.columns:
        return None
    aliases = {
        "train_oof": ("train_oof", "train_oof_original_only"),
        "validation": ("validation", "validation_original_only"),
    }
    accepted = aliases.get(dataset, (dataset,))
    rows = df[df[dataset_column].astype(str).isin(accepted)]
    if rows.empty:
        return None

    if "threshold_policy" in rows.columns:
        priorities = (
            (
                "oof_tuned",
                "oof_tuned_frozen",
                "oof_selected_frozen",
            )
            if dataset == "train_oof"
            else (
                "oof_tuned_frozen",
                "oof_selected_frozen",
                "oof_tuned",
            )
        )
        for policy in priorities:
            preferred = rows[
                rows["threshold_policy"].astype(str) == policy
            ]
            if not preferred.empty:
                return preferred.iloc[0]
    return rows.iloc[0]


def value(row: pd.Series | None, key: str) -> Any:
    if row is None or key not in row.index:
        return np.nan
    result = row[key]
    return np.nan if pd.isna(result) else result


def candidate_value(
    row: pd.Series | None,
    configuration: pd.Series | None,
) -> Any:
    candidate = value(row, "candidate_key")
    if not pd.isna(candidate):
        return candidate
    for key in ("winner", "selected_candidate", "candidate_key"):
        candidate = value(configuration, key)
        if not pd.isna(candidate):
            return candidate
    return np.nan


def model_size_value(
    row: pd.Series | None,
    configuration: pd.Series | None,
) -> Any:
    """Acepta los nombres usados por sklearn, Keras y late fusion."""

    keys = (
        "model_size_kb_joblib",
        "model_size_kb_float32_keras",
        "combined_model_size_kb",
        "model_size_kb",
    )
    for source in (row, configuration):
        for key in keys:
            result = value(source, key)
            if not pd.isna(result):
                return result
    return np.nan


def configuration_series(configuration: pd.DataFrame) -> pd.Series | None:
    """Normaliza configuraciones wide y parameter/value a una serie."""

    if configuration.empty:
        return None
    if {"parameter", "value"}.issubset(configuration.columns):
        return pd.Series(
            dict(
                zip(
                    configuration["parameter"].astype(str),
                    configuration["value"],
                )
            )
        )
    return configuration.iloc[0]


def build_summary() -> pd.DataFrame:
    rows = []
    for spec in EXPERIMENTS:
        metrics_path = absolute(spec.get("metrics"))
        train_row = None
        validation_row = None
        configuration_row = None
        status = "Finalizado"
        if metrics_path is not None and metrics_path.is_file():
            metrics = filter_metrics_for_spec(
                pd.read_csv(metrics_path), spec
            )
            train_row = first_dataset(metrics, "train_oof")
            validation_row = first_dataset(metrics, "validation")
            if validation_row is None:
                status = "Solo OOF"
        else:
            status = "En ejecución"

        config_path = absolute(spec.get("config"))
        if config_path is not None and config_path.is_file():
            configuration = pd.read_csv(config_path)
            configuration_row = configuration_series(configuration)

        # Mientras W05 sigue ejecutándose, incorporar la fase 1 OOF ya cerrada.
        projection_path = absolute(spec.get("projection"))
        if train_row is None and projection_path and projection_path.is_file():
            projection = pd.read_csv(projection_path)
            selected = projection[
                projection["selected_projection"].astype(str).str.lower()
                == "true"
            ]
            if selected.empty:
                selected = projection.sort_values(
                    "oof_tuned__macro_f1", ascending=False
                ).head(1)
            selected_row = selected.iloc[0]
            train_row = pd.Series(
                {
                    "candidate_key": selected_row.get("fixed_rf", ""),
                    "threshold": selected_row.get("threshold_oof", np.nan),
                    **{
                        metric: selected_row.get(
                            f"oof_tuned__{metric}", np.nan
                        )
                        for metric in METRIC_COLUMNS
                    },
                }
            )

        source = metrics_path if metrics_path and metrics_path.is_file() else projection_path
        rows.append(
            {
                "ID": spec["id"],
                "Estado": status,
                "Experimento": spec["name"],
                "Familia": spec["family"],
                "Unidad": spec["unit"],
                "Representación": spec["representation"],
                "Reducción/selección": spec["reduction"],
                "Balanceo": spec["balance"],
                "Clasificador": spec["classifier"],
                "Configuración ganadora": candidate_value(
                    validation_row if validation_row is not None else train_row,
                    configuration_row,
                ),
                "Umbral OOF": value(train_row, "threshold"),
                "Macro-F1 OOF": value(train_row, "macro_f1"),
                "Macro-F1 validation": value(validation_row, "macro_f1"),
                "Balanced acc validation": value(
                    validation_row, "balanced_accuracy"
                ),
                "Recall dry validation": value(validation_row, "dry_recall"),
                "Recall wet validation": value(validation_row, "wet_recall"),
                "Precision wet validation": value(
                    validation_row, "wet_precision"
                ),
                "AUC validation": value(validation_row, "roc_auc"),
                "AP wet validation": value(
                    validation_row, "average_precision_wet"
                ),
                "Modelo KB": model_size_value(
                    validation_row if validation_row is not None else train_row,
                    configuration_row,
                ),
                "Conclusión provisional": spec["note"],
                "Fuente": str(source.relative_to(SCRIPT_DIR)) if source and source.exists() else "",
            }
        )
    result = pd.DataFrame(rows)
    status_order = {"Finalizado": 0, "Solo OOF": 1, "En ejecución": 2}
    result["_status_order"] = result["Estado"].map(status_order).fillna(9)
    return (
        result.sort_values(
            ["_status_order", "Macro-F1 validation"],
            ascending=[True, False],
            na_position="last",
        )
        .drop(columns="_status_order")
        .reset_index(drop=True)
    )


def build_metric_detail() -> pd.DataFrame:
    frames = []
    for spec in EXPERIMENTS:
        path = absolute(spec.get("metrics"))
        if path is None or not path.is_file():
            continue
        data = pd.read_csv(path)
        data.insert(0, "experiment_name", spec["name"])
        data.insert(0, "experiment_id", spec["id"])
        data["source_file"] = str(path.relative_to(SCRIPT_DIR))
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_fold_detail() -> pd.DataFrame:
    frames = []
    for spec in EXPERIMENTS:
        path = absolute(spec.get("folds"))
        if path is None or not path.is_file():
            continue
        data = pd.read_csv(path)
        data.insert(0, "experiment_name", spec["name"])
        data.insert(0, "experiment_id", spec["id"])
        data["source_file"] = str(path.relative_to(SCRIPT_DIR))
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_candidate_detail() -> pd.DataFrame:
    """Recoge rejillas OOF declaradas por los experimentos."""
    frames = []
    for spec in EXPERIMENTS:
        path = absolute(spec.get("candidates"))
        if path is None or not path.is_file():
            continue
        data = pd.read_csv(path)
        data.insert(0, "experiment_name", spec["name"])
        data.insert(0, "experiment_id", spec["id"])
        data["source_file"] = str(path.relative_to(SCRIPT_DIR))
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_configurations() -> pd.DataFrame:
    rows = []
    for spec in EXPERIMENTS:
        for key in (
            "family",
            "unit",
            "representation",
            "reduction",
            "balance",
            "classifier",
        ):
            rows.append(
                {
                    "experiment_id": spec["id"],
                    "experiment_name": spec["name"],
                    "parameter": key,
                    "value": spec[key],
                    "source_file": "catalogo_del_generador",
                }
            )
        path = absolute(spec.get("config"))
        if path is None or not path.is_file():
            continue
        config = pd.read_csv(path)
        if config.empty:
            continue
        normalized = configuration_series(config)
        if normalized is None:
            continue
        for parameter, parameter_value in normalized.items():
            rows.append(
                {
                    "experiment_id": spec["id"],
                    "experiment_name": spec["name"],
                    "parameter": parameter,
                    "value": parameter_value,
                    "source_file": str(path.relative_to(SCRIPT_DIR)),
                }
            )
    return pd.DataFrame(rows)


def build_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    closed = summary[summary["Macro-F1 validation"].notna()].copy()
    closed["Ranking macro-F1"] = closed["Macro-F1 validation"].rank(
        method="min", ascending=False
    ).astype(int)
    closed["Ranking recall wet"] = closed["Recall wet validation"].rank(
        method="min", ascending=False
    ).astype(int)
    closed["Ranking AUC"] = closed["AUC validation"].rank(
        method="min", ascending=False
    ).astype(int)
    columns = [
        "Ranking macro-F1",
        "Ranking recall wet",
        "Ranking AUC",
        "ID",
        "Experimento",
        "Macro-F1 validation",
        "Balanced acc validation",
        "Recall dry validation",
        "Recall wet validation",
        "Precision wet validation",
        "AUC validation",
        "AP wet validation",
        "Modelo KB",
    ]
    return closed[columns].sort_values("Ranking macro-F1")


def build_notes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Protocolo", "Dry=0 y wet=1. TEST permanece reservado."),
            ("Protocolo", "Los hiperparámetros y umbrales se seleccionan con OOF de TRAIN."),
            ("Comparación", "Usar preferentemente filas validation y política oof_tuned_frozen cuando exista."),
            ("Duplicado control", "W02 conserva 644/644 features y reproduce W01; documenta que MI no compactó."),
            ("Referencia equilibrada", "W06 (WST+LR) mantiene el mejor compromiso actual: macro-F1 validation 0,6257 y recall wet 0,6049."),
            ("CNN móvil", "L01 alcanza macro-F1 validation 0,6216 con 5697 parámetros y 128,57 KB; recall wet 0,5309."),
            ("Fusión WST+CNN", "F05 maximiza macro-F1 validation hasta 0,6398, pero reduce recall wet a 0,4815 y su meta-CV es 0,6139; con restricción wet selecciona WST puro."),
            ("Modelo compacto", "W04 reduce mucho el RF, pero pierde macro-F1 de VALIDATION."),
            ("SMOTE", "W03 es un resultado negativo: no mejora la clase wet."),
            ("PCA+RF recording", "W05 finalizó: PCA fue descartado y no_pca obtuvo macro-F1 validation 0,5815."),
            ("Siguiente", "Comparar WST+LR y la CNN Log-Mel tras cuantización, incluyendo extracción de features, RAM y latencia móvil."),
            ("Fusión", "Las fusiones se alinean por original_uuid y se seleccionan con OOF/meta-CV; valorar siempre el coste conjunto y el recall wet."),
            ("Actualización", "Ejecutar: python .\\build_stage2_experiments_excel.py"),
        ],
        columns=["Categoría", "Detalle"],
    )


def build_wst_ablations() -> pd.DataFrame:
    """Resume las ablaciones WST sin mezclarlas con el ranking principal."""

    rows: list[dict[str, Any]] = []

    for temporal_label, relative_path in WST_TEMPORAL_ABLATIONS:
        path = absolute(relative_path)
        if path is None or not path.is_file():
            continue
        data = pd.read_csv(path)
        for _, result in data.iterrows():
            statistics = str(result["temporal_statistics"]).replace("|", "+")
            rows.append(
                {
                    "Estudio": "T y estadisticas temporales",
                    "Configuracion": f"{temporal_label}; {statistics}",
                    "T muestras": int(result["T_samples"]),
                    "T ms": float(result["T_seconds"]) * 1000.0,
                    "Ordenes WST": "S0+S1+S2",
                    "Estadisticas intraevento": statistics,
                    "Pooling entre eventos": "mean+std+max",
                    "Paths": int(result["path_count"]),
                    "Features por grabacion": int(
                        result["recording_feature_count"]
                    ),
                    "PCA": int(result["pca_components"]),
                    "C LR": float(result["C"]),
                    "Umbral OOF": float(result["threshold_oof"]),
                    "Macro-F1 OOF": float(result["oof__macro_f1"]),
                    "Balanced acc OOF": float(
                        result["oof__balanced_accuracy"]
                    ),
                    "Recall wet OOF": float(result["oof__wet_recall"]),
                    "AUC OOF": float(result["oof__roc_auc"]),
                    "Macro-F1 validation": float(
                        result["validation__macro_f1"]
                    ),
                    "Balanced acc validation": float(
                        result["validation__balanced_accuracy"]
                    ),
                    "Recall dry validation": float(
                        result["validation__dry_recall"]
                    ),
                    "Recall wet validation": float(
                        result["validation__wet_recall"]
                    ),
                    "AUC validation": float(result["validation__roc_auc"]),
                    "Fuente": str(path.relative_to(SCRIPT_DIR)),
                }
            )

    order_path = absolute(WST_ORDER_ABLATION)
    if order_path is not None and order_path.is_file():
        data = pd.read_csv(order_path)
        order_labels = {
            "s1": "S1",
            "s2": "S2",
            "s1_s2": "S1+S2",
            "s0_s1_s2": "S0+S1+S2",
        }
        for _, result in data.iterrows():
            representation = str(result["order_representation"])
            orders = order_labels.get(representation, representation)
            rows.append(
                {
                    "Estudio": "Ordenes de scattering",
                    "Configuracion": orders,
                    "T muestras": int(result["T_samples"]),
                    "T ms": float(result["T_samples"]) / 16.0,
                    "Ordenes WST": orders,
                    "Estadisticas intraevento": str(
                        result["within_event_temporal_pooling"]
                    ),
                    "Pooling entre eventos": str(
                        result["between_event_pooling"]
                    ).replace("|", "+"),
                    "Paths": int(result["path_count"]),
                    "Features por grabacion": int(
                        result["recording_feature_count"]
                    ),
                    "PCA": int(result["pca_components"]),
                    "C LR": float(result["C"]),
                    "Umbral OOF": float(result["threshold_oof"]),
                    "Macro-F1 OOF": float(result["oof__macro_f1"]),
                    "Balanced acc OOF": float(
                        result["oof__balanced_accuracy"]
                    ),
                    "Recall wet OOF": float(result["oof__wet_recall"]),
                    "AUC OOF": float(result["oof__roc_auc"]),
                    "Macro-F1 validation": float(
                        result["validation__macro_f1"]
                    ),
                    "Balanced acc validation": float(
                        result["validation__balanced_accuracy"]
                    ),
                    "Recall dry validation": float(
                        result["validation__dry_recall"]
                    ),
                    "Recall wet validation": float(
                        result["validation__wet_recall"]
                    ),
                    "AUC validation": float(result["validation__roc_auc"]),
                    "Fuente": str(order_path.relative_to(SCRIPT_DIR)),
                }
            )

    ablations = pd.DataFrame(rows)
    if ablations.empty:
        return ablations

    best_oof = ablations.groupby("Estudio")["Macro-F1 OOF"].transform("max")
    ablations.insert(
        2,
        "Ganador segun OOF",
        np.isclose(ablations["Macro-F1 OOF"], best_oof),
    )
    return ablations.sort_values(
        ["Estudio", "Macro-F1 OOF"], ascending=[True, False]
    ).reset_index(drop=True)


def projection_sheet() -> pd.DataFrame:
    spec = next(item for item in EXPERIMENTS if item["id"] == "W05")
    path = absolute(spec.get("projection"))
    if path is None or not path.is_file():
        return pd.DataFrame(
            [{"Estado": "W05 todavía no ha guardado projection_cv_comparison.csv"}]
        )
    data = pd.read_csv(path)
    data.insert(0, "experiment_id", "W05")
    data["source_file"] = str(path.relative_to(SCRIPT_DIR))
    return data


def safe_excel_value(value_to_write: Any) -> Any:
    if pd.isna(value_to_write):
        return None
    if isinstance(value_to_write, np.generic):
        return value_to_write.item()
    return value_to_write


def add_dataframe_sheet(writer: pd.ExcelWriter, name: str, df: pd.DataFrame) -> None:
    if df.empty:
        df = pd.DataFrame([{"Estado": "Sin datos disponibles"}])
    clean = df.map(safe_excel_value)
    clean.to_excel(writer, sheet_name=name, index=False)


def style_workbook(path: Path) -> None:
    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    status_colors = {
        "Finalizado": "C6E0B4",
        "Solo OOF": "FFE699",
        "En ejecución": "F4B183",
    }

    for sheet_index, worksheet in enumerate(workbook.worksheets, start=1):
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        worksheet.row_dimensions[1].height = 32

        if worksheet.max_row >= 2 and worksheet.max_column >= 1:
            table_name = f"Tabla{sheet_index}_{''.join(c for c in worksheet.title if c.isalnum())[:18]}"
            table = Table(displayName=table_name, ref=worksheet.dimensions)
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2",
                showFirstColumn=False,
                showLastColumn=False,
                showRowStripes=True,
                showColumnStripes=False,
            )
            worksheet.add_table(table)

        headers = {cell.value: cell.column for cell in worksheet[1]}
        for column_index in range(1, worksheet.max_column + 1):
            values = [
                str(worksheet.cell(row, column_index).value or "")
                for row in range(1, min(worksheet.max_row, 250) + 1)
            ]
            width = min(max(max(map(len, values)) + 2, 10), 42)
            worksheet.column_dimensions[get_column_letter(column_index)].width = width
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

        for header, column_index in headers.items():
            header_text = str(header or "").lower()
            if any(
                token in header_text
                for token in (
                    "macro_f1",
                    "macro-f1",
                    "accuracy",
                    "recall",
                    "precision",
                    "roc_auc",
                    "auc ",
                    "ap wet",
                    "threshold",
                    "umbral",
                    "variance_ratio",
                )
            ):
                for row_index in range(2, worksheet.max_row + 1):
                    worksheet.cell(row_index, column_index).number_format = "0.0000"

        if "Estado" in headers:
            column_index = headers["Estado"]
            for row_index in range(2, worksheet.max_row + 1):
                cell = worksheet.cell(row_index, column_index)
                color = status_colors.get(str(cell.value))
                if color:
                    cell.fill = PatternFill("solid", fgColor=color)

        metric_headers = [
            header
            for header in headers
            if any(token in str(header).lower() for token in ("macro-f1 validation", "recall wet validation", "auc validation"))
        ]
        for header in metric_headers:
            column_index = headers[header]
            column_letter = get_column_letter(column_index)
            worksheet.conditional_formatting.add(
                f"{column_letter}2:{column_letter}{worksheet.max_row}",
                ColorScaleRule(
                    start_type="min",
                    start_color="F8696B",
                    mid_type="percentile",
                    mid_value=50,
                    mid_color="FFEB84",
                    end_type="max",
                    end_color="63BE7B",
                ),
            )

    summary_sheet = workbook["Resumen"]
    headers = {cell.value: cell.column for cell in summary_sheet[1]}
    if "Macro-F1 validation" in headers and summary_sheet.max_row >= 3:
        chart = BarChart()
        chart.type = "bar"
        chart.style = 10
        chart.title = "Macro-F1 en VALIDATION"
        chart.y_axis.title = "Experimento"
        chart.x_axis.title = "Macro-F1"
        data = Reference(
            summary_sheet,
            min_col=headers["Macro-F1 validation"],
            min_row=1,
            max_row=summary_sheet.max_row,
        )
        categories = Reference(
            summary_sheet,
            min_col=headers["ID"],
            min_row=2,
            max_row=summary_sheet.max_row,
        )
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(categories)
        chart.height = 7
        chart.width = 13
        summary_sheet.add_chart(chart, "X2")

    workbook.save(path)


def main() -> None:
    summary = build_summary()
    detail = build_metric_detail()
    folds = build_fold_detail()
    candidates = build_candidate_detail()
    configurations = build_configurations()
    ranking = build_ranking(summary)
    notes = build_notes()
    pca_current = projection_sheet()
    wst_ablations = build_wst_ablations()

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        add_dataframe_sheet(writer, "Resumen", summary)
        add_dataframe_sheet(writer, "Ranking_validation", ranking)
        add_dataframe_sheet(writer, "Metricas_detalle", detail)
        add_dataframe_sheet(writer, "CV_por_fold", folds)
        add_dataframe_sheet(writer, "Candidatos_OOF", candidates)
        add_dataframe_sheet(writer, "PCA_RF_actual", pca_current)
        add_dataframe_sheet(writer, "Ablaciones_WST", wst_ablations)
        add_dataframe_sheet(writer, "Configuraciones", configurations)
        add_dataframe_sheet(writer, "Notas_y_protocolo", notes)
    style_workbook(OUTPUT_PATH)

    finalized = int((summary["Estado"] == "Finalizado").sum())
    running = int((summary["Estado"] == "En ejecución").sum())
    print(f"Excel guardado en: {OUTPUT_PATH}")
    print(f"Experimentos finalizados: {finalized}")
    print(f"Experimentos en ejecucion: {running}")
    print(f"Filas de metricas detalladas: {len(detail)}")
    print(f"Filas de candidatos OOF: {len(candidates)}")
    print(f"Filas de ablaciones WST: {len(wst_ablations)}")


if __name__ == "__main__":
    main()
