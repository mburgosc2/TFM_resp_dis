"""Construye tablas compactas y trazables con los experimentos clave del TFM.

El libro separa los protocolos que no deben compararse como si fueran el mismo:

* TEST interno de Stage 1.
* Validacion piloto externa de Stage 1.
* VALIDATION de Stage 2 (usada con umbral seleccionado exclusivamente en OOF).
* Unico TEST final disponible de Stage 2.

Los valores se leen de los CSV/Excel producidos por los scripts de entrenamiento;
no se copian manualmente en este generador.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


ROOT = Path(__file__).resolve().parent
DEMO_ROOT = ROOT.parent / "TFM_cough_detection_demo"
OUTPUT_DIR = ROOT / "tables_tfm_key_experiments"
OUTPUT_XLSX = ROOT / "tfm_key_experiments_comparison.xlsx"
STAGE2_CATALOG = ROOT / "stage2_dry_wet_experiments_summary.xlsx"


STAGE1_SPECS = (
    {
        "id": "S1-LR-01",
        "name": "MFCC117 + LR; CoughVID como clase positiva",
        "positive_data": "CoughVID",
        "augmentation": "Ninguno",
        "protocol": "Split aleatorio agrupado",
        "comparison_group": "A: referencia anterior; distinto conjunto positivo",
        "metrics": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_random"
        / "full"
        / "metrics_summary.csv",
        "config": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_random"
        / "full"
        / "experiment_configuration.csv",
        "source_recalls": None,
        "coughvid_is_total_recall": True,
    },
    {
        "id": "S1-LR-02",
        "name": "MFCC117 + LR; CoughVID + FSD50K",
        "positive_data": "CoughVID + toses FSD50K revisadas y segmentadas",
        "augmentation": "Ninguno",
        "protocol": "Split aleatorio agrupado",
        "comparison_group": "B: comparación principal; mismo TEST que S1-LR-03/04",
        "metrics": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_random"
        / "full"
        / "metrics_summary.csv",
        "config": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_random"
        / "full"
        / "experiment_configuration.csv",
        "source_recalls": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_random"
        / "full"
        / "test_cough_recall_by_source.csv",
    },
    {
        "id": "S1-LR-03",
        "name": "MFCC117 + LR; augmentation pitch + ruido",
        "positive_data": "CoughVID + toses FSD50K revisadas y segmentadas",
        "augmentation": "2 variantes/tos TRAIN: pitch ±1.5 o ruido moderado/alto",
        "protocol": "Split aleatorio agrupado; OOF/VAL/TEST solo originales",
        "comparison_group": "B: comparación principal; mismo TEST que S1-LR-02/04",
        "metrics": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_audio_aug_random"
        / "full"
        / "metrics_summary.csv",
        "config": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_audio_aug_random"
        / "full"
        / "experiment_configuration.csv",
        "source_recalls": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_audio_aug_random"
        / "full"
        / "test_cough_recall_by_source.csv",
    },
    {
        "id": "S1-LR-04",
        "name": "MFCC117 + LR; augmentation solo ruido",
        "positive_data": "CoughVID + toses FSD50K revisadas y segmentadas",
        "augmentation": "1 variante/tos TRAIN: mezcla a 10 o 20 dB SNR",
        "protocol": "Split aleatorio agrupado; OOF/VAL/TEST solo originales",
        "comparison_group": "B: comparación principal; mismo TEST que S1-LR-02/03",
        "metrics": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_noise_aug_random"
        / "full"
        / "metrics_summary.csv",
        "config": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_noise_aug_random"
        / "full"
        / "experiment_configuration.csv",
        "source_recalls": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_noise_aug_random"
        / "full"
        / "test_cough_recall_by_source.csv",
    },
    {
        "id": "S1-LR-05",
        "name": "MFCC117 + LR; stress test de calidad poor",
        "positive_data": "TRAIN/VAL: CoughVID + FSD50K; TEST: CoughVID poor",
        "augmentation": "Ninguno",
        "protocol": "Todas las toses CoughVID poor reservadas en TEST",
        "comparison_group": "C: análisis secundario de robustez; TEST diferente",
        "metrics": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_audio_quality_poor"
        / "full"
        / "metrics_summary.csv",
        "config": ROOT
        / "results_stage1_cough_no_cough"
        / "mfcc117_lr_fsd50k_cough_segments_audio_quality_poor"
        / "full"
        / "experiment_configuration.csv",
        "source_recalls": None,
        "coughvid_is_total_recall": True,
    },
)


EXTERNAL_SPECS = (
    {
        "id": "EXT-01",
        "name": "Sin augmentation",
        "directory": "stage1_mfcc117_lr_fsd50k_cough_segments",
    },
    {
        "id": "EXT-02",
        "name": "Pitch + ruido",
        "directory": "stage1_mfcc117_lr_fsd50k_cough_segments_audio_aug",
    },
    {
        "id": "EXT-03",
        "name": "Solo ruido",
        "directory": "stage1_mfcc117_lr_fsd50k_cough_segments_noise_aug",
    },
)


STAGE2_IDS = (
    "M01",  # MFCC + LR
    "C05",  # cocleograma + XGBoost
    "L01",  # Log-Mel + tiny CNN
    "W01",  # WST por evento + RF
    "W08",  # WST recording + SVM RBF
    "W09",  # WST recording + MLP
    "W11",  # WST temporal + CNN MIL
    "W13",  # WST LR + augmentation gold
    "W14",  # WST mejorado + LR
    "W15",  # WST + LR, búsqueda en dos fases
    "F05",  # late fusion WST + Log-Mel
)


STAGE2_REASON = {
    "M01": "Baseline MFCC lineal.",
    "C05": "Mejor resultado representativo con cocleograma.",
    "L01": "Alternativa deep learning muy compacta con Log-Mel.",
    "W01": "Primer baseline WST con Random Forest.",
    "W08": "Comparación WST con SVM no lineal.",
    "W09": "Comparación WST con MLP ligera.",
    "W11": "Conserva temporalidad y entrena a nivel recording mediante MIL.",
    "W13": "Ablación de data augmentation y pesos de consenso.",
    "W14": "Ablación de pooling ponderado, órdenes WST y amplitud.",
    "W15": "Modelo WST-LR de referencia y único evaluado finalmente en TEST.",
    "F05": "Mayor macro-F1 en VALIDATION mediante late fusion.",
}


def read_parameter_value(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = pd.read_csv(path)
    if not {"parameter", "value"}.issubset(data.columns):
        return {}
    return dict(zip(data["parameter"].astype(str), data["value"]))


def metric_row(data: pd.DataFrame, dataset: str) -> pd.Series:
    selected = data[data["dataset"].astype(str) == dataset]
    if selected.empty:
        raise ValueError(f"No existe la fila dataset={dataset!r}")
    return selected.iloc[0]


def source_recalls(path: Path | None) -> tuple[float, float]:
    if path is None or not path.is_file():
        return np.nan, np.nan
    data = pd.read_csv(path)
    mapping = dict(zip(data["cough_source"].astype(str), data["recall_cough"]))
    return (
        float(mapping.get("original_cough", np.nan)),
        float(mapping.get("new_fsd50k_cough", np.nan)),
    )


def build_stage1_internal() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in STAGE1_SPECS:
        metrics_path = Path(spec["metrics"])
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        metrics = pd.read_csv(metrics_path)
        oof = metric_row(metrics, "train_oof")
        validation = metric_row(metrics, "validation")
        test = metric_row(metrics, "test")
        configuration = read_parameter_value(Path(spec["config"]))
        coughvid_recall, fsd50k_recall = source_recalls(spec.get("source_recalls"))
        if spec.get("coughvid_is_total_recall", False):
            coughvid_recall = float(test["recall_cough"])
        validation_source_path = metrics_path.with_name(
            "validation_cough_recall_by_source.csv"
        )
        validation_coughvid_recall, validation_fsd50k_recall = source_recalls(
            validation_source_path
        )
        if not validation_source_path.is_file():
            validation_coughvid_recall = float(validation["recall_cough"])

        rows.append(
            {
                "ID": spec["id"],
                "Experimento": spec["name"],
                "Datos positivos": spec["positive_data"],
                "Augmentation en TRAIN": spec["augmentation"],
                "Protocolo": spec["protocol"],
                "Grupo de comparabilidad": spec["comparison_group"],
                "Configuración LR": configuration.get(
                    "winner", "lr__C0p3__l1__weight_none"
                ),
                "Umbral": float(test["threshold"]),
                "N VALIDATION": int(validation["n_samples"]),
                "No tos VALIDATION": int(validation["n_no_cough"]),
                "Tos VALIDATION": int(validation["n_cough"]),
                "Accuracy VALIDATION": float(validation["accuracy"]),
                "Balanced accuracy VALIDATION": float(
                    validation["balanced_accuracy"]
                ),
                "Sensibilidad / recall tos VALIDATION": float(
                    validation["recall_cough"]
                ),
                "Especificidad / recall no tos VALIDATION": float(
                    validation["specificity_no_cough"]
                ),
                "Precision tos VALIDATION": float(
                    validation["precision_cough"]
                ),
                "F1 tos VALIDATION": float(validation["f1_cough"]),
                "Macro-F1 VALIDATION": float(validation["macro_f1"]),
                "ROC-AUC VALIDATION": float(validation["roc_auc"]),
                "Recall toses CoughVID VALIDATION": (
                    validation_coughvid_recall
                ),
                "Recall toses FSD50K VALIDATION": validation_fsd50k_recall,
                "TN VALIDATION": int(validation["tn_no_cough_correct"]),
                "FP VALIDATION": int(
                    validation["fp_no_cough_as_cough"]
                ),
                "FN VALIDATION": int(
                    validation["fn_cough_as_no_cough"]
                ),
                "TP VALIDATION": int(validation["tp_cough_correct"]),
                "N TEST": int(test["n_samples"]),
                "No tos TEST": int(test["n_no_cough"]),
                "Tos TEST": int(test["n_cough"]),
                "Accuracy TEST": float(test["accuracy"]),
                "Balanced accuracy TEST": float(test["balanced_accuracy"]),
                "Sensibilidad / recall tos TEST": float(test["recall_cough"]),
                "Especificidad / recall no tos TEST": float(
                    test["specificity_no_cough"]
                ),
                "Precisión tos TEST": float(test["precision_cough"]),
                "F1 tos TEST": float(test["f1_cough"]),
                "Macro-F1 TEST": float(test["macro_f1"]),
                "ROC-AUC TEST": float(test["roc_auc"]),
                "Recall toses CoughVID TEST": coughvid_recall,
                "Recall toses FSD50K TEST": fsd50k_recall,
                "FP TEST": int(test["fp_no_cough_as_cough"]),
                "FN TEST": int(test["fn_cough_as_no_cough"]),
                "Macro-F1 OOF": float(oof["macro_f1"]),
                "Modelo KB": float(configuration.get("model_size_kb", np.nan)),
                "Fuente": str(metrics_path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def external_metric_mapping(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    return {
        str(metric): float(value)
        for metric, value in zip(data["metric"], data["value"])
    }


def external_cohort_signature(path: Path) -> tuple[tuple[str, ...], ...]:
    """Identifica la cohorte externa sin depender de las predicciones."""

    if not path.is_file():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, dtype=str).fillna("")
    columns = [
        "sample_id",
        "subject_id",
        "true_label",
        "sound_type",
        "device",
        "declared_file_name",
    ]
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en {path}: {', '.join(sorted(missing))}"
        )
    if data["sample_id"].duplicated().any():
        duplicates = sorted(
            data.loc[data["sample_id"].duplicated(), "sample_id"].unique()
        )
        raise ValueError(
            f"Hay sample_id duplicados en {path}: {duplicates[:10]}"
        )
    ordered = data.sort_values("sample_id", kind="stable")
    return tuple(
        tuple(str(value) for value in row)
        for row in ordered[columns].itertuples(index=False, name=None)
    )


def build_stage1_validation(stage1_internal: pd.DataFrame) -> pd.DataFrame:
    """Tabla compacta dedicada exclusivamente a VALIDATION de Stage 1."""

    target_configuration_column = "Configuraci\u00f3n LR"
    if target_configuration_column not in stage1_internal.columns:
        candidates = [
            column
            for column in stage1_internal.columns
            if column.startswith("Configuraci") and column.endswith(" LR")
        ]
        if len(candidates) != 1:
            raise ValueError(
                "No se pudo identificar de forma unica la columna de "
                "configuracion LR."
            )
        stage1_internal = stage1_internal.rename(
            columns={candidates[0]: target_configuration_column}
        )

    columns = [
        "ID",
        "Experimento",
        "Datos positivos",
        "Augmentation en TRAIN",
        "Protocolo",
        "Grupo de comparabilidad",
        "ConfiguraciÃ³n LR",
        "Umbral",
        "N VALIDATION",
        "No tos VALIDATION",
        "Tos VALIDATION",
        "Accuracy VALIDATION",
        "Balanced accuracy VALIDATION",
        "Sensibilidad / recall tos VALIDATION",
        "Especificidad / recall no tos VALIDATION",
        "Precision tos VALIDATION",
        "F1 tos VALIDATION",
        "Macro-F1 VALIDATION",
        "ROC-AUC VALIDATION",
        "Recall toses CoughVID VALIDATION",
        "Recall toses FSD50K VALIDATION",
        "TN VALIDATION",
        "FP VALIDATION",
        "FN VALIDATION",
        "TP VALIDATION",
        "Macro-F1 OOF",
        "Modelo KB",
        "Fuente",
    ]
    columns[6] = target_configuration_column
    missing = set(columns) - set(stage1_internal.columns)
    if missing:
        raise ValueError(
            "Faltan columnas para la tabla Stage 1 VALIDATION: "
            + ", ".join(sorted(missing))
        )
    return stage1_internal[columns].copy()


def build_stage1_external() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = DEMO_ROOT / "results_external_validation"
    reference_signature: tuple[tuple[str, ...], ...] | None = None
    reference_name = ""
    for spec in EXTERNAL_SPECS:
        result_dir = base / spec["directory"]
        path = result_dir / "external_validation_metrics.csv"
        predictions_path = result_dir / "external_validation_predictions.csv"
        configuration_path = result_dir / "external_validation_configuration.csv"
        m = external_metric_mapping(path)
        signature = external_cohort_signature(predictions_path)
        if int(m["n_samples"]) != len(signature):
            raise ValueError(
                f"{spec['name']}: las metricas indican {int(m['n_samples'])} "
                f"muestras, pero hay {len(signature)} predicciones."
            )
        if reference_signature is None:
            reference_signature = signature
            reference_name = str(spec["name"])
        elif signature != reference_signature:
            raise ValueError(
                "Las evaluaciones externas no contienen exactamente la misma "
                f"cohorte: {reference_name!r} y {spec['name']!r}. Ejecuta de "
                "nuevo los tres modelos con el mismo CSV de metadatos."
            )
        configuration = read_parameter_value(configuration_path)
        rows.append(
            {
                "ID": spec["id"],
                "Modelo": spec["name"],
                "Protocolo": (
                    "Evaluacion externa exploratoria post hoc; mismas "
                    f"{int(m['n_samples'])} muestras; sin ajuste de modelo "
                    "ni umbral"
                ),
                "Fecha evaluacion UTC": configuration.get("created_utc", ""),
                "SHA-256 metadatos": configuration.get("metadata_sha256", ""),
                "Umbral": 0.5,
                "N": int(m["n_samples"]),
                "No tos": int(m["n_no_cough"]),
                "Tos": int(m["n_cough"]),
                "Accuracy": m["accuracy"],
                "Balanced accuracy": m["balanced_accuracy"],
                "Sensibilidad / recall tos": m["recall_cough_sensitivity"],
                "Especificidad / recall no tos": m[
                    "recall_no_cough_specificity"
                ],
                "Precisión tos": m["precision_cough"],
                "F1 tos": m["f1_cough"],
                "Macro-F1": m["macro_f1"],
                "ROC-AUC": m["roc_auc"],
                "Average precision": m["average_precision_pr_auc"],
                "TN": int(m["true_negative"]),
                "FP": int(m["false_positive"]),
                "FN": int(m["false_negative"]),
                "TP": int(m["true_positive"]),
                "Fuente": str(path.relative_to(DEMO_ROOT)),
            }
        )
    return pd.DataFrame(rows)


def harmonic_f1(precision: pd.Series, recall: pd.Series) -> pd.Series:
    denominator = precision + recall
    return pd.Series(
        np.where(denominator > 0, 2.0 * precision * recall / denominator, 0.0),
        index=precision.index,
    )


def build_stage2_validation() -> pd.DataFrame:
    if not STAGE2_CATALOG.is_file():
        raise FileNotFoundError(STAGE2_CATALOG)
    catalog = pd.read_excel(STAGE2_CATALOG, sheet_name="Resumen")
    selected = catalog[catalog["ID"].isin(STAGE2_IDS)].copy()
    missing = set(STAGE2_IDS) - set(selected["ID"])
    if missing:
        raise ValueError(f"Faltan experimentos Stage 2 en el catálogo: {sorted(missing)}")

    # VALIDATION es común: 268 grabaciones dry y 81 wet.
    selected["N VALIDATION"] = 349
    selected["Dry VALIDATION"] = 268
    selected["Wet VALIDATION"] = 81
    selected["Accuracy VALIDATION"] = (
        268.0 * selected["Recall dry validation"]
        + 81.0 * selected["Recall wet validation"]
    ) / 349.0
    selected["F1 wet VALIDATION"] = harmonic_f1(
        selected["Precision wet validation"], selected["Recall wet validation"]
    )
    selected["Motivo de inclusión"] = selected["ID"].map(STAGE2_REASON)

    result = pd.DataFrame(
        {
            "ID": selected["ID"],
            "Experimento": selected["Experimento"],
            "Familia": selected["Familia"],
            "Unidad de clasificación": selected["Unidad"],
            "Clasificador": selected["Clasificador"],
            "Reducción/selección": selected["Reducción/selección"],
            "Configuración ganadora": selected["Configuración ganadora"],
            "Umbral OOF congelado": selected["Umbral OOF"],
            "N VALIDATION": selected["N VALIDATION"],
            "Dry VALIDATION": selected["Dry VALIDATION"],
            "Wet VALIDATION": selected["Wet VALIDATION"],
            "Accuracy VALIDATION": selected["Accuracy VALIDATION"],
            "Balanced accuracy VALIDATION": selected[
                "Balanced acc validation"
            ],
            "Sensibilidad / recall wet VALIDATION": selected[
                "Recall wet validation"
            ],
            "Especificidad / recall dry VALIDATION": selected[
                "Recall dry validation"
            ],
            "Precisión wet VALIDATION": selected["Precision wet validation"],
            "F1 wet VALIDATION": selected["F1 wet VALIDATION"],
            "Macro-F1 VALIDATION": selected["Macro-F1 validation"],
            "ROC-AUC VALIDATION": selected["AUC validation"],
            "Average precision wet VALIDATION": selected["AP wet validation"],
            "Macro-F1 OOF": selected["Macro-F1 OOF"],
            "Modelo KB": selected["Modelo KB"],
            "Motivo de inclusión": selected["Motivo de inclusión"],
            "Fuente": selected["Fuente"],
        }
    )
    return result.sort_values("Macro-F1 VALIDATION", ascending=False).reset_index(
        drop=True
    )


def build_stage2_test() -> pd.DataFrame:
    comparison_path = (
        ROOT
        / "results_stage2_late_fusion_wst_logmel_cnn"
        / "full"
        / "stage2_test_model_comparison.csv"
    )
    if comparison_path.is_file():
        data = pd.read_csv(comparison_path)
        catalog = pd.read_excel(STAGE2_CATALOG, sheet_name="Resumen")
        configurations = dict(
            zip(catalog["ID"].astype(str), catalog["Configuración ganadora"])
        )
        wet_denominator = data["wet_precision"] + data["wet_recall"]
        wet_f1 = np.where(
            wet_denominator > 0,
            2.0 * data["wet_precision"] * data["wet_recall"] / wet_denominator,
            0.0,
        )
        result = pd.DataFrame(
            {
                "ID": data["ID"],
                "Experimento": data["experiment"],
                "Protocolo": (
                    "Mismo TEST final; modelos, configuración, fusión y umbrales "
                    "congelados mediante OOF antes de evaluarlo"
                ),
                "Configuración": data["ID"].map(configurations),
                "Umbral": data["threshold"],
                "N TEST": data["n_test"],
                "Dry TEST": data["n_dry"],
                "Wet TEST": data["n_wet"],
                "Accuracy TEST": data["accuracy"],
                "Balanced accuracy TEST": data["balanced_accuracy"],
                "Sensibilidad / recall wet TEST": data["wet_recall"],
                "Especificidad / recall dry TEST": data["dry_recall"],
                "Precisión wet TEST": data["wet_precision"],
                "F1 wet TEST": wet_f1,
                "Macro-F1 TEST": data["macro_f1"],
                "ROC-AUC TEST": data["roc_auc"],
                "Average precision wet TEST": data["average_precision_wet"],
                "TN dry": data["tn_dry_correct"],
                "FP dry→wet": data["fp_dry_as_wet"],
                "FN wet→dry": data["fn_wet_as_dry"],
                "TP wet": data["tp_wet_correct"],
                "Modelo KB": data["model_size_kb"],
                "Fuente": str(comparison_path.relative_to(ROOT)),
            }
        )
        return result.sort_values("Macro-F1 TEST", ascending=False).reset_index(
            drop=True
        )

    path = (
        ROOT
        / "results_stage2_dry_wet_wst_recording_lr_two_phase_search"
        / "paper_q8_q1_t500_full"
        / "full"
        / "metrics_summary.csv"
    )
    data = pd.read_csv(path)
    row = metric_row(data, "test")
    return pd.DataFrame(
        [
            {
                "ID": "W15",
                "Experimento": "WST recording + LR; búsqueda en dos fases",
                "Protocolo": (
                    "TEST final tras congelar configuración y umbral mediante OOF; "
                    "no usado para selección"
                ),
                "Configuración": row["candidate_key"],
                "Umbral": float(row["threshold"]),
                "N TEST": int(
                    row["tn_dry_correct"]
                    + row["fp_dry_as_wet"]
                    + row["fn_wet_as_dry"]
                    + row["tp_wet_correct"]
                ),
                "Dry TEST": int(row["tn_dry_correct"] + row["fp_dry_as_wet"]),
                "Wet TEST": int(row["fn_wet_as_dry"] + row["tp_wet_correct"]),
                "Accuracy TEST": float(row["accuracy"]),
                "Balanced accuracy TEST": float(row["balanced_accuracy"]),
                "Sensibilidad / recall wet TEST": float(row["wet_recall"]),
                "Especificidad / recall dry TEST": float(row["dry_recall"]),
                "Precisión wet TEST": float(row["wet_precision"]),
                "F1 wet TEST": float(
                    2
                    * row["wet_precision"]
                    * row["wet_recall"]
                    / (row["wet_precision"] + row["wet_recall"])
                ),
                "Macro-F1 TEST": float(row["macro_f1"]),
                "ROC-AUC TEST": float(row["roc_auc"]),
                "Average precision wet TEST": float(row["average_precision_wet"]),
                "TN dry": int(row["tn_dry_correct"]),
                "FP dry→wet": int(row["fp_dry_as_wet"]),
                "FN wet→dry": int(row["fn_wet_as_dry"]),
                "TP wet": int(row["tp_wet_correct"]),
                "Modelo KB": float(row["model_size_kb_joblib"]),
                "Fuente": str(path.relative_to(ROOT)),
            }
        ]
    )


def build_takeaways() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Sección": "Stage 1 — comparación interna principal",
                "Lectura": (
                    "Sin augmentation ofrece el mejor equilibrio global del grupo B "
                    "en TEST (macro-F1 0.9594, especificidad 0.9778)."
                ),
            },
            {
                "Sección": "Stage 1 — augmentation",
                "Lectura": (
                    "Pitch+ruido maximiza sensibilidad (0.9813) y recall de toses "
                    "FSD50K (0.6800), pero aumenta los falsos positivos y reduce "
                    "macro-F1 y especificidad. Solo ruido queda como compromiso intermedio."
                ),
            },
            {
                "Sección": "Stage 1 — calidad poor",
                "Lectura": (
                    "El stress test obtiene sensibilidad 0.9370 y especificidad 0.9630; "
                    "apoya robustez frente a peor calidad, pero usa un TEST distinto."
                ),
            },
            {
                "Sección": "Stage 1 — validación piloto externa",
                "Lectura": (
                    "Sin augmentation conserva la mejor balanced accuracy y ROC-AUC; "
                    "la augmentation incrementa sensibilidad a costa de especificidad. "
                    "La muestra es pequeña y exploratoria."
                ),
            },
            {
                "Sección": "Stage 2 — VALIDATION",
                "Lectura": (
                    "Late fusion alcanza el mayor macro-F1 (0.6398), pero WST+LR "
                    "en dos fases ofrece la mayor sensibilidad wet (0.6049) y balanced "
                    "accuracy (0.6588) entre los experimentos seleccionados."
                ),
            },
            {
                "Sección": "Stage 2 — TEST",
                "Lectura": (
                    "Con los cuatro modelos congelados sobre el mismo TEST, la late "
                    "fusion obtiene el mayor macro-F1 (0.6751), balanced accuracy "
                    "(0.6729) y ROC-AUC (0.7220). Log-Mel alcanza el mayor recall wet "
                    "(0.5500). WST-LR mejora modestamente al baseline MFCC-LR."
                ),
            },
        ]
    )


def build_metric_notes(external_n: int | None = None) -> pd.DataFrame:
    external_sample_text = (
        f"{external_n} muestras" if external_n is not None else "una muestra reducida"
    )
    return pd.DataFrame(
        [
            (
                "Sensibilidad / recall positivo",
                "Son la misma métrica: TP/(TP+FN). En Stage 1 la clase positiva es tos; en Stage 2 es wet.",
            ),
            (
                "Especificidad / recall negativo",
                "Son equivalentes en binario: TN/(TN+FP). En Stage 1 corresponde a no tos y en Stage 2 a dry.",
            ),
            (
                "Balanced accuracy",
                "Media de sensibilidad y especificidad; resulta más informativa que accuracy con clases desbalanceadas.",
            ),
            (
                "Macro-F1",
                "Media del F1 de ambas clases con el mismo peso; métrica principal para valorar equilibrio global.",
            ),
            (
                "ROC-AUC",
                "Capacidad de ordenar positivos por encima de negativos sin fijar un único umbral.",
            ),
            (
                "Average precision",
                "Resumen de la curva precisión-recall; especialmente útil para la clase wet minoritaria.",
            ),
            (
                "Comparabilidad Stage 1",
                "Solo S1-LR-02/03/04 comparten exactamente el mismo TEST. S1-LR-01 y S1-LR-05 responden a preguntas distintas.",
            ),
            (
                "Comparabilidad Stage 2",
                "Las filas de la hoja Stage2_VALIDATION comparten las mismas 349 grabaciones. El TEST final se mantiene separado.",
            ),
            (
                "Validación externa",
                f"Evaluacion exploratoria con {external_sample_text}; no debe "
                "presentarse como estimacion clinica ni usarse para reajustar "
                "el modelo retrospectivamente.",
            ),
        ],
        columns=["Concepto", "Interpretación"],
    )


def safe_excel_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def add_sheet(writer: pd.ExcelWriter, name: str, data: pd.DataFrame) -> None:
    data.map(safe_excel_value).to_excel(writer, sheet_name=name, index=False)


def style_workbook(path: Path) -> None:
    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)

    for index, worksheet in enumerate(workbook.worksheets, start=1):
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.row_dimensions[1].height = 42
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(
                horizontal="center", vertical="center", wrap_text=True
            )
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

        for column_index in range(1, worksheet.max_column + 1):
            values = [
                str(worksheet.cell(row, column_index).value or "")
                for row in range(1, min(worksheet.max_row, 100) + 1)
            ]
            width = min(max(max(map(len, values), default=8) + 2, 11), 38)
            worksheet.column_dimensions[get_column_letter(column_index)].width = width

        headers = {cell.value: cell.column for cell in worksheet[1]}
        metric_columns = []
        for header, column_index in headers.items():
            text = str(header or "").lower()
            if any(
                token in text
                for token in (
                    "accuracy",
                    "sensibilidad",
                    "especificidad",
                    "precisión",
                    "recall",
                    "f1",
                    "auc",
                    "average precision",
                    "umbral",
                )
            ):
                metric_columns.append(column_index)
                for row_index in range(2, worksheet.max_row + 1):
                    worksheet.cell(row_index, column_index).number_format = "0.0000"
        for column_index in metric_columns:
            if worksheet.max_row < 3:
                continue
            letter = get_column_letter(column_index)
            worksheet.conditional_formatting.add(
                f"{letter}2:{letter}{worksheet.max_row}",
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

        if worksheet.max_row >= 2 and worksheet.max_column >= 1:
            table = Table(
                displayName=f"TFMTable{index}",
                ref=f"A1:{get_column_letter(worksheet.max_column)}{worksheet.max_row}",
            )
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2",
                showFirstColumn=False,
                showLastColumn=False,
                showRowStripes=True,
                showColumnStripes=False,
            )
            worksheet.add_table(table)

    workbook.save(path)


def main() -> None:
    stage1_internal = build_stage1_internal()
    stage1_validation = build_stage1_validation(stage1_internal)
    stage1_external = build_stage1_external()
    stage2_validation = build_stage2_validation()
    stage2_test = build_stage2_test()
    takeaways = build_takeaways()
    external_sizes = stage1_external["N"].dropna().astype(int).unique()
    if len(external_sizes) != 1:
        raise ValueError(
            "Las tres evaluaciones externas no tienen el mismo numero de muestras."
        )
    metric_notes = build_metric_notes(int(external_sizes[0]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_tables = {
        "stage1_internal_key_results.csv": stage1_internal,
        "stage1_validation_key_results.csv": stage1_validation,
        "stage1_external_pilot_results.csv": stage1_external,
        "stage2_validation_key_results.csv": stage2_validation,
        "stage2_final_test_result.csv": stage2_test,
    }
    for filename, data in csv_tables.items():
        data.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")

    output_xlsx = OUTPUT_XLSX

    def write_excel(path: Path) -> None:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            add_sheet(writer, "Lectura_rapida", takeaways)
            add_sheet(writer, "Stage1_TEST_interno", stage1_internal)
            add_sheet(writer, "Stage1_VALIDATION", stage1_validation)
            add_sheet(writer, "Stage1_externo", stage1_external)
            add_sheet(writer, "Stage2_VALIDATION", stage2_validation)
            add_sheet(writer, "Stage2_TEST_final", stage2_test)
            add_sheet(writer, "Definiciones", metric_notes)

    try:
        write_excel(output_xlsx)
    except PermissionError:
        output_xlsx = OUTPUT_XLSX.with_name(
            f"{OUTPUT_XLSX.stem}_updated{OUTPUT_XLSX.suffix}"
        )
        write_excel(output_xlsx)
        print(
            "El Excel principal estaba abierto/bloqueado; "
            f"se creó una copia actualizada: {output_xlsx}"
        )
    style_workbook(output_xlsx)

    print(f"Excel guardado: {output_xlsx}")
    print(f"CSV guardados: {OUTPUT_DIR}")
    print(f"Stage 1 interno: {len(stage1_internal)} experimentos")
    print(f"Stage 1 validation: {len(stage1_validation)} experimentos")
    print(f"Stage 1 externo: {len(stage1_external)} modelos")
    print(f"Stage 2 validation: {len(stage2_validation)} experimentos clave")
    print(f"Stage 2 test: {len(stage2_test)} modelo final")


if __name__ == "__main__":
    main()
