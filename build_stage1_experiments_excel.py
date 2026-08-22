"""Genera el Excel reproducible de experimentos Stage 1 (no tos / tos)."""

from __future__ import annotations

import re
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


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "stage1_cough_no_cough_experiments_summary.xlsx"
HISTORICAL_TEXT = ROOT / "graphs_results_experiment_random" / "results_training.txt"
CURRENT_RESULTS = ROOT / "results_stage1_cough_no_cough" / "mfcc117_rf" / "full"


def historical_metrics() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary: dict[str, Any] = {
        "ID": "S1-H01",
        "Estado": "Histórico",
        "Experimento": "MFCC117 + RF (splits antiguos)",
        "Features": "13 MFCC + delta + delta2; mean/std/max = 117",
        "Clasificador": "Random Forest",
        "Datos/splits": "features_extracted_experiment_random (10-08-2026)",
        "Umbral": 0.5,
        "Nota": "Resultado original; metadatos anteriores al consenso actualizado.",
        "Fuente": str(HISTORICAL_TEXT),
    }
    details: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    if not HISTORICAL_TEXT.is_file():
        summary["Estado"] = "No encontrado"
        return summary, pd.DataFrame(), pd.DataFrame()

    text = HISTORICAL_TEXT.read_text(encoding="utf-8", errors="ignore")
    for fold, score in re.findall(r"Fold\s+(\d+)\s+-\s+F1-Score:\s*([0-9.]+)", text):
        folds.append({"ID": "S1-H01", "fold": int(fold), "f1_cough": float(score)})
    match = re.search(r"VALIDACI[^:]*:\s*([0-9.]+)", text, flags=re.IGNORECASE)
    if match:
        summary["F1 CV medio"] = float(match.group(1))
    if folds:
        fold_scores = [row["f1_cough"] for row in folds]
        summary["F1 CV std"] = float(np.std(fold_scores, ddof=0))

    block_pattern = re.compile(
        r"Conjunto\s+(TRAIN|VALIDATION|TEST \(BLIND\))\s*:(.*?)(?=Conjunto|Pipeline|\Z)",
        flags=re.DOTALL,
    )
    metric_patterns = {
        "accuracy": r"Accuracy:\s*([0-9.]+)",
        "precision_cough": r"Precision:\s*([0-9.]+)",
        "recall_cough": r"Recall:\s*([0-9.]+)",
        "f1_cough": r"F1-Score:\s*([0-9.]+)",
        "roc_auc": r"ROC-AUC:\s*([0-9.]+)",
    }
    split_names = {"TRAIN": "train_fit", "VALIDATION": "validation", "TEST (BLIND)": "test"}
    for raw_split, block in block_pattern.findall(text):
        row: dict[str, Any] = {"ID": "S1-H01", "dataset": split_names[raw_split]}
        for name, pattern in metric_patterns.items():
            found = re.search(pattern, block)
            row[name] = float(found.group(1)) if found else np.nan
        details.append(row)

    detail_df = pd.DataFrame(details)
    validation = detail_df[detail_df["dataset"] == "validation"]
    test = detail_df[detail_df["dataset"] == "test"]
    if not validation.empty:
        row = validation.iloc[0]
        summary.update(
            {
                "F1 validation": row.get("f1_cough", np.nan),
                "Recall tos validation": row.get("recall_cough", np.nan),
                "Precision tos validation": row.get("precision_cough", np.nan),
                "AUC validation": row.get("roc_auc", np.nan),
            }
        )
    if not test.empty:
        row = test.iloc[0]
        summary.update(
            {
                "F1 test": row.get("f1_cough", np.nan),
                "Recall tos test": row.get("recall_cough", np.nan),
                "AUC test": row.get("roc_auc", np.nan),
            }
        )
    return summary, detail_df, pd.DataFrame(folds)


def current_metrics() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_path = CURRENT_RESULTS / "metrics_summary.csv"
    summary: dict[str, Any] = {
        "ID": "S1-C01",
        "Estado": "Pendiente de ejecución",
        "Experimento": "MFCC117 + RF (splits actuales)",
        "Features": "13 MFCC + delta + delta2; mean/std/max = 117",
        "Clasificador": "Random Forest",
        "Datos/splits": "multiclass_4c actual; 0=no_tos, 1/2/3=tos",
        "Umbral": 0.5,
        "Nota": "Repetición controlada con consenso y folds actuales.",
        "Fuente": str(metrics_path),
    }
    if not metrics_path.is_file():
        return summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    metrics = pd.read_csv(metrics_path)
    metrics.insert(0, "ID", "S1-C01")
    folds_path = CURRENT_RESULTS / "cv_fold_metrics.csv"
    folds = pd.read_csv(folds_path) if folds_path.is_file() else pd.DataFrame()
    if not folds.empty:
        folds.insert(0, "ID", "S1-C01")
    config_path = CURRENT_RESULTS / "experiment_configuration.csv"
    config = pd.read_csv(config_path) if config_path.is_file() else pd.DataFrame()
    if not config.empty:
        config.insert(0, "ID", "S1-C01")

    summary["Estado"] = "Finalizado"
    oof = metrics[metrics["dataset"] == "train_oof"]
    validation = metrics[metrics["dataset"] == "validation"]
    test = metrics[metrics["dataset"] == "test"]
    if not oof.empty:
        summary["Macro-F1 OOF"] = oof.iloc[0].get("macro_f1", np.nan)
    if not folds.empty:
        summary["F1 CV medio"] = folds["f1_cough"].mean()
        summary["F1 CV std"] = folds["f1_cough"].std(ddof=0)
    if not validation.empty:
        row = validation.iloc[0]
        summary.update(
            {
                "F1 validation": row.get("f1_cough", np.nan),
                "Macro-F1 validation": row.get("macro_f1", np.nan),
                "Recall tos validation": row.get("recall_cough", np.nan),
                "Especificidad validation": row.get("specificity_no_cough", np.nan),
                "Precision tos validation": row.get("precision_cough", np.nan),
                "AUC validation": row.get("roc_auc", np.nan),
            }
        )
    if not test.empty:
        row = test.iloc[0]
        summary.update(
            {
                "F1 test": row.get("f1_cough", np.nan),
                "Recall tos test": row.get("recall_cough", np.nan),
                "AUC test": row.get("roc_auc", np.nan),
            }
        )
    return summary, metrics, folds, config


def add_table(ws, name: str) -> None:
    if ws.max_row < 2 or ws.max_column < 1:
        return
    table = Table(displayName=name, ref=f"A1:{get_column_letter(ws.max_column)}{ws.max_row}")
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
        showRowStripes=True, showColumnStripes=False
    )
    ws.add_table(table)


def style_workbook() -> None:
    workbook = load_workbook(OUTPUT_PATH)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for ws in workbook.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for column in ws.columns:
            values = [str(cell.value or "") for cell in column[: min(ws.max_row, 100)]]
            width = min(max(max(map(len, values), default=8) + 2, 10), 45)
            ws.column_dimensions[get_column_letter(column[0].column)].width = width
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

    summary = workbook["Resumen"]
    header = {cell.value: cell.column for cell in summary[1]}
    for metric in ["F1 validation", "Recall tos validation", "AUC validation"]:
        if metric in header and summary.max_row >= 3:
            letter = get_column_letter(header[metric])
            summary.conditional_formatting.add(
                f"{letter}2:{letter}{summary.max_row}",
                ColorScaleRule(start_type="min", start_color="F8696B", mid_type="percentile",
                               mid_value=50, mid_color="FFEB84", end_type="max", end_color="63BE7B"),
            )
    if "F1 validation" in header:
        chart = BarChart()
        chart.title = "F1 de tos en validation"
        chart.y_axis.title = "F1"
        chart.height = 7
        chart.width = 14
        chart.add_data(
            Reference(summary, min_col=header["F1 validation"], min_row=1, max_row=summary.max_row),
            titles_from_data=True,
        )
        chart.set_categories(Reference(summary, min_col=3, min_row=2, max_row=summary.max_row))
        summary.add_chart(chart, f"A{summary.max_row + 4}")
    workbook.save(OUTPUT_PATH)


def build_workbook() -> Path:
    historical_summary, historical_detail, historical_folds = historical_metrics()
    current_summary, current_detail, current_folds, current_config = current_metrics()
    summary = pd.DataFrame([historical_summary, current_summary])
    preferred_columns = [
        "ID", "Estado", "Experimento", "Features", "Clasificador", "Datos/splits", "Umbral",
        "F1 CV medio", "F1 CV std", "Macro-F1 OOF", "F1 validation", "Macro-F1 validation",
        "Recall tos validation", "Especificidad validation", "Precision tos validation",
        "AUC validation", "F1 test", "Recall tos test", "AUC test", "Nota", "Fuente",
    ]
    summary = summary.reindex(columns=preferred_columns)
    details = pd.concat([historical_detail, current_detail], ignore_index=True, sort=False)
    folds = pd.concat([historical_folds, current_folds], ignore_index=True, sort=False)
    notes = pd.DataFrame(
        {
            "Aspecto": ["Objetivo", "Mapeo", "Comparabilidad", "TEST", "Actualización"],
            "Descripción": [
                "Stage 1 detecta tos frente a no tos.",
                "Clase 0=no_cough; dry, wet y unknown se consideran tos (clase 1).",
                "S1-H01 usa splits antiguos; S1-C01 usa splits y consenso actuales.",
                "El test ya fue consultado en agosto y no debe describirse como completamente virgen.",
                "Ejecutar build_stage1_experiments_excel.py tras añadir resultados nuevos.",
            ],
        }
    )

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Resumen", index=False)
        details.to_excel(writer, sheet_name="Metricas_detalle", index=False)
        folds.to_excel(writer, sheet_name="CV_por_fold", index=False)
        current_config.to_excel(writer, sheet_name="Configuraciones", index=False)
        notes.to_excel(writer, sheet_name="Notas_y_protocolo", index=False)

    workbook = load_workbook(OUTPUT_PATH)
    for ws, name in [
        (workbook["Resumen"], "Stage1Summary"),
        (workbook["Metricas_detalle"], "Stage1Metrics"),
        (workbook["CV_por_fold"], "Stage1Folds"),
        (workbook["Configuraciones"], "Stage1Configs"),
        (workbook["Notas_y_protocolo"], "Stage1Notes"),
    ]:
        add_table(ws, name)
    workbook.save(OUTPUT_PATH)
    style_workbook()
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_workbook()
    print(f"Excel Stage 1 generado: {path}")
