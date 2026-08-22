"""Analiza complementariedad recording-level entre WST-LR y MFCC-LR.

Las probabilidades se alinean uno-a-uno mediante ``original_uuid``. Se
guardan Pearson, Spearman, desacuerdo de decisiones y recuperacion cruzada
de errores para TRAIN OOF y VALIDATION. Los umbrales son los seleccionados
previamente con OOF de TRAIN. TEST no se lee.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results_stage2_complementarity_wst_mfcc_recording"
WST_DIR = (
    ROOT
    / "results_stage2_dry_wet_wavelet_scattering_recording_linear"
    / "paper_q8_q1_t500_full"
    / "full"
    / "logistic_regression"
)
MFCC_DIR = (
    ROOT
    / "results_stage2_dry_wet_mfcc_recording_linear"
    / "mfcc117"
    / "full"
    / "logistic_regression"
)


def prediction_path(result_dir: Path, split: str) -> Path:
    filename = (
        "best_oof_predictions.csv"
        if split == "train_oof"
        else "validation_predictions.csv"
    )
    return result_dir / filename


def load_threshold(result_dir: Path) -> float:
    path = result_dir / "metrics_summary.csv"
    metrics = pd.read_csv(path)
    rows = metrics[
        (metrics["dataset"].astype(str) == "train_oof")
        & (metrics["threshold_policy"].astype(str) == "oof_tuned_frozen")
    ]
    if len(rows) != 1:
        raise ValueError(f"No hay un unico umbral OOF congelado en {path}.")
    return float(rows.iloc[0]["threshold"])


def load_aligned(split: str) -> pd.DataFrame:
    columns = [
        "original_uuid",
        "y_true",
        "cough_type",
        "cough_type_consensus",
        "fold",
        "score",
    ]
    wst = pd.read_csv(prediction_path(WST_DIR, split), usecols=columns).rename(
        columns={"score": "wst_probability"}
    )
    mfcc = pd.read_csv(prediction_path(MFCC_DIR, split), usecols=columns).rename(
        columns={"score": "mfcc_probability"}
    )
    if wst["original_uuid"].duplicated().any():
        raise ValueError(f"WST contiene UUID duplicados en {split}.")
    if mfcc["original_uuid"].duplicated().any():
        raise ValueError(f"MFCC contiene UUID duplicados en {split}.")

    aligned = wst.merge(
        mfcc,
        on="original_uuid",
        how="inner",
        suffixes=("", "__mfcc"),
        validate="one_to_one",
    )
    if len(aligned) != len(wst) or len(aligned) != len(mfcc):
        raise ValueError(f"WST y MFCC no contienen los mismos UUID en {split}.")
    for column in ("y_true", "cough_type", "cough_type_consensus", "fold"):
        other = f"{column}__mfcc"
        if not aligned[column].equals(aligned[other]):
            raise ValueError(f"WST y MFCC discrepan en {column} para {split}.")
        aligned = aligned.drop(columns=other)
    for column in ("wst_probability", "mfcc_probability"):
        values = aligned[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{column} no contiene probabilidades validas.")
    return aligned


def summarize(
    split: str,
    data: pd.DataFrame,
    wst_threshold: float,
    mfcc_threshold: float,
) -> dict[str, float | int | str]:
    y = data["y_true"].to_numpy(dtype=int)
    wst_score = data["wst_probability"].to_numpy(dtype=float)
    mfcc_score = data["mfcc_probability"].to_numpy(dtype=float)
    wst_prediction = wst_score >= wst_threshold
    mfcc_prediction = mfcc_score >= mfcc_threshold
    wst_correct = wst_prediction == y
    mfcc_correct = mfcc_prediction == y
    return {
        "dataset": split,
        "recording_count": len(data),
        "wst_threshold_oof": wst_threshold,
        "mfcc_threshold_oof": mfcc_threshold,
        "pearson_score": float(
            pd.Series(wst_score).corr(pd.Series(mfcc_score), method="pearson")
        ),
        "spearman_score": float(
            pd.Series(wst_score).corr(pd.Series(mfcc_score), method="spearman")
        ),
        "prediction_disagreement_pct": float(
            np.mean(wst_prediction != mfcc_prediction) * 100.0
        ),
        "correctness_pearson": float(
            pd.Series(wst_correct.astype(int)).corr(
                pd.Series(mfcc_correct.astype(int)), method="pearson"
            )
        ),
        "both_correct": int(np.sum(wst_correct & mfcc_correct)),
        "wst_only_correct": int(np.sum(wst_correct & ~mfcc_correct)),
        "mfcc_only_correct": int(np.sum(~wst_correct & mfcc_correct)),
        "both_wrong": int(np.sum(~wst_correct & ~mfcc_correct)),
        "wet_wst_errors_recovered_by_mfcc": int(
            np.sum((y == 1) & ~wst_correct & mfcc_correct)
        ),
        "dry_wst_errors_recovered_by_mfcc": int(
            np.sum((y == 0) & ~wst_correct & mfcc_correct)
        ),
        "wet_mfcc_errors_recovered_by_wst": int(
            np.sum((y == 1) & ~mfcc_correct & wst_correct)
        ),
        "dry_mfcc_errors_recovered_by_wst": int(
            np.sum((y == 0) & ~mfcc_correct & wst_correct)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Complementariedad OOF entre WST-LR y MFCC-LR."
    )
    parser.add_argument("--action", choices=["check", "analyze"], default="check")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("COMPLEMENTARIEDAD STAGE 2 — WST-LR + MFCC-LR")
    print("=" * 78)
    print(f"Accion: {args.action}")
    print("TEST no sera leido ni procesado.")

    wst_threshold = load_threshold(WST_DIR)
    mfcc_threshold = load_threshold(MFCC_DIR)
    train = load_aligned("train_oof")
    validation = load_aligned("validation")
    if set(train["original_uuid"]) & set(validation["original_uuid"]):
        raise ValueError("TRAIN OOF y VALIDATION comparten UUID.")
    summaries = pd.DataFrame(
        [
            summarize("train_oof", train, wst_threshold, mfcc_threshold),
            summarize("validation", validation, wst_threshold, mfcc_threshold),
        ]
    )
    print(
        summaries[
            [
                "dataset",
                "recording_count",
                "pearson_score",
                "spearman_score",
                "prediction_disagreement_pct",
                "mfcc_only_correct",
                "both_wrong",
            ]
        ].to_string(index=False)
    )
    if args.action == "check":
        print("Comprobacion completada. No se guardaron resultados.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(
        OUTPUT_DIR / "complementarity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "alignment_key": "original_uuid",
                "unit": "recording",
                "wst_model": str(WST_DIR),
                "mfcc_model": str(MFCC_DIR),
                "threshold_source": "TRAIN OOF",
                "validation_used_for_selection": False,
                "test_processed": False,
            }
        ]
    ).to_csv(
        OUTPUT_DIR / "analysis_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Resultados guardados en: {OUTPUT_DIR}")
    print("TEST permanece reservado.")


if __name__ == "__main__":
    main()
