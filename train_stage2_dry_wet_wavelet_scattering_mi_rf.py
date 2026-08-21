"""Seleccion por informacion mutua sobre WST y Random Forest fijo.

Compara 1, 5, 10, 30, 50 y 100 % de los 644 caminos Wavelet Scattering.
El ranking de informacion mutua se aprende dentro de cada fold usando una
media de eventos por ``original_uuid``. El clasificador sigue entrenandose
por evento con pesos equilibrados y sus probabilidades se promedian por
grabacion para calcular metricas.

El Random Forest se mantiene fijo en la configuracion ganadora del baseline
WST completo para medir solamente el efecto de la seleccion. No aplica PCA,
SMOTE ni estandarizacion. TEST no se lee ni se procesa.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_rf as baseline


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results_stage2_dry_wet_wavelet_scattering_mi_rf"
MODELS_ROOT = SCRIPT_DIR / "models_stage2_dry_wet_wavelet_scattering_mi_rf"
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wavelet_scattering_mi_rf"
)

RANDOM_STATE = 42
MI_NEIGHBORS = 3
SELECTION_TOLERANCE_MACRO_F1 = 0.005
FULL_FRACTIONS = (0.01, 0.05, 0.10, 0.30, 0.50, 1.00)
QUICK_FRACTIONS = (0.10, 1.00)
PRESETS = baseline.PRESETS
FIXED_RF = baseline.RFCandidate(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    max_features=0.1,
)


@dataclass(frozen=True)
class FractionSpec:
    fraction: float
    feature_count: int

    @property
    def percent(self) -> int:
        return int(round(self.fraction * 100))

    @property
    def key(self) -> str:
        return f"mi_{self.percent:03d}pct__k{self.feature_count:03d}"


@dataclass
class FractionEvaluation:
    spec: FractionSpec
    threshold: float
    metrics: dict[str, float | int]
    oof_recordings: pd.DataFrame
    fold_metrics: pd.DataFrame


def feature_specs(feature_count: int, quick: bool) -> list[FractionSpec]:
    fractions = QUICK_FRACTIONS if quick else FULL_FRACTIONS
    specs = []
    for fraction in fractions:
        count = max(1, int(round(feature_count * fraction)))
        if math.isclose(fraction, 1.0):
            count = feature_count
        specs.append(FractionSpec(fraction=fraction, feature_count=count))
    if len({spec.feature_count for spec in specs}) != len(specs):
        raise RuntimeError("Dos porcentajes MI generan el mismo numero de features.")
    return specs


def load_feature_layout(preset: str) -> pd.DataFrame:
    path = (
        baseline.FEATURES_ROOT
        / preset
        / "wavelet_scattering_feature_layout.csv"
    )
    layout = pd.read_csv(path)
    required = {"feature_index", "feature_name", "scattering_order"}
    missing = required - set(layout.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el layout WST: {sorted(missing)}")
    if layout["feature_index"].tolist() != list(range(len(layout))):
        raise ValueError("feature_index no es consecutivo en el layout WST.")
    return layout


def aggregate_event_features_by_recording(
    x_values: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x_values) != len(metadata):
        raise ValueError("Features y metadatos tienen longitudes distintas.")

    feature_rows = []
    labels = []
    uuids = []
    for original_uuid, positions in metadata.groupby(
        "original_uuid", sort=False
    ).indices.items():
        indices = np.asarray(positions, dtype=int)
        group_labels = metadata.iloc[indices]["stage2_target"].unique()
        if len(group_labels) != 1:
            raise ValueError(f"Etiquetas inconsistentes en {original_uuid}.")
        feature_rows.append(
            np.mean(x_values[indices], axis=0, dtype=np.float64)
        )
        labels.append(int(group_labels[0]))
        uuids.append(str(original_uuid))

    recording_features = np.asarray(feature_rows, dtype=np.float32)
    recording_labels = np.asarray(labels, dtype=np.int64)
    recording_uuids = np.asarray(uuids, dtype=str)
    if not np.isfinite(recording_features).all():
        raise RuntimeError("La agregacion por grabacion genero NaN/Inf.")
    if set(recording_labels.tolist()) != {0, 1}:
        raise ValueError("La agregacion por grabacion no contiene ambas clases.")
    return recording_features, recording_labels, recording_uuids


def mutual_information_ranking(
    x_events: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:
    x_recordings, y_recordings, _ = aggregate_event_features_by_recording(
        x_events,
        metadata,
    )
    scores = mutual_info_classif(
        x_recordings,
        y_recordings,
        discrete_features=False,
        n_neighbors=MI_NEIGHBORS,
        copy=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (x_events.shape[1],) or not np.isfinite(scores).all():
        raise RuntimeError("Informacion mutua produjo valores invalidos.")

    # Mayor MI primero; ante empates se usa el indice original.
    ranking = np.lexsort((np.arange(len(scores)), -scores))
    return scores, ranking.astype(int), len(x_recordings)


def selected_indices(
    ranking: np.ndarray,
    spec: FractionSpec,
    total_features: int,
) -> np.ndarray:
    if spec.feature_count == total_features:
        # Mantener el orden original hace que 100 % reproduzca exactamente
        # el baseline WST completo con el mismo random_state del RF.
        return np.arange(total_features, dtype=int)
    return np.sort(ranking[: spec.feature_count]).astype(int)


def evaluate_fractions_oof(
    data: common.DataView,
    specs: list[FractionSpec],
) -> tuple[list[FractionEvaluation], pd.DataFrame]:
    oof_scores = {
        spec.key: np.full(len(data.x_train), np.nan, dtype=float)
        for spec in specs
    }
    ranking_rows: list[dict[str, float | int]] = []
    total_features = data.x_train.shape[1]

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = data.train_metadata["fold"].to_numpy() == fold
        training_mask = ~validation_mask
        training_metadata = data.train_metadata.loc[training_mask].reset_index(
            drop=True
        )
        x_training = data.x_train[training_mask]
        y_training = training_metadata["stage2_target"].to_numpy(dtype=int)
        sample_weights = common.compute_training_weights(training_metadata)

        print(f"  Fold {fold}: calculando MI solo con grabaciones de ajuste...")
        mi_scores, ranking, recording_count = mutual_information_ranking(
            x_training,
            training_metadata,
        )
        rank_positions = np.empty(total_features, dtype=int)
        rank_positions[ranking] = np.arange(1, total_features + 1)
        for feature_index in range(total_features):
            ranking_rows.append(
                {
                    "fold": fold,
                    "training_recording_count": recording_count,
                    "feature_index": feature_index,
                    "mi_score": mi_scores[feature_index],
                    "mi_rank": int(rank_positions[feature_index]),
                }
            )

        for spec_index, spec in enumerate(specs, start=1):
            indices = selected_indices(ranking, spec, total_features)
            classifier = baseline.build_rf(FIXED_RF)
            classifier.fit(
                x_training[:, indices],
                y_training,
                sample_weight=sample_weights,
            )
            oof_scores[spec.key][validation_mask] = baseline.rf_scores(
                classifier,
                data.x_train[validation_mask][:, indices],
            )
            print(
                f"    MI {spec_index:02d}/{len(specs):02d}: "
                f"{spec.key}"
            )

    evaluations: list[FractionEvaluation] = []
    for spec in specs:
        event_scores = oof_scores[spec.key]
        if not np.isfinite(event_scores).all():
            raise RuntimeError(f"OOF incompleto para {spec.key}.")
        recordings = common.aggregate_scores_by_recording(
            data.train_metadata,
            event_scores,
        )
        y_true = recordings["y_true"].to_numpy(dtype=int)
        scores = recordings["score"].to_numpy(dtype=float)
        threshold = common.tune_threshold(y_true, scores, 0.5)
        metrics = common.binary_metrics(y_true, scores, threshold)
        recordings["y_pred"] = (scores >= threshold).astype(int)
        recordings["candidate_key"] = spec.key

        fold_rows = []
        for fold, fold_df in recordings.groupby("fold", sort=True):
            fold_rows.append(
                {
                    "fold": int(fold),
                    "recording_count": len(fold_df),
                    "threshold": threshold,
                    **common.binary_metrics(
                        fold_df["y_true"].to_numpy(dtype=int),
                        fold_df["score"].to_numpy(dtype=float),
                        threshold,
                    ),
                }
            )
        evaluations.append(
            FractionEvaluation(
                spec=spec,
                threshold=threshold,
                metrics=metrics,
                oof_recordings=recordings,
                fold_metrics=pd.DataFrame(fold_rows),
            )
        )
    return evaluations, pd.DataFrame(ranking_rows)


def select_fraction(
    evaluations: list[FractionEvaluation],
) -> tuple[FractionEvaluation, float, list[str]]:
    best_macro_f1 = max(float(item.metrics["macro_f1"]) for item in evaluations)
    cutoff = best_macro_f1 - SELECTION_TOLERANCE_MACRO_F1
    eligible = [
        item for item in evaluations if float(item.metrics["macro_f1"]) >= cutoff
    ]
    selected = min(
        eligible,
        key=lambda item: (
            item.spec.feature_count,
            -float(item.metrics["macro_f1"]),
            -float(item.metrics["balanced_accuracy"]),
            -float(item.metrics["roc_auc"]),
        ),
    )
    return selected, best_macro_f1, [item.spec.key for item in eligible]


def forest_complexity(classifier: object) -> dict[str, float | int]:
    return baseline.forest_complexity(classifier)


def save_oof_results(
    evaluations: list[FractionEvaluation],
    selected: FractionEvaluation,
    ranking_by_fold: pd.DataFrame,
    feature_layout: pd.DataFrame,
    result_dir: Path,
    elapsed_seconds: float,
    best_macro_f1: float,
    eligible_keys: list[str],
) -> None:
    rows = []
    for evaluation in evaluations:
        rows.append(
            {
                "candidate_key": evaluation.spec.key,
                "feature_fraction": evaluation.spec.fraction,
                "feature_percent": evaluation.spec.percent,
                "selected_feature_count": evaluation.spec.feature_count,
                "fixed_rf": FIXED_RF.key,
                "threshold_oof": evaluation.threshold,
                "elapsed_seconds_total_search": elapsed_seconds,
                "within_0p005_of_best_macro_f1": (
                    evaluation.spec.key in eligible_keys
                ),
                "selected_by_compact_rule": (
                    evaluation.spec.key == selected.spec.key
                ),
                **evaluation.metrics,
            }
        )
    pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy", "roc_auc"],
        ascending=False,
    ).to_csv(
        result_dir / "fraction_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.oof_recordings.to_csv(
        result_dir / "selected_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.fold_metrics.to_csv(
        result_dir / "selected_cv_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranking_with_layout = ranking_by_fold.merge(
        feature_layout,
        on="feature_index",
        how="left",
        validate="many_to_one",
    )
    ranking_with_layout.to_csv(
        result_dir / "mi_ranking_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "best_macro_f1_oof": best_macro_f1,
                "macro_f1_tolerance": SELECTION_TOLERANCE_MACRO_F1,
                "eligible_candidates": "|".join(eligible_keys),
                "selected_candidate": selected.spec.key,
                "selection_rule": (
                    "smallest_feature_count_with_macro_f1_within_0.005_of_best"
                ),
            }
        ]
    ).to_csv(
        result_dir / "selection_decision.csv",
        index=False,
        encoding="utf-8-sig",
    )


def final_feature_tables(
    data: common.DataView,
    feature_layout: pd.DataFrame,
    selected: FractionEvaluation,
    ranking_by_fold: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_scores, full_ranking, recording_count = mutual_information_ranking(
        data.x_train,
        data.train_metadata,
    )
    indices = selected_indices(
        full_ranking,
        selected.spec,
        data.x_train.shape[1],
    )
    full_rank_positions = np.empty(len(full_scores), dtype=int)
    full_rank_positions[full_ranking] = np.arange(1, len(full_scores) + 1)

    full_table = feature_layout.copy()
    full_table["mi_score_full_train"] = full_scores
    full_table["mi_rank_full_train"] = full_rank_positions
    full_table["selected_final"] = False
    full_table.loc[indices, "selected_final"] = True
    full_table["full_train_recording_count"] = recording_count
    full_table = full_table.sort_values("mi_rank_full_train").reset_index(drop=True)

    selected_table = full_table[full_table["selected_final"]].copy()
    selected_table = selected_table.sort_values("mi_rank_full_train")

    fold_selected = ranking_by_fold[
        ranking_by_fold["mi_rank"] <= selected.spec.feature_count
    ]
    stability = (
        ranking_by_fold.groupby("feature_index")
        .agg(
            mean_mi_score_folds=("mi_score", "mean"),
            std_mi_score_folds=("mi_score", "std"),
            mean_mi_rank_folds=("mi_rank", "mean"),
        )
        .reset_index()
    )
    selected_counts = fold_selected["feature_index"].value_counts()
    stability["selected_fold_count"] = (
        stability["feature_index"].map(selected_counts).fillna(0).astype(int)
    )
    stability = stability.merge(
        full_table[
            [
                "feature_index",
                "feature_name",
                "scattering_order",
                "mi_score_full_train",
                "mi_rank_full_train",
                "selected_final",
            ]
        ],
        on="feature_index",
        how="left",
        validate="one_to_one",
    ).sort_values(
        ["selected_fold_count", "mean_mi_score_folds"],
        ascending=False,
    )

    order_summary = (
        selected_table.groupby("scattering_order")
        .size()
        .rename("selected_feature_count")
        .reset_index()
    )
    order_summary["selected_fraction"] = selected.spec.fraction
    return indices, full_table, selected_table, stability.merge(
        order_summary,
        on="scattering_order",
        how="left",
    )


def train(
    data: common.DataView,
    extraction_configuration: pd.DataFrame,
    feature_layout: pd.DataFrame,
    preset: str,
    quick: bool,
) -> None:
    run_name = "quick" if quick else "full"
    result_dir = RESULTS_ROOT / preset / run_name
    model_dir = MODELS_ROOT / preset / run_name
    graph_dir = GRAPHS_ROOT / preset / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    specs = feature_specs(data.x_train.shape[1], quick)
    print("\n" + "=" * 78)
    print("CV WST -> INFORMACION MUTUA -> RF FIJO")
    print("=" * 78)
    print(f"Porcentajes: {[spec.percent for spec in specs]}")
    print(f"RF fijo: {FIXED_RF.key}")
    start_time = time.perf_counter()
    evaluations, ranking_by_fold = evaluate_fractions_oof(data, specs)
    elapsed_seconds = time.perf_counter() - start_time

    for evaluation in evaluations:
        print(
            f"{evaluation.spec.key} | "
            f"macro-F1={evaluation.metrics['macro_f1']:.4f} | "
            f"bal-acc={evaluation.metrics['balanced_accuracy']:.4f} | "
            f"wet-recall={evaluation.metrics['wet_recall']:.4f} | "
            f"AUC={evaluation.metrics['roc_auc']:.4f}"
        )

    selected, best_macro_f1, eligible_keys = select_fraction(evaluations)
    save_oof_results(
        evaluations,
        selected,
        ranking_by_fold,
        feature_layout,
        result_dir,
        elapsed_seconds,
        best_macro_f1,
        eligible_keys,
    )
    oof_summary = {
        "dataset": "train_oof",
        "experiment": "wavelet_scattering_mi_rf_event",
        "candidate_key": selected.spec.key,
        "threshold": selected.threshold,
        "input_feature_count": data.x_train.shape[1],
        "selected_feature_count": selected.spec.feature_count,
        "selected_feature_fraction": selected.spec.fraction,
        "model_size_kb_joblib": np.nan,
        **selected.metrics,
    }

    if quick:
        pd.DataFrame([oof_summary]).to_csv(
            result_dir / "metrics_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\nPrueba rapida MI completada.")
        print(f"Mejor macro-F1 observado: {best_macro_f1:.4f}")
        print(f"Seleccion compacta: {selected.spec.key}")
        print("VALIDATION no se ha evaluado.")
        print(f"Resultados: {result_dir}")
        return

    print("\nCalculando ranking MI final exclusivamente con todo TRAIN...")
    (
        final_indices,
        full_feature_table,
        selected_feature_table,
        stability_table,
    ) = final_feature_tables(
        data,
        feature_layout,
        selected,
        ranking_by_fold,
    )
    full_feature_table.to_csv(
        result_dir / "mi_ranking_full_train.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_feature_table.to_csv(
        result_dir / "selected_features_final.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stability_table.to_csv(
        result_dir / "selected_feature_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Ajustando RF final con las features seleccionadas...")
    classifier = baseline.build_rf(FIXED_RF)
    classifier.fit(
        data.x_train[:, final_indices],
        data.train_metadata["stage2_target"].to_numpy(dtype=int),
        sample_weight=common.compute_training_weights(data.train_metadata),
    )
    validation_event_scores = baseline.rf_scores(
        classifier,
        data.x_validation[:, final_indices],
    )
    validation_recordings = common.aggregate_scores_by_recording(
        data.validation_metadata,
        validation_event_scores,
    )
    validation_scores = validation_recordings["score"].to_numpy(dtype=float)
    validation_recordings["y_pred"] = (
        validation_scores >= selected.threshold
    ).astype(int)
    validation_recordings["candidate_key"] = selected.spec.key
    validation_metrics = common.binary_metrics(
        validation_recordings["y_true"].to_numpy(dtype=int),
        validation_scores,
        selected.threshold,
    )

    complexity = forest_complexity(classifier)
    model_package = {
        "classifier": classifier,
        "selected_feature_indices": final_indices,
        "selected_feature_names": feature_layout.set_index("feature_index")
        .loc[final_indices, "feature_name"]
        .astype(str)
        .tolist(),
        "selected_feature_count": selected.spec.feature_count,
        "selected_feature_fraction": selected.spec.fraction,
        "mi_scores_full_train": full_feature_table.sort_values(
            "feature_index"
        )["mi_score_full_train"].to_numpy(dtype=np.float32),
        "preset": preset,
        "experiment": "wavelet_scattering_mi_rf_event",
        "candidate_key": selected.spec.key,
        "fixed_rf": FIXED_RF.key,
        "threshold": selected.threshold,
        "label_mapping": common.LABEL_TO_NAME,
        "recording_aggregation": "mean_event_probability",
        "mi_ranking_unit": "mean_event_features_per_original_uuid",
        "selection_rule": (
            "smallest_feature_count_with_macro_f1_within_0.005_of_best"
        ),
        "pca_used": False,
        "smote_used": False,
        "standardization_used": False,
        "classifier_weighting": (
            "class_balanced_and_1_over_events_per_recording"
        ),
        "extraction_configuration": extraction_configuration.iloc[0].to_dict(),
        "random_state": RANDOM_STATE,
        **complexity,
    }
    model_path = model_dir / "wavelet_scattering_mi_rf_event_model.joblib"
    joblib.dump(model_package, model_path, compress=3)
    model_size_kb = model_path.stat().st_size / 1024.0

    oof_summary.update(
        {"model_size_kb_joblib": model_size_kb, **complexity}
    )
    validation_summary = {
        "dataset": "validation",
        "experiment": "wavelet_scattering_mi_rf_event",
        "candidate_key": selected.spec.key,
        "threshold": selected.threshold,
        "input_feature_count": data.x_train.shape[1],
        "selected_feature_count": selected.spec.feature_count,
        "selected_feature_fraction": selected.spec.fraction,
        "model_size_kb_joblib": model_size_kb,
        **complexity,
        **validation_metrics,
    }
    pd.DataFrame([oof_summary, validation_summary]).to_csv(
        result_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_recordings.to_csv(
        result_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "input_feature_count": data.x_train.shape[1],
                "selected_feature_count": selected.spec.feature_count,
                "selected_feature_fraction": selected.spec.fraction,
                "mi_neighbors": MI_NEIGHBORS,
                "mi_ranking_unit": "mean_event_features_per_original_uuid",
                "macro_f1_tolerance": SELECTION_TOLERANCE_MACRO_F1,
                "fixed_rf": FIXED_RF.key,
                "pca_used": False,
                "smote_used": False,
                "validation_used_for_feature_selection": False,
                "test_processed": False,
                **complexity,
            }
        ]
    ).to_csv(
        result_dir / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    graph_path = graph_dir / "validation_wavelet_scattering_mi_rf_event.png"
    common.create_validation_graph(
        validation_recordings,
        selected.threshold,
        "wavelet_scattering_mi_rf_event",
        selected.spec,
        graph_path,
    )
    order_counts = selected_feature_table["scattering_order"].value_counts()
    print("\n" + "=" * 78)
    print("RESULTADO WST + INFORMACION MUTUA + RF FIJO")
    print("=" * 78)
    print(f"Mejor macro-F1 observado OOF: {best_macro_f1:.4f}")
    print(f"Seleccion compacta: {selected.spec.key}")
    print(f"Umbral OOF: {selected.threshold:.6f}")
    print(f"Macro-F1 OOF seleccionado: {selected.metrics['macro_f1']:.4f}")
    print(f"Macro-F1 validation: {validation_metrics['macro_f1']:.4f}")
    print(
        "Recalls validation dry/wet: "
        f"{validation_metrics['dry_recall']:.4f} / "
        f"{validation_metrics['wet_recall']:.4f}"
    )
    print(
        "Features seleccionadas por orden 0/1/2: "
        f"{int(order_counts.get(0, 0))} / "
        f"{int(order_counts.get(1, 0))} / "
        f"{int(order_counts.get(2, 0))}"
    )
    print(
        f"RF: {complexity['rf_tree_count']} arboles, "
        f"{complexity['rf_total_nodes']} nodos"
    )
    print(f"Modelo joblib: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print(f"Grafica: {graph_path}")
    print("TEST permanece reservado.")


def print_check(data: common.DataView, quick: bool) -> None:
    specs = feature_specs(data.x_train.shape[1], quick)
    train_recordings = data.train_metadata.drop_duplicates("original_uuid")
    validation_recordings = data.validation_metadata.drop_duplicates(
        "original_uuid"
    )
    print("=" * 78)
    print("CHECK - WST + INFORMACION MUTUA + RF FIJO")
    print("=" * 78)
    print(f"X train: {data.x_train.shape} {data.x_train.dtype}")
    print(f"X validation: {data.x_validation.shape} {data.x_validation.dtype}")
    print(
        "Grabaciones train dry/wet: "
        f"{(train_recordings['stage2_target'] == 0).sum()} / "
        f"{(train_recordings['stage2_target'] == 1).sum()}"
    )
    print(
        "Grabaciones validation dry/wet: "
        f"{(validation_recordings['stage2_target'] == 0).sum()} / "
        f"{(validation_recordings['stage2_target'] == 1).sum()}"
    )
    print(
        "Porcentajes/features: "
        + ", ".join(
            f"{spec.percent}%={spec.feature_count}" for spec in specs
        )
    )
    print(f"RF fijo: {FIXED_RF.key}")
    print(
        "Regla: menor numero de features dentro de 0.005 del mejor "
        "macro-F1 OOF."
    )
    print("MI se ajusta por grabacion dentro de cada fold.")
    print("Sin PCA, SMOTE ni uso de VALIDATION para seleccionar.")
    print("TEST no sera leido ni procesado.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seleccion MI de caminos WST con Random Forest fijo."
    )
    parser.add_argument(
        "--action",
        choices=["check", "train"],
        default="check",
        help="check valida datos; train ejecuta CV y, salvo quick, validation.",
    )
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default="paper_q8_q1_t500_full",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Compara 10% y 100% mediante OOF sin evaluar VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST + INFORMACION MUTUA + RANDOM FOREST FIJO")
    print("=" * 78)
    print(f"Preset: {args.preset}")
    print(f"Accion: {args.action}")
    print("TEST no sera leido ni procesado.")
    data, extraction_configuration = baseline.load_wavelet_data(args.preset)
    feature_layout = load_feature_layout(args.preset)
    if len(feature_layout) != data.x_train.shape[1]:
        raise ValueError("El layout no coincide con la matriz WST.")
    print_check(data, args.quick)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(
        data,
        extraction_configuration,
        feature_layout,
        args.preset,
        args.quick,
    )


if __name__ == "__main__":
    main()
