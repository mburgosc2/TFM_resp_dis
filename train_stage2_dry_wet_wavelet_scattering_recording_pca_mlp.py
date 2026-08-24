"""Stage 2 dry/wet: WST recording -> StandardScaler -> PCA128 -> MLP.

Parte de una fila por grabacion con mean+std+max de los 644 caminos WST
(1932 features). StandardScaler y PCA128 se ajustan dentro de cada fold. Una
particion interna del bloque de entrenamiento decide las epocas mediante early
stopping; despues el modelo del fold se vuelve a ajustar con todo el bloque de
entrenamiento durante esas epocas y genera la prediccion OOF exterior.

Arquitectura, regularizacion y umbral se eligen exclusivamente con OOF de
TRAIN. En modo completo, el ganador se ajusta con todo TRAIN y el umbral OOF se
aplica congelado a VALIDATION. TEST no se lee ni se procesa. No se usa SMOTE.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_mlp as mlp_base
import train_stage2_dry_wet_wavelet_scattering_rf as event_baseline


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (
    SCRIPT_DIR / "results_stage2_dry_wet_wavelet_scattering_recording_pca_mlp"
)
MODELS_ROOT = (
    SCRIPT_DIR / "models_stage2_dry_wet_wavelet_scattering_recording_pca_mlp"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wavelet_scattering_recording_pca_mlp"
)

RANDOM_STATE = 42
PCA_COMPONENTS = 128
PCA_RANDOM_STATE = 42
PRESETS = event_baseline.PRESETS
DEFAULT_PRESET = "paper_q8_q1_t500_full"
EXPERIMENT_KEY = "wavelet_scattering_recording_pca128_mlp"
RECORDING_POOLING = mlp_base.RECORDING_POOLING
MLPCandidate = mlp_base.MLPCandidate
CandidateEvaluation = mlp_base.CandidateEvaluation


def candidate_specs(quick: bool) -> list[MLPCandidate]:
    if quick:
        return [
            MLPCandidate(
                name="pca128_tiny",
                hidden_units=(32, 8),
                dropout_rate=0.20,
                l2_strength=1e-3,
                learning_rate=5e-4,
                max_epochs=40,
                patience=6,
            )
        ]

    candidates = []
    for name, hidden_units, dropout_rate in (
        ("pca128_tiny", (32, 8), 0.20),
        ("pca128_compact", (64, 16), 0.25),
    ):
        for l2_strength in (1e-4, 1e-3, 1e-2):
            candidates.append(
                MLPCandidate(
                    name=name,
                    hidden_units=hidden_units,
                    dropout_rate=dropout_rate,
                    l2_strength=l2_strength,
                    learning_rate=5e-4,
                )
            )
    return candidates


def fit_projection(x_fit: np.ndarray) -> tuple[StandardScaler, PCA]:
    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_fit).astype(np.float32)
    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        n_oversamples=12,
        iterated_power=4,
        power_iteration_normalizer="auto",
        whiten=False,
        random_state=PCA_RANDOM_STATE,
    )
    pca.fit(x_fit_scaled)
    return scaler, pca


def transform_projection(
    x_values: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    scaled = scaler.transform(x_values).astype(np.float32)
    projected = pca.transform(scaled).astype(np.float32)
    if projected.shape != (len(x_values), PCA_COMPONENTS):
        raise RuntimeError(f"Proyeccion PCA inesperada: {projected.shape}.")
    if not np.isfinite(projected).all():
        raise RuntimeError("StandardScaler/PCA produjo NaN o Inf.")
    return projected


def train_with_early_stopping(
    candidate: MLPCandidate,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    validation_weights: np.ndarray,
    seed: int,
) -> tuple[int, float, int, int]:
    tf.keras.backend.clear_session()
    mlp_base.set_seed(seed)
    model = mlp_base.build_model(candidate, PCA_COMPONENTS)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=candidate.patience,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=0,
    )
    history = model.fit(
        x_train,
        y_train,
        sample_weight=train_weights,
        validation_data=(x_validation, y_validation, validation_weights),
        epochs=candidate.max_epochs,
        batch_size=candidate.batch_size,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=0,
    )
    validation_losses = np.asarray(history.history["val_loss"], dtype=float)
    best_epoch = int(np.argmin(validation_losses) + 1)
    return (
        best_epoch,
        float(validation_losses[best_epoch - 1]),
        len(validation_losses),
        int(model.count_params()),
    )


def refit_outer_and_predict(
    candidate: MLPCandidate,
    x_training: np.ndarray,
    y_training: np.ndarray,
    training_weights: np.ndarray,
    x_validation: np.ndarray,
    epochs: int,
    seed: int,
) -> np.ndarray:
    tf.keras.backend.clear_session()
    mlp_base.set_seed(seed)
    model = mlp_base.build_model(candidate, PCA_COMPONENTS)
    model.fit(
        x_training,
        y_training,
        sample_weight=training_weights,
        epochs=epochs,
        batch_size=candidate.batch_size,
        shuffle=True,
        verbose=0,
    )
    scores = model.predict(
        x_validation,
        batch_size=candidate.batch_size,
        verbose=0,
    ).reshape(-1)
    if scores.shape != (len(x_validation),) or not np.isfinite(scores).all():
        raise RuntimeError("La MLP genero scores OOF invalidos.")
    return scores


def evaluate_candidate_oof(
    data: common.DataView,
    candidate: MLPCandidate,
) -> tuple[CandidateEvaluation, pd.DataFrame]:
    started = time.perf_counter()
    metadata = data.train_metadata.reset_index(drop=True)
    folds = metadata["fold"].to_numpy(dtype=int)
    y_all = metadata["stage2_target"].to_numpy(dtype=np.float32)
    oof_scores = np.full(len(metadata), np.nan, dtype=float)
    training_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, float | int | str]] = []
    parameter_count: int | None = None

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        outer_validation_indices = np.flatnonzero(folds == fold)
        outer_training_indices = np.flatnonzero(folds != fold)
        seed = RANDOM_STATE + fold
        inner_train_indices, inner_validation_indices = (
            mlp_base.split_inner_training(
                outer_training_indices, y_all, seed
            )
        )

        inner_scaler, inner_pca = fit_projection(
            data.x_train[inner_train_indices]
        )
        x_inner_train = transform_projection(
            data.x_train[inner_train_indices], inner_scaler, inner_pca
        )
        x_inner_validation = transform_projection(
            data.x_train[inner_validation_indices], inner_scaler, inner_pca
        )
        inner_train_weights = common.compute_training_weights(
            metadata.iloc[inner_train_indices]
        ).astype(np.float32)
        inner_validation_weights = common.compute_training_weights(
            metadata.iloc[inner_validation_indices]
        ).astype(np.float32)
        best_epoch, best_loss, epochs_run, current_parameters = (
            train_with_early_stopping(
                candidate,
                x_inner_train,
                y_all[inner_train_indices],
                inner_train_weights,
                x_inner_validation,
                y_all[inner_validation_indices],
                inner_validation_weights,
                seed,
            )
        )
        if parameter_count is None:
            parameter_count = current_parameters
        elif parameter_count != current_parameters:
            raise RuntimeError("El numero de parametros cambio entre folds.")

        # Reajuste estricto con todo el TRAIN exterior. El fold OOF no decide
        # ni PCA, ni pesos, ni epocas.
        outer_scaler, outer_pca = fit_projection(
            data.x_train[outer_training_indices]
        )
        x_outer_train = transform_projection(
            data.x_train[outer_training_indices], outer_scaler, outer_pca
        )
        x_outer_validation = transform_projection(
            data.x_train[outer_validation_indices], outer_scaler, outer_pca
        )
        outer_weights = common.compute_training_weights(
            metadata.iloc[outer_training_indices]
        ).astype(np.float32)
        fold_scores = refit_outer_and_predict(
            candidate,
            x_outer_train,
            y_all[outer_training_indices],
            outer_weights,
            x_outer_validation,
            best_epoch,
            seed,
        )
        oof_scores[outer_validation_indices] = fold_scores
        native_metrics = common.binary_metrics(
            y_all[outer_validation_indices].astype(int), fold_scores, 0.5
        )

        inner_variance = float(np.sum(inner_pca.explained_variance_ratio_))
        outer_variance = float(np.sum(outer_pca.explained_variance_ratio_))
        variance_rows.extend(
            [
                {
                    "candidate_key": candidate.key,
                    "fold": fold,
                    "projection_fit": "inner_train_for_early_stopping",
                    "pca_components": PCA_COMPONENTS,
                    "cumulative_explained_variance_ratio": inner_variance,
                    "cumulative_explained_variance_percent": inner_variance * 100.0,
                },
                {
                    "candidate_key": candidate.key,
                    "fold": fold,
                    "projection_fit": "outer_train_for_oof_prediction",
                    "pca_components": PCA_COMPONENTS,
                    "cumulative_explained_variance_ratio": outer_variance,
                    "cumulative_explained_variance_percent": outer_variance * 100.0,
                },
            ]
        )
        training_rows.append(
            {
                "candidate_key": candidate.key,
                "outer_fold": fold,
                "seed": seed,
                "outer_train_recordings": len(outer_training_indices),
                "inner_train_recordings": len(inner_train_indices),
                "inner_validation_recordings": len(inner_validation_indices),
                "outer_validation_recordings": len(outer_validation_indices),
                "epochs_run_inner": epochs_run,
                "best_epoch_inner": best_epoch,
                "best_inner_val_loss": best_loss,
                "outer_refit_epochs": best_epoch,
                "outer_macro_f1_at_0p5": native_metrics["macro_f1"],
                "outer_wet_recall_at_0p5": native_metrics["wet_recall"],
            }
        )
        print(
            f"  Fold {fold}: best_epoch interno={best_epoch}, "
            f"varianza PCA outer={outer_variance * 100.0:.2f}%, "
            f"macro-F1 OOF@0.5={native_metrics['macro_f1']:.4f}, "
            f"wet-recall@0.5={native_metrics['wet_recall']:.4f}"
        )

    if not np.isfinite(oof_scores).all():
        raise RuntimeError(f"Predicciones OOF incompletas para {candidate.key}.")
    predictions = mlp_base.recording_predictions(metadata, oof_scores)
    y_true = predictions["y_true"].to_numpy(dtype=int)
    threshold = common.tune_threshold(y_true, oof_scores, 0.5)
    metrics = common.binary_metrics(y_true, oof_scores, threshold)
    metrics_fixed = common.binary_metrics(y_true, oof_scores, 0.5)
    predictions["y_pred_oof_threshold"] = (oof_scores >= threshold).astype(int)
    predictions["y_pred_fixed_0p5"] = (oof_scores >= 0.5).astype(int)
    predictions["candidate_key"] = candidate.key
    evaluation = CandidateEvaluation(
        candidate=candidate,
        threshold=threshold,
        metrics=metrics,
        metrics_fixed_0p5=metrics_fixed,
        oof_predictions=predictions,
        fold_metrics=mlp_base.fold_metrics(predictions, threshold),
        training_summary=pd.DataFrame(training_rows),
        parameter_count=int(parameter_count),
        elapsed_seconds=time.perf_counter() - started,
    )
    return evaluation, pd.DataFrame(variance_rows)


def candidate_results_frame(
    results: list[CandidateEvaluation],
) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "candidate_key": result.candidate.key,
                "pca_components": PCA_COMPONENTS,
                "hidden_units": "x".join(map(str, result.candidate.hidden_units)),
                "dropout_rate": result.candidate.dropout_rate,
                "l2_strength": result.candidate.l2_strength,
                "learning_rate": result.candidate.learning_rate,
                "parameter_count": result.parameter_count,
                "threshold_oof": result.threshold,
                "elapsed_seconds": result.elapsed_seconds,
                **{f"oof_tuned__{k}": v for k, v in result.metrics.items()},
                **{
                    f"fixed_0p5__{k}": v
                    for k, v in result.metrics_fixed_0p5.items()
                },
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "oof_tuned__macro_f1",
            "oof_tuned__balanced_accuracy",
            "oof_tuned__roc_auc",
        ],
        ascending=False,
    )


def fit_final_model(
    data: common.DataView,
    winner: CandidateEvaluation,
) -> tuple[tf.keras.Model, int, int, StandardScaler, PCA]:
    best_epochs = winner.training_summary["best_epoch_inner"].to_numpy(dtype=int)
    final_epochs = max(1, int(np.rint(np.median(best_epochs))))
    y_train = data.train_metadata["stage2_target"].to_numpy(dtype=np.float32)
    weights = common.compute_training_weights(data.train_metadata).astype(np.float32)

    scaler, pca = fit_projection(data.x_train)
    x_train_pca = transform_projection(data.x_train, scaler, pca)
    tf.keras.backend.clear_session()
    mlp_base.set_seed(RANDOM_STATE)
    core_model = mlp_base.build_model(winner.candidate, PCA_COMPONENTS)
    core_model.fit(
        x_train_pca,
        y_train,
        sample_weight=weights,
        epochs=final_epochs,
        batch_size=winner.candidate.batch_size,
        shuffle=True,
        verbose=2,
    )

    # StandardScaler y PCA son dos transformaciones afines consecutivas. Se
    # fusionan en una unica Dense no entrenable para reproducir exactamente
    # sklearn y evitar el epsilon interno de keras.layers.Normalization.
    raw_inputs = tf.keras.Input(
        shape=(data.x_train.shape[1],), name="wst_recording_raw"
    )
    pca_layer = tf.keras.layers.Dense(
        PCA_COMPONENTS,
        use_bias=True,
        trainable=False,
        name="standard_scaler_plus_pca128",
    )
    pca_values = pca_layer(raw_inputs)
    outputs = core_model(pca_values)
    deployment_model = tf.keras.Model(
        raw_inputs, outputs, name="wst_recording_pca128_mlp"
    )
    # ((x - scaler.mean_) / scaler.scale_ - pca.mean_) @ components_.T
    # se expresa como x @ combined_weights + combined_bias.
    combined_weights = (
        pca.components_.T / scaler.scale_[:, np.newaxis]
    ).astype(np.float32)
    combined_bias = np.matmul(
        (-scaler.mean_ / scaler.scale_) - pca.mean_,
        pca.components_.T,
    ).astype(np.float32)
    pca_layer.set_weights(
        [combined_weights, combined_bias]
    )
    return (
        deployment_model,
        final_epochs,
        int(core_model.count_params()),
        scaler,
        pca,
    )


def metric_row(
    split: str,
    threshold_policy: str,
    threshold: float,
    predictions: pd.DataFrame,
    winner: CandidateEvaluation,
    model_size_kb: float,
) -> dict[str, Any]:
    return {
        "split": split,
        "experiment": EXPERIMENT_KEY,
        "candidate_key": winner.candidate.key,
        "pooling_key": RECORDING_POOLING.key,
        "pca_components": PCA_COMPONENTS,
        "threshold_policy": threshold_policy,
        "threshold": threshold,
        "parameter_count": winner.parameter_count,
        "model_size_kb_float32_keras": model_size_kb,
        **common.binary_metrics(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["score"].to_numpy(dtype=float),
            threshold,
        ),
    }


def save_reference_comparison(
    result_dir: Path,
    winner: CandidateEvaluation,
    validation_metrics: dict[str, Any] | None,
) -> None:
    rows = [
        {
            "model": "wst_recording_pca128_mlp",
            "candidate": winner.candidate.key,
            "oof_macro_f1": winner.metrics["macro_f1"],
            "validation_macro_f1": (
                np.nan if validation_metrics is None else validation_metrics["macro_f1"]
            ),
            "validation_wet_recall": (
                np.nan if validation_metrics is None else validation_metrics["wet_recall"]
            ),
        }
    ]
    references = (
        (
            "wst_recording_pca128_logistic_regression",
            SCRIPT_DIR
            / "results_stage2_dry_wet_wavelet_scattering_recording_linear"
            / DEFAULT_PRESET
            / "full"
            / "logistic_regression"
            / "metrics_summary.csv",
        ),
        (
            "wst_recording_no_pca_mlp",
            SCRIPT_DIR
            / "results_stage2_dry_wet_wavelet_scattering_recording_mlp"
            / DEFAULT_PRESET
            / "full"
            / "metrics_summary.csv",
        ),
    )
    for reference_name, path in references:
        if not path.is_file():
            continue
        data = pd.read_csv(path)
        split_column = "dataset" if "dataset" in data.columns else "split"
        oof = data[
            (data[split_column] == "train_oof")
            & data["threshold_policy"].astype(str).isin(
                ["oof_tuned", "oof_tuned_frozen"]
            )
        ]
        validation = data[
            (data[split_column] == "validation")
            & (data["threshold_policy"] == "oof_tuned_frozen")
        ]
        if oof.empty or validation.empty:
            continue
        rows.append(
            {
                "model": reference_name,
                "candidate": validation.iloc[0].get("candidate_key", ""),
                "oof_macro_f1": oof.iloc[0]["macro_f1"],
                "validation_macro_f1": validation.iloc[0]["macro_f1"],
                "validation_wet_recall": validation.iloc[0]["wet_recall"],
            }
        )
    pd.DataFrame(rows).to_csv(
        result_dir / "reference_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )


def train(
    data: common.DataView,
    extraction_configuration: pd.DataFrame,
    preset: str,
    quick: bool,
) -> None:
    mode = "quick" if quick else "full"
    result_dir = RESULTS_ROOT / preset / mode
    model_dir = MODELS_ROOT / preset / mode
    graph_dir = GRAPHS_ROOT / preset / mode
    result_dir.mkdir(parents=True, exist_ok=True)

    results: list[CandidateEvaluation] = []
    variance_frames = []
    print("\n" + "=" * 78)
    print("CV - WST RECORDING + PCA128 + MLP LIGERA")
    print("=" * 78)
    specs = candidate_specs(quick)
    for index, candidate in enumerate(specs, start=1):
        print(f"\n[{index}/{len(specs)}] Candidato: {candidate.key}")
        result, variance = evaluate_candidate_oof(data, candidate)
        results.append(result)
        variance_frames.append(variance)
        print(
            f"{candidate.key} | macro-F1={result.metrics['macro_f1']:.4f} | "
            f"bal-acc={result.metrics['balanced_accuracy']:.4f} | "
            f"AUC={result.metrics['roc_auc']:.4f} | "
            f"params={result.parameter_count}"
        )

    winner = max(results, key=mlp_base.selection_key)
    candidate_results_frame(results).to_csv(
        result_dir / "candidate_cv_results.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(variance_frames, ignore_index=True).to_csv(
        result_dir / "pca128_explained_variance_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        [result.training_summary for result in results], ignore_index=True
    ).to_csv(
        result_dir / "cv_training_summary.csv", index=False, encoding="utf-8-sig"
    )
    winner.oof_predictions.to_csv(
        result_dir / "best_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    winner.fold_metrics.to_csv(
        result_dir / "best_cv_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )

    oof_rows = [
        metric_row(
            "train_oof", "fixed_0p5", 0.5,
            winner.oof_predictions, winner, np.nan
        ),
        metric_row(
            "train_oof", "oof_tuned", winner.threshold,
            winner.oof_predictions, winner, np.nan
        ),
    ]
    if quick:
        pd.DataFrame(oof_rows).to_csv(
            result_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig"
        )
        save_reference_comparison(result_dir, winner, None)
        print("\nPrueba rapida PCA128 completada.")
        print(f"Mejor OOF: {winner.candidate.key}")
        print(f"Macro-F1 OOF: {winner.metrics['macro_f1']:.4f}")
        print("VALIDATION no se ha evaluado.")
        print(f"Resultados: {result_dir}")
        return

    print("\nAjustando StandardScaler, PCA128 y MLP finales con todo TRAIN...")
    final_model, final_epochs, core_parameters, _, final_pca = fit_final_model(
        data, winner
    )
    validation_scores = final_model.predict(
        data.x_validation,
        batch_size=winner.candidate.batch_size,
        verbose=0,
    ).reshape(-1)
    validation_predictions = mlp_base.recording_predictions(
        data.validation_metadata.reset_index(drop=True), validation_scores
    )
    validation_predictions["y_pred_oof_threshold"] = (
        validation_scores >= winner.threshold
    ).astype(int)
    validation_predictions["y_pred_fixed_0p5"] = (
        validation_scores >= 0.5
    ).astype(int)
    validation_predictions["candidate_key"] = winner.candidate.key
    validation_predictions.to_csv(
        result_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "wst_recording_pca128_mlp.keras"
    final_model.save(model_path)
    model_size_kb = model_path.stat().st_size / 1024.0
    for row in oof_rows:
        row["model_size_kb_float32_keras"] = model_size_kb
    validation_rows = [
        metric_row(
            "validation", "fixed_0p5", 0.5,
            validation_predictions, winner, model_size_kb
        ),
        metric_row(
            "validation", "oof_tuned_frozen", winner.threshold,
            validation_predictions, winner, model_size_kb
        ),
    ]
    pd.DataFrame([*oof_rows, *validation_rows]).to_csv(
        result_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig"
    )

    final_variance = float(np.sum(final_pca.explained_variance_ratio_))
    configuration = {
        "preset": preset,
        "unit_of_training": "recording",
        "pooling_key": RECORDING_POOLING.key,
        "input_feature_count": data.x_train.shape[1],
        "standard_scaler": "inside_each_fold_and_embedded_in_final_model",
        "pca_components": PCA_COMPONENTS,
        "pca_solver": "randomized",
        "pca_whiten": False,
        "final_pca_explained_variance_ratio": final_variance,
        "final_pca_explained_variance_percent": final_variance * 100.0,
        "smote_used": False,
        "class_balance": "balanced_sample_weight_per_recording",
        "inner_validation_fraction": mlp_base.INNER_VALIDATION_SIZE,
        "inner_split_purpose": "select_epochs_only",
        "outer_refit": "all_outer_train_for_selected_inner_epochs",
        "selection_metric": "train_oof_macro_f1",
        "winner": winner.candidate.key,
        "threshold_policy": "oof_tuned_frozen",
        "threshold": winner.threshold,
        "final_epochs_from_cv_median": final_epochs,
        "core_trainable_parameter_count": core_parameters,
        "saved_model_parameter_count_including_preprocessing": int(
            final_model.count_params()
        ),
        "model_size_kb_float32_keras": model_size_kb,
        "validation_used_for_selection": False,
        "test_read": False,
        **{f"mlp_{key}": value for key, value in asdict(winner.candidate).items()},
        **{
            f"wst_{key}": value
            for key, value in extraction_configuration.iloc[0].to_dict().items()
        },
    }
    pd.DataFrame([configuration]).to_csv(
        result_dir / "experiment_configuration.csv", index=False, encoding="utf-8-sig"
    )
    validation_metrics = validation_rows[-1]
    save_reference_comparison(result_dir, winner, validation_metrics)
    graph_path = graph_dir / "validation_wst_recording_pca128_mlp.png"
    mlp_base.create_validation_graph(
        validation_predictions,
        winner.threshold,
        winner.candidate.key,
        graph_path,
    )

    print("\n" + "=" * 78)
    print("RESULTADO WST RECORDING + PCA128 + MLP LIGERA")
    print("=" * 78)
    print(f"Mejor configuracion: {winner.candidate.key}")
    print(f"Varianza explicada PCA128 final: {final_variance * 100.0:.2f}%")
    print(f"Umbral OOF congelado: {winner.threshold:.6f}")
    print(f"Macro-F1 OOF: {winner.metrics['macro_f1']:.4f}")
    print(f"Macro-F1 validation: {validation_metrics['macro_f1']:.4f}")
    print(
        "Recalls validation dry/wet: "
        f"{validation_metrics['dry_recall']:.4f} / "
        f"{validation_metrics['wet_recall']:.4f}"
    )
    print(f"Parametros entrenables de la MLP: {core_parameters}")
    print(f"Modelo Keras con scaler y PCA: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print(f"Grafica: {graph_path}")
    print("TEST permanece reservado.")


def print_check(data: common.DataView, quick: bool) -> None:
    train_counts = data.train_metadata["stage2_target"].value_counts()
    validation_counts = data.validation_metadata["stage2_target"].value_counts()
    print("=" * 78)
    print("CHECK - WST RECORDING + PCA128 + MLP")
    print("=" * 78)
    print(f"X TRAIN original: {data.x_train.shape} {data.x_train.dtype}")
    print(f"X VALIDATION original: {data.x_validation.shape} {data.x_validation.dtype}")
    print(f"Proyeccion: {data.x_train.shape[1]} -> {PCA_COMPONENTS}")
    print(f"TRAIN dry/wet: {train_counts[0]} / {train_counts[1]}")
    print(f"VALIDATION dry/wet: {validation_counts[0]} / {validation_counts[1]}")
    print(f"Candidatos: {len(candidate_specs(quick))}")
    for candidate in candidate_specs(quick):
        tf.keras.backend.clear_session()
        model = mlp_base.build_model(candidate, PCA_COMPONENTS)
        print(f"  {candidate.key}: {model.count_params()} parametros entrenables")
    print("Scaler y PCA128 se ajustan dentro de cada bloque de entrenamiento.")
    print("El 15% interno selecciona epocas; despues se reajusta todo TRAIN exterior.")
    print("Sin SMOTE. VALIDATION no selecciona. TEST no se procesa.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WST recording StandardScaler PCA128 con MLP ligeras."
    )
    parser.add_argument("--action", choices=["check", "train"], default="check")
    parser.add_argument("--preset", choices=PRESETS, default=DEFAULT_PRESET)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Un candidato; no evalua VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST RECORDING + STANDARD SCALER + PCA128 + MLP")
    print("=" * 78)
    print(f"Preset: {args.preset} | Accion: {args.action} | Quick: {args.quick}")
    print("TEST no sera leido ni procesado.")
    event_data, extraction_configuration = event_baseline.load_wavelet_data(args.preset)
    data = mlp_base.build_recording_data(event_data)
    print_check(data, args.quick)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(data, extraction_configuration, args.preset, args.quick)


if __name__ == "__main__":
    main()
