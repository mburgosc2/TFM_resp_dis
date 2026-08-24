"""Stage 2 dry/wet: WST recording-level con MLP ligeras.

Reutiliza los 644 coeficientes Wavelet Scattering ya extraidos por evento y
construye una fila por grabacion mediante mean+std+max (1932 features). Compara
dos arquitecturas pequenas, 32->8 y 64->16, con dos intensidades de L2.

El protocolo evita usar VALIDATION para tomar decisiones:

* Los cinco folds exteriores de TRAIN generan predicciones OOF.
* Dentro del bloque de ajuste de cada fold se crea una particion estratificada
  independiente para early stopping.
* StandardScaler se ajusta solo con el entrenamiento interno.
* Arquitectura y umbral se seleccionan solo con las predicciones OOF.
* En modo completo se ajusta el ganador con todo TRAIN y se evalua una unica
  vez en VALIDATION. TEST no se lee ni se procesa.

No usa PCA ni SMOTE. La unidad de entrenamiento es la grabacion, de forma que
cada original_uuid aporta una sola fila y una sola etiqueta.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_rf as recording
import train_stage2_dry_wet_wavelet_scattering_rf as event_baseline


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (
    SCRIPT_DIR / "results_stage2_dry_wet_wavelet_scattering_recording_mlp"
)
MODELS_ROOT = (
    SCRIPT_DIR / "models_stage2_dry_wet_wavelet_scattering_recording_mlp"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wavelet_scattering_recording_mlp"
)

RANDOM_STATE = 42
PRESETS = event_baseline.PRESETS
DEFAULT_PRESET = "paper_q8_q1_t500_full"
EXPECTED_FEATURE_COUNT = 1932
INNER_VALIDATION_SIZE = 0.15
EXPERIMENT_KEY = "wavelet_scattering_recording_mlp"
RECORDING_POOLING = next(
    spec for spec in recording.POOLING_SPECS if spec.key == "mean_std_max"
)


@dataclass(frozen=True)
class MLPCandidate:
    name: str
    hidden_units: tuple[int, ...]
    dropout_rate: float
    l2_strength: float
    learning_rate: float
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10

    @property
    def key(self) -> str:
        hidden = "x".join(map(str, self.hidden_units))
        dropout = f"{self.dropout_rate:g}".replace(".", "p")
        l2_value = f"{self.l2_strength:g}".replace(".", "p")
        learning_rate = f"{self.learning_rate:g}".replace(".", "p")
        return (
            f"{self.name}__h{hidden}__drop{dropout}__"
            f"l2{l2_value}__lr{learning_rate}"
        )


@dataclass
class CandidateEvaluation:
    candidate: MLPCandidate
    threshold: float
    metrics: dict[str, float | int]
    metrics_fixed_0p5: dict[str, float | int]
    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    training_summary: pd.DataFrame
    parameter_count: int
    elapsed_seconds: float


@tf.keras.utils.register_keras_serializable(package="TFM")
class FixedStandardScaler(tf.keras.layers.Layer):
    """StandardScaler exacto y serializable, sin recorte epsilon de varianza."""

    def __init__(
        self,
        mean_values: list[float] | np.ndarray,
        scale_values: list[float] | np.ndarray,
        **kwargs: Any,
    ) -> None:
        kwargs["trainable"] = False
        super().__init__(**kwargs)
        self.mean_values = np.asarray(mean_values, dtype=np.float32)
        self.scale_values = np.asarray(scale_values, dtype=np.float32)
        if self.mean_values.shape != self.scale_values.shape:
            raise ValueError("Mean y scale del StandardScaler no coinciden.")
        if np.any(self.scale_values <= 0) or not np.isfinite(
            self.scale_values
        ).all():
            raise ValueError("Scale invalido para FixedStandardScaler.")

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        dimension = int(input_shape[-1])
        if self.mean_values.shape != (dimension,):
            raise ValueError(
                "Dimension de FixedStandardScaler incompatible con la entrada."
            )
        self.fixed_mean = self.add_weight(
            name="mean",
            shape=(dimension,),
            initializer=tf.keras.initializers.Constant(self.mean_values),
            trainable=False,
        )
        self.fixed_scale = self.add_weight(
            name="scale",
            shape=(dimension,),
            initializer=tf.keras.initializers.Constant(self.scale_values),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return (inputs - self.fixed_mean) / self.fixed_scale

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "mean_values": self.mean_values.tolist(),
                "scale_values": self.scale_values.tolist(),
            }
        )
        return config


def candidate_specs(quick: bool) -> list[MLPCandidate]:
    if quick:
        return [
            MLPCandidate(
                name="tiny",
                hidden_units=(32, 8),
                dropout_rate=0.25,
                l2_strength=1e-3,
                learning_rate=5e-4,
                max_epochs=25,
                patience=5,
            )
        ]

    candidates = []
    for name, hidden_units, dropout_rate in (
        ("tiny", (32, 8), 0.25),
        ("compact", (64, 16), 0.30),
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


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_recording_data(event_data: common.DataView) -> common.DataView:
    data = recording.build_recording_view(event_data, RECORDING_POOLING)
    if data.x_train.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Se esperaban 1932 features WST recording mean+std+max, "
            f"pero se obtuvo {data.x_train.shape}."
        )
    return data


def build_model(
    candidate: MLPCandidate,
    input_dimension: int = EXPECTED_FEATURE_COUNT,
) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.l2(candidate.l2_strength)
    inputs = tf.keras.Input(shape=(input_dimension,), name="wst_recording_scaled")
    x = inputs
    for layer_index, units in enumerate(candidate.hidden_units, start=1):
        x = tf.keras.layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizer,
            kernel_initializer="he_normal",
            name=f"dense_{layer_index}",
        )(x)
        x = tf.keras.layers.Dropout(
            candidate.dropout_rate,
            name=f"dropout_{layer_index}",
        )(x)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="wet_probability",
    )(x)
    model = tf.keras.Model(inputs, outputs, name=candidate.name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=candidate.learning_rate
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        weighted_metrics=[],
    )
    return model


def recording_predictions(
    metadata: pd.DataFrame,
    scores: np.ndarray,
) -> pd.DataFrame:
    if scores.shape != (len(metadata),):
        raise ValueError("Scores y filas recording tienen longitudes distintas.")
    predictions = metadata[
        [
            "original_uuid",
            "stage2_target",
            "cough_type",
            "cough_type_consensus",
            "fold",
            "event_count",
        ]
    ].copy()
    predictions = predictions.rename(columns={"stage2_target": "y_true"})
    predictions["score"] = scores
    if predictions["original_uuid"].duplicated().any():
        raise RuntimeError("La vista recording contiene UUID duplicados.")
    if not np.isfinite(predictions["score"]).all():
        raise RuntimeError("Se generaron scores no finitos.")
    return predictions


def fold_metrics(
    predictions: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for fold, group in predictions.groupby("fold", sort=True):
        rows.append(
            {
                "fold": int(fold),
                "recording_count": len(group),
                "threshold": threshold,
                **common.binary_metrics(
                    group["y_true"].to_numpy(dtype=int),
                    group["score"].to_numpy(dtype=float),
                    threshold,
                ),
            }
        )
    return pd.DataFrame(rows)


def split_inner_training(
    outer_training_indices: np.ndarray,
    y_all: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    inner_train, inner_validation = train_test_split(
        outer_training_indices,
        test_size=INNER_VALIDATION_SIZE,
        stratify=y_all[outer_training_indices],
        random_state=seed,
    )
    return np.asarray(inner_train, dtype=int), np.asarray(
        inner_validation, dtype=int
    )


def evaluate_candidate_oof(
    data: common.DataView,
    candidate: MLPCandidate,
) -> CandidateEvaluation:
    started = time.perf_counter()
    metadata = data.train_metadata.reset_index(drop=True)
    folds = metadata["fold"].to_numpy(dtype=int)
    y_all = metadata["stage2_target"].to_numpy(dtype=np.float32)
    oof_scores = np.full(len(metadata), np.nan, dtype=float)
    training_rows: list[dict[str, Any]] = []
    parameter_count: int | None = None

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        outer_validation_indices = np.flatnonzero(folds == fold)
        outer_training_indices = np.flatnonzero(folds != fold)
        seed = RANDOM_STATE + fold
        inner_train_indices, inner_validation_indices = split_inner_training(
            outer_training_indices, y_all, seed
        )

        scaler = StandardScaler()
        x_inner_train = scaler.fit_transform(
            data.x_train[inner_train_indices]
        ).astype(np.float32)
        x_inner_validation = scaler.transform(
            data.x_train[inner_validation_indices]
        ).astype(np.float32)
        x_outer_validation = scaler.transform(
            data.x_train[outer_validation_indices]
        ).astype(np.float32)

        metadata_inner_train = metadata.iloc[inner_train_indices]
        metadata_inner_validation = metadata.iloc[inner_validation_indices]
        train_weights = common.compute_training_weights(
            metadata_inner_train
        ).astype(np.float32)
        validation_weights = common.compute_training_weights(
            metadata_inner_validation
        ).astype(np.float32)

        tf.keras.backend.clear_session()
        set_seed(seed)
        model = build_model(candidate, data.x_train.shape[1])
        current_parameters = int(model.count_params())
        if parameter_count is None:
            parameter_count = current_parameters
        elif parameter_count != current_parameters:
            raise RuntimeError("El numero de parametros cambio entre folds.")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=candidate.patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=0,
        )
        history = model.fit(
            x_inner_train,
            y_all[inner_train_indices],
            sample_weight=train_weights,
            validation_data=(
                x_inner_validation,
                y_all[inner_validation_indices],
                validation_weights,
            ),
            epochs=candidate.max_epochs,
            batch_size=candidate.batch_size,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=0,
        )
        fold_scores = model.predict(
            x_outer_validation,
            batch_size=candidate.batch_size,
            verbose=0,
        ).reshape(-1)
        oof_scores[outer_validation_indices] = fold_scores

        validation_losses = np.asarray(history.history["val_loss"], dtype=float)
        best_epoch = int(np.argmin(validation_losses) + 1)
        fold_native = common.binary_metrics(
            y_all[outer_validation_indices].astype(int),
            fold_scores,
            0.5,
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
                "epochs_run": len(validation_losses),
                "best_epoch": best_epoch,
                "best_inner_val_loss": float(validation_losses[best_epoch - 1]),
                "outer_macro_f1_at_0p5": fold_native["macro_f1"],
                "outer_wet_recall_at_0p5": fold_native["wet_recall"],
            }
        )
        print(
            f"  Fold {fold}: best_epoch={best_epoch}, "
            f"inner_val_loss={validation_losses[best_epoch - 1]:.5f}, "
            f"outer_macro-F1@0.5={fold_native['macro_f1']:.4f}, "
            f"outer_wet_recall@0.5={fold_native['wet_recall']:.4f}"
        )
        del model

    if not np.isfinite(oof_scores).all():
        raise RuntimeError(f"Predicciones OOF incompletas para {candidate.key}.")

    predictions = recording_predictions(metadata, oof_scores)
    y_true = predictions["y_true"].to_numpy(dtype=int)
    threshold = common.tune_threshold(y_true, oof_scores, 0.5)
    metrics = common.binary_metrics(y_true, oof_scores, threshold)
    metrics_fixed = common.binary_metrics(y_true, oof_scores, 0.5)
    predictions["y_pred_oof_threshold"] = (oof_scores >= threshold).astype(int)
    predictions["y_pred_fixed_0p5"] = (oof_scores >= 0.5).astype(int)
    predictions["candidate_key"] = candidate.key

    return CandidateEvaluation(
        candidate=candidate,
        threshold=threshold,
        metrics=metrics,
        metrics_fixed_0p5=metrics_fixed,
        oof_predictions=predictions,
        fold_metrics=fold_metrics(predictions, threshold),
        training_summary=pd.DataFrame(training_rows),
        parameter_count=int(parameter_count),
        elapsed_seconds=time.perf_counter() - started,
    )


def selection_key(result: CandidateEvaluation) -> tuple[float, ...]:
    return (
        float(result.metrics["macro_f1"]),
        float(result.metrics["balanced_accuracy"]),
        float(result.metrics["roc_auc"]),
        -float(result.parameter_count),
    )


def candidate_results_frame(
    results: list[CandidateEvaluation],
) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "candidate_key": result.candidate.key,
                "hidden_units": "x".join(
                    map(str, result.candidate.hidden_units)
                ),
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
) -> tuple[tf.keras.Model, int, int]:
    best_epochs = winner.training_summary["best_epoch"].to_numpy(dtype=int)
    final_epochs = max(1, int(np.rint(np.median(best_epochs))))
    y_train = data.train_metadata["stage2_target"].to_numpy(dtype=np.float32)
    weights = common.compute_training_weights(data.train_metadata).astype(
        np.float32
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(data.x_train).astype(np.float32)

    tf.keras.backend.clear_session()
    set_seed(RANDOM_STATE)
    core_model = build_model(winner.candidate, data.x_train.shape[1])
    core_model.fit(
        x_train_scaled,
        y_train,
        sample_weight=weights,
        epochs=final_epochs,
        batch_size=winner.candidate.batch_size,
        shuffle=True,
        verbose=2,
    )

    raw_inputs = tf.keras.Input(
        shape=(data.x_train.shape[1],), name="wst_recording_raw"
    )
    normalization = FixedStandardScaler(
        scaler.mean_,
        scaler.scale_,
        name="standard_scaler",
    )
    outputs = core_model(normalization(raw_inputs))
    deployment_model = tf.keras.Model(
        raw_inputs,
        outputs,
        name="wst_recording_mlp_with_scaler",
    )
    return deployment_model, final_epochs, int(core_model.count_params())


def create_validation_graph(
    predictions: pd.DataFrame,
    threshold: float,
    candidate_key: str,
    output_path: Path,
) -> None:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    y_pred = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_percent = matrix / matrix.sum(axis=1, keepdims=True) * 100.0

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[0].text(
                column,
                row,
                f"{matrix[row, column]}\n({row_percent[row, column]:.1f}%)",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axes[0].set_xticks([0, 1], ["Dry", "Wet"])
    axes[0].set_yticks([0, 1], ["Dry", "Wet"])
    axes[0].set_xlabel("Prediccion")
    axes[0].set_ylabel("Etiqueta real")
    axes[0].set_title("Matriz de confusion")
    figure.colorbar(image, ax=axes[0], fraction=0.046)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    axes[1].plot(
        false_positive_rate,
        true_positive_rate,
        label=f"AUC={roc_auc_score(y_true, scores):.3f}",
    )
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC recording-level")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    axes[2].plot(
        recall,
        precision,
        label=f"AP wet={average_precision_score(y_true, scores):.3f}",
    )
    axes[2].axhline(
        float(np.mean(y_true == 1)),
        linestyle="--",
        color="grey",
        label="Baseline",
    )
    axes[2].set_xlabel("Recall wet")
    axes[2].set_ylabel("Precision wet")
    axes[2].set_title("Precision-recall")
    axes[2].legend()
    axes[2].grid(alpha=0.2)

    figure.suptitle(
        f"WST recording MLP - VALIDATION - {candidate_key} - umbral={threshold:.4f}"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


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
            "model": "wst_recording_mlp",
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
    reference_path = (
        SCRIPT_DIR
        / "results_stage2_dry_wet_wavelet_scattering_recording_linear"
        / DEFAULT_PRESET
        / "full"
        / "logistic_regression"
        / "metrics_summary.csv"
    )
    if reference_path.is_file():
        reference = pd.read_csv(reference_path)
        split_column = "dataset" if "dataset" in reference.columns else "split"
        oof = reference[
            (reference[split_column] == "train_oof")
            & (reference["threshold_policy"] == "oof_tuned_frozen")
        ]
        validation = reference[
            (reference[split_column] == "validation")
            & (reference["threshold_policy"] == "oof_tuned_frozen")
        ]
        if not oof.empty and not validation.empty:
            rows.append(
                {
                    "model": "wst_recording_logistic_regression_reference",
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

    results = []
    print("\n" + "=" * 78)
    print("CV - WST RECORDING + MLP LIGERA")
    print("=" * 78)
    for index, candidate in enumerate(candidate_specs(quick), start=1):
        print(
            f"\n[{index}/{len(candidate_specs(quick))}] Candidato: {candidate.key}"
        )
        result = evaluate_candidate_oof(data, candidate)
        results.append(result)
        print(
            f"{candidate.key} | macro-F1={result.metrics['macro_f1']:.4f} | "
            f"bal-acc={result.metrics['balanced_accuracy']:.4f} | "
            f"AUC={result.metrics['roc_auc']:.4f} | "
            f"params={result.parameter_count}"
        )

    winner = max(results, key=selection_key)
    candidate_results_frame(results).to_csv(
        result_dir / "candidate_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        [result.training_summary for result in results], ignore_index=True
    ).to_csv(
        result_dir / "cv_training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    winner.oof_predictions.to_csv(
        result_dir / "best_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    winner.fold_metrics.to_csv(
        result_dir / "best_cv_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof_rows = [
        metric_row(
            "train_oof",
            "fixed_0p5",
            0.5,
            winner.oof_predictions,
            winner,
            np.nan,
        ),
        metric_row(
            "train_oof",
            "oof_tuned",
            winner.threshold,
            winner.oof_predictions,
            winner,
            np.nan,
        ),
    ]
    if quick:
        pd.DataFrame(oof_rows).to_csv(
            result_dir / "metrics_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_reference_comparison(result_dir, winner, None)
        print("\nPrueba rapida completada.")
        print(f"Mejor OOF: {winner.candidate.key}")
        print(f"Macro-F1 OOF: {winner.metrics['macro_f1']:.4f}")
        print("VALIDATION no se ha evaluado.")
        print(f"Resultados: {result_dir}")
        return

    print("\nAjustando MLP final exclusivamente con todo TRAIN...")
    final_model, final_epochs, core_parameter_count = fit_final_model(data, winner)
    validation_scores = final_model.predict(
        data.x_validation,
        batch_size=winner.candidate.batch_size,
        verbose=0,
    ).reshape(-1)
    validation_predictions = recording_predictions(
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
        result_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "wst_recording_mlp_with_scaler.keras"
    final_model.save(model_path)
    model_size_kb = model_path.stat().st_size / 1024.0

    for row in oof_rows:
        row["model_size_kb_float32_keras"] = model_size_kb
    validation_rows = [
        metric_row(
            "validation",
            "fixed_0p5",
            0.5,
            validation_predictions,
            winner,
            model_size_kb,
        ),
        metric_row(
            "validation",
            "oof_tuned_frozen",
            winner.threshold,
            validation_predictions,
            winner,
            model_size_kb,
        ),
    ]
    pd.DataFrame([*oof_rows, *validation_rows]).to_csv(
        result_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    configuration = {
        "preset": preset,
        "unit_of_training": "recording",
        "pooling_key": RECORDING_POOLING.key,
        "pooling_statistics": "|".join(RECORDING_POOLING.statistics),
        "input_feature_count": data.x_train.shape[1],
        "standard_scaler": "inside_each_fold_and_embedded_in_final_model",
        "pca_used": False,
        "smote_used": False,
        "class_balance": "balanced_sample_weight_per_recording",
        "outer_cv_folds": 5,
        "inner_validation_fraction": INNER_VALIDATION_SIZE,
        "inner_split_purpose": "early_stopping_only",
        "selection_metric": "train_oof_macro_f1",
        "winner": winner.candidate.key,
        "threshold_policy": "oof_tuned_frozen",
        "threshold": winner.threshold,
        "final_epochs_from_cv_median": final_epochs,
        "core_parameter_count": core_parameter_count,
        "saved_model_parameter_count_including_normalizer": int(
            final_model.count_params()
        ),
        "model_size_kb_float32_keras": model_size_kb,
        "validation_used_for_selection": False,
        "test_read": False,
        **{
            f"mlp_{key}": value
            for key, value in asdict(winner.candidate).items()
        },
        **{
            f"wst_{key}": value
            for key, value in extraction_configuration.iloc[0].to_dict().items()
        },
    }
    pd.DataFrame([configuration]).to_csv(
        result_dir / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_metrics = validation_rows[-1]
    save_reference_comparison(result_dir, winner, validation_metrics)
    graph_path = graph_dir / "validation_wst_recording_mlp.png"
    create_validation_graph(
        validation_predictions,
        winner.threshold,
        winner.candidate.key,
        graph_path,
    )

    print("\n" + "=" * 78)
    print("RESULTADO WST RECORDING + MLP LIGERA")
    print("=" * 78)
    print(f"Mejor configuracion: {winner.candidate.key}")
    print(f"Umbral OOF congelado: {winner.threshold:.6f}")
    print(f"Macro-F1 OOF: {winner.metrics['macro_f1']:.4f}")
    print(f"Macro-F1 validation: {validation_metrics['macro_f1']:.4f}")
    print(
        "Recalls validation dry/wet: "
        f"{validation_metrics['dry_recall']:.4f} / "
        f"{validation_metrics['wet_recall']:.4f}"
    )
    print(f"Parametros entrenables: {core_parameter_count}")
    print(f"Modelo Keras con scaler: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print(f"Grafica: {graph_path}")
    print("TEST permanece reservado.")


def print_check(data: common.DataView, quick: bool) -> None:
    train_counts = data.train_metadata["stage2_target"].value_counts()
    validation_counts = data.validation_metadata["stage2_target"].value_counts()
    print("=" * 78)
    print("CHECK - WST RECORDING + MLP LIGERA")
    print("=" * 78)
    print(f"X TRAIN: {data.x_train.shape} {data.x_train.dtype}")
    print(f"X VALIDATION: {data.x_validation.shape} {data.x_validation.dtype}")
    print(f"TRAIN dry/wet: {train_counts[0]} / {train_counts[1]}")
    print(f"VALIDATION dry/wet: {validation_counts[0]} / {validation_counts[1]}")
    print(f"Pooling: {RECORDING_POOLING.key}")
    print(f"Candidatos: {len(candidate_specs(quick))}")
    for candidate in candidate_specs(quick):
        tf.keras.backend.clear_session()
        model = build_model(candidate, data.x_train.shape[1])
        print(f"  {candidate.key}: {model.count_params()} parametros")
    print("StandardScaler se ajusta dentro de cada fold.")
    print("Early stopping usa un 15% interno; no usa el fold OOF exterior.")
    print("Sin PCA ni SMOTE. VALIDATION no selecciona. TEST no se procesa.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WST recording mean+std+max con MLP ligeras."
    )
    parser.add_argument("--action", choices=["check", "train"], default="check")
    parser.add_argument("--preset", choices=PRESETS, default=DEFAULT_PRESET)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Un candidato y pocas epocas; no evalua VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST RECORDING + MLP LIGERA")
    print("=" * 78)
    print(f"Preset: {args.preset} | Accion: {args.action} | Quick: {args.quick}")
    print("TEST no sera leido ni procesado.")
    event_data, extraction_configuration = event_baseline.load_wavelet_data(
        args.preset
    )
    data = build_recording_data(event_data)
    print_check(data, args.quick)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(data, extraction_configuration, args.preset, args.quick)


if __name__ == "__main__":
    main()
