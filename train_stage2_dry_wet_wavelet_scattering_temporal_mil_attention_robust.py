"""Stage 2 dry/wet: WST temporal + MIL attention robusto.

Este experimento conserva una unica etiqueta y una unica perdida por
``original_uuid``. Reutiliza las bolsas WST temporales del experimento MIL
original, pero:

* compara cuatro arquitecturas gated-attention pequenas;
* usa tres inner splits por outer fold para seleccionar las epocas;
* usa la mediana de las tres epocas para el refit de cada outer fold;
* registra loss, precision, recall, macro-F1, AUC y learning rate por epoca;
* selecciona arquitectura y umbral exclusivamente con OOF de TRAIN;
* evalua VALIDATION una sola vez para el ganador en modo full;
* nunca lee ni procesa TEST.

Las metricas por epoca pertenecen al inner-validation y se calculan con
umbral fijo 0.5. No se optimiza un umbral en cada epoca, para evitar adaptar
las curvas al propio inner-validation.
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
from sklearn.model_selection import train_test_split

import train_stage2_dry_wet_wavelet_scattering_temporal_mil_cnn as base


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (
    SCRIPT_DIR
    / "results_stage2_dry_wet_wavelet_scattering_temporal_mil_attention_robust"
)
MODELS_ROOT = (
    SCRIPT_DIR
    / "models_stage2_dry_wet_wavelet_scattering_temporal_mil_attention_robust"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wavelet_scattering_temporal_mil_attention_robust"
)

PRESETS = base.PRESETS
DEFAULT_PRESET = base.DEFAULT_PRESET
INNER_REPEATS = 3
SELECTION_TOLERANCE = 0.005
EXPERIMENT_KEY = "wavelet_scattering_temporal_mil_attention_robust"


@dataclass(frozen=True)
class RobustMILCandidate(base.MILCandidate):
    encoder_spatial_dropout_rate: float = 0.0
    embedding_dropout_rate: float = 0.0

    @property
    def key(self) -> str:
        base_key = super().key
        spatial = f"{self.encoder_spatial_dropout_rate:g}".replace(".", "p")
        embedding = f"{self.embedding_dropout_rate:g}".replace(".", "p")
        return f"{base_key}__spdrop{spatial}__embdrop{embedding}"


def candidate_specs(quick: bool) -> list[RobustMILCandidate]:
    reference = RobustMILCandidate(
        pooling="attention",
        projection_channels=16,
        embedding_units=8,
        attention_units=8,
        dropout_rate=0.20,
        l2_strength=1e-3,
        learning_rate=5e-4,
        max_epochs=120,
        patience=12,
    )
    if quick:
        return [reference]
    return [
        reference,
        RobustMILCandidate(
            pooling="attention",
            projection_channels=16,
            embedding_units=8,
            attention_units=8,
            dropout_rate=0.30,
            l2_strength=1e-3,
            learning_rate=5e-4,
            max_epochs=120,
            patience=12,
            encoder_spatial_dropout_rate=0.15,
            embedding_dropout_rate=0.20,
        ),
        RobustMILCandidate(
            pooling="attention",
            projection_channels=16,
            embedding_units=8,
            attention_units=8,
            dropout_rate=0.35,
            l2_strength=3e-3,
            learning_rate=3e-4,
            max_epochs=120,
            patience=12,
            encoder_spatial_dropout_rate=0.20,
            embedding_dropout_rate=0.25,
        ),
        RobustMILCandidate(
            pooling="attention",
            projection_channels=32,
            embedding_units=16,
            attention_units=16,
            dropout_rate=0.35,
            l2_strength=3e-3,
            learning_rate=3e-4,
            max_epochs=120,
            patience=12,
            encoder_spatial_dropout_rate=0.20,
            embedding_dropout_rate=0.25,
        ),
    ]


def build_model(
    candidate: RobustMILCandidate,
    channel_mean: np.ndarray,
    channel_scale: np.ndarray,
) -> tf.keras.Model:
    """Construye el encoder WST y permite regularizarlo antes del MIL."""

    regularizer = tf.keras.regularizers.l2(candidate.l2_strength)
    event_input = tf.keras.Input(
        shape=(base.EXPECTED_TIME_COUNT, base.EXPECTED_PATH_COUNT),
        name="wst_event",
    )
    x = base.FixedChannelStandardizer(
        channel_mean,
        channel_scale,
        name="channel_standardizer",
    )(event_input)
    x = tf.keras.layers.Conv1D(
        candidate.projection_channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name="path_projection_1x1",
    )(x)
    x = tf.keras.layers.ReLU(name="projection_relu")(x)
    x = tf.keras.layers.SeparableConv1D(
        candidate.projection_channels,
        kernel_size=3,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizer,
        pointwise_regularizer=regularizer,
        name="temporal_separable_conv",
    )(x)
    x = tf.keras.layers.ReLU(name="temporal_relu")(x)
    if candidate.encoder_spatial_dropout_rate > 0:
        x = tf.keras.layers.SpatialDropout1D(
            candidate.encoder_spatial_dropout_rate,
            name="encoder_spatial_dropout",
        )(x)
    mean_time = tf.keras.layers.GlobalAveragePooling1D(
        name="temporal_mean"
    )(x)
    max_time = tf.keras.layers.GlobalMaxPooling1D(name="temporal_max")(x)
    x = tf.keras.layers.Concatenate(name="temporal_mean_max")(
        [mean_time, max_time]
    )
    x = tf.keras.layers.Dense(
        candidate.embedding_units,
        activation="relu",
        kernel_regularizer=regularizer,
        name="event_embedding",
    )(x)
    if candidate.embedding_dropout_rate > 0:
        x = tf.keras.layers.Dropout(
            candidate.embedding_dropout_rate,
            name="event_embedding_dropout",
        )(x)
    event_encoder = tf.keras.Model(event_input, x, name="event_encoder")

    events = tf.keras.Input(
        shape=(
            None,
            base.EXPECTED_TIME_COUNT,
            base.EXPECTED_PATH_COUNT,
        ),
        name="events",
    )
    event_mask = tf.keras.Input(shape=(None,), name="event_mask")
    embeddings = tf.keras.layers.TimeDistributed(
        event_encoder,
        name="encode_events",
    )(events)
    recording_embedding = base.GatedAttentionPooling(
        candidate.attention_units,
        name="gated_attention_pooling",
    )([embeddings, event_mask])
    recording_embedding = tf.keras.layers.Dropout(
        candidate.dropout_rate,
        name="recording_dropout",
    )(recording_embedding)
    score = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=regularizer,
        name="wet_probability",
    )(recording_embedding)
    model = tf.keras.Model(
        {"events": events, "event_mask": event_mask},
        score,
        name="wst_temporal_mil_attention_robust",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(candidate.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        weighted_metrics=[],
    )
    return model


def current_learning_rate(model: tf.keras.Model) -> float:
    value = model.optimizer.learning_rate
    if callable(value):
        value = value(model.optimizer.iterations)
    return float(tf.keras.backend.get_value(value))


class InnerValidationMetrics(tf.keras.callbacks.Callback):
    """Calcula metricas recording-level del inner-validation por epoca."""

    def __init__(
        self,
        prediction_sequence: base.RecordingBagSequence,
        labels: np.ndarray,
        candidate_key: str,
        outer_fold: int,
        inner_repeat: int,
    ) -> None:
        super().__init__()
        self.prediction_sequence = prediction_sequence
        self.labels = np.asarray(labels, dtype=int)
        self.candidate_key = candidate_key
        self.outer_fold = int(outer_fold)
        self.inner_repeat = int(inner_repeat)
        self.rows: list[dict[str, Any]] = []

    def _scores(self) -> np.ndarray:
        batches = []
        for batch_index in range(len(self.prediction_sequence)):
            inputs = self.prediction_sequence[batch_index]
            values = self.model(inputs, training=False).numpy().reshape(-1)
            batches.append(values)
        scores = np.concatenate(batches).astype(float, copy=False)
        if scores.shape != self.labels.shape or not np.isfinite(scores).all():
            raise RuntimeError("Scores inner-validation invalidos.")
        return scores

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        logs = logs or {}
        scores = self._scores()
        metrics = base.common.binary_metrics(self.labels, scores, 0.5)
        self.rows.append(
            {
                "candidate_key": self.candidate_key,
                "outer_fold": self.outer_fold,
                "inner_repeat": self.inner_repeat,
                "epoch": int(epoch + 1),
                "learning_rate": current_learning_rate(self.model),
                "train_loss": float(logs.get("loss", np.nan)),
                "validation_loss": float(logs.get("val_loss", np.nan)),
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )


def fit_one_inner_split(
    data: base.TemporalMILData,
    candidate: RobustMILCandidate,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
    channel_mean: np.ndarray,
    channel_scale: np.ndarray,
    seed: int,
    outer_fold: int,
    inner_repeat: int,
) -> tuple[int, float, int, int, pd.DataFrame]:
    train_weights = base.recording_class_weights(
        data.train_recordings, inner_train
    )
    validation_weights = base.recording_class_weights(
        data.train_recordings, inner_validation
    )
    train_sequence = base.make_sequence(
        data,
        "train",
        inner_train,
        candidate,
        True,
        True,
        train_weights,
        seed,
    )
    validation_sequence = base.make_sequence(
        data,
        "train",
        inner_validation,
        candidate,
        False,
        True,
        validation_weights,
        seed,
    )
    prediction_sequence = base.make_sequence(
        data,
        "train",
        inner_validation,
        candidate,
        False,
        False,
        seed=seed,
    )
    labels = data.train_recordings.iloc[inner_validation][
        "stage2_target"
    ].to_numpy(dtype=int)

    tf.keras.backend.clear_session()
    base.set_seed(seed)
    model = build_model(candidate, channel_mean, channel_scale)
    epoch_metrics = InnerValidationMetrics(
        prediction_sequence,
        labels,
        candidate.key,
        outer_fold,
        inner_repeat,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=4,
        min_delta=1e-4,
        min_lr=1e-5,
        verbose=0,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=candidate.patience,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=0,
    )
    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=candidate.max_epochs,
        callbacks=[epoch_metrics, reduce_lr, early_stopping],
        verbose=0,
    )
    losses = np.asarray(history.history["val_loss"], dtype=float)
    best_epoch = int(np.argmin(losses) + 1)
    epochs_run = int(len(losses))
    rows = pd.DataFrame(epoch_metrics.rows)
    if len(rows) != epochs_run:
        raise RuntimeError("Faltan metricas de alguna epoca inner.")
    rows["best_epoch_inner"] = best_epoch
    rows["epochs_run_inner"] = epochs_run
    return (
        best_epoch,
        float(losses[best_epoch - 1]),
        epochs_run,
        int(model.count_params()),
        rows,
    )


def refit_and_predict_outer(
    data: base.TemporalMILData,
    candidate: RobustMILCandidate,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    channel_mean: np.ndarray,
    channel_scale: np.ndarray,
    epochs: int,
    seed: int,
) -> np.ndarray:
    weights = base.recording_class_weights(data.train_recordings, outer_train)
    train_sequence = base.make_sequence(
        data,
        "train",
        outer_train,
        candidate,
        True,
        True,
        weights,
        seed,
    )
    validation_sequence = base.make_sequence(
        data,
        "train",
        outer_validation,
        candidate,
        False,
        False,
        seed=seed,
    )
    tf.keras.backend.clear_session()
    base.set_seed(seed)
    model = build_model(candidate, channel_mean, channel_scale)
    model.fit(train_sequence, epochs=epochs, verbose=0)
    scores = model.predict(validation_sequence, verbose=0).reshape(-1)
    if scores.shape != (len(outer_validation),) or not np.isfinite(scores).all():
        raise RuntimeError("Scores OOF MIL robusto invalidos.")
    return scores


def plot_inner_histories(
    histories: pd.DataFrame,
    candidate: RobustMILCandidate,
    outer_fold: int,
    output_path: Path,
) -> None:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), sharex=False)
    panels = [
        ("Loss", [("train_loss", "train", "--"), ("validation_loss", "val", "-")]),
        (
            "Macro-F1 y balanced accuracy (umbral 0.5)",
            [
                ("validation_macro_f1", "macro-F1", "-"),
                ("validation_balanced_accuracy", "bal-acc", "--"),
            ],
        ),
        (
            "Clase wet (umbral 0.5)",
            [
                ("validation_wet_recall", "recall", "-"),
                ("validation_wet_precision", "precision", "--"),
            ],
        ),
        (
            "Clase dry (umbral 0.5)",
            [
                ("validation_dry_recall", "recall", "-"),
                ("validation_dry_precision", "precision", "--"),
            ],
        ),
        (
            "Metricas sin umbral",
            [
                ("validation_roc_auc", "ROC-AUC", "-"),
                (
                    "validation_average_precision_wet",
                    "AP wet",
                    "--",
                ),
            ],
        ),
        ("Learning rate", [("learning_rate", "LR", "-")]),
    ]
    for axis, (title, series) in zip(axes.ravel(), panels):
        for repeat, group in histories.groupby("inner_repeat", sort=True):
            color = colors[int(repeat) % len(colors)]
            for column, label, linestyle in series:
                axis.plot(
                    group["epoch"],
                    group[column],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.6,
                    label=f"split {int(repeat)} - {label}",
                )
            best_epoch = int(group["best_epoch_inner"].iloc[0])
            axis.axvline(best_epoch, color=color, alpha=0.30, linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Epoca")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    axes[2, 1].set_yscale("log")
    fig.suptitle(
        f"Inner-validation por epoca - outer fold {outer_fold}\n{candidate.key}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate_candidate_oof(
    data: base.TemporalMILData,
    candidate: RobustMILCandidate,
    graph_dir: Path,
) -> tuple[base.CandidateEvaluation, pd.DataFrame]:
    started = time.perf_counter()
    recordings = data.train_recordings
    folds = recordings["fold"].to_numpy(dtype=int)
    labels = recordings["stage2_target"].to_numpy(dtype=int)
    oof_scores = np.full(len(recordings), np.nan, dtype=float)
    training_rows: list[dict[str, Any]] = []
    epoch_frames: list[pd.DataFrame] = []
    parameter_count: int | None = None

    for fold in sorted(base.common.EXPECTED_TRAIN_FOLDS):
        outer_validation = np.flatnonzero(folds == fold)
        outer_train = np.flatnonzero(folds != fold)
        best_epochs: list[int] = []
        best_losses: list[float] = []
        epochs_run_values: list[int] = []
        fold_histories: list[pd.DataFrame] = []

        for repeat in range(INNER_REPEATS):
            seed = base.RANDOM_STATE + fold * 100 + repeat
            inner_train, inner_validation = train_test_split(
                outer_train,
                test_size=base.INNER_VALIDATION_SIZE,
                stratify=labels[outer_train],
                random_state=seed,
            )
            inner_train = np.asarray(inner_train, dtype=int)
            inner_validation = np.asarray(inner_validation, dtype=int)
            inner_mean, inner_scale = base.fit_channel_scaler(
                data.x_train, data.train_event_indices, inner_train
            )
            (
                best_epoch,
                best_loss,
                epochs_run,
                current_parameters,
                epoch_history,
            ) = fit_one_inner_split(
                data,
                candidate,
                inner_train,
                inner_validation,
                inner_mean,
                inner_scale,
                seed,
                fold,
                repeat,
            )
            if parameter_count is None:
                parameter_count = current_parameters
            elif parameter_count != current_parameters:
                raise RuntimeError("Los parametros cambiaron entre ajustes.")
            best_epochs.append(best_epoch)
            best_losses.append(best_loss)
            epochs_run_values.append(epochs_run)
            fold_histories.append(epoch_history)
            epoch_frames.append(epoch_history)
            print(
                f"    inner {repeat}: best_epoch={best_epoch}, "
                f"epochs_run={epochs_run}, val_loss={best_loss:.5f}"
            )

        selected_epochs = max(1, int(np.rint(np.median(best_epochs))))
        histories = pd.concat(fold_histories, ignore_index=True)
        plot_inner_histories(
            histories,
            candidate,
            fold,
            graph_dir / candidate.key / f"outer_fold_{fold}.png",
        )

        outer_mean, outer_scale = base.fit_channel_scaler(
            data.x_train, data.train_event_indices, outer_train
        )
        outer_seed = base.RANDOM_STATE + fold
        fold_scores = refit_and_predict_outer(
            data,
            candidate,
            outer_train,
            outer_validation,
            outer_mean,
            outer_scale,
            selected_epochs,
            outer_seed,
        )
        oof_scores[outer_validation] = fold_scores
        native = base.common.binary_metrics(
            labels[outer_validation], fold_scores, 0.5
        )
        training_rows.append(
            {
                "candidate_key": candidate.key,
                "outer_fold": fold,
                "outer_train_recordings": len(outer_train),
                "outer_validation_recordings": len(outer_validation),
                "inner_repeats": INNER_REPEATS,
                "best_epochs_inner": "|".join(map(str, best_epochs)),
                "best_losses_inner": "|".join(
                    f"{value:.8f}" for value in best_losses
                ),
                "epochs_run_inner": "|".join(
                    map(str, epochs_run_values)
                ),
                "best_epoch_inner": selected_epochs,
                "outer_refit_epochs": selected_epochs,
                "outer_macro_f1_at_0p5": native["macro_f1"],
                "outer_roc_auc": native["roc_auc"],
                "outer_dry_recall_at_0p5": native["dry_recall"],
                "outer_wet_recall_at_0p5": native["wet_recall"],
            }
        )
        print(
            f"  Fold {fold}: inner_epochs={best_epochs} -> "
            f"median={selected_epochs} | macro-F1@0.5="
            f"{native['macro_f1']:.4f} | wet-recall@0.5="
            f"{native['wet_recall']:.4f}"
        )

    if not np.isfinite(oof_scores).all():
        raise RuntimeError(f"OOF incompleto para {candidate.key}.")
    predictions = base.prediction_frame(recordings, oof_scores)
    threshold = base.common.tune_threshold(labels, oof_scores, 0.5)
    metrics = base.common.binary_metrics(labels, oof_scores, threshold)
    metrics_fixed = base.common.binary_metrics(labels, oof_scores, 0.5)
    predictions["y_pred_oof_threshold"] = (
        oof_scores >= threshold
    ).astype(int)
    predictions["y_pred_fixed_0p5"] = (oof_scores >= 0.5).astype(int)
    predictions["candidate_key"] = candidate.key
    result = base.CandidateEvaluation(
        candidate=candidate,
        threshold=threshold,
        metrics=metrics,
        metrics_fixed_0p5=metrics_fixed,
        oof_predictions=predictions,
        fold_metrics=base.fold_metrics(predictions, threshold),
        training_summary=pd.DataFrame(training_rows),
        parameter_count=int(parameter_count),
        elapsed_seconds=time.perf_counter() - started,
    )
    return result, pd.concat(epoch_frames, ignore_index=True)


def stability_values(result: base.CandidateEvaluation) -> dict[str, float]:
    folds = result.fold_metrics
    return {
        "fold_macro_f1_mean": float(folds["macro_f1"].mean()),
        "fold_macro_f1_std": float(folds["macro_f1"].std(ddof=0)),
        "fold_macro_f1_min": float(folds["macro_f1"].min()),
        "fold_wet_recall_min": float(folds["wet_recall"].min()),
        "fold_roc_auc_min": float(folds["roc_auc"].min()),
    }


def candidate_results_frame(
    results: list[base.CandidateEvaluation],
) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "candidate_key": result.candidate.key,
                "parameter_count": result.parameter_count,
                "threshold_oof": result.threshold,
                "elapsed_seconds": result.elapsed_seconds,
                **asdict(result.candidate),
                **{f"oof_tuned__{k}": v for k, v in result.metrics.items()},
                **{
                    f"fixed_0p5__{k}": v
                    for k, v in result.metrics_fixed_0p5.items()
                },
                **stability_values(result),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "oof_tuned__macro_f1",
            "oof_tuned__roc_auc",
            "fold_macro_f1_std",
        ],
        ascending=[False, False, True],
    )


def select_winner(
    results: list[base.CandidateEvaluation],
) -> base.CandidateEvaluation:
    best_macro = max(float(item.metrics["macro_f1"]) for item in results)
    contenders = [
        item
        for item in results
        if float(item.metrics["macro_f1"])
        >= best_macro - SELECTION_TOLERANCE
    ]
    return max(
        contenders,
        key=lambda item: (
            float(item.metrics["roc_auc"]),
            stability_values(item)["fold_wet_recall_min"],
            -stability_values(item)["fold_macro_f1_std"],
            -item.parameter_count,
        ),
    )


def metric_row(
    split: str,
    policy: str,
    threshold: float,
    predictions: pd.DataFrame,
    winner: base.CandidateEvaluation,
    model_size_kb: float,
) -> dict[str, Any]:
    return {
        "split": split,
        "experiment": EXPERIMENT_KEY,
        "candidate_key": winner.candidate.key,
        "threshold_policy": policy,
        "threshold": threshold,
        "parameter_count": winner.parameter_count,
        "model_size_kb_float32_keras": model_size_kb,
        **base.common.binary_metrics(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["score"].to_numpy(dtype=float),
            threshold,
        ),
    }


def fit_final_model(
    data: base.TemporalMILData,
    winner: base.CandidateEvaluation,
) -> tuple[tf.keras.Model, int, np.ndarray, np.ndarray]:
    candidate = winner.candidate
    if not isinstance(candidate, RobustMILCandidate):
        raise TypeError("El ganador no contiene la configuracion robusta.")
    epochs = max(
        1,
        int(
            np.rint(
                np.median(
                    winner.training_summary[
                        "best_epoch_inner"
                    ].to_numpy(dtype=int)
                )
            )
        ),
    )
    all_indices = np.arange(len(data.train_recordings), dtype=int)
    mean, scale = base.fit_channel_scaler(
        data.x_train,
        data.train_event_indices,
        all_indices,
    )
    weights = base.recording_class_weights(
        data.train_recordings,
        all_indices,
    )
    sequence = base.make_sequence(
        data,
        "train",
        all_indices,
        candidate,
        True,
        True,
        weights,
        base.RANDOM_STATE,
    )
    tf.keras.backend.clear_session()
    base.set_seed(base.RANDOM_STATE)
    model = build_model(candidate, mean, scale)
    model.fit(sequence, epochs=epochs, verbose=2)
    return model, epochs, mean, scale


def train(
    data: base.TemporalMILData,
    extraction_configuration: pd.DataFrame,
    preset: str,
    quick: bool,
) -> None:
    mode = "quick" if quick else "full"
    result_dir = RESULTS_ROOT / preset / mode
    model_dir = MODELS_ROOT / preset / mode
    graph_dir = GRAPHS_ROOT / preset / mode
    result_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    specs = candidate_specs(quick)
    results: list[base.CandidateEvaluation] = []
    epoch_frames: list[pd.DataFrame] = []
    print("\n" + "=" * 78)
    print("CV ROBUSTA - WST TEMPORAL + MIL ATTENTION")
    print("=" * 78)
    for index, candidate in enumerate(specs, start=1):
        print(f"\n[{index}/{len(specs)}] {candidate.key}")
        result, epoch_frame = evaluate_candidate_oof(
            data, candidate, graph_dir / "inner_epoch_curves"
        )
        results.append(result)
        epoch_frames.append(epoch_frame)
        stability = stability_values(result)
        print(
            f"{candidate.key} | macro-F1={result.metrics['macro_f1']:.4f} | "
            f"bal-acc={result.metrics['balanced_accuracy']:.4f} | "
            f"AUC={result.metrics['roc_auc']:.4f} | "
            f"peor wet-recall={stability['fold_wet_recall_min']:.4f} | "
            f"params={result.parameter_count}"
        )

    winner = select_winner(results)
    candidate_results_frame(results).to_csv(
        result_dir / "candidate_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        [item.training_summary for item in results], ignore_index=True
    ).to_csv(
        result_dir / "cv_training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(epoch_frames, ignore_index=True).to_csv(
        result_dir / "inner_epoch_metrics.csv",
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
        print("\nPrueba robusta rapida completada.")
        print(f"Ganador OOF: {winner.candidate.key}")
        print(f"Macro-F1 OOF: {winner.metrics['macro_f1']:.4f}")
        print("VALIDATION no se ha evaluado.")
        print(f"Resultados: {result_dir}")
        print(f"Curvas: {graph_dir / 'inner_epoch_curves'}")
        return

    print("\nAjustando MIL attention final exclusivamente con todo TRAIN...")
    model, final_epochs, channel_mean, channel_scale = fit_final_model(
        data, winner
    )
    validation_indices = np.arange(
        len(data.validation_recordings), dtype=int
    )
    validation_sequence = base.make_sequence(
        data,
        "validation",
        validation_indices,
        winner.candidate,
        False,
        False,
    )
    validation_scores = model.predict(
        validation_sequence, verbose=0
    ).reshape(-1)
    validation_predictions = base.prediction_frame(
        data.validation_recordings, validation_scores
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
    model_path = model_dir / "wst_temporal_mil_attention_robust.keras"
    model.save(model_path)
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
    base.save_attention_weights(
        model,
        data,
        result_dir / "validation_event_attention_weights.csv",
    )
    configuration = {
        "preset": preset,
        "unit_of_training": "recording_bag",
        "pooling": "gated_attention",
        "inner_repeats_per_outer_fold": INNER_REPEATS,
        "outer_epoch_policy": "median_best_epoch_from_three_inner_splits",
        "inner_early_stopping_monitor": "val_loss",
        "inner_reduce_lr_monitor": "val_loss",
        "epoch_metric_threshold": 0.5,
        "winner": winner.candidate.key,
        "selection_policy": (
            "macro_f1_tolerance_0p005_then_auc_then_worst_wet_recall_"
            "then_fold_std_then_size"
        ),
        "threshold_policy": "oof_tuned_frozen",
        "threshold": winner.threshold,
        "final_epochs_from_outer_median": final_epochs,
        "parameter_count": winner.parameter_count,
        "model_size_kb_float32_keras": model_size_kb,
        "class_balance": "balanced_sample_weight_per_recording",
        "smote_used": False,
        "validation_used_for_selection": False,
        "test_read": False,
        "channel_mean_min": float(np.min(channel_mean)),
        "channel_mean_max": float(np.max(channel_mean)),
        "channel_scale_min": float(np.min(channel_scale)),
        "channel_scale_max": float(np.max(channel_scale)),
        **{f"cnn_{key}": value for key, value in asdict(winner.candidate).items()},
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
    validation_graph = graph_dir / "validation_wst_temporal_mil_attention_robust.png"
    base.graph_helpers.create_validation_graph(
        validation_predictions,
        winner.threshold,
        winner.candidate.key,
        validation_graph,
    )
    validation_metrics = validation_rows[-1]
    print("\n" + "=" * 78)
    print("RESULTADO WST TEMPORAL + MIL ATTENTION ROBUSTO")
    print("=" * 78)
    print(f"Ganador: {winner.candidate.key}")
    print(f"Umbral OOF congelado: {winner.threshold:.6f}")
    print(f"Macro-F1 OOF: {winner.metrics['macro_f1']:.4f}")
    print(f"Macro-F1 validation: {validation_metrics['macro_f1']:.4f}")
    print(
        "Recalls validation dry/wet: "
        f"{validation_metrics['dry_recall']:.4f} / "
        f"{validation_metrics['wet_recall']:.4f}"
    )
    print(f"Parametros: {winner.parameter_count}")
    print(f"Modelo Keras: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print(f"Curvas por epoca: {graph_dir / 'inner_epoch_curves'}")
    print("TEST permanece reservado.")


def print_check(data: base.TemporalMILData, quick: bool) -> None:
    train_counts = data.train_recordings["stage2_target"].value_counts()
    validation_counts = data.validation_recordings[
        "stage2_target"
    ].value_counts()
    print("=" * 78)
    print("CHECK - WST TEMPORAL + MIL ATTENTION ROBUSTO")
    print("=" * 78)
    print(
        f"Grabaciones TRAIN/VALIDATION: {len(data.train_recordings)} / "
        f"{len(data.validation_recordings)}"
    )
    print(f"TRAIN dry/wet: {train_counts[0]} / {train_counts[1]}")
    print(
        f"VALIDATION dry/wet: {validation_counts[0]} / "
        f"{validation_counts[1]}"
    )
    print(f"Inner splits por outer fold: {INNER_REPEATS}")
    print(f"Arquitecturas: {len(candidate_specs(quick))}")
    all_indices = np.arange(len(data.train_recordings), dtype=int)
    mean, scale = base.fit_channel_scaler(
        data.x_train, data.train_event_indices, all_indices
    )
    for candidate in candidate_specs(quick):
        tf.keras.backend.clear_session()
        model = build_model(candidate, mean, scale)
        print(f"  {candidate.key}: {model.count_params()} parametros")
    print("Las curvas por epoca usan solo inner-validation y umbral 0.5.")
    print("Quick evalua solo la arquitectura base con OOF.")
    print("Full compara cuatro arquitecturas y evalua VALIDATION para el ganador.")
    print("TEST no se lee ni se procesa.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena WST temporal + gated-attention MIL con seleccion robusta "
            "de epocas y curvas por inner split."
        )
    )
    parser.add_argument(
        "--action", choices=["check", "train"], default="check"
    )
    parser.add_argument(
        "--preset", choices=PRESETS, default=DEFAULT_PRESET
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Solo arquitectura base, tres inner splits por fold y OOF; "
            "no evalua VALIDATION."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST TEMPORAL + MIL ATTENTION ROBUSTO")
    print("=" * 78)
    print(f"Preset: {args.preset} | Accion: {args.action} | Quick: {args.quick}")
    print("TEST no sera leido ni procesado.")
    data, extraction_configuration = base.load_data(args.preset)
    print_check(data, args.quick)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(data, extraction_configuration, args.preset, args.quick)


if __name__ == "__main__":
    main()
