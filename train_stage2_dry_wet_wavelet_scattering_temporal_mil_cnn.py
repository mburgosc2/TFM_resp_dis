"""Stage 2 dry/wet: WST temporal + tiny CNN + MIL por grabacion.

Cada original_uuid es una bolsa de 1-9 eventos. Una CNN compartida transforma
cada tensor WST temporal (5 posiciones x 644 caminos) en un embedding y un
pooling MIL genera una unica representacion y una unica prediccion dry/wet por
grabacion. No se impone una perdida individual a cada evento.

Compara tres poolings con el mismo encoder: mean, mean+max y gated attention.
Los folds, normalizacion, epocas, arquitectura y umbral se seleccionan solo con
TRAIN OOF. VALIDATION se evalua una vez para el ganador y TEST no se procesa.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import train_stage2_dry_wet_cochleograms as common
import train_stage2_dry_wet_wavelet_scattering_recording_mlp as graph_helpers


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_ROOT = (
    SCRIPT_DIR / "features_extracted_stage2_dry_wet_wavelet_scattering_temporal"
)
RESULTS_ROOT = (
    SCRIPT_DIR / "results_stage2_dry_wet_wavelet_scattering_temporal_mil_cnn"
)
MODELS_ROOT = (
    SCRIPT_DIR / "models_stage2_dry_wet_wavelet_scattering_temporal_mil_cnn"
)
GRAPHS_ROOT = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "training_wavelet_scattering_temporal_mil_cnn"
)

PRESETS = ("paper_q8_q1_t500_full",)
DEFAULT_PRESET = PRESETS[0]
RANDOM_STATE = 42
EXPECTED_PATH_COUNT = 644
EXPECTED_TIME_COUNT = 5
INNER_VALIDATION_SIZE = 0.15
EXPERIMENT_KEY = "wavelet_scattering_temporal_mil_cnn"


@dataclass
class TemporalMILData:
    x_train: np.ndarray
    train_event_metadata: pd.DataFrame
    train_recordings: pd.DataFrame
    train_event_indices: list[np.ndarray]
    x_validation: np.ndarray
    validation_event_metadata: pd.DataFrame
    validation_recordings: pd.DataFrame
    validation_event_indices: list[np.ndarray]
    max_events: int


@dataclass(frozen=True)
class MILCandidate:
    pooling: str
    projection_channels: int = 16
    embedding_units: int = 8
    attention_units: int = 8
    dropout_rate: float = 0.20
    l2_strength: float = 1e-3
    learning_rate: float = 5e-4
    batch_size: int = 32
    max_epochs: int = 120
    patience: int = 12

    @property
    def key(self) -> str:
        dropout = f"{self.dropout_rate:g}".replace(".", "p")
        l2_value = f"{self.l2_strength:g}".replace(".", "p")
        learning_rate = f"{self.learning_rate:g}".replace(".", "p")
        return (
            f"mil_{self.pooling}__proj{self.projection_channels}__"
            f"emb{self.embedding_units}__att{self.attention_units}__"
            f"drop{dropout}__l2{l2_value}__lr{learning_rate}"
        )


@dataclass
class CandidateEvaluation:
    candidate: MILCandidate
    threshold: float
    metrics: dict[str, float | int]
    metrics_fixed_0p5: dict[str, float | int]
    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    training_summary: pd.DataFrame
    parameter_count: int
    elapsed_seconds: float


def candidate_specs(quick: bool) -> list[MILCandidate]:
    if quick:
        return [
            MILCandidate(
                pooling="attention",
                max_epochs=35,
                patience=6,
            )
        ]
    return [
        MILCandidate(pooling="mean"),
        MILCandidate(pooling="mean_max"),
        MILCandidate(pooling="attention"),
    ]


def validate_event_split(
    split_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    folds: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    required = {
        "feature_row",
        "event_id",
        "event_index",
        "original_uuid",
        "cough_type",
        "cough_type_consensus",
        "stage2_target",
        "fold",
        "split",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {split_name}: {sorted(missing)}")
    expected_shape = (len(metadata), EXPECTED_PATH_COUNT, EXPECTED_TIME_COUNT)
    if x_values.shape != expected_shape:
        raise ValueError(f"Forma WST {split_name} invalida: {x_values.shape}.")
    if x_values.dtype != np.float32:
        raise ValueError(f"WST {split_name} debe ser float32.")
    if y_values.shape != (len(metadata),) or folds.shape != (len(metadata),):
        raise ValueError(f"y/folds de {split_name} no coinciden con metadata.")
    if metadata["feature_row"].tolist() != list(range(len(metadata))):
        raise ValueError(f"feature_row desalineado en {split_name}.")
    if not np.array_equal(
        y_values.astype(int), metadata["stage2_target"].to_numpy(dtype=int)
    ):
        raise ValueError(f"y no coincide con metadata en {split_name}.")
    if not np.array_equal(
        folds.astype(int), metadata["fold"].to_numpy(dtype=int)
    ):
        raise ValueError(f"folds no coincide con metadata en {split_name}.")
    if set(y_values.astype(int)) != {0, 1}:
        raise ValueError(f"{split_name} no contiene dry y wet.")
    unique_folds = set(folds.astype(int))
    if split_name == "train" and unique_folds != common.EXPECTED_TRAIN_FOLDS:
        raise ValueError(f"Folds TRAIN inesperados: {unique_folds}.")
    if split_name == "validation" and unique_folds != {-1}:
        raise ValueError(f"Folds VALIDATION inesperados: {unique_folds}.")
    for column in (
        "stage2_target",
        "cough_type",
        "cough_type_consensus",
        "fold",
        "split",
    ):
        if metadata.groupby("original_uuid")[column].nunique(dropna=False).max() != 1:
            raise ValueError(
                f"Una grabacion de {split_name} tiene varios valores de {column}."
            )


def build_recording_view(
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    event_indices: list[np.ndarray] = []
    for original_uuid, positions in metadata.groupby(
        "original_uuid", sort=False
    ).indices.items():
        indices = np.asarray(positions, dtype=int)
        group = metadata.iloc[indices]
        rows.append(
            {
                "recording_row": len(rows),
                "original_uuid": str(original_uuid),
                "stage2_target": int(group["stage2_target"].iloc[0]),
                "cough_type": str(group["cough_type"].iloc[0]),
                "cough_type_consensus": str(
                    group["cough_type_consensus"].iloc[0]
                ),
                "fold": int(group["fold"].iloc[0]),
                "split": str(group["split"].iloc[0]),
                "event_count": len(indices),
            }
        )
        event_indices.append(indices)
    recordings = pd.DataFrame(rows)
    if recordings["original_uuid"].duplicated().any():
        raise RuntimeError("La vista recording contiene UUID duplicados.")
    return recordings, event_indices


def load_data(preset: str) -> tuple[TemporalMILData, pd.DataFrame]:
    root = FEATURE_ROOT / preset
    configuration_path = root / "wavelet_scattering_temporal_configuration.csv"
    configuration = pd.read_csv(configuration_path)
    if len(configuration) != 1:
        raise ValueError("Configuracion WST temporal invalida.")
    config = configuration.iloc[0]
    if int(config["path_count"]) != EXPECTED_PATH_COUNT:
        raise ValueError("Numero de caminos WST inesperado.")
    if int(config["time_position_count"]) != EXPECTED_TIME_COUNT:
        raise ValueError("Numero de posiciones temporales inesperado.")
    if str(config["stored_axis_order"]) != "event,path,time":
        raise ValueError("Orden de ejes WST temporal inesperado.")
    if bool(config["test_processed"]):
        raise ValueError("La configuracion indica que TEST fue procesado.")

    loaded: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for split_name in ("train", "validation"):
        x_values = np.load(
            root / f"X_events_{split_name}.npy", mmap_mode="r"
        )
        y_values = np.load(root / f"y_events_{split_name}.npy")
        folds = np.load(root / f"folds_events_{split_name}.npy")
        metadata = pd.read_csv(
            root / f"metadata_events_features_{split_name}.csv"
        )
        validate_event_split(
            split_name, x_values, y_values, folds, metadata
        )
        loaded[split_name] = (x_values, metadata)

    train_x, train_metadata = loaded["train"]
    validation_x, validation_metadata = loaded["validation"]
    overlap = set(train_metadata["original_uuid"].astype(str)) & set(
        validation_metadata["original_uuid"].astype(str)
    )
    if overlap:
        raise ValueError(f"TRAIN y VALIDATION comparten UUID: {sorted(overlap)[:5]}")
    train_recordings, train_indices = build_recording_view(train_metadata)
    validation_recordings, validation_indices = build_recording_view(
        validation_metadata
    )
    max_events = int(
        max(
            train_recordings["event_count"].max(),
            validation_recordings["event_count"].max(),
        )
    )
    return (
        TemporalMILData(
            x_train=train_x,
            train_event_metadata=train_metadata,
            train_recordings=train_recordings,
            train_event_indices=train_indices,
            x_validation=validation_x,
            validation_event_metadata=validation_metadata,
            validation_recordings=validation_recordings,
            validation_event_indices=validation_indices,
            max_events=max_events,
        ),
        configuration,
    )


def recording_class_weights(
    recordings: pd.DataFrame,
    indices: np.ndarray,
) -> np.ndarray:
    labels = recordings.iloc[indices]["stage2_target"].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("El subconjunto recording no contiene ambas clases.")
    factors = len(labels) / (2.0 * counts.astype(float))
    weights = factors[labels].astype(np.float32)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("Pesos recording invalidos.")
    return weights


def fit_channel_scaler(
    x_events: np.ndarray,
    event_indices: list[np.ndarray],
    recording_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Media/std por camino dando el mismo peso total a cada grabacion."""

    sum_means = np.zeros(EXPECTED_PATH_COUNT, dtype=np.float64)
    sum_squares = np.zeros(EXPECTED_PATH_COUNT, dtype=np.float64)
    for recording_index in recording_indices:
        indices = event_indices[int(recording_index)]
        values = np.asarray(x_events[indices], dtype=np.float64)
        # values=(event,path,time); cada grabacion aporta una media por camino.
        sum_means += values.mean(axis=(0, 2))
        sum_squares += np.square(values).mean(axis=(0, 2))
    count = float(len(recording_indices))
    mean = sum_means / count
    variance = np.maximum(sum_squares / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    near_constant = scale < 1e-12
    scale[near_constant] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise RuntimeError("Normalizacion WST produjo NaN o Inf.")
    return mean.astype(np.float32), scale.astype(np.float32)


class RecordingBagSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        x_events: np.ndarray,
        recordings: pd.DataFrame,
        event_indices: list[np.ndarray],
        recording_indices: np.ndarray,
        max_events: int,
        batch_size: int,
        shuffle: bool,
        include_targets: bool,
        sample_weights: np.ndarray | None = None,
        seed: int = RANDOM_STATE,
    ) -> None:
        super().__init__()
        self.x_events = x_events
        self.recordings = recordings
        self.event_indices = event_indices
        self.recording_indices = np.asarray(recording_indices, dtype=int)
        self.max_events = int(max_events)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.include_targets = bool(include_targets)
        self.sample_weights = sample_weights
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.recording_indices), dtype=int)
        if sample_weights is not None and sample_weights.shape != (
            len(self.recording_indices),
        ):
            raise ValueError("sample_weights no coincide con recording_indices.")
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.order) / self.batch_size)

    def __getitem__(self, batch_index: int):
        positions = self.order[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        selected_recordings = self.recording_indices[positions]
        batch_max_events = max(
            len(self.event_indices[int(index)]) for index in selected_recordings
        )
        # Nunca se descartan eventos. El padding se limita al maximo del batch.
        if batch_max_events > self.max_events:
            raise RuntimeError("Una grabacion supera max_events.")
        bags = np.zeros(
            (
                len(selected_recordings),
                batch_max_events,
                EXPECTED_TIME_COUNT,
                EXPECTED_PATH_COUNT,
            ),
            dtype=np.float32,
        )
        masks = np.zeros(
            (len(selected_recordings), batch_max_events), dtype=np.float32
        )
        for batch_row, recording_index in enumerate(selected_recordings):
            indices = self.event_indices[int(recording_index)]
            events = np.asarray(self.x_events[indices], dtype=np.float32)
            bags[batch_row, : len(indices)] = np.transpose(events, (0, 2, 1))
            masks[batch_row, : len(indices)] = 1.0
        inputs = {"events": bags, "event_mask": masks}
        if not self.include_targets:
            return inputs
        y_values = self.recordings.iloc[selected_recordings][
            "stage2_target"
        ].to_numpy(dtype=np.float32)
        if self.sample_weights is None:
            return inputs, y_values
        return inputs, y_values, self.sample_weights[positions]

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.order)


@tf.keras.utils.register_keras_serializable(package="TFM")
class FixedChannelStandardizer(tf.keras.layers.Layer):
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
        if self.mean_values.shape != (EXPECTED_PATH_COUNT,):
            raise ValueError("Mean WST por canal invalida.")
        if self.scale_values.shape != (EXPECTED_PATH_COUNT,):
            raise ValueError("Scale WST por canal invalida.")

    def build(self, input_shape) -> None:
        self.fixed_mean = self.add_weight(
            name="mean",
            shape=(EXPECTED_PATH_COUNT,),
            initializer=tf.keras.initializers.Constant(self.mean_values),
            trainable=False,
        )
        self.fixed_scale = self.add_weight(
            name="scale",
            shape=(EXPECTED_PATH_COUNT,),
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


@tf.keras.utils.register_keras_serializable(package="TFM")
class MaskedMeanPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        embeddings, mask = inputs
        expanded = tf.cast(mask[..., tf.newaxis], embeddings.dtype)
        numerator = tf.reduce_sum(embeddings * expanded, axis=1)
        denominator = tf.maximum(tf.reduce_sum(expanded, axis=1), 1.0)
        return numerator / denominator


@tf.keras.utils.register_keras_serializable(package="TFM")
class MaskedMeanMaxPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        embeddings, mask = inputs
        expanded = tf.cast(mask[..., tf.newaxis], embeddings.dtype)
        mean = tf.reduce_sum(embeddings * expanded, axis=1) / tf.maximum(
            tf.reduce_sum(expanded, axis=1), 1.0
        )
        masked = tf.where(
            expanded > 0,
            embeddings,
            tf.cast(-1e9, embeddings.dtype),
        )
        maximum = tf.reduce_max(masked, axis=1)
        return tf.concat([mean, maximum], axis=-1)


@tf.keras.utils.register_keras_serializable(package="TFM")
class GatedAttentionPooling(tf.keras.layers.Layer):
    def __init__(self, attention_units: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.attention_units = int(attention_units)
        self.tanh_projection = tf.keras.layers.Dense(
            self.attention_units, activation="tanh", name="attention_tanh"
        )
        self.sigmoid_projection = tf.keras.layers.Dense(
            self.attention_units,
            activation="sigmoid",
            name="attention_sigmoid",
        )
        self.logit_projection = tf.keras.layers.Dense(
            1, use_bias=False, name="attention_logit"
        )

    def compute_attention(self, embeddings, mask):
        gated = self.tanh_projection(embeddings) * self.sigmoid_projection(
            embeddings
        )
        logits = tf.squeeze(self.logit_projection(gated), axis=-1)
        logits = tf.where(
            tf.cast(mask, tf.bool),
            logits,
            tf.cast(-1e9, logits.dtype),
        )
        return tf.nn.softmax(logits, axis=1)

    def call(self, inputs):
        embeddings, mask = inputs
        weights = self.compute_attention(embeddings, mask)
        return tf.reduce_sum(embeddings * weights[..., tf.newaxis], axis=1)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["attention_units"] = self.attention_units
        return config


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_model(
    candidate: MILCandidate,
    channel_mean: np.ndarray,
    channel_scale: np.ndarray,
) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.l2(candidate.l2_strength)
    event_input = tf.keras.Input(
        shape=(EXPECTED_TIME_COUNT, EXPECTED_PATH_COUNT),
        name="wst_event",
    )
    x = FixedChannelStandardizer(
        channel_mean, channel_scale, name="channel_standardizer"
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
    event_encoder = tf.keras.Model(
        event_input, x, name="event_encoder"
    )

    events = tf.keras.Input(
        shape=(None, EXPECTED_TIME_COUNT, EXPECTED_PATH_COUNT),
        name="events",
    )
    event_mask = tf.keras.Input(shape=(None,), name="event_mask")
    embeddings = tf.keras.layers.TimeDistributed(
        event_encoder, name="encode_events"
    )(events)
    if candidate.pooling == "mean":
        recording_embedding = MaskedMeanPooling(name="masked_mean_pooling")(
            [embeddings, event_mask]
        )
    elif candidate.pooling == "mean_max":
        recording_embedding = MaskedMeanMaxPooling(
            name="masked_mean_max_pooling"
        )([embeddings, event_mask])
    elif candidate.pooling == "attention":
        recording_embedding = GatedAttentionPooling(
            candidate.attention_units,
            name="gated_attention_pooling",
        )([embeddings, event_mask])
    else:
        raise ValueError(f"Pooling MIL desconocido: {candidate.pooling}")
    recording_embedding = tf.keras.layers.Dropout(
        candidate.dropout_rate, name="recording_dropout"
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
        name=f"wst_temporal_mil_{candidate.pooling}",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(candidate.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        weighted_metrics=[],
    )
    return model


def make_sequence(
    data: TemporalMILData,
    split_name: str,
    recording_indices: np.ndarray,
    candidate: MILCandidate,
    shuffle: bool,
    include_targets: bool,
    sample_weights: np.ndarray | None = None,
    seed: int = RANDOM_STATE,
) -> RecordingBagSequence:
    if split_name == "train":
        return RecordingBagSequence(
            data.x_train,
            data.train_recordings,
            data.train_event_indices,
            recording_indices,
            data.max_events,
            candidate.batch_size,
            shuffle,
            include_targets,
            sample_weights,
            seed,
        )
    if split_name == "validation":
        return RecordingBagSequence(
            data.x_validation,
            data.validation_recordings,
            data.validation_event_indices,
            recording_indices,
            data.max_events,
            candidate.batch_size,
            shuffle,
            include_targets,
            sample_weights,
            seed,
        )
    raise ValueError(f"Split desconocido: {split_name}")


def fold_metrics(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
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


def prediction_frame(
    recordings: pd.DataFrame,
    scores: np.ndarray,
) -> pd.DataFrame:
    if scores.shape != (len(recordings),):
        raise ValueError("Scores y grabaciones no coinciden.")
    result = recordings[
        [
            "original_uuid",
            "stage2_target",
            "cough_type",
            "cough_type_consensus",
            "fold",
            "event_count",
        ]
    ].rename(columns={"stage2_target": "y_true"}).copy()
    result["score"] = scores
    return result


def fit_early_stopping(
    data: TemporalMILData,
    candidate: MILCandidate,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
    channel_mean: np.ndarray,
    channel_scale: np.ndarray,
    seed: int,
) -> tuple[int, float, int, int]:
    train_weights = recording_class_weights(data.train_recordings, inner_train)
    validation_weights = recording_class_weights(
        data.train_recordings, inner_validation
    )
    train_sequence = make_sequence(
        data, "train", inner_train, candidate, True, True,
        train_weights, seed
    )
    validation_sequence = make_sequence(
        data, "train", inner_validation, candidate, False, True,
        validation_weights, seed
    )
    tf.keras.backend.clear_session()
    set_seed(seed)
    model = build_model(candidate, channel_mean, channel_scale)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=4,
        min_delta=1e-4,
        min_lr=1e-5,
        verbose=0,
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
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
        callbacks=[
            reduce_lr_callback,
            early_stopping_callback,
        ],
        verbose=0,
    )
    losses = np.asarray(history.history["val_loss"], dtype=float)
    best_epoch = int(np.argmin(losses) + 1)
    return best_epoch, float(losses[best_epoch - 1]), len(losses), int(
        model.count_params()
    )


def refit_and_predict_outer(
    data: TemporalMILData,
    candidate: MILCandidate,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    channel_mean: np.ndarray,
    channel_scale: np.ndarray,
    epochs: int,
    seed: int,
) -> np.ndarray:
    weights = recording_class_weights(data.train_recordings, outer_train)
    train_sequence = make_sequence(
        data, "train", outer_train, candidate, True, True, weights, seed
    )
    validation_sequence = make_sequence(
        data, "train", outer_validation, candidate, False, False, seed=seed
    )
    tf.keras.backend.clear_session()
    set_seed(seed)
    model = build_model(candidate, channel_mean, channel_scale)
    model.fit(train_sequence, epochs=epochs, verbose=0)
    scores = model.predict(validation_sequence, verbose=0).reshape(-1)
    if scores.shape != (len(outer_validation),) or not np.isfinite(scores).all():
        raise RuntimeError("Scores OOF MIL invalidos.")
    return scores


def evaluate_candidate_oof(
    data: TemporalMILData,
    candidate: MILCandidate,
) -> CandidateEvaluation:
    started = time.perf_counter()
    recordings = data.train_recordings
    folds = recordings["fold"].to_numpy(dtype=int)
    labels = recordings["stage2_target"].to_numpy(dtype=int)
    oof_scores = np.full(len(recordings), np.nan, dtype=float)
    training_rows: list[dict[str, Any]] = []
    parameter_count: int | None = None

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        outer_validation = np.flatnonzero(folds == fold)
        outer_train = np.flatnonzero(folds != fold)
        seed = RANDOM_STATE + fold
        inner_train, inner_validation = train_test_split(
            outer_train,
            test_size=INNER_VALIDATION_SIZE,
            stratify=labels[outer_train],
            random_state=seed,
        )
        inner_mean, inner_scale = fit_channel_scaler(
            data.x_train, data.train_event_indices, np.asarray(inner_train)
        )
        best_epoch, best_loss, epochs_run, current_parameters = (
            fit_early_stopping(
                data,
                candidate,
                np.asarray(inner_train),
                np.asarray(inner_validation),
                inner_mean,
                inner_scale,
                seed,
            )
        )
        if parameter_count is None:
            parameter_count = current_parameters
        elif parameter_count != current_parameters:
            raise RuntimeError("Los parametros cambiaron entre folds.")
        outer_mean, outer_scale = fit_channel_scaler(
            data.x_train, data.train_event_indices, outer_train
        )
        fold_scores = refit_and_predict_outer(
            data,
            candidate,
            outer_train,
            outer_validation,
            outer_mean,
            outer_scale,
            best_epoch,
            seed,
        )
        oof_scores[outer_validation] = fold_scores
        native = common.binary_metrics(
            labels[outer_validation], fold_scores, 0.5
        )
        training_rows.append(
            {
                "candidate_key": candidate.key,
                "outer_fold": fold,
                "outer_train_recordings": len(outer_train),
                "inner_train_recordings": len(inner_train),
                "inner_validation_recordings": len(inner_validation),
                "outer_validation_recordings": len(outer_validation),
                "epochs_run_inner": epochs_run,
                "best_epoch_inner": best_epoch,
                "best_inner_val_loss": best_loss,
                "outer_refit_epochs": best_epoch,
                "outer_macro_f1_at_0p5": native["macro_f1"],
                "outer_wet_recall_at_0p5": native["wet_recall"],
            }
        )
        print(
            f"  Fold {fold}: best_epoch={best_epoch}, "
            f"macro-F1 OOF@0.5={native['macro_f1']:.4f}, "
            f"wet-recall@0.5={native['wet_recall']:.4f}"
        )

    if not np.isfinite(oof_scores).all():
        raise RuntimeError(f"OOF incompleto para {candidate.key}.")
    predictions = prediction_frame(recordings, oof_scores)
    threshold = common.tune_threshold(labels, oof_scores, 0.5)
    metrics = common.binary_metrics(labels, oof_scores, threshold)
    metrics_fixed = common.binary_metrics(labels, oof_scores, 0.5)
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


def candidate_results_frame(results: list[CandidateEvaluation]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "candidate_key": result.candidate.key,
                "pooling": result.candidate.pooling,
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
        ["oof_tuned__macro_f1", "oof_tuned__balanced_accuracy", "oof_tuned__roc_auc"],
        ascending=False,
    )


def fit_final_model(
    data: TemporalMILData,
    winner: CandidateEvaluation,
) -> tuple[tf.keras.Model, int, np.ndarray, np.ndarray]:
    epochs = max(
        1,
        int(np.rint(np.median(
            winner.training_summary["best_epoch_inner"].to_numpy(dtype=int)
        ))),
    )
    all_indices = np.arange(len(data.train_recordings), dtype=int)
    mean, scale = fit_channel_scaler(
        data.x_train, data.train_event_indices, all_indices
    )
    weights = recording_class_weights(data.train_recordings, all_indices)
    sequence = make_sequence(
        data,
        "train",
        all_indices,
        winner.candidate,
        True,
        True,
        weights,
        RANDOM_STATE,
    )
    tf.keras.backend.clear_session()
    set_seed(RANDOM_STATE)
    model = build_model(winner.candidate, mean, scale)
    model.fit(sequence, epochs=epochs, verbose=2)
    return model, epochs, mean, scale


def save_attention_weights(
    model: tf.keras.Model,
    data: TemporalMILData,
    output_path: Path,
) -> None:
    if "gated_attention_pooling" not in [layer.name for layer in model.layers]:
        return
    encoder = model.get_layer("encode_events")
    attention = model.get_layer("gated_attention_pooling")
    rows: list[dict[str, Any]] = []
    for recording_index, recording in data.validation_recordings.iterrows():
        indices = data.validation_event_indices[recording_index]
        events = np.transpose(
            np.asarray(data.x_validation[indices], dtype=np.float32),
            (0, 2, 1),
        )[np.newaxis, ...]
        mask = np.ones((1, len(indices)), dtype=np.float32)
        embeddings = encoder(events, training=False)
        weights = attention.compute_attention(embeddings, mask).numpy()[0]
        event_rows = data.validation_event_metadata.iloc[indices]
        for event_position, (_, event) in enumerate(event_rows.iterrows()):
            rows.append(
                {
                    "original_uuid": recording["original_uuid"],
                    "event_id": event["event_id"],
                    "event_index": int(event["event_index"]),
                    "event_position_in_bag": event_position,
                    "attention_weight": float(weights[event_position]),
                    "event_count": len(indices),
                    "y_true": int(recording["stage2_target"]),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def metric_row(
    split: str,
    policy: str,
    threshold: float,
    predictions: pd.DataFrame,
    winner: CandidateEvaluation,
    model_size_kb: float,
) -> dict[str, Any]:
    return {
        "split": split,
        "experiment": EXPERIMENT_KEY,
        "candidate_key": winner.candidate.key,
        "pooling": winner.candidate.pooling,
        "threshold_policy": policy,
        "threshold": threshold,
        "parameter_count": winner.parameter_count,
        "model_size_kb_float32_keras": model_size_kb,
        **common.binary_metrics(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["score"].to_numpy(dtype=float),
            threshold,
        ),
    }


def train(
    data: TemporalMILData,
    extraction_configuration: pd.DataFrame,
    preset: str,
    quick: bool,
) -> None:
    mode = "quick" if quick else "full"
    result_dir = RESULTS_ROOT / preset / mode
    model_dir = MODELS_ROOT / preset / mode
    graph_dir = GRAPHS_ROOT / preset / mode
    result_dir.mkdir(parents=True, exist_ok=True)
    specs = candidate_specs(quick)
    results: list[CandidateEvaluation] = []
    print("\n" + "=" * 78)
    print("CV - WST TEMPORAL + TINY CNN + MIL RECORDING-LEVEL")
    print("=" * 78)
    for index, candidate in enumerate(specs, start=1):
        print(f"\n[{index}/{len(specs)}] {candidate.key}")
        result = evaluate_candidate_oof(data, candidate)
        results.append(result)
        print(
            f"{candidate.pooling}: macro-F1={result.metrics['macro_f1']:.4f} | "
            f"bal-acc={result.metrics['balanced_accuracy']:.4f} | "
            f"AUC={result.metrics['roc_auc']:.4f} | params={result.parameter_count}"
        )
    winner = max(results, key=selection_key)
    candidate_results_frame(results).to_csv(
        result_dir / "candidate_cv_results.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(
        [item.training_summary for item in results], ignore_index=True
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
        print("\nPrueba rapida MIL completada.")
        print(f"Macro-F1 OOF: {winner.metrics['macro_f1']:.4f}")
        print("VALIDATION no se ha evaluado.")
        print(f"Resultados: {result_dir}")
        return

    print("\nAjustando CNN MIL final exclusivamente con todo TRAIN...")
    model, final_epochs, channel_mean, channel_scale = fit_final_model(data, winner)
    validation_indices = np.arange(len(data.validation_recordings), dtype=int)
    validation_sequence = make_sequence(
        data, "validation", validation_indices, winner.candidate,
        False, False
    )
    validation_scores = model.predict(validation_sequence, verbose=0).reshape(-1)
    validation_predictions = prediction_frame(
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
        result_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "wst_temporal_mil_cnn.keras"
    model.save(model_path)
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
    if winner.candidate.pooling == "attention":
        save_attention_weights(
            model, data, result_dir / "validation_event_attention_weights.csv"
        )
    configuration = {
        "preset": preset,
        "unit_of_training": "recording_bag",
        "event_label_loss_used": False,
        "recording_label_loss_used": True,
        "max_events_per_recording": data.max_events,
        "events_discarded": False,
        "wst_input_per_event": "5_time_positions_x_644_paths",
        "channel_standardization": "fit_equal_weight_per_recording_inside_fold",
        "winner": winner.candidate.key,
        "threshold_policy": "oof_tuned_frozen",
        "threshold": winner.threshold,
        "final_epochs_from_cv_median": final_epochs,
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
        result_dir / "experiment_configuration.csv", index=False, encoding="utf-8-sig"
    )
    graph_path = graph_dir / "validation_wst_temporal_mil_cnn.png"
    graph_helpers.create_validation_graph(
        validation_predictions, winner.threshold, winner.candidate.key, graph_path
    )
    validation_metrics = validation_rows[-1]
    print("\n" + "=" * 78)
    print("RESULTADO WST TEMPORAL + TINY CNN + MIL")
    print("=" * 78)
    print(f"Pooling ganador: {winner.candidate.pooling}")
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
    print(f"Grafica: {graph_path}")
    print("TEST permanece reservado.")


def print_check(data: TemporalMILData, quick: bool) -> None:
    train_counts = data.train_recordings["stage2_target"].value_counts()
    validation_counts = data.validation_recordings["stage2_target"].value_counts()
    print("=" * 78)
    print("CHECK - WST TEMPORAL + CNN MIL RECORDING-LEVEL")
    print("=" * 78)
    print(f"Eventos TRAIN/VALIDATION: {len(data.x_train)} / {len(data.x_validation)}")
    print(
        f"Grabaciones TRAIN/VALIDATION: {len(data.train_recordings)} / "
        f"{len(data.validation_recordings)}"
    )
    print(f"TRAIN dry/wet: {train_counts[0]} / {train_counts[1]}")
    print(f"VALIDATION dry/wet: {validation_counts[0]} / {validation_counts[1]}")
    print(f"Eventos maximos por grabacion: {data.max_events}")
    print("Entrada evento CNN: (5 posiciones temporales, 644 caminos).")
    print("Una etiqueta y una perdida por original_uuid; no por evento.")
    print(f"Candidatos: {len(candidate_specs(quick))}")
    probe_indices = np.arange(len(data.train_recordings), dtype=int)
    mean, scale = fit_channel_scaler(
        data.x_train, data.train_event_indices, probe_indices
    )
    for candidate in candidate_specs(quick):
        tf.keras.backend.clear_session()
        model = build_model(candidate, mean, scale)
        print(f"  {candidate.pooling}: {model.count_params()} parametros")
    print("Quick evalua solo attention con OOF; no consulta VALIDATION.")
    print("Full compara mean, mean_max y attention; TEST permanece cerrado.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena tiny CNN MIL sobre WST temporal para dry/wet."
    )
    parser.add_argument("--action", choices=["check", "train"], default="check")
    parser.add_argument("--preset", choices=PRESETS, default=DEFAULT_PRESET)
    parser.add_argument(
        "--quick", action="store_true",
        help="Solo attention, menos epocas y sin evaluar VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - WST TEMPORAL + TINY CNN + MIL RECORDING-LEVEL")
    print("=" * 78)
    print(f"Preset: {args.preset} | Accion: {args.action} | Quick: {args.quick}")
    print("TEST no sera leido ni procesado.")
    data, extraction_configuration = load_data(args.preset)
    print_check(data, args.quick)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(data, extraction_configuration, args.preset, args.quick)


if __name__ == "__main__":
    main()
