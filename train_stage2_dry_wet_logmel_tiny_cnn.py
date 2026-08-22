"""Entrena una tiny CNN Log-Mel para Stage 2 dry/wet.

La CNN recibe un evento Log-Mel de 1,5 s y produce una probabilidad wet.
Los folds se definen por original_uuid, cada grabacion aporta el mismo peso
total y las probabilidades de sus eventos se promedian antes de calcular
metricas. Arquitectura, epocas y umbral se seleccionan exclusivamente con
OOF de TRAIN. VALIDATION se usa una sola vez en el modo completo y TEST no
se lee.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train_stage2_dry_wet_cochleograms as common


ROOT = Path(__file__).resolve().parent
FEATURES_ROOT = ROOT / "features_extracted_stage2_dry_wet_logmel"
RESULTS_ROOT = ROOT / "results_stage2_dry_wet_logmel_tiny_cnn"
MODELS_ROOT = ROOT / "models_stage2_dry_wet_logmel_tiny_cnn"
GRAPHS_ROOT = (
    ROOT
    / "graphs_results_stage2_dry_wet_random"
    / "training_logmel_tiny_cnn"
)

PRESETS = ("logmel64_win32_hop16",)
EXPECTED_INPUT_SHAPE = (64, 92, 1)
RANDOM_STATE = 42


@dataclass(frozen=True)
class CNNCandidate:
    name: str
    filters: tuple[int, int, int]
    dense_units: int
    dropout_rate: float
    l2_strength: float
    learning_rate: float
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 7

    @property
    def key(self) -> str:
        filter_name = "x".join(map(str, self.filters))
        dropout = str(self.dropout_rate).replace(".", "p")
        l2_name = f"{self.l2_strength:g}".replace(".", "p")
        lr_name = f"{self.learning_rate:g}".replace(".", "p")
        return (
            f"{self.name}__f{filter_name}__dense{self.dense_units}__"
            f"drop{dropout}__l2{l2_name}__lr{lr_name}"
        )


@dataclass
class LogMelData:
    x_train: np.ndarray
    train_metadata: pd.DataFrame
    x_validation: np.ndarray
    validation_metadata: pd.DataFrame


@dataclass
class CandidateEvaluation:
    candidate: CNNCandidate
    threshold: float
    metrics: dict[str, float | int]
    oof_recordings: pd.DataFrame
    fold_metrics: pd.DataFrame
    training_summary: pd.DataFrame
    parameter_count: int
    elapsed_seconds: float


def feature_paths(input_dir: Path) -> dict[str, Path]:
    return {
        "x_train": input_dir / "X_events_train.npy",
        "y_train": input_dir / "y_events_train.npy",
        "folds_train": input_dir / "folds_events_train.npy",
        "metadata_train": input_dir / "metadata_events_features_train.csv",
        "x_validation": input_dir / "X_events_validation.npy",
        "y_validation": input_dir / "y_events_validation.npy",
        "folds_validation": input_dir / "folds_events_validation.npy",
        "metadata_validation": (
            input_dir / "metadata_events_features_validation.csv"
        ),
        "configuration": input_dir / "logmel_configuration.csv",
    }


def validate_split(
    split_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    folds: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    required_columns = {
        "feature_row",
        "event_id",
        "original_uuid",
        "cough_type",
        "cough_type_consensus",
        "stage2_target",
        "fold",
        "split",
    }
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en metadata {split_name}: {sorted(missing)}"
        )
    expected_shape = (len(metadata), *EXPECTED_INPUT_SHAPE)
    if x_values.shape != expected_shape:
        raise ValueError(
            f"X {split_name} tiene forma {x_values.shape}; "
            f"esperada {expected_shape}."
        )
    if x_values.dtype != np.float32:
        raise ValueError(f"X debe ser float32, no {x_values.dtype}.")
    if not np.isfinite(x_values).all():
        raise ValueError(f"X {split_name} contiene NaN o infinito.")
    if float(x_values.min()) < 0.0 or float(x_values.max()) > 1.0:
        raise ValueError(f"X {split_name} no esta dentro de [0, 1].")
    if y_values.shape != (len(metadata),):
        raise ValueError(f"Forma de y {split_name} invalida: {y_values.shape}.")
    if folds.shape != (len(metadata),):
        raise ValueError(
            f"Forma de folds {split_name} invalida: {folds.shape}."
        )
    if metadata["feature_row"].tolist() != list(range(len(metadata))):
        raise ValueError("feature_row no coincide con el orden de X.")
    if metadata["event_id"].duplicated().any():
        raise ValueError(f"Hay event_id duplicados en {split_name}.")
    if not np.array_equal(
        y_values.astype(int),
        metadata["stage2_target"].to_numpy(dtype=int),
    ):
        raise ValueError("y no coincide con stage2_target.")
    if not np.array_equal(
        folds.astype(int),
        metadata["fold"].to_numpy(dtype=int),
    ):
        raise ValueError("folds no coincide con metadata.")
    expected_targets = metadata["cough_type"].map({"dry": 0, "wet": 1})
    if expected_targets.isna().any() or not np.array_equal(
        expected_targets.to_numpy(dtype=int),
        y_values.astype(int),
    ):
        raise ValueError("cough_type y stage2_target no coinciden.")
    if set(y_values.astype(int)) != {0, 1}:
        raise ValueError("Las clases deben ser exactamente dry=0 y wet=1.")
    if set(metadata["split"].astype(str)) != {split_name}:
        raise ValueError(f"La columna split no coincide con {split_name}.")

    unique_folds = set(folds.astype(int))
    if split_name == "train" and unique_folds != common.EXPECTED_TRAIN_FOLDS:
        raise ValueError(f"Folds TRAIN inesperados: {unique_folds}.")
    if split_name == "validation" and unique_folds != {-1}:
        raise ValueError(f"Folds VALIDATION inesperados: {unique_folds}.")
    grouped = metadata.groupby("original_uuid")
    if grouped["stage2_target"].nunique().max() != 1:
        raise ValueError("Una grabacion contiene varias etiquetas.")
    if grouped["fold"].nunique().max() != 1:
        raise ValueError("Una grabacion aparece en varios folds.")


def load_data(preset: str) -> tuple[LogMelData, pd.DataFrame]:
    input_dir = FEATURES_ROOT / preset
    paths = feature_paths(input_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos Log-Mel. Ejecuta primero el extractor:\n"
            + "\n".join(missing)
        )

    x_train = np.load(paths["x_train"], mmap_mode="r")
    y_train = np.load(paths["y_train"])
    folds_train = np.load(paths["folds_train"])
    train_metadata = pd.read_csv(paths["metadata_train"])
    x_validation = np.load(paths["x_validation"], mmap_mode="r")
    y_validation = np.load(paths["y_validation"])
    folds_validation = np.load(paths["folds_validation"])
    validation_metadata = pd.read_csv(paths["metadata_validation"])
    configuration = pd.read_csv(paths["configuration"])

    if len(configuration) != 1:
        raise ValueError("La configuracion Log-Mel debe tener una sola fila.")
    config = configuration.iloc[0]
    observed_shape = (
        int(config["n_mels"]),
        int(config["n_frames"]),
        1,
    )
    if observed_shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"Configuracion Log-Mel incompatible: {observed_shape}."
        )
    if str(config["scaling"]) != (
        "per_event_relative_power_db_clipped_then_0_1"
    ):
        raise ValueError("Escalado Log-Mel no reconocido.")

    validate_split(
        "train", x_train, y_train, folds_train, train_metadata
    )
    validate_split(
        "validation",
        x_validation,
        y_validation,
        folds_validation,
        validation_metadata,
    )
    overlap = set(train_metadata["original_uuid"].astype(str)) & set(
        validation_metadata["original_uuid"].astype(str)
    )
    if overlap:
        raise ValueError(
            f"TRAIN y VALIDATION comparten UUID: {sorted(overlap)[:10]}"
        )

    return (
        LogMelData(
            x_train=x_train,
            train_metadata=train_metadata,
            x_validation=x_validation,
            validation_metadata=validation_metadata,
        ),
        configuration,
    )


def candidate_specs(quick: bool) -> list[CNNCandidate]:
    if quick:
        return [
            CNNCandidate(
                name="tiny",
                filters=(8, 16, 32),
                dense_units=16,
                dropout_rate=0.15,
                l2_strength=1e-4,
                learning_rate=1e-3,
                max_epochs=12,
                patience=3,
            )
        ]
    return [
        CNNCandidate(
            name="tiny",
            filters=(8, 16, 32),
            dense_units=16,
            dropout_rate=0.15,
            l2_strength=1e-4,
            learning_rate=1e-3,
        ),
        CNNCandidate(
            name="compact_regularized",
            filters=(12, 24, 48),
            dense_units=24,
            dropout_rate=0.25,
            l2_strength=3e-4,
            learning_rate=5e-4,
        ),
        CNNCandidate(
            name="small",
            filters=(16, 32, 64),
            dense_units=32,
            dropout_rate=0.30,
            l2_strength=5e-4,
            learning_rate=5e-4,
        ),
    ]


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_model(
    candidate: CNNCandidate,
    input_shape: tuple[int, int, int] = EXPECTED_INPUT_SHAPE,
) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.l2(candidate.l2_strength)
    inputs = tf.keras.Input(shape=input_shape, name="logmel")

    x = tf.keras.layers.Conv2D(
        candidate.filters[0],
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name="conv_initial",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_initial")(x)
    x = tf.keras.layers.ReLU(name="relu_initial")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="pool_initial")(x)

    for block_index, filters in enumerate(candidate.filters[1:], start=1):
        x = tf.keras.layers.SeparableConv2D(
            filters,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            depthwise_regularizer=regularizer,
            pointwise_regularizer=regularizer,
            name=f"separable_{block_index}",
        )(x)
        x = tf.keras.layers.BatchNormalization(
            name=f"bn_separable_{block_index}"
        )(x)
        x = tf.keras.layers.ReLU(
            name=f"relu_separable_{block_index}"
        )(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average")(x)
    x = tf.keras.layers.Dense(
        candidate.dense_units,
        activation="relu",
        kernel_regularizer=regularizer,
        name="dense_embedding",
    )(x)
    x = tf.keras.layers.Dropout(
        candidate.dropout_rate,
        name="dropout",
    )(x)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="wet_probability",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=candidate.name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=candidate.learning_rate
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        weighted_metrics=[],
    )
    return model


def fold_metrics(
    recordings: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for fold, group in recordings.groupby("fold", sort=True):
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


def evaluate_candidate_oof(
    data: LogMelData,
    candidate: CNNCandidate,
) -> CandidateEvaluation:
    start = time.perf_counter()
    metadata = data.train_metadata
    folds = metadata["fold"].to_numpy(dtype=int)
    y_all = metadata["stage2_target"].to_numpy(dtype=np.float32)
    oof_scores = np.full(len(metadata), np.nan, dtype=float)
    training_rows: list[dict[str, float | int | str]] = []
    parameter_count: int | None = None

    for fold in sorted(common.EXPECTED_TRAIN_FOLDS):
        validation_mask = folds == fold
        training_mask = ~validation_mask
        metadata_fit = metadata.loc[training_mask].reset_index(drop=True)
        metadata_fold = metadata.loc[validation_mask].reset_index(drop=True)
        fit_weights = common.compute_training_weights(metadata_fit).astype(
            np.float32
        )
        validation_weights = common.compute_training_weights(
            metadata_fold
        ).astype(np.float32)

        tf.keras.backend.clear_session()
        seed = RANDOM_STATE + fold
        set_seed(seed)
        model = build_model(candidate)
        if parameter_count is None:
            parameter_count = int(model.count_params())
        elif parameter_count != int(model.count_params()):
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
            data.x_train[training_mask],
            y_all[training_mask],
            sample_weight=fit_weights,
            validation_data=(
                data.x_train[validation_mask],
                y_all[validation_mask],
                validation_weights,
            ),
            epochs=candidate.max_epochs,
            batch_size=candidate.batch_size,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=0,
        )
        fold_scores = model.predict(
            data.x_train[validation_mask],
            batch_size=candidate.batch_size,
            verbose=0,
        ).reshape(-1)
        if not np.isfinite(fold_scores).all():
            raise RuntimeError(f"Scores no finitos en fold {fold}.")
        oof_scores[validation_mask] = fold_scores

        validation_losses = np.asarray(
            history.history["val_loss"], dtype=float
        )
        best_epoch = int(np.argmin(validation_losses) + 1)
        training_rows.append(
            {
                "candidate_key": candidate.key,
                "fold": fold,
                "seed": seed,
                "epochs_run": len(validation_losses),
                "best_epoch": best_epoch,
                "best_val_loss": float(validation_losses[best_epoch - 1]),
                "final_train_loss": float(history.history["loss"][-1]),
            }
        )
        print(
            f"  {candidate.name} fold {fold}: "
            f"best_epoch={best_epoch}, "
            f"val_loss={validation_losses[best_epoch - 1]:.5f}"
        )
        del model

    if not np.isfinite(oof_scores).all():
        raise RuntimeError(f"OOF incompleto para {candidate.key}.")
    recordings = common.aggregate_scores_by_recording(metadata, oof_scores)
    y_true = recordings["y_true"].to_numpy(dtype=int)
    scores = recordings["score"].to_numpy(dtype=float)
    threshold = common.tune_threshold(y_true, scores, 0.5)
    metrics = common.binary_metrics(y_true, scores, threshold)
    recordings["y_pred"] = (scores >= threshold).astype(int)
    recordings["candidate_key"] = candidate.key

    return CandidateEvaluation(
        candidate=candidate,
        threshold=threshold,
        metrics=metrics,
        oof_recordings=recordings,
        fold_metrics=fold_metrics(recordings, threshold),
        training_summary=pd.DataFrame(training_rows),
        parameter_count=int(parameter_count),
        elapsed_seconds=time.perf_counter() - start,
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
                "filters": "x".join(map(str, result.candidate.filters)),
                "dense_units": result.candidate.dense_units,
                "dropout_rate": result.candidate.dropout_rate,
                "l2_strength": result.candidate.l2_strength,
                "learning_rate": result.candidate.learning_rate,
                "batch_size": result.candidate.batch_size,
                "max_epochs": result.candidate.max_epochs,
                "patience": result.candidate.patience,
                "parameter_count": result.parameter_count,
                "threshold_oof": result.threshold,
                "elapsed_seconds": result.elapsed_seconds,
                **result.metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy", "roc_auc"],
        ascending=False,
    )


def create_validation_graph(
    predictions: pd.DataFrame,
    threshold: float,
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
                color=(
                    "white"
                    if matrix[row, column] > matrix.max() / 2
                    else "black"
                ),
            )
    axes[0].set_xticks([0, 1], ["Dry", "Wet"])
    axes[0].set_yticks([0, 1], ["Dry", "Wet"])
    axes[0].set_xlabel("Prediccion")
    axes[0].set_ylabel("Etiqueta real")
    axes[0].set_title("Matriz de confusion")
    figure.colorbar(image, ax=axes[0], fraction=0.046)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    axes[1].plot(false_positive_rate, true_positive_rate, linewidth=2)
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC recording-level")
    axes[1].grid(alpha=0.25)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    axes[2].plot(recall, precision, linewidth=2)
    axes[2].set_xlabel("Recall wet")
    axes[2].set_ylabel("Precision wet")
    axes[2].set_title("Precision-Recall wet")
    axes[2].grid(alpha=0.25)

    figure.suptitle(
        f"Tiny CNN Log-Mel - VALIDATION - umbral OOF={threshold:.4f}"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def fit_final_model(
    data: LogMelData,
    winner: CandidateEvaluation,
) -> tuple[tf.keras.Model, int]:
    best_epochs = winner.training_summary["best_epoch"].to_numpy(dtype=int)
    final_epochs = max(1, int(np.rint(np.median(best_epochs))))
    weights = common.compute_training_weights(data.train_metadata).astype(
        np.float32
    )
    y_train = data.train_metadata["stage2_target"].to_numpy(np.float32)

    tf.keras.backend.clear_session()
    set_seed(RANDOM_STATE)
    model = build_model(winner.candidate)
    model.fit(
        data.x_train,
        y_train,
        sample_weight=weights,
        epochs=final_epochs,
        batch_size=winner.candidate.batch_size,
        shuffle=True,
        verbose=2,
    )
    return model, final_epochs


def train(
    data: LogMelData,
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
    print("\n" + "=" * 78)
    print("CV LOG-MEL - TINY CNN - METRICAS RECORDING-LEVEL")
    print("=" * 78)
    for candidate in candidate_specs(quick):
        print(f"\nCandidato: {candidate.key}")
        result = evaluate_candidate_oof(data, candidate)
        results.append(result)
        print(
            f"{candidate.name}: macro-F1={result.metrics['macro_f1']:.4f} | "
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
        [result.training_summary for result in results],
        ignore_index=True,
    ).to_csv(
        result_dir / "cv_training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    winner.oof_recordings.to_csv(
        result_dir / "best_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    winner.fold_metrics.to_csv(
        result_dir / "best_cv_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if quick:
        print("\nPrueba rapida completada.")
        print(f"Mejor OOF: {winner.candidate.key}")
        print(f"Macro-F1 OOF: {winner.metrics['macro_f1']:.4f}")
        print("VALIDATION no se ha evaluado.")
        print(f"Resultados: {result_dir}")
        return

    print("\nAjustando la CNN final exclusivamente con todo TRAIN...")
    final_model, final_epochs = fit_final_model(data, winner)
    validation_event_scores = final_model.predict(
        data.x_validation,
        batch_size=winner.candidate.batch_size,
        verbose=0,
    ).reshape(-1)
    validation_predictions = common.aggregate_scores_by_recording(
        data.validation_metadata,
        validation_event_scores,
    )
    validation_predictions["y_pred"] = (
        validation_predictions["score"].to_numpy(dtype=float)
        >= winner.threshold
    ).astype(int)
    validation_predictions.to_csv(
        result_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_metrics = common.binary_metrics(
        validation_predictions["y_true"].to_numpy(dtype=int),
        validation_predictions["score"].to_numpy(dtype=float),
        winner.threshold,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "logmel_tiny_cnn.keras"
    final_model.save(model_path)
    model_size_kb = model_path.stat().st_size / 1024.0

    metrics_rows = [
        {
            "split": "train_oof",
            "threshold_policy": "oof_tuned",
            "threshold": winner.threshold,
            **winner.metrics,
        },
        {
            "split": "validation",
            "threshold_policy": "oof_tuned_frozen",
            "threshold": winner.threshold,
            **validation_metrics,
        },
    ]
    pd.DataFrame(metrics_rows).to_csv(
        result_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    experiment_configuration = {
        "preset": preset,
        "unit_of_training": "event",
        "unit_of_evaluation": "recording",
        "recording_score_pooling": "mean_event_probability",
        "class_balance": "balanced_class_and_equal_total_per_recording",
        "winner": winner.candidate.key,
        "threshold_policy": "oof_tuned_frozen",
        "threshold": winner.threshold,
        "final_epochs_from_cv_median": final_epochs,
        "parameter_count": winner.parameter_count,
        "model_size_kb_float32_keras": model_size_kb,
        "augmentation": "none_initial_baseline",
        "test_read": False,
        **{
            f"logmel_{key}": value
            for key, value in extraction_configuration.iloc[0].to_dict().items()
        },
        **{
            f"cnn_{key}": value
            for key, value in asdict(winner.candidate).items()
        },
    }
    pd.DataFrame([experiment_configuration]).to_csv(
        result_dir / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    graph_path = graph_dir / "validation_logmel_tiny_cnn.png"
    create_validation_graph(
        validation_predictions,
        winner.threshold,
        graph_path,
    )

    print("\n" + "=" * 78)
    print("RESULTADO LOG-MEL + TINY CNN")
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
    print(f"Parametros: {winner.parameter_count}")
    print(f"Modelo Keras float32: {model_size_kb:.2f} KB")
    print(f"Resultados: {result_dir}")
    print(f"Grafica: {graph_path}")
    print("TEST permanece reservado.")


def print_check(data: LogMelData, quick: bool) -> None:
    train_recordings = data.train_metadata.drop_duplicates("original_uuid")
    validation_recordings = data.validation_metadata.drop_duplicates(
        "original_uuid"
    )
    print("=" * 78)
    print("CHECK - ENTRENAMIENTO LOG-MEL TINY CNN")
    print("=" * 78)
    print(f"X TRAIN: {data.x_train.shape} {data.x_train.dtype}")
    print(f"X VALIDATION: {data.x_validation.shape} {data.x_validation.dtype}")
    print(
        "Grabaciones TRAIN dry/wet: "
        f"{(train_recordings['stage2_target'] == 0).sum()} / "
        f"{(train_recordings['stage2_target'] == 1).sum()}"
    )
    print(
        "Grabaciones VALIDATION dry/wet: "
        f"{(validation_recordings['stage2_target'] == 0).sum()} / "
        f"{(validation_recordings['stage2_target'] == 1).sum()}"
    )
    print(f"Candidatos: {len(candidate_specs(quick))}")
    for candidate in candidate_specs(quick):
        tf.keras.backend.clear_session()
        model = build_model(candidate)
        print(f"  {candidate.name}: {model.count_params()} parametros")
    print("Prediccion por evento; evaluacion y umbral por original_uuid.")
    print("VALIDATION solo se evaluara en entrenamiento full. TEST no se lee.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena tiny CNN Log-Mel para Stage 2 dry/wet."
    )
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default=PRESETS[0],
    )
    parser.add_argument(
        "--action",
        choices=["check", "train"],
        default="check",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Una arquitectura y pocas epocas; no evalua VALIDATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("STAGE 2 - LOG-MEL + TINY DEPTHWISE CNN")
    print("=" * 78)
    print(f"Preset: {args.preset} | Accion: {args.action}")
    print("TEST no sera leido ni procesado.")

    data, extraction_configuration = load_data(args.preset)
    print_check(data, args.quick)
    if args.action == "check":
        print("Comprobacion completada. No se entreno ningun modelo.")
        return
    train(data, extraction_configuration, args.preset, args.quick)


if __name__ == "__main__":
    main()
