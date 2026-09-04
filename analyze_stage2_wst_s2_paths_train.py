"""Analisis exploratorio recording-level de paths WST de segundo orden.

El script usa exclusivamente TRAIN. Reduce el eje temporal de cada evento con
mean(S), agrega despues los eventos mediante una media por ``original_uuid`` y
compara directamente las distribuciones dry/wet de cada path S2.

No carga VALIDATION ni TEST, no entrena modelos y no aplica PCA.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import kymatio
import kymatio.scattering1d.filter_bank as kymatio_filter_bank
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_ROOT = (
    SCRIPT_DIR
    / "features_extracted_stage2_dry_wet_wavelet_scattering_temporal"
)
OUTPUT_ROOT = SCRIPT_DIR / "analysis_stage2_dry_wet_wst_s2_paths"

DEFAULT_PRESET = "paper_q8_q1_t500_full"
SAMPLE_RATE = 16_000
EXPECTED_RECORDINGS = 1_623
EXPECTED_S2_PATHS = 549
LOADED_SPLITS = ("train",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interpreta los paths S2 WST usando una observacion por "
            "grabacion de TRAIN."
        )
    )
    parser.add_argument(
        "--action",
        choices=("check", "analyze"),
        default="check",
        help="check valida entradas; analyze genera CSV y figuras.",
    )
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def feature_directory(preset: str) -> Path:
    return FEATURE_ROOT / preset


def required_train_files() -> tuple[str, ...]:
    """Lista deliberadamente limitada a TRAIN y metadata global de paths."""

    return (
        "X_events_train.npy",
        "y_events_train.npy",
        "folds_events_train.npy",
        "metadata_events_features_train.csv",
        "wavelet_scattering_temporal_path_layout.csv",
        "wavelet_scattering_temporal_configuration.csv",
    )


def verify_xi_frequency_convention(path_layout: pd.DataFrame) -> str:
    """Verifica que xi esta expresada en ciclos por muestra.

    Kymatio 0.3.0 construye los filtros Morlet sobre la rejilla devuelta por
    ``numpy.fft.fftfreq`` y documenta ``xi`` como su frecuencia central. Esa
    rejilla, cuando no se proporciona un periodo fisico, usa ciclos/muestra.
    Por ello, la conversion a Hz es ``xi * sample_rate``.
    """

    source = inspect.getsource(kymatio_filter_bank.morlet_1d)
    if "np.fft.fftfreq(N)" not in source:
        raise RuntimeError(
            "No se pudo verificar en la version instalada de Kymatio que "
            "xi usa la rejilla normalizada de numpy.fft.fftfreq."
        )

    for column in ("xi_1", "xi_2"):
        values = path_layout[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contiene valores ausentes o no finitos.")
        if np.any(values <= 0.0) or np.any(values > 0.5):
            raise ValueError(
                f"{column} no pertenece al intervalo esperado (0, 0.5] "
                "para frecuencias normalizadas de tiempo discreto."
            )

    return (
        f"Kymatio {kymatio.__version__}: xi verificada como frecuencia "
        "normalizada en ciclos/muestra mediante morlet_1d y "
        "numpy.fft.fftfreq; Hz = xi * 16000"
    )


def load_train_only(
    preset: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.Series, str]:
    """Carga solo el tensor y la metadata de TRAIN."""

    if LOADED_SPLITS != ("train",):
        raise RuntimeError("La lista de splits permitidos debe contener solo TRAIN.")

    directory = feature_directory(preset)
    missing = [
        filename
        for filename in required_train_files()
        if not (directory / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Faltan entradas en {directory}: {', '.join(missing)}"
        )

    # No se referencia, abre ni carga ningun fichero de VALIDATION o TEST.
    x_train = np.load(directory / "X_events_train.npy", mmap_mode="r")
    y_train = np.load(directory / "y_events_train.npy", mmap_mode="r")
    metadata = pd.read_csv(directory / "metadata_events_features_train.csv")
    path_layout = pd.read_csv(
        directory / "wavelet_scattering_temporal_path_layout.csv"
    )
    configuration = pd.read_csv(
        directory / "wavelet_scattering_temporal_configuration.csv"
    )

    if len(configuration) != 1:
        raise ValueError("La configuracion WST debe contener exactamente una fila.")
    config = configuration.iloc[0]
    expected = {
        "sample_rate": SAMPLE_RATE,
        "J": 13,
        "max_order": 2,
        "invariance_samples": 8000,
    }
    for column, expected_value in expected.items():
        if int(config[column]) != expected_value:
            raise ValueError(
                f"Configuracion inesperada: {column}={config[column]}; "
                f"se esperaba {expected_value}."
            )
    q_text = str(config["Q"]).replace(" ", "")
    if q_text not in {"(8,1)", "[8,1]"}:
        raise ValueError(f"Se esperaba Q=(8,1), recibido {q_text}.")
    if str(config["stored_axis_order"]) != "event,path,time":
        raise ValueError("El tensor no tiene el orden event,path,time esperado.")

    required_metadata = {
        "feature_row",
        "original_uuid",
        "stage2_target",
        "cough_type",
        "split",
    }
    missing_metadata = required_metadata - set(metadata.columns)
    if missing_metadata:
        raise ValueError(
            "Faltan columnas en la metadata TRAIN: "
            + ", ".join(sorted(missing_metadata))
        )
    required_layout = {
        "path_index",
        "path_name",
        "scattering_order",
        "j_1",
        "j_2",
        "xi_1",
        "xi_2",
    }
    missing_layout = required_layout - set(path_layout.columns)
    if missing_layout:
        raise ValueError(
            "Faltan columnas en path_layout: "
            + ", ".join(sorted(missing_layout))
        )

    if x_train.ndim != 3:
        raise ValueError(
            f"Se esperaba tensor (evento,path,tiempo), recibido {x_train.shape}."
        )
    if x_train.shape[0] != len(metadata) or len(y_train) != len(metadata):
        raise ValueError("Tensor, etiquetas y metadata TRAIN no estan alineados.")
    if x_train.shape[1] != len(path_layout):
        raise ValueError("El numero de paths no coincide con path_layout.")
    if metadata["feature_row"].tolist() != list(range(len(metadata))):
        raise ValueError("feature_row no es consecutivo desde cero.")
    if path_layout["path_index"].tolist() != list(range(len(path_layout))):
        raise ValueError("path_index no es consecutivo desde cero.")
    if set(metadata["split"].astype(str)) != {"train"}:
        raise ValueError("La metadata cargada no pertenece exclusivamente a TRAIN.")
    if not np.array_equal(
        np.asarray(y_train, dtype=int),
        metadata["stage2_target"].to_numpy(dtype=int),
    ):
        raise ValueError("Las etiquetas del tensor y de la metadata no coinciden.")
    if set(metadata["stage2_target"].astype(int)) != {0, 1}:
        raise ValueError("TRAIN debe contener las etiquetas dry=0 y wet=1.")

    labels_per_recording = metadata.groupby("original_uuid")[
        "stage2_target"
    ].nunique(dropna=False)
    if not labels_per_recording.eq(1).all():
        invalid = labels_per_recording[~labels_per_recording.eq(1)]
        raise ValueError(
            f"Hay {len(invalid)} original_uuid con mas de una etiqueta."
        )
    type_per_recording = metadata.groupby("original_uuid")["cough_type"].nunique(
        dropna=False
    )
    if not type_per_recording.eq(1).all():
        raise ValueError("Hay original_uuid con mas de un cough_type.")

    recording_count = int(metadata["original_uuid"].nunique())
    if recording_count != EXPECTED_RECORDINGS:
        raise ValueError(
            f"Se esperaban {EXPECTED_RECORDINGS} grabaciones TRAIN, "
            f"pero se encontraron {recording_count}."
        )

    s2_layout = path_layout[
        path_layout["scattering_order"].astype(int).eq(2)
    ].copy()
    if len(s2_layout) != EXPECTED_S2_PATHS:
        raise ValueError(
            f"Se esperaban {EXPECTED_S2_PATHS} paths S2, recibidos "
            f"{len(s2_layout)}."
        )
    if not s2_layout["scattering_order"].astype(int).eq(2).all():
        raise AssertionError("La seleccion contiene paths que no son S2.")

    convention = verify_xi_frequency_convention(s2_layout)
    return x_train, metadata, s2_layout, config, convention


def aggregate_s2_by_recording(
    x_train: np.ndarray,
    metadata: pd.DataFrame,
    s2_layout: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aplica mean temporal y despues mean de eventos por original_uuid."""

    s2_indices = s2_layout["path_index"].to_numpy(dtype=int)
    if not np.all(
        s2_layout["scattering_order"].to_numpy(dtype=int) == 2
    ):
        raise AssertionError("Solo pueden agregarse paths de orden 2.")

    event_s2 = np.mean(
        x_train[:, s2_indices, :], axis=2, dtype=np.float32
    )
    if event_s2.shape != (len(metadata), len(s2_layout)):
        raise ValueError(f"Forma event_s2 inesperada: {event_s2.shape}.")
    if not np.isfinite(event_s2).all():
        raise ValueError("event_s2 contiene NaN o infinitos.")

    uuid_values = metadata["original_uuid"].astype(str).to_numpy()
    recording_codes, recording_ids = pd.factorize(uuid_values, sort=False)
    if np.any(recording_codes < 0):
        raise ValueError("Hay original_uuid ausentes.")

    n_recordings = len(recording_ids)
    sums = np.zeros((n_recordings, len(s2_layout)), dtype=np.float64)
    np.add.at(sums, recording_codes, event_s2)
    event_counts = np.bincount(recording_codes, minlength=n_recordings)
    if np.any(event_counts <= 0):
        raise ValueError("Alguna grabacion no contiene eventos.")
    recording_s2 = sums / event_counts[:, np.newaxis]

    label_map = metadata.groupby("original_uuid", sort=False)[
        "stage2_target"
    ].first()
    labels = label_map.reindex(recording_ids).to_numpy(dtype=int)
    if recording_s2.shape != (EXPECTED_RECORDINGS, EXPECTED_S2_PATHS):
        raise ValueError(
            "La matriz final no tiene una fila por grabacion TRAIN: "
            f"{recording_s2.shape}."
        )
    if len(np.unique(recording_ids)) != len(recording_ids):
        raise ValueError("La salida contiene original_uuid duplicados.")
    if not np.isfinite(recording_s2).all():
        raise ValueError("Las features recording-level contienen NaN o Inf.")

    return (
        recording_s2,
        labels,
        np.asarray(recording_ids, dtype=str),
        event_counts,
    )


def cliffs_delta_from_ranks(wet: np.ndarray, dry: np.ndarray) -> float:
    """Cliff's delta con empates tratados mediante rangos promedio."""

    n_wet = len(wet)
    n_dry = len(dry)
    combined = np.concatenate((wet, dry))
    ranks = rankdata(combined, method="average")
    u_wet = ranks[:n_wet].sum() - n_wet * (n_wet + 1) / 2.0
    return float(2.0 * u_wet / (n_wet * n_dry) - 1.0)


def effect_direction(
    delta: float,
    wet_median: float,
    dry_median: float,
    wet_mean: float,
    dry_mean: float,
) -> str:
    if delta > 0:
        return "wet_higher"
    if delta < 0:
        return "dry_higher"
    if wet_median != dry_median:
        return "wet_higher" if wet_median > dry_median else "dry_higher"
    return "wet_higher" if wet_mean >= dry_mean else "dry_higher"


def analyze_paths(
    recording_s2: np.ndarray,
    labels: np.ndarray,
    s2_layout: pd.DataFrame,
) -> pd.DataFrame:
    dry = recording_s2[labels == 0]
    wet = recording_s2[labels == 1]
    n_dry, n_wet = len(dry), len(wet)
    if (n_dry, n_wet) != (1247, 376):
        raise ValueError(
            f"Recuentos dry/wet inesperados: {n_dry}/{n_wet}."
        )

    rows: list[dict[str, Any]] = []
    for local_index, (_, path_metadata) in enumerate(s2_layout.iterrows()):
        dry_values = dry[:, local_index]
        wet_values = wet[:, local_index]
        all_values = recording_s2[:, local_index]

        dry_mean = float(np.mean(dry_values))
        wet_mean = float(np.mean(wet_values))
        dry_median = float(np.median(dry_values))
        wet_median = float(np.median(wet_values))
        auc = float(roc_auc_score(labels, all_values))
        delta = cliffs_delta_from_ranks(wet_values, dry_values)

        # Para una variable continua, delta = 2*AUC-1 (con el mismo manejo de
        # empates). Esta comprobacion detecta errores de orientacion dry/wet.
        if not np.isclose(delta, 2.0 * auc - 1.0, atol=1e-10):
            raise RuntimeError(
                "Cliff's delta y ROC-AUC no son coherentes para el path "
                f"{path_metadata['path_index']}."
            )

        row = path_metadata.to_dict()
        row.update(
            {
                "order": int(path_metadata["scattering_order"]),
                "acoustic_frequency_hz": float(path_metadata["xi_1"])
                * SAMPLE_RATE,
                "modulation_frequency_hz": float(path_metadata["xi_2"])
                * SAMPLE_RATE,
                "dry_mean": dry_mean,
                "wet_mean": wet_mean,
                "dry_median": dry_median,
                "wet_median": wet_median,
                "mean_difference_wet_minus_dry": wet_mean - dry_mean,
                "median_difference_wet_minus_dry": (
                    wet_median - dry_median
                ),
                "auc": auc,
                "auc_separation": abs(auc - 0.5),
                "cliffs_delta": delta,
                "abs_cliffs_delta": abs(delta),
                "direction": effect_direction(
                    delta,
                    wet_median,
                    dry_median,
                    wet_mean,
                    dry_mean,
                ),
                "n_dry": n_dry,
                "n_wet": n_wet,
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["abs_cliffs_delta", "auc_separation"],
        ascending=[False, False],
    )
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    if len(result) != EXPECTED_S2_PATHS:
        raise ValueError("El resultado no contiene todos los paths S2.")
    if not result["order"].eq(2).all():
        raise AssertionError("El resultado contiene ordenes distintos de S2.")

    preferred_columns = [
        "rank",
        "path_index",
        "path_name",
        "order",
        "scattering_order",
        "j_1",
        "j_2",
        "xi_1",
        "xi_2",
        "acoustic_frequency_hz",
        "modulation_frequency_hz",
        "dry_mean",
        "wet_mean",
        "dry_median",
        "wet_median",
        "mean_difference_wet_minus_dry",
        "median_difference_wet_minus_dry",
        "auc",
        "auc_separation",
        "cliffs_delta",
        "abs_cliffs_delta",
        "direction",
        "n_dry",
        "n_wet",
    ]
    remaining = [
        column for column in result.columns if column not in preferred_columns
    ]
    return result[preferred_columns + remaining].reset_index(drop=True)


def plot_frequency_scatter(
    results: pd.DataFrame,
    output_path: Path,
    title: str = "S2 wet-dry separation across acoustic and modulation frequencies",
) -> None:
    colors = {
        "wet_higher": "#167D8D",
        "dry_higher": "#D97735",
    }
    magnitude = results["abs_cliffs_delta"].to_numpy(dtype=float)
    sizes = 18.0 + 380.0 * magnitude

    fig, axis = plt.subplots(figsize=(10.5, 7.0))
    for direction in ("wet_higher", "dry_higher"):
        mask = results["direction"].eq(direction).to_numpy()
        axis.scatter(
            results.loc[mask, "acoustic_frequency_hz"],
            results.loc[mask, "modulation_frequency_hz"],
            s=sizes[mask],
            color=colors[direction],
            alpha=0.62,
            linewidth=0.35,
            edgecolor="white",
            label=direction.replace("_", " "),
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("First-order acoustic centre frequency (Hz)")
    axis.set_ylabel("Second-order modulation centre frequency (Hz)")
    axis.set_title(f"{title}\nTRAIN only; one observation per recording")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_paths(
    results: pd.DataFrame,
    top_k: int,
    output_path: Path,
    title: str | None = None,
) -> None:
    top = results.head(top_k).copy().sort_values(
        "abs_cliffs_delta", ascending=True
    )
    colors = top["direction"].map(
        {"wet_higher": "#167D8D", "dry_higher": "#D97735"}
    )
    labels = [
        (
            f"#{int(row['rank'])} path {int(row['path_index'])}  "
            f"({row['acoustic_frequency_hz']:.0f} Hz / "
            f"{row['modulation_frequency_hz']:.1f} Hz)"
        )
        for _, row in top.iterrows()
    ]

    fig, axis = plt.subplots(figsize=(11.5, 8.5))
    axis.barh(labels, top["cliffs_delta"], color=colors)
    axis.axvline(0.0, color="#333333", linewidth=0.9)
    axis.set_xlabel("Cliff's delta (positive: wet higher; negative: dry higher)")
    axis.set_ylabel("S2 path (acoustic Hz / modulation Hz)")
    if title is None:
        title = f"Top {top_k} second-order scattering paths"
    axis.set_title(
        f"{title}\nTRAIN only; ranked by absolute Cliff's delta"
    )
    axis.grid(True, axis="x", alpha=0.25)
    legend = [
        Line2D([0], [0], color="#167D8D", lw=8, label="wet higher"),
        Line2D([0], [0], color="#D97735", lw=8, label="dry higher"),
    ]
    axis.legend(handles=legend, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_check(
    preset: str,
    x_train: np.ndarray,
    metadata: pd.DataFrame,
    s2_layout: pd.DataFrame,
    convention: str,
) -> None:
    recording_labels = metadata.drop_duplicates("original_uuid")[
        "stage2_target"
    ].value_counts().sort_index()
    print("=" * 78)
    print("CHECK - ANALISIS S2 WST WET VS DRY (TRAIN ONLY)")
    print("=" * 78)
    print(f"Preset: {preset}")
    print(f"Splits cargados: {LOADED_SPLITS}")
    print(f"Tensor TRAIN: {x_train.shape}")
    print(
        f"Grabaciones TRAIN: {metadata['original_uuid'].nunique()} | "
        f"dry/wet={recording_labels.to_dict()}"
    )
    print(f"Paths S2: {len(s2_layout)}")
    print(f"Metadata disponible: {', '.join(s2_layout.columns)}")
    print(f"Convencion xi: {convention}")
    print("VALIDATION y TEST no se cargan ni se procesan.")


def print_top_paths(results: pd.DataFrame, top_k: int) -> None:
    table = results.head(top_k)[
        [
            "rank",
            "acoustic_frequency_hz",
            "modulation_frequency_hz",
            "wet_median",
            "dry_median",
            "cliffs_delta",
            "auc",
            "direction",
        ]
    ].copy()
    table.columns = [
        "rank",
        "f_acoustic_Hz",
        "f_modulation_Hz",
        "wet_median",
        "dry_median",
        "effect_size",
        "AUC",
        "direction",
    ]
    formats = {
        "f_acoustic_Hz": lambda value: f"{value:.2f}",
        "f_modulation_Hz": lambda value: f"{value:.2f}",
        "wet_median": lambda value: f"{value:.6e}",
        "dry_median": lambda value: f"{value:.6e}",
        "effect_size": lambda value: f"{value:+.4f}",
        "AUC": lambda value: f"{value:.4f}",
    }
    print("\n" + "=" * 78)
    print(f"TOP {top_k} PATHS S2 POR SEPARACION WET/DRY")
    print("=" * 78)
    print(table.to_string(index=False, formatters=formats))


def save_configuration(
    output_dir: Path,
    preset: str,
    x_train: np.ndarray,
    metadata: pd.DataFrame,
    recording_s2: np.ndarray,
    labels: np.ndarray,
    convention: str,
) -> None:
    row = {
        "analysis": "stage2_wst_s2_path_analysis_train_only",
        "preset": preset,
        "loaded_splits": "train_only",
        "validation_loaded": False,
        "test_loaded": False,
        "pca_applied": False,
        "classifier_trained": False,
        "sample_rate": SAMPLE_RATE,
        "event_temporal_pooling": "mean",
        "recording_event_pooling": "mean",
        "event_count": len(metadata),
        "recording_count": len(recording_s2),
        "dry_recording_count": int(np.sum(labels == 0)),
        "wet_recording_count": int(np.sum(labels == 1)),
        "s2_path_count": recording_s2.shape[1],
        "input_tensor_shape": "x".join(map(str, x_train.shape)),
        "recording_matrix_shape": "x".join(map(str, recording_s2.shape)),
        "xi_frequency_convention": convention,
        "frequency_conversion": "frequency_hz = xi * 16000",
        "ranking_primary": "abs_cliffs_delta",
        "ranking_secondary": "auc_separation",
    }
    pd.DataFrame([row]).to_csv(
        output_dir / "s2_wet_dry_path_analysis_configuration.csv",
        index=False,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k <= 0 or args.top_k > EXPECTED_S2_PATHS:
        raise ValueError(
            f"--top-k debe estar entre 1 y {EXPECTED_S2_PATHS}."
        )

    x_train, metadata, s2_layout, _, convention = load_train_only(args.preset)
    print_check(args.preset, x_train, metadata, s2_layout, convention)
    if args.action == "check":
        print("Comprobacion completada. No se generaron resultados.")
        return

    recording_s2, labels, recording_ids, event_counts = (
        aggregate_s2_by_recording(x_train, metadata, s2_layout)
    )
    if len(recording_ids) != EXPECTED_RECORDINGS:
        raise AssertionError("No hay exactamente una fila por original_uuid.")

    results = analyze_paths(recording_s2, labels, s2_layout)
    output_dir = OUTPUT_ROOT / args.preset / "train_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "s2_wet_dry_path_analysis_train.csv"
    results.to_csv(csv_path, index=False)

    plot_frequency_scatter(
        results,
        output_dir / "s2_frequency_modulation_effect_scatter_train.png",
    )
    plot_top_paths(
        results,
        args.top_k,
        output_dir / "s2_top_paths_cliffs_delta_train.png",
    )
    save_configuration(
        output_dir,
        args.preset,
        x_train,
        metadata,
        recording_s2,
        labels,
        convention,
    )
    pd.DataFrame(
        {
            "original_uuid": recording_ids,
            "stage2_target": labels,
            "cough_type": np.where(labels == 1, "wet", "dry"),
            "event_count": event_counts,
        }
    ).to_csv(output_dir / "recording_analysis_index_train.csv", index=False)

    print_top_paths(results, args.top_k)
    print("\n" + "=" * 78)
    print("ANALISIS S2 COMPLETADO - SOLO TRAIN")
    print("=" * 78)
    print(f"Matriz recording-level: {recording_s2.shape}")
    print(f"CSV: {csv_path}")
    print(
        "Scatter: "
        f"{output_dir / 's2_frequency_modulation_effect_scatter_train.png'}"
    )
    print(
        "Barras: "
        f"{output_dir / 's2_top_paths_cliffs_delta_train.png'}"
    )
    print("VALIDATION y TEST no se cargaron ni se procesaron.")


if __name__ == "__main__":
    main()
