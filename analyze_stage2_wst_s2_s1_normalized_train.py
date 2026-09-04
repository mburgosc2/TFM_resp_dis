"""Analiza S2 normalizado por su coeficiente S1 padre usando solo TRAIN.

Para cada grabacion y path de segundo orden se calcula:

    R(lambda_1, lambda_2) = mean_events(S2) /
                            (mean_events(S1_parent) + epsilon)

La division se realiza despues de agregar eventos para evitar que eventos con
un denominador casi nulo dominen la media de una grabacion. Tambien se compara
la magnitud S2 global por grabacion antes de normalizar.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import analyze_stage2_wst_s2_paths_train as raw_analysis


DEFAULT_EPSILON = 1e-8
OUTPUT_SUBDIRECTORY = "s1_parent_normalized"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara dry/wet mediante S2/S1_parent con una observacion por "
            "grabacion de TRAIN."
        )
    )
    parser.add_argument(
        "--action", choices=("check", "analyze"), default="check"
    )
    parser.add_argument("--preset", default=raw_analysis.DEFAULT_PRESET)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    return parser


def load_full_path_layout(preset: str) -> pd.DataFrame:
    path = (
        raw_analysis.feature_directory(preset)
        / "wavelet_scattering_temporal_path_layout.csv"
    )
    layout = pd.read_csv(path)
    required = {
        "path_index",
        "path_name",
        "scattering_order",
        "j_1",
        "xi_1",
    }
    missing = required - set(layout.columns)
    if missing:
        raise ValueError(
            "Faltan columnas en path_layout: " + ", ".join(sorted(missing))
        )
    return layout


def build_parent_mapping(
    full_layout: pd.DataFrame,
    s2_layout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Empareja cada S2 con el unico S1 que comparte j_1 y xi_1."""

    s1_layout = full_layout[
        full_layout["scattering_order"].astype(int).eq(1)
    ].copy().reset_index(drop=True)
    if len(s1_layout) != 94:
        raise ValueError(f"Se esperaban 94 paths S1, recibidos {len(s1_layout)}.")
    if s1_layout[["j_1", "xi_1"]].duplicated().any():
        raise ValueError("La clave (j_1, xi_1) no identifica un S1 unico.")

    mapping_rows = []
    for _, s2_path in s2_layout.iterrows():
        same_j = s1_layout["j_1"].eq(float(s2_path["j_1"]))
        same_xi = np.isclose(
            s1_layout["xi_1"].to_numpy(dtype=float),
            float(s2_path["xi_1"]),
            rtol=1e-12,
            atol=0.0,
        )
        matches = s1_layout[same_j.to_numpy() & same_xi]
        if len(matches) != 1:
            raise ValueError(
                "No se encontro un unico padre S1 para el path S2 "
                f"{int(s2_path['path_index'])}: coincidencias={len(matches)}."
            )
        parent = matches.iloc[0]
        mapping_rows.append(
            {
                "path_index": int(s2_path["path_index"]),
                "parent_s1_local_index": int(matches.index[0]),
                "parent_s1_path_index": int(parent["path_index"]),
                "parent_s1_path_name": str(parent["path_name"]),
                "parent_s1_j_1": float(parent["j_1"]),
                "parent_s1_xi_1": float(parent["xi_1"]),
                "parent_s1_frequency_hz": float(parent["xi_1"])
                * raw_analysis.SAMPLE_RATE,
            }
        )

    mapping = pd.DataFrame(mapping_rows)
    if len(mapping) != raw_analysis.EXPECTED_S2_PATHS:
        raise ValueError("No se asigno un padre a cada path S2.")
    if mapping["path_index"].duplicated().any():
        raise ValueError("La tabla de padres contiene paths S2 duplicados.")
    if not np.allclose(
        mapping["parent_s1_xi_1"],
        s2_layout["xi_1"],
        rtol=1e-12,
        atol=0.0,
    ):
        raise AssertionError("S2 y su S1 padre no comparten xi_1.")
    return s1_layout, mapping


def aggregate_paths_by_recording(
    x_train: np.ndarray,
    metadata: pd.DataFrame,
    path_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean temporal por evento y mean de eventos por grabacion."""

    event_values = np.mean(
        x_train[:, path_indices, :], axis=2, dtype=np.float32
    )
    if not np.isfinite(event_values).all():
        raise ValueError("Los coeficientes por evento contienen NaN o Inf.")

    codes, recording_ids = pd.factorize(
        metadata["original_uuid"].astype(str).to_numpy(), sort=False
    )
    sums = np.zeros((len(recording_ids), len(path_indices)), dtype=np.float64)
    np.add.at(sums, codes, event_values)
    event_counts = np.bincount(codes, minlength=len(recording_ids))
    recording_values = sums / event_counts[:, np.newaxis]
    if not np.isfinite(recording_values).all():
        raise ValueError("La agregacion por grabacion contiene NaN o Inf.")
    return recording_values, np.asarray(recording_ids, dtype=str)


def global_s2_summary(values: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    dry = values[labels == 0]
    wet = values[labels == 1]
    auc = float(roc_auc_score(labels, values))
    delta = raw_analysis.cliffs_delta_from_ranks(wet, dry)
    if not np.isclose(delta, 2.0 * auc - 1.0, atol=1e-10):
        raise RuntimeError("AUC y Cliff's delta globales no son coherentes.")
    dry_mean = float(np.mean(dry))
    wet_mean = float(np.mean(wet))
    dry_median = float(np.median(dry))
    wet_median = float(np.median(wet))
    return pd.DataFrame(
        [
            {
                "feature": "mean_raw_s2_across_549_paths",
                "n_dry": len(dry),
                "n_wet": len(wet),
                "dry_mean": dry_mean,
                "wet_mean": wet_mean,
                "dry_median": dry_median,
                "wet_median": wet_median,
                "mean_difference_wet_minus_dry": wet_mean - dry_mean,
                "median_difference_wet_minus_dry": wet_median - dry_median,
                "auc": auc,
                "auc_separation": abs(auc - 0.5),
                "cliffs_delta": delta,
                "abs_cliffs_delta": abs(delta),
                "direction": raw_analysis.effect_direction(
                    delta,
                    wet_median,
                    dry_median,
                    wet_mean,
                    dry_mean,
                ),
            }
        ]
    )


def plot_global_s2(
    values: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    dry = values[labels == 0]
    wet = values[labels == 1]
    fig, axis = plt.subplots(figsize=(7.0, 5.8))
    box = axis.boxplot(
        [dry, wet],
        tick_labels=["dry", "wet"],
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    for patch, color in zip(box["boxes"], ("#D97735", "#167D8D")):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axis.set_yscale("log")
    axis.set_ylabel("Mean raw S2 magnitude across 549 paths")
    axis.set_title(
        "Global second-order scattering magnitude\n"
        "TRAIN only; one observation per recording"
    )
    axis.grid(True, axis="y", which="both", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k <= 0 or args.top_k > raw_analysis.EXPECTED_S2_PATHS:
        raise ValueError(
            f"--top-k debe estar entre 1 y {raw_analysis.EXPECTED_S2_PATHS}."
        )
    if not np.isfinite(args.epsilon) or args.epsilon <= 0:
        raise ValueError("--epsilon debe ser positivo y finito.")

    x_train, metadata, s2_layout, _, convention = (
        raw_analysis.load_train_only(args.preset)
    )
    full_layout = load_full_path_layout(args.preset)
    s1_layout, parent_mapping = build_parent_mapping(full_layout, s2_layout)
    raw_analysis.print_check(
        args.preset, x_train, metadata, s2_layout, convention
    )
    print(f"Paths S1 disponibles: {len(s1_layout)}")
    print(
        "Padres S1 usados por S2: "
        f"{parent_mapping['parent_s1_path_index'].nunique()}"
    )
    print(f"Epsilon: {args.epsilon:.3e}")
    print(
        "Definicion: mean_eventos(S2) / "
        "(mean_eventos(S1_padre) + epsilon)"
    )
    if args.action == "check":
        print("Comprobacion completada. No se generaron resultados.")
        return

    recording_s2, labels, recording_ids, event_counts = (
        raw_analysis.aggregate_s2_by_recording(x_train, metadata, s2_layout)
    )
    recording_s1, s1_recording_ids = aggregate_paths_by_recording(
        x_train,
        metadata,
        s1_layout["path_index"].to_numpy(dtype=int),
    )
    if not np.array_equal(recording_ids, s1_recording_ids):
        raise ValueError("S1 y S2 no tienen el mismo orden de original_uuid.")

    parent_local_indices = parent_mapping[
        "parent_s1_local_index"
    ].to_numpy(dtype=int)
    denominators = recording_s1[:, parent_local_indices]
    normalized_s2 = recording_s2 / (denominators + args.epsilon)
    if normalized_s2.shape != (
        raw_analysis.EXPECTED_RECORDINGS,
        raw_analysis.EXPECTED_S2_PATHS,
    ):
        raise ValueError(f"Forma S2/S1 inesperada: {normalized_s2.shape}.")
    if not np.isfinite(normalized_s2).all():
        raise ValueError("S2/S1 contiene NaN o infinitos.")

    results = raw_analysis.analyze_paths(normalized_s2, labels, s2_layout)
    results = results.merge(
        parent_mapping.drop(columns="parent_s1_local_index"),
        on="path_index",
        how="left",
        validate="one_to_one",
    )
    results["normalization"] = "S2_recording_mean/S1_parent_recording_mean"
    results["epsilon"] = args.epsilon
    results = results.sort_values(
        ["abs_cliffs_delta", "auc_separation"], ascending=[False, False]
    ).reset_index(drop=True)
    results["rank"] = np.arange(1, len(results) + 1)

    global_values = np.mean(recording_s2, axis=1)
    global_summary = global_s2_summary(global_values, labels)
    s1_positive = recording_s1[recording_s1 > 0]
    one_percent_quantile = float(np.quantile(s1_positive, 0.01))
    denominator_below_epsilon = float(np.mean(denominators < args.epsilon))

    output_dir = (
        raw_analysis.OUTPUT_ROOT
        / args.preset
        / "train_only"
        / OUTPUT_SUBDIRECTORY
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "s2_over_s1_path_analysis_train.csv"
    results.to_csv(result_path, index=False)
    parent_mapping.drop(columns="parent_s1_local_index").to_csv(
        output_dir / "s2_to_s1_parent_mapping.csv", index=False
    )

    recording_index = pd.DataFrame(
        {
            "original_uuid": recording_ids,
            "stage2_target": labels,
            "cough_type": np.where(labels == 1, "wet", "dry"),
            "event_count": event_counts,
            "mean_raw_s2_across_549_paths": global_values,
        }
    )
    recording_index.to_csv(
        output_dir / "global_s2_magnitude_by_recording_train.csv", index=False
    )
    global_summary.to_csv(
        output_dir / "global_s2_magnitude_comparison_train.csv", index=False
    )

    raw_analysis.plot_frequency_scatter(
        results,
        output_dir / "s2_over_s1_frequency_effect_scatter_train.png",
        title="S2/S1-parent wet-dry separation across frequency pairs",
    )
    raw_analysis.plot_top_paths(
        results,
        args.top_k,
        output_dir / "s2_over_s1_top_paths_cliffs_delta_train.png",
        title=f"Top {args.top_k} S2/S1-parent paths",
    )
    plot_global_s2(
        global_values,
        labels,
        output_dir / "global_s2_magnitude_boxplot_train.png",
    )

    pd.DataFrame(
        [
            {
                "analysis": "stage2_wst_s2_normalized_by_s1_parent_train_only",
                "preset": args.preset,
                "loaded_splits": "train_only",
                "validation_loaded": False,
                "test_loaded": False,
                "pca_applied": False,
                "classifier_trained": False,
                "sample_rate": raw_analysis.SAMPLE_RATE,
                "event_temporal_pooling": "mean",
                "recording_event_pooling": "mean",
                "normalization_order": "aggregate_events_then_divide",
                "normalization": (
                    "mean_events(S2)/(mean_events(S1_parent)+epsilon)"
                ),
                "epsilon": args.epsilon,
                "s1_positive_1pct_quantile": one_percent_quantile,
                "fraction_parent_denominators_below_epsilon": (
                    denominator_below_epsilon
                ),
                "recording_count": len(recording_ids),
                "s1_path_count": len(s1_layout),
                "s1_parent_count_used": parent_mapping[
                    "parent_s1_path_index"
                ].nunique(),
                "s2_path_count": len(s2_layout),
                "xi_frequency_convention": convention,
            }
        ]
    ).to_csv(output_dir / "s2_over_s1_analysis_configuration.csv", index=False)

    raw_analysis.print_top_paths(results, args.top_k)
    global_row = global_summary.iloc[0]
    print("\n" + "=" * 78)
    print("MAGNITUD S2 GLOBAL ANTES DE NORMALIZAR")
    print("=" * 78)
    print(
        f"AUC={global_row['auc']:.4f} | "
        f"Cliff's delta={global_row['cliffs_delta']:+.4f} | "
        f"direccion={global_row['direction']}"
    )
    print(
        f"Mediana dry/wet: {global_row['dry_median']:.6e} / "
        f"{global_row['wet_median']:.6e}"
    )
    print(
        f"Percentil 1 de S1 positivo={one_percent_quantile:.3e} | "
        "fraccion de denominadores por debajo de epsilon="
        f"{denominator_below_epsilon:.4%}"
    )
    print("\n" + "=" * 78)
    print("ANALISIS S2/S1 COMPLETADO - SOLO TRAIN")
    print("=" * 78)
    print(f"CSV normalizado: {result_path}")
    print(f"Resultados: {output_dir}")
    print("VALIDATION y TEST no se cargaron ni se procesaron.")


if __name__ == "__main__":
    main()
