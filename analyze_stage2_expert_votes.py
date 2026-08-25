"""Audita los votos cough_type_1..4 utilizados en Stage 2 dry/wet.

El script une los metadatos originales de COUGHVID con los splits maestros ya
congelados. No modifica etiquetas ni splits. Genera una fila por grabacion,
tablas resumen y graficas que permiten distinguir:

* cuantas personas expertas evaluaron cada audio;
* si una muestra weak procede de 1 evaluador, 2 de acuerdo, 2 contra 1, etc.;
* cuantos empates fueron convertidos en unknown/ambiguous_expert;
* como se distribuyen los patrones en TRAIN, VALIDATION y TEST.

TEST se usa exclusivamente para describir metadatos/votos preexistentes. No se
leen audios, features ni predicciones, y no se calcula rendimiento de modelos.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_METADATA = SCRIPT_DIR.parent / "DATA" / "metadata_compiled.csv"
SPLITS_DIR = SCRIPT_DIR / "metadata_splits_multi_experiment_random"
OUTPUT_DIR = SCRIPT_DIR / "metadata_stage2_expert_vote_analysis"
GRAPH_DIR = (
    SCRIPT_DIR
    / "graphs_results_stage2_dry_wet_random"
    / "expert_vote_analysis"
)

SPLIT_FILES = {
    "train": "metadata_train_multiclass_4c.csv",
    "validation": "metadata_val_multiclass_4c.csv",
    "test": "metadata_test_multiclass_4c.csv",
}
VOTE_COLUMNS = [f"cough_type_{index}" for index in range(1, 5)]
VALID_VOTES = ("dry", "wet", "unknown")
SPLIT_ORDER = ("train", "validation", "test")
SPLIT_COLORS = {
    "train": "#4C78A8",
    "validation": "#F58518",
    "test": "#54A24B",
}
CONSENSUS_ORDER = (
    "gold_expert",
    "weak_expert",
    "ambiguous_expert",
    "not_applicable",
)


def normalize_vote(value: object) -> str | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "nan", "none"}:
        return None
    if normalized not in VALID_VOTES:
        raise ValueError(f"Voto cough_type inesperado: {value!r}")
    return normalized


def weak_reason(
    evaluator_count: int,
    maximum_agreement: int,
    count_pattern: str,
) -> str:
    if evaluator_count == 1:
        return "only_1_evaluator"
    if evaluator_count == 2 and maximum_agreement == 2:
        return "2_evaluators_unanimous"
    if evaluator_count == 3 and maximum_agreement == 2:
        return "3_evaluators_2_vs_1"
    if evaluator_count == 4 and count_pattern == "2-1-1":
        return "4_evaluators_2_vs_1_vs_1"
    return f"other_weak__n{evaluator_count}__{count_pattern}"


def gold_reason(
    evaluator_count: int,
    maximum_agreement: int,
) -> str:
    if evaluator_count == maximum_agreement:
        return f"{evaluator_count}_evaluators_unanimous"
    return f"{maximum_agreement}_of_{evaluator_count}_agree"


def summarize_votes(row: pd.Series) -> pd.Series:
    votes = [
        vote
        for vote in (normalize_vote(row[column]) for column in VOTE_COLUMNS)
        if vote is not None
    ]
    counts = Counter(votes)
    evaluator_count = len(votes)
    dry_votes = int(counts.get("dry", 0))
    wet_votes = int(counts.get("wet", 0))
    unknown_votes = int(counts.get("unknown", 0))
    vote_pattern = (
        f"dry{dry_votes}_wet{wet_votes}_unknown{unknown_votes}"
    )

    if evaluator_count == 0:
        return pd.Series(
            {
                "expert_evaluator_count": 0,
                "dry_vote_count": 0,
                "wet_vote_count": 0,
                "unknown_vote_count": 0,
                "maximum_agreement_count": 0,
                "agreement_ratio": np.nan,
                "count_pattern": "no_votes",
                "vote_pattern": vote_pattern,
                "top_vote_tie": False,
                "computed_cough_type": "no_votes",
                "computed_consensus": "no_votes",
                "consensus_reason": "no_type_votes",
            }
        )

    maximum_agreement = max(counts.values())
    top_labels = sorted(
        label for label, count in counts.items() if count == maximum_agreement
    )
    top_vote_tie = len(top_labels) > 1
    count_pattern = "-".join(
        str(value) for value in sorted(counts.values(), reverse=True)
    )

    if top_vote_tie:
        computed_cough_type = "unknown"
        computed_consensus = "ambiguous_expert"
        reason = f"tie_{count_pattern}__{evaluator_count}_evaluators"
    else:
        computed_cough_type = top_labels[0]
        if maximum_agreement >= 3:
            computed_consensus = "gold_expert"
            reason = gold_reason(evaluator_count, maximum_agreement)
        else:
            computed_consensus = "weak_expert"
            reason = weak_reason(
                evaluator_count,
                maximum_agreement,
                count_pattern,
            )

    return pd.Series(
        {
            "expert_evaluator_count": evaluator_count,
            "dry_vote_count": dry_votes,
            "wet_vote_count": wet_votes,
            "unknown_vote_count": unknown_votes,
            "maximum_agreement_count": int(maximum_agreement),
            "agreement_ratio": maximum_agreement / evaluator_count,
            "count_pattern": count_pattern,
            "vote_pattern": vote_pattern,
            "top_vote_tie": top_vote_tie,
            "computed_cough_type": computed_cough_type,
            "computed_consensus": computed_consensus,
            "consensus_reason": reason,
        }
    )


def load_original_votes() -> pd.DataFrame:
    if not ORIGINAL_METADATA.is_file():
        raise FileNotFoundError(f"No existe {ORIGINAL_METADATA}")
    original = pd.read_csv(
        ORIGINAL_METADATA,
        usecols=["uuid", *VOTE_COLUMNS],
        dtype={"uuid": str},
    )
    if original["uuid"].duplicated().any():
        raise ValueError("metadata_compiled.csv contiene uuid duplicados.")
    original["uuid"] = original["uuid"].astype(str).str.strip()
    vote_summary = original.apply(summarize_votes, axis=1)
    return pd.concat([original, vote_summary], axis=1)


def load_split_recordings() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = {
        "original_uuid",
        "dataset_origin",
        "cough_type",
        "cough_type_consensus",
        "stage2_eligible",
        "stage2_gold_eval",
        "stage2_reject_challenge",
        "fold",
        "split",
    }
    for split_name, filename in SPLIT_FILES.items():
        path = SPLITS_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"No existe {path}")
        frame = pd.read_csv(path, dtype={"original_uuid": str})
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Faltan columnas en {filename}: {sorted(missing)}")
        if set(frame["split"].astype(str)) != {split_name}:
            raise ValueError(f"Split inesperado en {filename}.")
        frame = frame[
            frame["dataset_origin"].astype(str).str.upper() == "COUGHVID"
        ].copy()
        consistency_columns = [
            "cough_type",
            "cough_type_consensus",
            "stage2_eligible",
            "stage2_gold_eval",
            "stage2_reject_challenge",
            "fold",
            "split",
        ]
        inconsistent = (
            frame.groupby("original_uuid")[consistency_columns]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if inconsistent.any():
            examples = inconsistent[inconsistent].index[:10].tolist()
            raise ValueError(
                f"Grabaciones inconsistentes en {split_name}: {examples}"
            )
        frame = frame.drop_duplicates("original_uuid")
        frames.append(frame[["original_uuid", *consistency_columns]])

    recordings = pd.concat(frames, ignore_index=True)
    if recordings["original_uuid"].duplicated().any():
        raise ValueError("Una grabacion COUGHVID aparece en varios splits.")
    recordings["original_uuid"] = (
        recordings["original_uuid"].astype(str).str.strip()
    )
    return recordings


def merge_and_validate() -> pd.DataFrame:
    votes = load_original_votes()
    recordings = load_split_recordings()
    merged = recordings.merge(
        votes,
        left_on="original_uuid",
        right_on="uuid",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        examples = merged.loc[unmatched, "original_uuid"].head(10).tolist()
        raise ValueError(f"Grabaciones sin votos originales: {examples}")
    merged = merged.drop(columns="_merge")

    with_votes = merged["expert_evaluator_count"] > 0
    consensus_mismatch = (
        with_votes
        & (
            merged["computed_consensus"]
            != merged["cough_type_consensus"]
        )
    )
    type_mismatch = (
        with_votes
        & (merged["computed_cough_type"] != merged["cough_type"])
    )
    if consensus_mismatch.any() or type_mismatch.any():
        columns = [
            "original_uuid",
            "cough_type",
            "computed_cough_type",
            "cough_type_consensus",
            "computed_consensus",
            *VOTE_COLUMNS,
        ]
        examples = merged.loc[
            consensus_mismatch | type_mismatch,
            columns,
        ].head(20)
        raise ValueError(
            "Los votos no reproducen los metadatos actuales:\n"
            f"{examples.to_string(index=False)}"
        )

    merged["analysis_scope"] = "other_coughvid"
    merged.loc[
        merged["stage2_eligible"].astype(str).str.lower().isin({"true", "1"}),
        "analysis_scope",
    ] = "stage2_dry_wet"
    merged.loc[
        merged["cough_type_consensus"] == "ambiguous_expert",
        "analysis_scope",
    ] = "ambiguous_expert"
    merged.loc[
        merged["expert_evaluator_count"] == 0,
        "analysis_scope",
    ] = "no_type_votes"
    return merged.sort_values(
        ["split", "fold", "cough_type", "original_uuid"]
    ).reset_index(drop=True)


def percentage_summary(
    data: pd.DataFrame,
    group_columns: list[str],
    denominator_columns: list[str],
) -> pd.DataFrame:
    summary = (
        data.groupby(group_columns, dropna=False)
        .size()
        .rename("recording_count")
        .reset_index()
    )
    totals = summary.groupby(denominator_columns)["recording_count"].transform(
        "sum"
    )
    summary["percentage"] = 100.0 * summary["recording_count"] / totals
    return summary


def save_summaries(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    stage2 = data[data["analysis_scope"] == "stage2_dry_wet"].copy()
    weak = stage2[stage2["cough_type_consensus"] == "weak_expert"].copy()
    ambiguous = data[
        data["cough_type_consensus"] == "ambiguous_expert"
    ].copy()

    summaries = {
        "evaluator_count_by_split": percentage_summary(
            data,
            ["split", "expert_evaluator_count"],
            ["split"],
        ),
        "stage2_consensus_by_split_class": percentage_summary(
            stage2,
            ["split", "cough_type", "cough_type_consensus"],
            ["split", "cough_type"],
        ),
        "weak_reasons_by_split": percentage_summary(
            weak,
            ["split", "consensus_reason"],
            ["split"],
        ),
        "weak_patterns_by_split_class": percentage_summary(
            weak,
            ["split", "cough_type", "vote_pattern"],
            ["split", "cough_type"],
        ),
        "ambiguous_reasons_by_split": percentage_summary(
            ambiguous,
            ["split", "consensus_reason"],
            ["split"],
        ),
        "vote_count_by_consensus": percentage_summary(
            data,
            [
                "split",
                "cough_type_consensus",
                "expert_evaluator_count",
            ],
            ["split", "cough_type_consensus"],
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, summary in summaries.items():
        summary.to_csv(
            OUTPUT_DIR / f"summary_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return summaries


def grouped_percentage_bars(
    summary: pd.DataFrame,
    category_column: str,
    category_order: list[object],
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(max(10, len(category_order) * 1.8), 6))
    x = np.arange(len(category_order), dtype=float)
    width = 0.24
    for split_index, split in enumerate(SPLIT_ORDER):
        group = summary[summary["split"] == split].set_index(category_column)
        percentages = np.array(
            [
                float(group.loc[value, "percentage"])
                if value in group.index
                else 0.0
                for value in category_order
            ]
        )
        counts = np.array(
            [
                int(group.loc[value, "recording_count"])
                if value in group.index
                else 0
                for value in category_order
            ]
        )
        positions = x + (split_index - 1) * width
        bars = axis.bar(
            positions,
            percentages,
            width=width,
            color=SPLIT_COLORS[split],
            label=split,
        )
        for bar, percentage, count in zip(bars, percentages, counts):
            if count:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.7,
                    f"{percentage:.1f}%\nn={count}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    axis.set_xticks(x)
    axis.set_xticklabels([str(value) for value in category_order], rotation=25)
    axis.set_ylabel("Porcentaje dentro del split")
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    upper = max(105.0, axis.get_ylim()[1] * 1.10)
    axis.set_ylim(0, upper)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_weak_patterns(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    patterns = sorted(summary["vote_pattern"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(max(11, len(patterns) * 1.5), 11))
    for axis, cough_type in zip(axes, ("dry", "wet")):
        subset = summary[summary["cough_type"] == cough_type]
        x = np.arange(len(patterns), dtype=float)
        width = 0.24
        for split_index, split in enumerate(SPLIT_ORDER):
            group = subset[subset["split"] == split].set_index("vote_pattern")
            values = [
                float(group.loc[item, "percentage"])
                if item in group.index
                else 0.0
                for item in patterns
            ]
            axis.bar(
                x + (split_index - 1) * width,
                values,
                width,
                color=SPLIT_COLORS[split],
                label=split,
            )
        axis.set_title(f"Weak expert con etiqueta final {cough_type}")
        axis.set_ylabel("Porcentaje dentro de clase y split")
        axis.set_xticks(x)
        axis.set_xticklabels(patterns, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("Patrones exactos de votos en las muestras weak dry/wet")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def create_graphs(summaries: dict[str, pd.DataFrame]) -> None:
    evaluator_summary = summaries["evaluator_count_by_split"]
    grouped_percentage_bars(
        evaluator_summary,
        "expert_evaluator_count",
        [1, 2, 3, 4],
        "Numero de evaluadores con voto de tipo de tos",
        "Numero de evaluadores",
        GRAPH_DIR / "expert_evaluator_count_by_split.png",
    )

    weak_summary = summaries["weak_reasons_by_split"]
    weak_order = [
        "only_1_evaluator",
        "2_evaluators_unanimous",
        "3_evaluators_2_vs_1",
        "4_evaluators_2_vs_1_vs_1",
    ]
    weak_order.extend(
        sorted(set(weak_summary["consensus_reason"]) - set(weak_order))
    )
    grouped_percentage_bars(
        weak_summary,
        "consensus_reason",
        weak_order,
        "Por que una muestra dry/wet es weak_expert",
        "Causa del consenso weak",
        GRAPH_DIR / "weak_reasons_by_split.png",
    )

    ambiguous_summary = summaries["ambiguous_reasons_by_split"]
    ambiguous_order = sorted(ambiguous_summary["consensus_reason"].unique())
    if ambiguous_order:
        grouped_percentage_bars(
            ambiguous_summary,
            "consensus_reason",
            ambiguous_order,
            "Empates y ausencia de votos convertidos en ambiguous_expert",
            "Patron ambiguo",
            GRAPH_DIR / "ambiguous_reasons_by_split.png",
        )

    plot_weak_patterns(
        summaries["weak_patterns_by_split_class"],
        GRAPH_DIR / "weak_vote_patterns_by_split_and_class.png",
    )


def print_key_results(
    data: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
) -> None:
    stage2 = data[data["analysis_scope"] == "stage2_dry_wet"]
    weak = stage2[stage2["cough_type_consensus"] == "weak_expert"]
    weak_ties = int(weak["top_vote_tie"].sum())
    print("\n" + "=" * 78)
    print("RESULTADO - AUDITORIA DE VOTOS EXPERTOS")
    print("=" * 78)
    print(f"Grabaciones COUGHVID en splits: {len(data)}")
    print(f"Grabaciones Stage 2 dry/wet: {len(stage2)}")
    print(f"Muestras weak Stage 2: {len(weak)}")
    print(f"Muestras weak con empate: {weak_ties}")
    print("\nCausas de weak_expert por split:")
    print(
        summaries["weak_reasons_by_split"]
        .sort_values(["split", "recording_count"], ascending=[True, False])
        .to_string(index=False)
    )
    print(f"\nDetalle: {OUTPUT_DIR / 'stage2_expert_votes_detailed.csv'}")
    print(f"Resumenes: {OUTPUT_DIR}")
    print(f"Graficas: {GRAPH_DIR}")
    print("No se han modificado etiquetas, splits, audios ni features.")


def main() -> None:
    print("=" * 78)
    print("AUDITORIA DE VOTOS EXPERTOS COUGHVID - STAGE 2")
    print("=" * 78)
    print("Se describen metadatos de TRAIN, VALIDATION y TEST.")
    print("No se entrenan modelos ni se consultan audios/features/predicciones.")
    data = merge_and_validate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        OUTPUT_DIR / "stage2_expert_votes_detailed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summaries = save_summaries(data)
    create_graphs(summaries)
    print_key_results(data, summaries)


if __name__ == "__main__":
    main()
