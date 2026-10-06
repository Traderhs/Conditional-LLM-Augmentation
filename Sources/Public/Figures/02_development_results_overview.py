import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEVELOPMENT_INPUT = (
    PROJECT_ROOT
    / "Results"
    / "BinaryMatchedSizeExperiment"
    / "Downstream"
    / "DevelopmentGrid"
    / "paired_results.csv"
)
WEIGHT_INPUT = (
    PROJECT_ROOT
    / "Results"
    / "BinaryMatchedSizeExperiment"
    / "Downstream"
    / "SyntheticWeightGrid"
    / "paired_weight_results.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "Figures"
    / "02_development_results_overview.pdf"
)


DATASETS = [
    ("en_binary_sst2", "English"),
    ("ko_binary_nsmc", "Korean"),
    ("bn_binary_cinexdrama", "Bengali"),
    ("ha_binary_hausa_movie_review", "Hausa"),
    ("ml_binary_dravidian_codemix", "Malayalam"),
]

RATIOS = [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
    3.50,
    4.00,
]

SYNTHETIC_WEIGHTS = [0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a compact cross-dataset overview of the three "
            "development-stage comparisons discussed in the Results section."
        )
    )
    parser.add_argument(
        "--development-input",
        type=Path,
        default=DEVELOPMENT_INPUT,
        help="Development-grid paired_results.csv.",
    )
    parser.add_argument(
        "--weight-input",
        type=Path,
        default=WEIGHT_INPUT,
        help="Synthetic-weight-grid paired_weight_results.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output PDF path.",
    )
    return parser.parse_args()


def select_figure_font() -> str:
    preferred_families = [
        "Source Sans 3",
        "IBM Plex Sans",
        "Noto Sans",
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
    ]
    installed = {
        font.name
        for font in font_manager.fontManager.ttflist
    }
    for family in preferred_families:
        if family in installed:
            return family
    return "sans-serif"


FIGURE_FONT = select_figure_font()
SEMIBOLD = 600
VALUE_LABEL_SIZE = 11.5
FOOTER_SIZE = 11.5
PANEL_TITLE_SIZE = 16.5
PANEL_NOTE_SIZE = 11.5

# Deliberately muted, publication-oriented palettes. Each overview panel uses
# its own palette so the three result stages are visually distinct without
# falling back to Matplotlib's default blue/orange cycle.
FILTERING_COLORS = ["#0072B2", "#E69F00"]
RATIO_COLORS = ["#59648F", "#C69A3B"]
WEIGHTING_COLORS = ["#009E73", "#CC79A7"]


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    parsed = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )
    if parsed.isna().any():
        invalid = sorted(normalized.loc[parsed.isna()].unique())
        raise ValueError(f"Unexpected boolean values: {invalid}")
    return parsed.astype(bool)


def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    label: str,
) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{label} is missing required columns: "
            + ", ".join(missing)
        )


def prepare_development(paired: pd.DataFrame) -> pd.DataFrame:
    paired = paired.copy()
    require_columns(
        paired,
        {
            "cell_id",
            "repeat_seed",
            "ratio",
            "similarity_condition_id",
            "similarity_enabled",
            "similarity_lower",
            "similarity_upper",
            "feasible",
            "hybrid_macro_f1",
            "matched_all_real_macro_f1",
            "hybrid_auroc",
            "matched_all_real_auroc",
        },
        "development results",
    )
    paired["feasible"] = parse_bool(paired["feasible"])
    paired["similarity_enabled"] = parse_bool(
        paired["similarity_enabled"]
    )
    numeric = [
        "repeat_seed",
        "ratio",
        "similarity_lower",
        "similarity_upper",
        "hybrid_macro_f1",
        "matched_all_real_macro_f1",
        "hybrid_auroc",
        "matched_all_real_auroc",
    ]
    for column in numeric:
        paired[column] = pd.to_numeric(
            paired[column],
            errors="coerce" if column.startswith("similarity_") else "raise",
        )
    return paired


def build_filtering_summary(paired: pd.DataFrame) -> pd.DataFrame:
    off = paired.loc[
        paired["similarity_condition_id"].eq("OFF"),
        [
            "cell_id",
            "repeat_seed",
            "ratio",
            "hybrid_macro_f1",
            "hybrid_auroc",
        ],
    ].rename(
        columns={
            "hybrid_macro_f1": "off_macro_f1",
            "hybrid_auroc": "off_auroc",
        }
    )
    expected_off = len(DATASETS) * len(RATIOS) * 50
    if len(off) != expected_off:
        raise ValueError(
            f"Unexpected unfiltered row count: {len(off)}; "
            f"expected {expected_off}."
        )

    filtered = paired.loc[
        paired["similarity_enabled"]
    ].merge(
        off,
        on=["cell_id", "repeat_seed", "ratio"],
        how="left",
        validate="many_to_one",
    )
    if filtered[["off_macro_f1", "off_auroc"]].isna().any().any():
        raise ValueError("Filtered rows are missing unfiltered anchors.")

    filtered["delta_macro_f1"] = (
        filtered["hybrid_macro_f1"] - filtered["off_macro_f1"]
    )
    filtered["delta_auroc"] = (
        filtered["hybrid_auroc"] - filtered["off_auroc"]
    )

    condition_keys = [
        "cell_id",
        "similarity_lower",
        "similarity_upper",
        "ratio",
    ]
    fully_feasible_groups: list[pd.DataFrame] = []
    for keys, group in filtered.groupby(condition_keys, dropna=False):
        cell_id, lower, upper, ratio = keys
        fully_feasible = len(group) == 50 and bool(group["feasible"].all())
        if not fully_feasible:
            continue
        fully_feasible_groups.append(group)

    if not fully_feasible_groups:
        raise ValueError("No fully feasible filtering cells were found.")
    feasible = pd.concat(fully_feasible_groups, ignore_index=True)
    records: list[dict[str, object]] = []
    for cell_id, group in feasible.groupby("cell_id"):
        for metric, column in [
            ("Macro-F1", "delta_macro_f1"),
            ("AUROC", "delta_auroc"),
        ]:
            values = group[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Non-finite filtering values for {cell_id}, {metric}."
                )
            records.append(
                {
                    "cell_id": cell_id,
                    "metric": metric,
                    "mean": float(values.mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def build_ratio_summary(paired: pd.DataFrame) -> pd.DataFrame:
    off = paired.loc[
        paired["similarity_condition_id"].eq("OFF")
        & paired["feasible"]
    ].copy()
    expected = len(DATASETS) * len(RATIOS) * 50
    if len(off) != expected:
        raise ValueError(
            f"Unexpected similarity-OFF row count: {len(off)}; "
            f"expected {expected}."
        )
    records: list[dict[str, object]] = []
    for (cell_id, ratio), group in off.groupby(["cell_id", "ratio"]):
        if len(group) != 50:
            raise ValueError(
                f"Expected 50 unfiltered rows for {cell_id}, ratio={ratio}; "
                f"found {len(group)}."
            )
        for metric, suffix in [
            ("Macro-F1", "macro_f1"),
            ("AUROC", "auroc"),
        ]:
            base = group[f"base_{suffix}"].to_numpy(dtype=float)
            comparisons = [
                ("Hybrid", group[f"hybrid_{suffix}"].to_numpy(dtype=float)),
                (
                    "Matched All-real",
                    group[f"matched_all_real_{suffix}"].to_numpy(dtype=float),
                ),
            ]
            for arm, values in comparisons:
                delta = values - base
                if not np.isfinite(delta).all():
                    raise ValueError(
                        f"Non-finite ratio values for {cell_id}, ratio={ratio}, "
                        f"{metric}, {arm}."
                    )
                records.append(
                    {
                        "cell_id": cell_id,
                        "ratio": float(ratio),
                        "metric": metric,
                        "arm": arm,
                        "mean": float(delta.mean()),
                    }
                )
    return pd.DataFrame.from_records(records)


def prepare_weights(paired: pd.DataFrame) -> pd.DataFrame:
    paired = paired.copy()
    require_columns(
        paired,
        {
            "cell_id",
            "repeat_seed",
            "ratio",
            "synthetic_weight",
            "weighted_status",
            "delta_vs_base_macro_f1",
            "delta_vs_base_auroc",
        },
        "synthetic-weight results",
    )
    numeric = [
        "repeat_seed",
        "ratio",
        "synthetic_weight",
        "delta_vs_base_macro_f1",
        "delta_vs_base_auroc",
    ]
    for column in numeric:
        paired[column] = pd.to_numeric(paired[column], errors="raise")
    incomplete = paired.loc[~paired["weighted_status"].eq("completed")]
    if not incomplete.empty:
        raise ValueError("Incomplete synthetic-weight rows were found.")
    expected = (
        len(DATASETS)
        * len(RATIOS)
        * len(SYNTHETIC_WEIGHTS)
        * 50
    )
    if len(paired) != expected:
        raise ValueError(
            f"Unexpected synthetic-weight row count: {len(paired)}; "
            f"expected {expected}."
        )
    return paired


def build_weight_summary(paired: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    specifications = [
        ("Macro-F1", "delta_vs_base_macro_f1"),
        ("AUROC", "delta_vs_base_auroc"),
    ]
    for cell_id, cell in paired.groupby("cell_id"):
        for metric, column in specifications:
            means = (
                cell.groupby("synthetic_weight")[column]
                .mean()
                .sort_index()
            )
            if len(means) != len(SYNTHETIC_WEIGHTS):
                raise ValueError(
                    f"Unexpected weight count for {cell_id}, {metric}: "
                    f"{len(means)}."
                )
            if not np.isfinite(means.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"Non-finite weight means for {cell_id}, {metric}."
                )
            maximum = float(means.max())
            winners = means.index[
                np.isclose(
                    means.to_numpy(dtype=float),
                    maximum,
                    rtol=0.0,
                    atol=1e-12,
                )
            ].to_numpy(dtype=float)
            if winners.size != 1:
                raise ValueError(
                    "Expected a unique descriptive best weight for "
                    f"{cell_id}, {metric}; found {winners.tolist()}."
                )
            records.append(
                {
                    "cell_id": cell_id,
                    "metric": metric,
                    "best_weight": float(winners[0]),
                }
            )
    return pd.DataFrame.from_records(records)


def values_by_dataset(
    data: pd.DataFrame,
    metric: str,
    value_column: str,
) -> np.ndarray:
    dataset_order = [cell_id for cell_id, _ in DATASETS]
    subset = (
        data.loc[data["metric"].eq(metric)]
        .set_index("cell_id")
        .reindex(dataset_order)
    )
    if subset[value_column].isna().any():
        raise ValueError(f"Missing overview values for {metric}.")
    values = subset[value_column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite overview values for {metric}.")
    return values


def format_signed(value: float) -> str:
    absolute = abs(value)
    if absolute < 0.001:
        return f"{value:+.4f}"
    if absolute < 0.01:
        return f"{value:+.3f}"
    return f"{value:+.2f}"


def format_weight(value: float) -> str:
    return f"{value:.3f}"


def add_value_labels(
    axis: plt.Axes,
    bars,
    formatter,
    *,
    vertical: bool = True,
) -> None:
    ymin, ymax = axis.get_ylim()
    span = ymax - ymin
    for bar in bars:
        height = float(bar.get_height())
        if vertical:
            offset = 0.025 * span
            y = height + offset if height >= 0 else height - offset
            va = "bottom" if height >= 0 else "top"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                formatter(height),
                ha="center",
                va=va,
                fontsize=VALUE_LABEL_SIZE,
                fontweight=SEMIBOLD,
            )


def draw_effect_bars(
    axis: plt.Axes,
    data: pd.DataFrame,
    title: str,
    ylabel: str,
    footer: str,
    colors: list[str],
) -> tuple[object, object]:
    x = np.arange(len(DATASETS), dtype=float)
    width = 0.34
    macro = values_by_dataset(data, "Macro-F1", "mean")
    auroc = values_by_dataset(data, "AUROC", "mean")
    lower = min(0.0, float(np.min([macro.min(), auroc.min()])))
    upper = max(0.0, float(np.max([macro.max(), auroc.max()])))
    span = upper - lower
    padding = 0.16 * span if span > 0 else 0.001
    axis.set_ylim(lower - padding, upper + padding)
    macro_bars = axis.bar(
        x - width / 2,
        macro,
        width,
        label="Macro-F1",
        color=colors[0],
        zorder=3,
    )
    auroc_bars = axis.bar(
        x + width / 2,
        auroc,
        width,
        label="AUROC",
        color=colors[1],
        zorder=3,
    )
    axis.axhline(0.0, color="black", linewidth=0.8, zorder=2)
    axis.grid(axis="y", linewidth=0.45, alpha=0.28, zorder=1)
    axis.set_title(title, pad=8, fontweight=SEMIBOLD)
    axis.set_ylabel(ylabel, fontweight=SEMIBOLD)
    axis.set_xticks(x)
    axis.set_xticklabels([label for _, label in DATASETS])
    axis.tick_params(axis="x", length=0, pad=6)
    axis.tick_params(axis="y", length=3)
    add_value_labels(axis, macro_bars, format_signed)
    add_value_labels(axis, auroc_bars, format_signed)
    axis.text(
        0.5,
        -0.20,
        footer,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=FOOTER_SIZE,
    )
    return macro_bars, auroc_bars


def draw_weight_bars(
    axis: plt.Axes,
    data: pd.DataFrame,
    colors: list[str],
) -> tuple[object, object]:
    x = np.arange(len(DATASETS), dtype=float)
    width = 0.34
    macro = values_by_dataset(data, "Macro-F1", "best_weight")
    auroc = values_by_dataset(data, "AUROC", "best_weight")
    axis.set_ylim(0.0, 1.18)
    macro_bars = axis.bar(
        x - width / 2,
        macro,
        width,
        label="Macro-F1",
        color=colors[0],
        zorder=3,
    )
    auroc_bars = axis.bar(
        x + width / 2,
        auroc,
        width,
        label="AUROC",
        color=colors[1],
        zorder=3,
    )
    axis.grid(axis="y", linewidth=0.45, alpha=0.28, zorder=1)
    axis.set_title("(c) Synthetic sample weighting", pad=8, fontweight=SEMIBOLD)
    axis.set_ylabel("Best synthetic weight", fontweight=SEMIBOLD)
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.set_xticks(x)
    axis.set_xticklabels([label for _, label in DATASETS])
    axis.tick_params(axis="x", length=0, pad=6)
    axis.tick_params(axis="y", length=3)
    add_value_labels(axis, macro_bars, format_weight)
    add_value_labels(axis, auroc_bars, format_weight)
    axis.text(
        0.5,
        -0.20,
        "0 = Base / no synthetic influence   ·   1 = full synthetic weight",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=FOOTER_SIZE,
    )
    return macro_bars, auroc_bars


def selected_ratio_by_language(data: pd.DataFrame) -> dict[str, float]:
    selected: dict[str, float] = {}
    hybrid_macro = data.loc[
        data["metric"].eq("Macro-F1")
        & data["arm"].eq("Hybrid")
    ]
    for cell_id, _ in DATASETS:
        rows = hybrid_macro.loc[
            hybrid_macro["cell_id"].eq(cell_id)
        ].sort_values("ratio")
        if len(rows) != len(RATIOS):
            raise ValueError(
                f"Expected {len(RATIOS)} Hybrid Macro-F1 ratio rows for "
                f"{cell_id}; found {len(rows)}."
            )
        maximum = float(rows["mean"].max())
        winners = rows.loc[
            np.isclose(
                rows["mean"],
                maximum,
                rtol=0.0,
                atol=1e-15,
            ),
            "ratio",
        ].to_numpy(dtype=float)
        if winners.size != 1:
            raise ValueError(
                f"Expected one best Hybrid Macro-F1 ratio for {cell_id}; "
                f"found {winners.tolist()}."
            )
        selected[cell_id] = float(winners[0])
    return selected


def draw_ratio_dumbbell(
    axis: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    selected_ratios: dict[str, float],
    *,
    show_language_labels: bool,
) -> tuple[object, object]:
    dataset_order = [cell_id for cell_id, _ in DATASETS]
    base_labels = [label for _, label in DATASETS]
    labels = [
        f"{label}\n(r*={selected_ratios[cell_id]:g})"
        for cell_id, label in DATASETS
    ]
    y = np.arange(len(DATASETS), dtype=float)
    metric_data = data.loc[data["metric"].eq(metric)]

    hybrid_values: list[float] = []
    matched_values: list[float] = []
    for cell_id in dataset_order:
        ratio_value = selected_ratios[cell_id]
        for arm, target in [
            ("Hybrid", hybrid_values),
            ("Matched All-real", matched_values),
        ]:
            row = metric_data.loc[
                metric_data["cell_id"].eq(cell_id)
                & metric_data["arm"].eq(arm)
                & np.isclose(metric_data["ratio"], ratio_value)
            ]
            if len(row) != 1:
                raise ValueError(
                    f"Expected one {arm} row for {cell_id}, {metric}, "
                    f"r={ratio_value}; found {len(row)}."
                )
            target.append(float(row.iloc[0]["mean"]))

    hybrid = np.asarray(hybrid_values, dtype=float)
    matched = np.asarray(matched_values, dtype=float)
    if not np.isfinite(hybrid).all() or not np.isfinite(matched).all():
        raise ValueError(f"Non-finite ratio overview values for {metric}.")

    for row_index, (left, right) in enumerate(
        zip(hybrid, matched, strict=True)
    ):
        axis.plot(
            [left, right],
            [row_index, row_index],
            color="0.72",
            linewidth=1.6,
            zorder=1,
        )
    hybrid_handle = axis.scatter(
        hybrid,
        y,
        marker="o",
        s=52,
        facecolor=RATIO_COLORS[0],
        edgecolor=RATIO_COLORS[0],
        label="Hybrid",
        zorder=3,
    )
    matched_handle = axis.scatter(
        matched,
        y,
        marker="D",
        s=48,
        facecolor=RATIO_COLORS[1],
        edgecolor=RATIO_COLORS[1],
        linewidth=1.2,
        label="Matched All-real",
        zorder=3,
    )
    axis.axvline(0.0, color="black", linewidth=0.8, zorder=2)
    axis.grid(axis="x", linewidth=0.45, alpha=0.28, zorder=1)
    axis.set_title(metric, pad=6, fontweight=SEMIBOLD)
    axis.set_xlabel("Mean Δ vs Base", fontweight=SEMIBOLD)
    axis.set_yticks(y)
    axis.set_yticklabels(labels if show_language_labels else [])
    axis.set_ylim(len(DATASETS) - 0.5, -0.5)
    axis.tick_params(axis="y", length=0, pad=7)
    axis.tick_params(axis="x", length=3)
    return hybrid_handle, matched_handle


def main() -> None:
    args = parse_args()
    development_path = args.development_input.expanduser().resolve()
    weight_path = args.weight_input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in [development_path, weight_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
    if output_path.suffix.lower() != ".pdf":
        raise ValueError("Overview output must use a .pdf extension.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    development = prepare_development(
        pd.read_csv(development_path, low_memory=False)
    )
    weights = prepare_weights(
        pd.read_csv(weight_path, low_memory=False)
    )
    filtering = build_filtering_summary(development)
    ratio = build_ratio_summary(development)
    weighting = build_weight_summary(weights)
    selected_ratios = selected_ratio_by_language(ratio)

    plt.rcParams.update(
        {
            "font.family": FIGURE_FONT,
            "font.size": 14.0,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.titleweight": SEMIBOLD,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(10.8, 11.8))
    outer = figure.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.0, 1.25, 1.0],
        hspace=0.82,
    )
    axis_filtering = figure.add_subplot(outer[0, 0])
    ratio_grid = outer[1, 0].subgridspec(1, 2, wspace=0.16)
    axis_ratio_macro = figure.add_subplot(ratio_grid[0, 0])
    axis_ratio_auroc = figure.add_subplot(ratio_grid[0, 1])
    axis_weighting = figure.add_subplot(outer[2, 0])
    first_macro, first_auroc = draw_effect_bars(
        axis_filtering,
        filtering,
        "(a) Similarity filtering",
        "Mean filtered − unfiltered",
        "Below zero = unfiltered selection performed better on average",
        FILTERING_COLORS,
    )
    ratio_hybrid, ratio_matched = draw_ratio_dumbbell(
        axis_ratio_macro,
        ratio,
        "Macro-F1",
        selected_ratios,
        show_language_labels=True,
    )
    draw_ratio_dumbbell(
        axis_ratio_auroc,
        ratio,
        "AUROC",
        selected_ratios,
        show_language_labels=False,
    )
    draw_weight_bars(
        axis_weighting,
        weighting,
        WEIGHTING_COLORS,
    )
    weighting_legend = [
        plt.Rectangle((0, 0), 1, 1, facecolor=WEIGHTING_COLORS[0]),
        plt.Rectangle((0, 0), 1, 1, facecolor=WEIGHTING_COLORS[1]),
    ]
    figure.subplots_adjust(
        top=0.94,
        bottom=0.06,
        left=0.12,
        right=0.98,
    )
    figure.canvas.draw()
    macro_position = axis_ratio_macro.get_position()
    auroc_position = axis_ratio_auroc.get_position()
    ratio_group_center = (macro_position.x0 + auroc_position.x1) / 2
    ratio_group_title_y = max(macro_position.y1, auroc_position.y1) + 0.034
    macro_tight_box = axis_ratio_macro.get_tightbbox(
        figure.canvas.get_renderer()
    ).transformed(figure.transFigure.inverted())
    figure.text(
        ratio_group_center,
        ratio_group_title_y,
        "(b) Augmentation ratio",
        ha="center",
        va="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight=SEMIBOLD,
    )
    figure.text(
        macro_tight_box.x0 + 0.035,
        macro_position.y0 - 0.25 * macro_position.height,
        "r* = ratio with the highest mean unweighted-Hybrid Macro-F1",
        ha="left",
        va="top",
        fontsize=PANEL_NOTE_SIZE,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    filtering_title_box = axis_filtering.title.get_window_extent(
        renderer=renderer
    ).transformed(figure.transFigure.inverted())
    weighting_title_box = axis_weighting.title.get_window_extent(
        renderer=renderer
    ).transformed(figure.transFigure.inverted())
    filtering_title_center_y = (
        filtering_title_box.y0 + filtering_title_box.y1
    ) / 2
    weighting_title_center_y = (
        weighting_title_box.y0 + weighting_title_box.y1
    ) / 2
    axis_filtering.legend(
        [first_macro, first_auroc],
        ["Macro-F1", "AUROC"],
        loc="center right",
        bbox_to_anchor=(axis_filtering.get_position().x1, filtering_title_center_y),
        bbox_transform=figure.transFigure,
        ncol=2,
        handlelength=1.55,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
        frameon=False,
    )
    axis_ratio_auroc.legend(
        [ratio_hybrid, ratio_matched],
        ["Hybrid", "Matched All-real"],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.965),
        ncol=1,
        fontsize=11.0,
        handletextpad=0.40,
        markerscale=0.9,
        borderaxespad=0.0,
        frameon=False,
    )
    axis_weighting.legend(
        weighting_legend,
        ["Macro-F1", "AUROC"],
        loc="center right",
        bbox_to_anchor=(axis_weighting.get_position().x1, weighting_title_center_y),
        bbox_transform=figure.transFigure,
        ncol=2,
        handlelength=1.65,
        handletextpad=0.42,
        columnspacing=0.9,
        borderaxespad=0.0,
        frameon=False,
    )

    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
