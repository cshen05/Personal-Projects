# generate_fp4_figures.py
#
# Generates publication-style graphs for FP4 paper tables.
# - Bar charts only (clean + consistent with your preferred style)
# - Keeps dense scaling tables as tables (does NOT generate 4 workload line charts)
# - Saves PNGs to the same directory as this code file
#
# Style requirements implemented:
# - Remove x-axis gridlines (no vertical lines)
# - Y-axis gridlines behind bars
# - Horizontal value labels
# - Multipliers + values stay inside figure
# - Legend above axes (below title)
# - Precision in parentheses: "MI355X (MXFP4)" (no "+")
#
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ============================================================
# Output directory: same directory as this code file
# ============================================================
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT_DIR = BASE_DIR

# ============================================================
# Brand colors (match your earlier graphs)
# ============================================================
BLUE = "#2f56ec"
RED  = "#ff3132"

plt.rcParams.update({
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def comma_fmt(x, pos=None):
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def style_axes_bar(ax):
    """Y-grid only, behind bars; no x-grid/vertical lines."""
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(axis="y", alpha=0.22, zorder=0)

def put_legend_above(ax, ncol=2):
    """Legend above axes, under title (outside plot area)."""
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=ncol,
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.5,
        handletextpad=0.6
    )

def add_headroom(ax, y, headroom=0.22):
    ymax = float(np.nanmax(y)) if len(y) else 1.0
    ax.set_ylim(0, ymax * (1.0 + headroom))

def annotate_bar_values(ax, bars, fmt, dy_frac=0.012, fontsize=10):
    """Horizontal value labels above bars."""
    ymax = ax.get_ylim()[1]
    dy = ymax * dy_frac
    for b in bars:
        h = float(b.get_height())
        if h <= 0 or not math.isfinite(h):
            continue
        ax.text(
            b.get_x() + b.get_width()/2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=fontsize,
            clip_on=True
        )

def annotate_pair_multipliers(ax, x_centers, tops, ratios, dy_frac=0.055, fmt="{:.1f}x", fontsize=13):
    """Multipliers above the taller bar, kept inside canvas."""
    ymax = ax.get_ylim()[1]
    dy = ymax * dy_frac
    for x, t, r in zip(x_centers, tops, ratios):
        if not (math.isfinite(r) and r > 0):
            continue
        ax.text(
            float(x),
            float(t) + dy,
            fmt.format(float(r)),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            clip_on=True
        )

def grouped_bar_clean(
    *,
    title: str,
    xlabels: List[str],
    y_left: np.ndarray,
    y_right: np.ndarray,
    left_label: str,
    right_label: str,
    ylabel: str,
    xlabel: str,
    filename: str,
    value_fmt: str = "{:,.0f}",
    ratio_fmt: str = "{:.1f}x",
    headroom: float = 0.22,
    figsize=(12, 5.6),
):
    x = np.arange(len(xlabels))
    w = 0.36

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)

    b1 = ax.bar(x - w/2, y_left,  w, label=left_label,  color=BLUE, zorder=3)
    b2 = ax.bar(x + w/2, y_right, w, label=right_label, color=RED,  zorder=3)

    ax.set_title(title, pad=18)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    ax.yaxis.set_major_formatter(FuncFormatter(comma_fmt))
    style_axes_bar(ax)

    add_headroom(ax, np.r_[y_left, y_right], headroom=headroom)

    # Legend above plot
    put_legend_above(ax, ncol=2)

    # Horizontal labels
    annotate_bar_values(ax, b1, value_fmt, dy_frac=0.012)
    annotate_bar_values(ax, b2, value_fmt, dy_frac=0.012)

    # Multipliers (left / right)
    ratios = np.array([a / b if b else np.nan for a, b in zip(y_left, y_right)], dtype=float)
    tops = np.maximum(y_left, y_right)
    annotate_pair_multipliers(ax, x, tops, ratios, dy_frac=0.055, fmt=ratio_fmt)

    # Ensure space for legend + title
    fig.subplots_adjust(top=0.70)

    out_path = OUT_DIR / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved:", out_path)

# ============================================================
# DATA (update these if the paper changes)
# ============================================================

# Table 1 — Peak throughput across all tested concurrency levels (MI355X MXFP4 vs MI300X FP8)
TABLE1_PEAK: Dict[str, Tuple[int, int]] = {
    "128 / 128":   (33106,  7799),
    "128 / 2048":  (43916,  8278),
    "2048 / 128":  (10806,  1777),
    "2048 / 2048": (18395,  4096),
}

# Table 2 snapshot — throughput by workload profile at ~512 concurrent on MI355X (MXFP4 vs FP8)
# (These correspond to the “profile comparison at 512 concurrent” style figure)
TABLE2_SNAPSHOT_512: Dict[str, Tuple[int, int]] = {
    "128 / 128":   (10503, 8359),
    "128 / 2048":  (9621,  7385),
    "2048 / 128":  (4862,  4191),
    "2048 / 2048": (7000,  5838),
}

# Table 4 — max concurrency within 100ms per-token SLA (MI355X MXFP4 vs MI300X FP8)
TABLE4_SLA_MAX_CONC: Dict[str, Tuple[int, int]] = {
    "128 / 128":   (2048, 256),
    "128 / 2048":  (2048, 512),
    "2048 / 128":  (256,  64),
    "2048 / 2048": (512,  128),
}

# Table 5 — generational throughput scaling (pick one “hero” workload for a bar chart: 2048/2048)
# Rows: concurrency -> (MI355X MXFP4 tok/s, MI300X FP8 tok/s)
TABLE5_2048_2048: Dict[int, Tuple[int, int]] = {
    256:  (4513, 2453),
    1024: (9942, 4002),
    4096: (16647, 4083),
    8192: (18395, 4089),
}

# Table 6 — power-normalized throughput at 512 concurrency (tokens/W)
TABLE6_TOK_PER_WATT: Dict[str, Tuple[float, float]] = {
    "128 / 128":   (4.46, 0.66),
    "128 / 2048":  (4.12, 0.77),
    "2048 / 128":  (2.07, 0.21),
    "2048 / 2048": (2.97, 0.44),
}

# ============================================================
# FIGURE GENERATORS
# ============================================================

def fig_table1_peak_throughput():
    xlabels = list(TABLE1_PEAK.keys())
    left = np.array([TABLE1_PEAK[k][0] for k in xlabels], dtype=float)
    right = np.array([TABLE1_PEAK[k][1] for k in xlabels], dtype=float)

    grouped_bar_clean(
        title="Peak Throughput by Workload",
        xlabels=xlabels,
        y_left=left,
        y_right=right,
        left_label="MI355X (MXFP4)",
        right_label="MI300X (FP8)",
        ylabel="Tokens / second",
        xlabel="Input / Output Token Length",
        filename="Table1_Peak_Throughput_by_Workload.png",
        value_fmt="{:,.0f}",
        ratio_fmt="{:.1f}x",
        headroom=0.26,
        figsize=(12.5, 5.8),
    )

def fig_table2_snapshot_512():
    xlabels = list(TABLE2_SNAPSHOT_512.keys())
    left = np.array([TABLE2_SNAPSHOT_512[k][0] for k in xlabels], dtype=float)
    right = np.array([TABLE2_SNAPSHOT_512[k][1] for k in xlabels], dtype=float)

    grouped_bar_clean(
        title="Throughput by Workload Profile (≈512 Concurrent, MI355X)",
        xlabels=xlabels,
        y_left=left,
        y_right=right,
        left_label="MI355X (MXFP4)",
        right_label="MI355X (FP8)",
        ylabel="Tokens / second",
        xlabel="Input / Output Token Length",
        filename="Table2_Snapshot_512_Concurrent_MI355X.png",
        value_fmt="{:,.0f}",
        ratio_fmt="{:.1f}x",
        headroom=0.26,
        figsize=(12.5, 5.8),
    )

def fig_table4_sla_max_concurrency():
    xlabels = list(TABLE4_SLA_MAX_CONC.keys())
    left = np.array([TABLE4_SLA_MAX_CONC[k][0] for k in xlabels], dtype=float)   # MI355X
    right = np.array([TABLE4_SLA_MAX_CONC[k][1] for k in xlabels], dtype=float)  # MI300X

    grouped_bar_clean(
        title="Max Concurrency Within 100ms Per-Token SLA",
        xlabels=xlabels,
        y_left=left,
        y_right=right,
        left_label="MI355X (MXFP4)",
        right_label="MI300X (FP8)",
        ylabel="Max Concurrent Requests",
        xlabel="Workload (Input / Output)",
        filename="Table4_Max_Concurrency_100ms_SLA.png",
        value_fmt="{:,.0f}",
        ratio_fmt="{:.1f}x",
        headroom=0.30,
        figsize=(12.5, 5.8),
    )

def fig_table5_2048_2048_generational_bar():
    conc = sorted(TABLE5_2048_2048.keys())
    xlabels = [f"{c:,}" for c in conc]
    left = np.array([TABLE5_2048_2048[c][0] for c in conc], dtype=float)   # MI355X MXFP4
    right = np.array([TABLE5_2048_2048[c][1] for c in conc], dtype=float)  # MI300X FP8

    grouped_bar_clean(
        title="Generational Throughput Scaling (2048 / 2048)",
        xlabels=xlabels,
        y_left=left,
        y_right=right,
        left_label="MI355X (MXFP4)",
        right_label="MI300X (FP8)",
        ylabel="Tokens / second",
        xlabel="Concurrent Requests",
        filename="Table5_GenScaling_2048_2048.png",
        value_fmt="{:,.0f}",
        ratio_fmt="{:.1f}x",
        headroom=0.28,
        figsize=(12.5, 5.8),
    )

def fig_table6_power_normalized():
    xlabels = list(TABLE6_TOK_PER_WATT.keys())
    left = np.array([TABLE6_TOK_PER_WATT[k][0] for k in xlabels], dtype=float)   # MI355X
    right = np.array([TABLE6_TOK_PER_WATT[k][1] for k in xlabels], dtype=float)  # MI300X

    grouped_bar_clean(
        title="Power-Normalized Throughput at 512 Concurrent Requests",
        xlabels=xlabels,
        y_left=left,
        y_right=right,
        left_label="MI355X (MXFP4)",
        right_label="MI300X (FP8)",
        ylabel="Tokens / Watt",
        xlabel="Workload (Input / Output)",
        filename="Table6_Power_Normalized_Throughput_512.png",
        value_fmt="{:.2f}",
        ratio_fmt="{:.1f}x",
        headroom=0.34,
        figsize=(12.5, 5.8),
    )

# ============================================================
# MAIN
# ============================================================
def main():
    print("Saving figures to:", OUT_DIR)

    # High-value, publication-friendly charts:
    fig_table1_peak_throughput()
    fig_table2_snapshot_512()
    fig_table4_sla_max_concurrency()
    fig_table5_2048_2048_generational_bar()
    fig_table6_power_normalized()

    print("\nDone.")

if __name__ == "__main__":
    main()