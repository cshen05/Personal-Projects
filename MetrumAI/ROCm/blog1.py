import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ============================================================
# Brand Colors (Primary)
# ============================================================
METRUM_BLUE = "#2f56ec"
ORANGE      = "#ff3132"

# Supporting neutrals (for readability)
BG_COLOR    = "#f4f6fb"
GRID_COLOR  = "#d9dce3"
TEXT_COLOR  = "#111111"
NEUTRAL_3RD = "#6b7280"   # for 3rd series only (Table 1 has 3 workloads)

# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
})

def comma_int_fmt(x, _pos=None):
    """Integer + comma formatting (good for ms, tokens/s)."""
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return str(x)

def style_axes(ax, title, ylabel, xlabel, xticks, xticklabels, y_formatter=None):
    """
    y_formatter: optional matplotlib formatter. If None, uses comma_int_fmt.
    """
    ax.set_title(title, fontweight="bold", pad=18, color=TEXT_COLOR)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=11)

    ax.grid(axis="y", color=GRID_COLOR, linestyle="-", linewidth=1, alpha=0.6)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if y_formatter is None:
        ax.yaxis.set_major_formatter(FuncFormatter(comma_int_fmt))
    else:
        ax.yaxis.set_major_formatter(y_formatter)

def add_inside_labels(ax, bars, fmt_func, y_frac=0.90, fontsize=11):
    """White, bold labels inside bars."""
    for b in bars:
        h = b.get_height()
        if h <= 0:
            continue
        ax.text(
            b.get_x() + b.get_width()/2,
            h * y_frac,
            fmt_func(h),
            ha="center",
            va="top",
            color="white",
            fontsize=fontsize,
            fontweight="bold"
        )

def make_figure(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    return fig, ax

def save_or_show(fig, outpath=None):
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, bbox_inches="tight")
    plt.show()

# ============================================================
# DATA FROM PAPER
# ============================================================
# Table 1 (Throughput on single 8-GPU MI355X node)
concurrency_t1 = np.array([128, 256, 512, 1024])
tput_short = np.array([454, 701, 1117, 1838])
tput_gen   = np.array([424, 691, 1032, 1785])
tput_long  = np.array([393, 662, 1016, 1606])

# Table 2 (Tokens/Watt at production concurrency for Generation workload)
concurrency_t2 = np.array([32, 64, 128, 256, 512, 1024])
tokens_per_w   = np.array([0.087, 0.094, 0.195, 0.318, 0.475, 0.821])

# Table 3 (TTFT p50/p90)
concurrency_t3 = np.array([32, 64, 128, 256, 512, 1024])
ttft_p50 = np.array([652, 965, 954, 1358, 1960, 1860])
ttft_p90 = np.array([999, 1201, 1468, 2857, 4233, 7295])

# ============================================================
# CHART 1: Table 1 Throughput (Grouped bar, 3 series)
# ============================================================
fig, ax = make_figure(figsize=(11, 5.4))

x = np.arange(len(concurrency_t1))
w = 0.25

bars_short = ax.bar(x - w, tput_short, width=w, color=METRUM_BLUE, label="Short Context (128/128)")
bars_gen   = ax.bar(x,      tput_gen,   width=w, color=ORANGE,      label="Generation (128/2048)")
bars_long  = ax.bar(x + w,  tput_long,  width=w, color=NEUTRAL_3RD, label="Long Context (2048/2048)")

style_axes(
    ax,
    title="Throughput Scaling: Output Tokens per Second (MI355X)",
    ylabel="Output Tokens / Second",
    xlabel="Concurrent Requests",
    xticks=x,
    xticklabels=[str(v) for v in concurrency_t1],
)

add_inside_labels(ax, bars_short, fmt_func=lambda v: f"{v:,.0f}", y_frac=0.90)
add_inside_labels(ax, bars_gen,   fmt_func=lambda v: f"{v:,.0f}", y_frac=0.90)
add_inside_labels(ax, bars_long,  fmt_func=lambda v: f"{v:,.0f}", y_frac=0.90, fontsize=10)

ax.legend(frameon=False, loc="upper left")

save_or_show(fig, outpath="table1_throughput_bar_styled.png")

# ============================================================
# CHART 2: Table 2 Tokens/Watt (Single-series bar)
# FIX: Use a decimal y-axis formatter (so values < 1 don't display as 0).
# ============================================================
fig, ax = make_figure(figsize=(11, 5.0))

x2 = np.arange(len(concurrency_t2))
bars_eff = ax.bar(x2, tokens_per_w, width=0.6, color=ORANGE, label="Tokens/Watt")

decimal_formatter = FuncFormatter(lambda v, _pos: f"{v:.3f}")

style_axes(
    ax,
    title="Efficiency Scaling: Output Tokens per Watt (MI355X, 128/2048)",
    ylabel="Output Tokens / Watt",
    xlabel="Concurrent Requests",
    xticks=x2,
    xticklabels=[str(v) for v in concurrency_t2],
    y_formatter=decimal_formatter,   # <-- key fix
)

add_inside_labels(ax, bars_eff, fmt_func=lambda v: f"{v:.3f}", y_frac=0.90)

ax.legend(frameon=False, loc="upper left")

save_or_show(fig, outpath="table2_tokens_per_watt_bar_styled.png")

# ============================================================
# CHART 3: Table 3 TTFT (Grouped bar, 2 series)
# Change: Bars only (no multipliers).
# ============================================================
fig, ax = make_figure(figsize=(11, 5.4))

x3 = np.arange(len(concurrency_t3))
w3 = 0.36

bars_p50 = ax.bar(x3 - w3/2, ttft_p50, width=w3, color=METRUM_BLUE, label="TTFT p50 (ms)")
bars_p90 = ax.bar(x3 + w3/2, ttft_p90, width=w3, color=ORANGE,      label="TTFT p90 (ms)")

style_axes(
    ax,
    title="Latency Under Load: Time to First Token (TTFT) (MI355X, 128/2048)",
    ylabel="TTFT (ms)",
    xlabel="Concurrent Requests",
    xticks=x3,
    xticklabels=[str(v) for v in concurrency_t3],
)

add_inside_labels(ax, bars_p50, fmt_func=lambda v: f"{v:,.0f}", y_frac=0.90)
add_inside_labels(ax, bars_p90, fmt_func=lambda v: f"{v:,.0f}", y_frac=0.90)

ax.legend(frameon=False, loc="upper left")

save_or_show(fig, outpath="table3_ttft_bar_styled.png")