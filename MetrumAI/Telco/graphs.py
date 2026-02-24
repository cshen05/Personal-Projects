import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ============================================================
# Brand colors
# ============================================================
BLUE = "#2f56ec"  # MI355X
RED  = "#ff3132"  # MI300X

# ============================================================
# Data (from your paper tables)
# ============================================================
bbus = np.array([3, 6, 15, 30])
rrhs = np.array([9, 18, 45, 90])

# Events/min
events_355x = np.array([539, 828, 1515, 2253])
events_300x = np.array([323, 512, 766, 1470])

# Tokens/sec
tokens_355x = np.array([2231, 2897, 6823, 7773])
tokens_300x = np.array([2078, 3954, 4342, 5476])

# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    "savefig.dpi": 300,
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
})

def style_axes(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, linewidth=0.8, alpha=0.12)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def thousands(x, pos):
    return f"{int(x):,}"

def save(fig, filename):
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

def label_inside_bars(ax, bars, fmt="{:,}", pad_frac=0.035):
    """
    Put numeric labels inside each bar near the top so they never collide with titles.
    """
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    pad = y_range * pad_frac

    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            max(h - pad, 0),
            fmt.format(int(round(h))),
            ha="center",
            va="top",
            fontsize=11,
            color="white"
        )

def add_group_multiplier_labels(ax, x_centers, left_vals, right_vals, ratios, y_pad_frac=0.03):
    """
    Place the multiplier label directly above each pair of bars (per-group),
    centered at x[i], at y = max(pair)+pad.
    """
    y0, y1 = ax.get_ylim()
    pad = (y1 - y0) * y_pad_frac

    for i, r in enumerate(ratios):
        y = max(left_vals[i], right_vals[i]) + pad
        ax.text(
            x_centers[i],
            y,
            f"{r:.1f}×",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold"
        )

# ============================================================
# FIGURE 1: Events/min scalability + per-group multipliers
# ============================================================
def fig_01_events_scalability_with_multipliers():
    fig, ax = plt.subplots(figsize=(9.8, 5.3))

    x = np.arange(len(bbus))
    w = 0.36

    b1 = ax.bar(x - w/2, events_355x, width=w, color=BLUE, label="MI355X")
    b2 = ax.bar(x + w/2, events_300x, width=w, color=RED,  label="MI300X")

    ax.set_title("Scalability: Events Processed per Minute\n(MI355X vs MI300X)", pad=12)
    ax.set_ylabel("Events / min")
    ax.set_xlabel("Monitoring Scale (BBUs / RRHs)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} BBUs\n{r} RRHs" for b, r in zip(bbus, rrhs)])

    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    style_axes(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper left")

    ymax = max(events_355x.max(), events_300x.max())
    ax.set_ylim(0, ymax * 1.18)

    # Labels inside bars
    label_inside_bars(ax, b1)
    label_inside_bars(ax, b2)

    # Multipliers per group
    ratios = events_355x / events_300x
    add_group_multiplier_labels(ax, x, events_355x, events_300x, ratios, y_pad_frac=0.03)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig_01_events_scalability_multipliers.png")

# ============================================================
# FIGURE 2: Tokens/sec scalability + per-group multipliers
# ============================================================
def fig_02_tokens_scalability_with_multipliers():
    fig, ax = plt.subplots(figsize=(9.8, 5.3))

    x = np.arange(len(bbus))
    w = 0.36

    b1 = ax.bar(x - w/2, tokens_355x, width=w, color=BLUE, label="MI355X")
    b2 = ax.bar(x + w/2, tokens_300x, width=w, color=RED,  label="MI300X")

    ax.set_title("Scalability: Inference Throughput (Tokens/sec)\n(MI355X vs MI300X)", pad=12)
    ax.set_ylabel("Tokens / sec")
    ax.set_xlabel("Monitoring Scale (BBUs / RRHs)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} BBUs\n{r} RRHs" for b, r in zip(bbus, rrhs)])

    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    style_axes(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper left")

    ymax = max(tokens_355x.max(), tokens_300x.max())
    ax.set_ylim(0, ymax * 1.18)

    # Labels inside bars
    label_inside_bars(ax, b1)
    label_inside_bars(ax, b2)

    # Multipliers per group
    ratios = tokens_355x / tokens_300x
    add_group_multiplier_labels(ax, x, tokens_355x, tokens_300x, ratios, y_pad_frac=0.03)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig_02_tokens_scalability_multipliers.png")

# ============================================================
# FIGURE 3: Tokens per event (proxy for per-incident token budget)
# NOT redundant with Figures 1-2 (this is efficiency/workload characterization)
# ============================================================
def fig_03_tokens_per_event():
    # tokens_per_event = 60 * tokens/sec / (events/min)
    tpe_355x = 60.0 * tokens_355x / events_355x
    tpe_300x = 60.0 * tokens_300x / events_300x

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    x = np.arange(len(bbus))
    w = 0.36

    b1 = ax.bar(x - w/2, tpe_355x, width=w, color=BLUE, label="MI355X")
    b2 = ax.bar(x + w/2, tpe_300x, width=w, color=RED,  label="MI300X")

    ax.set_title("Per-Event Inference Output: Tokens per Event (Proxy)\nHigher can indicate more generated text per incident", pad=12)
    ax.set_ylabel("Tokens per event")
    ax.set_xlabel("Monitoring Scale (BBUs / RRHs)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} BBUs\n{r} RRHs" for b, r in zip(bbus, rrhs)])

    style_axes(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper left")

    ymax = max(tpe_355x.max(), tpe_300x.max())
    ax.set_ylim(0, ymax * 1.22)

    # Value labels above bars (small chart; safe with headroom)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + ymax*0.03, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig_03_tokens_per_event.png")

# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    fig_01_events_scalability_with_multipliers()
    fig_02_tokens_scalability_with_multipliers()
    fig_03_tokens_per_event()