import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ============================================================
# Brand colors
# ============================================================
BLUE   = "#2f56ec"  # MI355X
ORANGE = "#ff3132"  # MI300X

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def comma_fmt(x, pos):
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def style_axes(ax):
    # Grid behind bars
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25, zorder=0)

def add_value_labels(ax, bars, fmt="{:,.0f}", y_pad_frac=0.015, fontsize=11):
    ymax = ax.get_ylim()[1]
    dy = ymax * y_pad_frac
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width()/2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False
        )

def add_advantage_labels(ax, xs, tops, labels, y_pad_frac=0.055, fontsize=16):
    ymax = ax.get_ylim()[1]
    dy = ymax * y_pad_frac
    for x, top, lab in zip(xs, tops, labels):
        ax.text(
            x, top + dy, lab,
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold",
            clip_on=False
        )

def grouped_bars(ax, xlabels, y_left, y_right,
                 left_label, right_label,
                 left_color, right_color,
                 ylabel, xlabel,
                 value_fmt_left="{:,.0f}", value_fmt_right="{:,.0f}",
                 adv_labels=None,
                 title=None,
                 legend_loc="upper left",
                 headroom_frac=0.18):
    x = np.arange(len(xlabels))
    w = 0.36

    # Bars above gridlines
    b1 = ax.bar(x - w/2, y_left,  width=w, label=left_label,  color=left_color,  zorder=3)
    b2 = ax.bar(x + w/2, y_right, width=w, label=right_label, color=right_color, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in xlabels])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, pad=14)

    ax.yaxis.set_major_formatter(FuncFormatter(comma_fmt))
    style_axes(ax)
    ax.legend(frameon=False, loc=legend_loc)

    # Y headroom to prevent top collisions
    ymax = max(float(np.max(y_left)), float(np.max(y_right)))
    ax.set_ylim(0, ymax * (1 + headroom_frac))

    # Value labels
    add_value_labels(ax, b1, fmt=value_fmt_left,  y_pad_frac=0.012)
    add_value_labels(ax, b2, fmt=value_fmt_right, y_pad_frac=0.012)

    # Advantage labels (x multipliers)
    if adv_labels is not None:
        tops = np.maximum(y_left, y_right)
        add_advantage_labels(ax, x, tops, adv_labels, y_pad_frac=0.06)

    return b1, b2

# ============================================================
# DATA (from your tables)
# ============================================================

# Table 10
concurrency_10 = np.array([512, 1024, 2048, 4096, 8192])
mi300x_tok_10  = np.array([7015, 10143, 8805, 9568, 9649])
mi355x_tok_10  = np.array([10449, 16097, 24743, 39930, 43801])
adv_10 = [f"{(a/b):.1f}×" for a, b in zip(mi355x_tok_10, mi300x_tok_10)]

# Table 15 (TTFT P95 ms)
concurrency_15 = np.array([32, 64, 128, 256, 512, 1024])
mi300x_ttft_15 = np.array([1523, 3185, 5397, 8983, 17640, 35648])
mi355x_ttft_15 = np.array([ 380,  693, 1172, 1978,  3779, 10013])
adv_15 = [f"{(a/b):.1f}× faster" for a, b in zip(mi300x_ttft_15, mi355x_ttft_15)]  # MI300X / MI355X

# Table 16 (tok/W)
concurrency_16 = np.array([512, 1024, 2048, 4096, 8192])
mi355x_tokW_16 = np.array([1.36, 1.88, 2.79, 4.37, 4.59])
mi300x_tokW_16 = np.array([1.27, 1.86, 1.77, 2.09, 2.12])
adv_16 = [f"{(a/b):.1f}×" for a, b in zip(mi355x_tokW_16, mi300x_tokW_16)]

# Table 18 (tok/s throughput)
concurrency_18 = np.array([128, 256, 512, 1024, 2048, 4096, 8192])
mi355x_tok_18  = np.array([3050, 4654, 6339, 7328, 7343, 7438, 7709])
mi300x_tok_18  = np.array([1426, 1890, 2075, 2235, 2266, 2295, 2352])
adv_18_tok = [f"{(a/b):.1f}×" for a, b in zip(mi355x_tok_18, mi300x_tok_18)]

# Table 18 (tok/W efficiency)
mi355x_tokW_18 = np.array([0.321, 0.479, 0.635, 0.714, 0.721, 0.725, 0.752])
mi300x_gpuW_18 = np.array([5398, 5457, 5365, 4701, 4750, 4737, 4724])
mi300x_tokW_18 = mi300x_tok_18 / mi300x_gpuW_18
adv_18_tokW = [f"{(a/b):.1f}×" for a, b in zip(mi355x_tokW_18, mi300x_tokW_18)]

# Table 19 (capacity/hr)
use_cases_19 = [
    "AML Transaction\nScreening",
    "KYC Document\nReview",
    "Trade Surveillance\nReports",
    "Regulatory Filing\nClassification",
]
mi300x_cap_19 = np.array([ 8.5,  4230,  4230,  4230], dtype=float)   # first is M tokens/hr
mi355x_cap_19 = np.array([27.8, 13880, 13880, 13880], dtype=float)

# ============================================================
# PLOTS (with improved spacing + grid behind bars)
# ============================================================

# ---- Figure: Concurrency scaling (tok/s)
fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
grouped_bars(
    ax,
    concurrency_10,
    mi355x_tok_10, mi300x_tok_10,
    "MI355X", "MI300X",
    BLUE, ORANGE,
    ylabel="Tokens / sec",
    xlabel="Concurrent Sessions",
    adv_labels=adv_10,
    title="Concurrency Scaling (128/128 Short-Form)",
    legend_loc="upper left",
    headroom_frac=0.22
)
plt.savefig("table10_concurrency_scaling_128_128.png", bbox_inches="tight")
plt.show()

# ---- Figure: TTFT P95
fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
grouped_bars(
    ax,
    concurrency_15,
    mi300x_ttft_15, mi355x_ttft_15,
    "MI300X", "MI355X",
    ORANGE, BLUE,
    ylabel="TTFT P95 (ms)",
    xlabel="Concurrent Sessions",
    adv_labels=adv_15,
    title="Time to First Token (TTFT P95), 2,048/128 Compliance",
    legend_loc="upper left",
    headroom_frac=0.26
)
plt.savefig("table15_ttft_p95_2048_128.png", bbox_inches="tight")
plt.show()

# ---- Figure: Tokens/W scaling
fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
grouped_bars(
    ax,
    concurrency_16,
    mi355x_tokW_16, mi300x_tokW_16,
    "MI355X", "MI300X",
    BLUE, ORANGE,
    ylabel="Tokens / Watt",
    xlabel="Concurrent Requests",
    value_fmt_left="{:.2f}",
    value_fmt_right="{:.2f}",
    adv_labels=adv_16,
    title="Tokens per GPU Watt Scaling (128/128 Short-Form)",
    legend_loc="upper left",
    headroom_frac=0.24
)
plt.savefig("table16_tokens_per_watt_128_128.png", bbox_inches="tight")
plt.show()

# ---- Figure: Compliance throughput (tok/s)
fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
grouped_bars(
    ax,
    concurrency_18,
    mi355x_tok_18, mi300x_tok_18,
    "MI355X", "MI300X",
    BLUE, ORANGE,
    ylabel="Tokens / sec",
    xlabel="Concurrent Sessions",
    adv_labels=adv_18_tok,
    title="Concurrency Scaling (2,048/128 Compliance) — Throughput",
    legend_loc="upper left",
    headroom_frac=0.22
)
plt.savefig("table18_compliance_throughput_2048_128.png", bbox_inches="tight")
plt.show()

# ---- Figure: Compliance efficiency (tok/W)
fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
grouped_bars(
    ax,
    concurrency_18,
    mi355x_tokW_18, mi300x_tokW_18,
    "MI355X", "MI300X",
    BLUE, ORANGE,
    ylabel="Tokens / Watt",
    xlabel="Concurrent Sessions",
    value_fmt_left="{:.3f}",
    value_fmt_right="{:.3f}",
    adv_labels=adv_18_tokW,
    title="Concurrency Scaling (2,048/128 Compliance) — Efficiency",
    legend_loc="upper left",
    headroom_frac=0.26
)
plt.savefig("table18_compliance_tokens_per_watt_2048_128.png", bbox_inches="tight")
plt.show()

# ---- Figure: Estimated capacity (keep single axis, fix label spacing)
fig, ax = plt.subplots(figsize=(12.5, 5.8), constrained_layout=True)
x = np.arange(len(use_cases_19))
w = 0.36

b1 = ax.bar(x - w/2, mi355x_cap_19, width=w, label="MI355X", color=BLUE, zorder=3)
b2 = ax.bar(x + w/2, mi300x_cap_19, width=w, label="MI300X", color=ORANGE, zorder=3)

ax.set_title("Estimated Document Processing Capacity (Peak Throughput)", pad=14)
ax.set_ylabel("Capacity / hour (see labels)")
ax.set_xticks(x)
ax.set_xticklabels(use_cases_19)
style_axes(ax)
ax.legend(frameon=False, loc="upper right")

# headroom
ymax = max(mi355x_cap_19.max(), mi300x_cap_19.max())
ax.set_ylim(0, ymax * 1.20)

# annotate: special-case first category (tokens/hr) so it doesn't collide at baseline
# ---- Fix AML annotation spacing ----
for i in range(len(use_cases_19)):
    if i == 0:
        # Raise MI355X label higher
        ax.text(
            b1[i].get_x() + b1[i].get_width()/2,
            ymax * 0.07,   # was 0.03 — increased for spacing
            f"~{mi355x_cap_19[i]:.1f}M tokens/hr",
            ha="center", va="bottom",
            fontsize=11, clip_on=False
        )

        # Keep MI300X slightly below it
        ax.text(
            b2[i].get_x() + b2[i].get_width()/2,
            ymax * 0.035,  # slight separation
            f"~{mi300x_cap_19[i]:.1f}M tokens/hr",
            ha="center", va="bottom",
            fontsize=11, clip_on=False
        )
    else:
        ax.text(b1[i].get_x()+b1[i].get_width()/2, b1[i].get_height() + ymax*0.02, f"~{int(mi355x_cap_19[i]):,}/hr",
                ha="center", va="bottom", fontsize=11, clip_on=False)
        ax.text(b2[i].get_x()+b2[i].get_width()/2, b2[i].get_height() + ymax*0.02, f"~{int(mi300x_cap_19[i]):,}/hr",
                ha="center", va="bottom", fontsize=11, clip_on=False)

plt.savefig("table19_estimated_capacity_peak.png", bbox_inches="tight")
plt.show()

print("Regenerated PNGs (overwritten in current directory):")
print(" - table10_concurrency_scaling_128_128.png")
print(" - table15_ttft_p95_2048_128.png")
print(" - table16_tokens_per_watt_128_128.png")
print(" - table18_compliance_throughput_2048_128.png")
print(" - table18_compliance_tokens_per_watt_2048_128.png")
print(" - table19_estimated_capacity_peak.png")