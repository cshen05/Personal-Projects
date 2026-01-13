import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Module-level flag to control brand frame rendering
USE_BRAND_FRAME = False

plt.rcParams.update({
    "figure.figsize": (14, 8),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})


def add_brand_frame(fig,
                    top_line_color="#111111",
                    frame_color="#111111",
                    line_start=0.03,
                    line_end=0.40,
                    line_y=0.985,
                    frame_margin=0.012):
    """
    Adds an outer frame and short horizontal accent line at the top-left
    of the figure, similar to the Solidigm VectorDBBench style.
    Coordinates are in figure fractions so they scale with size.
    """

    if not USE_BRAND_FRAME:
        return
    # Outer rectangular frame
    fig.add_artist(
        Rectangle(
            (frame_margin, frame_margin),
            1 - 2 * frame_margin,
            1 - 2 * frame_margin,
            transform=fig.transFigure,
            fill=False,
            linewidth=1.2,
            edgecolor=frame_color,
        )
    )

# Reusable branded callout helper (rounded box)

def add_callout(ax, text, x, y, ha="left", va="top",
                fontsize=12, fontweight="bold", color=None):
    """Draws a branded callout box in axes coordinates that pops visually."""
    if color is None:
        color = PRIMARY_BLUE
    txt = ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",   # neutral background
            edgecolor=color,     # match bar/line color
            linewidth=1.4,
            alpha=1.0,
        ),
    )
    return txt

# ===== COLOR PALETTE (your colors) ==========================
PRIMARY_BLUE = "#0080ff"   # palette blue
MAGENTA      = "#ff00ae"   # palette magenta
GREEN        = "#a3ff04"   # palette green (not yet used, available)
# ============================================================


# ============================================================
# 1) LLM Throughput With and Without FP8 Optimization
# ============================================================

def plot_llm_throughput_comparison():
    racks = np.array([1, 50, 100, 200])

    q30  = np.array([499,  7495, 12644, 17783])   # Qwen3-30B
    q235 = np.array([550,  5641,  8214,  7454])   # Qwen3-235B FP8

    x = np.arange(len(racks))
    width = 0.35

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    add_brand_frame(fig)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(False)

    # More headroom for title + subtitles
    fig.subplots_adjust(top=0.85)

    bars30  = ax.bar(x - width/2, q30,  width, label="Qwen3-30B (BF16)",
                     color=PRIMARY_BLUE)
    bars235 = ax.bar(x + width/2, q235, width, label="Qwen3-235B (FP8)",
                     color=MAGENTA)

    ax.set_xticks(x)
    ax.set_xticklabels(racks)
    ax.set_xlabel("Racks Monitored")
    ax.set_ylabel("Max LLM Throughput (tokens/sec)")

    leg = ax.legend(loc="upper left", fontsize=12, frameon=True)
    leg.get_frame().set_alpha(0.9)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.15)
    ax.xaxis.grid(False)

    # numeric labels on bars
    offset = 0.01 * max(q30.max(), q235.max())
    for bar, val in zip(bars30, q30):
        ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=11)
    for bar, val in zip(bars235, q235):
        ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=11)

    # Brand-style callout: central, bold text without cross-model comparison
    last_idx = -1
    scale_factor_30b = q30[last_idx] / q30[0]
    add_callout(
        ax,
        f"Qwen3-30B (BF16) scales {scale_factor_30b:.1f}×\nfrom 1 to {racks[last_idx]} racks",
        x=0.25,
        y=0.80,
        ha="center",
        va="top",
        fontsize=15,
        color=PRIMARY_BLUE,
    )

    # Additional callout for Qwen3-235B (FP8): highlight saturation behavior
    idx_100 = np.where(racks == 100)[0][0]
    add_callout(
        ax,
        f"Qwen3-235B (FP8) reaches\n{q235[idx_100]:,} tokens/sec at 100 racks\nand then saturates",
        x=0.25,
        y=0.50,
        ha="center",
        va="bottom",
        fontsize=15,
        color=MAGENTA,
    )

    fig.suptitle(
        "LLM Throughput Comparison: Qwen3-30B (BF16) vs Qwen3-235B (FP8)",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5, 0.93,
        "Measuring Qwen Model Performance on Supermicro Servers",
        ha="center", va="top",
        fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, 0.89,
        "LLM Throughput for Different Rack and Server Counts (Higher is Better)",
        ha="center", va="top",
        fontsize=12, color="gray",
    )

    plt.savefig("01_llm_throughput_fp8_vs_30b.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Helper for single-model bar charts (thicker bars)
# ============================================================

def plot_single_model_bar(filename, main_title, model_label,
                          conc_vals, tps_vals, color,
                          subtitle="Measured on AMD Instinct™ MI325X Accelerators (Higher is Better)"):

    idx = np.arange(len(conc_vals))        # evenly spaced positions
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    add_brand_frame(fig)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(False)

    # More headroom for title + subtitles
    fig.subplots_adjust(top=0.85)

    bars = ax.bar(idx, tps_vals, width=0.7, color=color)

    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Tokens/Second")
    ax.set_xticks(idx)
    ax.set_xticklabels(conc_vals)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.15)
    ax.xaxis.grid(False)

    offset = 0.01 * tps_vals.max()
    for bar, val in zip(bars, tps_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=11)

    # Brand-style callout: summary box, left-aligned
    peak_idx = np.argmax(tps_vals)
    peak_val = tps_vals[peak_idx]
    baseline_val = tps_vals[0]
    improvement_vs_baseline = peak_val / baseline_val
    add_callout(
        ax,
        f"Peak: {peak_val:,.0f} tokens/sec\n"
        f"{improvement_vs_baseline:.1f}× vs {baseline_val:,.0f} at {conc_vals[0]} req",
        x=0.02,
        y=0.88,
        ha="left",
        va="top",
        fontsize=13,
        color=color,
    )

    fig.suptitle(main_title, fontsize=20, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.93,
        model_label,
        ha="center", va="top",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.89,
        subtitle,
        ha="center", va="top",
        fontsize=11, color="gray",
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 2) Qwen3-235B FP8 – 2048/2048
# ============================================================

def plot_q235_2048():
    conc_vals = np.array([1, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    tps_vals  = np.array([45, 1236, 2033, 3504, 5686,
                          8253, 10904, 11514, 10732])

    plot_single_model_bar(
        filename="02_q235_2048_2048.png",
        main_title="Throughput for 2048 input tokens and 2048 output tokens",
        model_label="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        conc_vals=conc_vals,
        tps_vals=tps_vals,
        color=MAGENTA,
    )


# ============================================================
# 3) Qwen3-235B FP8 – 128/128
# ============================================================

def plot_q235_128():
    conc_vals = np.array([1, 32, 64, 128, 256, 512,
                          1024, 2048, 4096, 8192])
    tps_vals  = np.array([12, 1281, 2288, 4018, 6922,
                          11167, 15352, 20133, 22787, 23178])

    plot_single_model_bar(
        filename="03_q235_128_128.png",
        main_title="Throughput for 128 input tokens and 128 output tokens",
        model_label="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        conc_vals=conc_vals,
        tps_vals=tps_vals,
        color=MAGENTA,
    )


# ============================================================
# 4) Qwen3-30B – 2048/2048
# ============================================================

def plot_q30_2048():
    conc_vals = np.array([1, 32, 64, 128, 256, 512,
                          1024, 2048, 4096, 8192])
    tps_vals  = np.array([49, 2972, 4852, 8447, 13899,
                          22681, 31111, 41151, 52685, 53710])

    plot_single_model_bar(
        filename="04_q30_2048_2048.png",
        main_title="Throughput for 2048 input tokens and 2048 output tokens",
        model_label="Qwen/Qwen3-30B-A3B-Thinking-2507-BF16",
        conc_vals=conc_vals,
        tps_vals=tps_vals,
        color=PRIMARY_BLUE,
    )


# ============================================================
# 5) Qwen3-30B – 128/128
# ============================================================

def plot_q30_128():
    conc_vals = np.array([1, 32, 64, 128, 256, 512,
                          1024, 2048, 4096, 8192])
    tps_vals  = np.array([9, 3165, 5275, 9421, 16638,
                          29530, 44491, 69641, 96437, 116317])

    plot_single_model_bar(
        filename="05_q30_128_128.png",
        main_title="Throughput for 128 input tokens and 128 output tokens",
        model_label="Qwen/Qwen3-30B-A3B-Thinking-2507-BF16",
        conc_vals=conc_vals,
        tps_vals=tps_vals,
        color=PRIMARY_BLUE,
    )


# ============================================================
# 6) Redfish telemetry endpoints processed vs LLM Throughput
# ============================================================

def plot_redfish_vs_llm():
    redfish_30  = np.array([65.79, 3300.00, 11176.47, 18945.69])
    llm_30      = np.array([499, 7495, 12644, 17783])

    redfish_235 = np.array([66.00, 3300.00, 6600.00, 13198.04])
    llm_235     = np.array([550, 5641, 8214, 7454])

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    add_brand_frame(fig)

    # More headroom for title + subtitles
    fig.subplots_adjust(top=0.85)

    ax.plot(redfish_30,  llm_30,  marker="o", markersize=8, linewidth=3,
            label="Qwen3-30B (BF16)", color=PRIMARY_BLUE)
    ax.plot(redfish_235, llm_235, marker="o", markersize=8, linewidth=3,
            label="Qwen3-235B (FP8)", color=MAGENTA)

    ax.set_xlabel("Redfish telemetry endpoints processed/min")
    ax.set_ylabel("LLM Throughput (tokens/sec)")

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.15)
    ax.xaxis.grid(True, linestyle="-", alpha=0.15)

    leg = ax.legend(loc="lower right")
    leg.get_frame().set_alpha(0.9)

    # Brand-style callouts: bold text in corners
    last_idx = -1
    # Quantify Qwen3-30B BF16 scaling explicitly for better precision in the storyline
    start_eps = redfish_30[0]
    end_eps = redfish_30[last_idx]
    start_tps = llm_30[0]
    end_tps = llm_30[last_idx]
    scale_factor_30b = end_tps / start_tps

    add_callout(
        ax,
        f"Qwen3-30B BF16 grows from {start_tps:,.0f} to {end_tps:,.0f} tokens/sec\n"
        f"as telemetry scales from {start_eps:,.0f} to {end_eps:,.0f} endpoints/min\n"
        f"({scale_factor_30b:.1f}× throughput increase with no saturation observed)",
        x=0.03,
        y=0.86,
        ha="left",
        va="top",
        fontsize=13,
        color=PRIMARY_BLUE,
    )
    add_callout(
        ax,
        "Qwen3-235B FP8 curve flattens\n"
        "beyond ~6,600 endpoints/min",
        x=0.66,
        y=0.30,
        ha="left",
        va="bottom",
        fontsize=12,
        color=MAGENTA,
    )

    fig.suptitle(
        "Redfish Telemetry Endpoints Processed vs LLM Throughput",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5, 0.93,
        "Comparing Qwen3-30B BF16 and Qwen3-235B FP8 Under Increasing Telemetry Load",
        ha="center", va="top",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.89,
        "(Higher is Better)",
        ha="center", va="top",
        fontsize=11,
        color="gray",
    )

    plt.savefig("06_redfish_vs_llm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7) SUM of Redfish metrics processed/min vs Number of Racks
# ============================================================

def plot_redfish_metrics_vs_racks():
    # Qwen3-30B BF16: SUM of Redfish metrics processed per minute per rack count
    racks_r     = np.array([1, 50, 100, 200])
    redfish_sum = np.array([65.79, 3300.00, 11176.47, 18945.69])

    idx = np.arange(len(racks_r))

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    add_brand_frame(fig)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(False)

    # More headroom for title + subtitles
    fig.subplots_adjust(top=0.85)

    bars = ax.bar(idx, redfish_sum, width=0.7, color=PRIMARY_BLUE)

    ax.set_xlabel("Number of Racks Monitored")
    ax.set_ylabel("Qwen3-30B BF16: Redfish metrics processed/min")
    ax.set_xticks(idx)
    ax.set_xticklabels(racks_r)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.15)
    ax.xaxis.grid(False)

    offset = 0.01 * redfish_sum.max()
    for bar, val in zip(bars, redfish_sum):
        ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=11)

    # Brand-style callout: left-aligned, similar style to Graph 8
    last_idx = -1
    add_callout(
        ax,
        f"{redfish_sum[last_idx]:,.0f} metrics/min\n"
        f"at {racks_r[last_idx]} racks (Qwen3-30B BF16)",
        x=0.03,
        y=0.86,
        ha="left",
        va="top",
        fontsize=13,
        color=PRIMARY_BLUE,
    )

    fig.suptitle(
        "Qwen3-30B BF16: Redfish Metrics Processed/Min vs Number of Racks",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5, 0.93,
        "Simulated Redfish Telemetry for Qwen3-30B BF16 on Supermicro Liquid-Cooled Racks",
        ha="center", va="top",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.89,
        "(Higher is Better)",
        ha="center", va="top",
        fontsize=11,
        color="gray",
    )

    plt.savefig("07_redfish_metrics_vs_racks.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8) Redfish telemetry endpoints processed/min vs Number of Racks
# ============================================================

def plot_redfish_endpoints_vs_racks():
    # Same underlying data, explicitly framed as telemetry endpoints/min
    racks_r     = np.array([1, 50, 100, 150, 200])
    redfish_eps = np.array([66, 3300, 6600, 9900, 13198])

    idx = np.arange(len(racks_r))

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    add_brand_frame(fig)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(False)

    # More headroom for title + subtitles
    fig.subplots_adjust(top=0.85)

    bars = ax.bar(idx, redfish_eps, width=0.7, color=PRIMARY_BLUE)

    ax.set_xlabel("Number of Racks Monitored")
    ax.set_ylabel("Redfish Telemetry Endpoints Processed/Min")
    ax.set_xticks(idx)
    ax.set_xticklabels(racks_r)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.15)
    ax.xaxis.grid(False)

    offset = 0.01 * redfish_eps.max()
    for bar, val in zip(bars, redfish_eps):
        ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=11)

    # Brand-style callout: left-aligned
    last_idx = -1
    add_callout(
        ax,
        f"{redfish_eps[last_idx]:,} endpoints/min\n"
        f"monitored at {racks_r[last_idx]} racks",
        x=0.03,
        y=0.86,
        ha="left",
        va="top",
        fontsize=13,
        color=PRIMARY_BLUE,
    )

    fig.suptitle(
        "Qwen3-235B FP8: Redfish Metrics Processed/Min vs Number of Racks",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5, 0.93,
        "Scaling Redfish Telemetry Endpoints With Increasing Rack Counts",
        ha="center", va="top",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.89,
        "(Higher is Better)",
        ha="center", va="top",
        fontsize=11,
        color="gray",
    )

    plt.savefig("08_redfish_endpoints_vs_racks.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Run all plots
# ============================================================

if __name__ == "__main__":
    plot_llm_throughput_comparison()
    plot_q235_2048()
    plot_q235_128()
    plot_q30_2048()
    plot_q30_128()
    plot_redfish_vs_llm()
    plot_redfish_metrics_vs_racks()
    plot_redfish_endpoints_vs_racks()
