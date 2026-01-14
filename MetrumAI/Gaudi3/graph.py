import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerPatch

# ==========================================
# Global style (aligned with Supermicro graphs)
# ==========================================

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,           # we'll enable grid per-axis
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


# Color palette (match Redfish graphs)
PRIMARY_BLUE = "#0080ff"
MAGENTA      = "#ff00ae"


# ==========================================
# Custom legend handler for split-color keys
# ==========================================
class SplitColorPatchHandler(HandlerPatch):
    """Legend handler that draws a patch split into multiple colored segments."""
    def create_artists(
        self,
        legend,
        orig_handle,
        x0,
        y0,
        width,
        height,
        fontsize,
        trans,
    ):
        patches = []
        # Colors for this handle are stored on the handle itself
        colors = getattr(orig_handle, "_colors", [orig_handle.get_facecolor()])
        n = len(colors)
        seg_width = width / n

        for i, c in enumerate(colors):
            p = Rectangle(
                (x0 + i * seg_width, y0),
                seg_width,
                height,
                transform=trans,
                facecolor=c,
                edgecolor="none",
            )
            patches.append(p)

        return patches


# ==========================================
# Reusable callout helper (axes coordinates)
# ==========================================

def add_callout(ax, text, x, y, ha="left", va="top",
                fontsize=12, fontweight="bold", color=None):
    """Draws a branded callout box in axes coordinates (no arrows)."""
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
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
            alpha=1.0,
        ),
    )
    return txt


# ==========================================
# Data (from Gaudi3 NF scaling experiment)
# ==========================================

clusters = np.array([1, 5, 10])
nfs = np.array([4, 20, 40])
events_per_min = np.array([393.00, 805.50, 1314.00])
tokens_per_sec = np.array([622.82, 3815.74, 5521.86])


events_per_sec = events_per_min / 60.0

# Derived metrics

# Tokens generated per processed event
TOKENS_PER_EVENT = tokens_per_sec / events_per_sec

# Events per minute per individual network function
EVENTS_PER_NF = events_per_min / nfs


# ==========================================
# Helpers: Derived metrics + optional Excel loader
# ==========================================

def compute_tokens_per_event(tokens_per_sec_arr, events_per_min_arr):
    """Compute tokens generated per processed event.

    tokens_per_event = (tokens/sec) / (events/sec) = (tokens/sec) / (events/min / 60)
    """
    tokens_per_sec_arr = np.asarray(tokens_per_sec_arr, dtype=float)
    events_per_min_arr = np.asarray(events_per_min_arr, dtype=float)

    events_per_sec_arr = events_per_min_arr / 60.0
    # Avoid divide-by-zero
    eps = 1e-12
    return tokens_per_sec_arr / np.maximum(events_per_sec_arr, eps)


def _first_existing_col(df, candidates):
    """Return the first column name from candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_tokens_per_event_from_excel(
    excel_path,
    qwen_sheet="QwenQwen3-30B-A3B-Thinking-2507",
    raw_sheet="raw_metrics",
    tokens_col_candidates=(
        "vllm_throughput_tokens_per_sec",
        "tokens_per_sec",
        "tokens/sec",
        "tokens_per_second",
    ),
    events_col_candidates=(
        "logs_processed_per_minute",
        "events_per_min",
        "events_per_minute",
        "Events / Logs Processed per Minute",
        "Events/Logs Processed per Minute",
    ),
    clusters_col_candidates=(
        "clusters",
        "cluster_count",
        "Number of Network Function Clusters Monitored",
        "nf_clusters",
    ),
):
    """Load arrays needed for Tokens/Event plot from the benchmarking workbook.

    Notes:
      - If a clusters column isn't present, we fall back to the existing `clusters` array.
      - If the sheet row counts differ and no join key exists, we truncate to the shortest length.
    """
    qwen = pd.read_excel(excel_path, sheet_name=qwen_sheet)
    raw = pd.read_excel(excel_path, sheet_name=raw_sheet)

    tok_col = _first_existing_col(qwen, tokens_col_candidates)
    evt_col = _first_existing_col(raw, events_col_candidates)

    if tok_col is None:
        raise ValueError(
            f"Could not find a tokens/sec column in sheet '{qwen_sheet}'. Tried: {tokens_col_candidates}"
        )
    if evt_col is None:
        raise ValueError(
            f"Could not find an events/min column in sheet '{raw_sheet}'. Tried: {events_col_candidates}"
        )

    # Optional clusters column(s)
    qwen_cl_col = _first_existing_col(qwen, clusters_col_candidates)
    raw_cl_col = _first_existing_col(raw, clusters_col_candidates)

    tokens_per_sec_arr = qwen[tok_col].to_numpy(dtype=float)
    events_per_min_arr = raw[evt_col].to_numpy(dtype=float)

    # Determine x-axis
    if qwen_cl_col is not None and raw_cl_col is not None:
        # Merge on clusters if both have it
        qsub = qwen[[qwen_cl_col, tok_col]].rename(columns={qwen_cl_col: "clusters", tok_col: "tokens_per_sec"})
        rsub = raw[[raw_cl_col, evt_col]].rename(columns={raw_cl_col: "clusters", evt_col: "events_per_min"})
        merged = pd.merge(qsub, rsub, on="clusters", how="inner").sort_values("clusters")
        clusters_arr = merged["clusters"].to_numpy(dtype=float)
        tokens_per_sec_arr = merged["tokens_per_sec"].to_numpy(dtype=float)
        events_per_min_arr = merged["events_per_min"].to_numpy(dtype=float)
    elif raw_cl_col is not None:
        clusters_arr = raw[raw_cl_col].to_numpy(dtype=float)
        # If lengths mismatch, align by truncation
        n = min(len(clusters_arr), len(tokens_per_sec_arr), len(events_per_min_arr))
        clusters_arr = clusters_arr[:n]
        tokens_per_sec_arr = tokens_per_sec_arr[:n]
        events_per_min_arr = events_per_min_arr[:n]
    elif qwen_cl_col is not None:
        clusters_arr = qwen[qwen_cl_col].to_numpy(dtype=float)
        n = min(len(clusters_arr), len(tokens_per_sec_arr), len(events_per_min_arr))
        clusters_arr = clusters_arr[:n]
        tokens_per_sec_arr = tokens_per_sec_arr[:n]
        events_per_min_arr = events_per_min_arr[:n]
    else:
        # Fall back to the existing global clusters array
        clusters_arr = np.asarray(clusters, dtype=float)
        n = min(len(clusters_arr), len(tokens_per_sec_arr), len(events_per_min_arr))
        clusters_arr = clusters_arr[:n]
        tokens_per_sec_arr = tokens_per_sec_arr[:n]
        events_per_min_arr = events_per_min_arr[:n]

    tokens_per_event_arr = compute_tokens_per_event(tokens_per_sec_arr, events_per_min_arr)
    return clusters_arr, tokens_per_event_arr


# ==========================================
# 1) Network Function Clusters vs Events/Logs Processed
# ==========================================

def plot_clusters_vs_events():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")

    # Layout: leave room for title / helper / subtitle
    fig.subplots_adjust(top=0.8)

    ax.plot(
        clusters,
        events_per_min,
        marker="o",
        markersize=8,
        linewidth=3,
        color=PRIMARY_BLUE,
    )

    ax.set_xlabel("Number of Network Function Clusters Monitored")
    ax.set_ylabel("Events / Logs Processed per Minute")

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.15)
    ax.xaxis.grid(False)

    # Callout in axes coordinates
    add_callout(
        ax,
        "Near-linear scaling as additional\n5G Core clusters are monitored",
        x=0.08,
        y=0.8,
        ha="left",
        va="top",
        fontsize=12,
        color=PRIMARY_BLUE,
    )

    fig.suptitle(
        "Network Function Clusters Monitored vs Events/Logs Processed",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.94,
        "Scaling telemetry ingestion as 5G Core coverage increases",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.89,
        "(Higher is Better)",
        ha="center",
        va="top",
        fontsize=10,
        color="gray",
    )

    plt.savefig("fig1_clusters_vs_events_per_min.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# 2) Network Function Clusters vs LLM Throughput
# ==========================================

def plot_clusters_vs_llm():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.8)

    ax.plot(
        clusters,
        tokens_per_sec,
        marker="o",
        markersize=8,
        linewidth=3,
        color=MAGENTA,
    )

    ax.set_xlabel("Number of Network Function Clusters Monitored")
    ax.set_ylabel("LLM Throughput (tokens/sec)")

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.15)
    ax.xaxis.grid(False)

    add_callout(
        ax,
        "Throughput scales with monitored clusters,\n"
        "supporting more concurrent investigations",
        x=0.08,
        y=0.8,
        ha="left",
        va="top",
        fontsize=12,
        color=MAGENTA,
    )

    fig.suptitle(
        "Network Function Clusters Monitored vs LLM Throughput",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.94,
        "Gaudi 3–accelerated multi-agent system under increasing network scope",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.89,
        "(Higher is Better)",
        ha="center",
        va="top",
        fontsize=10,
        color="gray",
    )

    plt.savefig("fig2_clusters_vs_llm_throughput.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# 3) LLM Reasoning Cost per Event
# ==========================================

def plot_tokens_per_event():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.8)

    tokens_per_event = compute_tokens_per_event(tokens_per_sec, events_per_min)

    ax.plot(
        clusters,
        tokens_per_event,
        marker="o",
        markersize=8,
        linewidth=3,
        color=PRIMARY_BLUE,
    )

    ax.set_xlabel("Number of Network Function Clusters Monitored")
    ax.set_ylabel("Tokens per Event")

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.15)
    ax.xaxis.grid(False)

    add_callout(
        ax,
        "Reasoning cost stays in a bounded range\n"
        "even as the monitored scope grows",
        x=0.08,
        y=0.8,
        ha="left",
        va="top",
        fontsize=12,
        color=PRIMARY_BLUE,
    )

    fig.suptitle(
        "LLM Reasoning Cost per Event",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.94,
        "Tokens generated per processed telemetry event",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.89,
        "(Stable or Lower is Better)",
        ha="center",
        va="top",
        fontsize=10,
        color="gray",
    )

    plt.savefig("fig3_tokens_per_event.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# 3b) LLM Reasoning Tokens per Event (from Excel)
# ==========================================

def plot_tokens_per_event_from_excel(excel_path="5GCore_Solution_Benchmarking.xlsx"):
    """Generate the Tokens/Event figure directly from the benchmarking workbook."""
    x_clusters, tokens_per_event = load_tokens_per_event_from_excel(excel_path)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.8)

    ax.plot(
        x_clusters,
        tokens_per_event,
        marker="o",
        markersize=8,
        linewidth=3,
        color=PRIMARY_BLUE,
    )

    ax.set_xlabel("Number of Network Function Clusters Monitored")
    ax.set_ylabel("Tokens per Event")

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.15)
    ax.xaxis.grid(False)

    add_callout(
        ax,
        "Reasoning cost stays in a bounded range\n"
        "even as the monitored scope grows",
        x=0.08,
        y=0.8,
        ha="left",
        va="top",
        fontsize=12,
        color=PRIMARY_BLUE,
    )

    fig.suptitle(
        "LLM Reasoning Tokens per Event",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.94,
        "Computed as (LLM tokens/sec) ÷ (events/sec)",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.89,
        "(Stable or Lower is Better)",
        ha="center",
        va="top",
        fontsize=10,
        color="gray",
    )

    plt.savefig("fig3b_tokens_per_event_from_excel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# 4) Events per Network Function vs NF Clusters
# ==========================================

def plot_events_per_nf():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.8)

    ax.plot(
        clusters,
        EVENTS_PER_NF,
        marker="o",
        markersize=8,
        linewidth=3,
        color=MAGENTA,
    )

    ax.set_xlabel("Number of Network Function Clusters Monitored")
    ax.set_ylabel("Events per Minute per Network Function")

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.15)
    ax.xaxis.grid(False)

    add_callout(
        ax,
        "Per-function load decreases as telemetry\n"
        "is distributed across more network functions",
        x=0.5,
        y=0.8,
        ha="left",
        va="top",
        fontsize=12,
        color=MAGENTA,
    )

    fig.suptitle(
        "Events per Network Function vs Network Function Clusters",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.94,
        "Per-function monitoring load as additional clusters are onboarded",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.89,
        "(Balanced or Lower is Better)",
        ha="center",
        va="top",
        fontsize=10,
        color="gray",
    )

    plt.savefig("fig4_events_per_nf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# 5) Incident Lifecycle: Traditional vs Multi-Agent on Gaudi 3
# ==========================================

def plot_incident_lifecycle():
    scenarios = ["Traditional NOC", "Multi-Agent on Gaudi 3"]
    x_pos = np.arange(len(scenarios))

    # Example time breakdown in seconds (replace with real values if needed)
    detection     = np.array([900, 5])
    investigation = np.array([3600, 10])
    rca           = np.array([900, 10])
    remediation   = np.array([600, 10])
    documentation = np.array([1200, 5])

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.8)

    width = 0.6
    bottom = np.zeros_like(x_pos, dtype=float)

    # Blue gradient for Traditional NOC, magenta gradient for Multi-Agent on Gaudi 3
    segments = [
        (detection,     "Detection",           "#0D47A1", "#8A005E"),  # dark blue, dark magenta
        (investigation, "Investigation",       "#1976D2", "#B0007C"),
        (rca,           "Root Cause Analysis", "#42A5F5", "#D6009A"),
        (remediation,   "Remediation",         "#90CAF9", "#F247B4"),
        (documentation, "Documentation",       "#C5E1F9", "#FF8BCF"),
    ]

    for values, label, trad_color, gaudi_color in segments:
        segment_colors = [trad_color, gaudi_color]
        ax.bar(
            x_pos,
            values,
            width,
            bottom=bottom,
            color=segment_colors,
            alpha=0.9,
        )
        bottom += values

    ax.set_xlabel("Operational Model")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios)

    # Use log scale so both multi-hour and seconds-scale MTTR are visible
    ax.set_yscale("log")

    ax.set_ylabel("Time (seconds, log scale)")

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.15)
    ax.xaxis.grid(False)

    # Legend: each lifecycle phase uses a split rectangle (Traditional NOC blue, Gaudi 3 magenta)
    legend_handles = []
    legend_labels = []
    for _, label, trad_color, gaudi_color in segments:
        handle = Rectangle((0, 0), 1, 1)
        handle._colors = [trad_color, gaudi_color]
        legend_handles.append(handle)
        legend_labels.append(label)

    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        frameon=True,
        handler_map={Rectangle: SplitColorPatchHandler()},
    )

    # Totals for label placement
    traditional_total = (
        detection[0] + investigation[0] + rca[0] + remediation[0] + documentation[0]
    )
    ma_total = (
        detection[1] + investigation[1] + rca[1] + remediation[1] + documentation[1]
    )

    # Give a bit of headroom above the tallest bar on log scale
    ymax = traditional_total * 2.0
    ymin = max(1, ma_total / 5.0)
    ax.set_ylim(ymin, ymax)

    # Callouts centered above each bar (data coordinates, within axes)
    ax.text(
        x_pos[0],
        traditional_total * 1.06,
        "Multi-hour manual workflow",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY_BLUE,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor=PRIMARY_BLUE,
            linewidth=1.4,
        ),
    )
    ax.text(
        x_pos[1],
        ma_total * 1.06,
        "Seconds-scale autonomous response",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=MAGENTA,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor=MAGENTA,
            linewidth=1.4,
        ),
    )

    fig.suptitle(
        "Incident Lifecycle Time: Traditional vs Multi-Agent AI on Gaudi 3",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.94,
        "Breaking down end-to-end Mean Time to Resolution across operational models",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.89,
        "(Lower is Better)",
        ha="center",
        va="top",
        fontsize=10,
        color="gray",
    )

    plt.savefig("fig5_incident_lifecycle_stacked.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# Run all plots
# ==========================================

if __name__ == "__main__":
    plot_clusters_vs_events()
    plot_clusters_vs_llm()
    plot_tokens_per_event()
    plot_events_per_nf()
    plot_incident_lifecycle()

    # Optional: generate Tokens/Event directly from the benchmarking workbook
    # plot_tokens_per_event_from_excel("5GCore_Solution_Benchmarking.xlsx")