"""
generate_url_benchmarks.py

Generates four figures with a dark, whitepaper-style theme:
  Fig 5  – URLs/Second vs Batch Size (Single vs Multi-node)
  Fig 6  – Learning Rate vs Input Requests
  Fig 7  – URLs Evaluated per Minute vs Input Requests
  Fig 8  – End-to-End Response Time vs Batch Size
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from pathlib import Path

# ============================================================
# 1) RAW DATA FROM YOUR SHEETS
# ============================================================

# ---------- Table 1: Learning & URLs/min ----------
input_requests = np.array([1, 5, 15])

# Single-node rows (No. of nodes = 1)
learning_single = np.array([0.41, 1.68, 3.58])   # Learning Rate column
urls_eval_single = np.array([0.41, 1.68, 3.58])  # URLs evaluated/min column

# Multi-node rows (No. of nodes = 2)
learning_multi = np.array([0.86, 2.34, 5.15])    # Learning Rate column
urls_eval_multi = np.array([0.86, 2.34, 5.83])   # URLs evaluated/min column

# ---------- Table 2: Throughput & Latency ----------
batch_sizes = np.array([1, 50, 100, 300, 350])

# URLs/second
urls_per_sec_single = np.array([90225, 74184, 805312, 875151, 955864])
urls_per_sec_multi  = np.array([221000, 1770000, 1610000, 1810000, 1930000])

# End-to-end response time in ms
response_time_single = np.array([8.867, 539.201, 99.34, 274.238, 292.929])
response_time_multi  = np.array([7.0,   45.0,    99.0, 265.0,   290.0])  # last is ~290ms

# ============================================================
# 2) GLOBAL STYLE HELPERS
# ============================================================

# Dark background + clean axes, shared across all plots
BG_COLOR = "#050816"      # deep navy
AXIS_COLOR = "#E0E6F3"    # light desaturated for ticks/labels
GRID_COLOR = "#2A3555"
LINE_SINGLE = "#4FC3F7"   # cyan / blue accent
LINE_MULTI = "#FFB74D"    # warm orange accent

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.facecolor": BG_COLOR,
    "figure.facecolor": BG_COLOR,
    "axes.edgecolor": AXIS_COLOR,
    "axes.labelcolor": AXIS_COLOR,
    "xtick.color": AXIS_COLOR,
    "ytick.color": AXIS_COLOR,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "grid.color": GRID_COLOR,
    "legend.edgecolor": "none",
})

def style_axes(ax, title, xlabel, ylabel):
    """Apply consistent styling to an axes object."""
    ax.set_title(title, color=AXIS_COLOR, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Only left/bottom spines for a cleaner look
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_linewidth(1.0)

    ax.tick_params(axis="both", which="both", length=4)

    # Legend styling
    leg = ax.legend(frameon=True, facecolor="#10172F")
    for text in leg.get_texts():
        text.set_color(AXIS_COLOR)

# Formatter for the URLs/second axis (1.8M, 900K, etc.)
def millions_formatter(x, pos):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    elif x >= 1_000:
        return f"{x/1_000:.0f}K"
    return f"{int(x)}"

# ============================================================
# HELPER FUNCTIONS FOR LLM TOKENS-PER-WATT BAR CHARTS
# ============================================================

def load_tokens_per_watt(path: Path, model_name: str, node_label: str) -> pd.DataFrame:
    """
    Load a tokens_per_watt_pivot_table CSV and normalize columns for a given model.
    """
    df = pd.read_csv(path)

    # Forward-fill metadata columns (model, token lengths, etc.)
    meta_cols = ["model", "input_tokens_length", "max_output_tokens"]
    for col in meta_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Normalize tokens/sec/watt column name
    if "tokens/sec/watt" in df.columns:
        eff_col = "tokens/sec/watt"
    elif "Tokens/sec/watt" in df.columns:
        eff_col = "Tokens/sec/watt"
    else:
        raise ValueError(f"Efficiency column not found in {path}")

    df = df.rename(columns={eff_col: "tokens_per_watt"})

    # Keep only the model we care about
    if "model" in df.columns:
        df = df[df["model"].astype(str).str.contains(model_name, na=False)].copy()
    df["node"] = node_label
    return df


def summarize_best_by_tokens(df: pd.DataFrame, node_label: str) -> pd.DataFrame:
    """
    For each (input_tokens, max_output_tokens) combination, find the row with
    the highest tokens_completion_per_second and record both throughput and
    efficiency at that operating point.
    """
    if df.empty:
        return pd.DataFrame(columns=["node", "input_tokens", "output_tokens", "tokens_per_second", "tokens_per_watt"])

    rows = []
    grouped = df.groupby(["input_tokens_length", "max_output_tokens"])

    for (inp, out), grp in grouped:
        if pd.isna(inp) or pd.isna(out):
            continue

        # Row with max completion rate
        idx = grp["tokens_completion_per_second"].idxmax()
        best = grp.loc[idx]

        rows.append(
            {
                "node": node_label,
                "input_tokens": int(inp),
                "output_tokens": int(out),
                "tokens_per_second": float(best["tokens_completion_per_second"]),
                "tokens_per_watt": float(best["tokens_per_watt"]),
            }
        )

    return pd.DataFrame(rows)

# ============================================================
# 3) FIGURE 5 – URLs/Second vs Batch Size (Single vs Multi)
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    batch_sizes,
    urls_per_sec_single,
    marker="o",
    linewidth=2.2,
    markersize=6,
    label="Single node",
    color=LINE_SINGLE,
)
ax.plot(
    batch_sizes,
    urls_per_sec_multi,
    marker="o",
    linewidth=2.2,
    markersize=6,
    label="Multi-node",
    color=LINE_MULTI,
)

style_axes(
    ax,
    "URLs/Second vs. Batch Size by Number of Nodes",
    "Batch Size",
    "URLs per Second",
)

ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

fig.tight_layout()
fig.savefig("fig5_urls_per_second_vs_batch_size.png", transparent=False)
plt.close(fig)

# ============================================================
# 4) FIGURE 6 – Learning Rate vs Input Requests
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    input_requests,
    learning_single,
    marker="o",
    linewidth=2.2,
    markersize=6,
    label="Single node",
    color=LINE_SINGLE,
)
ax.plot(
    input_requests,
    learning_multi,
    marker="o",
    linewidth=2.2,
    markersize=6,
    label="Multi-node",
    color=LINE_MULTI,
)

style_axes(
    ax,
    "Learning Rate vs. Input Requests by Number of Nodes",
    "Input Requests",
    "Learning Rate (malicious URLs updated to cache / min)",
)

fig.tight_layout()
fig.savefig("fig6_learning_rate_vs_input_requests.png", transparent=False)
plt.close(fig)

# ============================================================
# 5) FIGURE 7 – URLs Evaluated per Minute vs Input Requests
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    input_requests,
    urls_eval_single,
    marker="o",
    linewidth=2.2,
    markersize=6,
    label="Single node",
    color=LINE_SINGLE,
)
ax.plot(
    input_requests,
    urls_eval_multi,
    marker="o",
    linewidth=2.2,
    markersize=6,
    label="Multi-node",
    color=LINE_MULTI,
)

style_axes(
    ax,
    "URLs Evaluated per Minute vs. Input Requests by Number of Nodes",
    "Input Requests",
    "URLs Evaluated per Minute",
)

fig.tight_layout()
fig.savefig("fig7_urls_per_min_vs_input_requests.png", transparent=False)
plt.close(fig)

# ============================================================
# 6) FIGURE 8 – End-to-End Response Time vs Batch Size
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    batch_sizes,
    response_time_single,
    marker="o",
    linewidth=2.2,
    markersize=6,
    label="Single node",
    color=LINE_SINGLE,
)
ax.plot(
    batch_sizes,
    response_time_multi,
    marker="o",
    linewidth=2.2,
    markersize=6,
    label="Multi-node",
    color=LINE_MULTI,
)

style_axes(
    ax,
    "End-to-End Response Time vs. Batch Size by Number of Nodes",
    "Batch Size",
    "End-to-End Response Time (ms)",
)

fig.tight_layout()
fig.savefig("fig8_latency_vs_batch_size.png", transparent=False)
plt.close(fig)

# ============================================================
# 7) LLM TOKEN THROUGHPUT & EFFICIENCY BAR CHARTS (TAKEAWAY 2)
# ============================================================

# Paths to the tokens-per-watt pivot tables (adjust if needed)
single_tokens_path = Path("vonage-url-analysis-agent-solution-benchmark - tokens_per_watt_pivot_table_single_node.csv")
multi_tokens_path = Path("vonage-url-analysis-agent-solution-benchmark - tokens_per_watt_pivot_table_multi_node.csv")

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TARGET_TOKEN_WINDOWS = [(128, 128), (2048, 2048)]  # (input, output)

llm_figs_created = []

try:
    single_raw = load_tokens_per_watt(single_tokens_path, MODEL_NAME, "Single node")
    multi_raw = load_tokens_per_watt(multi_tokens_path, MODEL_NAME, "Multi-node")

    single_summary = summarize_best_by_tokens(single_raw, "Single node")
    multi_summary = summarize_best_by_tokens(multi_raw, "Multi-node")

    summary = pd.concat([single_summary, multi_summary], ignore_index=True)

    # Filter to the token windows we care about (128/128 and 2048/2048)
    if not summary.empty:
        mask = summary.apply(
            lambda r: (r["input_tokens"], r["output_tokens"]) in TARGET_TOKEN_WINDOWS,
            axis=1,
        )
        summary = summary[mask].copy()

    # If still empty, fall back to all available windows
    if summary.empty and not single_summary.empty and not multi_summary.empty:
        summary = pd.concat([single_summary, multi_summary], ignore_index=True)

    if not summary.empty:
        # Build grouped data by token window and node
        windows = sorted({(int(r["input_tokens"]), int(r["output_tokens"])) for _, r in summary.iterrows()})
        labels = [f"{inp}/{out}" for inp, out in windows]
        x = np.arange(len(labels))
        width = 0.35

        def get_values(node_label, field):
            vals = []
            for inp, out in windows:
                row = summary[(summary["node"] == node_label) &
                              (summary["input_tokens"] == inp) &
                              (summary["output_tokens"] == out)]
                if not row.empty:
                    vals.append(float(row.iloc[0][field]))
                else:
                    vals.append(0.0)
            return vals

        single_tokens = get_values("Single node", "tokens_per_second")
        multi_tokens = get_values("Multi-node", "tokens_per_second")

        single_eff = get_values("Single node", "tokens_per_watt")
        multi_eff = get_values("Multi-node", "tokens_per_watt")

        # BAR CHART 1: COMPLETION RATE (TOKENS/SEC)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(x - width / 2, single_tokens, width, label="Single node", color=LINE_SINGLE)
        ax.bar(x + width / 2, multi_tokens, width, label="Multi-node", color=LINE_MULTI)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        style_axes(
            ax,
            "Qwen3-30B Max Completion Rate by Node & Token Window",
            "Token Window (input/output tokens)",
            "Max Completion Rate (tokens/sec)",
        )
        ax.grid(False)
        # Add centered labels above bars
        for i, val in enumerate(single_tokens):
            ax.text(x[i] - width/2, val + (val * 0.02 if val != 0 else 0.02),
                    f"{val:.0f}", ha="center", va="bottom", color=AXIS_COLOR, fontsize=10)
        for i, val in enumerate(multi_tokens):
            ax.text(x[i] + width/2, val + (val * 0.02 if val != 0 else 0.02),
                    f"{val:.0f}", ha="center", va="bottom", color=AXIS_COLOR, fontsize=10)
        fig.tight_layout()
        fig.savefig("fig9_llm_completion_rate_takeaway2.png", transparent=False)
        plt.close(fig)
        llm_figs_created.append("fig9_llm_completion_rate_takeaway2.png")

        # BAR CHART 2: ENERGY EFFICIENCY (TOKENS/SEC/WATT)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(x - width / 2, single_eff, width, label="Single node", color=LINE_SINGLE)
        ax.bar(x + width / 2, multi_eff, width, label="Multi-node", color=LINE_MULTI)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        style_axes(
            ax,
            "Qwen3-30B Energy Efficiency by Node & Token Window",
            "Token Window (input/output tokens)",
            "Max Energy Efficiency (tokens/sec/watt)",
        )
        ax.grid(False)
        # Add centered labels above bars
        for i, val in enumerate(single_eff):
            ax.text(x[i] - width/2, val + (val * 0.02 if val != 0 else 0.02),
                    f"{val:.2f}", ha="center", va="bottom", color=AXIS_COLOR, fontsize=10)
        for i, val in enumerate(multi_eff):
            ax.text(x[i] + width/2, val + (val * 0.02 if val != 0 else 0.02),
                    f"{val:.2f}", ha="center", va="bottom", color=AXIS_COLOR, fontsize=10)
        fig.tight_layout()
        fig.savefig("fig10_llm_efficiency_takeaway2.png", transparent=False)
        plt.close(fig)
        llm_figs_created.append("fig10_llm_efficiency_takeaway2.png")

except FileNotFoundError:
    # If the CSVs are not present, just skip LLM charts gracefully
    pass

print("Saved figures (whitepaper style):")
print("  fig5_urls_per_second_vs_batch_size.png")
print("  fig6_learning_rate_vs_input_requests.png")
print("  fig7_urls_per_min_vs_input_requests.png")
print("  fig8_latency_vs_batch_size.png")
if llm_figs_created:
    for fname in llm_figs_created:
        print(f"  {fname}")