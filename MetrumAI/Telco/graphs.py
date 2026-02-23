import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Brand colors
# -----------------------------
BLUE = "#2f56ec"   # MI355X / primary
RED  = "#ff3132"   # MI300X / comparison baseline

# -----------------------------
# Data (from the paper)
# -----------------------------
bbus = np.array([3, 6, 15, 30])

# Table 6: MI355X scalability
events_355x = np.array([539, 828, 1515, 2253])

# Table 6a / generational comparison: MI300X
events_300x = np.array([323, 512, 766, 1470])

# Latency profile (section text)
ttft_ms = (200, 245)        # ms
workflow_s = (102, 463)     # seconds
p95_s = (43, 390)           # seconds
min_workflow_s = 45         # seconds

# -----------------------------
# Styling helpers
# -----------------------------
def style_axes(ax):
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

def save(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

# -----------------------------
# FIGURE 1: Scalability (Events/min vs BBUs)
# -----------------------------
def fig_scalability_events():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.plot(bbus, events_355x, marker="o", linewidth=2.6, color=BLUE, label="MI355X")
    ax.plot(bbus, events_300x, marker="o", linewidth=2.6, linestyle="--", color=RED, label="MI300X")

    ax.set_title("Scalability: Events Processed per Minute vs BBUs", fontsize=12)
    ax.set_xlabel("BBUs Monitored")
    ax.set_ylabel("Events / min")
    style_axes(ax)
    ax.legend(frameon=False)

    # Peak callout
    ax.annotate("Peak: 2,253 events/min",
                xy=(30, 2253), xytext=(16, 2420),
                arrowprops=dict(arrowstyle="->", lw=1.2, alpha=0.85),
                fontsize=10)

    save(fig, "fig_01_scalability_events_per_min.png")

# -----------------------------
# FIGURE 2: Peak Load Comparison (30 BBUs)
# -----------------------------
def fig_peak_load_events_and_uplift():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    categories = ["Events/min (30 BBUs)"]
    mi355x_vals = [events_355x[-1]]
    mi300x_vals = [events_300x[-1]]

    x = np.arange(len(categories))
    w = 0.38

    ax.bar(x - w/2, mi355x_vals, width=w, color=BLUE, label="MI355X")
    ax.bar(x + w/2, mi300x_vals, width=w, color=RED, label="MI300X")

    ax.set_title("Peak Load Comparison (30 BBUs / 90 RRHs)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Events / min")
    style_axes(ax)
    ax.legend(frameon=False)

    uplift_events = (events_355x[-1] / events_300x[-1] - 1) * 100
    ax.text(x[0], max(mi355x_vals[0], mi300x_vals[0]) * 1.06,
            f"MI355X is +{uplift_events:.0f}% vs MI300X",
            ha="center", fontsize=11)

    # Value labels
    ax.text(x[0] - w/2, mi355x_vals[0] + 35, f"{mi355x_vals[0]:,}", ha="center", fontsize=10)
    ax.text(x[0] + w/2, mi300x_vals[0] + 35, f"{mi300x_vals[0]:,}", ha="center", fontsize=10)

    save(fig, "fig_02_peak_load_comparison_events.png")

# -----------------------------
# FIGURE 3: Latency Profile (range bars)
# -----------------------------
def fig_latency_profile():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    # Convert TTFT to seconds for a common axis
    metrics = ["TTFT (s)", "Workflow Duration (s)", "P95 Inference Latency (s)"]
    starts = [ttft_ms[0] / 1000.0, workflow_s[0], p95_s[0]]
    ends   = [ttft_ms[1] / 1000.0, workflow_s[1], p95_s[1]]
    widths = [e - s for s, e in zip(starts, ends)]
    y = np.arange(len(metrics))

    ax.barh(y, widths, left=starts, color=BLUE, alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds")
    ax.set_title("Latency Profile (Ranges)", fontsize=12)
    style_axes(ax)

    # Mark the minimum workflow completion (45s) on the Workflow line
    ax.scatter([min_workflow_s], [1], color=RED, s=40, zorder=3)
    ax.text(min_workflow_s + 8, 1, "Min completion: 45s", va="center", fontsize=10)

    # Range labels
    for i, (s, e) in enumerate(zip(starts, ends)):
        label = f"{s:.3g}–{e:.3g}"
        offset = 0.03 if i == 0 else 8
        ax.text(e + offset, i, label, va="center", fontsize=10)

    save(fig, "fig_03_latency_profile_ranges.png")

# -----------------------------
# Run selected figures
# -----------------------------
if __name__ == "__main__":
    fig_scalability_events()
    fig_peak_load_events_and_uplift()
    fig_latency_profile()