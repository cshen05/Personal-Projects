import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ============================================================
# Brand Colors
# ============================================================
BLUE = "#2f56ec"
RED = "#ff3132"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
})

def style_axes(ax):
    ax.grid(True, axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def thousands(x, pos):
    return f"{int(x):,}"

# ============================================================
# Common categorical axis for Figures 6 & 7
# ============================================================

labels = ["300","350","400","450","500","550","600","650","700","750","800","850"]
x = np.arange(len(labels))

# ============================================================
# FIGURE 6 — Holdings Scaling (BLUE)
# ============================================================

holdings = np.array([
    557.91, 654.96, 740.73, 817.28, 890.51, 953.68,
    1017.79, 1088.12, 1189.66, 1282.24, 1288.75, 1415.39
])

fig, ax = plt.subplots(figsize=(9.5,5.2))

bars = ax.bar(x, holdings, color=BLUE, width=0.7)

ax.set_title("Holdings Analyzed per Minute by Concurrent Portfolio Count", pad=12)
ax.set_xlabel("Concurrent Portfolios")
ax.set_ylabel("Holdings / minute")

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylim(0, 1550)
style_axes(ax)

for b in bars:
    h = b.get_height()
    ax.text(
        b.get_x() + b.get_width()/2,
        h + 20,
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.savefig("fig6_holdings_scaling.png")
plt.close()

# ============================================================
# FIGURE 7 — Compliance Scaling (RED)
# ============================================================

compliance = np.array([
    53.35, 61.01, 71.04, 77.15, 84.18, 89.80,
    94.92, 97.86, 101.87, 104.87, 107.89, 108.90
])

fig, ax = plt.subplots(figsize=(9.5,5.2))

bars = ax.bar(x, compliance, color=RED, width=0.7)

ax.set_title("Compliance Checks per Minute by Concurrent Portfolio Count", pad=12)
ax.set_xlabel("Concurrent Portfolios")
ax.set_ylabel("Compliance Checks / minute")

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylim(0, 120)
style_axes(ax)

for b in bars:
    h = b.get_height()
    ax.text(
        b.get_x() + b.get_width()/2,
        h + 2,
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.savefig("fig7_compliance_scaling.png")
plt.close()

# ============================================================
# FIGURE 8 — Raw Inference Scaling (Grouped Bars)
# MI355X vs MI300X
# ============================================================

raw_concurrency = np.array([128, 512, 1024, 2048, 4096])
mi355x = np.array([39930, 10481, 16856, 24767, 39830])
mi300x = np.array([9568, 7426, 9498, 9488, 10441])

x2 = np.arange(len(raw_concurrency))
width = 0.36

fig, ax = plt.subplots(figsize=(10,5.5))

bars1 = ax.bar(
    x2 - width/2,
    mi355x,
    width,
    label="MI355X",
    color=BLUE
)

bars2 = ax.bar(
    x2 + width/2,
    mi300x,
    width,
    label="MI300X",
    color=RED
)

ax.set_title(
    "Raw Inference Throughput Scaling\nMI355X vs MI300X (Qwen3-235B)",
    pad=12
)

ax.set_ylabel("Tokens / sec")
ax.set_xlabel("Concurrent Requests")

ax.set_xticks(x2)
ax.set_xticklabels(raw_concurrency)

ax.yaxis.set_major_formatter(FuncFormatter(thousands))
style_axes(ax)
ax.legend(frameon=False)

ax.set_ylim(0, 45000)

# Multiplier labels centered above each pair
ratios = mi355x / mi300x

for i in range(len(raw_concurrency)):
    ymax = max(mi355x[i], mi300x[i])
    ax.text(
        i,
        ymax + 1500,
        f"{ratios[i]:.1f}×",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig("fig8_raw_inference_scaling.png")
plt.close()

print("All IPRA replacement figures generated successfully.")