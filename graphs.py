import matplotlib.pyplot as plt
import numpy as np

# --- data ---
racks = np.array([1, 50, 100, 200])

# Max LLM throughput (tokens/sec)
q30   = np.array([550, 5641,  8214,  7454])    # Qwen3-30B
q235  = np.array([499, 7495, 12644, 17783])    # Qwen3-235B FP8

x = np.arange(len(racks))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")

# --- bars ---
bars30  = ax.bar(x - width/2, q30,  width, label="Qwen3-30B",      color="#c2185b")
bars235 = ax.bar(x + width/2, q235, width, label="Qwen3-235B FP8", color="#2979ff")

# axes labels
ax.set_xticks(x)
ax.set_xticklabels(racks)
ax.set_xlabel("Racks Monitored")
ax.set_ylabel("Max LLM Throughput (Tokens/sec)")

# legend (bigger key)
leg = ax.legend(loc="upper left", fontsize=14, frameon=True)
leg.get_frame().set_alpha(0.9)

# grid: horizontal only, behind bars
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="-", alpha=0.15)
ax.xaxis.grid(False)

# value labels
offset = 0.01 * q235.max()
for bar, val in zip(bars30, q30):
    ax.text(bar.get_x() + bar.get_width()/2, val + offset,
            f"{val:.0f}", ha="center", va="bottom", fontsize=11)

for bar, val in zip(bars235, q235):
    ax.text(bar.get_x() + bar.get_width()/2, val + offset,
            f"{val:.0f}", ha="center", va="bottom", fontsize=11)

# ---------- % labels for ALL rack points ----------
# percent change of FP8 vs non-FP8
pct = np.round((q235 - q30) / q30 * 100).astype(int)
# -> array([-9, 33, 54, 139])

for i, p in enumerate(pct):
    x_pos = x[i] + width/2
    is_last = (i == 3)

    # text & sign
    pct_text  = f"{abs(p)}%"
    sign_text = "Higher" if p >= 0 else "Lower"

    if is_last:
        # inside the 200-rack blue bar
        color = "white"
        y_pct  = q235[i] * 0.62
        y_word = q235[i] * 0.52
        va_pct  = "center"
        va_word = "center"
    else:
        # above the blue bar
        color = "black"
        y_pct  = q235[i] + 0.18 * q235.max()
        y_word = y_pct - 300
        va_pct  = "bottom"
        va_word = "top"

    ax.text(x_pos, y_pct, pct_text,
            ha="center", va=va_pct,
            fontsize=22, fontweight="bold", color=color)
    ax.text(x_pos, y_word, sign_text,
            ha="center", va=va_word,
            fontsize=14, fontweight="bold", color=color)

# leave room at top for titles
fig.subplots_adjust(top=0.78)

# --- title block (same style as telemetry chart) ---
fig.suptitle(
    "LLM Throughput With and Without FP8 Optimization",
    fontsize=26,
    fontweight="bold",
    y=0.97,
)

fig.text(
    0.5, 0.92,
    "Measuring Qwen Model Performance on Supermicro Servers",
    ha="center", va="top",
    fontsize=18, fontweight="bold", color="black",
)

fig.text(
    0.5, 0.87,
    "LLM Throughput for Various Rack and Server Counts",
    ha="center", va="top",
    fontsize=16, fontweight="bold", color="gray",
)

fig.text(
    0.5, 0.83,
    "(Higher is Better)",
    ha="center", va="top",
    fontsize=14, fontweight="normal", color="gray",
)

plt.savefig("llm_throughput_final_with_all_pcts.png", dpi=300, bbox_inches="tight")
plt.close(fig)