import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# Synthetic jitter data (same as before – replace with real data when ready)
# ============================================================
rng = np.random.default_rng(42)
n_samples = 20000

jitter_6767 = rng.lognormal(mean=np.log(1.5), sigma=0.05, size=n_samples)
jitter_6980 = rng.lognormal(mean=np.log(1.5), sigma=0.06, size=n_samples)

core_8592_main = rng.lognormal(mean=np.log(1.0), sigma=0.08, size=int(n_samples * 0.9))
core_8592_tail = rng.lognormal(mean=np.log(2.0), sigma=0.25, size=int(n_samples * 0.1))
jitter_8592 = np.concatenate([core_8592_main, core_8592_tail])

cpus = ["Xeon 8592+", "Xeon 6767P", "Xeon 6980P"]
jitter_data = [jitter_8592, jitter_6767, jitter_6980]

# If you have real samples, overwrite jitter_... arrays above.


# ============================================================
# 1) CLEANER Jitter Density Overlay
# ============================================================
# Idea: zoom into the main jitter band and use fewer bins so curves are smoother.

plt.figure(figsize=(8, 3.5))

# Focus on the interesting region (adjust if your real data differs)
x_min, x_max = 0.8, 2.2
bins = np.linspace(x_min, x_max, 60)

for cpu, data in zip(cpus, jitter_data):
    # Clip to focus region so tails don’t dominate scaling
    clipped = data[(data >= x_min) & (data <= x_max)]
    plt.hist(
        clipped,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label=cpu,
    )

plt.xlabel("Wake-up Latency (µs)")
plt.ylabel("Density (a.u.)")
plt.title("Jitter Density Overlay by CPU SKU")
plt.xlim(x_min, x_max)
plt.margins(x=0.01)
plt.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.7)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("jitter_density_overlay_clean.png", dpi=300)
plt.show()


# ============================================================
# 2) CLEANER Jitter CDF Curve
# ============================================================
# Idea: same zoom, slightly thicker lines, light grid.

plt.figure(figsize=(8, 3.5))

for cpu, data in zip(cpus, jitter_data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, linewidth=1.8, label=cpu)

plt.xlabel("Wake-up Latency (µs)")
plt.ylabel("Cumulative Probability")
plt.title("Jitter CDF by CPU SKU")
plt.xlim(x_min, x_max)
plt.ylim(0.0, 1.01)
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("jitter_cdf_overlay_clean.png", dpi=300)
plt.show()


# ============================================================
# 3) Benchmark Comparison Heatmap (unchanged except labels tightened)
# ============================================================
benchmarks = ["jitter-c", "perf", "SPEC"]
features = [
    "Latency distribution\nshape",
    "p99 / p99.9\nlatencies",
    "Per-core\ngranularity",
    "Real-time\nscheduling",
    "Throughput /\nIPC focus",
]

data = [
    [1, 1, 1, 1, 0],  # jitter-c
    [0, 0, 0, 0, 1],  # perf
    [0, 0, 0, 0, 1],  # SPEC
]

df = pd.DataFrame(data, index=benchmarks, columns=features)

plt.figure(figsize=(7, 3.2))
plt.imshow(df.values, aspect="auto")
plt.xticks(range(len(features)), features, rotation=0, ha="center")
plt.yticks(range(len(benchmarks)), benchmarks)
plt.colorbar(label="Feature Present (1 = Yes, 0 = No)")
plt.title("Benchmark Capability Comparison")
plt.tight_layout()
plt.savefig("benchmark_comparison_heatmap_clean.png", dpi=300)
plt.show()
