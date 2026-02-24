import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Match reference style exactly: blue = primary (MI355X/MXFP4), red = comparison
BLUE = '#2563EB'
RED = '#EF4444'
BLACK = '#000000'
DARK_GRAY = '#333333'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.facecolor': '#FFFFFF',
    'figure.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CCCCCC',
    'axes.grid': True,
    'grid.color': '#E8E8E8',
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.axisbelow': True,
})

outdir = '.'

def fmt_val(v):
    """Format value for bar label."""
    if isinstance(v, int) or (isinstance(v, float) and v == int(v) and v >= 10):
        return f'{int(v):,}'
    elif isinstance(v, float) and v >= 10:
        return f'{v:,.0f}'
    elif isinstance(v, float):
        return f'{v:.2f}'
    return str(v)

def styled_chart(ax, x, vals1, vals2, bars1, bars2, label1, label2, suffix=''):
    """Add value labels inside bars and ratio above, matching reference style."""
    for i in range(len(vals1)):
        v1, v2 = vals1[i], vals2[i]
        b1, b2 = bars1[i], bars2[i]

        # Value inside bar 1 (blue) - white text
        lbl1 = fmt_val(v1) + suffix
        y1 = b1.get_height()
        ax.text(b1.get_x() + b1.get_width()/2, y1 * 0.45,
                lbl1, ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        # Value inside bar 2 (red) - white text
        lbl2 = fmt_val(v2) + suffix
        y2 = b2.get_height()
        ax.text(b2.get_x() + b2.get_width()/2, y2 * 0.45,
                lbl2, ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        # Ratio above the pair in bold black
        if v2 > 0:
            ratio = v1 / v2
            top = max(y1, y2)
            mid_x = (b1.get_x() + b1.get_width()/2 + b2.get_x() + b2.get_width()/2) / 2
            ax.text(mid_x, top * 1.05,
                    f'{ratio:.1f}x', ha='center', va='bottom',
                    fontsize=13, color=BLACK, fontweight='bold')

# ================================================================
# CHART 1: MXFP4 vs FP8 Throughput Scaling (128/2048)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

conc = [32, 64, 128, 256, 512, 1024, 2048, 8192]
mxfp4 = [1598, 2070, 3367, 5641, 9621, 15318, 23824, 43916]
fp8 =   [1133, 1428, 2468, 4007, 7385, 12588, 18739, 20540]

x = np.arange(len(conc))
w = 0.35
bars1 = ax.bar(x - w/2, mxfp4, w, label='MXFP4', color=BLUE, edgecolor='white', linewidth=0.3)
bars2 = ax.bar(x + w/2, fp8, w, label='FP8', color=RED, edgecolor='white', linewidth=0.3)

ax.set_xlabel('Concurrent Requests', fontsize=12, color=DARK_GRAY, fontweight='bold')
ax.set_ylabel('Output Throughput (tokens/s)', fontsize=12, color=DARK_GRAY, fontweight='bold')
ax.set_title('Throughput Scaling: MXFP4 vs. FP8\n(128 / 2048 Workload, MI355X)', fontsize=14, color=BLACK, fontweight='bold', pad=14)
ax.set_xticks(x)
ax.set_xticklabels([f'{c:,}' for c in conc], fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.legend(loc='upper left', framealpha=0.95, fontsize=11, edgecolor='#CCCCCC')
ax.set_ylim(0, max(mxfp4) * 1.18)

styled_chart(ax, x, mxfp4, fp8, bars1, bars2, 'MXFP4', 'FP8')

plt.tight_layout()
fig.savefig(f'{outdir}/chart1_throughput_scaling.png', dpi=200, bbox_inches='tight')
plt.close()
print("Chart 1 done")

# ================================================================
# CHART 2: Workload Profiles at 512 Concurrency
# ================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))

workloads = ['128 / 128', '128 / 2048', '2048 / 128', '2048 / 2048']
mxfp4_512 = [10503, 9621, 4862, 7000]
fp8_512 = [8359, 7385, 4191, 5838]

x = np.arange(len(workloads))
w = 0.35
bars1 = ax.bar(x - w/2, mxfp4_512, w, label='MXFP4', color=BLUE, edgecolor='white', linewidth=0.3)
bars2 = ax.bar(x + w/2, fp8_512, w, label='FP8', color=RED, edgecolor='white', linewidth=0.3)

ax.set_xlabel('Input / Output Token Length', fontsize=12, color=DARK_GRAY, fontweight='bold')
ax.set_ylabel('Output Throughput (tokens/s)', fontsize=12, color=DARK_GRAY, fontweight='bold')
ax.set_title('Throughput by Workload Profile\n(512 Concurrent Requests, MI355X)', fontsize=14, color=BLACK, fontweight='bold', pad=14)
ax.set_xticks(x)
ax.set_xticklabels(workloads, fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.legend(loc='upper right', framealpha=0.95, fontsize=11, edgecolor='#CCCCCC')
ax.set_ylim(0, max(mxfp4_512) * 1.18)

styled_chart(ax, x, mxfp4_512, fp8_512, bars1, bars2, 'MXFP4', 'FP8')

plt.tight_layout()
fig.savefig(f'{outdir}/chart2_workload_profiles.png', dpi=200, bbox_inches='tight')
plt.close()
print("Chart 2 done")

# ================================================================
# CHART 3: MI355X MXFP4 vs MI300X FP8 (2048/2048)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

conc_gen = [128, 256, 512, 1024, 2048, 4096]
mi355x = [3125, 4513, 7000, 9942, 14032, 16647]
mi300x = [1680, 2453, 3204, 4002, 4096, 4083]

x = np.arange(len(conc_gen))
w = 0.35
bars1 = ax.bar(x - w/2, mi355x, w, label='MI355X + MXFP4', color=BLUE, edgecolor='white', linewidth=0.3)
bars2 = ax.bar(x + w/2, mi300x, w, label='MI300X + FP8', color=RED, edgecolor='white', linewidth=0.3)

ax.set_xlabel('Concurrent Requests', fontsize=12, color=DARK_GRAY, fontweight='bold')
ax.set_ylabel('Output Throughput (tokens/s)', fontsize=12, color=DARK_GRAY, fontweight='bold')
ax.set_title('Generational Gain: MI355X MXFP4 vs. MI300X FP8\n(2048 / 2048 Workload)', fontsize=14, color=BLACK, fontweight='bold', pad=14)
ax.set_xticks(x)
ax.set_xticklabels([f'{c:,}' for c in conc_gen], fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.legend(loc='upper left', framealpha=0.95, fontsize=11, edgecolor='#CCCCCC')
ax.set_ylim(0, max(mi355x) * 1.18)

styled_chart(ax, x, mi355x, mi300x, bars1, bars2, 'MI355X', 'MI300X')

plt.tight_layout()
fig.savefig(f'{outdir}/chart3_generational.png', dpi=200, bbox_inches='tight')
plt.close()
print("Chart 3 done")

# ================================================================
# CHART 4: Power-Normalized Efficiency
# ================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))

workloads_pwr = ['128 / 128', '128 / 2048', '2048 / 128', '2048 / 2048']
eff_355x = [4.46, 4.12, 2.07, 2.97]
eff_300x = [0.66, 0.77, 0.21, 0.44]

x = np.arange(len(workloads_pwr))
w = 0.35
bars1 = ax.bar(x - w/2, eff_355x, w, label='MI355X + MXFP4', color=BLUE, edgecolor='white', linewidth=0.3)
bars2 = ax.bar(x + w/2, eff_300x, w, label='MI300X + FP8', color=RED, edgecolor='white', linewidth=0.3)

ax.set_xlabel('Input / Output Token Length', fontsize=12, color=DARK_GRAY, fontweight='bold')
ax.set_ylabel('Tokens / Second / Watt', fontsize=12, color=DARK_GRAY, fontweight='bold')
ax.set_title('Power-Normalized Throughput\n(512 Concurrent Requests)', fontsize=14, color=BLACK, fontweight='bold', pad=14)
ax.set_xticks(x)
ax.set_xticklabels(workloads_pwr, fontsize=11)
ax.legend(loc='upper right', framealpha=0.95, fontsize=11, edgecolor='#CCCCCC')
ax.set_ylim(0, max(eff_355x) * 1.18)

styled_chart(ax, x, eff_355x, eff_300x, bars1, bars2, 'MI355X', 'MI300X')

plt.tight_layout()
fig.savefig(f'{outdir}/chart4_power_efficiency.png', dpi=200, bbox_inches='tight')
plt.close()
print("Chart 4 done")

# ================================================================
# CHART 5: TTFT at High Concurrency (2048/2048)
# ================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))

conc_ttft = [128, 512, 1024]
ttft_mxfp4 = [0.451, 0.763, 1.169]
ttft_fp8 = [0.483, 1.069, 24.4]

x = np.arange(len(conc_ttft))
w = 0.35
bars1 = ax.bar(x - w/2, ttft_mxfp4, w, label='MXFP4', color=BLUE, edgecolor='white', linewidth=0.3)
bars2 = ax.bar(x + w/2, ttft_fp8, w, label='FP8', color=RED, edgecolor='white', linewidth=0.3)

ax.set_xlabel('Concurrent Requests', fontsize=12, color=DARK_GRAY, fontweight='bold')
ax.set_ylabel('Time to First Token (seconds)', fontsize=12, color=DARK_GRAY, fontweight='bold')
ax.set_title('Time to First Token: MXFP4 vs. FP8\n(2048 / 2048 Workload, MI355X)', fontsize=14, color=BLACK, fontweight='bold', pad=14)
ax.set_xticks(x)
ax.set_xticklabels([f'{c:,}' for c in conc_ttft], fontsize=11)
ax.legend(loc='upper left', framealpha=0.95, fontsize=11, edgecolor='#CCCCCC')
ax.set_ylim(0, max(ttft_fp8) * 1.18)

# For TTFT, lower is better, so show ratio as FP8/MXFP4 improvement
for i in range(len(conc_ttft)):
    v1, v2 = ttft_mxfp4[i], ttft_fp8[i]
    b1, b2 = bars1[i], bars2[i]

    ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() * 0.45,
            f'{v1:.1f}s', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() * 0.45,
            f'{v2:.1f}s', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ratio = v2 / v1  # how many times faster MXFP4 is
    top = max(v1, v2)
    mid_x = (b1.get_x() + b1.get_width()/2 + b2.get_x() + b2.get_width()/2) / 2
    ax.text(mid_x, top * 1.05,
            f'{ratio:.1f}x', ha='center', va='bottom',
            fontsize=13, color=BLACK, fontweight='bold')

plt.tight_layout()
fig.savefig(f'{outdir}/chart5_ttft.png', dpi=200, bbox_inches='tight')
plt.close()
print("Chart 5 done")

print("\nAll charts regenerated in reference style.")