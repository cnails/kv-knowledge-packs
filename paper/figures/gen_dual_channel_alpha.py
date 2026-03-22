"""Generate dual-channel alpha trade-off figure for the paper."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
})

# N=200 results
alphas =     [0.0,  0.1,  0.2,  0.5,  0.7,  1.0,  1.5,  2.0]
em =         [72.5, 73.0, 73.0, 72.0, 73.0, 67.0, 55.0, 40.5]
formality =  [0.20, 0.21, 0.21, 0.23, 0.24, 0.25, 0.28, 0.32]
words =      [35,   37,   39,   43,   45,   49,   53,   55]

fig, ax1 = plt.subplots(figsize=(7, 4.5))

# Green zone: alpha <= 0.7
ax1.axvspan(-0.05, 0.75, alpha=0.12, color='#2ecc71', label='Safe zone (α ≤ 0.7)')

# EM on left axis
color_em = '#2c3e50'
ax1.set_xlabel('Steering strength α', fontsize=13)
ax1.set_ylabel('Exact Match (%)', color=color_em, fontsize=13)
line1 = ax1.plot(alphas, em, 'o-', color=color_em, linewidth=2.5, markersize=8,
                  label='EM (accuracy)', zorder=5)
ax1.tick_params(axis='y', labelcolor=color_em)
ax1.set_ylim(30, 80)
ax1.set_xlim(-0.05, 2.1)

# Formality on right axis
ax2 = ax1.twinx()
color_form = '#e74c3c'
ax2.set_ylabel('Formality score', color=color_form, fontsize=13)
line2 = ax2.plot(alphas, formality, 's--', color=color_form, linewidth=2.5, markersize=8,
                  label='Formality (steering)', zorder=5)
ax2.tick_params(axis='y', labelcolor=color_form)
ax2.set_ylim(0.15, 0.38)

# Annotations
ax1.annotate('No accuracy loss\nα ≤ 0.7', xy=(0.35, 73), fontsize=10,
             ha='center', va='bottom', color='#27ae60', fontweight='bold')
ax1.annotate('Accuracy\ndegrades', xy=(1.5, 52), fontsize=10,
             ha='center', va='top', color='#c0392b', fontweight='bold')

# Vertical line at boundary
ax1.axvline(x=0.75, color='#27ae60', linestyle=':', linewidth=1.5, alpha=0.7)

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
# Add green zone to legend
from matplotlib.patches import Patch
legend_elements = lines + [Patch(facecolor='#2ecc71', alpha=0.12, label='Safe zone (α ≤ 0.7)')]
legend_labels = labels + ['Safe zone (α ≤ 0.7)']
ax1.legend(legend_elements, legend_labels, loc='lower left', fontsize=10, framealpha=0.9)

ax1.set_title('Dual-Channel Trade-off: Knowledge + Steering (N=200)', fontsize=14, pad=12)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/cnails/kv-knowledge-packs/paper/figures/dual_channel_alpha.png',
            dpi=300, bbox_inches='tight')
print("Saved dual_channel_alpha.png")
