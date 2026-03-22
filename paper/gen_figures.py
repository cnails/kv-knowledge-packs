"""Generate all paper figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# NeurIPS-friendly style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Colors
C_KV = '#2171b5'
C_RAG = '#cb181d'
C_QWEN = '#2171b5'
C_LLAMA = '#e6550d'


# ================================================================
# Figure 1: Accumulation scaling — token cost
# ================================================================
def fig_accumulation():
    searches = [1, 2, 3, 5]

    # Qwen3-8B
    kv_tok_qwen = [35, 35, 35, 35]
    rag_tok_qwen = [176, 299, 438, 739]

    # Llama-3.1-8B
    kv_tok_llama = [31, 31, 31, 31]
    rag_tok_llama = [188, 305, 437, 724]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)

    for ax, kv, rag, name in [
        (ax1, kv_tok_qwen, rag_tok_qwen, 'Qwen3-8B'),
        (ax2, kv_tok_llama, rag_tok_llama, 'Llama-3.1-8B'),
    ]:
        ax.plot(searches, rag, 's-', color=C_RAG, label='RAG (text in prompt)',
                markersize=7, linewidth=2)
        ax.plot(searches, kv, 'o-', color=C_KV, label='KV injection',
                markersize=7, linewidth=2)
        ax.fill_between(searches, kv, rag, alpha=0.12, color=C_RAG)

        # Annotate savings at 5 searches
        mid_y = (rag[-1] + kv[-1]) / 2
        ax.annotate(f'{rag[-1] - kv[-1]} tokens\nsaved (95%)',
                     xy=(5, mid_y), fontsize=9, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='gray', alpha=0.9))

        ax.set_xlabel('Retrieval steps')
        ax.set_title(name)
        ax.set_xticks(searches)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel('Prompt tokens (knowledge)')

    fig.suptitle('Token Cost: Constant (KV) vs. Linear (RAG)', y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'accumulation_scaling.png'))
    plt.close(fig)
    print('  ✓ accumulation_scaling.png')


# ================================================================
# Figure 2: Dual-channel alpha sweep
# ================================================================
def fig_dual_alpha():
    alphas = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0]
    em =     [84.0, 84.0, 84.0, 84.0, 82.0, 80.0, 74.0, 64.0]
    words =  [34, 36, 37, 50, 63, 83, 148, 191]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.8))
    ax2 = ax1.twinx()

    l1, = ax1.plot(alphas, em, 'o-', color=C_KV, linewidth=2, markersize=6,
                    label='Exact Match (%)')
    l2, = ax2.plot(alphas, words, 's--', color=C_LLAMA, linewidth=2, markersize=6,
                    label='Avg. words')

    # Mark zero-interference zone
    ax1.axvspan(0, 0.5, alpha=0.08, color='green')
    ax1.annotate('zero interference\nzone', xy=(0.25, 66), fontsize=9,
                  ha='center', color='#2d6a2d',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='#2d6a2d', alpha=0.8))

    ax1.set_xlabel('Steering strength (α)')
    ax1.set_ylabel('Exact Match (%)', color=C_KV)
    ax2.set_ylabel('Avg. response words', color=C_LLAMA)
    ax1.set_ylim(55, 90)
    ax2.set_ylim(0, 220)
    ax1.tick_params(axis='y', labelcolor=C_KV)
    ax2.tick_params(axis='y', labelcolor=C_LLAMA)

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='lower left')
    ax1.set_title('Dual-Channel: Knowledge Accuracy vs. Steering Strength')
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'dual_channel_alpha.png'))
    plt.close(fig)
    print('  ✓ dual_channel_alpha.png')


# ================================================================
# Figure 3: Layer-selective steering (grouped bar chart)
# ================================================================
def fig_layer_steering():
    ranges = ['Baseline', 'Early\n(0–33%)', 'Mid\n(33–66%)', 'Late\n(66–100%)', 'All\n(0–100%)']
    qwen =  [0.60, 1.20, 1.93, 0.67, 1.93]
    llama = [1.47, 1.33, 2.27, 1.47, 2.47]

    x = np.arange(len(ranges))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars1 = ax.bar(x - w/2, qwen, w, label='Qwen3-8B (α=2.0)', color=C_QWEN, alpha=0.85)
    bars2 = ax.bar(x + w/2, llama, w, label='Llama-3.1-8B (α=1.5)', color=C_LLAMA, alpha=0.85)

    # Highlight mid layers
    ax.axvspan(x[2] - 0.5, x[2] + 0.5, alpha=0.08, color='green')

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Defensive coding score (0–9)')
    ax.set_title('Layer-Selective Value Steering')
    ax.set_xticks(x)
    ax.set_xticklabels(ranges)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 3.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'layer_steering.png'))
    plt.close(fig)
    print('  ✓ layer_steering.png')


if __name__ == '__main__':
    print('Generating figures...')
    fig_accumulation()
    fig_dual_alpha()
    fig_layer_steering()
    print('Done.')
