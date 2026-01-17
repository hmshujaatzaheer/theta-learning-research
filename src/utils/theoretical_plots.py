"""
Theoretical Complexity Plots

This module generates the theoretical complexity comparison plots shown in the proposal.

STATUS: ✅ IMPLEMENTED
These plots are THEORETICAL based on mathematical complexity analysis.
They are NOT empirical measurements.

IMPORTANT DISCLAIMER:
- These plots show O(nd) vs O(d²) vs O(rd) based on operation counts
- Actual wall-clock time depends on memory bandwidth, parallelization, etc.
- Empirical validation is required (Phase 2 of research plan)

PROPOSAL CORRESPONDENCE:
- Figure 3: Complexity comparison
- Figure 5: Theoretical speedup
- Table 12: Speedup ratios
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
import warnings


# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colors matching proposal
COLORS = {
    'transformer': '#B7352D',  # ETH Red
    'theta_mn': '#0066B3',     # ETH Blue
    'theta_mn_lr': '#629D3B',  # ETH Green
    'mamba': '#F39100',        # ETH Orange
}


def compute_flops_per_token(
    architecture: str,
    n: int,
    d: int = 4096,
    r: int = 512,
    s: int = 16  # SSM state size
) -> float:
    """
    Compute theoretical FLOPs per token for different architectures.
    
    Based on complexity analysis in the proposal:
    - Transformer: O(nd) per token (attending to all previous tokens)
    - θMN: O(d²) per token (fixed matrix operations)
    - θMN(r): O(rd) per token (low-rank operations)
    - Mamba/SSM: O(ds) per token (state update)
    
    Args:
        architecture: One of 'transformer', 'theta_mn', 'theta_mn_lr', 'mamba'
        n: Current sequence position (context length so far)
        d: Model dimension
        r: Rank for low-rank variant
        s: State size for SSM
        
    Returns:
        Approximate FLOP count
    """
    if architecture == 'transformer':
        # Attention: Q·K^T (n·d) + softmax (n) + attn·V (n·d) ≈ 2nd + n
        # Per head, summed over heads (assuming d_head * n_heads = d)
        return 2 * n * d + n
    
    elif architecture == 'theta_mn':
        # encode (d²) + predict (d²) + gradient (d²) + update (d²) + output (d²)
        return 5 * d * d
    
    elif architecture == 'theta_mn_lr':
        # encode (d²) + compress (rd) + predict (dr) + updates (4rd) + output (d²)
        # Core memory operations: 6rd
        return 2 * d * d + 6 * r * d
    
    elif architecture == 'mamba':
        # SSM update: A·h (s²) + B·x (sd) + C·h (sd) ≈ s² + 2sd
        # Typically s << d, so O(sd)
        return s * s + 2 * s * d
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def generate_complexity_comparison_plot(
    save_path: Optional[Path] = None,
    d: int = 4096,
    r: int = 512,
    max_context: int = 200000
) -> plt.Figure:
    """
    Generate Figure 3: Complexity comparison across context lengths.
    
    This shows how per-token cost scales with context length.
    
    THEORETICAL - based on O() analysis, not empirical measurement.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Context lengths (log scale)
    context_lengths = np.logspace(3, np.log10(max_context), 100).astype(int)
    
    # Compute costs
    transformer_costs = [compute_flops_per_token('transformer', n, d) for n in context_lengths]
    theta_mn_costs = [compute_flops_per_token('theta_mn', n, d) for n in context_lengths]
    theta_mn_lr_costs = [compute_flops_per_token('theta_mn_lr', n, d, r) for n in context_lengths]
    mamba_costs = [compute_flops_per_token('mamba', n, d) for n in context_lengths]
    
    # Normalize to d=4096 baseline
    baseline = d  # Normalize so Transformer at n=d is 1
    
    # Plot
    ax.loglog(context_lengths, np.array(transformer_costs)/baseline, 
              color=COLORS['transformer'], linewidth=2.5, label='Transformer $O(nd)$')
    ax.loglog(context_lengths, np.array(theta_mn_costs)/baseline, 
              color=COLORS['theta_mn'], linewidth=2.5, label='θMN $O(d^2)$')
    ax.loglog(context_lengths, np.array(theta_mn_lr_costs)/baseline, 
              color=COLORS['theta_mn_lr'], linewidth=2.5, label=f'θMN(r={r}) $O(rd)$')
    ax.loglog(context_lengths, np.array(mamba_costs)/baseline, 
              color=COLORS['mamba'], linewidth=2.5, linestyle='--', label='Mamba $O(sd)$')
    
    # Mark crossover point where θMN becomes faster
    crossover_n = d
    ax.axvline(x=crossover_n, color='gray', linestyle=':', alpha=0.7)
    ax.annotate(f'n = d = {d}', xy=(crossover_n, 10), fontsize=10, 
                ha='center', color='gray')
    
    # Labels
    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Relative Computation Cost (normalized)')
    ax.set_title('THEORETICAL Complexity Comparison\n(Based on O() analysis, not empirical measurement)',
                 fontsize=13)
    
    # Format x-axis
    ax.set_xticks([1000, 4096, 16000, 64000, 128000])
    ax.set_xticklabels(['1K', '4K', '16K', '64K', '128K'])
    
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add disclaimer
    ax.text(0.98, 0.02, 
            'THEORETICAL: Empirical validation required (Section 9, Phase 2)',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            style='italic', color='gray')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
        
    return fig


def generate_speedup_plot(
    save_path: Optional[Path] = None,
    d: int = 4096,
    ranks: List[int] = [256, 512, 1024]
) -> plt.Figure:
    """
    Generate Figure 5: Theoretical speedup over Transformer.
    
    Shows speedup ratio n/r for different rank values.
    
    THEORETICAL - actual speedup depends on implementation.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    context_lengths = np.logspace(3, 6, 100).astype(int)  # 1K to 1M
    
    # θMN full rank (becomes faster when n > d)
    theta_speedup = [max(1, n / d) for n in context_lengths]
    ax.loglog(context_lengths, theta_speedup, 
              color=COLORS['theta_mn'], linewidth=2.5, 
              label=f'θMN vs Transformer: $n/d$')
    
    # θMN(r) for different ranks
    for r in ranks:
        speedup = [max(1, n / r) for n in context_lengths]
        alpha = 0.5 + 0.5 * (ranks.index(r) / len(ranks))
        ax.loglog(context_lengths, speedup, 
                  color=COLORS['theta_mn_lr'], linewidth=2, alpha=alpha,
                  label=f'θMN(r={r}): $n/r$')
    
    # Horizontal line at 1x (no speedup)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    # Labels
    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Theoretical Speedup (×)')
    ax.set_title('THEORETICAL Speedup over Transformer\n(Based on FLOP ratio, not wall-clock measurement)',
                 fontsize=13)
    
    # Format axes
    ax.set_xticks([1000, 10000, 100000, 1000000])
    ax.set_xticklabels(['1K', '10K', '100K', '1M'])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(['1×', '10×', '100×', '1000×'])
    
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add specific speedup annotations
    for n, label in [(128000, '128K'), (1000000, '1M')]:
        for r in [512]:
            speedup = n / r
            ax.annotate(f'{speedup:.0f}×', 
                        xy=(n, speedup), 
                        xytext=(n*1.5, speedup),
                        fontsize=9, ha='left')
    
    # Disclaimer
    ax.text(0.98, 0.02, 
            'THEORETICAL: Wall-clock speedup requires empirical validation',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            style='italic', color='gray')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
        
    return fig


def generate_memory_comparison_plot(
    save_path: Optional[Path] = None,
    d: int = 4096,
    r: int = 512
) -> plt.Figure:
    """
    Generate memory footprint comparison.
    
    - Transformer KV-cache: O(n) - grows with context
    - θMN: O(d²) - constant
    - θMN(r): O(rd) - constant, smaller
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    context_lengths = np.logspace(3, 6, 100).astype(int)
    
    # Memory in bytes (assuming float16)
    bytes_per_param = 2
    
    # Transformer KV-cache: 2 * n * d (K and V) per layer
    # Assuming 32 layers
    n_layers = 32
    transformer_memory = [2 * n * d * n_layers * bytes_per_param / 1e9 
                         for n in context_lengths]
    
    # θMN: d² parameters
    theta_memory = d * d * bytes_per_param / 1e9
    
    # θMN(r): 2dr parameters (U and V)
    theta_lr_memory = 2 * d * r * bytes_per_param / 1e9
    
    # Plot
    ax.semilogy(context_lengths, transformer_memory, 
                color=COLORS['transformer'], linewidth=2.5, 
                label=f'Transformer KV-cache $O(n)$')
    ax.axhline(y=theta_memory, color=COLORS['theta_mn'], linewidth=2.5,
               label=f'θMN $O(d^2)$ = {theta_memory:.2f} GB')
    ax.axhline(y=theta_lr_memory, color=COLORS['theta_mn_lr'], linewidth=2.5,
               label=f'θMN(r={r}) $O(rd)$ = {theta_lr_memory:.3f} GB')
    
    # Mark OOM region (80GB A100)
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5)
    ax.fill_between(context_lengths, 80, 1000, alpha=0.1, color='red')
    ax.text(2000, 100, 'OOM (80GB GPU)', fontsize=10, color='red')
    
    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Memory Footprint (GB)')
    ax.set_title('Memory Footprint Comparison\n(KV-cache grows linearly; θMN is constant)',
                 fontsize=13)
    
    ax.set_xscale('log')
    ax.set_xticks([1000, 10000, 100000, 1000000])
    ax.set_xticklabels(['1K', '10K', '100K', '1M'])
    
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.001, 500)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
        
    return fig


def generate_all_theoretical_plots(output_dir: Path):
    """Generate all theoretical plots for the proposal."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Theoretical Plots")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: These are THEORETICAL plots based on O() analysis.")
    print("    Empirical validation is required (Phase 2 of research plan).\n")
    
    # Figure 3: Complexity comparison
    print("Generating complexity comparison plot...")
    generate_complexity_comparison_plot(
        save_path=output_dir / 'complexity_comparison.pdf'
    )
    generate_complexity_comparison_plot(
        save_path=output_dir / 'complexity_comparison.png'
    )
    
    # Figure 5: Speedup
    print("Generating speedup plot...")
    generate_speedup_plot(
        save_path=output_dir / 'speedup_theoretical.pdf'
    )
    generate_speedup_plot(
        save_path=output_dir / 'speedup_theoretical.png'
    )
    
    # Memory comparison
    print("Generating memory comparison plot...")
    generate_memory_comparison_plot(
        save_path=output_dir / 'memory_comparison.pdf'
    )
    generate_memory_comparison_plot(
        save_path=output_dir / 'memory_comparison.png'
    )
    
    print("\n" + "=" * 60)
    print("✅ All theoretical plots generated")
    print(f"   Output directory: {output_dir}")
    print("=" * 60)
    
    # Create a summary file
    summary = """# Theoretical Plots Summary

## Generated Files

1. `complexity_comparison.pdf/png` - Figure 3 in proposal
   - Shows O(nd) vs O(d²) vs O(rd) scaling
   - THEORETICAL based on operation counts

2. `speedup_theoretical.pdf/png` - Figure 5 in proposal  
   - Shows theoretical speedup ratio n/r
   - THEORETICAL based on FLOP ratio

3. `memory_comparison.pdf/png` - Memory footprint comparison
   - Shows KV-cache O(n) vs θMN O(d²) vs θMN(r) O(rd)
   - THEORETICAL based on parameter counts

## Important Disclaimer

These plots are based on **theoretical complexity analysis**, not empirical
measurements. Actual performance depends on:

- Memory bandwidth (often the bottleneck on GPUs)
- Parallelization efficiency  
- Implementation optimization (e.g., FlashAttention)
- Hardware architecture

**Empirical validation is required** as specified in Phase 2 of the research plan.

## Reproducing

```bash
python src/utils/theoretical_plots.py
```

Or from the scripts directory:

```bash
python scripts/generate_theoretical_plots.py
```
"""
    
    with open(output_dir / 'README.md', 'w') as f:
        f.write(summary)


if __name__ == "__main__":
    # Default output to figures directory
    import sys
    
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        # Assume we're in src/utils, go up to repo root
        output_dir = Path(__file__).parent.parent.parent / 'figures'
    
    generate_all_theoretical_plots(output_dir)
    plt.show()
