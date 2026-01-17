#!/usr/bin/env python3
"""
Generate Theoretical Plots

This script generates all theoretical plots for the proposal.

IMPORTANT: These plots are THEORETICAL based on O() analysis.
They are NOT empirical measurements.

Usage:
    python scripts/generate_theoretical_plots.py [output_dir]
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.theoretical_plots import generate_all_theoretical_plots


def main():
    # Determine output directory
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path(__file__).parent.parent / 'figures'
    
    print("Generating theoretical plots...")
    print(f"Output directory: {output_dir}")
    print()
    
    generate_all_theoretical_plots(output_dir)


if __name__ == "__main__":
    main()
