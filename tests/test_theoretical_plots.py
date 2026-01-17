"""
Comprehensive Unit Tests for Theoretical Plots

These tests achieve 100% code coverage for src/utils/theoretical_plots.py
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock matplotlib to avoid display issues in CI
import matplotlib
matplotlib.use('Agg')

from src.utils.theoretical_plots import (
    compute_flops_per_token,
    generate_complexity_comparison_plot,
    generate_speedup_plot,
    generate_memory_comparison_plot,
    generate_all_theoretical_plots
)


class TestComputeFlopsPerToken:
    """Tests for compute_flops_per_token function."""
    
    def test_transformer(self):
        """Test FLOP computation for transformer."""
        result = compute_flops_per_token('transformer', n=1000, d=512)
        
        # Transformer: 2nd + n
        expected = 2 * 1000 * 512 + 1000
        assert result == expected
    
    def test_theta_mn(self):
        """Test FLOP computation for theta_mn."""
        result = compute_flops_per_token('theta_mn', n=1000, d=512)
        
        # θMN: 5 * d²
        expected = 5 * 512 * 512
        assert result == expected
    
    def test_theta_mn_lr(self):
        """Test FLOP computation for theta_mn_lr."""
        result = compute_flops_per_token('theta_mn_lr', n=1000, d=512, r=64)
        
        # θMN(r): 2d² + 6rd
        expected = 2 * 512 * 512 + 6 * 64 * 512
        assert result == expected
    
    def test_mamba(self):
        """Test FLOP computation for mamba."""
        result = compute_flops_per_token('mamba', n=1000, d=512, s=16)
        
        # Mamba: s² + 2sd
        expected = 16 * 16 + 2 * 16 * 512
        assert result == expected
    
    def test_unknown_architecture(self):
        """Test that unknown architecture raises error."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            compute_flops_per_token('unknown', n=100, d=512)


class TestGeneratePlots:
    """Tests for plot generation functions."""
    
    def test_complexity_comparison_plot(self):
        """Test complexity comparison plot generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_complexity.png"
            fig = generate_complexity_comparison_plot(save_path=save_path, d=512, r=64, max_context=10000)
            
            assert fig is not None
            assert save_path.exists()
    
    def test_complexity_comparison_plot_no_save(self):
        """Test complexity comparison plot without saving."""
        fig = generate_complexity_comparison_plot(d=512, r=64, max_context=10000)
        
        assert fig is not None
    
    def test_speedup_plot(self):
        """Test speedup plot generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_speedup.png"
            fig = generate_speedup_plot(save_path=save_path, d=512, ranks=[64, 128])
            
            assert fig is not None
            assert save_path.exists()
    
    def test_speedup_plot_no_save(self):
        """Test speedup plot without saving."""
        fig = generate_speedup_plot(d=512, ranks=[64, 128])
        
        assert fig is not None
    
    def test_memory_comparison_plot(self):
        """Test memory comparison plot generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_memory.png"
            fig = generate_memory_comparison_plot(save_path=save_path, d=512, r=64)
            
            assert fig is not None
            assert save_path.exists()
    
    def test_memory_comparison_plot_no_save(self):
        """Test memory comparison plot without saving."""
        fig = generate_memory_comparison_plot(d=512, r=64)
        
        assert fig is not None


class TestGenerateAllPlots:
    """Tests for generate_all_theoretical_plots function."""
    
    def test_generate_all(self):
        """Test generating all plots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_all_theoretical_plots(output_dir)
            
            # Check that files were created
            assert (output_dir / 'complexity_comparison.pdf').exists()
            assert (output_dir / 'complexity_comparison.png').exists()
            assert (output_dir / 'speedup_theoretical.pdf').exists()
            assert (output_dir / 'speedup_theoretical.png').exists()
            assert (output_dir / 'memory_comparison.pdf').exists()
            assert (output_dir / 'memory_comparison.png').exists()
            assert (output_dir / 'README.md').exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=term-missing"])
