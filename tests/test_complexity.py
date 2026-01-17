"""
Comprehensive Unit Tests for Complexity Verification Utility

These tests achieve 100% code coverage for src/utils/complexity.py
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.complexity import (
    FLOPCount,
    count_matrix_vector_multiply,
    count_outer_product,
    count_vector_subtraction,
    count_scalar_matrix_multiply,
    count_matrix_addition,
    count_flops,
    summarize_flops,
    verify_complexity_claims
)


class TestFLOPCount:
    """Tests for FLOPCount dataclass."""
    
    def test_flop_count_creation(self):
        """Test FLOPCount creation."""
        flop = FLOPCount(
            multiplications=100,
            additions=50,
            total=150,
            description="Test operation"
        )
        
        assert flop.multiplications == 100
        assert flop.additions == 50
        assert flop.total == 150
        assert flop.description == "Test operation"
    
    def test_flop_count_complexity(self):
        """Test complexity property."""
        flop = FLOPCount(
            multiplications=100,
            additions=50,
            total=150,
            description="Test"
        )
        
        assert flop.complexity == "O(150)"


class TestCountMatrixVectorMultiply:
    """Tests for count_matrix_vector_multiply."""
    
    def test_square_matrix(self):
        """Test with square matrix."""
        result = count_matrix_vector_multiply(64, 64)
        
        assert result.multiplications == 64 * 64
        assert result.additions == 64 * 63
        assert result.total == 64 * 64 + 64 * 63
    
    def test_rectangular_matrix(self):
        """Test with rectangular matrix."""
        result = count_matrix_vector_multiply(32, 64)
        
        assert result.multiplications == 32 * 64
        assert result.additions == 32 * 63


class TestCountOuterProduct:
    """Tests for count_outer_product."""
    
    def test_outer_product(self):
        """Test outer product counting."""
        result = count_outer_product(64, 32)
        
        assert result.multiplications == 64 * 32
        assert result.additions == 0
        assert result.total == 64 * 32


class TestCountVectorSubtraction:
    """Tests for count_vector_subtraction."""
    
    def test_vector_subtraction(self):
        """Test vector subtraction counting."""
        result = count_vector_subtraction(64)
        
        assert result.multiplications == 0
        assert result.additions == 64
        assert result.total == 64


class TestCountScalarMatrixMultiply:
    """Tests for count_scalar_matrix_multiply."""
    
    def test_scalar_matrix_multiply(self):
        """Test scalar-matrix multiplication."""
        result = count_scalar_matrix_multiply(64, 32)
        
        assert result.multiplications == 64 * 32
        assert result.additions == 0
        assert result.total == 64 * 32


class TestCountMatrixAddition:
    """Tests for count_matrix_addition."""
    
    def test_matrix_addition(self):
        """Test matrix addition counting."""
        result = count_matrix_addition(64, 32)
        
        assert result.multiplications == 0
        assert result.additions == 64 * 32
        assert result.total == 64 * 32


class TestCountFlops:
    """Tests for count_flops function."""
    
    def test_theta_mn(self):
        """Test FLOP counting for theta_mn."""
        d = 64
        flops = count_flops('theta_mn', d=d)
        
        assert 'encode' in flops
        assert 'predict' in flops
        assert 'error' in flops
        assert 'gradient' in flops
        assert 'update_scale' in flops
        assert 'update_add' in flops
        assert 'output' in flops
        
        # Verify encode is d² FLOPs
        assert flops['encode'].total == d * d + d * (d - 1)
    
    def test_theta_mn_lr(self):
        """Test FLOP counting for theta_mn_lr."""
        d = 64
        r = 16
        flops = count_flops('theta_mn_lr', d=d, r=r)
        
        assert 'encode' in flops
        assert 'compress' in flops
        assert 'predict' in flops
        assert 'error' in flops
        assert 'U_gradient' in flops
        assert 'V_gradient' in flops
        assert 'output' in flops
        
        # Verify compress is rd FLOPs
        assert flops['compress'].total == r * d + r * (d - 1)
    
    def test_theta_mn_lr_missing_rank(self):
        """Test that missing rank raises error."""
        with pytest.raises(ValueError, match="Must specify rank"):
            count_flops('theta_mn_lr', d=64)
    
    def test_transformer(self):
        """Test FLOP counting for transformer."""
        d = 64
        n = 100
        flops = count_flops('transformer', d=d, n=n)
        
        assert 'query' in flops
        assert 'key' in flops
        assert 'value' in flops
        assert 'attention_scores' in flops
        assert 'softmax' in flops
        assert 'attention_output' in flops
        assert 'output' in flops
    
    def test_transformer_missing_n(self):
        """Test that missing n raises error."""
        with pytest.raises(ValueError, match="Must specify sequence position"):
            count_flops('transformer', d=64)
    
    def test_unknown_architecture(self):
        """Test that unknown architecture raises error."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            count_flops('unknown', d=64)


class TestSummarizeFlops:
    """Tests for summarize_flops function."""
    
    def test_summarize(self):
        """Test FLOP summarization."""
        flops = count_flops('theta_mn', d=64)
        summary = summarize_flops(flops)
        
        assert 'total' in summary
        assert 'multiplications' in summary
        assert 'additions' in summary
        assert 'operations' in summary
        
        # Total should be sum of all operation totals
        expected_total = sum(f.total for f in flops.values())
        assert summary['total'] == expected_total


class TestVerifyComplexityClaims:
    """Tests for verify_complexity_claims function."""
    
    def test_verify_runs(self, capsys):
        """Test that verify_complexity_claims runs without error."""
        verify_complexity_claims(d=64, r=16, n=128)
        
        captured = capsys.readouterr()
        assert "THEOREM 3 VERIFICATION" in captured.out
        assert "θMN (Full Rank)" in captured.out
        assert "θMN(r=" in captured.out
        assert "Transformer" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=term-missing"])
