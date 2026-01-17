"""
Comprehensive Unit Tests for θ-Memory Network Low-Rank (θMN(r))

These tests achieve 100% code coverage for src/core/theta_mn_lr.py
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.theta_mn_lr import ThetaMemoryNetworkLowRank


class TestThetaMemoryNetworkLowRankInit:
    """Tests for ThetaMemoryNetworkLowRank initialization."""
    
    def test_initialization_default(self):
        """Test model initializes with default parameters."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        assert model.d == d
        assert model.r == r
        assert model.lr == 0.01  # default
        assert model.U.shape == (d, r)
        assert model.V.shape == (r, d)
        assert model.W_e.shape == (d, d)
        assert model.W_o.shape == (d, d)
        assert model.device == torch.device('cpu')
    
    def test_initialization_custom_lr(self):
        """Test model initializes with custom learning rate."""
        d = 32
        r = 8
        lr = 0.05
        model = ThetaMemoryNetworkLowRank(d=d, r=r, lr=lr)
        
        assert model.lr == lr
    
    def test_initialization_custom_init_scale(self):
        """Test model initializes with custom init scale."""
        d = 32
        r = 8
        init_scale = 0.1
        model = ThetaMemoryNetworkLowRank(d=d, r=r, init_scale=init_scale)
        
        assert model.U.std().item() < init_scale * 3
        assert model.V.std().item() < init_scale * 3
    
    def test_initialization_with_device(self):
        """Test model initializes with specified device."""
        d = 32
        r = 8
        device = torch.device('cpu')
        model = ThetaMemoryNetworkLowRank(d=d, r=r, device=device)
        
        assert model.device == device
        assert model.U.device == device
        assert model.V.device == device
    
    def test_initialization_invalid_rank(self):
        """Test that rank > d raises assertion error."""
        d = 32
        r = 64  # r > d is invalid
        
        with pytest.raises(AssertionError):
            ThetaMemoryNetworkLowRank(d=d, r=r)


class TestThetaMemoryNetworkLowRankTheta:
    """Tests for theta property."""
    
    def test_theta_property(self):
        """Test that theta property reconstructs U @ V."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        theta = model.theta
        expected = model.U @ model.V
        
        assert theta.shape == (d, d)
        assert torch.allclose(theta, expected)


class TestThetaMemoryNetworkLowRankEncode:
    """Tests for encode method."""
    
    def test_encode_single(self):
        """Test encode with single input."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        x = torch.randn(d)
        h = model.encode(x)
        
        assert h.shape == (d,)
    
    def test_encode_batched(self):
        """Test encode with batched input."""
        d = 64
        r = 16
        batch_size = 8
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        x = torch.randn(batch_size, d)
        h = model.encode(x)
        
        assert h.shape == (batch_size, d)


class TestThetaMemoryNetworkLowRankCompress:
    """Tests for compress method."""
    
    def test_compress_single(self):
        """Test compress with single input."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        h = torch.randn(d)
        z = model.compress(h)
        
        assert z.shape == (r,)
    
    def test_compress_batched(self):
        """Test compress with batched input."""
        d = 64
        r = 16
        batch_size = 8
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        h = torch.randn(batch_size, d)
        z = model.compress(h)
        
        assert z.shape == (batch_size, r)


class TestThetaMemoryNetworkLowRankPredict:
    """Tests for predict method."""
    
    def test_predict_single(self):
        """Test predict with single input."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        z = torch.randn(r)
        y = model.predict(z)
        
        assert y.shape == (d,)
    
    def test_predict_batched(self):
        """Test predict with batched input."""
        d = 64
        r = 16
        batch_size = 8
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        z = torch.randn(batch_size, r)
        y = model.predict(z)
        
        assert y.shape == (batch_size, d)


class TestThetaMemoryNetworkLowRankComputeUpdates:
    """Tests for compute_updates method."""
    
    def test_compute_updates_single(self):
        """Test compute_updates with single input."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        h = torch.randn(d)
        z = torch.randn(r)
        target = torch.randn(d)
        prediction = torch.randn(d)
        
        U_grad, V_grad = model.compute_updates(h, z, target, prediction)
        
        assert U_grad.shape == (d, r)
        assert V_grad.shape == (r, d)
    
    def test_compute_updates_batched(self):
        """Test compute_updates with batched input."""
        d = 64
        r = 16
        batch_size = 8
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        h = torch.randn(batch_size, d)
        z = torch.randn(batch_size, r)
        target = torch.randn(batch_size, d)
        prediction = torch.randn(batch_size, d)
        
        U_grad, V_grad = model.compute_updates(h, z, target, prediction)
        
        assert U_grad.shape == (d, r)
        assert V_grad.shape == (r, d)


class TestThetaMemoryNetworkLowRankUpdateMemory:
    """Tests for update_memory method."""
    
    def test_update_memory(self):
        """Test that memory is updated correctly."""
        d = 64
        r = 16
        lr = 0.1
        model = ThetaMemoryNetworkLowRank(d=d, r=r, lr=lr)
        
        initial_U = model.U.data.clone()
        initial_V = model.V.data.clone()
        
        U_grad = torch.randn(d, r)
        V_grad = torch.randn(r, d)
        
        model.update_memory(U_grad, V_grad)
        
        expected_U = initial_U + lr * U_grad
        expected_V = initial_V + lr * V_grad
        
        assert torch.allclose(model.U.data, expected_U)
        assert torch.allclose(model.V.data, expected_V)


class TestThetaMemoryNetworkLowRankResetMemory:
    """Tests for reset_memory method."""
    
    def test_reset_memory(self):
        """Test that memory resets to initial state."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r, lr=0.1)
        
        initial_U = model.U.data.clone()
        initial_V = model.V.data.clone()
        
        # Make updates
        for _ in range(10):
            x = torch.randn(d)
            target = torch.randn(d)
            model.forward_and_update(x, target)
        
        # Verify memory changed
        assert not torch.allclose(model.U.data, initial_U)
        
        # Reset
        model.reset_memory()
        
        # Verify reset
        assert torch.allclose(model.U.data, initial_U)
        assert torch.allclose(model.V.data, initial_V)


class TestThetaMemoryNetworkLowRankForward:
    """Tests for forward method."""
    
    def test_forward_single(self):
        """Test forward pass with single input."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        x = torch.randn(d)
        y = model.forward(x)
        
        assert y.shape == (d,)
    
    def test_forward_batched(self):
        """Test forward pass with batched input."""
        d = 64
        r = 16
        batch_size = 8
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        x = torch.randn(batch_size, d)
        y = model.forward(x)
        
        assert y.shape == (batch_size, d)


class TestThetaMemoryNetworkLowRankForwardAndUpdate:
    """Tests for forward_and_update method."""
    
    def test_forward_and_update_single(self):
        """Test forward_and_update with single input."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        x = torch.randn(d)
        target = torch.randn(d)
        
        initial_U = model.U.data.clone()
        y = model.forward_and_update(x, target)
        
        assert y.shape == (d,)
        assert not torch.allclose(model.U.data, initial_U)
    
    def test_forward_and_update_batched(self):
        """Test forward_and_update with batched input."""
        d = 64
        r = 16
        batch_size = 8
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        x = torch.randn(batch_size, d)
        target = torch.randn(batch_size, d)
        
        y = model.forward_and_update(x, target)
        
        assert y.shape == (batch_size, d)


class TestThetaMemoryNetworkLowRankProcessSequence:
    """Tests for process_sequence method."""
    
    def test_process_sequence_2d_with_targets(self):
        """Test sequence processing with 2D input and targets."""
        d = 64
        r = 16
        seq_len = 20
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        sequence = torch.randn(seq_len, d)
        targets = torch.randn(seq_len, d)
        
        outputs = model.process_sequence(sequence, targets)
        
        assert outputs.shape == (seq_len, d)
    
    def test_process_sequence_2d_without_targets(self):
        """Test sequence processing with 2D input without targets."""
        d = 64
        r = 16
        seq_len = 20
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        sequence = torch.randn(seq_len, d)
        
        outputs = model.process_sequence(sequence)
        
        assert outputs.shape == (seq_len, d)
    
    def test_process_sequence_3d_with_targets(self):
        """Test sequence processing with 3D batched input."""
        d = 64
        r = 16
        seq_len = 20
        batch_size = 4
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        sequence = torch.randn(batch_size, seq_len, d)
        targets = torch.randn(batch_size, seq_len, d)
        
        outputs = model.process_sequence(sequence, targets)
        
        assert outputs.shape == (batch_size, seq_len, d)
    
    def test_process_sequence_3d_without_targets(self):
        """Test sequence processing with 3D batched input without targets."""
        d = 64
        r = 16
        seq_len = 20
        batch_size = 4
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        sequence = torch.randn(batch_size, seq_len, d)
        
        outputs = model.process_sequence(sequence)
        
        assert outputs.shape == (batch_size, seq_len, d)
    
    def test_process_sequence_invalid_dim(self):
        """Test that invalid dimensions raise error."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        sequence = torch.randn(d)  # 1D - invalid
        
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            model.process_sequence(sequence)


class TestThetaMemoryNetworkLowRankFlopCount:
    """Tests for count_flops_per_token method."""
    
    def test_count_flops(self):
        """Test FLOP counting."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        flops = model.count_flops_per_token()
        
        assert 'encode' in flops
        assert 'compress' in flops
        assert 'predict' in flops
        assert 'error' in flops
        assert 'core_memory_ops' in flops
        assert 'complexity_core' in flops
        
        # Verify core memory ops is O(rd)
        assert flops['core_memory_ops'] == 7 * d * r


class TestThetaMemoryNetworkLowRankCompressionRatio:
    """Tests for get_compression_ratio method."""
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        ratio = model.get_compression_ratio()
        
        expected = (d * d) / (2 * d * r)
        assert ratio == expected
        assert ratio == 2.0  # d/(2r) = 64/(2*16) = 2


class TestNumericalStabilityLowRank:
    """Tests for numerical stability."""
    
    def test_large_values(self):
        """Test with large input values."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        x = torch.randn(d) * 1000
        y = model.forward(x)
        
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
    
    def test_small_values(self):
        """Test with small input values."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        x = torch.randn(d) * 1e-6
        y = model.forward(x)
        
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
    
    def test_zero_values(self):
        """Test with zero input values."""
        d = 64
        r = 16
        model = ThetaMemoryNetworkLowRank(d=d, r=r)
        
        x = torch.zeros(d)
        y = model.forward(x)
        
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
    
    def test_long_sequence(self):
        """Test with long sequence."""
        d = 64
        r = 16
        seq_len = 500
        model = ThetaMemoryNetworkLowRank(d=d, r=r, lr=0.001)
        
        sequence = torch.randn(seq_len, d)
        targets = torch.randn(seq_len, d)
        
        outputs = model.process_sequence(sequence, targets)
        
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=term-missing"])
