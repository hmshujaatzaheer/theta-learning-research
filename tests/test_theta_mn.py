"""
Comprehensive Unit Tests for θ-Memory Network (θMN)

These tests achieve 100% code coverage for src/core/theta_mn.py
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.theta_mn import ThetaMemoryNetwork, ThetaMemoryLayer


class TestThetaMemoryNetworkInit:
    """Tests for ThetaMemoryNetwork initialization."""
    
    def test_initialization_default(self):
        """Test model initializes with default parameters."""
        d = 64
        model = ThetaMemoryNetwork(d=d)
        
        assert model.d == d
        assert model.lr == 0.01  # default
        assert model.theta.shape == (d, d)
        assert model.W_e.shape == (d, d)
        assert model.W_o.shape == (d, d)
        assert model.device == torch.device('cpu')
    
    def test_initialization_custom_lr(self):
        """Test model initializes with custom learning rate."""
        d = 32
        lr = 0.05
        model = ThetaMemoryNetwork(d=d, lr=lr)
        
        assert model.lr == lr
    
    def test_initialization_custom_init_scale(self):
        """Test model initializes with custom init scale."""
        d = 32
        init_scale = 0.1
        model = ThetaMemoryNetwork(d=d, init_scale=init_scale)
        
        # Check that weights are scaled appropriately (statistically)
        # Mean should be near 0, std should be near init_scale
        assert model.theta.std().item() < init_scale * 3
    
    def test_initialization_with_device(self):
        """Test model initializes with specified device."""
        d = 32
        device = torch.device('cpu')
        model = ThetaMemoryNetwork(d=d, device=device)
        
        assert model.device == device
        assert model.theta.device == device


class TestThetaMemoryNetworkEncode:
    """Tests for encode method."""
    
    def test_encode_single(self):
        """Test encode with single input."""
        d = 64
        model = ThetaMemoryNetwork(d=d)
        
        x = torch.randn(d)
        h = model.encode(x)
        
        assert h.shape == (d,)
    
    def test_encode_batched(self):
        """Test encode with batched input."""
        d = 64
        batch_size = 8
        model = ThetaMemoryNetwork(d=d)
        
        x = torch.randn(batch_size, d)
        h = model.encode(x)
        
        assert h.shape == (batch_size, d)


class TestThetaMemoryNetworkPredict:
    """Tests for predict method."""
    
    def test_predict_single(self):
        """Test predict with single input."""
        d = 64
        model = ThetaMemoryNetwork(d=d)
        
        h = torch.randn(d)
        y = model.predict(h)
        
        assert y.shape == (d,)
    
    def test_predict_batched(self):
        """Test predict with batched input."""
        d = 64
        batch_size = 8
        model = ThetaMemoryNetwork(d=d)
        
        h = torch.randn(batch_size, d)
        y = model.predict(h)
        
        assert y.shape == (batch_size, d)


class TestThetaMemoryNetworkComputeUpdate:
    """Tests for compute_update method."""
    
    def test_compute_update_single(self):
        """Test compute_update with single input."""
        d = 64
        model = ThetaMemoryNetwork(d=d)
        
        h = torch.randn(d)
        target = torch.randn(d)
        prediction = torch.randn(d)
        
        gradient = model.compute_update(h, target, prediction)
        
        assert gradient.shape == (d, d)
    
    def test_compute_update_batched(self):
        """Test compute_update with batched input."""
        d = 64
        batch_size = 8
        model = ThetaMemoryNetwork(d=d)
        
        h = torch.randn(batch_size, d)
        target = torch.randn(batch_size, d)
        prediction = torch.randn(batch_size, d)
        
        gradient = model.compute_update(h, target, prediction)
        
        assert gradient.shape == (d, d)


class TestThetaMemoryNetworkUpdateMemory:
    """Tests for update_memory method."""
    
    def test_update_memory(self):
        """Test that memory is updated correctly."""
        d = 64
        lr = 0.1
        model = ThetaMemoryNetwork(d=d, lr=lr)
        
        initial_theta = model.theta.data.clone()
        gradient = torch.randn(d, d)
        
        model.update_memory(gradient)
        
        expected = initial_theta + lr * gradient
        assert torch.allclose(model.theta.data, expected)


class TestThetaMemoryNetworkForward:
    """Tests for forward method."""
    
    def test_forward_single(self):
        """Test forward pass with single input."""
        d = 64
        model = ThetaMemoryNetwork(d=d)
        
        x = torch.randn(d)
        y = model.forward(x)
        
        assert y.shape == (d,)
    
    def test_forward_batched(self):
        """Test forward pass with batched input."""
        d = 64
        batch_size = 8
        model = ThetaMemoryNetwork(d=d)
        
        x = torch.randn(batch_size, d)
        y = model.forward(x)
        
        assert y.shape == (batch_size, d)


class TestThetaMemoryNetworkForwardAndUpdate:
    """Tests for forward_and_update method."""
    
    def test_forward_and_update_single(self):
        """Test forward_and_update with single input."""
        d = 64
        model = ThetaMemoryNetwork(d=d)
        
        x = torch.randn(d)
        target = torch.randn(d)
        
        initial_theta = model.theta.data.clone()
        y = model.forward_and_update(x, target)
        
        assert y.shape == (d,)
        # Memory should have changed
        assert not torch.allclose(model.theta.data, initial_theta)
    
    def test_forward_and_update_batched(self):
        """Test forward_and_update with batched input."""
        d = 64
        batch_size = 8
        model = ThetaMemoryNetwork(d=d)
        
        x = torch.randn(batch_size, d)
        target = torch.randn(batch_size, d)
        
        y = model.forward_and_update(x, target)
        
        assert y.shape == (batch_size, d)


class TestThetaMemoryNetworkResetMemory:
    """Tests for reset_memory method."""
    
    def test_reset_memory(self):
        """Test that memory resets to initial state."""
        d = 64
        model = ThetaMemoryNetwork(d=d, lr=0.1)
        
        initial_theta = model.theta.data.clone()
        
        # Make updates
        for _ in range(10):
            x = torch.randn(d)
            target = torch.randn(d)
            model.forward_and_update(x, target)
        
        # Verify memory changed
        assert not torch.allclose(model.theta.data, initial_theta)
        
        # Reset
        model.reset_memory()
        
        # Verify reset
        assert torch.allclose(model.theta.data, initial_theta)


class TestThetaMemoryNetworkProcessSequence:
    """Tests for process_sequence method."""
    
    def test_process_sequence_2d_with_targets(self):
        """Test sequence processing with 2D input and targets."""
        d = 64
        seq_len = 20
        model = ThetaMemoryNetwork(d=d)
        
        sequence = torch.randn(seq_len, d)
        targets = torch.randn(seq_len, d)
        
        outputs = model.process_sequence(sequence, targets)
        
        assert outputs.shape == (seq_len, d)
    
    def test_process_sequence_2d_without_targets(self):
        """Test sequence processing with 2D input without targets."""
        d = 64
        seq_len = 20
        model = ThetaMemoryNetwork(d=d)
        
        sequence = torch.randn(seq_len, d)
        
        outputs = model.process_sequence(sequence)
        
        assert outputs.shape == (seq_len, d)
    
    def test_process_sequence_3d_with_targets(self):
        """Test sequence processing with 3D batched input."""
        d = 64
        seq_len = 20
        batch_size = 4
        model = ThetaMemoryNetwork(d=d)
        
        sequence = torch.randn(batch_size, seq_len, d)
        targets = torch.randn(batch_size, seq_len, d)
        
        outputs = model.process_sequence(sequence, targets)
        
        assert outputs.shape == (batch_size, seq_len, d)
    
    def test_process_sequence_3d_without_targets(self):
        """Test sequence processing with 3D batched input without targets."""
        d = 64
        seq_len = 20
        batch_size = 4
        model = ThetaMemoryNetwork(d=d)
        
        sequence = torch.randn(batch_size, seq_len, d)
        
        outputs = model.process_sequence(sequence)
        
        assert outputs.shape == (batch_size, seq_len, d)
    
    def test_process_sequence_invalid_dim(self):
        """Test that invalid dimensions raise error."""
        d = 64
        model = ThetaMemoryNetwork(d=d)
        
        sequence = torch.randn(d)  # 1D - invalid
        
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            model.process_sequence(sequence)


class TestThetaMemoryNetworkQuery:
    """Tests for query method."""
    
    def test_query(self):
        """Test query method."""
        d = 64
        model = ThetaMemoryNetwork(d=d)
        
        # Process some context
        context = torch.randn(10, d)
        targets = torch.randn(10, d)
        model.process_sequence(context, targets)
        
        # Query
        query = torch.randn(d)
        answer = model.query(query)
        
        assert answer.shape == (d,)


class TestThetaMemoryNetworkFlopCount:
    """Tests for count_flops_per_token method."""
    
    def test_count_flops(self):
        """Test FLOP counting."""
        d = 64
        model = ThetaMemoryNetwork(d=d)
        
        flops = model.count_flops_per_token()
        
        assert 'encode' in flops
        assert 'predict' in flops
        assert 'error' in flops
        assert 'gradient' in flops
        assert 'update' in flops
        assert 'output' in flops
        assert 'total' in flops
        assert 'complexity' in flops
        
        # Verify values
        assert flops['encode'] == d * d
        assert flops['predict'] == d * d
        assert flops['error'] == d
        assert flops['gradient'] == d * d
        assert flops['total'] == 5 * d * d + d


class TestThetaMemoryLayer:
    """Tests for ThetaMemoryLayer class."""
    
    def test_initialization(self):
        """Test layer initialization."""
        d_model = 64
        n_heads = 4
        layer = ThetaMemoryLayer(d_model=d_model, n_heads=n_heads)
        
        assert layer.d_model == d_model
        assert layer.n_heads == n_heads
        assert layer.d_head == d_model // n_heads
        assert len(layer.theta_heads) == n_heads
    
    def test_forward_without_targets(self):
        """Test forward pass without targets."""
        d_model = 64
        n_heads = 4
        batch_size = 2
        seq_len = 10
        
        layer = ThetaMemoryLayer(d_model=d_model, n_heads=n_heads)
        
        x = torch.randn(batch_size, seq_len, d_model)
        y = layer.forward(x)
        
        assert y.shape == (batch_size, seq_len, d_model)
    
    def test_forward_with_targets(self):
        """Test forward pass with targets."""
        d_model = 64
        n_heads = 4
        batch_size = 2
        seq_len = 10
        
        layer = ThetaMemoryLayer(d_model=d_model, n_heads=n_heads)
        
        x = torch.randn(batch_size, seq_len, d_model)
        targets = torch.randn(batch_size, seq_len, d_model)
        y = layer.forward(x, targets)
        
        assert y.shape == (batch_size, seq_len, d_model)


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_large_values(self):
        """Test with large input values."""
        d = 64
        model = ThetaMemoryNetwork(d=d)
        
        x = torch.randn(d) * 1000
        y = model.forward(x)
        
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
    
    def test_small_values(self):
        """Test with small input values."""
        d = 64
        model = ThetaMemoryNetwork(d=d)
        
        x = torch.randn(d) * 1e-6
        y = model.forward(x)
        
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
    
    def test_zero_values(self):
        """Test with zero input values."""
        d = 64
        model = ThetaMemoryNetwork(d=d)
        
        x = torch.zeros(d)
        y = model.forward(x)
        
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
    
    def test_long_sequence(self):
        """Test with long sequence."""
        d = 64
        seq_len = 500
        model = ThetaMemoryNetwork(d=d, lr=0.001)
        
        sequence = torch.randn(seq_len, d)
        targets = torch.randn(seq_len, d)
        
        outputs = model.process_sequence(sequence, targets)
        
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=term-missing"])
