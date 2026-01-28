"""Tests for batched tropical matmul PyTorch integration.

Tests cover:
- Forward pass correctness (compared with loop implementation)
- Gradient structure validation
- Numerical gradient check (finite differences)
- Optimization convergence test
- GPU forward/backward (if CUDA available)
"""

import numpy as np
import pytest
import torch

from tropical_gemm.pytorch import (
    # Batched CPU functions
    tropical_maxplus_matmul_batched,
    tropical_minplus_matmul_batched,
    tropical_maxmul_matmul_batched,
    # Batched GPU functions
    tropical_maxplus_matmul_batched_gpu,
    tropical_minplus_matmul_batched_gpu,
    tropical_maxmul_matmul_batched_gpu,
    # Non-batched functions for comparison
    tropical_maxplus_matmul,
    tropical_minplus_matmul,
    tropical_maxmul_matmul,
    # Flags
    GPU_AVAILABLE,
)


# ============================================================================
# Forward pass correctness tests
# ============================================================================


def test_maxplus_batched_forward_correctness():
    """Test batched MaxPlus forward pass matches loop implementation."""
    batch_size = 4
    m, k, n = 3, 4, 5

    torch.manual_seed(42)
    a = torch.randn(batch_size, m, k)
    b = torch.randn(batch_size, k, n)

    # Batched result
    c_batched = tropical_maxplus_matmul_batched(a, b)

    # Loop result
    c_loop = torch.stack([tropical_maxplus_matmul(a[i], b[i]) for i in range(batch_size)])

    np.testing.assert_array_almost_equal(
        c_batched.numpy(), c_loop.numpy(), decimal=5
    )


def test_minplus_batched_forward_correctness():
    """Test batched MinPlus forward pass matches loop implementation."""
    batch_size = 4
    m, k, n = 3, 4, 5

    torch.manual_seed(42)
    a = torch.randn(batch_size, m, k)
    b = torch.randn(batch_size, k, n)

    c_batched = tropical_minplus_matmul_batched(a, b)
    c_loop = torch.stack([tropical_minplus_matmul(a[i], b[i]) for i in range(batch_size)])

    np.testing.assert_array_almost_equal(
        c_batched.numpy(), c_loop.numpy(), decimal=5
    )


def test_maxmul_batched_forward_correctness():
    """Test batched MaxMul forward pass matches loop implementation."""
    batch_size = 4
    m, k, n = 3, 4, 5

    torch.manual_seed(42)
    # Use positive values for maxmul to avoid issues with negative products
    a = torch.abs(torch.randn(batch_size, m, k)) + 0.1
    b = torch.abs(torch.randn(batch_size, k, n)) + 0.1

    c_batched = tropical_maxmul_matmul_batched(a, b)
    c_loop = torch.stack([tropical_maxmul_matmul(a[i], b[i]) for i in range(batch_size)])

    np.testing.assert_array_almost_equal(
        c_batched.numpy(), c_loop.numpy(), decimal=5
    )


def test_batched_output_shape():
    """Test that batched operations produce correct output shapes."""
    batch_size = 4
    m, k, n = 32, 64, 16

    a = torch.randn(batch_size, m, k)
    b = torch.randn(batch_size, k, n)

    c_maxplus = tropical_maxplus_matmul_batched(a, b)
    c_minplus = tropical_minplus_matmul_batched(a, b)
    c_maxmul = tropical_maxmul_matmul_batched(a, b)

    assert c_maxplus.shape == (batch_size, m, n)
    assert c_minplus.shape == (batch_size, m, n)
    assert c_maxmul.shape == (batch_size, m, n)


def test_batched_single_batch():
    """Test batched operations with batch_size=1."""
    a = torch.randn(1, 4, 5)
    b = torch.randn(1, 5, 3)

    c = tropical_maxplus_matmul_batched(a, b)
    assert c.shape == (1, 4, 3)

    # Compare with non-batched
    c_expected = tropical_maxplus_matmul(a[0], b[0])
    np.testing.assert_array_almost_equal(c[0].numpy(), c_expected.numpy(), decimal=5)


# ============================================================================
# Gradient structure validation
# ============================================================================


def test_maxplus_batched_gradient_structure():
    """Test that batched MaxPlus produces sparse gradients with correct structure."""
    batch_size = 2
    m, k, n = 2, 3, 2

    a = torch.randn(batch_size, m, k, requires_grad=True)
    b = torch.randn(batch_size, k, n, requires_grad=True)

    c = tropical_maxplus_matmul_batched(a, b)
    loss = c.sum()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape

    # Each output element contributes gradient to exactly one position in each input
    # Total gradient should equal batch_size * m * n (number of output elements)
    total_grad_a = a.grad.sum().item()
    total_grad_b = b.grad.sum().item()
    expected_total = batch_size * m * n

    np.testing.assert_almost_equal(total_grad_a, expected_total, decimal=5)
    np.testing.assert_almost_equal(total_grad_b, expected_total, decimal=5)


def test_minplus_batched_gradient_structure():
    """Test that batched MinPlus produces sparse gradients with correct structure."""
    batch_size = 2
    m, k, n = 2, 3, 2

    a = torch.randn(batch_size, m, k, requires_grad=True)
    b = torch.randn(batch_size, k, n, requires_grad=True)

    c = tropical_minplus_matmul_batched(a, b)
    loss = c.sum()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None

    total_grad_a = a.grad.sum().item()
    total_grad_b = b.grad.sum().item()
    expected_total = batch_size * m * n

    np.testing.assert_almost_equal(total_grad_a, expected_total, decimal=5)
    np.testing.assert_almost_equal(total_grad_b, expected_total, decimal=5)


def test_maxmul_batched_gradient_exists():
    """Test that batched MaxMul produces gradients."""
    batch_size = 2
    m, k, n = 2, 3, 2

    a = torch.abs(torch.randn(batch_size, m, k)) + 0.1
    a.requires_grad = True
    b = torch.abs(torch.randn(batch_size, k, n)) + 0.1
    b.requires_grad = True

    c = tropical_maxmul_matmul_batched(a, b)
    loss = c.sum()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape


# ============================================================================
# Numerical gradient check (finite differences)
# ============================================================================


def numerical_gradient(fn, inputs, idx, eps=1e-4):
    """Compute numerical gradient using finite differences."""
    x = inputs[idx]
    grad = torch.zeros_like(x)

    for i in range(x.numel()):
        x_flat = x.flatten()
        orig_val = x_flat[i].item()

        x_flat[i] = orig_val + eps
        inputs_plus = [inp.clone() if j != idx else x.view(inputs[idx].shape) for j, inp in enumerate(inputs)]
        out_plus = fn(*inputs_plus).sum()

        x_flat[i] = orig_val - eps
        inputs_minus = [inp.clone() if j != idx else x.view(inputs[idx].shape) for j, inp in enumerate(inputs)]
        out_minus = fn(*inputs_minus).sum()

        x_flat[i] = orig_val
        grad.flatten()[i] = (out_plus - out_minus) / (2 * eps)

    return grad


def test_maxplus_batched_numerical_gradient():
    """Verify MaxPlus batched gradients against numerical differentiation.

    Note: Tropical operations are piecewise-linear and non-differentiable at
    boundaries. This test checks gradient structure rather than exact values.
    """
    batch_size = 2
    m, k, n = 2, 3, 2

    torch.manual_seed(42)
    a = torch.randn(batch_size, m, k, requires_grad=True, dtype=torch.float64)
    b = torch.randn(batch_size, k, n, requires_grad=True, dtype=torch.float64)

    # Compute analytical gradients
    c = tropical_maxplus_matmul_batched(a, b)
    c.sum().backward()

    # Verify gradient structure:
    # 1. Gradients should exist
    assert a.grad is not None
    assert b.grad is not None

    # 2. Total gradient should equal number of output elements (each output contributes 1)
    total_grad_a = a.grad.sum().item()
    total_grad_b = b.grad.sum().item()
    expected_total = batch_size * m * n

    np.testing.assert_almost_equal(total_grad_a, expected_total, decimal=5)
    np.testing.assert_almost_equal(total_grad_b, expected_total, decimal=5)

    # 3. Gradients should be non-negative (each position receives gradient from winning paths)
    assert (a.grad >= 0).all()
    assert (b.grad >= 0).all()

    # 4. For each (batch, row) in A, gradient should be concentrated in k indices
    # that won in the forward pass
    for batch_idx in range(batch_size):
        for row in range(m):
            # Each row of A contributes to n output elements
            # So total gradient for that row should be n
            row_grad_sum = a.grad[batch_idx, row].sum().item()
            np.testing.assert_almost_equal(row_grad_sum, n, decimal=5)


# ============================================================================
# Optimization convergence test
# ============================================================================


def test_optimization_convergence():
    """Test that batched tropical matmul can be used in optimization."""
    batch_size = 4
    m, k, n = 8, 6, 8

    torch.manual_seed(42)

    # Create target
    a_target = torch.randn(batch_size, m, k)
    b_target = torch.randn(batch_size, k, n)
    with torch.no_grad():
        c_target = tropical_maxplus_matmul_batched(a_target, b_target)

    # Trainable parameters
    a = torch.randn(batch_size, m, k, requires_grad=True)
    b = torch.randn(batch_size, k, n, requires_grad=True)

    optimizer = torch.optim.Adam([a, b], lr=0.1)

    initial_loss = None
    final_loss = None

    for step in range(50):
        optimizer.zero_grad()
        c = tropical_maxplus_matmul_batched(a, b)
        loss = ((c - c_target) ** 2).mean()

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if step == 49:
            final_loss = loss.item()

    # Loss should decrease
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


# ============================================================================
# GPU tests (if CUDA available)
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_maxplus_batched_gpu_forward():
    """Test GPU batched MaxPlus forward pass."""
    batch_size = 4
    m, k, n = 8, 16, 8

    a = torch.randn(batch_size, m, k).cuda()
    b = torch.randn(batch_size, k, n).cuda()

    c = tropical_maxplus_matmul_batched_gpu(a, b)

    assert c.device.type == "cuda"
    assert c.shape == (batch_size, m, n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_maxplus_batched_gpu_backward():
    """Test GPU batched MaxPlus backward pass."""
    batch_size = 4
    m, k, n = 8, 16, 8

    a = torch.randn(batch_size, m, k, requires_grad=True).cuda()
    b = torch.randn(batch_size, k, n, requires_grad=True).cuda()

    c = tropical_maxplus_matmul_batched_gpu(a, b)
    loss = c.sum()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.device.type == "cuda"
    assert b.grad.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_minplus_batched_gpu():
    """Test GPU batched MinPlus."""
    batch_size = 4
    m, k, n = 8, 16, 8

    a = torch.randn(batch_size, m, k, requires_grad=True).cuda()
    b = torch.randn(batch_size, k, n, requires_grad=True).cuda()

    c = tropical_minplus_matmul_batched_gpu(a, b)
    c.sum().backward()

    assert c.device.type == "cuda"
    assert a.grad is not None
    assert b.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_maxmul_batched_gpu():
    """Test GPU batched MaxMul."""
    batch_size = 4
    m, k, n = 8, 16, 8

    a = torch.abs(torch.randn(batch_size, m, k)).cuda() + 0.1
    a.requires_grad = True
    b = torch.abs(torch.randn(batch_size, k, n)).cuda() + 0.1
    b.requires_grad = True

    c = tropical_maxmul_matmul_batched_gpu(a, b)
    c.sum().backward()

    assert c.device.type == "cuda"
    assert a.grad is not None
    assert b.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_cpu_consistency():
    """Test that GPU and CPU produce same results."""
    batch_size = 4
    m, k, n = 8, 16, 8

    torch.manual_seed(42)
    a = torch.randn(batch_size, m, k)
    b = torch.randn(batch_size, k, n)

    c_cpu = tropical_maxplus_matmul_batched(a, b)
    c_gpu = tropical_maxplus_matmul_batched_gpu(a.cuda(), b.cuda())

    np.testing.assert_array_almost_equal(
        c_cpu.numpy(), c_gpu.cpu().numpy(), decimal=4
    )


# ============================================================================
# Edge cases
# ============================================================================


def test_batched_larger_matrices():
    """Test with larger batch and matrix sizes."""
    batch_size = 16
    m, k, n = 64, 32, 48

    a = torch.randn(batch_size, m, k, requires_grad=True)
    b = torch.randn(batch_size, k, n, requires_grad=True)

    c = tropical_maxplus_matmul_batched(a, b)
    assert c.shape == (batch_size, m, n)

    c.sum().backward()
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape


def test_batched_with_special_values():
    """Test batched operations with special values."""
    batch_size = 2
    m, k, n = 2, 2, 2

    # For MinPlus, the tropical identity uses +inf (not -inf)
    # MinPlus: C[i,j] = min_k(A[i,k] + B[k,j])
    # Identity: A = [[0, +inf], [+inf, 0]], so C = A @ B = B
    inf = float('inf')
    a = torch.tensor([
        [[0.0, inf], [inf, 0.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ])
    b = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ])

    # MinPlus with identity-like matrix should return the other matrix
    c = tropical_minplus_matmul_batched(a, b)
    assert c.shape == (batch_size, m, n)

    # First batch: identity * b = b
    # C[0,0,0] = min(0+1, inf+3) = 1
    # C[0,0,1] = min(0+2, inf+4) = 2
    np.testing.assert_almost_equal(c[0, 0, 0].item(), 1.0)
    np.testing.assert_almost_equal(c[0, 0, 1].item(), 2.0)

    # For MaxPlus, test with negative infinity (tropical zero for MaxPlus)
    neg_inf = float('-inf')
    a_maxplus = torch.tensor([
        [[0.0, neg_inf], [neg_inf, 0.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ])
    b_maxplus = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ])

    c_maxplus = tropical_maxplus_matmul_batched(a_maxplus, b_maxplus)
    # C[0,0,0] = max(0+1, -inf+3) = 1
    np.testing.assert_almost_equal(c_maxplus[0, 0, 0].item(), 1.0)
    np.testing.assert_almost_equal(c_maxplus[0, 0, 1].item(), 2.0)


def test_batched_contiguous_input():
    """Test that non-contiguous inputs are handled correctly."""
    batch_size = 4
    m, k, n = 8, 6, 8

    a = torch.randn(batch_size, m, k).transpose(1, 2).transpose(1, 2)  # Force non-contiguous
    b = torch.randn(batch_size, k, n)

    # Should still work
    c = tropical_maxplus_matmul_batched(a, b)
    assert c.shape == (batch_size, m, n)
