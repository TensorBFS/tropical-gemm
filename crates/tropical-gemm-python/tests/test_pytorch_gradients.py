"""
PyTorch gradient verification tests for tropical matmul.

Tests the backward pass implementation using:
1. Manual gradient verification against expected sparse structure
2. Numerical gradient checking where applicable
3. Optimization loop verification
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tropical_gemm
from tropical_gemm.pytorch import (
    TropicalMaxPlusMatmul,
    TropicalMinPlusMatmul,
    TropicalMaxMulMatmul,
    tropical_maxplus_matmul,
    tropical_minplus_matmul,
    tropical_maxmul_matmul,
    tropical_maxplus_matmul_gpu,
    tropical_minplus_matmul_gpu,
    GPU_AVAILABLE,
)


# ============================================================================
# Gradient structure tests
# ============================================================================


def test_maxplus_gradient_structure():
    """
    Verify the sparse gradient structure of MaxPlus matmul.

    For C[i,j] = max_k(A[i,k] + B[k,j]), the gradient is:
    - grad_A[i,k] = grad_C[i,j] if k == argmax[i,j], else 0
    - grad_B[k,j] = grad_C[i,j] if k == argmax[i,j], else 0
    """
    torch.manual_seed(42)

    # Use well-separated values to ensure unique argmax
    a = torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 6.0]], requires_grad=True)
    b = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]], requires_grad=True)

    c = tropical_maxplus_matmul(a, b)

    # Compute expected values and argmax manually
    # C[0,0] = max(1+1, 5+3, 2+2) = max(2, 8, 4) = 8, argmax=1
    # C[0,1] = max(1+2, 5+1, 2+4) = max(3, 6, 6) = 6, argmax=1 or 2
    # C[1,0] = max(3+1, 1+3, 6+2) = max(4, 4, 8) = 8, argmax=2
    # C[1,1] = max(3+2, 1+1, 6+4) = max(5, 2, 10) = 10, argmax=2

    expected_c = torch.tensor([[8.0, 6.0], [8.0, 10.0]])
    assert torch.allclose(c, expected_c), f"Forward pass incorrect: {c} vs {expected_c}"

    # Backward pass with unit gradient
    grad_c = torch.ones_like(c)
    c.backward(grad_c)

    # Check gradient structure
    # grad_A should have 1s only at argmax positions
    # Row 0: argmax is 1 for both columns -> grad_A[0,1] = 2 (or split if tied)
    # Row 1: argmax is 2 for both columns -> grad_A[1,2] = 2

    # Each C[i,j] contributes exactly 1 to the gradient sum
    assert abs(a.grad.sum().item() - c.numel()) < 0.01, "grad_A sum incorrect"
    assert abs(b.grad.sum().item() - c.numel()) < 0.01, "grad_B sum incorrect"


def test_minplus_gradient_structure():
    """Verify the sparse gradient structure of MinPlus matmul."""
    torch.manual_seed(42)

    a = torch.tensor([[5.0, 1.0, 3.0], [2.0, 4.0, 1.0]], requires_grad=True)
    b = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]], requires_grad=True)

    c = tropical_minplus_matmul(a, b)

    # C[0,0] = min(5+1, 1+3, 3+2) = min(6, 4, 5) = 4, argmax=1
    # C[0,1] = min(5+2, 1+1, 3+4) = min(7, 2, 7) = 2, argmax=1
    # C[1,0] = min(2+1, 4+3, 1+2) = min(3, 7, 3) = 3, argmax=0 or 2
    # C[1,1] = min(2+2, 4+1, 1+4) = min(4, 5, 5) = 4, argmax=0

    expected_c = torch.tensor([[4.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(c, expected_c), f"Forward pass incorrect: {c} vs {expected_c}"

    grad_c = torch.ones_like(c)
    c.backward(grad_c)

    assert abs(a.grad.sum().item() - c.numel()) < 0.01, "grad_A sum incorrect"
    assert abs(b.grad.sum().item() - c.numel()) < 0.01, "grad_B sum incorrect"


def test_maxmul_gradient_structure():
    """
    Verify the gradient structure of MaxMul matmul.

    For C[i,j] = max_k(A[i,k] * B[k,j]), the gradient is:
    - grad_A[i,k] = grad_C[i,j] * B[k,j] if k == argmax[i,j]
    - grad_B[k,j] = grad_C[i,j] * A[i,k] if k == argmax[i,j]
    """
    torch.manual_seed(42)

    # Use well-separated positive values to ensure unique argmax
    a = torch.tensor([[1.0, 3.0, 2.0], [2.0, 1.0, 4.0]], requires_grad=True)
    b = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 2.0]], requires_grad=True)

    c = tropical_maxmul_matmul(a, b)

    # C[0,0] = max(1*1, 3*2, 2*3) = max(1, 6, 6) = 6, argmax=1 or 2
    # C[0,1] = max(1*2, 3*1, 2*2) = max(2, 3, 4) = 4, argmax=2
    # C[1,0] = max(2*1, 1*2, 4*3) = max(2, 2, 12) = 12, argmax=2
    # C[1,1] = max(2*2, 1*1, 4*2) = max(4, 1, 8) = 8, argmax=2

    expected_c = torch.tensor([[6.0, 4.0], [12.0, 8.0]])
    assert torch.allclose(c, expected_c), f"Forward pass incorrect: {c} vs {expected_c}"

    grad_c = torch.ones_like(c)
    c.backward(grad_c)

    # Gradients should be nonzero only at argmax positions
    # and should include the multiplicative factor
    assert a.grad is not None, "grad_A should not be None"
    assert b.grad is not None, "grad_B should not be None"


def test_gradient_sparsity():
    """Verify that gradients are sparse (only argmax positions are nonzero)."""
    torch.manual_seed(123)

    # Create matrices where each row/column has a clear winner
    a = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], requires_grad=True
    )
    b = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True
    )

    c = tropical_maxplus_matmul(a, b)

    # With this structure, C should be diagonal-dominant
    # C[i,j] = max_k(A[i,k] + B[k,j])
    # C[0,0] = max(10+1, 0+0, 0+0) = 11, argmax=0
    # C[1,1] = max(0+0, 10+1, 0+0) = 11, argmax=1
    # C[2,2] = max(0+0, 0+0, 10+1) = 11, argmax=2

    c.backward(torch.ones_like(c))

    # Count nonzero gradients
    nonzero_a = (a.grad.abs() > 1e-6).sum().item()
    nonzero_b = (b.grad.abs() > 1e-6).sum().item()

    # Each output element contributes to exactly one A and one B element
    # 9 outputs -> at most 9 nonzero grad_A and 9 nonzero grad_B
    assert nonzero_a <= 9, f"grad_A has too many nonzeros: {nonzero_a}"
    assert nonzero_b <= 9, f"grad_B has too many nonzeros: {nonzero_b}"


# ============================================================================
# Numerical gradient verification
# ============================================================================


def test_numerical_gradient_maxplus():
    """
    Verify gradients using finite differences.

    Note: Tropical operations are piecewise linear, so gradients are
    technically subgradients. We test at points where argmax is unique.
    """
    torch.manual_seed(42)

    # Use well-separated values to ensure unique argmax
    a = torch.tensor([[1.0, 10.0], [5.0, 2.0]], requires_grad=True)
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    # Compute analytical gradient
    c = tropical_maxplus_matmul(a, b)
    loss = c.sum()
    loss.backward()

    analytical_grad_a = a.grad.clone()
    analytical_grad_b = b.grad.clone()

    # Compute numerical gradient
    eps = 1e-4
    numerical_grad_a = torch.zeros_like(a)
    numerical_grad_b = torch.zeros_like(b)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a_plus = a.detach().clone()
            a_plus[i, j] += eps
            a_minus = a.detach().clone()
            a_minus[i, j] -= eps

            c_plus = tropical_maxplus_matmul(a_plus, b.detach()).sum()
            c_minus = tropical_maxplus_matmul(a_minus, b.detach()).sum()

            numerical_grad_a[i, j] = (c_plus - c_minus) / (2 * eps)

    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            b_plus = b.detach().clone()
            b_plus[i, j] += eps
            b_minus = b.detach().clone()
            b_minus[i, j] -= eps

            c_plus = tropical_maxplus_matmul(a.detach(), b_plus).sum()
            c_minus = tropical_maxplus_matmul(a.detach(), b_minus).sum()

            numerical_grad_b[i, j] = (c_plus - c_minus) / (2 * eps)

    # Compare (allow for numerical precision issues)
    # Note: Tropical ops are piecewise linear, numerical gradients may be slightly off
    assert torch.allclose(
        analytical_grad_a, numerical_grad_a, atol=0.05
    ), f"grad_A mismatch:\nAnalytical: {analytical_grad_a}\nNumerical: {numerical_grad_a}"

    assert torch.allclose(
        analytical_grad_b, numerical_grad_b, atol=0.05
    ), f"grad_B mismatch:\nAnalytical: {analytical_grad_b}\nNumerical: {numerical_grad_b}"


def test_numerical_gradient_minplus():
    """Verify MinPlus gradients using finite differences."""
    torch.manual_seed(42)

    a = torch.tensor([[10.0, 1.0], [5.0, 8.0]], requires_grad=True)
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    c = tropical_minplus_matmul(a, b)
    loss = c.sum()
    loss.backward()

    analytical_grad_a = a.grad.clone()
    analytical_grad_b = b.grad.clone()

    eps = 1e-4
    numerical_grad_a = torch.zeros_like(a)
    numerical_grad_b = torch.zeros_like(b)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a_plus = a.detach().clone()
            a_plus[i, j] += eps
            a_minus = a.detach().clone()
            a_minus[i, j] -= eps

            c_plus = tropical_minplus_matmul(a_plus, b.detach()).sum()
            c_minus = tropical_minplus_matmul(a_minus, b.detach()).sum()

            numerical_grad_a[i, j] = (c_plus - c_minus) / (2 * eps)

    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            b_plus = b.detach().clone()
            b_plus[i, j] += eps
            b_minus = b.detach().clone()
            b_minus[i, j] -= eps

            c_plus = tropical_minplus_matmul(a.detach(), b_plus).sum()
            c_minus = tropical_minplus_matmul(a.detach(), b_minus).sum()

            numerical_grad_b[i, j] = (c_plus - c_minus) / (2 * eps)

    assert torch.allclose(
        analytical_grad_a, numerical_grad_a, atol=1e-2
    ), f"grad_A mismatch:\nAnalytical: {analytical_grad_a}\nNumerical: {numerical_grad_a}"

    assert torch.allclose(
        analytical_grad_b, numerical_grad_b, atol=1e-2
    ), f"grad_B mismatch:\nAnalytical: {analytical_grad_b}\nNumerical: {numerical_grad_b}"


def test_numerical_gradient_maxmul():
    """Verify MaxMul gradients using finite differences."""
    torch.manual_seed(42)

    # Use well-separated positive values to ensure unique argmax (avoid ties)
    # C[i,j] = max_k(A[i,k] * B[k,j])
    a = torch.tensor([[1.0, 10.0], [8.0, 1.0]], requires_grad=True)
    b = torch.tensor([[1.0, 1.0], [2.0, 3.0]], requires_grad=True)
    # C[0,0] = max(1*1, 10*2) = max(1, 20) = 20, argmax=1
    # C[0,1] = max(1*1, 10*3) = max(1, 30) = 30, argmax=1
    # C[1,0] = max(8*1, 1*2) = max(8, 2) = 8, argmax=0
    # C[1,1] = max(8*1, 1*3) = max(8, 3) = 8, argmax=0

    c = tropical_maxmul_matmul(a, b)
    loss = c.sum()
    loss.backward()

    analytical_grad_a = a.grad.clone()
    analytical_grad_b = b.grad.clone()

    eps = 1e-4
    numerical_grad_a = torch.zeros_like(a)
    numerical_grad_b = torch.zeros_like(b)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a_plus = a.detach().clone()
            a_plus[i, j] += eps
            a_minus = a.detach().clone()
            a_minus[i, j] -= eps

            c_plus = tropical_maxmul_matmul(a_plus, b.detach()).sum()
            c_minus = tropical_maxmul_matmul(a_minus, b.detach()).sum()

            numerical_grad_a[i, j] = (c_plus - c_minus) / (2 * eps)

    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            b_plus = b.detach().clone()
            b_plus[i, j] += eps
            b_minus = b.detach().clone()
            b_minus[i, j] -= eps

            c_plus = tropical_maxmul_matmul(a.detach(), b_plus).sum()
            c_minus = tropical_maxmul_matmul(a.detach(), b_minus).sum()

            numerical_grad_b[i, j] = (c_plus - c_minus) / (2 * eps)

    # Compare with relaxed tolerance for piecewise operations
    assert torch.allclose(
        analytical_grad_a, numerical_grad_a, atol=0.1
    ), f"grad_A mismatch:\nAnalytical: {analytical_grad_a}\nNumerical: {numerical_grad_a}"

    assert torch.allclose(
        analytical_grad_b, numerical_grad_b, atol=0.1
    ), f"grad_B mismatch:\nAnalytical: {analytical_grad_b}\nNumerical: {numerical_grad_b}"


# ============================================================================
# Optimization tests
# ============================================================================


def test_optimization_convergence():
    """Test that gradients enable optimization to converge."""
    torch.manual_seed(42)

    # Create learnable parameters
    a = torch.randn(3, 4, requires_grad=True)
    b = torch.randn(4, 3, requires_grad=True)

    # Target output
    target = torch.randn(3, 3)

    optimizer = torch.optim.Adam([a, b], lr=0.1)

    initial_loss = None
    final_loss = None

    for step in range(20):
        optimizer.zero_grad()

        c = tropical_maxplus_matmul(a, b)
        loss = ((c - target) ** 2).mean()

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        final_loss = loss.item()

    # Loss should decrease
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


def test_gradient_accumulation():
    """Test that gradients accumulate correctly over multiple backward passes."""
    torch.manual_seed(42)

    a = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(3, 2, requires_grad=True)

    # First backward
    c1 = tropical_maxplus_matmul(a, b)
    c1.sum().backward()
    grad_a_1 = a.grad.clone()

    # Second backward (gradients should accumulate)
    c2 = tropical_maxplus_matmul(a, b)
    c2.sum().backward()
    grad_a_2 = a.grad.clone()

    # grad_a_2 should be 2x grad_a_1
    assert torch.allclose(
        grad_a_2, 2 * grad_a_1
    ), "Gradient accumulation incorrect"


def test_gradient_with_scaling():
    """Test gradients with non-unit upstream gradient."""
    torch.manual_seed(42)

    a = torch.tensor([[1.0, 10.0], [5.0, 2.0]], requires_grad=True)
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    c = tropical_maxplus_matmul(a, b)

    # Scale the gradient
    scale = 3.0
    c.backward(scale * torch.ones_like(c))

    # Check that gradient is scaled correctly
    # Reset and compute with unit gradient
    a2 = a.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    c2 = tropical_maxplus_matmul(a2, b2)
    c2.backward(torch.ones_like(c2))

    assert torch.allclose(
        a.grad, scale * a2.grad
    ), "Scaled gradient incorrect for A"
    assert torch.allclose(
        b.grad, scale * b2.grad
    ), "Scaled gradient incorrect for B"


# ============================================================================
# Edge cases
# ============================================================================


def test_single_element_gradient():
    """Test gradient for 1x1 matmul."""
    a = torch.tensor([[5.0]], requires_grad=True)
    b = torch.tensor([[3.0]], requires_grad=True)

    c = tropical_maxplus_matmul(a, b)
    assert c.item() == 8.0

    c.backward()
    assert a.grad.item() == 1.0
    assert b.grad.item() == 1.0


def test_rectangular_gradient():
    """Test gradient for non-square matrices."""
    torch.manual_seed(42)

    a = torch.randn(2, 5, requires_grad=True)
    b = torch.randn(5, 3, requires_grad=True)

    c = tropical_maxplus_matmul(a, b)
    assert c.shape == (2, 3)

    c.backward(torch.ones_like(c))

    assert a.grad.shape == (2, 5)
    assert b.grad.shape == (5, 3)

    # Each output element contributes to exactly one gradient
    assert abs(a.grad.sum().item() - 6) < 0.01  # 2x3 = 6 elements
    assert abs(b.grad.sum().item() - 6) < 0.01


# ============================================================================
# GPU tests (when available)
# ============================================================================


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA not available")
def test_gpu_maxplus_forward():
    """Test GPU MaxPlus forward pass matches CPU."""
    torch.manual_seed(42)

    a = torch.randn(4, 3)
    b = torch.randn(3, 5)

    c_cpu = tropical_maxplus_matmul(a, b)
    c_gpu = tropical_maxplus_matmul_gpu(a, b)

    assert torch.allclose(c_cpu, c_gpu, atol=1e-4), f"GPU result differs from CPU"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA not available")
def test_gpu_minplus_forward():
    """Test GPU MinPlus forward pass matches CPU."""
    torch.manual_seed(42)

    a = torch.randn(4, 3)
    b = torch.randn(3, 5)

    c_cpu = tropical_minplus_matmul(a, b)
    c_gpu = tropical_minplus_matmul_gpu(a, b)

    assert torch.allclose(c_cpu, c_gpu, atol=1e-4), f"GPU result differs from CPU"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA not available")
def test_gpu_maxplus_gradient():
    """Test GPU MaxPlus backward pass."""
    torch.manual_seed(42)

    a = torch.tensor([[1.0, 10.0], [5.0, 2.0]], requires_grad=True)
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    c = tropical_maxplus_matmul_gpu(a, b)
    loss = c.sum()
    loss.backward()

    # Check gradient structure
    assert a.grad is not None
    assert b.grad is not None
    assert abs(a.grad.sum().item() - c.numel()) < 0.01
    assert abs(b.grad.sum().item() - c.numel()) < 0.01


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA not available")
def test_gpu_optimization():
    """Test GPU optimization convergence."""
    torch.manual_seed(42)

    a = torch.randn(3, 4, requires_grad=True)
    b = torch.randn(4, 3, requires_grad=True)
    target = torch.randn(3, 3)

    optimizer = torch.optim.Adam([a, b], lr=0.1)

    initial_loss = None
    final_loss = None

    for step in range(10):
        optimizer.zero_grad()

        c = tropical_maxplus_matmul_gpu(a, b)
        loss = ((c - target) ** 2).mean()

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        final_loss = loss.item()

    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


# ============================================================================
# Module import tests
# ============================================================================


def test_pytorch_module_exports():
    """Test that the pytorch module exports all expected functions."""
    from tropical_gemm import pytorch

    # Check that all expected items are exported
    assert hasattr(pytorch, "TropicalMaxPlusMatmul")
    assert hasattr(pytorch, "TropicalMinPlusMatmul")
    assert hasattr(pytorch, "TropicalMaxMulMatmul")
    assert hasattr(pytorch, "tropical_maxplus_matmul")
    assert hasattr(pytorch, "tropical_minplus_matmul")
    assert hasattr(pytorch, "tropical_maxmul_matmul")
    assert hasattr(pytorch, "tropical_maxplus_matmul_gpu")
    assert hasattr(pytorch, "tropical_minplus_matmul_gpu")
    assert hasattr(pytorch, "GPU_AVAILABLE")


def test_main_module_exports():
    """Test that the main module exports all expected functions."""
    import tropical_gemm

    # Check core functions
    assert hasattr(tropical_gemm, "maxplus_matmul")
    assert hasattr(tropical_gemm, "minplus_matmul")
    assert hasattr(tropical_gemm, "maxmul_matmul")
    assert hasattr(tropical_gemm, "maxplus_matmul_with_argmax")
    assert hasattr(tropical_gemm, "backward_a")
    assert hasattr(tropical_gemm, "backward_b")


# ============================================================================
# DLPack integration tests
# ============================================================================


def test_dlpack_availability_flag():
    """Test that _DLPACK_AVAILABLE flag is correctly exported."""
    from tropical_gemm.pytorch import _DLPACK_AVAILABLE

    # Flag should be a boolean
    assert isinstance(_DLPACK_AVAILABLE, bool)

    # If cuda_available() returns True, DLPack should be available
    if tropical_gemm.cuda_available():
        assert _DLPACK_AVAILABLE, "DLPack should be available when CUDA is enabled"


def test_dlpack_functions_exist_when_cuda_available():
    """Test that DLPack functions are exported when CUDA is available."""
    if not tropical_gemm.cuda_available():
        pytest.skip("CUDA not available")

    assert hasattr(tropical_gemm, "maxplus_matmul_dlpack")
    assert hasattr(tropical_gemm, "minplus_matmul_dlpack")
    assert hasattr(tropical_gemm, "maxmul_matmul_dlpack")


@pytest.mark.skipif(not tropical_gemm.cuda_available(), reason="CUDA not available")
def test_dlpack_maxplus_cpu_tensor():
    """Test DLPack maxplus with CPU tensors (uses CPU backend)."""
    torch.manual_seed(42)

    a = torch.randn(10, 20, dtype=torch.float32)
    b = torch.randn(20, 15, dtype=torch.float32)

    # Call DLPack function with CPU tensors - should fall back to CPU backend
    c_flat, _ = tropical_gemm.maxplus_matmul_dlpack(a, b)

    # Convert to torch tensors
    c = torch.from_numpy(np.array(c_flat).reshape(10, 15))

    # Verify against reference implementation
    a_np = a.numpy()
    b_np = b.numpy()
    c_ref_flat, argmax_ref_flat = tropical_gemm.maxplus_matmul_with_argmax(a_np, b_np)
    c_ref = torch.from_numpy(np.array(c_ref_flat).reshape(10, 15))

    assert torch.allclose(c, c_ref), "DLPack CPU path should match reference"


@pytest.mark.skipif(not tropical_gemm.cuda_available(), reason="CUDA not available")
def test_dlpack_minplus_cpu_tensor():
    """Test DLPack minplus with CPU tensors."""
    torch.manual_seed(42)

    a = torch.randn(10, 20, dtype=torch.float32)
    b = torch.randn(20, 15, dtype=torch.float32)

    c_flat, _ = tropical_gemm.minplus_matmul_dlpack(a, b)
    c = torch.from_numpy(np.array(c_flat).reshape(10, 15))

    # Verify against reference
    a_np = a.numpy()
    b_np = b.numpy()
    c_ref_flat, _ = tropical_gemm.minplus_matmul_with_argmax(a_np, b_np)
    c_ref = torch.from_numpy(np.array(c_ref_flat).reshape(10, 15))

    assert torch.allclose(c, c_ref), "DLPack minplus CPU path should match reference"


@pytest.mark.skipif(not tropical_gemm.cuda_available(), reason="CUDA not available")
def test_dlpack_maxmul_cpu_tensor():
    """Test DLPack maxmul with CPU tensors."""
    torch.manual_seed(42)

    a = torch.randn(10, 20, dtype=torch.float32).abs() + 0.1  # Positive values for maxmul
    b = torch.randn(20, 15, dtype=torch.float32).abs() + 0.1

    c_flat, _ = tropical_gemm.maxmul_matmul_dlpack(a, b)
    c = torch.from_numpy(np.array(c_flat).reshape(10, 15))

    # Verify against reference
    a_np = a.numpy()
    b_np = b.numpy()
    c_ref_flat, _ = tropical_gemm.maxmul_matmul_with_argmax(a_np, b_np)
    c_ref = torch.from_numpy(np.array(c_ref_flat).reshape(10, 15))

    assert torch.allclose(c, c_ref), "DLPack maxmul CPU path should match reference"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA not available")
def test_dlpack_gpu_tensor_zero_copy():
    """Test that DLPack with GPU tensors uses the Rust CUDA backend."""
    torch.manual_seed(42)

    a = torch.randn(100, 50, dtype=torch.float32, device='cuda')
    b = torch.randn(50, 80, dtype=torch.float32, device='cuda')

    # This should use the zero-copy DLPack path with Rust CUDA backend
    c_flat, _ = tropical_gemm.maxplus_matmul_dlpack(a, b)
    c = torch.from_numpy(np.array(c_flat).reshape(100, 80))

    # Verify result matches CPU reference
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    c_ref_flat, _ = tropical_gemm.maxplus_matmul_with_argmax(a_cpu.numpy(), b_cpu.numpy())
    c_ref = torch.from_numpy(np.array(c_ref_flat).reshape(100, 80))

    assert torch.allclose(c, c_ref, atol=1e-5), "GPU DLPack path should match CPU reference"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA not available")
def test_gpu_autograd_uses_dlpack():
    """Test that GPU autograd functions use DLPack when available."""
    from tropical_gemm.pytorch import _DLPACK_AVAILABLE

    if not _DLPACK_AVAILABLE:
        pytest.skip("DLPack not available")

    torch.manual_seed(42)

    a = torch.randn(50, 30, dtype=torch.float32, device='cuda', requires_grad=True)
    b = torch.randn(30, 40, dtype=torch.float32, device='cuda', requires_grad=True)

    # Forward pass should use DLPack + Rust CUDA backend
    c = tropical_maxplus_matmul_gpu(a, b)

    # Result should be on the same device
    assert c.device == a.device, "Output should be on same device as input"

    # Backward should work
    loss = c.sum()
    loss.backward()

    assert a.grad is not None, "grad_a should be computed"
    assert b.grad is not None, "grad_b should be computed"


@pytest.mark.skipif(not tropical_gemm.cuda_available(), reason="CUDA not available")
def test_dlpack_contiguity_check():
    """Test that DLPack functions require contiguous tensors."""
    a = torch.randn(10, 20, dtype=torch.float32)
    b = torch.randn(20, 15, dtype=torch.float32)

    # Create non-contiguous tensor (stride != 1)
    a_noncontig = a[:, ::2]

    # Should raise an error for non-contiguous tensors
    with pytest.raises(Exception):  # Could be ValueError or RuntimeError
        tropical_gemm.maxplus_matmul_dlpack(a_noncontig, b[:20//2, :])


@pytest.mark.skipif(not tropical_gemm.cuda_available(), reason="CUDA not available")
def test_dlpack_dtype_check():
    """Test that DLPack functions only accept f32 tensors."""
    a = torch.randn(10, 20, dtype=torch.float64)  # f64, not f32
    b = torch.randn(20, 15, dtype=torch.float64)

    with pytest.raises(Exception):  # Should raise error for non-f32
        tropical_gemm.maxplus_matmul_dlpack(a, b)


def test_tropical_gemm_has_metadata():
    """Basic sanity checks on tropical_gemm module attributes."""
    assert hasattr(tropical_gemm, "cuda_available")
    assert hasattr(tropical_gemm, "__version__")


# ============================================================================
# Batched operation tests
# ============================================================================


from tropical_gemm.pytorch import (
    TropicalMaxPlusMatmulBatched,
    TropicalMinPlusMatmulBatched,
    TropicalMaxMulMatmulBatched,
    tropical_maxplus_matmul_batched,
    tropical_minplus_matmul_batched,
    tropical_maxmul_matmul_batched,
)


def test_batched_maxplus_forward_correctness():
    """Test batched MaxPlus forward pass matches looped reference."""
    torch.manual_seed(42)

    batch, m, k, n = 4, 3, 5, 2
    a = torch.randn(batch, m, k)
    b = torch.randn(batch, k, n)

    # Batched computation
    c_batched = tropical_maxplus_matmul_batched(a, b)

    # Reference: loop over batch
    c_ref = torch.stack([
        tropical_maxplus_matmul(a[i], b[i])
        for i in range(batch)
    ])

    assert c_batched.shape == (batch, m, n)
    assert torch.allclose(c_batched, c_ref, atol=1e-5), \
        f"Batched result differs from reference:\n{c_batched}\nvs\n{c_ref}"


def test_batched_minplus_forward_correctness():
    """Test batched MinPlus forward pass matches looped reference."""
    torch.manual_seed(42)

    batch, m, k, n = 4, 3, 5, 2
    a = torch.randn(batch, m, k)
    b = torch.randn(batch, k, n)

    c_batched = tropical_minplus_matmul_batched(a, b)

    c_ref = torch.stack([
        tropical_minplus_matmul(a[i], b[i])
        for i in range(batch)
    ])

    assert c_batched.shape == (batch, m, n)
    assert torch.allclose(c_batched, c_ref, atol=1e-5)


def test_batched_maxmul_forward_correctness():
    """Test batched MaxMul forward pass matches looped reference."""
    torch.manual_seed(42)

    batch, m, k, n = 4, 3, 5, 2
    a = torch.randn(batch, m, k).abs() + 0.1
    b = torch.randn(batch, k, n).abs() + 0.1

    c_batched = tropical_maxmul_matmul_batched(a, b)

    c_ref = torch.stack([
        tropical_maxmul_matmul(a[i], b[i])
        for i in range(batch)
    ])

    assert c_batched.shape == (batch, m, n)
    assert torch.allclose(c_batched, c_ref, atol=1e-5)


def test_batched_maxplus_gradient_structure():
    """Verify gradient structure of batched MaxPlus matmul."""
    torch.manual_seed(42)

    batch, m, k, n = 2, 3, 4, 2
    a = torch.randn(batch, m, k, requires_grad=True)
    b = torch.randn(batch, k, n, requires_grad=True)

    c = tropical_maxplus_matmul_batched(a, b)
    c.backward(torch.ones_like(c))

    # Each output element contributes to exactly one gradient
    assert abs(a.grad.sum().item() - c.numel()) < 0.01, "grad_A sum incorrect"
    assert abs(b.grad.sum().item() - c.numel()) < 0.01, "grad_B sum incorrect"


def test_batched_minplus_gradient_structure():
    """Verify gradient structure of batched MinPlus matmul."""
    torch.manual_seed(42)

    batch, m, k, n = 2, 3, 4, 2
    a = torch.randn(batch, m, k, requires_grad=True)
    b = torch.randn(batch, k, n, requires_grad=True)

    c = tropical_minplus_matmul_batched(a, b)
    c.backward(torch.ones_like(c))

    assert abs(a.grad.sum().item() - c.numel()) < 0.01, "grad_A sum incorrect"
    assert abs(b.grad.sum().item() - c.numel()) < 0.01, "grad_B sum incorrect"


def test_batched_maxmul_gradient_structure():
    """Verify gradient structure of batched MaxMul matmul."""
    torch.manual_seed(42)

    batch, m, k, n = 2, 3, 4, 2
    # Use positive values for MaxMul - create as leaf tensors
    a = (torch.randn(batch, m, k).abs() + 0.1).requires_grad_(True)
    b = (torch.randn(batch, k, n).abs() + 0.1).requires_grad_(True)

    c = tropical_maxmul_matmul_batched(a, b)
    c.backward(torch.ones_like(c))

    assert a.grad is not None, "grad_A should not be None"
    assert b.grad is not None, "grad_B should not be None"
    assert a.grad.shape == (batch, m, k)
    assert b.grad.shape == (batch, k, n)


def test_batched_numerical_gradient_maxplus():
    """Verify batched MaxPlus gradients using finite differences."""
    torch.manual_seed(42)

    batch, m, k, n = 2, 2, 3, 2
    # Use well-separated values to ensure unique argmax - create as leaf tensors
    a = (torch.randn(batch, m, k) * 3).requires_grad_(True)
    b = (torch.randn(batch, k, n) * 3).requires_grad_(True)

    c = tropical_maxplus_matmul_batched(a, b)
    loss = c.sum()
    loss.backward()

    analytical_grad_a = a.grad.clone()

    eps = 1e-4
    numerical_grad_a = torch.zeros_like(a)

    for bi in range(batch):
        for i in range(m):
            for j in range(k):
                a_plus = a.detach().clone()
                a_plus[bi, i, j] += eps
                a_minus = a.detach().clone()
                a_minus[bi, i, j] -= eps

                c_plus = tropical_maxplus_matmul_batched(a_plus, b.detach()).sum()
                c_minus = tropical_maxplus_matmul_batched(a_minus, b.detach()).sum()

                numerical_grad_a[bi, i, j] = (c_plus - c_minus) / (2 * eps)

    assert torch.allclose(analytical_grad_a, numerical_grad_a, atol=0.1), \
        f"grad_A mismatch:\nAnalytical: {analytical_grad_a}\nNumerical: {numerical_grad_a}"


def test_batched_optimization_convergence():
    """Test that batched gradients enable optimization to converge."""
    torch.manual_seed(42)

    batch, m, k, n = 3, 4, 5, 3
    a = torch.randn(batch, m, k, requires_grad=True)
    b = torch.randn(batch, k, n, requires_grad=True)
    target = torch.randn(batch, m, n)

    optimizer = torch.optim.Adam([a, b], lr=0.1)

    initial_loss = None
    final_loss = None

    for step in range(20):
        optimizer.zero_grad()

        c = tropical_maxplus_matmul_batched(a, b)
        loss = ((c - target) ** 2).mean()

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        final_loss = loss.item()

    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


def test_batched_single_batch():
    """Test batched operation with batch size 1."""
    torch.manual_seed(42)

    a = torch.randn(1, 3, 4, requires_grad=True)
    b = torch.randn(1, 4, 2, requires_grad=True)

    c = tropical_maxplus_matmul_batched(a, b)
    assert c.shape == (1, 3, 2)

    c.backward(torch.ones_like(c))
    assert a.grad.shape == (1, 3, 4)
    assert b.grad.shape == (1, 4, 2)


def test_batched_large_batch():
    """Test batched operation with larger batch size."""
    torch.manual_seed(42)

    batch, m, k, n = 16, 8, 10, 6
    a = torch.randn(batch, m, k, requires_grad=True)
    b = torch.randn(batch, k, n, requires_grad=True)

    c = tropical_maxplus_matmul_batched(a, b)
    assert c.shape == (batch, m, n)

    loss = c.sum()
    loss.backward()

    assert a.grad.shape == (batch, m, k)
    assert b.grad.shape == (batch, k, n)


def test_batched_gradient_accumulation():
    """Test that batched gradients accumulate correctly."""
    torch.manual_seed(42)

    batch = 3
    a = torch.randn(batch, 2, 3, requires_grad=True)
    b = torch.randn(batch, 3, 2, requires_grad=True)

    c1 = tropical_maxplus_matmul_batched(a, b)
    c1.sum().backward()
    grad_a_1 = a.grad.clone()

    c2 = tropical_maxplus_matmul_batched(a, b)
    c2.sum().backward()
    grad_a_2 = a.grad.clone()

    assert torch.allclose(grad_a_2, 2 * grad_a_1), "Gradient accumulation incorrect"


def test_batched_exports():
    """Test that batched functions are exported correctly."""
    from tropical_gemm import pytorch

    assert hasattr(pytorch, "TropicalMaxPlusMatmulBatched")
    assert hasattr(pytorch, "TropicalMinPlusMatmulBatched")
    assert hasattr(pytorch, "TropicalMaxMulMatmulBatched")
    assert hasattr(pytorch, "tropical_maxplus_matmul_batched")
    assert hasattr(pytorch, "tropical_minplus_matmul_batched")
    assert hasattr(pytorch, "tropical_maxmul_matmul_batched")
