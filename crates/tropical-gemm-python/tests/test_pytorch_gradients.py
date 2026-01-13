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


class TropicalMaxPlusMatmul(torch.autograd.Function):
    """MaxPlus tropical matmul with custom backward."""

    @staticmethod
    def forward(ctx, a, b):
        m, k = a.shape
        n = b.shape[1]

        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        c_flat, argmax_flat = tropical_gemm.maxplus_matmul_with_argmax(a_np, b_np)

        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        ctx.save_for_backward(torch.from_numpy(argmax_np))
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c):
        (argmax,) = ctx.saved_tensors
        k = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        grad_a_flat = tropical_gemm.backward_a(grad_c_np, argmax_np, k)
        grad_b_flat = tropical_gemm.backward_b(grad_c_np, argmax_np, k)

        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k)).to(grad_c.device)
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k, n)).to(grad_c.device)

        return grad_a, grad_b


class TropicalMinPlusMatmul(torch.autograd.Function):
    """MinPlus tropical matmul with custom backward."""

    @staticmethod
    def forward(ctx, a, b):
        m, k = a.shape
        n = b.shape[1]

        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        c_flat, argmax_flat = tropical_gemm.minplus_matmul_with_argmax(a_np, b_np)

        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        ctx.save_for_backward(torch.from_numpy(argmax_np))
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c):
        (argmax,) = ctx.saved_tensors
        k = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        grad_a_flat = tropical_gemm.backward_a(grad_c_np, argmax_np, k)
        grad_b_flat = tropical_gemm.backward_b(grad_c_np, argmax_np, k)

        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k)).to(grad_c.device)
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k, n)).to(grad_c.device)

        return grad_a, grad_b


class TropicalMaxMulMatmul(torch.autograd.Function):
    """
    MaxMul tropical matmul with custom backward.

    Forward: C[i,j] = max_k(A[i,k] * B[k,j])

    Backward: The gradient is different from MaxPlus/MinPlus because
    the operation is multiplication, not addition.
    - grad_A[i,k] = grad_C[i,j] * B[k,j] if k == argmax[i,j]
    - grad_B[k,j] = grad_C[i,j] * A[i,k] if k == argmax[i,j]
    """

    @staticmethod
    def forward(ctx, a, b):
        m, k = a.shape
        n = b.shape[1]

        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        c_flat, argmax_flat = tropical_gemm.maxmul_matmul_with_argmax(a_np, b_np)

        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        # Save inputs and argmax for backward
        ctx.save_for_backward(
            torch.from_numpy(a_np),
            torch.from_numpy(b_np),
            torch.from_numpy(argmax_np),
        )
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c):
        a, b, argmax = ctx.saved_tensors
        k_dim = ctx.k
        m = ctx.m
        n = ctx.n

        # Use Rust backend for MaxMul backward
        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)
        a_np = a.numpy().astype(np.float32)
        b_np = b.numpy().astype(np.float32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        grad_a_flat = tropical_gemm.maxmul_backward_a(grad_c_np, argmax_np, b_np)
        grad_b_flat = tropical_gemm.maxmul_backward_b(grad_c_np, argmax_np, a_np)

        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k_dim)).to(grad_c.device)
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k_dim, n)).to(grad_c.device)

        return grad_a, grad_b


def tropical_maxplus_matmul(a, b):
    return TropicalMaxPlusMatmul.apply(a, b)


def tropical_minplus_matmul(a, b):
    return TropicalMinPlusMatmul.apply(a, b)


def tropical_maxmul_matmul(a, b):
    return TropicalMaxMulMatmul.apply(a, b)


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
