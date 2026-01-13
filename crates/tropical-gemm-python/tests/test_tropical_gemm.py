"""Tests for tropical_gemm Python bindings."""

import numpy as np
import pytest
from tropical_gemm import (
    maxplus_matmul,
    minplus_matmul,
    maxplus_matmul_with_argmax,
    minplus_matmul_with_argmax,
    backward_a,
    backward_b,
)


def test_maxplus_matmul_basic():
    """Test basic MaxPlus matmul: C[i,j] = max_k(A[i,k] + B[k,j])."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    c = maxplus_matmul(a, b)
    c = c.reshape(2, 2)

    # C[0,0] = max(1+1, 2+3, 3+5) = max(2, 5, 8) = 8
    # C[0,1] = max(1+2, 2+4, 3+6) = max(3, 6, 9) = 9
    # C[1,0] = max(4+1, 5+3, 6+5) = max(5, 8, 11) = 11
    # C[1,1] = max(4+2, 5+4, 6+6) = max(6, 9, 12) = 12
    expected = np.array([[8.0, 9.0], [11.0, 12.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(c, expected)


def test_minplus_matmul_basic():
    """Test basic MinPlus matmul: C[i,j] = min_k(A[i,k] + B[k,j])."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    c = minplus_matmul(a, b)
    c = c.reshape(2, 2)

    # C[0,0] = min(1+1, 2+3, 3+5) = min(2, 5, 8) = 2
    # C[0,1] = min(1+2, 2+4, 3+6) = min(3, 6, 9) = 3
    # C[1,0] = min(4+1, 5+3, 6+5) = min(5, 8, 11) = 5
    # C[1,1] = min(4+2, 5+4, 6+6) = min(6, 9, 12) = 6
    expected = np.array([[2.0, 3.0], [5.0, 6.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(c, expected)


def test_maxplus_matmul_with_argmax():
    """Test MaxPlus matmul with argmax tracking."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    c, argmax = maxplus_matmul_with_argmax(a, b)
    c = c.reshape(2, 2)
    argmax = argmax.reshape(2, 2)

    # Values should match maxplus_matmul
    expected = np.array([[8.0, 9.0], [11.0, 12.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(c, expected)

    # argmax[i,j] = k that gave the max value
    # All should be k=2 (third column) since 3+5, 3+6, 6+5, 6+6 are maxes
    expected_argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)
    np.testing.assert_array_equal(argmax, expected_argmax)


def test_minplus_matmul_with_argmax():
    """Test MinPlus matmul with argmax tracking."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    c, argmax = minplus_matmul_with_argmax(a, b)
    c = c.reshape(2, 2)
    argmax = argmax.reshape(2, 2)

    # Values should match minplus_matmul
    expected = np.array([[2.0, 3.0], [5.0, 6.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(c, expected)

    # argmax[i,j] = k that gave the min value
    # All should be k=0 (first column) since 1+1, 1+2, 4+1, 4+2 are mins
    expected_argmax = np.array([[0, 0], [0, 0]], dtype=np.int32)
    np.testing.assert_array_equal(argmax, expected_argmax)


def test_backward_a():
    """Test backward pass for gradient w.r.t. A."""
    m, k, n = 2, 3, 2

    grad_c = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    # argmax indicates which k produced each output
    argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)

    grad_a = backward_a(grad_c, argmax, k)
    grad_a = grad_a.reshape(m, k)

    # grad_a[i,k] = sum_j { grad_c[i,j] if argmax[i,j] == k }
    # Since all argmax are 2:
    # grad_a[0,2] = grad_c[0,0] + grad_c[0,1] = 1 + 2 = 3
    # grad_a[1,2] = grad_c[1,0] + grad_c[1,1] = 3 + 4 = 7
    expected = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 7.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(grad_a, expected)


def test_backward_b():
    """Test backward pass for gradient w.r.t. B."""
    m, k, n = 2, 3, 2

    grad_c = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)

    grad_b = backward_b(grad_c, argmax, k)
    grad_b = grad_b.reshape(k, n)

    # grad_b[k,j] = sum_i { grad_c[i,j] if argmax[i,j] == k }
    # Since all argmax are 2:
    # grad_b[2,0] = grad_c[0,0] + grad_c[1,0] = 1 + 3 = 4
    # grad_b[2,1] = grad_c[0,1] + grad_c[1,1] = 2 + 4 = 6
    expected = np.array([[0.0, 0.0], [0.0, 0.0], [4.0, 6.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(grad_b, expected)


def test_dimension_mismatch():
    """Test that dimension mismatch raises error."""
    a = np.array([[1.0, 2.0]], dtype=np.float32)  # 1x2
    b = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)  # 3x1

    with pytest.raises(ValueError, match="Dimension mismatch"):
        maxplus_matmul(a, b)


def test_square_matrix():
    """Test with square matrices."""
    n = 4
    a = np.arange(n * n, dtype=np.float32).reshape(n, n)
    b = np.arange(n * n, dtype=np.float32).reshape(n, n)

    c = maxplus_matmul(a, b)
    assert c.shape == (n * n,)

    c, argmax = maxplus_matmul_with_argmax(a, b)
    assert c.shape == (n * n,)
    assert argmax.shape == (n * n,)


def test_single_element():
    """Test with 1x1 matrices."""
    a = np.array([[5.0]], dtype=np.float32)
    b = np.array([[3.0]], dtype=np.float32)

    c = maxplus_matmul(a, b)
    assert c[0] == 8.0  # 5 + 3

    c = minplus_matmul(a, b)
    assert c[0] == 8.0  # 5 + 3 (same for single element)


def test_negative_values():
    """Test with negative values."""
    a = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    b = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)

    c = maxplus_matmul(a, b)
    c = c.reshape(2, 2)

    # C[0,0] = max(-1+-1, -2+-3) = max(-2, -5) = -2
    assert c[0, 0] == -2.0


def test_infinity_handling():
    """Test with infinity values (tropical zero)."""
    inf = np.float32('inf')
    a = np.array([[0.0, inf], [inf, 0.0]], dtype=np.float32)
    b = np.array([[0.0, inf], [inf, 0.0]], dtype=np.float32)

    c = minplus_matmul(a, b)
    c = c.reshape(2, 2)

    # MinPlus: C[i,j] = min_k(A[i,k] + B[k,j])
    # C[0,0] = min(0+0, inf+inf) = 0
    # C[0,1] = min(0+inf, inf+0) = 0 (inf is tropical zero)
    assert c[0, 0] == 0.0
