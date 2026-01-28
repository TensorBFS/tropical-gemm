"""Tests for tropical_gemm Python bindings.

Tests cover all semiring types (MaxPlus, MinPlus, MaxMul) and
all scalar types (f32, f64, i32, i64).
"""

import numpy as np
import pytest
from tropical_gemm import (
    # f32 operations
    maxplus_matmul,
    minplus_matmul,
    maxmul_matmul,
    maxplus_matmul_with_argmax,
    minplus_matmul_with_argmax,
    maxmul_matmul_with_argmax,
    backward_a,
    backward_b,
    maxmul_backward_a,
    maxmul_backward_b,
    # f64 operations
    maxplus_matmul_f64,
    minplus_matmul_f64,
    maxmul_matmul_f64,
    maxplus_matmul_with_argmax_f64,
    minplus_matmul_with_argmax_f64,
    maxmul_matmul_with_argmax_f64,
    backward_a_f64,
    backward_b_f64,
    maxmul_backward_a_f64,
    maxmul_backward_b_f64,
    # i32 operations
    maxplus_matmul_i32,
    minplus_matmul_i32,
    maxmul_matmul_i32,
    # i64 operations
    maxplus_matmul_i64,
    minplus_matmul_i64,
    maxmul_matmul_i64,
)


# ============================================================================
# MaxPlus f32 tests
# ============================================================================


def test_maxplus_matmul_f32():
    """Test basic MaxPlus matmul (f32): C[i,j] = max_k(A[i,k] + B[k,j])."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    c = maxplus_matmul(a, b).reshape(2, 2)

    # C[0,0] = max(1+1, 2+3, 3+5) = max(2, 5, 8) = 8
    # C[0,1] = max(1+2, 2+4, 3+6) = max(3, 6, 9) = 9
    # C[1,0] = max(4+1, 5+3, 6+5) = max(5, 8, 11) = 11
    # C[1,1] = max(4+2, 5+4, 6+6) = max(6, 9, 12) = 12
    expected = np.array([[8.0, 9.0], [11.0, 12.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(c, expected)


def test_maxplus_matmul_with_argmax_f32():
    """Test MaxPlus matmul with argmax tracking (f32)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    c, argmax = maxplus_matmul_with_argmax(a, b)
    c = c.reshape(2, 2)
    argmax = argmax.reshape(2, 2)

    expected = np.array([[8.0, 9.0], [11.0, 12.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(c, expected)

    # All should be k=2 (third column) since 3+5, 3+6, 6+5, 6+6 are maxes
    expected_argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)
    np.testing.assert_array_equal(argmax, expected_argmax)


# ============================================================================
# MinPlus f32 tests
# ============================================================================


def test_minplus_matmul_f32():
    """Test basic MinPlus matmul (f32): C[i,j] = min_k(A[i,k] + B[k,j])."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    c = minplus_matmul(a, b).reshape(2, 2)

    # C[0,0] = min(1+1, 2+3, 3+5) = min(2, 5, 8) = 2
    # C[0,1] = min(1+2, 2+4, 3+6) = min(3, 6, 9) = 3
    # C[1,0] = min(4+1, 5+3, 6+5) = min(5, 8, 11) = 5
    # C[1,1] = min(4+2, 5+4, 6+6) = min(6, 9, 12) = 6
    expected = np.array([[2.0, 3.0], [5.0, 6.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(c, expected)


def test_minplus_matmul_with_argmax_f32():
    """Test MinPlus matmul with argmax tracking (f32)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    c, argmax = minplus_matmul_with_argmax(a, b)
    c = c.reshape(2, 2)
    argmax = argmax.reshape(2, 2)

    expected = np.array([[2.0, 3.0], [5.0, 6.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(c, expected)

    # All should be k=0 (first column) since 1+1, 1+2, 4+1, 4+2 are mins
    expected_argmax = np.array([[0, 0], [0, 0]], dtype=np.int32)
    np.testing.assert_array_equal(argmax, expected_argmax)


# ============================================================================
# MaxMul f32 tests
# ============================================================================


def test_maxmul_matmul_f32():
    """Test basic MaxMul matmul (f32): C[i,j] = max_k(A[i,k] * B[k,j])."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    c = maxmul_matmul(a, b).reshape(2, 2)

    # C[0,0] = max(1*1, 2*3, 3*5) = max(1, 6, 15) = 15
    # C[0,1] = max(1*2, 2*4, 3*6) = max(2, 8, 18) = 18
    # C[1,0] = max(4*1, 5*3, 6*5) = max(4, 15, 30) = 30
    # C[1,1] = max(4*2, 5*4, 6*6) = max(8, 20, 36) = 36
    expected = np.array([[15.0, 18.0], [30.0, 36.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(c, expected)


def test_maxmul_matmul_with_argmax_f32():
    """Test MaxMul matmul with argmax tracking (f32)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    c, argmax = maxmul_matmul_with_argmax(a, b)
    c = c.reshape(2, 2)
    argmax = argmax.reshape(2, 2)

    expected = np.array([[15.0, 18.0], [30.0, 36.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(c, expected)

    # All should be k=2 (third column) since 3*5, 3*6, 6*5, 6*6 are maxes
    expected_argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)
    np.testing.assert_array_equal(argmax, expected_argmax)


# ============================================================================
# f64 tests
# ============================================================================


def test_maxplus_matmul_f64():
    """Test MaxPlus matmul (f64)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    c = maxplus_matmul_f64(a, b).reshape(2, 2)

    expected = np.array([[8.0, 9.0], [11.0, 12.0]], dtype=np.float64)
    np.testing.assert_array_almost_equal(c, expected)


def test_minplus_matmul_f64():
    """Test MinPlus matmul (f64)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    c = minplus_matmul_f64(a, b).reshape(2, 2)

    expected = np.array([[2.0, 3.0], [5.0, 6.0]], dtype=np.float64)
    np.testing.assert_array_almost_equal(c, expected)


def test_maxmul_matmul_f64():
    """Test MaxMul matmul (f64)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    c = maxmul_matmul_f64(a, b).reshape(2, 2)

    expected = np.array([[15.0, 18.0], [30.0, 36.0]], dtype=np.float64)
    np.testing.assert_array_almost_equal(c, expected)


def test_maxplus_matmul_with_argmax_f64():
    """Test MaxPlus matmul with argmax (f64)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    c, argmax = maxplus_matmul_with_argmax_f64(a, b)
    c = c.reshape(2, 2)
    argmax = argmax.reshape(2, 2)

    expected = np.array([[8.0, 9.0], [11.0, 12.0]], dtype=np.float64)
    np.testing.assert_array_almost_equal(c, expected)
    expected_argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)
    np.testing.assert_array_equal(argmax, expected_argmax)


def test_minplus_matmul_with_argmax_f64():
    """Test MinPlus matmul with argmax (f64)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    c, argmax = minplus_matmul_with_argmax_f64(a, b)
    c = c.reshape(2, 2)
    argmax = argmax.reshape(2, 2)

    expected = np.array([[2.0, 3.0], [5.0, 6.0]], dtype=np.float64)
    np.testing.assert_array_almost_equal(c, expected)
    expected_argmax = np.array([[0, 0], [0, 0]], dtype=np.int32)
    np.testing.assert_array_equal(argmax, expected_argmax)


def test_maxmul_matmul_with_argmax_f64():
    """Test MaxMul matmul with argmax (f64)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    c, argmax = maxmul_matmul_with_argmax_f64(a, b)
    c = c.reshape(2, 2)
    argmax = argmax.reshape(2, 2)

    expected = np.array([[15.0, 18.0], [30.0, 36.0]], dtype=np.float64)
    np.testing.assert_array_almost_equal(c, expected)
    expected_argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)
    np.testing.assert_array_equal(argmax, expected_argmax)


def test_backward_a_f64():
    """Test backward pass for gradient w.r.t. A (f64)."""
    m, k, n = 2, 3, 2

    grad_c = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)

    grad_a = backward_a_f64(grad_c, argmax, k).reshape(m, k)

    expected = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 7.0]], dtype=np.float64)
    np.testing.assert_array_almost_equal(grad_a, expected)


def test_backward_b_f64():
    """Test backward pass for gradient w.r.t. B (f64)."""
    m, k, n = 2, 3, 2

    grad_c = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)

    grad_b = backward_b_f64(grad_c, argmax, k).reshape(k, n)

    expected = np.array([[0.0, 0.0], [0.0, 0.0], [4.0, 6.0]], dtype=np.float64)
    np.testing.assert_array_almost_equal(grad_b, expected)


# ============================================================================
# MaxMul backward tests (different from MaxPlus/MinPlus)
# ============================================================================


def test_maxmul_backward_a():
    """Test MaxMul backward pass for gradient w.r.t. A (f32).

    For MaxMul: C[i,j] = max_k(A[i,k] * B[k,j])
    grad_A[i,k] = sum_j { grad_C[i,j] * B[k,j] if argmax[i,j] == k }
    """
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    # All argmax are k=2
    grad_c = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)

    grad_a = maxmul_backward_a(grad_c, argmax, b).reshape(2, 3)

    # grad_A[0,2] = grad_C[0,0]*B[2,0] + grad_C[0,1]*B[2,1] = 1*5 + 1*6 = 11
    # grad_A[1,2] = grad_C[1,0]*B[2,0] + grad_C[1,1]*B[2,1] = 1*5 + 1*6 = 11
    expected = np.array([[0.0, 0.0, 11.0], [0.0, 0.0, 11.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(grad_a, expected)


def test_maxmul_backward_b():
    """Test MaxMul backward pass for gradient w.r.t. B (f32).

    For MaxMul: C[i,j] = max_k(A[i,k] * B[k,j])
    grad_B[k,j] = sum_i { grad_C[i,j] * A[i,k] if argmax[i,j] == k }
    """
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    grad_c = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)

    grad_b = maxmul_backward_b(grad_c, argmax, a).reshape(3, 2)

    # grad_B[2,0] = grad_C[0,0]*A[0,2] + grad_C[1,0]*A[1,2] = 1*3 + 1*6 = 9
    # grad_B[2,1] = grad_C[0,1]*A[0,2] + grad_C[1,1]*A[1,2] = 1*3 + 1*6 = 9
    expected = np.array([[0.0, 0.0], [0.0, 0.0], [9.0, 9.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(grad_b, expected)


def test_maxmul_backward_a_f64():
    """Test MaxMul backward pass for gradient w.r.t. A (f64)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    grad_c = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)

    grad_a = maxmul_backward_a_f64(grad_c, argmax, b).reshape(2, 3)

    expected = np.array([[0.0, 0.0, 11.0], [0.0, 0.0, 11.0]], dtype=np.float64)
    np.testing.assert_array_almost_equal(grad_a, expected)


def test_maxmul_backward_b_f64():
    """Test MaxMul backward pass for gradient w.r.t. B (f64)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    grad_c = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)

    grad_b = maxmul_backward_b_f64(grad_c, argmax, a).reshape(3, 2)

    expected = np.array([[0.0, 0.0], [0.0, 0.0], [9.0, 9.0]], dtype=np.float64)
    np.testing.assert_array_almost_equal(grad_b, expected)


# ============================================================================
# i32 tests
# ============================================================================


def test_maxplus_matmul_i32():
    """Test MaxPlus matmul (i32)."""
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)

    c = maxplus_matmul_i32(a, b).reshape(2, 2)

    expected = np.array([[8, 9], [11, 12]], dtype=np.int32)
    np.testing.assert_array_equal(c, expected)


def test_minplus_matmul_i32():
    """Test MinPlus matmul (i32)."""
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)

    c = minplus_matmul_i32(a, b).reshape(2, 2)

    expected = np.array([[2, 3], [5, 6]], dtype=np.int32)
    np.testing.assert_array_equal(c, expected)


def test_maxmul_matmul_i32():
    """Test MaxMul matmul (i32)."""
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)

    c = maxmul_matmul_i32(a, b).reshape(2, 2)

    expected = np.array([[15, 18], [30, 36]], dtype=np.int32)
    np.testing.assert_array_equal(c, expected)


# ============================================================================
# i64 tests
# ============================================================================


def test_maxplus_matmul_i64():
    """Test MaxPlus matmul (i64)."""
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)

    c = maxplus_matmul_i64(a, b).reshape(2, 2)

    expected = np.array([[8, 9], [11, 12]], dtype=np.int64)
    np.testing.assert_array_equal(c, expected)


def test_minplus_matmul_i64():
    """Test MinPlus matmul (i64)."""
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)

    c = minplus_matmul_i64(a, b).reshape(2, 2)

    expected = np.array([[2, 3], [5, 6]], dtype=np.int64)
    np.testing.assert_array_equal(c, expected)


def test_maxmul_matmul_i64():
    """Test MaxMul matmul (i64)."""
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)

    c = maxmul_matmul_i64(a, b).reshape(2, 2)

    expected = np.array([[15, 18], [30, 36]], dtype=np.int64)
    np.testing.assert_array_equal(c, expected)


# ============================================================================
# Backward pass tests (f32 only, gradients don't apply to integers)
# ============================================================================


def test_backward_a():
    """Test backward pass for gradient w.r.t. A."""
    m, k, n = 2, 3, 2

    grad_c = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    # argmax indicates which k produced each output
    argmax = np.array([[2, 2], [2, 2]], dtype=np.int32)

    grad_a = backward_a(grad_c, argmax, k).reshape(m, k)

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

    grad_b = backward_b(grad_c, argmax, k).reshape(k, n)

    # grad_b[k,j] = sum_i { grad_c[i,j] if argmax[i,j] == k }
    # Since all argmax are 2:
    # grad_b[2,0] = grad_c[0,0] + grad_c[1,0] = 1 + 3 = 4
    # grad_b[2,1] = grad_c[0,1] + grad_c[1,1] = 2 + 4 = 6
    expected = np.array([[0.0, 0.0], [0.0, 0.0], [4.0, 6.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(grad_b, expected)


# ============================================================================
# Error handling tests
# ============================================================================


def test_dimension_mismatch():
    """Test that dimension mismatch raises error."""
    a = np.array([[1.0, 2.0]], dtype=np.float32)  # 1x2
    b = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)  # 3x1

    with pytest.raises(ValueError, match="Dimension mismatch"):
        maxplus_matmul(a, b)


def test_dimension_mismatch_i32():
    """Test dimension mismatch with i32."""
    a = np.array([[1, 2]], dtype=np.int32)
    b = np.array([[1], [2], [3]], dtype=np.int32)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        maxplus_matmul_i32(a, b)


# ============================================================================
# Edge case tests
# ============================================================================


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
    assert c[0] == 8.0  # 5 + 3

    c = maxmul_matmul(a, b)
    assert c[0] == 15.0  # 5 * 3


def test_negative_values():
    """Test with negative values."""
    a = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    b = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)

    c = maxplus_matmul(a, b).reshape(2, 2)
    # C[0,0] = max(-1+-1, -2+-3) = max(-2, -5) = -2
    assert c[0, 0] == -2.0


def test_negative_integers():
    """Test with negative integer values."""
    a = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    b = np.array([[-1, -2], [-3, -4]], dtype=np.int32)

    c = maxplus_matmul_i32(a, b).reshape(2, 2)
    assert c[0, 0] == -2


def test_infinity_handling():
    """Test with infinity values (tropical zero for MinPlus)."""
    inf = np.float32("inf")
    a = np.array([[0.0, inf], [inf, 0.0]], dtype=np.float32)
    b = np.array([[0.0, inf], [inf, 0.0]], dtype=np.float32)

    c = minplus_matmul(a, b).reshape(2, 2)
    # MinPlus: C[i,j] = min_k(A[i,k] + B[k,j])
    # C[0,0] = min(0+0, inf+inf) = 0
    assert c[0, 0] == 0.0


def test_maxmul_with_zeros():
    """Test MaxMul with zeros (tropical zero)."""
    a = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)

    c = maxmul_matmul(a, b).reshape(2, 2)
    # MaxMul: C[i,j] = max_k(A[i,k] * B[k,j])
    # C[0,0] = max(0*0, 1*2) = max(0, 2) = 2
    # C[0,1] = max(0*1, 1*0) = max(0, 0) = 0
    # C[1,0] = max(2*0, 0*2) = max(0, 0) = 0
    # C[1,1] = max(2*1, 0*0) = max(2, 0) = 2
    expected = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(c, expected)


def test_larger_matrix():
    """Test with a larger matrix to verify correctness at scale."""
    n = 16
    np.random.seed(42)
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)

    c = maxplus_matmul(a, b).reshape(n, n)

    # Verify a few elements manually
    for i in range(min(3, n)):
        for j in range(min(3, n)):
            expected = np.max(a[i, :] + b[:, j])
            np.testing.assert_almost_equal(c[i, j], expected, decimal=5)


# ============================================================================
# Strided batched operations tests
# ============================================================================


def test_maxplus_matmul_strided_batched():
    """Test batched MaxPlus matmul."""
    from tropical_gemm import maxplus_matmul_strided_batched

    batch_size = 2
    m, k, n = 2, 3, 2

    # Create batched input: (batch, m, k) and (batch, k, n)
    a = np.array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # batch 0
        [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],  # batch 1
    ], dtype=np.float32)

    b = np.array([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # batch 0
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # batch 1
    ], dtype=np.float32)

    c_flat = maxplus_matmul_strided_batched(a, b)
    c = c_flat.reshape(batch_size, m, n)

    # Batch 0: same as single matrix test
    # C[0,0] = max(1+1, 2+3, 3+5) = 8
    assert c[0, 0, 0] == 8.0
    # C[0,1] = max(4+2, 5+4, 6+6) = 12
    assert c[0, 1, 1] == 12.0

    # Batch 1: shifted by 1
    # C[0,0] = max(2+1, 3+3, 4+5) = 9
    assert c[1, 0, 0] == 9.0


def test_minplus_matmul_strided_batched():
    """Test batched MinPlus matmul."""
    from tropical_gemm import minplus_matmul_strided_batched

    batch_size = 2
    m, k, n = 2, 3, 2

    a = np.array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
    ], dtype=np.float32)

    b = np.array([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    ], dtype=np.float32)

    c_flat = minplus_matmul_strided_batched(a, b)
    c = c_flat.reshape(batch_size, m, n)

    # Batch 0:
    # C[0,0] = min(1+1, 2+3, 3+5) = 2
    assert c[0, 0, 0] == 2.0

    # Batch 1:
    # C[0,0] = min(2+1, 3+3, 4+5) = 3
    assert c[1, 0, 0] == 3.0


def test_maxmul_matmul_strided_batched():
    """Test batched MaxMul matmul."""
    from tropical_gemm import maxmul_matmul_strided_batched

    batch_size = 2
    m, k, n = 2, 2, 2

    a = np.array([
        [[2.0, 3.0], [4.0, 5.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ], dtype=np.float32)

    b = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ], dtype=np.float32)

    c_flat = maxmul_matmul_strided_batched(a, b)
    c = c_flat.reshape(batch_size, m, n)

    # Batch 0:
    # C[0,0] = max(2*1, 3*3) = max(2, 9) = 9
    assert c[0, 0, 0] == 9.0
    # C[0,1] = max(2*2, 3*4) = max(4, 12) = 12
    assert c[0, 0, 1] == 12.0


def test_maxplus_matmul_strided_batched_with_argmax():
    """Test batched MaxPlus matmul with argmax tracking."""
    from tropical_gemm import maxplus_matmul_strided_batched_with_argmax

    batch_size = 2
    m, k, n = 2, 3, 2

    a = np.array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[1.0, 2.0, 10.0], [4.0, 5.0, 6.0]],  # batch 1 has k=2 winning for [0,0]
    ], dtype=np.float32)

    b = np.array([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    ], dtype=np.float32)

    c_flat, argmax_flat = maxplus_matmul_strided_batched_with_argmax(a, b)
    c = c_flat.reshape(batch_size, m, n)
    argmax = argmax_flat.reshape(batch_size, m, n)

    # Batch 0: all argmax should be 2 (k=2 wins for all)
    assert argmax[0, 0, 0] == 2
    assert argmax[0, 1, 1] == 2

    # Batch 1: with a[0,2]=10, C[0,0] = max(1+1, 2+3, 10+5) = 15, argmax=2
    assert c[1, 0, 0] == 15.0
    assert argmax[1, 0, 0] == 2


def test_backward_a_strided_batched():
    """Test batched backward pass for gradient w.r.t. A."""
    from tropical_gemm import backward_a_strided_batched

    batch_size = 2
    m, k, n = 2, 3, 2

    grad_c = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, 1.0], [1.0, 1.0]],
    ], dtype=np.float32)

    # All argmax are k=2
    argmax = np.array([
        [[2, 2], [2, 2]],
        [[2, 2], [2, 2]],
    ], dtype=np.int32)

    grad_a_flat = backward_a_strided_batched(grad_c, argmax, k)
    grad_a = grad_a_flat.reshape(batch_size, m, k)

    # Batch 0:
    # grad_a[0,0,2] = grad_c[0,0,0] + grad_c[0,0,1] = 1 + 2 = 3
    # grad_a[0,1,2] = grad_c[0,1,0] + grad_c[0,1,1] = 3 + 4 = 7
    assert grad_a[0, 0, 2] == 3.0
    assert grad_a[0, 1, 2] == 7.0
    assert grad_a[0, 0, 0] == 0.0
    assert grad_a[0, 0, 1] == 0.0

    # Batch 1:
    # grad_a[1,0,2] = 1 + 1 = 2
    # grad_a[1,1,2] = 1 + 1 = 2
    assert grad_a[1, 0, 2] == 2.0
    assert grad_a[1, 1, 2] == 2.0


def test_backward_b_strided_batched():
    """Test batched backward pass for gradient w.r.t. B."""
    from tropical_gemm import backward_b_strided_batched

    batch_size = 2
    m, k, n = 2, 3, 2

    grad_c = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, 1.0], [1.0, 1.0]],
    ], dtype=np.float32)

    argmax = np.array([
        [[2, 2], [2, 2]],
        [[2, 2], [2, 2]],
    ], dtype=np.int32)

    grad_b_flat = backward_b_strided_batched(grad_c, argmax, k)
    grad_b = grad_b_flat.reshape(batch_size, k, n)

    # Batch 0:
    # grad_b[0,2,0] = grad_c[0,0,0] + grad_c[0,1,0] = 1 + 3 = 4
    # grad_b[0,2,1] = grad_c[0,0,1] + grad_c[0,1,1] = 2 + 4 = 6
    assert grad_b[0, 2, 0] == 4.0
    assert grad_b[0, 2, 1] == 6.0
    assert grad_b[0, 0, 0] == 0.0
    assert grad_b[0, 1, 0] == 0.0


def test_maxmul_backward_strided_batched():
    """Test batched MaxMul backward pass."""
    from tropical_gemm import (
        maxmul_backward_a_strided_batched,
        maxmul_backward_b_strided_batched,
    )

    batch_size = 2
    m, k, n = 2, 3, 2

    a = np.array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    ], dtype=np.float32)

    b = np.array([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    ], dtype=np.float32)

    grad_c = np.array([
        [[1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0]],
    ], dtype=np.float32)

    # All argmax are k=2
    argmax = np.array([
        [[2, 2], [2, 2]],
        [[2, 2], [2, 2]],
    ], dtype=np.int32)

    grad_a_flat = maxmul_backward_a_strided_batched(grad_c, argmax, b)
    grad_b_flat = maxmul_backward_b_strided_batched(grad_c, argmax, a)

    grad_a = grad_a_flat.reshape(batch_size, m, k)
    grad_b = grad_b_flat.reshape(batch_size, k, n)

    # Batch 0:
    # grad_A[0,0,2] = grad_C[0,0,0]*B[0,2,0] + grad_C[0,0,1]*B[0,2,1] = 1*5 + 1*6 = 11
    assert grad_a[0, 0, 2] == 11.0
    assert grad_a[0, 1, 2] == 11.0

    # grad_B[0,2,0] = grad_C[0,0,0]*A[0,0,2] + grad_C[0,1,0]*A[0,1,2] = 1*3 + 1*6 = 9
    assert grad_b[0, 2, 0] == 9.0
    assert grad_b[0, 2, 1] == 9.0


def test_strided_batched_dimension_mismatch():
    """Test that dimension mismatch raises error."""
    from tropical_gemm import maxplus_matmul_strided_batched

    a = np.random.randn(2, 3, 4).astype(np.float32)  # batch=2, m=3, k=4
    b = np.random.randn(3, 5, 2).astype(np.float32)  # batch=3 (mismatch!)

    with pytest.raises(ValueError, match="Batch size mismatch"):
        maxplus_matmul_strided_batched(a, b)
