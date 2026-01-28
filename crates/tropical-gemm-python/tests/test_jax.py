"""
Tests for JAX integration of tropical matrix multiplication.
"""

import pytest
import numpy as np

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax import grad, jit, vmap

from tropical_gemm.jax import (
    tropical_maxplus_matmul,
    tropical_minplus_matmul,
    tropical_maxmul_matmul,
    tropical_maxplus_matmul_batched,
    tropical_minplus_matmul_batched,
    tropical_maxmul_matmul_batched,
)


# ===========================================================================
# Naive implementations for reference
# ===========================================================================

def naive_maxplus_2d(a, b):
    return (a[:, :, None] + b[None, :, :]).max(axis=1)

def naive_minplus_2d(a, b):
    return (a[:, :, None] + b[None, :, :]).min(axis=1)

def naive_maxmul_2d(a, b):
    return (a[:, :, None] * b[None, :, :]).max(axis=1)

def naive_maxplus_3d(a, b):
    return (a[:, :, :, None] + b[:, None, :, :]).max(axis=2)

def naive_minplus_3d(a, b):
    return (a[:, :, :, None] + b[:, None, :, :]).min(axis=2)

def naive_maxmul_3d(a, b):
    return (a[:, :, :, None] * b[:, None, :, :]).max(axis=2)


# ===========================================================================
# 2D Forward Tests
# ===========================================================================

class TestMaxPlus2D:
    def test_basic(self):
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        result = tropical_maxplus_matmul(a, b)
        expected = naive_maxplus_2d(a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_random(self):
        key = jax.random.PRNGKey(42)
        a = jax.random.normal(key, (32, 64))
        b = jax.random.normal(jax.random.PRNGKey(43), (64, 48))

        result = tropical_maxplus_matmul(a, b)
        expected = naive_maxplus_2d(a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_shape(self):
        a = jnp.ones((100, 50))
        b = jnp.ones((50, 80))

        result = tropical_maxplus_matmul(a, b)
        assert result.shape == (100, 80)


class TestMinPlus2D:
    def test_basic(self):
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        result = tropical_minplus_matmul(a, b)
        expected = naive_minplus_2d(a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_random(self):
        key = jax.random.PRNGKey(42)
        a = jax.random.normal(key, (32, 64))
        b = jax.random.normal(jax.random.PRNGKey(43), (64, 48))

        result = tropical_minplus_matmul(a, b)
        expected = naive_minplus_2d(a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestMaxMul2D:
    def test_basic(self):
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        result = tropical_maxmul_matmul(a, b)
        expected = naive_maxmul_2d(a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_positive_values(self):
        key = jax.random.PRNGKey(42)
        a = jnp.abs(jax.random.normal(key, (32, 64))) + 0.1
        b = jnp.abs(jax.random.normal(jax.random.PRNGKey(43), (64, 48))) + 0.1

        result = tropical_maxmul_matmul(a, b)
        expected = naive_maxmul_2d(a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-5)


# ===========================================================================
# 3D Batched Forward Tests
# ===========================================================================

class TestMaxPlusBatched:
    def test_basic(self):
        key = jax.random.PRNGKey(42)
        a = jax.random.normal(key, (4, 16, 32))
        b = jax.random.normal(jax.random.PRNGKey(43), (4, 32, 24))

        result = tropical_maxplus_matmul_batched(a, b)
        expected = naive_maxplus_3d(a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_shape(self):
        a = jnp.ones((8, 32, 64))
        b = jnp.ones((8, 64, 48))

        result = tropical_maxplus_matmul_batched(a, b)
        assert result.shape == (8, 32, 48)


class TestMinPlusBatched:
    def test_basic(self):
        key = jax.random.PRNGKey(42)
        a = jax.random.normal(key, (4, 16, 32))
        b = jax.random.normal(jax.random.PRNGKey(43), (4, 32, 24))

        result = tropical_minplus_matmul_batched(a, b)
        expected = naive_minplus_3d(a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestMaxMulBatched:
    def test_basic(self):
        key = jax.random.PRNGKey(42)
        a = jnp.abs(jax.random.normal(key, (4, 16, 32))) + 0.1
        b = jnp.abs(jax.random.normal(jax.random.PRNGKey(43), (4, 32, 24))) + 0.1

        result = tropical_maxmul_matmul_batched(a, b)
        expected = naive_maxmul_3d(a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-5)


# ===========================================================================
# Gradient Tests
# ===========================================================================

class TestMaxPlusGradient:
    def test_grad_exists(self):
        """Test that gradients can be computed."""
        def loss_fn(a, b):
            return tropical_maxplus_matmul(a, b).sum()

        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        grad_a = grad(loss_fn, argnums=0)(a, b)
        grad_b = grad(loss_fn, argnums=1)(a, b)

        assert grad_a.shape == a.shape
        assert grad_b.shape == b.shape

    def test_grad_structure(self):
        """Test gradient structure: total gradient equals number of outputs."""
        def loss_fn(a, b):
            return tropical_maxplus_matmul(a, b).sum()

        m, k, n = 8, 16, 12
        a = jax.random.normal(jax.random.PRNGKey(42), (m, k))
        b = jax.random.normal(jax.random.PRNGKey(43), (k, n))

        grad_a = grad(loss_fn, argnums=0)(a, b)
        grad_b = grad(loss_fn, argnums=1)(a, b)

        # Each output element has exactly one argmax, gradient flows to both A and B
        # Total gradient should equal 2*M*N (each output contributes 1 to grad_a and 1 to grad_b)
        total_grad = grad_a.sum() + grad_b.sum()
        np.testing.assert_allclose(total_grad, 2 * m * n, rtol=1e-5)

    def test_grad_nonnegative(self):
        """Gradients should be non-negative for sum loss."""
        def loss_fn(a, b):
            return tropical_maxplus_matmul(a, b).sum()

        a = jax.random.normal(jax.random.PRNGKey(42), (16, 32))
        b = jax.random.normal(jax.random.PRNGKey(43), (32, 24))

        grad_a = grad(loss_fn, argnums=0)(a, b)
        grad_b = grad(loss_fn, argnums=1)(a, b)

        assert jnp.all(grad_a >= 0)
        assert jnp.all(grad_b >= 0)


class TestMinPlusGradient:
    def test_grad_exists(self):
        def loss_fn(a, b):
            return tropical_minplus_matmul(a, b).sum()

        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        grad_a = grad(loss_fn, argnums=0)(a, b)
        grad_b = grad(loss_fn, argnums=1)(a, b)

        assert grad_a.shape == a.shape
        assert grad_b.shape == b.shape


class TestMaxMulGradient:
    def test_grad_exists(self):
        def loss_fn(a, b):
            return tropical_maxmul_matmul(a, b).sum()

        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        grad_a = grad(loss_fn, argnums=0)(a, b)
        grad_b = grad(loss_fn, argnums=1)(a, b)

        assert grad_a.shape == a.shape
        assert grad_b.shape == b.shape


class TestBatchedGradient:
    def test_maxplus_batched_grad(self):
        def loss_fn(a, b):
            return tropical_maxplus_matmul_batched(a, b).sum()

        a = jax.random.normal(jax.random.PRNGKey(42), (4, 16, 32))
        b = jax.random.normal(jax.random.PRNGKey(43), (4, 32, 24))

        grad_a = grad(loss_fn, argnums=0)(a, b)
        grad_b = grad(loss_fn, argnums=1)(a, b)

        assert grad_a.shape == a.shape
        assert grad_b.shape == b.shape

    def test_minplus_batched_grad(self):
        def loss_fn(a, b):
            return tropical_minplus_matmul_batched(a, b).sum()

        a = jax.random.normal(jax.random.PRNGKey(42), (4, 16, 32))
        b = jax.random.normal(jax.random.PRNGKey(43), (4, 32, 24))

        grad_a = grad(loss_fn, argnums=0)(a, b)
        grad_b = grad(loss_fn, argnums=1)(a, b)

        assert grad_a.shape == a.shape
        assert grad_b.shape == b.shape

    def test_maxmul_batched_grad(self):
        def loss_fn(a, b):
            return tropical_maxmul_matmul_batched(a, b).sum()

        a = jnp.abs(jax.random.normal(jax.random.PRNGKey(42), (4, 16, 32))) + 0.1
        b = jnp.abs(jax.random.normal(jax.random.PRNGKey(43), (4, 32, 24))) + 0.1

        grad_a = grad(loss_fn, argnums=0)(a, b)
        grad_b = grad(loss_fn, argnums=1)(a, b)

        assert grad_a.shape == a.shape
        assert grad_b.shape == b.shape


# ===========================================================================
# JIT Compilation Tests
# ===========================================================================

class TestJIT:
    def test_maxplus_jit(self):
        """Test that JIT compilation works."""
        @jit
        def compute(a, b):
            return tropical_maxplus_matmul(a, b)

        a = jax.random.normal(jax.random.PRNGKey(42), (32, 64))
        b = jax.random.normal(jax.random.PRNGKey(43), (64, 48))

        result = compute(a, b)
        expected = naive_maxplus_2d(a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_batched_jit(self):
        @jit
        def compute(a, b):
            return tropical_maxplus_matmul_batched(a, b)

        a = jax.random.normal(jax.random.PRNGKey(42), (4, 32, 64))
        b = jax.random.normal(jax.random.PRNGKey(43), (4, 64, 48))

        result = compute(a, b)
        expected = naive_maxplus_3d(a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_grad_jit(self):
        """Test that JIT-compiled gradient works."""
        @jit
        def loss_and_grad(a, b):
            def loss_fn(a, b):
                return tropical_maxplus_matmul(a, b).sum()
            return loss_fn(a, b), grad(loss_fn, argnums=(0, 1))(a, b)

        a = jax.random.normal(jax.random.PRNGKey(42), (16, 32))
        b = jax.random.normal(jax.random.PRNGKey(43), (32, 24))

        loss, (grad_a, grad_b) = loss_and_grad(a, b)

        assert grad_a.shape == a.shape
        assert grad_b.shape == b.shape
