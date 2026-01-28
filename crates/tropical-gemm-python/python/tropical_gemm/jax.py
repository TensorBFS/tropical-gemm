"""
JAX integration for tropical matrix multiplication.

This module provides JAX-compatible functions with custom VJP (vector-Jacobian product)
support for automatic differentiation.

Tropical semirings are useful for:
- Shortest path problems (MinPlus)
- Longest path problems (MaxPlus)
- Dynamic programming on graphs
- Probabilistic inference (log-space operations)

Example:
    >>> import jax.numpy as jnp
    >>> from tropical_gemm.jax import tropical_maxplus_matmul
    >>>
    >>> a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> b = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    >>> c = tropical_maxplus_matmul(a, b)  # c[i,j] = max_k(a[i,k] + b[k,j])
"""

try:
    import jax
    import jax.numpy as jnp
    from jax import custom_vjp
except ImportError:
    raise ImportError(
        "JAX is required for this module. "
        "Install it with: pip install jax jaxlib"
    )

import numpy as np
import tropical_gemm


# ===========================================================================
# Helper functions to call Rust backend
# ===========================================================================

def _call_maxplus_with_argmax(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Call Rust MaxPlus matmul with argmax tracking."""
    m, k = a.shape
    n = b.shape[1]

    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    c_flat, argmax_flat = tropical_gemm.maxplus_matmul_with_argmax(a, b)

    c = np.array(c_flat, dtype=np.float32).reshape(m, n)
    argmax = np.array(argmax_flat, dtype=np.int32).reshape(m, n)

    return c, argmax


def _call_minplus_with_argmax(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Call Rust MinPlus matmul with argmax tracking."""
    m, k = a.shape
    n = b.shape[1]

    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    c_flat, argmax_flat = tropical_gemm.minplus_matmul_with_argmax(a, b)

    c = np.array(c_flat, dtype=np.float32).reshape(m, n)
    argmax = np.array(argmax_flat, dtype=np.int32).reshape(m, n)

    return c, argmax


def _call_maxmul_with_argmax(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Call Rust MaxMul matmul with argmax tracking."""
    m, k = a.shape
    n = b.shape[1]

    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    c_flat, argmax_flat = tropical_gemm.maxmul_matmul_with_argmax(a, b)

    c = np.array(c_flat, dtype=np.float32).reshape(m, n)
    argmax = np.array(argmax_flat, dtype=np.int32).reshape(m, n)

    return c, argmax


# Batched versions
def _call_maxplus_batched_with_argmax(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Call Rust batched MaxPlus matmul with argmax tracking."""
    batch, m, k = a.shape
    n = b.shape[2]

    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    c_flat, argmax_flat = tropical_gemm.maxplus_matmul_strided_batched_with_argmax(a, b)

    c = np.array(c_flat, dtype=np.float32).reshape(batch, m, n)
    argmax = np.array(argmax_flat, dtype=np.int32).reshape(batch, m, n)

    return c, argmax


def _call_minplus_batched_with_argmax(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Call Rust batched MinPlus matmul with argmax tracking."""
    batch, m, k = a.shape
    n = b.shape[2]

    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    c_flat, argmax_flat = tropical_gemm.minplus_matmul_strided_batched_with_argmax(a, b)

    c = np.array(c_flat, dtype=np.float32).reshape(batch, m, n)
    argmax = np.array(argmax_flat, dtype=np.int32).reshape(batch, m, n)

    return c, argmax


def _call_maxmul_batched_with_argmax(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Call Rust batched MaxMul matmul with argmax tracking."""
    batch, m, k = a.shape
    n = b.shape[2]

    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    c_flat, argmax_flat = tropical_gemm.maxmul_matmul_strided_batched_with_argmax(a, b)

    c = np.array(c_flat, dtype=np.float32).reshape(batch, m, n)
    argmax = np.array(argmax_flat, dtype=np.int32).reshape(batch, m, n)

    return c, argmax


# ===========================================================================
# 2D Operations with custom_vjp
# ===========================================================================

@custom_vjp
def tropical_maxplus_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Tropical max-plus matrix multiplication: C[i,j] = max_k(A[i,k] + B[k,j])

    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)

    Returns:
        Output matrix C of shape (M, N)
    """
    def impl(a, b):
        c, _ = _call_maxplus_with_argmax(np.asarray(a), np.asarray(b))
        return c

    return jax.pure_callback(
        impl,
        jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.float32),
        a, b
    )


def _maxplus_fwd(a, b):
    """Forward pass: compute result and save argmax for backward."""
    def impl(a, b):
        return _call_maxplus_with_argmax(np.asarray(a), np.asarray(b))

    c, argmax = jax.pure_callback(
        impl,
        (
            jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.float32),
            jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.int32),
        ),
        a, b
    )
    return c, (argmax, a.shape[1])


def _maxplus_bwd(res, g):
    """Backward pass: sparse gradient routing via argmax."""
    argmax, k = res
    m, n = g.shape

    # grad_a[i, argmax[i,j]] += g[i,j] for all j
    row_idx = jnp.arange(m)[:, None]
    grad_a = jnp.zeros((m, k), dtype=g.dtype).at[row_idx, argmax].add(g)

    # grad_b[argmax[i,j], j] += g[i,j] for all i
    col_idx = jnp.arange(n)[None, :]
    grad_b = jnp.zeros((k, n), dtype=g.dtype).at[argmax, col_idx].add(g)

    return grad_a, grad_b


tropical_maxplus_matmul.defvjp(_maxplus_fwd, _maxplus_bwd)


@custom_vjp
def tropical_minplus_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Tropical min-plus matrix multiplication: C[i,j] = min_k(A[i,k] + B[k,j])

    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)

    Returns:
        Output matrix C of shape (M, N)
    """
    def impl(a, b):
        c, _ = _call_minplus_with_argmax(np.asarray(a), np.asarray(b))
        return c

    return jax.pure_callback(
        impl,
        jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.float32),
        a, b
    )


def _minplus_fwd(a, b):
    def impl(a, b):
        return _call_minplus_with_argmax(np.asarray(a), np.asarray(b))

    c, argmax = jax.pure_callback(
        impl,
        (
            jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.float32),
            jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.int32),
        ),
        a, b
    )
    return c, (argmax, a.shape[1])


def _minplus_bwd(res, g):
    argmax, k = res
    m, n = g.shape

    row_idx = jnp.arange(m)[:, None]
    grad_a = jnp.zeros((m, k), dtype=g.dtype).at[row_idx, argmax].add(g)

    col_idx = jnp.arange(n)[None, :]
    grad_b = jnp.zeros((k, n), dtype=g.dtype).at[argmax, col_idx].add(g)

    return grad_a, grad_b


tropical_minplus_matmul.defvjp(_minplus_fwd, _minplus_bwd)


@custom_vjp
def tropical_maxmul_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Tropical max-mul matrix multiplication: C[i,j] = max_k(A[i,k] * B[k,j])

    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)

    Returns:
        Output matrix C of shape (M, N)
    """
    def impl(a, b):
        c, _ = _call_maxmul_with_argmax(np.asarray(a), np.asarray(b))
        return c

    return jax.pure_callback(
        impl,
        jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.float32),
        a, b
    )


def _maxmul_fwd(a, b):
    def impl(a, b):
        return _call_maxmul_with_argmax(np.asarray(a), np.asarray(b))

    c, argmax = jax.pure_callback(
        impl,
        (
            jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.float32),
            jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.int32),
        ),
        a, b
    )
    return c, (argmax, a, b)


def _maxmul_bwd(res, g):
    """MaxMul backward: d/da(a*b) = b, d/db(a*b) = a at argmax positions."""
    argmax, a, b = res
    m, k = a.shape
    n = b.shape[1]

    row_idx = jnp.arange(m)[:, None]
    col_idx = jnp.arange(n)[None, :]

    # For MaxMul: grad = g * (derivative of a*b)
    # At argmax[i,j] = k*: grad_a[i,k*] += g[i,j] * b[k*,j], grad_b[k*,j] += g[i,j] * a[i,k*]
    b_at_argmax = b[argmax, col_idx]  # (M, N)
    a_at_argmax = a[row_idx, argmax]  # (M, N)

    grad_a = jnp.zeros((m, k), dtype=g.dtype).at[row_idx, argmax].add(g * b_at_argmax)
    grad_b = jnp.zeros((k, n), dtype=g.dtype).at[argmax, col_idx].add(g * a_at_argmax)

    return grad_a, grad_b


tropical_maxmul_matmul.defvjp(_maxmul_fwd, _maxmul_bwd)


# ===========================================================================
# 3D Batched Operations with custom_vjp
# ===========================================================================

@custom_vjp
def tropical_maxplus_matmul_batched(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Batched tropical max-plus matrix multiplication.

    Args:
        a: Input tensor A of shape (B, M, K)
        b: Input tensor B of shape (B, K, N)

    Returns:
        Output tensor C of shape (B, M, N) where C[b,i,j] = max_k(A[b,i,k] + B[b,k,j])
    """
    def impl(a, b):
        c, _ = _call_maxplus_batched_with_argmax(np.asarray(a), np.asarray(b))
        return c

    return jax.pure_callback(
        impl,
        jax.ShapeDtypeStruct((a.shape[0], a.shape[1], b.shape[2]), jnp.float32),
        a, b
    )


def _maxplus_batched_fwd(a, b):
    def impl(a, b):
        return _call_maxplus_batched_with_argmax(np.asarray(a), np.asarray(b))

    batch, m, k = a.shape
    n = b.shape[2]

    c, argmax = jax.pure_callback(
        impl,
        (
            jax.ShapeDtypeStruct((batch, m, n), jnp.float32),
            jax.ShapeDtypeStruct((batch, m, n), jnp.int32),
        ),
        a, b
    )
    return c, (argmax, k)


def _maxplus_batched_bwd(res, g):
    argmax, k = res
    batch, m, n = g.shape

    batch_idx = jnp.arange(batch)[:, None, None]
    row_idx = jnp.arange(m)[None, :, None]
    col_idx = jnp.arange(n)[None, None, :]

    grad_a = jnp.zeros((batch, m, k), dtype=g.dtype).at[batch_idx, row_idx, argmax].add(g)
    grad_b = jnp.zeros((batch, k, n), dtype=g.dtype).at[batch_idx, argmax, col_idx].add(g)

    return grad_a, grad_b


tropical_maxplus_matmul_batched.defvjp(_maxplus_batched_fwd, _maxplus_batched_bwd)


@custom_vjp
def tropical_minplus_matmul_batched(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Batched tropical min-plus matrix multiplication.

    Args:
        a: Input tensor A of shape (B, M, K)
        b: Input tensor B of shape (B, K, N)

    Returns:
        Output tensor C of shape (B, M, N) where C[b,i,j] = min_k(A[b,i,k] + B[b,k,j])
    """
    def impl(a, b):
        c, _ = _call_minplus_batched_with_argmax(np.asarray(a), np.asarray(b))
        return c

    return jax.pure_callback(
        impl,
        jax.ShapeDtypeStruct((a.shape[0], a.shape[1], b.shape[2]), jnp.float32),
        a, b
    )


def _minplus_batched_fwd(a, b):
    def impl(a, b):
        return _call_minplus_batched_with_argmax(np.asarray(a), np.asarray(b))

    batch, m, k = a.shape
    n = b.shape[2]

    c, argmax = jax.pure_callback(
        impl,
        (
            jax.ShapeDtypeStruct((batch, m, n), jnp.float32),
            jax.ShapeDtypeStruct((batch, m, n), jnp.int32),
        ),
        a, b
    )
    return c, (argmax, k)


def _minplus_batched_bwd(res, g):
    argmax, k = res
    batch, m, n = g.shape

    batch_idx = jnp.arange(batch)[:, None, None]
    row_idx = jnp.arange(m)[None, :, None]
    col_idx = jnp.arange(n)[None, None, :]

    grad_a = jnp.zeros((batch, m, k), dtype=g.dtype).at[batch_idx, row_idx, argmax].add(g)
    grad_b = jnp.zeros((batch, k, n), dtype=g.dtype).at[batch_idx, argmax, col_idx].add(g)

    return grad_a, grad_b


tropical_minplus_matmul_batched.defvjp(_minplus_batched_fwd, _minplus_batched_bwd)


@custom_vjp
def tropical_maxmul_matmul_batched(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Batched tropical max-mul matrix multiplication.

    Args:
        a: Input tensor A of shape (B, M, K)
        b: Input tensor B of shape (B, K, N)

    Returns:
        Output tensor C of shape (B, M, N) where C[b,i,j] = max_k(A[b,i,k] * B[b,k,j])
    """
    def impl(a, b):
        c, _ = _call_maxmul_batched_with_argmax(np.asarray(a), np.asarray(b))
        return c

    return jax.pure_callback(
        impl,
        jax.ShapeDtypeStruct((a.shape[0], a.shape[1], b.shape[2]), jnp.float32),
        a, b
    )


def _maxmul_batched_fwd(a, b):
    def impl(a, b):
        return _call_maxmul_batched_with_argmax(np.asarray(a), np.asarray(b))

    batch, m, k = a.shape
    n = b.shape[2]

    c, argmax = jax.pure_callback(
        impl,
        (
            jax.ShapeDtypeStruct((batch, m, n), jnp.float32),
            jax.ShapeDtypeStruct((batch, m, n), jnp.int32),
        ),
        a, b
    )
    return c, (argmax, a, b)


def _maxmul_batched_bwd(res, g):
    argmax, a, b = res
    batch, m, k = a.shape
    n = b.shape[2]

    batch_idx = jnp.arange(batch)[:, None, None]
    row_idx = jnp.arange(m)[None, :, None]
    col_idx = jnp.arange(n)[None, None, :]

    b_at_argmax = b[batch_idx, argmax, col_idx]
    a_at_argmax = a[batch_idx, row_idx, argmax]

    grad_a = jnp.zeros((batch, m, k), dtype=g.dtype).at[batch_idx, row_idx, argmax].add(g * b_at_argmax)
    grad_b = jnp.zeros((batch, k, n), dtype=g.dtype).at[batch_idx, argmax, col_idx].add(g * a_at_argmax)

    return grad_a, grad_b


tropical_maxmul_matmul_batched.defvjp(_maxmul_batched_fwd, _maxmul_batched_bwd)


# ===========================================================================
# Exports
# ===========================================================================

__all__ = [
    # 2D operations
    "tropical_maxplus_matmul",
    "tropical_minplus_matmul",
    "tropical_maxmul_matmul",
    # Batched operations
    "tropical_maxplus_matmul_batched",
    "tropical_minplus_matmul_batched",
    "tropical_maxmul_matmul_batched",
]
