"""
JAX integration for tropical matrix multiplication.

This module provides JAX-compatible functions with custom VJP (vector-Jacobian product)
support for automatic differentiation.

Supports both CPU and GPU:
- CPU: Uses optimized Rust SIMD backend
- GPU: Uses CUDA backend via DLPack zero-copy interface

Example:
    >>> import jax.numpy as jnp
    >>> from tropical_gemm.jax import tropical_maxplus_matmul
    >>>
    >>> # CPU
    >>> a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> b = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    >>> c = tropical_maxplus_matmul(a, b)
    >>>
    >>> # GPU (if available)
    >>> from tropical_gemm.jax import tropical_maxplus_matmul_gpu
    >>> a_gpu = jax.device_put(a, jax.devices('gpu')[0])
    >>> b_gpu = jax.device_put(b, jax.devices('gpu')[0])
    >>> c_gpu = tropical_maxplus_matmul_gpu(a_gpu, b_gpu)
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

# Check CUDA availability
GPU_AVAILABLE = tropical_gemm.cuda_available()


# ===========================================================================
# CPU Helper functions
# ===========================================================================

def _call_maxplus_with_argmax(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Call Rust MaxPlus matmul with argmax tracking."""
    m, n = a.shape[0], b.shape[1]
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c_flat, argmax_flat = tropical_gemm.maxplus_matmul_with_argmax(a, b)
    return np.array(c_flat, dtype=np.float32).reshape(m, n), np.array(argmax_flat, dtype=np.int32).reshape(m, n)


def _call_minplus_with_argmax(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Call Rust MinPlus matmul with argmax tracking."""
    m, n = a.shape[0], b.shape[1]
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c_flat, argmax_flat = tropical_gemm.minplus_matmul_with_argmax(a, b)
    return np.array(c_flat, dtype=np.float32).reshape(m, n), np.array(argmax_flat, dtype=np.int32).reshape(m, n)


def _call_maxmul_with_argmax(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Call Rust MaxMul matmul with argmax tracking."""
    m, n = a.shape[0], b.shape[1]
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c_flat, argmax_flat = tropical_gemm.maxmul_matmul_with_argmax(a, b)
    return np.array(c_flat, dtype=np.float32).reshape(m, n), np.array(argmax_flat, dtype=np.int32).reshape(m, n)


# Batched CPU versions
def _call_maxplus_batched_with_argmax(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch, m, n = a.shape[0], a.shape[1], b.shape[2]
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c_flat, argmax_flat = tropical_gemm.maxplus_matmul_strided_batched_with_argmax(a, b)
    return np.array(c_flat, dtype=np.float32).reshape(batch, m, n), np.array(argmax_flat, dtype=np.int32).reshape(batch, m, n)


def _call_minplus_batched_with_argmax(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch, m, n = a.shape[0], a.shape[1], b.shape[2]
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c_flat, argmax_flat = tropical_gemm.minplus_matmul_strided_batched_with_argmax(a, b)
    return np.array(c_flat, dtype=np.float32).reshape(batch, m, n), np.array(argmax_flat, dtype=np.int32).reshape(batch, m, n)


def _call_maxmul_batched_with_argmax(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch, m, n = a.shape[0], a.shape[1], b.shape[2]
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c_flat, argmax_flat = tropical_gemm.maxmul_matmul_strided_batched_with_argmax(a, b)
    return np.array(c_flat, dtype=np.float32).reshape(batch, m, n), np.array(argmax_flat, dtype=np.int32).reshape(batch, m, n)


# ===========================================================================
# GPU Helper functions (DLPack-based)
# ===========================================================================

class _JaxDLPackWrapper:
    """Wrapper to provide __dlpack__ interface for JAX arrays."""
    def __init__(self, arr):
        self._arr = arr

    def __dlpack__(self, stream=None):
        return jax.dlpack.to_dlpack(self._arr)

    def __dlpack_device__(self):
        # Return (device_type, device_id)
        # For CUDA: device_type = 2
        device = self._arr.device
        if hasattr(device, 'id'):
            return (2, device.id)
        return (2, 0)


def _call_maxplus_gpu_with_argmax(a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Call CUDA MaxPlus matmul with argmax via DLPack."""
    m, n = a.shape[0], b.shape[1]
    a_wrap, b_wrap = _JaxDLPackWrapper(a), _JaxDLPackWrapper(b)
    c_flat, argmax_flat = tropical_gemm.maxplus_matmul_gpu_with_argmax(a_wrap, b_wrap)
    c = jax.dlpack.from_dlpack(c_flat).reshape(m, n)
    argmax = jax.dlpack.from_dlpack(argmax_flat).reshape(m, n).astype(jnp.int32)
    return c, argmax


def _call_minplus_gpu_with_argmax(a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Call CUDA MinPlus matmul with argmax via DLPack."""
    m, n = a.shape[0], b.shape[1]
    a_wrap, b_wrap = _JaxDLPackWrapper(a), _JaxDLPackWrapper(b)
    c_flat, argmax_flat = tropical_gemm.minplus_matmul_gpu_with_argmax(a_wrap, b_wrap)
    c = jax.dlpack.from_dlpack(c_flat).reshape(m, n)
    argmax = jax.dlpack.from_dlpack(argmax_flat).reshape(m, n).astype(jnp.int32)
    return c, argmax


def _call_maxmul_gpu_with_argmax(a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Call CUDA MaxMul matmul with argmax via DLPack."""
    m, n = a.shape[0], b.shape[1]
    a_wrap, b_wrap = _JaxDLPackWrapper(a), _JaxDLPackWrapper(b)
    c_flat, argmax_flat = tropical_gemm.maxmul_matmul_gpu_with_argmax(a_wrap, b_wrap)
    c = jax.dlpack.from_dlpack(c_flat).reshape(m, n)
    argmax = jax.dlpack.from_dlpack(argmax_flat).reshape(m, n).astype(jnp.int32)
    return c, argmax


# Batched GPU versions
def _call_maxplus_batched_gpu_with_argmax(a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Call CUDA batched MaxPlus matmul with argmax via DLPack."""
    batch, m, n = a.shape[0], a.shape[1], b.shape[2]
    a_wrap, b_wrap = _JaxDLPackWrapper(a), _JaxDLPackWrapper(b)
    c_flat, argmax_flat = tropical_gemm.maxplus_matmul_gpu_strided_batched_with_argmax(a_wrap, b_wrap)
    c = jax.dlpack.from_dlpack(c_flat).reshape(batch, m, n)
    argmax = jax.dlpack.from_dlpack(argmax_flat).reshape(batch, m, n).astype(jnp.int32)
    return c, argmax


def _call_minplus_batched_gpu_with_argmax(a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Call CUDA batched MinPlus matmul with argmax via DLPack."""
    batch, m, n = a.shape[0], a.shape[1], b.shape[2]
    a_wrap, b_wrap = _JaxDLPackWrapper(a), _JaxDLPackWrapper(b)
    c_flat, argmax_flat = tropical_gemm.minplus_matmul_gpu_strided_batched_with_argmax(a_wrap, b_wrap)
    c = jax.dlpack.from_dlpack(c_flat).reshape(batch, m, n)
    argmax = jax.dlpack.from_dlpack(argmax_flat).reshape(batch, m, n).astype(jnp.int32)
    return c, argmax


def _call_maxmul_batched_gpu_with_argmax(a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Call CUDA batched MaxMul matmul with argmax via DLPack."""
    batch, m, n = a.shape[0], a.shape[1], b.shape[2]
    a_wrap, b_wrap = _JaxDLPackWrapper(a), _JaxDLPackWrapper(b)
    c_flat, argmax_flat = tropical_gemm.maxmul_matmul_gpu_strided_batched_with_argmax(a_wrap, b_wrap)
    c = jax.dlpack.from_dlpack(c_flat).reshape(batch, m, n)
    argmax = jax.dlpack.from_dlpack(argmax_flat).reshape(batch, m, n).astype(jnp.int32)
    return c, argmax


# ===========================================================================
# 2D CPU Operations with custom_vjp
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
    def impl(a, b):
        return _call_maxplus_with_argmax(np.asarray(a), np.asarray(b))

    c, argmax = jax.pure_callback(
        impl,
        (jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.float32),
         jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.int32)),
        a, b
    )
    return c, (argmax, a.shape[1])


def _maxplus_bwd(res, g):
    argmax, k = res
    m, n = g.shape
    row_idx = jnp.arange(m)[:, None]
    col_idx = jnp.arange(n)[None, :]
    grad_a = jnp.zeros((m, k), dtype=g.dtype).at[row_idx, argmax].add(g)
    grad_b = jnp.zeros((k, n), dtype=g.dtype).at[argmax, col_idx].add(g)
    return grad_a, grad_b


tropical_maxplus_matmul.defvjp(_maxplus_fwd, _maxplus_bwd)


@custom_vjp
def tropical_minplus_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Tropical min-plus matrix multiplication: C[i,j] = min_k(A[i,k] + B[k,j])"""
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
        (jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.float32),
         jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.int32)),
        a, b
    )
    return c, (argmax, a.shape[1])


def _minplus_bwd(res, g):
    argmax, k = res
    m, n = g.shape
    row_idx = jnp.arange(m)[:, None]
    col_idx = jnp.arange(n)[None, :]
    grad_a = jnp.zeros((m, k), dtype=g.dtype).at[row_idx, argmax].add(g)
    grad_b = jnp.zeros((k, n), dtype=g.dtype).at[argmax, col_idx].add(g)
    return grad_a, grad_b


tropical_minplus_matmul.defvjp(_minplus_fwd, _minplus_bwd)


@custom_vjp
def tropical_maxmul_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Tropical max-mul matrix multiplication: C[i,j] = max_k(A[i,k] * B[k,j])"""
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
        (jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.float32),
         jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.int32)),
        a, b
    )
    return c, (argmax, a, b)


def _maxmul_bwd(res, g):
    argmax, a, b = res
    m, k = a.shape
    n = b.shape[1]
    row_idx = jnp.arange(m)[:, None]
    col_idx = jnp.arange(n)[None, :]
    b_at_argmax = b[argmax, col_idx]
    a_at_argmax = a[row_idx, argmax]
    grad_a = jnp.zeros((m, k), dtype=g.dtype).at[row_idx, argmax].add(g * b_at_argmax)
    grad_b = jnp.zeros((k, n), dtype=g.dtype).at[argmax, col_idx].add(g * a_at_argmax)
    return grad_a, grad_b


tropical_maxmul_matmul.defvjp(_maxmul_fwd, _maxmul_bwd)


# ===========================================================================
# 3D Batched CPU Operations with custom_vjp
# ===========================================================================

@custom_vjp
def tropical_maxplus_matmul_batched(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Batched tropical max-plus: C[b,i,j] = max_k(A[b,i,k] + B[b,k,j])"""
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
        (jax.ShapeDtypeStruct((batch, m, n), jnp.float32),
         jax.ShapeDtypeStruct((batch, m, n), jnp.int32)),
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
    """Batched tropical min-plus: C[b,i,j] = min_k(A[b,i,k] + B[b,k,j])"""
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
        (jax.ShapeDtypeStruct((batch, m, n), jnp.float32),
         jax.ShapeDtypeStruct((batch, m, n), jnp.int32)),
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
    """Batched tropical max-mul: C[b,i,j] = max_k(A[b,i,k] * B[b,k,j])"""
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
        (jax.ShapeDtypeStruct((batch, m, n), jnp.float32),
         jax.ShapeDtypeStruct((batch, m, n), jnp.int32)),
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
# 2D GPU Operations with custom_vjp
# ===========================================================================

if GPU_AVAILABLE:
    @custom_vjp
    def tropical_maxplus_matmul_gpu(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """GPU tropical max-plus matrix multiplication: C[i,j] = max_k(A[i,k] + B[k,j])"""
        c, _ = _call_maxplus_gpu_with_argmax(a, b)
        return c

    def _maxplus_gpu_fwd(a, b):
        c, argmax = _call_maxplus_gpu_with_argmax(a, b)
        return c, (argmax, a.shape[1])

    def _maxplus_gpu_bwd(res, g):
        argmax, k = res
        m, n = g.shape
        row_idx = jnp.arange(m)[:, None]
        col_idx = jnp.arange(n)[None, :]
        grad_a = jnp.zeros((m, k), dtype=g.dtype).at[row_idx, argmax].add(g)
        grad_b = jnp.zeros((k, n), dtype=g.dtype).at[argmax, col_idx].add(g)
        return grad_a, grad_b

    tropical_maxplus_matmul_gpu.defvjp(_maxplus_gpu_fwd, _maxplus_gpu_bwd)

    @custom_vjp
    def tropical_minplus_matmul_gpu(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """GPU tropical min-plus matrix multiplication: C[i,j] = min_k(A[i,k] + B[k,j])"""
        c, _ = _call_minplus_gpu_with_argmax(a, b)
        return c

    def _minplus_gpu_fwd(a, b):
        c, argmax = _call_minplus_gpu_with_argmax(a, b)
        return c, (argmax, a.shape[1])

    def _minplus_gpu_bwd(res, g):
        argmax, k = res
        m, n = g.shape
        row_idx = jnp.arange(m)[:, None]
        col_idx = jnp.arange(n)[None, :]
        grad_a = jnp.zeros((m, k), dtype=g.dtype).at[row_idx, argmax].add(g)
        grad_b = jnp.zeros((k, n), dtype=g.dtype).at[argmax, col_idx].add(g)
        return grad_a, grad_b

    tropical_minplus_matmul_gpu.defvjp(_minplus_gpu_fwd, _minplus_gpu_bwd)

    @custom_vjp
    def tropical_maxmul_matmul_gpu(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """GPU tropical max-mul matrix multiplication: C[i,j] = max_k(A[i,k] * B[k,j])"""
        c, _ = _call_maxmul_gpu_with_argmax(a, b)
        return c

    def _maxmul_gpu_fwd(a, b):
        c, argmax = _call_maxmul_gpu_with_argmax(a, b)
        return c, (argmax, a, b)

    def _maxmul_gpu_bwd(res, g):
        argmax, a, b = res
        m, k = a.shape
        n = b.shape[1]
        row_idx = jnp.arange(m)[:, None]
        col_idx = jnp.arange(n)[None, :]
        b_at_argmax = b[argmax, col_idx]
        a_at_argmax = a[row_idx, argmax]
        grad_a = jnp.zeros((m, k), dtype=g.dtype).at[row_idx, argmax].add(g * b_at_argmax)
        grad_b = jnp.zeros((k, n), dtype=g.dtype).at[argmax, col_idx].add(g * a_at_argmax)
        return grad_a, grad_b

    tropical_maxmul_matmul_gpu.defvjp(_maxmul_gpu_fwd, _maxmul_gpu_bwd)

    # ===========================================================================
    # 3D Batched GPU Operations with custom_vjp
    # ===========================================================================

    @custom_vjp
    def tropical_maxplus_matmul_batched_gpu(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """GPU batched tropical max-plus: C[b,i,j] = max_k(A[b,i,k] + B[b,k,j])"""
        c, _ = _call_maxplus_batched_gpu_with_argmax(a, b)
        return c

    def _maxplus_batched_gpu_fwd(a, b):
        c, argmax = _call_maxplus_batched_gpu_with_argmax(a, b)
        return c, (argmax, a.shape[2])

    def _maxplus_batched_gpu_bwd(res, g):
        argmax, k = res
        batch, m, n = g.shape
        batch_idx = jnp.arange(batch)[:, None, None]
        row_idx = jnp.arange(m)[None, :, None]
        col_idx = jnp.arange(n)[None, None, :]
        grad_a = jnp.zeros((batch, m, k), dtype=g.dtype).at[batch_idx, row_idx, argmax].add(g)
        grad_b = jnp.zeros((batch, k, n), dtype=g.dtype).at[batch_idx, argmax, col_idx].add(g)
        return grad_a, grad_b

    tropical_maxplus_matmul_batched_gpu.defvjp(_maxplus_batched_gpu_fwd, _maxplus_batched_gpu_bwd)

    @custom_vjp
    def tropical_minplus_matmul_batched_gpu(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """GPU batched tropical min-plus: C[b,i,j] = min_k(A[b,i,k] + B[b,k,j])"""
        c, _ = _call_minplus_batched_gpu_with_argmax(a, b)
        return c

    def _minplus_batched_gpu_fwd(a, b):
        c, argmax = _call_minplus_batched_gpu_with_argmax(a, b)
        return c, (argmax, a.shape[2])

    def _minplus_batched_gpu_bwd(res, g):
        argmax, k = res
        batch, m, n = g.shape
        batch_idx = jnp.arange(batch)[:, None, None]
        row_idx = jnp.arange(m)[None, :, None]
        col_idx = jnp.arange(n)[None, None, :]
        grad_a = jnp.zeros((batch, m, k), dtype=g.dtype).at[batch_idx, row_idx, argmax].add(g)
        grad_b = jnp.zeros((batch, k, n), dtype=g.dtype).at[batch_idx, argmax, col_idx].add(g)
        return grad_a, grad_b

    tropical_minplus_matmul_batched_gpu.defvjp(_minplus_batched_gpu_fwd, _minplus_batched_gpu_bwd)

    @custom_vjp
    def tropical_maxmul_matmul_batched_gpu(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """GPU batched tropical max-mul: C[b,i,j] = max_k(A[b,i,k] * B[b,k,j])"""
        c, _ = _call_maxmul_batched_gpu_with_argmax(a, b)
        return c

    def _maxmul_batched_gpu_fwd(a, b):
        c, argmax = _call_maxmul_batched_gpu_with_argmax(a, b)
        return c, (argmax, a, b)

    def _maxmul_batched_gpu_bwd(res, g):
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

    tropical_maxmul_matmul_batched_gpu.defvjp(_maxmul_batched_gpu_fwd, _maxmul_batched_gpu_bwd)


# ===========================================================================
# Exports
# ===========================================================================

__all__ = [
    # 2D CPU operations
    "tropical_maxplus_matmul",
    "tropical_minplus_matmul",
    "tropical_maxmul_matmul",
    # Batched CPU operations
    "tropical_maxplus_matmul_batched",
    "tropical_minplus_matmul_batched",
    "tropical_maxmul_matmul_batched",
]

if GPU_AVAILABLE:
    __all__.extend([
        # 2D GPU operations
        "tropical_maxplus_matmul_gpu",
        "tropical_minplus_matmul_gpu",
        "tropical_maxmul_matmul_gpu",
        # Batched GPU operations
        "tropical_maxplus_matmul_batched_gpu",
        "tropical_minplus_matmul_batched_gpu",
        "tropical_maxmul_matmul_batched_gpu",
    ])
