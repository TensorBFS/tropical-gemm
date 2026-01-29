"""
Tropical GEMM - Fast tropical matrix multiplication with automatic differentiation support.

This package provides optimized implementations of tropical matrix multiplication
for various semirings (MaxPlus, MinPlus, MaxMul) with support for:

- Multiple data types: f32, f64, i32, i64
- Automatic differentiation via argmax tracking
- Optional GPU acceleration via CUDA
- PyTorch integration for neural network training

Example:
    >>> import numpy as np
    >>> import tropical_gemm
    >>>
    >>> # Basic MaxPlus matmul
    >>> a = np.random.randn(100, 50).astype(np.float32)
    >>> b = np.random.randn(50, 80).astype(np.float32)
    >>> c = tropical_gemm.maxplus_matmul(a, b)
    >>>
    >>> # With argmax for backpropagation
    >>> c, argmax = tropical_gemm.maxplus_matmul_with_argmax(a, b)
"""

# Import everything from the Rust extension module
from tropical_gemm._core import *

# Re-export for convenience
from tropical_gemm._core import (
    # f32 operations
    maxplus_matmul,
    minplus_matmul,
    maxmul_matmul,
    maxplus_matmul_with_argmax,
    minplus_matmul_with_argmax,
    maxmul_matmul_with_argmax,
    backward_a,
    backward_b,
    # f64 operations
    maxplus_matmul_f64,
    minplus_matmul_f64,
    maxmul_matmul_f64,
    maxplus_matmul_with_argmax_f64,
    minplus_matmul_with_argmax_f64,
    maxmul_matmul_with_argmax_f64,
    backward_a_f64,
    backward_b_f64,
    # MaxMul backward (multiplicative rule)
    maxmul_backward_a,
    maxmul_backward_b,
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
    # 2D output variants (f32)
    maxplus_matmul_2d,
    minplus_matmul_2d,
    maxmul_matmul_2d,
    # 2D output variants (f64)
    maxplus_matmul_2d_f64,
    minplus_matmul_2d_f64,
    maxmul_matmul_2d_f64,
    # 2D output variants (i32)
    maxplus_matmul_2d_i32,
    minplus_matmul_2d_i32,
    maxmul_matmul_2d_i32,
    # 2D output variants (i64)
    maxplus_matmul_2d_i64,
    minplus_matmul_2d_i64,
    maxmul_matmul_2d_i64,
    # CUDA availability
    cuda_available,
)

# Conditionally import GPU functions if available
if cuda_available():
    from tropical_gemm._core import (
        maxplus_matmul_gpu,
        minplus_matmul_gpu,
        maxmul_matmul_gpu,
        maxplus_matmul_gpu_with_argmax,
        minplus_matmul_gpu_with_argmax,
        maxmul_matmul_gpu_with_argmax,
    )

__version__ = "0.2.0"

__all__ = [
    # f32 operations
    "maxplus_matmul",
    "minplus_matmul",
    "maxmul_matmul",
    "maxplus_matmul_with_argmax",
    "minplus_matmul_with_argmax",
    "maxmul_matmul_with_argmax",
    "backward_a",
    "backward_b",
    # f64 operations
    "maxplus_matmul_f64",
    "minplus_matmul_f64",
    "maxmul_matmul_f64",
    "maxplus_matmul_with_argmax_f64",
    "minplus_matmul_with_argmax_f64",
    "maxmul_matmul_with_argmax_f64",
    "backward_a_f64",
    "backward_b_f64",
    # MaxMul backward
    "maxmul_backward_a",
    "maxmul_backward_b",
    "maxmul_backward_a_f64",
    "maxmul_backward_b_f64",
    # i32 operations
    "maxplus_matmul_i32",
    "minplus_matmul_i32",
    "maxmul_matmul_i32",
    # i64 operations
    "maxplus_matmul_i64",
    "minplus_matmul_i64",
    "maxmul_matmul_i64",
    # 2D output variants (f32)
    "maxplus_matmul_2d",
    "minplus_matmul_2d",
    "maxmul_matmul_2d",
    # 2D output variants (f64)
    "maxplus_matmul_2d_f64",
    "minplus_matmul_2d_f64",
    "maxmul_matmul_2d_f64",
    # 2D output variants (i32)
    "maxplus_matmul_2d_i32",
    "minplus_matmul_2d_i32",
    "maxmul_matmul_2d_i32",
    # 2D output variants (i64)
    "maxplus_matmul_2d_i64",
    "minplus_matmul_2d_i64",
    "maxmul_matmul_2d_i64",
    # CUDA
    "cuda_available",
]

# Add GPU functions to __all__ if available
if cuda_available():
    __all__.extend([
        "maxplus_matmul_gpu",
        "minplus_matmul_gpu",
        "maxmul_matmul_gpu",
        "maxplus_matmul_gpu_with_argmax",
        "minplus_matmul_gpu_with_argmax",
        "maxmul_matmul_gpu_with_argmax",
    ])
