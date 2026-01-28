"""
PyTorch integration for tropical matrix multiplication.

This module provides custom autograd functions that enable using tropical
matrix multiplication in PyTorch neural networks with full gradient support.

Tropical semirings are useful for:
- Shortest path problems (MinPlus)
- Longest path problems (MaxPlus)
- Dynamic programming on graphs
- Probabilistic inference (log-space operations)

Example:
    >>> import torch
    >>> from tropical_gemm.pytorch import tropical_maxplus_matmul
    >>>
    >>> a = torch.randn(100, 50, requires_grad=True)
    >>> b = torch.randn(50, 80, requires_grad=True)
    >>> c = tropical_maxplus_matmul(a, b)
    >>> loss = c.sum()
    >>> loss.backward()
"""

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required for this module. "
        "Install it with: pip install tropical-gemm[torch]"
    )

import tropical_gemm

# Check if GPU is available
GPU_AVAILABLE = tropical_gemm.cuda_available()


# ===========================================================================
# Helper: Convert tensors to C-contiguous numpy arrays
# ===========================================================================


def _to_contiguous_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to C_CONTIGUOUS numpy array."""
    arr = t.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def _to_contiguous_numpy_3d(t: torch.Tensor) -> np.ndarray:
    """Convert 3D PyTorch tensor to C_CONTIGUOUS numpy array."""
    arr = t.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr




# ===========================================================================
# CPU Autograd Functions (using optimized Rust SIMD backend)
# ===========================================================================


class TropicalMaxPlusMatmul(torch.autograd.Function):
    """
    Custom autograd function for MaxPlus tropical matrix multiplication.

    Forward: C[i,j] = max_k(A[i,k] + B[k,j])

    The gradient is sparse: only the argmax index contributes to each output.
    For each output C[i,j], the gradient flows back to:
    - A[i, argmax[i,j]]
    - B[argmax[i,j], j]
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute tropical MaxPlus matmul.

        Args:
            a: Input tensor of shape (M, K)
            b: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        m, k = a.shape
        n = b.shape[1]

        a_np = _to_contiguous_numpy(a)
        b_np = _to_contiguous_numpy(b)

        c_flat, argmax_flat = tropical_gemm.maxplus_matmul_with_argmax(a_np, b_np)

        c_np = np.asarray(c_flat).reshape(m, n)
        argmax_np = np.asarray(argmax_flat).reshape(m, n)

        ctx.save_for_backward(torch.from_numpy(argmax_np.copy()))
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np.copy()).to(a.device)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        """
        Backward pass: compute gradients w.r.t. A and B.

        The tropical matmul gradient is sparse because only the argmax
        index contributes to each output element.
        """
        (argmax,) = ctx.saved_tensors
        k = ctx.k
        m = ctx.m
        n = ctx.n

        # Ensure contiguous numpy arrays
        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        # Compute gradients using the Rust backend (returns flattened arrays)
        grad_a_flat = tropical_gemm.backward_a(grad_c_np, argmax_np, k)
        grad_b_flat = tropical_gemm.backward_b(grad_c_np, argmax_np, k)

        # Reshape to 2D
        grad_a = torch.from_numpy(np.asarray(grad_a_flat).reshape(m, k)).to(grad_c.device)
        grad_b = torch.from_numpy(np.asarray(grad_b_flat).reshape(k, n)).to(grad_c.device)

        return grad_a, grad_b


class TropicalMinPlusMatmul(torch.autograd.Function):
    """
    Custom autograd function for MinPlus tropical matrix multiplication.

    Forward: C[i,j] = min_k(A[i,k] + B[k,j])

    Useful for shortest path computations in graphs.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.shape
        n = b.shape[1]

        a_np = _to_contiguous_numpy(a)
        b_np = _to_contiguous_numpy(b)

        c_flat, argmax_flat = tropical_gemm.minplus_matmul_with_argmax(a_np, b_np)

        c_np = np.asarray(c_flat).reshape(m, n)
        argmax_np = np.asarray(argmax_flat).reshape(m, n)

        ctx.save_for_backward(torch.from_numpy(argmax_np.copy()))
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np.copy()).to(a.device)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
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

        grad_a = torch.from_numpy(np.asarray(grad_a_flat).reshape(m, k)).to(grad_c.device)
        grad_b = torch.from_numpy(np.asarray(grad_b_flat).reshape(k, n)).to(grad_c.device)

        return grad_a, grad_b


class TropicalMaxMulMatmul(torch.autograd.Function):
    """
    Custom autograd function for MaxMul tropical matrix multiplication.

    Forward: C[i,j] = max_k(A[i,k] * B[k,j])

    The backward pass differs from MaxPlus/MinPlus because the operation
    is multiplication, not addition. The chain rule gives:
    - grad_A[i,k] = grad_C[i,j] * B[k,j] if k == argmax[i,j]
    - grad_B[k,j] = grad_C[i,j] * A[i,k] if k == argmax[i,j]

    Useful for max-probability computations (non-log space).
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.shape
        n = b.shape[1]

        a_np = _to_contiguous_numpy(a)
        b_np = _to_contiguous_numpy(b)

        c_flat, argmax_flat = tropical_gemm.maxmul_matmul_with_argmax(a_np, b_np)

        c_np = np.asarray(c_flat).reshape(m, n)
        argmax_np = np.asarray(argmax_flat).reshape(m, n)

        # Save inputs and argmax for backward (needed for multiplicative gradient)
        ctx.save_for_backward(
            torch.from_numpy(a_np.copy()),
            torch.from_numpy(b_np.copy()),
            torch.from_numpy(argmax_np.copy()),
        )
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np.copy()).to(a.device)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        a, b, argmax = ctx.saved_tensors
        k_dim = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)
        a_np = a.numpy().astype(np.float32)
        b_np = b.numpy().astype(np.float32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        # Use multiplicative backward rule
        grad_a_flat = tropical_gemm.maxmul_backward_a(grad_c_np, argmax_np, b_np)
        grad_b_flat = tropical_gemm.maxmul_backward_b(grad_c_np, argmax_np, a_np)

        grad_a = torch.from_numpy(np.asarray(grad_a_flat).reshape(m, k_dim)).to(
            grad_c.device
        )
        grad_b = torch.from_numpy(np.asarray(grad_b_flat).reshape(k_dim, n)).to(
            grad_c.device
        )

        return grad_a, grad_b


# ===========================================================================
# GPU-Accelerated Autograd Functions (using Rust CUDA backend via DLPack)
# ===========================================================================

# Check if DLPack functions are available (CUDA build)
_DLPACK_AVAILABLE = hasattr(tropical_gemm, "maxplus_matmul_dlpack")


class TropicalMaxPlusMatmulGPU(torch.autograd.Function):
    """
    GPU-accelerated MaxPlus tropical matrix multiplication.

    Uses the optimized Rust CUDA backend via DLPack for zero-copy GPU tensor exchange.
    Input tensors stay on GPU; only the forward pass output requires a D2H transfer
    (Phase 1). The backward pass runs entirely on GPU using PyTorch scatter operations.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not _DLPACK_AVAILABLE:
            raise RuntimeError("CUDA support not available. Build with CUDA feature enabled.")
        if not a.is_cuda or not b.is_cuda:
            raise RuntimeError("GPU function requires CUDA tensors. Use tropical_maxplus_matmul for CPU tensors.")

        m, k = a.shape
        n = b.shape[1]

        # Rust side already applies the row-major -> column-major swap trick for external pointers.
        # Keep the Python API simple: pass (M,K) and (K,N) directly and reshape output as (M,N).
        a_c = a.detach().contiguous()
        b_c = b.detach().contiguous()

        c_flat, argmax_flat = tropical_gemm.maxplus_matmul_dlpack(a_c, b_c)

        c = torch.from_numpy(np.asarray(c_flat).reshape(m, n).copy()).to(a.device)
        argmax = (
            torch.from_numpy(np.asarray(argmax_flat).reshape(m, n).copy())
            .to(a.device)
            .long()
        )

        ctx.save_for_backward(argmax)
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return c

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        (argmax,) = ctx.saved_tensors
        k = ctx.k
        m = ctx.m
        n = ctx.n

        # Compute gradients on GPU using scatter operations
        grad_a = torch.zeros(m, k, device=grad_c.device, dtype=grad_c.dtype)
        grad_a.scatter_add_(1, argmax, grad_c)

        argmax_t = argmax.t()
        grad_c_t = grad_c.t()
        grad_b_t = torch.zeros(n, k, device=grad_c.device, dtype=grad_c.dtype)
        grad_b_t.scatter_add_(1, argmax_t, grad_c_t)
        grad_b = grad_b_t.t()

        return grad_a, grad_b


class TropicalMinPlusMatmulGPU(torch.autograd.Function):
    """
    GPU-accelerated MinPlus tropical matrix multiplication.

    Uses the optimized Rust CUDA backend via DLPack for zero-copy GPU tensor exchange.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not _DLPACK_AVAILABLE:
            raise RuntimeError("CUDA support not available. Build with CUDA feature enabled.")
        if not a.is_cuda or not b.is_cuda:
            raise RuntimeError("GPU function requires CUDA tensors. Use tropical_minplus_matmul for CPU tensors.")

        m, k = a.shape
        n = b.shape[1]

        a_c = a.detach().contiguous()
        b_c = b.detach().contiguous()

        c_flat, argmax_flat = tropical_gemm.minplus_matmul_dlpack(a_c, b_c)

        c = torch.from_numpy(np.asarray(c_flat).reshape(m, n).copy()).to(a.device)
        argmax = (
            torch.from_numpy(np.asarray(argmax_flat).reshape(m, n).copy())
            .to(a.device)
            .long()
        )

        ctx.save_for_backward(argmax)
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return c

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        (argmin,) = ctx.saved_tensors
        k = ctx.k
        m = ctx.m
        n = ctx.n

        grad_a = torch.zeros(m, k, device=grad_c.device, dtype=grad_c.dtype)
        grad_a.scatter_add_(1, argmin, grad_c)

        argmin_t = argmin.t()
        grad_c_t = grad_c.t()
        grad_b_t = torch.zeros(n, k, device=grad_c.device, dtype=grad_c.dtype)
        grad_b_t.scatter_add_(1, argmin_t, grad_c_t)
        grad_b = grad_b_t.t()

        return grad_a, grad_b


class TropicalMaxMulMatmulGPU(torch.autograd.Function):
    """
    GPU-accelerated MaxMul tropical matrix multiplication.

    Forward: C[i,j] = max_k(A[i,k] * B[k,j])

    Uses the optimized Rust CUDA backend via DLPack for zero-copy GPU tensor exchange.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not _DLPACK_AVAILABLE:
            raise RuntimeError("CUDA support not available. Build with CUDA feature enabled.")
        if not a.is_cuda or not b.is_cuda:
            raise RuntimeError("GPU function requires CUDA tensors. Use tropical_maxmul_matmul for CPU tensors.")

        m, k = a.shape
        n = b.shape[1]

        a_c = a.detach().contiguous()
        b_c = b.detach().contiguous()

        c_flat, argmax_flat = tropical_gemm.maxmul_matmul_dlpack(a_c, b_c)

        c = torch.from_numpy(np.asarray(c_flat).reshape(m, n).copy()).to(a.device)
        argmax = (
            torch.from_numpy(np.asarray(argmax_flat).reshape(m, n).copy())
            .to(a.device)
            .long()
        )

        ctx.save_for_backward(a.detach(), b.detach(), argmax)
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return c

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        a, b, argmax = ctx.saved_tensors
        k_dim = ctx.k
        m = ctx.m
        n = ctx.n

        # For maxmul: C[i,j] = A[i, argmax[i,j]] * B[argmax[i,j], j]
        # grad_a[i, argmax[i,j]] += grad_c[i,j] * B[argmax[i,j], j]
        # grad_b[argmax[i,j], j] += grad_c[i,j] * A[i, argmax[i,j]]

        # Get the winning values from B
        b_winning = b[argmax, torch.arange(n, device=b.device).unsqueeze(0).expand(m, -1)]

        # Get the winning values from A
        a_winning = torch.gather(a, 1, argmax)

        # Compute gradients
        grad_a = torch.zeros(m, k_dim, device=grad_c.device, dtype=grad_c.dtype)
        grad_a.scatter_add_(1, argmax, grad_c * b_winning)

        argmax_t = argmax.t()
        grad_c_a_t = (grad_c * a_winning).t()
        grad_b_t = torch.zeros(n, k_dim, device=grad_c.device, dtype=grad_c.dtype)
        grad_b_t.scatter_add_(1, argmax_t, grad_c_a_t)
        grad_b = grad_b_t.t()

        return grad_a, grad_b


# ===========================================================================
# Batched CPU Autograd Functions
# ===========================================================================


class TropicalMaxPlusMatmulBatched(torch.autograd.Function):
    """
    Batched MaxPlus tropical matrix multiplication.

    Forward: C[b,i,j] = max_k(A[b,i,k] + B[b,k,j])

    Args:
        a: Input tensor of shape (batch, M, K)
        b: Input tensor of shape (batch, K, N)

    Returns:
        Output tensor of shape (batch, M, N)
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch_size, m, k = a.shape
        n = b.shape[2]

        a_np = _to_contiguous_numpy_3d(a)
        b_np = _to_contiguous_numpy_3d(b)

        c_flat, argmax_flat = tropical_gemm.maxplus_matmul_strided_batched_with_argmax(
            a_np, b_np
        )

        c = torch.from_numpy(np.asarray(c_flat).reshape(batch_size, m, n)).to(a.device)
        argmax = torch.from_numpy(np.asarray(argmax_flat).reshape(batch_size, m, n)).to(
            a.device
        )

        ctx.save_for_backward(argmax)
        ctx.batch_size = batch_size
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return c

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        (argmax,) = ctx.saved_tensors
        batch_size = ctx.batch_size
        k = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.cpu().numpy().astype(np.int32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        grad_a_flat = tropical_gemm.backward_a_strided_batched(grad_c_np, argmax_np, k)
        grad_b_flat = tropical_gemm.backward_b_strided_batched(grad_c_np, argmax_np, k)

        grad_a = torch.from_numpy(np.asarray(grad_a_flat).reshape(batch_size, m, k)).to(
            grad_c.device
        )
        grad_b = torch.from_numpy(np.asarray(grad_b_flat).reshape(batch_size, k, n)).to(
            grad_c.device
        )

        return grad_a, grad_b


class TropicalMinPlusMatmulBatched(torch.autograd.Function):
    """
    Batched MinPlus tropical matrix multiplication.

    Forward: C[b,i,j] = min_k(A[b,i,k] + B[b,k,j])
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch_size, m, k = a.shape
        n = b.shape[2]

        a_np = _to_contiguous_numpy_3d(a)
        b_np = _to_contiguous_numpy_3d(b)

        c_flat, argmax_flat = tropical_gemm.minplus_matmul_strided_batched_with_argmax(
            a_np, b_np
        )

        c = torch.from_numpy(np.asarray(c_flat).reshape(batch_size, m, n)).to(a.device)
        argmax = torch.from_numpy(np.asarray(argmax_flat).reshape(batch_size, m, n)).to(
            a.device
        )

        ctx.save_for_backward(argmax)
        ctx.batch_size = batch_size
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return c

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        (argmax,) = ctx.saved_tensors
        batch_size = ctx.batch_size
        k = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.cpu().numpy().astype(np.int32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        grad_a_flat = tropical_gemm.backward_a_strided_batched(grad_c_np, argmax_np, k)
        grad_b_flat = tropical_gemm.backward_b_strided_batched(grad_c_np, argmax_np, k)

        grad_a = torch.from_numpy(np.asarray(grad_a_flat).reshape(batch_size, m, k)).to(
            grad_c.device
        )
        grad_b = torch.from_numpy(np.asarray(grad_b_flat).reshape(batch_size, k, n)).to(
            grad_c.device
        )

        return grad_a, grad_b


class TropicalMaxMulMatmulBatched(torch.autograd.Function):
    """
    Batched MaxMul tropical matrix multiplication.

    Forward: C[b,i,j] = max_k(A[b,i,k] * B[b,k,j])

    Uses multiplicative gradient rule in backward pass.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch_size, m, k = a.shape
        n = b.shape[2]

        a_np = _to_contiguous_numpy_3d(a)
        b_np = _to_contiguous_numpy_3d(b)

        c_flat, argmax_flat = tropical_gemm.maxmul_matmul_strided_batched_with_argmax(
            a_np, b_np
        )

        c = torch.from_numpy(np.asarray(c_flat).reshape(batch_size, m, n)).to(a.device)
        argmax = torch.from_numpy(np.asarray(argmax_flat).reshape(batch_size, m, n)).to(
            a.device
        )

        # Save inputs for backward pass
        ctx.save_for_backward(
            torch.from_numpy(a_np.copy()), torch.from_numpy(b_np.copy()), argmax
        )
        ctx.batch_size = batch_size
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return c

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        a, b, argmax = ctx.saved_tensors
        batch_size = ctx.batch_size
        k_dim = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)
        a_np = a.numpy().astype(np.float32)
        b_np = b.numpy().astype(np.float32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        grad_a_flat = tropical_gemm.maxmul_backward_a_strided_batched(
            grad_c_np, argmax_np, b_np
        )
        grad_b_flat = tropical_gemm.maxmul_backward_b_strided_batched(
            grad_c_np, argmax_np, a_np
        )

        grad_a = torch.from_numpy(np.asarray(grad_a_flat).reshape(batch_size, m, k_dim)).to(
            grad_c.device
        )
        grad_b = torch.from_numpy(np.asarray(grad_b_flat).reshape(batch_size, k_dim, n)).to(
            grad_c.device
        )

        return grad_a, grad_b


# ===========================================================================
# Batched GPU Autograd Functions
# ===========================================================================

# Check if GPU batched functions are available
_GPU_BATCHED_AVAILABLE = hasattr(tropical_gemm, "maxplus_matmul_gpu_strided_batched_with_argmax")


class TropicalMaxPlusMatmulBatchedGPU(torch.autograd.Function):
    """
    GPU-accelerated batched MaxPlus tropical matrix multiplication.

    Uses optimized Rust CUDA backend via the batched GPU functions.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch_size, m, k = a.shape
        n = b.shape[2]

        # Rust binding returns flattened row-major buffers (it transposes on download).
        a_np = _to_contiguous_numpy_3d(a)
        b_np = _to_contiguous_numpy_3d(b)

        c_flat, argmax_flat = tropical_gemm.maxplus_matmul_gpu_strided_batched_with_argmax(
            a_np, b_np
        )

        c = torch.from_numpy(np.asarray(c_flat).reshape(batch_size, m, n)).to(a.device)
        argmax = torch.from_numpy(np.asarray(argmax_flat).reshape(batch_size, m, n)).to(a.device).long()

        ctx.save_for_backward(argmax)
        ctx.batch_size = batch_size
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return c

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        (argmax,) = ctx.saved_tensors
        batch_size = ctx.batch_size
        k = ctx.k
        m = ctx.m
        n = ctx.n

        # Use GPU scatter operations for backward
        grad_a = torch.zeros(batch_size, m, k, device=grad_c.device, dtype=grad_c.dtype)
        grad_a.scatter_add_(2, argmax, grad_c)

        # For grad_b: need to accumulate over rows
        grad_b = torch.zeros(batch_size, k, n, device=grad_c.device, dtype=grad_c.dtype)
        # Transpose to accumulate correctly
        argmax_t = argmax.transpose(1, 2)  # (batch, n, m)
        grad_c_t = grad_c.transpose(1, 2)  # (batch, n, m)
        grad_b_t = torch.zeros(batch_size, n, k, device=grad_c.device, dtype=grad_c.dtype)
        grad_b_t.scatter_add_(2, argmax_t, grad_c_t)
        grad_b = grad_b_t.transpose(1, 2)

        return grad_a, grad_b


class TropicalMinPlusMatmulBatchedGPU(torch.autograd.Function):
    """
    GPU-accelerated batched MinPlus tropical matrix multiplication.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch_size, m, k = a.shape
        n = b.shape[2]

        # Rust binding returns flattened row-major buffers (it transposes on download).
        a_np = _to_contiguous_numpy_3d(a)
        b_np = _to_contiguous_numpy_3d(b)

        c_flat, argmax_flat = tropical_gemm.minplus_matmul_gpu_strided_batched_with_argmax(
            a_np, b_np
        )

        c = torch.from_numpy(np.asarray(c_flat).reshape(batch_size, m, n)).to(a.device)
        argmax = torch.from_numpy(np.asarray(argmax_flat).reshape(batch_size, m, n)).to(a.device).long()

        ctx.save_for_backward(argmax)
        ctx.batch_size = batch_size
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return c

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        (argmax,) = ctx.saved_tensors
        batch_size = ctx.batch_size
        k = ctx.k
        m = ctx.m
        n = ctx.n

        grad_a = torch.zeros(batch_size, m, k, device=grad_c.device, dtype=grad_c.dtype)
        grad_a.scatter_add_(2, argmax, grad_c)

        argmax_t = argmax.transpose(1, 2)
        grad_c_t = grad_c.transpose(1, 2)
        grad_b_t = torch.zeros(batch_size, n, k, device=grad_c.device, dtype=grad_c.dtype)
        grad_b_t.scatter_add_(2, argmax_t, grad_c_t)
        grad_b = grad_b_t.transpose(1, 2)

        return grad_a, grad_b


class TropicalMaxMulMatmulBatchedGPU(torch.autograd.Function):
    """
    GPU-accelerated batched MaxMul tropical matrix multiplication.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch_size, m, k = a.shape
        n = b.shape[2]

        # Rust binding returns flattened row-major buffers (it transposes on download).
        a_np = _to_contiguous_numpy_3d(a)
        b_np = _to_contiguous_numpy_3d(b)

        c_flat, argmax_flat = tropical_gemm.maxmul_matmul_gpu_strided_batched_with_argmax(
            a_np, b_np
        )

        c = torch.from_numpy(np.asarray(c_flat).reshape(batch_size, m, n)).to(a.device)
        argmax = torch.from_numpy(np.asarray(argmax_flat).reshape(batch_size, m, n)).to(a.device).long()

        ctx.save_for_backward(a.detach(), b.detach(), argmax)
        ctx.batch_size = batch_size
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return c

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        a, b, argmax = ctx.saved_tensors
        batch_size = ctx.batch_size
        k_dim = ctx.k
        m = ctx.m
        n = ctx.n

        # For maxmul: C[b,i,j] = A[b,i,argmax[b,i,j]] * B[b,argmax[b,i,j],j]
        # grad_a[b,i,argmax[b,i,j]] += grad_c[b,i,j] * B[b,argmax[b,i,j],j]
        # grad_b[b,argmax[b,i,j],j] += grad_c[b,i,j] * A[b,i,argmax[b,i,j]]

        # b_winning[b,i,j] = b[b, argmax[b,i,j], j]
        b_winning = torch.gather(b, 1, argmax)  # (batch, m, n) gathering along k axis

        # a_winning[b,i,j] = a[b, i, argmax[b,i,j]]
        a_winning = torch.gather(a, 2, argmax)  # (batch, m, n) gathering along k axis

        # grad_a
        grad_a = torch.zeros(batch_size, m, k_dim, device=grad_c.device, dtype=grad_c.dtype)
        grad_a.scatter_add_(2, argmax, grad_c * b_winning)

        # grad_b
        argmax_t = argmax.transpose(1, 2)  # (batch, n, m)
        grad_c_a_t = (grad_c * a_winning).transpose(1, 2)  # (batch, n, m)
        grad_b_t = torch.zeros(batch_size, n, k_dim, device=grad_c.device, dtype=grad_c.dtype)
        grad_b_t.scatter_add_(2, argmax_t, grad_c_a_t)
        grad_b = grad_b_t.transpose(1, 2)

        return grad_a, grad_b


# ===========================================================================
# Convenience functions
# ===========================================================================


def tropical_maxplus_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    MaxPlus tropical matrix multiplication: C[i,j] = max_k(A[i,k] + B[k,j])

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)
    """
    return TropicalMaxPlusMatmul.apply(a, b)


def tropical_minplus_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    MinPlus tropical matrix multiplication: C[i,j] = min_k(A[i,k] + B[k,j])

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)
    """
    return TropicalMinPlusMatmul.apply(a, b)


def tropical_maxmul_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    MaxMul tropical matrix multiplication: C[i,j] = max_k(A[i,k] * B[k,j])

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)
    """
    return TropicalMaxMulMatmul.apply(a, b)


def tropical_maxplus_matmul_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    GPU-accelerated MaxPlus tropical matrix multiplication.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)

    Raises:
        RuntimeError: If CUDA support is not available
    """
    return TropicalMaxPlusMatmulGPU.apply(a, b)


def tropical_minplus_matmul_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    GPU-accelerated MinPlus tropical matrix multiplication.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)

    Raises:
        RuntimeError: If CUDA support is not available
    """
    return TropicalMinPlusMatmulGPU.apply(a, b)


def tropical_maxmul_matmul_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    GPU-accelerated MaxMul tropical matrix multiplication.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)

    Raises:
        RuntimeError: If CUDA support is not available
    """
    return TropicalMaxMulMatmulGPU.apply(a, b)


# ===========================================================================
# Batched convenience functions
# ===========================================================================


def tropical_maxplus_matmul_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batched MaxPlus tropical matrix multiplication: C[b,i,j] = max_k(A[b,i,k] + B[b,k,j])

    Args:
        a: Input tensor of shape (batch, M, K)
        b: Input tensor of shape (batch, K, N)

    Returns:
        Output tensor of shape (batch, M, N)
    """
    return TropicalMaxPlusMatmulBatched.apply(a, b)


def tropical_minplus_matmul_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batched MinPlus tropical matrix multiplication: C[b,i,j] = min_k(A[b,i,k] + B[b,k,j])

    Args:
        a: Input tensor of shape (batch, M, K)
        b: Input tensor of shape (batch, K, N)

    Returns:
        Output tensor of shape (batch, M, N)
    """
    return TropicalMinPlusMatmulBatched.apply(a, b)


def tropical_maxmul_matmul_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batched MaxMul tropical matrix multiplication: C[b,i,j] = max_k(A[b,i,k] * B[b,k,j])

    Args:
        a: Input tensor of shape (batch, M, K)
        b: Input tensor of shape (batch, K, N)

    Returns:
        Output tensor of shape (batch, M, N)
    """
    return TropicalMaxMulMatmulBatched.apply(a, b)


def tropical_maxplus_matmul_batched_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    GPU-accelerated batched MaxPlus tropical matrix multiplication.

    Args:
        a: Input tensor of shape (batch, M, K)
        b: Input tensor of shape (batch, K, N)

    Returns:
        Output tensor of shape (batch, M, N)
    """
    return TropicalMaxPlusMatmulBatchedGPU.apply(a, b)


def tropical_minplus_matmul_batched_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    GPU-accelerated batched MinPlus tropical matrix multiplication.

    Args:
        a: Input tensor of shape (batch, M, K)
        b: Input tensor of shape (batch, K, N)

    Returns:
        Output tensor of shape (batch, M, N)
    """
    return TropicalMinPlusMatmulBatchedGPU.apply(a, b)


def tropical_maxmul_matmul_batched_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    GPU-accelerated batched MaxMul tropical matrix multiplication.

    Args:
        a: Input tensor of shape (batch, M, K)
        b: Input tensor of shape (batch, K, N)

    Returns:
        Output tensor of shape (batch, M, N)
    """
    return TropicalMaxMulMatmulBatchedGPU.apply(a, b)


__all__ = [
    # CPU autograd functions
    "TropicalMaxPlusMatmul",
    "TropicalMinPlusMatmul",
    "TropicalMaxMulMatmul",
    # GPU autograd functions
    "TropicalMaxPlusMatmulGPU",
    "TropicalMinPlusMatmulGPU",
    "TropicalMaxMulMatmulGPU",
    # Batched CPU autograd functions
    "TropicalMaxPlusMatmulBatched",
    "TropicalMinPlusMatmulBatched",
    "TropicalMaxMulMatmulBatched",
    # Batched GPU autograd functions
    "TropicalMaxPlusMatmulBatchedGPU",
    "TropicalMinPlusMatmulBatchedGPU",
    "TropicalMaxMulMatmulBatchedGPU",
    # Convenience functions
    "tropical_maxplus_matmul",
    "tropical_minplus_matmul",
    "tropical_maxmul_matmul",
    "tropical_maxplus_matmul_gpu",
    "tropical_minplus_matmul_gpu",
    "tropical_maxmul_matmul_gpu",
    # Batched convenience functions
    "tropical_maxplus_matmul_batched",
    "tropical_minplus_matmul_batched",
    "tropical_maxmul_matmul_batched",
    "tropical_maxplus_matmul_batched_gpu",
    "tropical_minplus_matmul_batched_gpu",
    "tropical_maxmul_matmul_batched_gpu",
    # GPU availability flag
    "GPU_AVAILABLE",
    # DLPack availability flag
    "_DLPACK_AVAILABLE",
    # GPU batched availability flag
    "_GPU_BATCHED_AVAILABLE",
]
