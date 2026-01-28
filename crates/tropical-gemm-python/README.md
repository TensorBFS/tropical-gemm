# tropical-gemm

Fast tropical matrix multiplication with automatic differentiation support.

## Installation

```bash
# From PyPI
pip install tropical-gemm

# With PyTorch support (for automatic differentiation)
pip install tropical-gemm[torch]

# With JAX support (requires Python >= 3.10)
pip install tropical-gemm[jax]

# For GPU support (requires CUDA toolkit)
pip install maturin
git clone https://github.com/TensorBFS/tropical-gemm
cd tropical-gemm/crates/tropical-gemm-python
maturin develop --features cuda
```

## Quick Start

```python
import numpy as np
import tropical_gemm

# Create matrices
a = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]], dtype=np.float32)
b = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]], dtype=np.float32)

# MaxPlus tropical matmul: C[i,j] = max_k(A[i,k] + B[k,j])
c = tropical_gemm.maxplus_matmul(a, b)
print("MaxPlus result:", c)

# MinPlus tropical matmul: C[i,j] = min_k(A[i,k] + B[k,j])
c = tropical_gemm.minplus_matmul(a, b)
print("MinPlus result:", c)

# MaxMul tropical matmul: C[i,j] = max_k(A[i,k] * B[k,j])
c = tropical_gemm.maxmul_matmul(a, b)
print("MaxMul result:", c)

# With argmax for backpropagation
c, argmax = tropical_gemm.maxplus_matmul_with_argmax(a, b)
print("Result:", c)
print("Argmax:", argmax)

# GPU acceleration (if compiled with CUDA)
if tropical_gemm.cuda_available():
    c = tropical_gemm.maxplus_matmul_gpu(a, b)
    c = tropical_gemm.minplus_matmul_gpu(a, b)
    c = tropical_gemm.maxmul_matmul_gpu(a, b)
```

## PyTorch Integration

The package includes a `pytorch` submodule with pre-built autograd functions:

```python
import torch
from tropical_gemm.pytorch import (
    # 2D CPU operations
    tropical_maxplus_matmul,
    tropical_minplus_matmul,
    tropical_maxmul_matmul,
    # 3D batched CPU operations
    tropical_maxplus_matmul_batched,
    tropical_minplus_matmul_batched,
    tropical_maxmul_matmul_batched,
    # GPU operations (requires CUDA)
    tropical_maxplus_matmul_gpu,
    tropical_minplus_matmul_gpu,
    tropical_maxmul_matmul_gpu,
    tropical_maxplus_matmul_batched_gpu,
    tropical_minplus_matmul_batched_gpu,
    tropical_maxmul_matmul_batched_gpu,
    GPU_AVAILABLE,
)

# Create tensors with gradient tracking
a = torch.randn(100, 50, requires_grad=True)
b = torch.randn(50, 80, requires_grad=True)

# Forward pass
c = tropical_maxplus_matmul(a, b)

# Backward pass - gradients computed automatically
loss = c.sum()
loss.backward()

print(f"grad_a shape: {a.grad.shape}")  # (100, 50)
print(f"grad_b shape: {b.grad.shape}")  # (50, 80)

# Batched operations for 3D tensors
a_batch = torch.randn(4, 32, 64, requires_grad=True)
b_batch = torch.randn(4, 64, 48, requires_grad=True)
c_batch = tropical_maxplus_matmul_batched(a_batch, b_batch)  # (4, 32, 48)

# Use GPU for larger matrices
if GPU_AVAILABLE:
    c = tropical_maxplus_matmul_gpu(a, b)
    c_batch = tropical_maxplus_matmul_batched_gpu(a_batch.cuda(), b_batch.cuda())
```

## JAX Integration

The package includes a `jax` submodule with differentiable operations using `custom_vjp`:

```python
import jax
import jax.numpy as jnp
from tropical_gemm.jax import (
    # 2D operations
    tropical_maxplus_matmul,
    tropical_minplus_matmul,
    tropical_maxmul_matmul,
    # 3D batched operations
    tropical_maxplus_matmul_batched,
    tropical_minplus_matmul_batched,
    tropical_maxmul_matmul_batched,
    # GPU operations (requires CUDA)
    tropical_maxplus_matmul_gpu,
    tropical_maxplus_matmul_batched_gpu,
    GPU_AVAILABLE,
)

# Create arrays
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (100, 50))
b = jax.random.normal(jax.random.PRNGKey(43), (50, 80))

# Forward pass
c = tropical_maxplus_matmul(a, b)

# Gradients via JAX autodiff
def loss_fn(a, b):
    return tropical_maxplus_matmul(a, b).sum()

grad_a = jax.grad(loss_fn, argnums=0)(a, b)
grad_b = jax.grad(loss_fn, argnums=1)(a, b)

# JIT compilation works seamlessly
@jax.jit
def compute(a, b):
    return tropical_maxplus_matmul(a, b)

c = compute(a, b)

# Batched operations
a_batch = jax.random.normal(key, (4, 32, 64))
b_batch = jax.random.normal(jax.random.PRNGKey(43), (4, 64, 48))
c_batch = tropical_maxplus_matmul_batched(a_batch, b_batch)  # (4, 32, 48)

# Use GPU for larger matrices
if GPU_AVAILABLE:
    c = tropical_maxplus_matmul_gpu(a, b)
```

### Gradient Semantics

The gradient computation depends on the semiring type:

**MaxPlus/MinPlus (additive rule):**
- `grad_A[i,k] = grad_C[i,j]` if `k == argmax[i,j]`
- `grad_B[k,j] = grad_C[i,j]` if `k == argmax[i,j]`

**MaxMul (multiplicative rule):**
- `grad_A[i,k] = grad_C[i,j] * B[k,j]` if `k == argmax[i,j]`
- `grad_B[k,j] = grad_C[i,j] * A[i,k]` if `k == argmax[i,j]`

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `maxplus_matmul(a, b)` | MaxPlus: C[i,j] = max_k(A[i,k] + B[k,j]) |
| `minplus_matmul(a, b)` | MinPlus: C[i,j] = min_k(A[i,k] + B[k,j]) |
| `maxmul_matmul(a, b)` | MaxMul: C[i,j] = max_k(A[i,k] * B[k,j]) |
| `*_with_argmax(a, b)` | Returns (result, argmax) for backpropagation |
| `backward_a(grad_c, argmax, k)` | Gradient w.r.t. A (additive rule) |
| `backward_b(grad_c, argmax, k)` | Gradient w.r.t. B (additive rule) |
| `maxmul_backward_a(grad_c, argmax, b)` | Gradient w.r.t. A (multiplicative rule) |
| `maxmul_backward_b(grad_c, argmax, a)` | Gradient w.r.t. B (multiplicative rule) |

### GPU Functions (requires CUDA)

| Function | Description |
|----------|-------------|
| `cuda_available()` | Check if CUDA support is available |
| `maxplus_matmul_gpu(a, b)` | GPU MaxPlus matmul |
| `minplus_matmul_gpu(a, b)` | GPU MinPlus matmul |
| `maxmul_matmul_gpu(a, b)` | GPU MaxMul matmul |
| `*_gpu_with_argmax(a, b)` | GPU matmul with argmax tracking |

### Data Types

All functions support:
- `f32` (default): `maxplus_matmul`, etc.
- `f64`: `maxplus_matmul_f64`, etc.
- `i32`: `maxplus_matmul_i32`, etc.
- `i64`: `maxplus_matmul_i64`, etc.

## License

MIT
