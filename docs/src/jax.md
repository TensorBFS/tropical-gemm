# JAX Integration

tropical-gemm provides Python bindings with full JAX autodiff support via `custom_vjp`.

## Installation

```bash
# From PyPI (requires Python >= 3.10)
pip install tropical-gemm[jax]

# For GPU support (requires CUDA toolkit)
pip install maturin
git clone https://github.com/TensorBFS/tropical-gemm
cd tropical-gemm/crates/tropical-gemm-python
maturin develop --features cuda
```

## Basic Usage

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
    # Check GPU availability
    GPU_AVAILABLE,
)

# Create arrays
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (100, 50))
b = jax.random.normal(jax.random.PRNGKey(43), (50, 80))

# MaxPlus: C[i,j] = max_k(A[i,k] + B[k,j])
c = tropical_maxplus_matmul(a, b)

# MinPlus: C[i,j] = min_k(A[i,k] + B[k,j])
c = tropical_minplus_matmul(a, b)

# MaxMul: C[i,j] = max_k(A[i,k] * B[k,j])
c = tropical_maxmul_matmul(a, b)
```

## Automatic Differentiation

JAX's autodiff works seamlessly with tropical operations:

```python
import jax
from jax import grad
from tropical_gemm.jax import tropical_maxplus_matmul

# Define a loss function
def loss_fn(a, b):
    c = tropical_maxplus_matmul(a, b)
    return c.sum()

# Create inputs
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (32, 64))
b = jax.random.normal(jax.random.PRNGKey(43), (64, 48))

# Compute gradients
grad_a = grad(loss_fn, argnums=0)(a, b)
grad_b = grad(loss_fn, argnums=1)(a, b)

print(f"grad_a shape: {grad_a.shape}")  # (32, 64)
print(f"grad_b shape: {grad_b.shape}")  # (64, 48)

# Both gradients at once
grads = grad(loss_fn, argnums=(0, 1))(a, b)
```

## JIT Compilation

All tropical operations are compatible with `jax.jit`:

```python
import jax
from jax import jit
from tropical_gemm.jax import tropical_maxplus_matmul

@jit
def compute(a, b):
    return tropical_maxplus_matmul(a, b)

# First call compiles, subsequent calls are fast
c = compute(a, b)
```

JIT also works with gradients:

```python
@jit
def loss_and_grad(a, b):
    def loss_fn(a, b):
        return tropical_maxplus_matmul(a, b).sum()
    return loss_fn(a, b), grad(loss_fn, argnums=(0, 1))(a, b)

loss, (grad_a, grad_b) = loss_and_grad(a, b)
```

## Batched Operations

For 3D tensors with a batch dimension:

```python
from tropical_gemm.jax import tropical_maxplus_matmul_batched

# Batched: (batch, M, K) @ (batch, K, N) -> (batch, M, N)
key = jax.random.PRNGKey(42)
a_batch = jax.random.normal(key, (4, 32, 64))
b_batch = jax.random.normal(jax.random.PRNGKey(43), (4, 64, 48))

c_batch = tropical_maxplus_matmul_batched(a_batch, b_batch)
print(c_batch.shape)  # (4, 32, 48)

# Gradients work with batched operations
def batched_loss(a, b):
    return tropical_maxplus_matmul_batched(a, b).sum()

grad_a = grad(batched_loss, argnums=0)(a_batch, b_batch)
print(grad_a.shape)  # (4, 32, 64)
```

## GPU Acceleration

For larger matrices, use GPU-accelerated functions:

```python
from tropical_gemm.jax import (
    tropical_maxplus_matmul_gpu,
    tropical_maxplus_matmul_batched_gpu,
    GPU_AVAILABLE,
)

if GPU_AVAILABLE:
    # 2D GPU operations
    a = jax.random.normal(jax.random.PRNGKey(0), (1024, 512))
    b = jax.random.normal(jax.random.PRNGKey(1), (512, 1024))

    c = tropical_maxplus_matmul_gpu(a, b)

    # 3D batched GPU operations
    a_batch = jax.random.normal(jax.random.PRNGKey(0), (8, 256, 512))
    b_batch = jax.random.normal(jax.random.PRNGKey(1), (8, 512, 256))

    c_batch = tropical_maxplus_matmul_batched_gpu(a_batch, b_batch)
```

## Available Functions

**2D Operations** (M, K) @ (K, N) -> (M, N)

| CPU Function | GPU Function | Operation |
|--------------|--------------|-----------|
| `tropical_maxplus_matmul` | `tropical_maxplus_matmul_gpu` | max_k(A[i,k] + B[k,j]) |
| `tropical_minplus_matmul` | `tropical_minplus_matmul_gpu` | min_k(A[i,k] + B[k,j]) |
| `tropical_maxmul_matmul` | `tropical_maxmul_matmul_gpu` | max_k(A[i,k] * B[k,j]) |

**3D Batched Operations** (B, M, K) @ (B, K, N) -> (B, M, N)

| CPU Function | GPU Function | Operation |
|--------------|--------------|-----------|
| `tropical_maxplus_matmul_batched` | `tropical_maxplus_matmul_batched_gpu` | max_k(A[b,i,k] + B[b,k,j]) |
| `tropical_minplus_matmul_batched` | `tropical_minplus_matmul_batched_gpu` | min_k(A[b,i,k] + B[b,k,j]) |
| `tropical_maxmul_matmul_batched` | `tropical_maxmul_matmul_batched_gpu` | max_k(A[b,i,k] * B[b,k,j]) |

## Gradient Semantics

The gradient computation follows the same rules as PyTorch:

### MaxPlus / MinPlus (Additive Rule)

For `C[i,j] = max_k(A[i,k] + B[k,j])`, let `k* = argmax_k(A[i,k] + B[k,j])`:

- `grad_A[i,k*] += grad_C[i,j]`
- `grad_B[k*,j] += grad_C[i,j]`

The gradient is **sparse** - only the winning index contributes.

### MaxMul (Multiplicative Rule)

For `C[i,j] = max_k(A[i,k] * B[k,j])`, let `k* = argmax_k(A[i,k] * B[k,j])`:

- `grad_A[i,k*] += grad_C[i,j] * B[k*,j]`
- `grad_B[k*,j] += grad_C[i,j] * A[i,k*]`

## Training Example

```python
import jax
import jax.numpy as jnp
from jax import grad, jit
from tropical_gemm.jax import tropical_maxplus_matmul

# Initialize parameters
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (64, 128))
b = jax.random.normal(jax.random.PRNGKey(43), (128, 32))
target = jax.random.normal(jax.random.PRNGKey(44), (64, 32))

@jit
def loss_fn(a, b):
    c = tropical_maxplus_matmul(a, b)
    return jnp.mean((c - target) ** 2)

@jit
def update(a, b, lr=0.1):
    loss, (grad_a, grad_b) = jax.value_and_grad(loss_fn, argnums=(0, 1))(a, b)
    a = a - lr * grad_a
    b = b - lr * grad_b
    return a, b, loss

# Training loop
for step in range(100):
    a, b, loss = update(a, b)
    if step % 20 == 0:
        print(f"Step {step}: loss = {loss:.4f}")
```

## Graph Algorithms

### Shortest Path (MinPlus)

```python
import jax.numpy as jnp
from tropical_gemm.jax import tropical_minplus_matmul

# Adjacency matrix (inf = no edge)
inf = jnp.inf
adj = jnp.array([
    [0.0, 1.0, inf, 4.0],
    [inf, 0.0, 2.0, inf],
    [inf, inf, 0.0, 1.0],
    [inf, inf, inf, 0.0],
])

# 2-hop shortest paths
two_hop = tropical_minplus_matmul(adj, adj)

# 3-hop shortest paths
three_hop = tropical_minplus_matmul(two_hop, adj)
```

### Longest Path (MaxPlus)

```python
import jax.numpy as jnp
from tropical_gemm.jax import tropical_maxplus_matmul

# Edge weights for critical path analysis
neg_inf = -jnp.inf
adj = jnp.array([
    [0.0, 3.0, 2.0, neg_inf],
    [neg_inf, 0.0, neg_inf, 4.0],
    [neg_inf, neg_inf, 0.0, 5.0],
    [neg_inf, neg_inf, neg_inf, 0.0],
])

# 2-hop longest paths
two_hop = tropical_maxplus_matmul(adj, adj)
```

## Comparison with PyTorch

Both JAX and PyTorch integrations produce identical results:

```python
import torch
import jax.numpy as jnp
import numpy as np

from tropical_gemm.pytorch import tropical_maxplus_matmul as torch_maxplus
from tropical_gemm.jax import tropical_maxplus_matmul as jax_maxplus

# Same input data
a_np = np.random.randn(32, 64).astype(np.float32)
b_np = np.random.randn(64, 48).astype(np.float32)

# PyTorch
a_torch = torch.from_numpy(a_np)
b_torch = torch.from_numpy(b_np)
c_torch = torch_maxplus(a_torch, b_torch).numpy()

# JAX
a_jax = jnp.array(a_np)
b_jax = jnp.array(b_np)
c_jax = np.array(jax_maxplus(a_jax, b_jax))

# Results match
np.testing.assert_allclose(c_torch, c_jax, rtol=1e-5)
```

## Implementation Notes

The JAX integration uses:
- `jax.custom_vjp` for custom vector-Jacobian products
- `jax.pure_callback` for calling the Rust backend
- DLPack for zero-copy GPU tensor exchange

This ensures efficient execution while maintaining full compatibility with JAX's transformation system (jit, grad, vmap, etc.).
