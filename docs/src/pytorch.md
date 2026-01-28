# PyTorch Integration

tropical-gemm provides Python bindings with full PyTorch autograd support.

## Installation

```bash
# From PyPI
pip install tropical-gemm

# With PyTorch support (recommended)
pip install tropical-gemm[torch]

# For GPU support (requires CUDA toolkit)
pip install maturin
git clone https://github.com/TensorBFS/tropical-gemm
cd tropical-gemm/crates/tropical-gemm-python
maturin develop --features cuda
```

## Basic NumPy Usage

```python
import numpy as np
import tropical_gemm

a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

# MaxPlus: C[i,j] = max_k(A[i,k] + B[k,j])
c = tropical_gemm.maxplus_matmul(a, b)

# MinPlus: C[i,j] = min_k(A[i,k] + B[k,j])
c = tropical_gemm.minplus_matmul(a, b)

# MaxMul: C[i,j] = max_k(A[i,k] * B[k,j])
c = tropical_gemm.maxmul_matmul(a, b)

# With argmax tracking for backpropagation
c, argmax = tropical_gemm.maxplus_matmul_with_argmax(a, b)
```

## PyTorch Module (Recommended)

The `tropical_gemm.pytorch` module provides pre-built autograd functions:

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
    # 2D GPU operations (requires CUDA)
    tropical_maxplus_matmul_gpu,
    tropical_minplus_matmul_gpu,
    tropical_maxmul_matmul_gpu,
    # 3D batched GPU operations (requires CUDA)
    tropical_maxplus_matmul_batched_gpu,
    tropical_minplus_matmul_batched_gpu,
    tropical_maxmul_matmul_batched_gpu,
    # Check GPU availability
    GPU_AVAILABLE,
)

# Create tensors with gradient tracking
a = torch.randn(100, 50, requires_grad=True)
b = torch.randn(50, 80, requires_grad=True)

# Forward pass - compute tropical matmul
c = tropical_maxplus_matmul(a, b)

# Backward pass - gradients computed automatically
loss = c.sum()
loss.backward()

print(f"grad_a shape: {a.grad.shape}")  # (100, 50)
print(f"grad_b shape: {b.grad.shape}")  # (50, 80)
```

### Batched Operations

For 3D tensors with a batch dimension:

```python
# Batched operations: (batch, M, K) @ (batch, K, N) -> (batch, M, N)
a_batch = torch.randn(4, 32, 64, requires_grad=True)
b_batch = torch.randn(4, 64, 48, requires_grad=True)

c_batch = tropical_maxplus_matmul_batched(a_batch, b_batch)
print(c_batch.shape)  # (4, 32, 48)

# Gradients work with batched operations
loss = c_batch.sum()
loss.backward()
print(a_batch.grad.shape)  # (4, 32, 64)
```

### GPU Acceleration

For larger matrices, use GPU-accelerated functions:

```python
if GPU_AVAILABLE:
    # 2D GPU operations
    a = torch.randn(1024, 512, requires_grad=True)
    b = torch.randn(512, 1024, requires_grad=True)

    c = tropical_maxplus_matmul_gpu(a, b)
    loss = c.sum()
    loss.backward()  # Gradients still work!

    # 3D batched GPU operations
    a_batch = torch.randn(8, 256, 512, requires_grad=True, device="cuda")
    b_batch = torch.randn(8, 512, 256, requires_grad=True, device="cuda")

    c_batch = tropical_maxplus_matmul_batched_gpu(a_batch, b_batch)
    c_batch.sum().backward()
```

### Available Functions

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

## Training Example

```python
import torch
from tropical_gemm.pytorch import tropical_maxplus_matmul

# Learnable parameters
a = torch.randn(64, 128, requires_grad=True)
b = torch.randn(128, 32, requires_grad=True)
target = torch.randn(64, 32)

optimizer = torch.optim.Adam([a, b], lr=0.1)

for step in range(100):
    optimizer.zero_grad()

    # Forward - tropical matmul
    c = tropical_maxplus_matmul(a, b)

    # Loss
    loss = ((c - target) ** 2).mean()

    # Backward - gradients flow through tropical operation
    loss.backward()

    # Update parameters
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
```

## Gradient Semantics

The gradient computation depends on the semiring type:

### MaxPlus / MinPlus (Additive Rule)

For `C[i,j] = max_k(A[i,k] + B[k,j])`, let `k* = argmax_k(A[i,k] + B[k,j])`:

- `grad_A[i,k*] += grad_C[i,j]`
- `grad_B[k*,j] += grad_C[i,j]`

The gradient is **sparse** - only the winning index contributes.

### MaxMul (Multiplicative Rule)

For `C[i,j] = max_k(A[i,k] * B[k,j])`, let `k* = argmax_k(A[i,k] * B[k,j])`:

- `grad_A[i,k*] += grad_C[i,j] * B[k*,j]`
- `grad_B[k*,j] += grad_C[i,j] * A[i,k*]`

| Semiring | Forward | Backward Rule |
|----------|---------|---------------|
| MaxPlus | max_k(A + B) | ∂C/∂A = 1 at argmax |
| MinPlus | min_k(A + B) | ∂C/∂A = 1 at argmin |
| MaxMul | max_k(A × B) | ∂C/∂A = B at argmax |

## Graph Algorithms

### Shortest Path (MinPlus)

```python
import torch
from tropical_gemm.pytorch import tropical_minplus_matmul

# Adjacency matrix (inf = no edge)
inf = float("inf")
adj = torch.tensor([
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
import torch
from tropical_gemm.pytorch import tropical_maxplus_matmul

# Edge weights for critical path analysis
neg_inf = float("-inf")
adj = torch.tensor([
    [0.0, 3.0, 2.0, neg_inf],
    [neg_inf, 0.0, neg_inf, 4.0],
    [neg_inf, neg_inf, 0.0, 5.0],
    [neg_inf, neg_inf, neg_inf, 0.0],
])

# 2-hop longest paths
two_hop = tropical_maxplus_matmul(adj, adj)
```

## Custom Autograd Function (Advanced)

If you need custom behavior, you can still define your own autograd function:

```python
import torch
import numpy as np
import tropical_gemm

class TropicalMaxPlusMatmul(torch.autograd.Function):
    """Custom differentiable MaxPlus: C[i,j] = max_k(A[i,k] + B[k,j])"""

    @staticmethod
    def forward(ctx, a, b):
        m, k = a.shape
        n = b.shape[1]

        # Convert to NumPy
        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        # Forward pass with argmax tracking
        c_flat, argmax_flat = tropical_gemm.maxplus_matmul_with_argmax(a_np, b_np)
        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        # Save for backward
        ctx.save_for_backward(torch.from_numpy(argmax_np))
        ctx.k, ctx.m, ctx.n = k, m, n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c):
        argmax, = ctx.saved_tensors
        k, m, n = ctx.k, ctx.m, ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)

        # Backward pass
        grad_a_flat = tropical_gemm.backward_a(grad_c_np, argmax_np, k)
        grad_b_flat = tropical_gemm.backward_b(grad_c_np, argmax_np, k)

        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k)).to(grad_c.device)
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k, n)).to(grad_c.device)

        return grad_a, grad_b
```

## Complete Example

See `crates/tropical-gemm-python/examples/pytorch_tropical.py` for:
- Gradient verification tests
- Shortest/longest path examples
- Optimization demos
- GPU benchmarks
