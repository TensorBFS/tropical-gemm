# PyTorch Integration

tropical-gemm provides Python bindings for seamless PyTorch integration.

## Installation

```bash
cd crates/tropical-gemm-python
pip install maturin torch
maturin develop --release
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
```

## PyTorch Autograd Function

For differentiable tropical operations, implement a custom `torch.autograd.Function`:

```python
import torch
import numpy as np
import tropical_gemm

class TropicalMaxPlusMatmul(torch.autograd.Function):
    """Differentiable MaxPlus: C[i,j] = max_k(A[i,k] + B[k,j])"""

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

## Using the Autograd Function

```python
# Create tensors with gradients
a = torch.randn(4, 5, requires_grad=True)
b = torch.randn(5, 3, requires_grad=True)

# Forward pass
c = TropicalMaxPlusMatmul.apply(a, b)

# Backward pass
loss = c.sum()
loss.backward()

print(a.grad.shape)  # (4, 5)
print(b.grad.shape)  # (5, 3)
```

## Training Example

```python
import torch

# Learnable parameters
a = torch.randn(4, 5, requires_grad=True)
b = torch.randn(5, 3, requires_grad=True)
target = torch.randn(4, 3)

optimizer = torch.optim.Adam([a, b], lr=0.1)

for step in range(100):
    optimizer.zero_grad()

    # Forward
    c = TropicalMaxPlusMatmul.apply(a, b)

    # Loss
    loss = ((c - target) ** 2).mean()

    # Backward
    loss.backward()

    # Update
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
```

## Gradient Semantics

| Semiring | Forward | Backward Rule |
|----------|---------|---------------|
| MaxPlus | `max_k(A + B)` | `∂C/∂A = 1` at argmax |
| MinPlus | `min_k(A + B)` | `∂C/∂A = 1` at argmin |
| MaxMul | `max_k(A × B)` | `∂C/∂A = B` at argmax |

The gradients are **sparse**: only the winning index contributes.

## Complete Example

See `crates/tropical-gemm-python/examples/pytorch_tropical.py` for:
- All three autograd functions (MaxPlus, MinPlus, MaxMul)
- Gradient verification tests
- Shortest/longest path examples
- Optimization demos
