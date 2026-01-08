# tropical-gemm

Fast tropical matrix multiplication with automatic differentiation support.

## Installation

```bash
# From source (requires Rust toolchain)
cd crates/tropical-gemm-python
pip install maturin
maturin develop

# Or build a wheel
maturin build --release
pip install target/wheels/tropical_gemm-*.whl
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

# With argmax for backpropagation
c, argmax = tropical_gemm.maxplus_matmul_with_argmax(a, b)
print("Result:", c)
print("Argmax:", argmax)
```

## PyTorch Integration

See `examples/pytorch_tropical.py` for a complete example of using tropical GEMM with PyTorch autograd.

```python
import torch
import tropical_gemm

class TropicalMaxPlusMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        a_np = a.detach().numpy()
        b_np = b.detach().numpy()
        c_np, argmax_np = tropical_gemm.maxplus_matmul_with_argmax(a_np, b_np)
        ctx.save_for_backward(torch.from_numpy(argmax_np))
        ctx.k = a.shape[1]
        return torch.from_numpy(c_np)

    @staticmethod
    def backward(ctx, grad_c):
        argmax, = ctx.saved_tensors
        k = ctx.k
        grad_c_np = grad_c.numpy()
        argmax_np = argmax.numpy()
        grad_a = torch.from_numpy(tropical_gemm.backward_a(grad_c_np, argmax_np, k))
        grad_b = torch.from_numpy(tropical_gemm.backward_b(grad_c_np, argmax_np, k))
        return grad_a, grad_b

# Use like a regular PyTorch operation
a = torch.randn(3, 4, requires_grad=True)
b = torch.randn(4, 5, requires_grad=True)
c = TropicalMaxPlusMatmul.apply(a, b)
c.sum().backward()
```

## API Reference

### Functions

- `maxplus_matmul(a, b)` - MaxPlus tropical matmul: C[i,j] = max_k(A[i,k] + B[k,j])
- `minplus_matmul(a, b)` - MinPlus tropical matmul: C[i,j] = min_k(A[i,k] + B[k,j])
- `maxplus_matmul_with_argmax(a, b)` - MaxPlus with argmax indices for backprop
- `minplus_matmul_with_argmax(a, b)` - MinPlus with argmax indices for backprop
- `backward_a(grad_c, argmax, k)` - Compute gradient w.r.t. A
- `backward_b(grad_c, argmax, k)` - Compute gradient w.r.t. B

## License

MIT
