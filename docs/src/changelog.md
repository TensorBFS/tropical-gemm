# Changelog

All notable changes to tropical-gemm.

## [Unreleased]

### Added
- **JAX Integration** - Full JAX support with `custom_vjp` for automatic differentiation
  - 2D operations: `tropical_maxplus_matmul`, `tropical_minplus_matmul`, `tropical_maxmul_matmul`
  - 3D batched operations: `tropical_*_matmul_batched`
  - GPU operations via DLPack zero-copy interface
  - Compatible with `jax.jit`, `jax.grad`, and other JAX transformations
- **Batched Operations** - Native support for 3D tensors with batch dimension
  - PyTorch: `tropical_*_matmul_batched` and `tropical_*_matmul_batched_gpu`
  - JAX: `tropical_*_matmul_batched` and `tropical_*_matmul_batched_gpu`
  - Full autograd/autodiff support for batched operations
- **Cross-Validation** - Benchmark tool to validate PyTorch vs JAX implementations
- mdBook documentation
- Comprehensive architecture documentation
- Performance tuning guide
- Troubleshooting guide

## [0.1.0] - Initial Release

### Features
- High-performance tropical matrix multiplication
- Support for three semirings: MaxPlus, MinPlus, MaxMul
- SIMD acceleration (AVX-512, AVX2, SSE4.1, NEON)
- CUDA GPU acceleration
- Argmax tracking for backpropagation
- Python bindings with NumPy support
- PyTorch autograd integration

### Crates
- `tropical-gemm`: Core CPU implementation
- `tropical-gemm-cuda`: CUDA GPU backend
- `tropical-gemm-python`: Python bindings

### Performance
- BLIS-style 5-loop cache blocking
- Runtime SIMD dispatch
- GPU speedup up to 800x for large matrices

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.2.0 | 2026 | JAX integration, batched operations |
| 0.1.0 | 2024 | Initial release |

## Migration Guides

### From NumPy Implementation

If migrating from a pure NumPy tropical matrix multiplication:

```python
# Before (NumPy)
def maxplus_matmul_numpy(a, b):
    m, k = a.shape
    n = b.shape[1]
    c = np.full((m, n), -np.inf)
    for i in range(m):
        for j in range(n):
            for kk in range(k):
                c[i, j] = max(c[i, j], a[i, kk] + b[kk, j])
    return c

# After (tropical-gemm)
import tropical_gemm
c = tropical_gemm.maxplus_matmul(a, b)
```

### API Changes

No breaking changes yet (this is the first release).
