# Changelog

All notable changes to tropical-gemm.

## [Unreleased]

### Added
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
