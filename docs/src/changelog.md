# Changelog

All notable changes to tropical-gemm.

## [0.2.0]

### Added
- **2D output functions**: New `*_matmul_2d` variants that return properly shaped 2D arrays instead of flattened 1D output. Available for all semirings (maxplus, minplus, maxmul) and data types (f32, f64, i32, i64):
  - `maxplus_matmul_2d`, `minplus_matmul_2d`, `maxmul_matmul_2d` (f32)
  - `maxplus_matmul_2d_f64`, `minplus_matmul_2d_f64`, `maxmul_matmul_2d_f64`
  - `maxplus_matmul_2d_i32`, `minplus_matmul_2d_i32`, `maxmul_matmul_2d_i32`
  - `maxplus_matmul_2d_i64`, `minplus_matmul_2d_i64`, `maxmul_matmul_2d_i64`
- mdBook documentation
- Comprehensive architecture documentation
- Performance tuning guide
- Troubleshooting guide

### Changed
- **GIL release during compute**: All CPU functions now release Python's GIL during heavy computation, allowing other Python threads to run concurrently. This improves performance in multi-threaded Python applications.

### Fixed
- **Batched CPU path copies**: Fixed unnecessary memory copies in batched PyTorch operations by using `np.asarray()` instead of `np.array()` for zero-copy array creation when possible.

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
