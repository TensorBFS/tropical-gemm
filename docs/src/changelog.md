# Changelog

All notable changes to tropical-gemm.

## [Unreleased]

### Added
- **CUDA: on-disk CUBIN cache (issue #41).** Kernels are compiled straight to a CUBIN
  (native SASS) for the device's architecture and cached on disk
  (`$XDG_CACHE_HOME` / `~/.cache/tropical-gemm/`), keyed on a stable hash of source +
  compile flags + GPU arch + NVRTC version. A warm `CudaContext::new()` drops from ~10.7 s
  to ~0.13 s (skips both the NVRTC compile and the driver's PTX→SASS JIT). The cache is
  load-validated and self-healing.

### Changed
- **CUDA: interior/boundary-split tile loads (issue #40).** The global→shared tile copies
  now give fully-interior blocks a bounds-check-free fast path; only edge blocks (last
  block-row/col, final K-tile) run the per-element `row < M && col < K` predicate. The old
  code ran that predicate on every element of every block — the entire throughput gap vs the
  C reference. On A40/CUDA 12.8 the kernels go from ~3-6% slower to **~3% faster** than
  [TropicalGemm_Cuda](https://github.com/ArrogantGao/TropicalGemm_Cuda), with identical
  results (full CUDA suite passes). The shared `LOAD_A_TILE` / `LOAD_B_TILE` /
  `STORE_C*_TILE` macros apply it across all f32/f64/i32/i64, argmax, and batched kernels.
- **CUDA: migrated to cudarc 0.19.7.** `CUDARC_CUDA_VERSION` selects the CUDA toolkit at
  build time, so `cargo check` works with no toolkit installed and no `nvcc` is needed at
  build time.
- **CUDA: `device_name()` now reports the real GPU model** (via `cuDeviceGetName`).

### Notes
- **Arch flags do not affect CUDA throughput (issue #40).** Compiling to an arch-targeted
  CUBIN (`-arch=sm_XX`) runs identically to driver-JIT'd PTX; the CUBIN build is purely a
  startup-latency win (issue #41). The throughput gap vs the C reference was the per-element
  bounds check above, not the toolchain or the CUDA version.

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
