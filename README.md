# tropical-gemm

[![CI](https://github.com/TensorBFS/tropical-gemm/actions/workflows/ci.yml/badge.svg)](https://github.com/TensorBFS/tropical-gemm/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/TensorBFS/tropical-gemm/branch/main/graph/badge.svg)](https://codecov.io/gh/TensorBFS/tropical-gemm)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://tensorbfs.github.io/tropical-gemm/tropical_gemm/)

High-performance tropical matrix multiplication in Rust with SIMD and CUDA backends.

## Features

- **Multiple Semirings**: MaxPlus, MinPlus, MaxMul, AndOr, Counting
- **SIMD Acceleration**: AVX-512, AVX2, SSE4.1, NEON auto-detection
- **CUDA Backend**: GPU-accelerated kernels via cudarc/NVRTC
- **Argmax Tracking**: For backpropagation in tropical neural networks
- **Cache-Optimized**: BLIS-style 5-loop blocking

## Installation

```toml
[dependencies]
tropical-gemm = "0.1"

# With CUDA support
tropical-gemm = { version = "0.1", features = ["cuda"] }
```

## Usage

### CPU (SIMD-optimized)

```rust
use tropical_gemm::prelude::*;

let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2

// C[i,j] = max_k(A[i,k] + B[k,j])
let c = tropical_matmul::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2);
```

### GPU (CUDA)

```rust
use tropical_gemm::cuda::{tropical_matmul_gpu, CudaContext, GpuMatrix};
use tropical_types::TropicalMaxPlus;

// One-shot API (handles memory automatically)
let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)?;

// Persistent context API (reuse for multiple operations)
let ctx = CudaContext::new()?;
let a_gpu = GpuMatrix::from_host_row_major(&ctx, &a, m, k)?;
let b_gpu = GpuMatrix::from_host_row_major(&ctx, &b, k, n)?;
let c_gpu = tropical_matmul_gpu_with_ctx::<TropicalMaxPlus<f32>>(&ctx, &a_gpu, &b_gpu)?;
let c = c_gpu.to_host_row_major(&ctx)?;
```

## Benchmark Results

Tested on NVIDIA RTX A4500 (Ampere, sm_86).

### GPU vs CPU Performance

| Size | CPU SIMD (ms) | GPU Kernel (ms) | Speedup |
|------|---------------|-----------------|---------|
| 256  | 4.1           | 0.032           | **128x** |
| 512  | 32.8          | 0.086           | **381x** |
| 1024 | 262.3         | 0.358           | **733x** |
| 2048 | 2091.6        | 2.510           | **833x** |

### Rust CUDA vs C Reference (TropicalGemm_Cuda)

Comparison with the optimized C library from [TropicalGemm_Cuda](https://github.com/ArrogantGao/TropicalGemm_Cuda):

| Size | C Library (ms) | Rust CUDA (ms) | Ratio |
|------|----------------|----------------|-------|
| 256  | 0.028          | 0.032          | 1.14x |
| 512  | 0.074          | 0.086          | 1.16x |
| 1024 | 0.315          | 0.358          | 1.14x |
| 2048 | 2.224          | 2.509          | 1.13x |

The C library is ~13-16% faster due to pre-compiled PTX vs runtime compilation. Both implementations use the same optimized kernel algorithm with shared memory tiling.

### All Semirings (GPU Kernel Time)

| Size | MaxPlus (ms) | MinPlus (ms) | MaxMul (ms) |
|------|--------------|--------------|-------------|
| 256  | 0.032        | 0.032        | 0.032       |
| 512  | 0.086        | 0.087        | 0.086       |
| 1024 | 0.358        | 0.358        | 0.359       |
| 2048 | 2.509        | 2.514        | 2.521       |

## Crate Structure

```
tropical-gemm/
├── tropical-types      # Semiring type definitions
├── tropical-gemm-core  # Portable GEMM algorithms
├── tropical-gemm-simd  # SIMD kernels (AVX-512, AVX2, NEON)
├── tropical-gemm-cuda  # CUDA backend
└── tropical-gemm       # Unified API
```

## Running Benchmarks

```bash
# CPU benchmark
cargo run --release --example bench_rust -p tropical-gemm

# CUDA vs CPU benchmark
LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
  cargo run --release --example bench_cuda_vs_cpu -p tropical-gemm-cuda
```

## License

MIT
