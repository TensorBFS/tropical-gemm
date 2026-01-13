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
- **Batched Operations**: Efficient batch processing for multiple matrices
- **Backward Pass**: Gradient computation for automatic differentiation
- **Python Bindings**: PyTorch integration via PyO3
- **Cache-Optimized**: BLIS-style 5-loop blocking

## Feature Matrix

### Supported Operations by Semiring Type

| Semiring | Scalar | CPU GEMM | CPU Batched | CPU Argmax | CPU Backward | GPU GEMM | GPU Batched | GPU Argmax | GPU Backward |
|----------|--------|----------|-------------|------------|--------------|----------|-------------|------------|--------------|
| `MaxPlus` | `f32` | ✅ SIMD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `MaxPlus` | `f64` | ✅ SIMD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `MaxPlus` | `i32` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |
| `MaxPlus` | `i64` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |
| `MinPlus` | `f32` | ✅ SIMD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `MinPlus` | `f64` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `MinPlus` | `i32` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |
| `MinPlus` | `i64` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |
| `MaxMul` | `f32` | ✅ SIMD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `MaxMul` | `f64` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `MaxMul` | `i32` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `MaxMul` | `i64` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |

**Legend:**
- ✅ SIMD: Optimized with AVX2/AVX-512/NEON vectorization
- ✅: Supported with portable implementation
- N/A: Not applicable (integers don't have gradients)
- ❌: Not yet implemented

### Semiring Definitions

| Type | Addition (⊕) | Multiplication (⊗) | Zero | One | Use Case |
|------|--------------|-------------------|------|-----|----------|
| `MaxPlus<T>` | max | + | -∞ | 0 | Viterbi, longest path |
| `MinPlus<T>` | min | + | +∞ | 0 | Shortest path, Dijkstra |
| `MaxMul<T>` | max | × | 0 | 1 | Probability (non-log) |
| `AndOr` | OR | AND | false | true | Graph reachability |
| `CountingTropical<T,C>` | max+count | +,× | (-∞,0) | (0,1) | Path counting |

## Installation

```toml
[dependencies]
tropical-gemm = "0.1"

# For GPU acceleration, add:
tropical-gemm-cuda = "0.1"
```

## Quick Start

### CPU (Matrix API)

```rust
use tropical_gemm::{Mat, MatRef, MaxPlus};

// Create matrices from raw data
let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);

// C[i,j] = max_k(A[i,k] + B[k,j])
let c = a.matmul(&b);            // Method syntax
let c = &a * &b;                  // Operator syntax
assert_eq!(c.get_value(0, 0), 8.0); // max(1+1, 2+3, 3+5) = 8

// Or use owned matrices with factory methods
let a = Mat::<MaxPlus<f32>>::from_row_major(&a_data, 2, 3);
let b = Mat::<MaxPlus<f32>>::from_row_major(&b_data, 3, 2);
let c = a.matmul(&b);
```

### GPU (CUDA)

```rust
use tropical_gemm::{MatRef, MaxPlus};
use tropical_gemm_cuda::{CudaContext, GpuMat};

let ctx = CudaContext::new()?;

// Upload to GPU
let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);

let a_gpu = GpuMat::from_matref(&ctx, &a)?;
let b_gpu = GpuMat::from_matref(&ctx, &b)?;

// Compute on GPU
let c_gpu = a_gpu.matmul(&ctx, &b_gpu)?;

// Download result
let c = c_gpu.to_mat(&ctx)?;
```

### Batched Operations

```rust
use tropical_gemm::{Mat, MaxPlus};

// Create batch of matrices
let a_batch = vec![
    Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2),
    Mat::<MaxPlus<f32>>::from_row_major(&[5.0, 6.0, 7.0, 8.0], 2, 2),
];
let b_batch = vec![
    Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2),
    Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2),
];

// Batched matmul (parallel by default)
let c_batch = Mat::matmul_batched(&a_batch, &b_batch);
```

### Argmax Tracking (for Backpropagation)

```rust
use tropical_gemm::{Mat, MaxPlus};

let a = Mat::<MaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
let b = Mat::<MaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

let result = a.matmul_argmax(&b);

// Get the optimal value and which k produced it
let value = result.get_value(0, 0); // 8.0
let k_idx = result.get_argmax(0, 0); // 2 (k=2 gave max)

// Compute gradients for backpropagation
let grad_c = vec![1.0f64; 4]; // upstream gradient
let grad_a = result.backward_a(&grad_c);
let grad_b = result.backward_b(&grad_c);
```

GPU argmax is also available:

```rust
use tropical_gemm::{MatRef, MaxPlus};
use tropical_gemm_cuda::{CudaContext, GpuMat};

let ctx = CudaContext::new()?;
let a_gpu = GpuMat::from_matref(&ctx, &a)?;
let b_gpu = GpuMat::from_matref(&ctx, &b)?;

let result = a_gpu.matmul_argmax(&ctx, &b_gpu)?;
let result_cpu = result.to_mat_with_argmax(&ctx)?;
// result_cpu.get_argmax(i, j) = k such that C[i,j] = A[i,k] + B[k,j]

// GPU backward pass
let grad_a = result.backward_a(&ctx, &grad_c_gpu)?;
let grad_b = result.backward_b(&ctx, &grad_c_gpu)?;
```

### Python Bindings (PyTorch Integration)

```python
import torch
import numpy as np
import tropical_gemm

# Create matrices
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

# MaxPlus matmul with argmax for backprop
c, argmax = tropical_gemm.maxplus_matmul_with_argmax(a, b)

# Compute gradients
grad_c = np.ones((2, 2), dtype=np.float32)
grad_a = tropical_gemm.backward_a(grad_c, argmax, k=3)
grad_b = tropical_gemm.backward_b(grad_c, argmax, k=3)
```

See `crates/tropical-gemm-python/examples/pytorch_tropical.py` for full PyTorch custom autograd integration.

## Benchmark Results

Tested on NVIDIA RTX A4500 (Ampere, sm_86) with AMD Ryzen 9 5900X.

### GPU vs CPU Performance (MaxPlus f32)

| Size | CPU SIMD (ms) | GPU Kernel (ms) | Speedup |
|------|---------------|-----------------|---------|
| 256  | 4.1           | 0.032           | **128x** |
| 512  | 32.8          | 0.086           | **381x** |
| 1024 | 262.3         | 0.358           | **733x** |
| 2048 | 2091.6        | 2.510           | **833x** |

### Rust CUDA vs C Reference

Comparison with [TropicalGemm_Cuda](https://github.com/ArrogantGao/TropicalGemm_Cuda):

| Size | C Library (ms) | Rust CUDA (ms) | Ratio |
|------|----------------|----------------|-------|
| 256  | 0.028          | 0.032          | 1.14x |
| 512  | 0.074          | 0.086          | 1.16x |
| 1024 | 0.315          | 0.358          | 1.14x |
| 2048 | 2.224          | 2.509          | 1.13x |

The C library is ~13-16% faster due to pre-compiled PTX vs runtime compilation.

### GPU Backward Pass Performance

| Size | Forward (ms) | Backward A (ms) | Backward B (ms) |
|------|--------------|-----------------|-----------------|
| 256  | 0.032        | 0.018           | 0.018           |
| 512  | 0.086        | 0.052           | 0.052           |
| 1024 | 0.358        | 0.183           | 0.184           |
| 2048 | 2.510        | 1.312           | 1.315           |

## Crate Structure

```
tropical-gemm/
├── tropical-gemm         # Main crate: types, CPU backend, public API
│   ├── mat/              # Matrix types (Mat, MatRef, MatMut)
│   ├── types/            # Semiring type definitions
│   ├── core/             # BLIS-style GEMM algorithms
│   └── simd/             # SIMD kernels (AVX-512, AVX2, NEON)
│
├── tropical-gemm-cuda    # GPU backend (optional)
│   └── kernels/          # CUDA kernel source
│
└── tropical-gemm-python  # Python bindings (PyO3)
    └── examples/         # PyTorch integration examples
```

## Running Benchmarks

```bash
# CPU benchmark
cargo run --release --example bench_rust -p tropical-gemm

# CUDA vs CPU benchmark
cargo run --release --example bench_cuda_vs_cpu -p tropical-gemm-cuda

# GPU backward pass benchmark
cargo run --release --example bench_backward -p tropical-gemm-cuda
```

## License

MIT
