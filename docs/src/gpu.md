# GPU Acceleration

tropical-gemm-cuda provides NVIDIA GPU acceleration via CUDA.

## Requirements

- NVIDIA GPU (compute capability 3.5+)
- CUDA Toolkit 11.0 or later
- `nvcc` in PATH

## Basic Usage

```rust
use tropical_gemm::{MatRef, MaxPlus};
use tropical_gemm_cuda::{CudaContext, GpuMat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create CUDA context (compiles kernels on first use)
    let ctx = CudaContext::new()?;

    // Prepare CPU data
    let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
    let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);

    // Upload to GPU
    let a_gpu = GpuMat::from_matref(&ctx, &a)?;
    let b_gpu = GpuMat::from_matref(&ctx, &b)?;

    // Compute on GPU
    let c_gpu = a_gpu.matmul(&ctx, &b_gpu)?;

    // Download result
    let c = c_gpu.to_mat(&ctx)?;

    println!("C[0,0] = {}", c.get_value(0, 0));
    Ok(())
}
```

## Context Reuse

The `CudaContext` compiles CUDA kernels on first use. **Always reuse contexts**
to avoid repeated compilation:

```rust
// GOOD: Reuse context
let ctx = CudaContext::new()?;
for _ in 0..100 {
    let c = a_gpu.matmul(&ctx, &b_gpu)?;
}

// BAD: Creates new context each iteration
for _ in 0..100 {
    let ctx = CudaContext::new()?;  // Slow!
    let c = a_gpu.matmul(&ctx, &b_gpu)?;
}
```

## GPU Argmax

For backpropagation with GPU computation:

```rust
let ctx = CudaContext::new()?;
let a_gpu = GpuMat::from_matref(&ctx, &a)?;
let b_gpu = GpuMat::from_matref(&ctx, &b)?;

// Forward pass with argmax tracking
let result = a_gpu.matmul_argmax(&ctx, &b_gpu)?;

// Download values and argmax
let result_cpu = result.to_mat_with_argmax(&ctx)?;
let value = result_cpu.get_value(0, 0);
let k_idx = result_cpu.get_argmax(0, 0);

// Backward pass on GPU
let grad_c_gpu = GpuMat::from_matref(&ctx, &grad_c)?;
let grad_a_gpu = result.backward_a(&ctx, &grad_c_gpu)?;
let grad_b_gpu = result.backward_b(&ctx, &grad_c_gpu)?;
```

## Batched GPU Operations

Process multiple matrices efficiently:

```rust
use tropical_gemm::{Mat, MaxPlus};
use tropical_gemm_cuda::{CudaContext, GpuMat};

let ctx = CudaContext::new()?;

// Upload batch to GPU
let a_batch: Vec<Mat<MaxPlus<f32>>> = /* ... */;
let b_batch: Vec<Mat<MaxPlus<f32>>> = /* ... */;

let a_gpu_batch = GpuMat::from_mats(&ctx, &a_batch)?;
let b_gpu_batch = GpuMat::from_mats(&ctx, &b_batch)?;

// Batched multiply
let c_gpu_batch = GpuMat::matmul_batched(&ctx, &a_gpu_batch, &b_gpu_batch)?;

// Download results
let c_batch = GpuMat::to_mats(&ctx, &c_gpu_batch)?;
```

## One-Shot API

For simple cases without context reuse:

```rust
use tropical_gemm::TropicalMaxPlus;
use tropical_gemm_cuda::tropical_matmul_gpu;

let a = vec![1.0f32; 64 * 64];
let b = vec![1.0f32; 64 * 64];

// One-shot GPU multiplication (creates temporary context)
let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, 64, 64, &b, 64)?;
```

## Performance Comparison

| Size | CPU SIMD | GPU | Speedup |
|------|----------|-----|---------|
| 256 | 4.1 ms | 0.032 ms | 128x |
| 512 | 32.8 ms | 0.086 ms | 381x |
| 1024 | 262.3 ms | 0.358 ms | 733x |
| 2048 | 2091.6 ms | 2.510 ms | 833x |

GPU becomes advantageous for matrices larger than ~256×256.
