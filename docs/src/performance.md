# Performance Guide

This guide helps you get the best performance from tropical-gemm.

## CPU vs GPU Selection

| Matrix Size | Recommendation | Reason |
|-------------|----------------|--------|
| < 128×128 | CPU | GPU transfer overhead dominates |
| 128-256 | CPU or GPU | Similar performance |
| > 256×256 | GPU | GPU computation advantage |
| > 1024×1024 | GPU (strongly) | 100-800x speedup |

### Benchmark Results (MaxPlus f32)

Tested on NVIDIA RTX A4500 (Ampere) with AMD Ryzen 9 5900X.

| Size | CPU AVX2 | GPU | GPU Speedup |
|------|----------|-----|-------------|
| 64 | 0.05 ms | 0.02 ms | 2.5x |
| 128 | 0.4 ms | 0.02 ms | 20x |
| 256 | 4.1 ms | 0.03 ms | 137x |
| 512 | 32.8 ms | 0.09 ms | 364x |
| 1024 | 262 ms | 0.36 ms | 728x |
| 2048 | 2092 ms | 2.5 ms | 837x |

### Rust CUDA vs C Reference

Comparison with [TropicalGemm_Cuda](https://github.com/ArrogantGao/TropicalGemm_Cuda):

| Size | C Library (ms) | Rust CUDA (ms) | Ratio |
|------|----------------|----------------|-------|
| 256 | 0.028 | 0.032 | 1.14x |
| 512 | 0.074 | 0.086 | 1.16x |
| 1024 | 0.315 | 0.358 | 1.14x |
| 2048 | 2.224 | 2.509 | 1.13x |

The C library is ~13-16% faster due to pre-compiled PTX vs runtime compilation.

### GPU Backward Pass Performance

| Size | Forward (ms) | Backward A (ms) | Backward B (ms) |
|------|--------------|-----------------|-----------------|
| 256 | 0.032 | 0.018 | 0.018 |
| 512 | 0.086 | 0.052 | 0.052 |
| 1024 | 0.358 | 0.183 | 0.184 |
| 2048 | 2.510 | 1.312 | 1.315 |

## CPU Optimization

### SIMD Detection

Ensure optimal SIMD is being used:

```rust
use tropical_gemm::{simd_level, SimdLevel};

match simd_level() {
    SimdLevel::Avx512 => println!("Best: AVX-512"),
    SimdLevel::Avx2 => println!("Good: AVX2"),
    SimdLevel::Sse41 => println!("Okay: SSE4.1"),
    SimdLevel::Neon => println!("ARM: NEON"),
    SimdLevel::None => println!("Slow: Portable fallback"),
}
```

### Memory Layout

Row-major contiguous data is fastest:

```rust
// GOOD: Contiguous row-major
let a = Mat::<MaxPlus<f32>>::from_fn(m, k, |i, j| data[i * k + j]);

// SLOWER: Non-contiguous requires packing overhead
let a_ref = MatRef::from_slice_strided(&data, m, k, stride);
```

### Cache Efficiency

For best cache utilization:

- **Square matrices**: Optimal blocking
- **Tall-skinny (M >> K)**: Good cache reuse for A
- **Short-wide (K >> M)**: May have cache pressure

## GPU Optimization

### Context Reuse

**Critical**: Reuse `CudaContext` to avoid repeated kernel compilation:

```rust
// GOOD: Create once, reuse many times
let ctx = CudaContext::new()?;  // ~1-2 seconds
for batch in batches {
    let c = a.matmul(&ctx, &b)?;  // Fast
}

// BAD: Creates new context each time
for batch in batches {
    let ctx = CudaContext::new()?;  // Slow!
    let c = a.matmul(&ctx, &b)?;
}
```

### Batched Operations

For multiple matrix multiplications, use batched API:

```rust
// GOOD: Single kernel launch for all matrices
let c_batch = GpuMat::matmul_batched(&ctx, &a_batch, &b_batch)?;

// SLOWER: Sequential kernel launches
let c_batch: Vec<_> = a_batch.iter()
    .zip(&b_batch)
    .map(|(a, b)| a.matmul(&ctx, b))
    .collect();
```

### Memory Transfer

Minimize CPU↔GPU transfers:

```rust
// GOOD: Keep data on GPU between operations
let a_gpu = GpuMat::from_matref(&ctx, &a)?;
let b_gpu = GpuMat::from_matref(&ctx, &b)?;

// Multiple operations without transfer
let c_gpu = a_gpu.matmul(&ctx, &b_gpu)?;
let d_gpu = c_gpu.matmul(&ctx, &b_gpu)?;
let e_gpu = d_gpu.matmul(&ctx, &b_gpu)?;

// Only transfer final result
let e = e_gpu.to_mat(&ctx)?;

// BAD: Transfer for each operation
for i in 0..3 {
    let a_gpu = GpuMat::from_matref(&ctx, &a)?;  // Upload
    let c_gpu = a_gpu.matmul(&ctx, &b_gpu)?;
    let c = c_gpu.to_mat(&ctx)?;  // Download
    a = c;  // Use result for next iteration
}
```

## PyTorch Training

### Keep Context Alive

```python
# Create context once at module initialization
class TropicalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Context created once
        self.ctx = tropical_gemm.CudaContext()

    def forward(self, a, b):
        # Reuse context
        return tropical_matmul_gpu(self.ctx, a, b)
```

### Batch Your Data

```python
# GOOD: Large batch, single kernel
output = tropical_matmul(large_batch_a, large_batch_b)

# SLOWER: Many small operations
outputs = [tropical_matmul(a, b) for a, b in zip(small_as, small_bs)]
```

## Python Threading

### GIL Release During Compute

All CPU functions release Python's GIL during heavy computation, allowing other Python threads to run concurrently:

```python
import threading
import tropical_gemm
import numpy as np

def background_task():
    # This can run while tropical_gemm computes
    print("Background task running")

a = np.random.randn(1000, 1000).astype(np.float32)
b = np.random.randn(1000, 1000).astype(np.float32)

# Start background thread
t = threading.Thread(target=background_task)
t.start()

# GIL is released during compute - background thread can run
c = tropical_gemm.maxplus_matmul(a, b)

t.join()
```

This is particularly useful in:
- Web servers (Flask, FastAPI) handling concurrent requests
- GUI applications that need to remain responsive
- Async applications using concurrent.futures

### Zero-Copy with 2D Functions

The `*_matmul_2d` functions return properly shaped 2D arrays without reshaping overhead:

```python
# Recommended: Use 2D functions for cleaner code
c = tropical_gemm.maxplus_matmul_2d(a, b)  # shape: (m, n)

# Older pattern requiring reshape
c_flat = tropical_gemm.maxplus_matmul(a, b)  # shape: (m*n,)
c = c_flat.reshape(m, n)
```

## Memory Considerations

### Argmax Memory

With argmax tracking, memory usage increases:

| Operation | Memory per element |
|-----------|-------------------|
| Standard GEMM | 4 bytes (f32) |
| With argmax | 8 bytes (f32 + i32) |

For large matrices, this can be significant:
- 4096×4096 standard: 64 MB
- 4096×4096 with argmax: 128 MB

### GPU Memory

Check available GPU memory:

```rust
let (free, total) = cuda_mem_info()?;
println!("GPU memory: {} MB free / {} MB total",
    free / 1024 / 1024,
    total / 1024 / 1024);
```

## Profiling

### CPU Profiling

```bash
# Linux perf
perf record --call-graph dwarf ./target/release/benchmark
perf report

# Flamegraph
cargo install flamegraph
cargo flamegraph --bin benchmark
```

### GPU Profiling

```bash
# NVIDIA Nsight
nsys profile ./target/release/gpu_benchmark
nsys-ui report.nsys-rep

# nvprof (older)
nvprof ./target/release/gpu_benchmark
```

## Troubleshooting Performance

### Unexpectedly Slow CPU

1. Check SIMD level (should be AVX2 or better on modern x86)
2. Ensure data is contiguous (avoid strided access)
3. Check for memory pressure (matrix too large for cache)

### Unexpectedly Slow GPU

1. Verify context reuse (compilation is slow)
2. Check transfer overhead (small matrices dominated by transfer)
3. Ensure sufficient GPU memory (avoid swapping)
4. Use batched API for multiple matrices

## Running Benchmarks

```bash
# CPU benchmark
cargo run --release --example bench_rust -p tropical-gemm

# CUDA vs CPU benchmark
cargo run --release --example bench_cuda_vs_cpu -p tropical-gemm-cuda

# GPU backward pass benchmark
cargo run --release --example bench_backward -p tropical-gemm-cuda
```

Or use the Makefile:

```bash
make bench          # Run all benchmarks
make bench-cpu      # CPU only
make bench-cuda     # CUDA only
```
