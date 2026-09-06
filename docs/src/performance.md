# Performance Guide

This guide helps you get the best performance from tropical-gemm.

## CPU vs GPU Selection

Choose using the actual matrix shape, thread count, need for argmax, and whether
inputs already live on the GPU. CPU/GPU crossover points vary with hardware and
transfer overhead. The [CPU SIMD/workspace and PyTorch comparison report](https://github.com/TensorBFS/tropical-gemm/blob/main/benchmarks/results/2026-09-07-cpu-completion.md)
contains current measurements and reproducible commands for both paths.

For single-matrix PyTorch calls, select `tropical_*_matmul_gpu` explicitly to
use CUDA. The ordinary `tropical_*_matmul` convenience functions use CPU
execution even when given CUDA tensors, adding host/device transfers.

### Historical Benchmark Results (MaxPlus f32)

Tested on NVIDIA RTX A4500 (Ampere) with AMD Ryzen 9 5900X. These older CPU
measurements predate single-GEMM threading; use the reproducible CPU benchmark
below when choosing a backend for your workload.

| Size | CPU AVX2 | GPU | GPU Speedup |
|------|----------|-----|-------------|
| 64 | 0.05 ms | 0.02 ms | 2.5x |
| 128 | 0.4 ms | 0.02 ms | 20x |
| 256 | 4.1 ms | 0.03 ms | 137x |
| 512 | 32.8 ms | 0.09 ms | 364x |
| 1024 | 262 ms | 0.36 ms | 728x |
| 2048 | 2092 ms | 2.5 ms | 837x |

### Rust CUDA vs C Reference

Re-measured on **NVIDIA A40 (sm_86), CUDA 12.8**, kernel-only (device-resident, no
host transfers; warmup + 100 iters), against
[TropicalGemm_Cuda](https://github.com/ArrogantGao/TropicalGemm_Cuda) built with
`nvcc -arch=sm_86`:

| Size | C ref (ms) | tropical-gemm (ms) | Ratio |
|------|-----------:|-------------------:|------:|
| 1024 | 0.272 | 0.263 | 0.97x |
| 2048 | 1.702 | 1.654 | 0.97x |
| 4096 | 13.40 | 13.01 | 0.97x |

The kernels are **~3% faster than the C reference** at large sizes (and more at small
sizes). This followed fixing the one real difference between the two kernels (issue #40):

- **Interior/boundary split in the tile loads.** Both kernels are the same tiled GEMM,
  but the old code ran a per-element bounds check (`if (row < M && col < K)`) on *every*
  element of *every* block. For a large matrix ~97% of blocks are fully interior and never
  need it. Splitting interior blocks (bare loads) from boundary blocks (guarded) removed
  that overhead and made the kernels ~7-8% faster — enough to pass the reference, which the
  old code trailed by ~3-6%. (We end up ahead because the reference's store path re-reads
  `C` and applies `alpha`/`beta`, while ours writes the accumulator directly.)
- **Compiling to an arch-targeted CUBIN (`-arch=sm_XX`) does not change throughput.**
  Default PTX (driver-JIT'd at load) and an offline CUBIN run identically — on a current
  driver the JIT runs ptxas and produces SASS equivalent to offline `nvcc`. The CUBIN build
  is a startup-latency win (issue #41), not a throughput one. (An earlier edition of this
  page blamed the gap on the CUDA toolkit version; that was wrong — it was the bounds-check.)

> NVRTC compiles device code optimized by default (`-dopt=on` is implicit unless `-G` is
> passed), so no `-O3` is needed — the `-O3` in `nvcc -O3` is a host-compiler flag and does
> not apply to these pure-device kernels.

### GPU Backward Pass Performance

| Size | Forward (ms) | Backward A (ms) | Backward B (ms) |
|------|--------------|-----------------|-----------------|
| 256 | 0.032 | 0.018 | 0.018 |
| 512 | 0.086 | 0.052 | 0.052 |
| 1024 | 0.358 | 0.183 | 0.184 |
| 2048 | 2.510 | 1.312 | 1.315 |

## CPU Optimization

### CPU threads

The default `parallel` feature parallelizes both batches and individual large
GEMMs, including argmax operations. Each task owns an output rectangle and
performs its entire K reduction, preserving serial reduction order and
first-winner argmax ties. Row or column boundaries align to microkernel tiles.

Small calls stay serial. The splitter currently requires at least 4,194,304
scalar products and budgets at least 2,097,152 per task, capped by the active
Rayon pool and the available output tiles. These thresholds are implementation
details, not an API guarantee. K alone is never split: very narrow outputs may
remain serial even with a large K.

Set `RAYON_NUM_THREADS` before the first CPU call (including Python calls), or
use a caller-owned Rayon pool in Rust:

```rust
use tropical_gemm::{tropical_matmul, TropicalMaxPlus};

let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
let a = vec![1.0f32; 512 * 512];
let b = vec![2.0f32; 512 * 512];
let c = pool.install(|| tropical_matmul::<TropicalMaxPlus<f32>>(&a, 512, 512, &b, 512));
assert_eq!(c[0].0, 3.0);
```

Nested batched calls reuse the same pool without creating another set of OS
threads. Compile the core crate with `default-features = false` for serial
execution. Custom kernels passed to the low-level GEMM functions must be `Sync`,
as they may be shared across workers. Packing buffers are bounded by each
submatrix's actual panel dimensions. Repeated calls can retain those buffers
with a caller-owned workspace, as shown below.

Measure your workload in release mode:

```bash
# M N K samples thread-counts; CSV medians include allocations and packing.
cargo run --release -p tropical-gemm --example bench_threads -- 512 512 512 7 1,2,4,8
```

The example covers values, argmax, and a batch of four matrices. Repeated
measurements on a shared server can vary with CPU contention, clock changes,
and memory placement. See the [CPU threading measurements](https://github.com/TensorBFS/tropical-gemm/blob/main/benchmarks/results/2026-09-07-cpu-threading.md)
for baseline comparisons and raw results.

### SIMD Detection

`simd_level()` reports hardware capabilities:

```rust
println!("CPU capabilities: {:?}", tropical_gemm::simd_level());
```

AVX2 and NEON implement f32/f64 argmax for MaxPlus, MinPlus, and MaxMul, plus
u32/u64 Bitwise value kernels. Argmax preserves the portable comparison rules,
including first-winner ties, signed zeros, and unordered NaN comparisons.
Unsupported architectures and other scalar types use the portable fallback.

An AVX-512-capable CPU currently uses AVX2 kernels; there is no dedicated AVX-512
kernel. Value-only floating-point SIMD coverage remains operation-dependent:
MaxPlus f32/f64 and MinPlus f32 have AVX2/NEON kernels, while MaxMul f32 has AVX2.
Other value-only combinations use portable kernels. Threading applies to both
SIMD and portable execution.

### Reusable packing workspace

`GemmWorkspace<Scalar>` retains packing buffers across calls. An exclusive mutable
borrow prevents concurrent reuse; parallel GEMM tasks borrow disjoint buffer
pairs internally. It can be reused across shapes, thread pools, and semirings
with the same scalar type. Storage grows to the largest panels/task count seen;
`capacity_bytes()` reports retained scalar storage and `clear()` releases it.

```rust
use tropical_gemm::{GemmWorkspace, TropicalGemm, TropicalMaxPlus};

let mut workspace = GemmWorkspace::<f32>::new();
let mut c = vec![TropicalMaxPlus(0.0); 4];
for _ in 0..10 {
    TropicalGemm::new(2, 2, 2).execute_with_workspace(
        &[1.0; 4], 2, &[2.0; 4], 2, &mut c, 2, &mut workspace,
    );
}
assert!(c.iter().all(|v| v.0 == 3.0));
```

For argmax, use `tropical_matmul_with_argmax_with_workspace`; it retains packing
storage but still allocates the returned values and indices. Low-level custom
kernel APIs also have workspace variants. Custom `KernelDispatch` implementations
can override `dispatch_gemm_with_workspace`; the default retains their existing
dispatch and may allocate.

A warmed serial builder call reuses both its output and packing storage without
allocating. Parallel calls can still allocate task metadata; this API does not
cache packed matrix contents, so inputs may change freely between calls.

```bash
# N threads samples: all float semirings and Bitwise, values and argmax.
cargo run --release -p tropical-gemm --example bench_cpu -- 512 8 7
# Compare fresh packing allocations with caller-owned workspace reuse.
cargo run --release -p tropical-gemm --example bench_workspace -- 128 1 21
# End-to-end forward interface versus PyTorch broadcast/reduce.
python benchmarks/bench_pytorch.py --device cpu --sizes 64 256 512 --threads 8
python benchmarks/bench_pytorch.py --device cuda --sizes 64 256 512 1024
```

### Memory Layout

The slice-based API accepts contiguous row-major scalars. `Mat` and `MatRef` use
column-major storage. Supply the layout expected by the API to avoid conversion:

```rust
use tropical_gemm::{Mat, MaxPlus};

// Column-major storage for the matrix [[1, 2], [3, 4]].
let a = Mat::<MaxPlus<f32>>::from_col_major(&[1.0, 3.0, 2.0, 4.0], 2, 2);
let c = &a * &a;
assert_eq!(c.get_value(0, 0), 5.0);
```

### Cache Efficiency

For best cache utilization:

- **Square matrices**: Optimal blocking
- **Tall-skinny (M >> K)**: Good cache reuse for A
- **Short-wide (K >> M)**: May have cache pressure

## GPU Optimization

### Context Reuse and the kernel cache

`CudaContext::new()` compiles all kernels to a CUBIN for the device's architecture and
**caches it on disk** (under `$XDG_CACHE_HOME` / `~/.cache/tropical-gemm/`), so the compile
cost is paid once per machine, not once per process:

- **cold** (first run, empty cache): ~10 s — full NVRTC compile.
- **warm** (cubin cached): **~0.13 s** — loads the cubin directly, skipping both the NVRTC
  compile *and* the driver's PTX→SASS JIT.

Still reuse a single `CudaContext` within a process to avoid even the warm ~0.13 s and keep
kernels resident:

```rust
// GOOD: create once, reuse many times
let ctx = CudaContext::new()?;  // ~10s cold / ~0.13s warm (disk-cached cubin)
for batch in batches {
    let c = a.matmul(&ctx, &b)?;  // fast
}

// BAD: new context each iteration
for batch in batches {
    let ctx = CudaContext::new()?;
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
