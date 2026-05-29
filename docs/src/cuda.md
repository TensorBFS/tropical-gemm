# CUDA Implementation

The GPU backend uses CUDA with runtime kernel compilation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User API                               │
│           (GpuMat::matmul, tropical_matmul_gpu)             │
├─────────────────────────────────────────────────────────────┤
│                    CudaContext                              │
│         (kernel compilation, device management)             │
├─────────────────────────────────────────────────────────────┤
│                      NVRTC                                  │
│           (runtime kernel compilation)                      │
├─────────────────────────────────────────────────────────────┤
│                   CUDA Kernels                              │
│       (tropical_gemm.cu, specialized per semiring)          │
└─────────────────────────────────────────────────────────────┘
```

## Runtime Compilation and the CUBIN Cache

Kernels are compiled from CUDA C source at runtime with NVRTC — straight to a **CUBIN**
(native SASS) for the device's compute capability — and the cubin is **cached on disk**.
Later processes load the cached cubin directly, skipping both the NVRTC compile and the
driver's PTX→SASS JIT.

```rust
// First CudaContext::new() on a machine: full NVRTC compile (~10s), cubin cached.
// Every later process: loads the cached cubin (~0.13s).
let ctx = CudaContext::new()?;

let c = a_gpu.matmul(&ctx, &b_gpu)?;  // just a kernel launch
```

The cache lives at `$XDG_CACHE_HOME` (or `~/.cache`)
`/tropical-gemm/<hash>_sm_<cc>_nvrtc<ver>.cubin`, keyed on the kernel source, compile flags,
GPU arch, and NVRTC version — so a different GPU or CUDA toolkit never reuses the wrong
cubin. A stale / corrupt / arch-incompatible file self-heals (it is deleted and recompiled).

Benefits:
- **No build-time CUDA dependency**: cudarc dynamic-loads CUDA at runtime, so users don't
  need `nvcc` (or any CUDA toolkit) at build time.
- **Fast startup**: the on-disk cubin cache makes a warm `CudaContext::new()` ~0.13 s.
- **Specialization**: kernels are compiled for the exact device architecture.

> **Selecting the toolkit:** at build time set `CUDARC_CUDA_VERSION` (e.g. `12080` for CUDA
> 12.8) to match the toolkit available at runtime. NVRTC compiles device code optimized by
> default (`-dopt=on` is implicit), so no `-O3` is needed — the `-O3` in `nvcc -O3` is a
> host-compiler flag and does not apply to these pure-device kernels.

## Kernel Design

### Thread Block Organization

```
Block size: 16×16 threads (256 threads per block)
Grid: ceil(M/16) × ceil(N/16) blocks

Each thread computes one output element C[i,j]
```

### Memory Access Pattern

```cuda
__global__ void tropical_maxplus_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float max_val = -INFINITY;
        for (int k = 0; k < K; k++) {
            float sum = A[row * K + k] + B[k * N + col];
            max_val = fmaxf(max_val, sum);
        }
        C[row * N + col] = max_val;
    }
}
```

### Shared Memory Tiling

For larger matrices, shared memory is used:

```cuda
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// Load tiles cooperatively
As[ty][tx] = A[row * K + (tile * TILE_SIZE + tx)];
Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
__syncthreads();

// Compute partial result from tile
for (int k = 0; k < TILE_SIZE; k++) {
    max_val = fmaxf(max_val, As[ty][k] + Bs[k][tx]);
}
```

## Argmax Kernels

For backpropagation, kernels track which k index achieved the max:

```cuda
__global__ void tropical_maxplus_gemm_argmax(
    const float* A, const float* B,
    float* C, int* argmax,
    int M, int N, int K
) {
    // ... setup ...

    float max_val = -INFINITY;
    int max_k = 0;

    for (int k = 0; k < K; k++) {
        float sum = A[row * K + k] + B[k * N + col];
        if (sum > max_val) {
            max_val = sum;
            max_k = k;
        }
    }

    C[row * N + col] = max_val;
    argmax[row * N + col] = max_k;
}
```

## Batched Kernels

For processing multiple matrices:

```cuda
// Strided batched: matrices stored contiguously
__global__ void tropical_maxplus_gemm_batched(
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_count,
    int stride_a, int stride_b, int stride_c
) {
    int batch = blockIdx.z;
    // ... standard GEMM with offset by batch * stride ...
}
```

## Memory Management

### Device Memory Allocation

```rust
// Allocate GPU memory
let d_ptr = cuda_malloc(size_bytes)?;

// Copy host → device
cuda_memcpy_h2d(d_ptr, h_data, size_bytes)?;

// Copy device → host
cuda_memcpy_d2h(h_data, d_ptr, size_bytes)?;

// Free
cuda_free(d_ptr)?;
```

### Pinned Memory (for faster transfers)

```rust
// For frequent CPU↔GPU transfers, use pinned memory
let pinned = cuda_malloc_host(size_bytes)?;
// ... 2-3x faster transfers ...
cuda_free_host(pinned)?;
```

## Error Handling

CUDA errors are wrapped in Rust Result types:

```rust
match CudaContext::new() {
    Ok(ctx) => { /* use context */ }
    Err(CudaError::NoDevice) => {
        println!("No CUDA device found, using CPU");
    }
    Err(CudaError::CompilationFailed(msg)) => {
        eprintln!("Kernel compilation failed: {}", msg);
    }
    Err(e) => return Err(e.into()),
}
```

## Code Location

- `tropical-gemm-cuda/src/context.rs`: CUDA context and compilation
- `tropical-gemm-cuda/src/gpu_mat.rs`: GPU matrix type
- `tropical-gemm-cuda/src/kernels.rs`: Kernel management
- `tropical-gemm-cuda/kernels/tropical_gemm.cu`: CUDA kernel source
