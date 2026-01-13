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

## Runtime Compilation

Kernels are compiled from CUDA C source at runtime using NVRTC:

```rust
// On first CudaContext::new()
let ctx = CudaContext::new()?;  // Compiles kernels (~1-2 seconds)

// Subsequent operations are fast
let c = a_gpu.matmul(&ctx, &b_gpu)?;  // Just kernel launch
```

Benefits:
- **No build-time CUDA dependency**: Users don't need nvcc at build time
- **Portability**: Works across CUDA versions
- **Specialization**: Kernels optimized for specific semirings

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
