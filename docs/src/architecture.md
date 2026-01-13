# Architecture

This section describes the internal architecture of tropical-gemm.

## Overview

tropical-gemm achieves high performance through:

1. **BLIS-style blocking** for cache efficiency
2. **SIMD microkernels** for vectorization
3. **Runtime dispatch** for optimal kernel selection
4. **CUDA kernels** for GPU acceleration

## Crate Structure

```
tropical-gemm/
├── src/
│   ├── lib.rs          # Public API
│   ├── api.rs          # Function-based API
│   ├── types/          # Semiring definitions
│   │   ├── traits.rs   # TropicalSemiring trait
│   │   ├── max_plus.rs
│   │   ├── min_plus.rs
│   │   └── max_mul.rs
│   ├── core/           # BLIS algorithm
│   │   ├── gemm.rs     # 5-loop blocking
│   │   ├── kernel.rs   # Microkernel trait
│   │   ├── packing.rs  # Matrix packing
│   │   └── tiling.rs   # Cache parameters
│   ├── simd/           # SIMD kernels
│   │   ├── dispatch.rs # Runtime selection
│   │   ├── detect.rs   # CPU detection
│   │   └── kernels/    # Per-architecture
│   └── mat/            # Matrix types

tropical-gemm-cuda/
├── src/
│   ├── lib.rs          # Public API
│   ├── context.rs      # CUDA context
│   ├── kernels.rs      # Kernel management
│   └── gpu_mat.rs      # GPU matrix type
└── kernels/
    └── tropical_gemm.cu  # CUDA source
```

## Performance Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    User API (Mat, MatRef)                   │
├─────────────────────────────────────────────────────────────┤
│                  Function API (tropical_matmul)             │
├─────────────────────────────────────────────────────────────┤
│               SIMD Dispatch (KernelDispatch)                │
├─────────────────────────────────────────────────────────────┤
│           BLIS 5-Loop Blocking (tropical_gemm_inner)        │
├─────────────────────────────────────────────────────────────┤
│                    SIMD Microkernel                         │
│            (AVX2 / AVX-512 / NEON / Portable)               │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Semiring as Type Parameter

Operations are generic over the semiring type, enabling compile-time specialization:

```rust
pub fn tropical_matmul<S: TropicalSemiring>(
    a: &[S::Scalar], m: usize, k: usize,
    b: &[S::Scalar], n: usize
) -> Vec<S>
```

### 2. Scalar vs Semiring Types

- **Input**: Raw scalar data (`&[f32]`, `&[f64]`)
- **Output**: Semiring-wrapped values (`Vec<MaxPlus<f32>>`)

This avoids unnecessary wrapping in hot paths.

### 3. Runtime SIMD Dispatch

CPU features are detected at runtime, not compile time:

```rust
match simd_level() {
    SimdLevel::Avx512 => avx512_kernel(...),
    SimdLevel::Avx2   => avx2_kernel(...),
    _                 => portable_kernel(...),
}
```

### 4. CUDA Runtime Compilation

Kernels are compiled from CUDA C source at runtime via NVRTC:

- No compile-time CUDA dependency
- Portability across CUDA versions
- Template-like specialization via macros
