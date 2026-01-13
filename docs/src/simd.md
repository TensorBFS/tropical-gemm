# SIMD Kernels

The microkernel is vectorized using SIMD instructions for maximum throughput.

## Supported Architectures

| Architecture | Instruction Set | Vector Width | f32 MR×NR | f64 MR×NR |
|--------------|-----------------|--------------|-----------|-----------|
| x86_64 | AVX-512 | 512-bit | 16×16 | 8×8 |
| x86_64 | AVX2 | 256-bit | 8×8 | 4×4 |
| x86_64 | SSE4.1 | 128-bit | 4×4 | 2×2 |
| aarch64 | NEON | 128-bit | 4×4 | 2×2 |
| Any | Portable | Scalar | 4×4 | 4×4 |

## Runtime Detection

CPU features are detected at runtime:

```rust
use tropical_gemm::{simd_level, SimdLevel};

match simd_level() {
    SimdLevel::Avx512 => println!("Using AVX-512"),
    SimdLevel::Avx2   => println!("Using AVX2"),
    SimdLevel::Sse41  => println!("Using SSE4.1"),
    SimdLevel::Neon   => println!("Using NEON"),
    SimdLevel::None   => println!("Using portable"),
}
```

## Microkernel Design

For MaxPlus f32 with AVX2 (8-wide vectors):

```rust
// Pseudocode for 8×8 microkernel
for k in 0..KC {
    // Load 8 elements from packed A
    let a_vec = _mm256_loadu_ps(a_ptr);

    // For each column in the 8-column output tile
    for j in 0..8 {
        // Broadcast scalar from packed B
        let b_scalar = _mm256_broadcast_ss(b_ptr + j);

        // Tropical multiply: a + b (element-wise)
        let prod = _mm256_add_ps(a_vec, b_scalar);

        // Tropical accumulate: max(c, prod)
        c_vec[j] = _mm256_max_ps(c_vec[j], prod);
    }

    a_ptr += 8;  // Next column in packed A
    b_ptr += 8;  // Next row in packed B
}
```

## Semiring-Specific Operations

| Semiring | Tropical Mul | Tropical Add |
|----------|--------------|--------------|
| MaxPlus | `_mm256_add_ps` | `_mm256_max_ps` |
| MinPlus | `_mm256_add_ps` | `_mm256_min_ps` |
| MaxMul | `_mm256_mul_ps` | `_mm256_max_ps` |

## Dispatch Mechanism

The `KernelDispatch` trait routes to the appropriate implementation:

```rust
impl KernelDispatch for TropicalMaxPlus<f32> {
    unsafe fn dispatch_gemm(...) {
        match simd_level() {
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                tropical_gemm_inner::<Self, Avx2MaxPlusF32>(...);
            }
            _ => {
                tropical_gemm_inner::<Self, PortableMicrokernel>(...);
            }
        }
    }
}
```

## Code Location

- `simd/detect.rs`: CPU feature detection
- `simd/dispatch.rs`: Runtime dispatch trait
- `simd/kernels/avx2.rs`: AVX2 implementations
- `simd/kernels/neon.rs`: NEON implementations
- `simd/kernels/portable.rs`: Fallback implementation
