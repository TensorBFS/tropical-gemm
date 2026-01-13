# API Reference

This page provides quick reference to the main APIs.

For complete documentation, see the [Rust API docs](../api/tropical_gemm/index.html).

## Crate Overview

| Crate | Purpose |
|-------|---------|
| `tropical-gemm` | CPU implementation with SIMD |
| `tropical-gemm-cuda` | GPU implementation with CUDA |
| `tropical-gemm-python` | Python bindings |

## Semiring Types

```rust
use tropical_gemm::{MaxPlus, MinPlus, MaxMul};
use tropical_gemm::types::{TropicalMaxPlus, TropicalMinPlus, TropicalMaxMul};

// Wrapper types (for storage)
let a: MaxPlus<f32> = MaxPlus::new(3.0);
let b: MinPlus<f64> = MinPlus::new(5.0);

// Marker types (for generic functions)
type S = TropicalMaxPlus<f32>;
```

## Matrix Types

### Mat (Owned)

```rust
use tropical_gemm::{Mat, MaxPlus};

// Create from function
let a = Mat::<MaxPlus<f32>>::from_fn(m, k, |i, j| value);

// Create from scalar slice
let a = Mat::<MaxPlus<f32>>::from_scalar_slice(&data, m, k);

// Access
let val = a.get_value(i, j);  // Returns f32
let dim = a.dim();            // Returns (rows, cols)
```

### MatRef (Borrowed)

```rust
use tropical_gemm::{MatRef, MaxPlus};

// From slice
let a = MatRef::<MaxPlus<f32>>::from_slice(&data, m, k);

// From Mat
let a_ref = a.as_ref();
```

### MatMut (Mutable)

```rust
use tropical_gemm::MatMut;

let mut c = Mat::zeros(m, n);
let c_mut = c.as_mut();
```

## Matrix Operations

### High-Level API (Mat)

```rust
use tropical_gemm::{Mat, MaxPlus};

let a = Mat::<MaxPlus<f32>>::from_scalar_slice(&a_data, m, k);
let b = Mat::<MaxPlus<f32>>::from_scalar_slice(&b_data, k, n);

// Standard multiply
let c = a.matmul(&b);

// With argmax tracking
let result = a.matmul_with_argmax(&b);
let value = result.get_value(i, j);
let argmax = result.get_argmax(i, j);
```

### Low-Level API (Functions)

```rust
use tropical_gemm::{tropical_matmul, tropical_matmul_with_argmax, TropicalMaxPlus};

// Standard multiply
let c = tropical_matmul::<TropicalMaxPlus<f32>>(&a, m, k, &b, n);

// With argmax
let (values, argmax) = tropical_matmul_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n);
```

## GPU API

### CudaContext

```rust
use tropical_gemm_cuda::CudaContext;

let ctx = CudaContext::new()?;  // Compiles kernels
```

### GpuMat

```rust
use tropical_gemm_cuda::GpuMat;
use tropical_gemm::{MatRef, MaxPlus};

// Upload
let a_gpu = GpuMat::from_matref(&ctx, &a)?;

// Compute
let c_gpu = a_gpu.matmul(&ctx, &b_gpu)?;

// With argmax
let result = a_gpu.matmul_argmax(&ctx, &b_gpu)?;

// Download
let c = c_gpu.to_mat(&ctx)?;
```

### Batched Operations

```rust
use tropical_gemm_cuda::GpuMat;

// Upload batch
let a_batch = GpuMat::from_mats(&ctx, &a_mats)?;
let b_batch = GpuMat::from_mats(&ctx, &b_mats)?;

// Batched multiply
let c_batch = GpuMat::matmul_batched(&ctx, &a_batch, &b_batch)?;

// Download batch
let c_mats = GpuMat::to_mats(&ctx, &c_batch)?;
```

## Python API

### NumPy Functions

```python
import tropical_gemm
import numpy as np

a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([[5, 6], [7, 8]], dtype=np.float32)

# Basic operations
c = tropical_gemm.maxplus_matmul(a, b)
c = tropical_gemm.minplus_matmul(a, b)
c = tropical_gemm.maxmul_matmul(a, b)

# With argmax
values, argmax = tropical_gemm.maxplus_matmul_with_argmax(a, b)
```

### Backward Pass

```python
# Gradient computation
grad_a = tropical_gemm.backward_a(grad_c, argmax, k)
grad_b = tropical_gemm.backward_b(grad_c, argmax, k)
```

## Utility Functions

### SIMD Detection

```rust
use tropical_gemm::{simd_level, SimdLevel};

match simd_level() {
    SimdLevel::Avx512 => { /* ... */ }
    SimdLevel::Avx2 => { /* ... */ }
    SimdLevel::Sse41 => { /* ... */ }
    SimdLevel::Neon => { /* ... */ }
    SimdLevel::None => { /* ... */ }
}
```

## Type Aliases

For convenience:

```rust
// These are equivalent:
use tropical_gemm::MaxPlus;
use tropical_gemm::types::max_plus::MaxPlus;

// Marker types for generics:
use tropical_gemm::TropicalMaxPlus;  // = TropicalSemiringImpl<MaxPlusTag, T>
use tropical_gemm::TropicalMinPlus;
use tropical_gemm::TropicalMaxMul;
```
