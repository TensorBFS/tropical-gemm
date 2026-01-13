# tropical-gemm

High-performance tropical matrix multiplication in Rust with SIMD and CUDA backends.

## What is Tropical Algebra?

**Tropical algebra** (also called max-plus or min-plus algebra) replaces standard arithmetic
operations with alternative ones:

| Standard | Tropical (MaxPlus) | Tropical (MinPlus) |
|----------|-------------------|-------------------|
| a + b | max(a, b) | min(a, b) |
| a × b | a + b | a + b |
| 0 | -∞ | +∞ |
| 1 | 0 | 0 |

## Applications

Tropical matrix multiplication appears in many algorithms:

- **Shortest/Longest Path**: Computing all-pairs shortest paths via matrix powers
- **Viterbi Algorithm**: Finding most likely sequences in HMMs
- **Dynamic Programming**: Optimizing over sequence alignments
- **Neural Networks**: Tropical neural networks with piecewise-linear activations
- **Combinatorics**: Counting optimal solutions

## Features

- **Multiple Semirings**: MaxPlus, MinPlus, MaxMul
- **SIMD Acceleration**: AVX-512, AVX2, SSE4.1, NEON auto-detection
- **CUDA Backend**: GPU-accelerated kernels via runtime compilation
- **Argmax Tracking**: For backpropagation in differentiable programs
- **Batched Operations**: Efficient batch processing
- **Python Bindings**: PyTorch integration via PyO3

## Feature Matrix

### Supported Operations by Semiring and Scalar Type

| Semiring | Scalar | CPU GEMM | CPU Batched | CPU Argmax | CPU Backward | GPU GEMM | GPU Batched | GPU Argmax | GPU Backward |
|----------|--------|----------|-------------|------------|--------------|----------|-------------|------------|--------------|
| MaxPlus | f32 | SIMD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MaxPlus | f64 | SIMD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MaxPlus | i32 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |
| MaxPlus | i64 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |
| MinPlus | f32 | SIMD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MinPlus | f64 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MinPlus | i32 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |
| MinPlus | i64 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |
| MaxMul | f32 | SIMD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MaxMul | f64 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MaxMul | i32 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |
| MaxMul | i64 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |

**Legend:**
- **SIMD**: Optimized with AVX2/AVX-512/NEON vectorization
- **✅**: Supported with portable implementation
- **N/A**: Not applicable (integers don't have gradients)

## Quick Example

```rust
use tropical_gemm::{MatRef, MaxPlus};

// Create 2x3 and 3x2 matrices
let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);

// C[i,j] = max_k(A[i,k] + B[k,j])
let c = &a * &b;
assert_eq!(c.get_value(0, 0), 8.0); // max(1+1, 2+3, 3+5) = 8
```
