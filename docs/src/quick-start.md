# Quick Start

This guide shows the basics of tropical matrix multiplication.

## Basic Matrix Multiplication

```rust
use tropical_gemm::{Mat, MatRef, MaxPlus};

// Create matrices from raw data (row-major order)
let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2 matrix

// Create matrix views
let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);

// Multiply using operator
let c = &a * &b;

// Or using method
let c = a.matmul(&b);

// Access result
println!("C[0,0] = {}", c.get_value(0, 0)); // 8.0 = max(1+1, 2+3, 3+5)
```

## Understanding the Result

For MaxPlus semiring, the multiplication computes:

```
C[i,j] = max_k(A[i,k] + B[k,j])
```

For the example above:
- C[0,0] = max(1+1, 2+3, 3+5) = max(2, 5, 8) = **8**
- C[0,1] = max(1+2, 2+4, 3+6) = max(3, 6, 9) = **9**
- C[1,0] = max(4+1, 5+3, 6+5) = max(5, 8, 11) = **11**
- C[1,1] = max(4+2, 5+4, 6+6) = max(6, 9, 12) = **12**

## Using Different Semirings

```rust
use tropical_gemm::{MatRef, MaxPlus, MinPlus, MaxMul};

let a_data = [1.0f32, 2.0, 3.0, 4.0];
let b_data = [1.0f32, 2.0, 3.0, 4.0];

// MaxPlus: C[i,j] = max_k(A[i,k] + B[k,j])
let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 2);
let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 2, 2);
let c_maxplus = &a * &b;

// MinPlus: C[i,j] = min_k(A[i,k] + B[k,j])
let a = MatRef::<MinPlus<f32>>::from_slice(&a_data, 2, 2);
let b = MatRef::<MinPlus<f32>>::from_slice(&b_data, 2, 2);
let c_minplus = &a * &b;

// MaxMul: C[i,j] = max_k(A[i,k] * B[k,j])
let a = MatRef::<MaxMul<f32>>::from_slice(&a_data, 2, 2);
let b = MatRef::<MaxMul<f32>>::from_slice(&b_data, 2, 2);
let c_maxmul = &a * &b;
```

## Factory Methods

```rust
use tropical_gemm::{Mat, MaxPlus};

// Create a zero matrix (all -∞ for MaxPlus)
let zeros = Mat::<MaxPlus<f32>>::zeros(3, 3);

// Create an identity matrix (0 on diagonal, -∞ elsewhere for MaxPlus)
let identity = Mat::<MaxPlus<f32>>::identity(3);

// Create from function
let mat = Mat::<MaxPlus<f32>>::from_fn(3, 3, |i, j| {
    MaxPlus::from_scalar((i + j) as f32)
});
```

## Next Steps

- [Semiring Types](./semirings.md) - Learn about different tropical semirings
- [Matrix API](./matrix-api.md) - Full matrix API reference
- [GPU Acceleration](./gpu.md) - Using CUDA for large matrices
