# Matrix API

tropical-gemm provides a matrix API inspired by [faer](https://github.com/sarah-ek/faer-rs).

## Matrix Types

| Type | Description |
|------|-------------|
| `Mat<S>` | Owned matrix with heap-allocated storage |
| `MatRef<'a, S>` | Immutable view into matrix data |
| `MatMut<'a, S>` | Mutable view into matrix data |
| `MatWithArgmax<S>` | Matrix with argmax indices for backpropagation |

## Creating Matrices

### From Raw Data

```rust
use tropical_gemm::{Mat, MatRef, MaxPlus};

// Create a view from a slice (no allocation)
let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
let a = MatRef::<MaxPlus<f32>>::from_slice(&data, 2, 3);

// Create an owned matrix from a slice (allocates)
let b = Mat::<MaxPlus<f32>>::from_row_major(&data, 2, 3);
```

### Factory Methods

```rust
use tropical_gemm::{Mat, MaxPlus, TropicalSemiring};

// Zero matrix (all elements = tropical zero)
let zeros = Mat::<MaxPlus<f32>>::zeros(3, 4);

// Identity matrix (diagonal = tropical one, off-diagonal = tropical zero)
let identity = Mat::<MaxPlus<f32>>::identity(3);

// From function
let mat = Mat::<MaxPlus<f32>>::from_fn(3, 3, |i, j| {
    MaxPlus::from_scalar((i * 3 + j) as f32)
});
```

## Matrix Multiplication

### Operator Syntax

```rust
use tropical_gemm::{MatRef, MaxPlus};

let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);

// Multiply using operators
let c = &a * &b;  // Returns Mat<S>
```

### Method Syntax

```rust
let c = a.matmul(&b);
```

## Accessing Elements

```rust
use tropical_gemm::{Mat, MaxPlus, TropicalSemiring};

let data = [1.0f32, 2.0, 3.0, 4.0];
let mat = Mat::<MaxPlus<f32>>::from_row_major(&data, 2, 2);

// Get the underlying scalar value
let value = mat.get_value(0, 1);  // 2.0

// Get the tropical element
let elem = mat[(0, 1)];  // MaxPlus(2.0)

// Dimensions
let (rows, cols) = (mat.nrows(), mat.ncols());
```

## Argmax Tracking

For backpropagation, track which k produced each output:

```rust
use tropical_gemm::{Mat, MaxPlus};

let a = Mat::<MaxPlus<f64>>::from_row_major(
    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3
);
let b = Mat::<MaxPlus<f64>>::from_row_major(
    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2
);

let result = a.matmul_argmax(&b);

// Get value and argmax
let value = result.get_value(0, 0);    // 8.0
let k_idx = result.get_argmax(0, 0);   // 2

// Compute gradients
let grad_c = vec![1.0f64; 4];  // upstream gradient (m × n)
let grad_a = result.backward_a(&grad_c);
let grad_b = result.backward_b(&grad_c);
```

## Batched Operations

Process multiple matrices in parallel:

```rust
use tropical_gemm::{Mat, MaxPlus};

let a_batch = vec![
    Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2),
    Mat::<MaxPlus<f32>>::from_row_major(&[5.0, 6.0, 7.0, 8.0], 2, 2),
];
let b_batch = vec![
    Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2),
    Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2),
];

// Batched matmul (parallel by default)
let c_batch = Mat::matmul_batched(&a_batch, &b_batch);

// With argmax
let results = Mat::matmul_batched_with_argmax(&a_batch, &b_batch);
```
