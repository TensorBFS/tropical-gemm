# Backward Rules for Tropical GEMM

This document describes the backward propagation (argmax tracking) implementation in the tropical-gemm library.

## Overview

In tropical matrix multiplication, each output element C[i,j] is computed as:

```
C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])
```

Where ⊕ is tropical addition and ⊗ is tropical multiplication.

For **backpropagation** in tropical neural networks, we need to track which k index produced the optimal value at each position. This is called **argmax tracking** (or argmin for MinPlus semiring).

## Implementation

### Trait Definition

The `TropicalWithArgmax` trait in `crates/tropical-types/src/traits.rs` defines the argmax interface:

```rust
pub trait TropicalWithArgmax: TropicalSemiring {
    type Index: Copy + Default + Debug + Send + Sync + 'static;

    fn tropical_add_argmax(
        self,
        self_idx: Self::Index,
        rhs: Self,
        rhs_idx: Self::Index,
    ) -> (Self, Self::Index);
}
```

### Semiring-Specific Implementations

#### MaxPlus (argmax)

For MaxPlus semiring where ⊕ = max and ⊗ = +, the argmax tracks which operand had the larger value:

```rust
// In crates/tropical-types/src/max_plus.rs
fn tropical_add_argmax(self, self_idx: u32, rhs: Self, rhs_idx: u32) -> (Self, u32) {
    if self.0 >= rhs.0 {
        (self, self_idx)
    } else {
        (rhs, rhs_idx)
    }
}
```

**Rule**: Returns the index of the **maximum** value. On ties, the left (existing) index is preserved.

#### MinPlus (argmin)

For MinPlus semiring where ⊕ = min and ⊗ = +, the tracking becomes argmin:

```rust
// In crates/tropical-types/src/min_plus.rs
fn tropical_add_argmax(self, self_idx: u32, rhs: Self, rhs_idx: u32) -> (Self, u32) {
    if self.0 <= rhs.0 {
        (self, self_idx)
    } else {
        (rhs, rhs_idx)
    }
}
```

**Rule**: Returns the index of the **minimum** value. On ties, the left (existing) index is preserved.

### Microkernel Implementation

The portable microkernel with argmax tracking is in `crates/tropical-gemm-core/src/kernel.rs`:

```rust
unsafe fn execute_with_argmax(
    &self,
    mr: usize, nr: usize, k: usize,
    k_offset: usize,  // Global k offset for multi-block accumulation
    a: *const T::Scalar, b: *const T::Scalar,
    c: *mut T, argmax: *mut u32, ldc: usize,
) {
    // Initialize accumulators from existing C and argmax
    let mut acc = [[T::tropical_zero(); NR]; MR];
    let mut idx = [[0u32; NR]; MR];
    for i in 0..mr {
        for j in 0..nr {
            acc[i][j] = *c.add(i * ldc + j);
            idx[i][j] = *argmax.add(i * ldc + j);
        }
    }

    // Main loop with argmax tracking
    for p in 0..k {
        let current_k = (k_offset + p) as u32;  // Global k index
        for i in 0..mr {
            let a_val = T::from_scalar(*a.add(p * MR + i));
            for j in 0..nr {
                let b_val = T::from_scalar(*b.add(p * NR + j));
                let product = a_val.tropical_mul(b_val);
                let (new_acc, new_idx) =
                    acc[i][j].tropical_add_argmax(idx[i][j], product, current_k);
                acc[i][j] = new_acc;
                idx[i][j] = new_idx;
            }
        }
    }

    // Write back results
    for i in 0..mr {
        for j in 0..nr {
            *c.add(i * ldc + j) = acc[i][j];
            *argmax.add(i * ldc + j) = idx[i][j];
        }
    }
}
```

**Key points**:
1. The `k_offset` parameter handles blocked GEMM where we process k in chunks
2. Argmax indices are **global** (k_offset + local_p), not local to the block
3. Existing accumulator values and indices are loaded at the start, allowing incremental updates

### Result Storage

The `GemmWithArgmax` struct in `crates/tropical-gemm-core/src/argmax.rs` stores both values and indices:

```rust
pub struct GemmWithArgmax<T: TropicalWithArgmax<Index = u32>> {
    pub values: Vec<T>,    // The result matrix values
    pub argmax: Vec<u32>,  // The k-index that produced each optimal value
    pub m: usize,
    pub n: usize,
    pub ld: usize,         // Leading dimension (stride between rows)
}
```

## Usage Example

```rust
use tropical_gemm::prelude::*;

// Matrix multiplication with argmax tracking
let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];  // 2x3
let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];  // 3x2

let result = tropical_matmul_with_argmax::<TropicalMaxPlus<f64>>(&a, 2, 3, &b, 2);

// For C[0,0]:
// k=0: A[0,0] + B[0,0] = 1 + 1 = 2
// k=1: A[0,1] + B[1,0] = 2 + 3 = 5
// k=2: A[0,2] + B[2,0] = 3 + 5 = 8  <-- max
assert_eq!(result.get(0, 0).value(), 8.0);
assert_eq!(result.get_argmax(0, 0), 2);  // k=2 produced the max
```

## Applications

### 1. Viterbi Algorithm (MaxPlus)
Track which state transition led to the maximum probability path.

### 2. Shortest Path (MinPlus)
Track which intermediate node was on the shortest path between two vertices.

### 3. Tropical Neural Networks
Compute gradients by knowing which path was optimal during forward pass.

### 4. Backpropagation in Tensor Networks
Determine optimal contraction sequences for gradient computation.

## Test Coverage

The backward rules are tested in:
- `crates/tropical-types/src/max_plus.rs` - argmax unit tests
- `crates/tropical-types/src/min_plus.rs` - argmin unit tests
- `crates/tropical-gemm-core/src/kernel.rs` - microkernel argmax tests
- `crates/tropical-gemm-core/src/gemm.rs` - full GEMM argmax tests
- `crates/tropical-gemm/src/api.rs` - API-level argmax tests

Key test scenarios:
- Basic argmax/argmin selection
- Tie-breaking (left/existing value wins)
- Chain accumulation (simulating k-loop)
- Global k_offset handling in blocked GEMM
- Different semirings (MaxPlus, MinPlus)
