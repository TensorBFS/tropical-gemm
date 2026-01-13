# Semiring Types

A **semiring** is an algebraic structure with two operations that generalize addition
and multiplication. Tropical semirings replace standard operations with max/min and addition.

## Available Semirings

| Type | ⊕ (add) | ⊗ (mul) | Zero | One | Use Case |
|------|---------|---------|------|-----|----------|
| `MaxPlus<T>` | max | + | -∞ | 0 | Longest path, Viterbi |
| `MinPlus<T>` | min | + | +∞ | 0 | Shortest path, Dijkstra |
| `MaxMul<T>` | max | × | 0 | 1 | Maximum probability |
| `AndOr` | OR | AND | false | true | Graph reachability |

## MaxPlus Semiring

The **MaxPlus** (or max-plus) semiring uses:
- Addition: `a ⊕ b = max(a, b)`
- Multiplication: `a ⊗ b = a + b`

```rust
use tropical_gemm::{MaxPlus, TropicalSemiring};

let a = MaxPlus::from_scalar(3.0f32);
let b = MaxPlus::from_scalar(5.0f32);

// Tropical add: max(3, 5) = 5
let sum = MaxPlus::tropical_add(a, b);
assert_eq!(sum.value(), 5.0);

// Tropical mul: 3 + 5 = 8
let product = MaxPlus::tropical_mul(a, b);
assert_eq!(product.value(), 8.0);
```

**Applications:**
- Longest path in graphs (Bellman-Ford with negated weights)
- Viterbi algorithm for HMM decoding
- Log-probability computations

## MinPlus Semiring

The **MinPlus** (or min-plus) semiring uses:
- Addition: `a ⊕ b = min(a, b)`
- Multiplication: `a ⊗ b = a + b`

```rust
use tropical_gemm::{MinPlus, TropicalSemiring};

let a = MinPlus::from_scalar(3.0f32);
let b = MinPlus::from_scalar(5.0f32);

// Tropical add: min(3, 5) = 3
let sum = MinPlus::tropical_add(a, b);
assert_eq!(sum.value(), 3.0);

// Tropical mul: 3 + 5 = 8
let product = MinPlus::tropical_mul(a, b);
assert_eq!(product.value(), 8.0);
```

**Applications:**
- Shortest path (Floyd-Warshall, Dijkstra)
- Edit distance computation
- Resource allocation

## MaxMul Semiring

The **MaxMul** semiring uses:
- Addition: `a ⊕ b = max(a, b)`
- Multiplication: `a ⊗ b = a × b`

```rust
use tropical_gemm::{MaxMul, TropicalSemiring};

let a = MaxMul::from_scalar(3.0f32);
let b = MaxMul::from_scalar(5.0f32);

// Tropical add: max(3, 5) = 5
let sum = MaxMul::tropical_add(a, b);
assert_eq!(sum.value(), 5.0);

// Tropical mul: 3 × 5 = 15
let product = MaxMul::tropical_mul(a, b);
assert_eq!(product.value(), 15.0);
```

**Applications:**
- Maximum probability paths (non-log space)
- Fuzzy set operations
- Reliability analysis

## Supported Scalar Types

Each semiring supports multiple scalar types:

| Scalar | MaxPlus | MinPlus | MaxMul | Notes |
|--------|---------|---------|--------|-------|
| `f32` | ✅ SIMD | ✅ SIMD | ✅ SIMD | Best performance |
| `f64` | ✅ SIMD | ✅ | ✅ | Higher precision |
| `i32` | ✅ | ✅ | ✅ | Integer operations |
| `i64` | ✅ | ✅ | ✅ | Large integers |

## Type Aliases

For convenience, shorter type aliases are provided:

```rust
use tropical_gemm::{MaxPlus, MinPlus, MaxMul, AndOr};

// These are equivalent:
type A = tropical_gemm::TropicalMaxPlus<f32>;
type B = MaxPlus<f32>;  // Preferred
```
