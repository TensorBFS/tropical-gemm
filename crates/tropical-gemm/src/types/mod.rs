//! Tropical semiring type definitions.
//!
//! This module defines the algebraic structures used for tropical matrix
//! multiplication. A **tropical semiring** replaces the standard arithmetic
//! operations (+, ×) with alternative operations, typically (max, +) or (min, +).
//!
//! # What is a Tropical Semiring?
//!
//! A semiring is an algebraic structure with two binary operations:
//! - **Addition** (⊕): An associative, commutative operation with identity (zero)
//! - **Multiplication** (⊗): An associative operation with identity (one)
//!
//! In tropical algebra, these become:
//!
//! | Type | ⊕ (add) | ⊗ (mul) | Zero | One | Use Case |
//! |------|---------|---------|------|-----|----------|
//! | [`TropicalMaxPlus<T>`] | max | + | -∞ | 0 | Longest path, Viterbi algorithm |
//! | [`TropicalMinPlus<T>`] | min | + | +∞ | 0 | Shortest path, Dijkstra |
//! | [`TropicalMaxMul<T>`] | max | × | 0 | 1 | Maximum probability paths |
//! | [`TropicalAndOr`] | OR | AND | false | true | Graph reachability |
//!
//! # Tropical Matrix Multiplication
//!
//! For matrices A (m×k) and B (k×n), the tropical product C = A ⊗ B is:
//!
//! ```text
//! C[i,j] = ⊕_{k} (A[i,k] ⊗ B[k,j])
//! ```
//!
//! For MaxPlus, this becomes: `C[i,j] = max_k(A[i,k] + B[k,j])`
//!
//! # Core Traits
//!
//! - [`TropicalSemiring`]: Defines the semiring operations (add, mul, zero, one)
//! - [`TropicalWithArgmax`]: Extends semiring with argmax tracking for backpropagation
//! - [`TropicalScalar`]: Trait for scalar types that can be used in tropical operations
//!
//! # Example
//!
//! ```rust
//! use tropical_gemm::types::{TropicalMaxPlus, TropicalSemiring};
//!
//! // Create tropical numbers
//! let a = TropicalMaxPlus::from_scalar(3.0f32);
//! let b = TropicalMaxPlus::from_scalar(5.0f32);
//!
//! // Tropical addition: max(3, 5) = 5
//! let sum = TropicalMaxPlus::tropical_add(a, b);
//! assert_eq!(sum.value(), 5.0);
//!
//! // Tropical multiplication: 3 + 5 = 8
//! let product = TropicalMaxPlus::tropical_mul(a, b);
//! assert_eq!(product.value(), 8.0);
//! ```
//!
//! # Type Aliases
//!
//! For convenience, the crate provides shorter type aliases:
//!
//! ```rust
//! use tropical_gemm::{MaxPlus, MinPlus, MaxMul, TropicalSemiring};
//!
//! let x: MaxPlus<f32> = MaxPlus::from_scalar(1.0);
//! let y: MinPlus<f64> = MinPlus::from_scalar(2.0);
//! let z: MaxMul<f32> = MaxMul::from_scalar(3.0);
//! ```

mod and_or;
mod counting;
mod max_mul;
mod max_plus;
mod min_plus;
mod scalar;
mod traits;

pub use and_or::TropicalAndOr;
pub use counting::CountingTropical;
pub use max_mul::TropicalMaxMul;
pub use max_plus::TropicalMaxPlus;
pub use min_plus::TropicalMinPlus;
pub use scalar::TropicalScalar;
pub use traits::{SimdTropical, TropicalSemiring, TropicalWithArgmax};
