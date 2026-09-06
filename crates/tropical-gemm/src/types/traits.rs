use super::scalar::TropicalScalar;
use std::fmt::Debug;

/// Core trait for tropical semiring operations.
///
/// A semiring (S, ⊕, ⊗) satisfies:
/// - (S, ⊕) is a commutative monoid with identity `tropical_zero`
/// - (S, ⊗) is a monoid with identity `tropical_one`
/// - ⊗ distributes over ⊕
/// - `tropical_zero` is absorbing: a ⊗ 0 = 0 ⊗ a = 0
pub trait TropicalSemiring: Copy + Clone + Send + Sync + Debug + PartialEq + 'static {
    /// The underlying scalar type.
    type Scalar: TropicalScalar;

    /// Returns the additive identity (zero element for ⊕).
    fn tropical_zero() -> Self;

    /// Returns the multiplicative identity (one element for ⊗).
    fn tropical_one() -> Self;

    /// Tropical addition (⊕).
    fn tropical_add(self, rhs: Self) -> Self;

    /// Tropical multiplication (⊗).
    fn tropical_mul(self, rhs: Self) -> Self;

    /// Get the underlying scalar value.
    fn value(&self) -> Self::Scalar;

    /// Create from a scalar value.
    fn from_scalar(s: Self::Scalar) -> Self;
}

/// Extension trait for tropical types that support argmax tracking.
///
/// This is used for backpropagation: during matrix multiplication,
/// we track which k index produced the optimal value for each C[i,j].
pub trait TropicalWithArgmax: TropicalSemiring {
    /// The index type used for argmax tracking.
    type Index: Copy + Default + Debug + Send + Sync + 'static;

    /// Tropical addition with argmax tracking.
    ///
    /// Returns the result of `tropical_add` along with the index
    /// corresponding to which operand "won" (produced the result).
    fn tropical_add_argmax(
        self,
        self_idx: Self::Index,
        rhs: Self,
        rhs_idx: Self::Index,
    ) -> (Self, Self::Index);

    /// Whether this (output) value is a tropical-zero "no contribution" cell
    /// whose argmax index should be canonicalized at GEMM write-back.
    ///
    /// Integer tropical zeros use a guard-free in-band sentinel, so a
    /// no-contribution cell's value drifts and its accumulated argmax adopts a
    /// spurious `k`. Returning `true` lets the kernel reset that index to the
    /// deterministic seed (`0`) so the whole repo agrees on one value for such
    /// cells (and the backward pass routes no gradient there once that seed
    /// becomes `-1`, a later step).
    ///
    /// Default `false`: exact-infinity types (floats) don't drift — their zero
    /// cells already keep the seed — so the branch folds away for them.
    #[inline(always)]
    fn is_no_contribution(&self) -> bool {
        false
    }
}

/// Marker trait for tropical types that support SIMD acceleration.
pub trait SimdTropical: TropicalSemiring {
    /// Whether SIMD operations are available for this type.
    const SIMD_AVAILABLE: bool;

    /// The SIMD width in elements.
    const SIMD_WIDTH: usize;
}
