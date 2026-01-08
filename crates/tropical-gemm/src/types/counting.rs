use super::scalar::TropicalScalar;
use super::traits::{SimdTropical, TropicalSemiring, TropicalWithArgmax};
use std::fmt;
use std::ops::{Add, Mul};

/// CountingTropical semiring: tracks both the tropical value and the count of optimal paths.
///
/// For TropicalMaxPlus semantics:
/// - Multiplication: (n₁, c₁) ⊗ (n₂, c₂) = (n₁ + n₂, c₁ × c₂)
/// - Addition: (n₁, c₁) ⊕ (n₂, c₂) =
///   - if n₁ > n₂: (n₁, c₁)
///   - if n₁ < n₂: (n₂, c₂)
///   - if n₁ = n₂: (n₁, c₁ + c₂)
///
/// This is used for:
/// - Counting optimal paths in dynamic programming
/// - Computing partition functions
/// - Gradient computations in certain neural network architectures
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct CountingTropical<T: TropicalScalar, C: TropicalScalar = T> {
    /// The tropical value (using MaxPlus semantics).
    pub value: T,
    /// The count of paths achieving this value.
    pub count: C,
}

impl<T: TropicalScalar, C: TropicalScalar> CountingTropical<T, C> {
    /// Create a new CountingTropical value.
    #[inline(always)]
    pub fn new(value: T, count: C) -> Self {
        Self { value, count }
    }

    /// Create a CountingTropical from a single value with count 1.
    #[inline(always)]
    pub fn from_value(value: T) -> Self {
        Self {
            value,
            count: C::scalar_one(),
        }
    }
}

impl<T: TropicalScalar, C: TropicalScalar> TropicalSemiring for CountingTropical<T, C> {
    type Scalar = T;

    #[inline(always)]
    fn tropical_zero() -> Self {
        Self {
            value: T::neg_infinity(),
            count: C::scalar_zero(),
        }
    }

    #[inline(always)]
    fn tropical_one() -> Self {
        Self {
            value: T::scalar_zero(),
            count: C::scalar_one(),
        }
    }

    #[inline(always)]
    fn tropical_add(self, rhs: Self) -> Self {
        if self.value > rhs.value {
            self
        } else if self.value < rhs.value {
            rhs
        } else {
            // Equal values: add counts
            Self {
                value: self.value,
                count: self.count.scalar_add(rhs.count),
            }
        }
    }

    #[inline(always)]
    fn tropical_mul(self, rhs: Self) -> Self {
        Self {
            value: self.value.scalar_add(rhs.value),
            count: self.count.scalar_mul(rhs.count),
        }
    }

    #[inline(always)]
    fn value(&self) -> T {
        self.value
    }

    #[inline(always)]
    fn from_scalar(s: T) -> Self {
        Self::from_value(s)
    }
}

impl<T: TropicalScalar, C: TropicalScalar> TropicalWithArgmax for CountingTropical<T, C> {
    type Index = u32;

    #[inline(always)]
    fn tropical_add_argmax(self, self_idx: u32, rhs: Self, rhs_idx: u32) -> (Self, u32) {
        if self.value > rhs.value {
            (self, self_idx)
        } else if self.value < rhs.value {
            (rhs, rhs_idx)
        } else {
            // Equal values: add counts, keep first index
            (
                Self {
                    value: self.value,
                    count: self.count.scalar_add(rhs.count),
                },
                self_idx,
            )
        }
    }
}

impl<T: TropicalScalar, C: TropicalScalar> SimdTropical for CountingTropical<T, C> {
    // SIMD for CountingTropical requires SOA layout
    const SIMD_AVAILABLE: bool = true;
    const SIMD_WIDTH: usize = 8;
}

impl<T: TropicalScalar, C: TropicalScalar> Add for CountingTropical<T, C> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.tropical_add(rhs)
    }
}

impl<T: TropicalScalar, C: TropicalScalar> Mul for CountingTropical<T, C> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.tropical_mul(rhs)
    }
}

impl<T: TropicalScalar, C: TropicalScalar> Default for CountingTropical<T, C> {
    #[inline(always)]
    fn default() -> Self {
        Self::tropical_zero()
    }
}

impl<T: TropicalScalar, C: TropicalScalar> fmt::Debug for CountingTropical<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CountingTropical({}, {})", self.value, self.count)
    }
}

impl<T: TropicalScalar, C: TropicalScalar> fmt::Display for CountingTropical<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.value, self.count)
    }
}

impl<T: TropicalScalar, C: TropicalScalar> From<T> for CountingTropical<T, C> {
    #[inline(always)]
    fn from(value: T) -> Self {
        Self::from_value(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semiring_identity() {
        let a = CountingTropical::<f64>::new(5.0, 2.0);
        let zero = CountingTropical::tropical_zero();
        let one = CountingTropical::tropical_one();

        // a ⊕ 0 = a
        let result = a.tropical_add(zero);
        assert_eq!(result.value, a.value);
        assert_eq!(result.count, a.count);

        // a ⊗ 1 = a
        let result = a.tropical_mul(one);
        assert_eq!(result.value, a.value);
        assert_eq!(result.count, a.count);
    }

    #[test]
    fn test_multiplication() {
        let a = CountingTropical::<f64>::new(3.0, 2.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        let result = a.tropical_mul(b);
        // value = 3 + 5 = 8
        assert_eq!(result.value, 8.0);
        // count = 2 * 3 = 6
        assert_eq!(result.count, 6.0);
    }

    #[test]
    fn test_addition_different_values() {
        let a = CountingTropical::<f64>::new(3.0, 2.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        let result = a.tropical_add(b);
        // max(3, 5) = 5, keep count of winner
        assert_eq!(result.value, 5.0);
        assert_eq!(result.count, 3.0);
    }

    #[test]
    fn test_addition_equal_values() {
        let a = CountingTropical::<f64>::new(5.0, 2.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        let result = a.tropical_add(b);
        // same value, add counts
        assert_eq!(result.value, 5.0);
        assert_eq!(result.count, 5.0);
    }

    #[test]
    fn test_addition_self_wins() {
        let a = CountingTropical::<f64>::new(7.0, 1.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        let result = a.tropical_add(b);
        // max(7, 5) = 7, keep count of winner
        assert_eq!(result.value, 7.0);
        assert_eq!(result.count, 1.0);
    }

    #[test]
    fn test_path_counting_example() {
        // Example: counting paths in a graph
        // Path A->B has value 3, count 1 (one path)
        // Path A->C->B has value 3, count 2 (two equivalent paths)
        // Total paths A->B with optimal value: 1 + 2 = 3

        let path1 = CountingTropical::<f64>::new(3.0, 1.0);
        let path2 = CountingTropical::<f64>::new(3.0, 2.0);

        let result = path1.tropical_add(path2);
        assert_eq!(result.value, 3.0);
        assert_eq!(result.count, 3.0);
    }

    #[test]
    fn test_operator_overloads() {
        let a = CountingTropical::<f64>::new(3.0, 2.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        // Add operator
        let result = a + b;
        assert_eq!(result.value, 5.0);
        assert_eq!(result.count, 3.0);

        // Mul operator
        let result = a * b;
        assert_eq!(result.value, 8.0);
        assert_eq!(result.count, 6.0);
    }

    #[test]
    fn test_default() {
        let d = CountingTropical::<f64>::default();
        assert!(d.value.is_infinite() && d.value < 0.0); // -inf
        assert_eq!(d.count, 0.0);
    }

    #[test]
    fn test_display_debug() {
        let a = CountingTropical::<f64>::new(3.0, 2.0);

        assert_eq!(format!("{}", a), "(3, 2)");
        assert_eq!(format!("{:?}", a), "CountingTropical(3, 2)");
    }

    #[test]
    fn test_from() {
        let a: CountingTropical<f64> = 5.0.into();
        assert_eq!(a.value, 5.0);
        assert_eq!(a.count, 1.0); // Default count is 1

        let b = CountingTropical::<f64>::from(3.0);
        assert_eq!(b.value, 3.0);
        assert_eq!(b.count, 1.0);
    }

    #[test]
    fn test_from_value() {
        let a = CountingTropical::<f64>::from_value(7.0);
        assert_eq!(a.value, 7.0);
        assert_eq!(a.count, 1.0);
    }

    #[test]
    fn test_value_and_from_scalar() {
        let a = CountingTropical::<f64>::new(5.0, 2.0);
        assert_eq!(a.value(), 5.0);

        let b = CountingTropical::<f64>::from_scalar(3.0);
        assert_eq!(b.value(), 3.0);
        assert_eq!(b.count, 1.0);
    }

    #[test]
    fn test_argmax_self_wins() {
        let a = CountingTropical::<f64>::new(7.0, 2.0);
        let b = CountingTropical::<f64>::new(3.0, 1.0);

        let (result, idx) = a.tropical_add_argmax(1, b, 2);
        assert_eq!(result.value, 7.0);
        assert_eq!(result.count, 2.0);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_argmax_rhs_wins() {
        let a = CountingTropical::<f64>::new(3.0, 1.0);
        let b = CountingTropical::<f64>::new(7.0, 2.0);

        let (result, idx) = a.tropical_add_argmax(1, b, 2);
        assert_eq!(result.value, 7.0);
        assert_eq!(result.count, 2.0);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_argmax_equal_counts_added() {
        // Equal values: counts are added, first index is kept
        let a = CountingTropical::<f64>::new(5.0, 2.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        let (result, idx) = a.tropical_add_argmax(1, b, 2);
        assert_eq!(result.value, 5.0);
        assert_eq!(result.count, 5.0); // 2 + 3
        assert_eq!(idx, 1); // First index is kept
    }

    #[test]
    fn test_argmax_chain() {
        let mut acc = CountingTropical::<f64>::tropical_zero();
        let mut idx = 0u32;

        // Values with different counts
        let values = [(3.0, 1.0), (7.0, 2.0), (7.0, 3.0), (5.0, 1.0)];
        for (k, &(val, count)) in values.iter().enumerate() {
            let candidate = CountingTropical::new(val, count);
            (acc, idx) = acc.tropical_add_argmax(idx, candidate, k as u32);
        }

        // Max value is 7.0, first encountered at k=1
        // Counts: 2 + 3 = 5 (both k=1 and k=2 have value 7.0)
        assert_eq!(acc.value, 7.0);
        assert_eq!(acc.count, 5.0);
        assert_eq!(idx, 1); // First index where max occurred
    }

    #[test]
    fn test_simd_tropical() {
        assert!(CountingTropical::<f64>::SIMD_AVAILABLE);
        assert_eq!(CountingTropical::<f64>::SIMD_WIDTH, 8);
    }

    #[test]
    fn test_clone_copy() {
        let a = CountingTropical::<f64>::new(5.0, 2.0);
        let a_copy = a;
        let a_clone = a.clone();

        assert_eq!(a.value, a_copy.value);
        assert_eq!(a.count, a_copy.count);
        assert_eq!(a.value, a_clone.value);
        assert_eq!(a.count, a_clone.count);
    }

    #[test]
    fn test_eq() {
        let a1 = CountingTropical::<f64>::new(5.0, 2.0);
        let a2 = CountingTropical::<f64>::new(5.0, 2.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }

    #[test]
    fn test_f32() {
        let a = CountingTropical::<f32>::new(3.0, 2.0);
        let b = CountingTropical::<f32>::new(5.0, 3.0);

        let result = a.tropical_mul(b);
        assert!((result.value - 8.0).abs() < 1e-6);
        assert!((result.count - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_different_count_type() {
        // Use different types for value and count
        let a = CountingTropical::<f64, f32>::new(3.0, 2.0);
        let b = CountingTropical::<f64, f32>::new(5.0, 3.0);

        let result = a.tropical_mul(b);
        assert_eq!(result.value, 8.0);
        assert!((result.count - 6.0).abs() < 1e-6);
    }
}
