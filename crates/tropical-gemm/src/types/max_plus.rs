use super::scalar::TropicalScalar;
use super::traits::{SimdTropical, TropicalSemiring, TropicalWithArgmax};
use std::fmt;
use std::ops::{Add, Mul};

/// TropicalMaxPlus semiring: (ℝ ∪ {-∞}, max, +)
///
/// - Addition (⊕) = max
/// - Multiplication (⊗) = +
/// - Zero = -∞
/// - One = 0
///
/// This is the classic tropical semiring used in:
/// - Viterbi algorithm
/// - Shortest path algorithms (with negated weights)
/// - Log-space probability computations
#[derive(Copy, Clone, PartialEq)]
#[repr(transparent)]
pub struct TropicalMaxPlus<T: TropicalScalar>(pub T);

impl<T: TropicalScalar> TropicalMaxPlus<T> {
    /// Create a new TropicalMaxPlus value.
    #[inline(always)]
    pub fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TropicalScalar> TropicalSemiring for TropicalMaxPlus<T> {
    type Scalar = T;

    #[inline(always)]
    fn tropical_zero() -> Self {
        Self(T::neg_infinity())
    }

    #[inline(always)]
    fn tropical_one() -> Self {
        Self(T::scalar_zero())
    }

    #[inline(always)]
    fn tropical_add(self, rhs: Self) -> Self {
        Self(self.0.scalar_max(rhs.0))
    }

    #[inline(always)]
    fn tropical_mul(self, rhs: Self) -> Self {
        Self(self.0.scalar_add(rhs.0))
    }

    #[inline(always)]
    fn value(&self) -> T {
        self.0
    }

    #[inline(always)]
    fn from_scalar(s: T) -> Self {
        Self(s)
    }
}

impl<T: TropicalScalar> TropicalWithArgmax for TropicalMaxPlus<T> {
    type Index = u32;

    #[inline(always)]
    fn tropical_add_argmax(self, self_idx: u32, rhs: Self, rhs_idx: u32) -> (Self, u32) {
        if self.0 >= rhs.0 {
            (self, self_idx)
        } else {
            (rhs, rhs_idx)
        }
    }
}

impl<T: TropicalScalar> SimdTropical for TropicalMaxPlus<T> {
    const SIMD_AVAILABLE: bool = true;
    const SIMD_WIDTH: usize = 8; // f32x8 for AVX2
}

impl<T: TropicalScalar> Add for TropicalMaxPlus<T> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.tropical_add(rhs)
    }
}

impl<T: TropicalScalar> Mul for TropicalMaxPlus<T> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.tropical_mul(rhs)
    }
}

impl<T: TropicalScalar> Default for TropicalMaxPlus<T> {
    #[inline(always)]
    fn default() -> Self {
        Self::tropical_zero()
    }
}

impl<T: TropicalScalar> fmt::Debug for TropicalMaxPlus<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TropicalMaxPlus({})", self.0)
    }
}

impl<T: TropicalScalar> fmt::Display for TropicalMaxPlus<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: TropicalScalar> From<T> for TropicalMaxPlus<T> {
    #[inline(always)]
    fn from(value: T) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semiring_identity() {
        let a = TropicalMaxPlus::new(5.0f64);
        let zero = TropicalMaxPlus::tropical_zero();
        let one = TropicalMaxPlus::tropical_one();

        // a ⊕ 0 = a
        assert_eq!(a.tropical_add(zero), a);
        // a ⊗ 1 = a
        assert_eq!(a.tropical_mul(one), a);
    }

    #[test]
    fn test_operations() {
        let a = TropicalMaxPlus::new(3.0f64);
        let b = TropicalMaxPlus::new(5.0f64);

        // max(3, 5) = 5
        assert_eq!(a.tropical_add(b).0, 5.0);
        // 3 + 5 = 8
        assert_eq!(a.tropical_mul(b).0, 8.0);
    }

    #[test]
    fn test_argmax() {
        let a = TropicalMaxPlus::new(3.0f64);
        let b = TropicalMaxPlus::new(5.0f64);

        let (result, idx) = a.tropical_add_argmax(0, b, 1);
        assert_eq!(result.0, 5.0);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_argmax_left_wins() {
        let a = TropicalMaxPlus::new(7.0f64);
        let b = TropicalMaxPlus::new(3.0f64);

        let (result, idx) = a.tropical_add_argmax(10, b, 20);
        assert_eq!(result.0, 7.0);
        assert_eq!(idx, 10); // Left wins, keep left index
    }

    #[test]
    fn test_argmax_equal_values() {
        // When values are equal, left (self) wins (>= comparison)
        let a = TropicalMaxPlus::new(5.0f64);
        let b = TropicalMaxPlus::new(5.0f64);

        let (result, idx) = a.tropical_add_argmax(1, b, 2);
        assert_eq!(result.0, 5.0);
        assert_eq!(idx, 1); // Equal, so left (self) wins
    }

    #[test]
    fn test_argmax_chain() {
        // Simulate accumulating through k iterations
        let mut acc = TropicalMaxPlus::tropical_zero();
        let mut idx = 0u32;

        let values = [3.0, 7.0, 2.0, 5.0]; // Max is at index 1
        for (k, &val) in values.iter().enumerate() {
            let candidate = TropicalMaxPlus::new(val);
            (acc, idx) = acc.tropical_add_argmax(idx, candidate, k as u32);
        }

        assert_eq!(acc.0, 7.0);
        assert_eq!(idx, 1); // Index where max occurred
    }

    #[test]
    fn test_argmax_neg_infinity() {
        let a = TropicalMaxPlus::tropical_zero(); // -inf
        let b = TropicalMaxPlus::new(-100.0f64);

        let (result, idx) = a.tropical_add_argmax(0, b, 1);
        assert_eq!(result.0, -100.0);
        assert_eq!(idx, 1); // -100 > -inf
    }

    #[test]
    fn test_absorbing_zero() {
        let a = TropicalMaxPlus::new(5.0f64);
        let zero = TropicalMaxPlus::tropical_zero();

        // a ⊗ 0 = a + (-inf) = -inf
        // In tropical max-plus, multiplying by zero (adding -inf) gives -inf
        let result = a.tropical_mul(zero);
        assert!(result.0.is_infinite() && result.0 < 0.0);
    }

    #[test]
    fn test_operator_overloads() {
        let a = TropicalMaxPlus::new(3.0f64);
        let b = TropicalMaxPlus::new(5.0f64);

        // Add operator (max)
        assert_eq!((a + b).0, 5.0);
        assert_eq!((b + a).0, 5.0);

        // Mul operator (add)
        assert_eq!((a * b).0, 8.0);
        assert_eq!((b * a).0, 8.0);
    }

    #[test]
    fn test_default() {
        let d = TropicalMaxPlus::<f64>::default();
        assert!(d.0.is_infinite() && d.0 < 0.0); // -inf
        assert_eq!(d, TropicalMaxPlus::tropical_zero());
    }

    #[test]
    fn test_display_debug() {
        let a = TropicalMaxPlus::new(5.0f64);

        assert_eq!(format!("{}", a), "5");
        assert_eq!(format!("{:?}", a), "TropicalMaxPlus(5)");
    }

    #[test]
    fn test_from() {
        let a: TropicalMaxPlus<f64> = 5.0.into();
        assert_eq!(a.0, 5.0);

        let b = TropicalMaxPlus::<f64>::from(3.0);
        assert_eq!(b.0, 3.0);
    }

    #[test]
    fn test_value_and_from_scalar() {
        let a = TropicalMaxPlus::new(5.0f64);
        assert_eq!(a.value(), 5.0);

        let b = TropicalMaxPlus::<f64>::from_scalar(3.0);
        assert_eq!(b.value(), 3.0);
    }

    #[test]
    fn test_simd_tropical() {
        assert!(TropicalMaxPlus::<f64>::SIMD_AVAILABLE);
        assert_eq!(TropicalMaxPlus::<f64>::SIMD_WIDTH, 8);
    }

    #[test]
    fn test_clone_copy() {
        let a = TropicalMaxPlus::new(5.0f64);
        let a_copy = a;
        let a_clone = a.clone();

        assert_eq!(a, a_copy);
        assert_eq!(a, a_clone);
    }

    #[test]
    fn test_eq() {
        let a1 = TropicalMaxPlus::new(5.0f64);
        let a2 = TropicalMaxPlus::new(5.0f64);
        let b = TropicalMaxPlus::new(3.0f64);

        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }

    #[test]
    fn test_f32() {
        let a = TropicalMaxPlus::new(3.0f32);
        let b = TropicalMaxPlus::new(5.0f32);

        assert!((a.tropical_add(b).0 - 5.0).abs() < 1e-6);
        assert!((a.tropical_mul(b).0 - 8.0).abs() < 1e-6);
    }
}
