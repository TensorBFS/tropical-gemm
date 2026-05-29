use super::scalar::TropicalScalar;
use super::traits::{SimdTropical, TropicalSemiring, TropicalWithArgmax};
use std::fmt;
use std::ops::{Add, Mul};

/// TropicalMaxMul semiring: (ℝ⁺, max, ×)
///
/// - Addition (⊕) = max
/// - Multiplication (⊗) = ×
/// - Zero = 0
/// - One = 1
///
/// This is used for:
/// - Probability computations (non-log space)
/// - Fuzzy logic with product t-norm
#[derive(Copy, Clone, PartialEq)]
#[repr(transparent)]
pub struct TropicalMaxMul<T: TropicalScalar>(pub T);

impl<T: TropicalScalar> TropicalMaxMul<T> {
    /// Create a new TropicalMaxMul value.
    #[inline(always)]
    pub fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TropicalScalar> TropicalSemiring for TropicalMaxMul<T> {
    type Scalar = T;

    #[inline(always)]
    fn tropical_zero() -> Self {
        Self(T::scalar_zero())
    }

    #[inline(always)]
    fn tropical_one() -> Self {
        Self(T::scalar_one())
    }

    #[inline(always)]
    fn tropical_add(self, rhs: Self) -> Self {
        Self(self.0.scalar_max(rhs.0))
    }

    #[inline(always)]
    fn tropical_mul(self, rhs: Self) -> Self {
        Self(self.0.scalar_mul(rhs.0))
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

impl<T: TropicalScalar> TropicalWithArgmax for TropicalMaxMul<T> {
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

impl<T: TropicalScalar> SimdTropical for TropicalMaxMul<T> {
    const SIMD_AVAILABLE: bool = true;
    const SIMD_WIDTH: usize = 8;
}

impl<T: TropicalScalar> Add for TropicalMaxMul<T> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.tropical_add(rhs)
    }
}

impl<T: TropicalScalar> Mul for TropicalMaxMul<T> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.tropical_mul(rhs)
    }
}

impl<T: TropicalScalar> Default for TropicalMaxMul<T> {
    #[inline(always)]
    fn default() -> Self {
        Self::tropical_zero()
    }
}

impl<T: TropicalScalar> fmt::Debug for TropicalMaxMul<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TropicalMaxMul({})", self.0)
    }
}

impl<T: TropicalScalar> fmt::Display for TropicalMaxMul<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: TropicalScalar> From<T> for TropicalMaxMul<T> {
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
        let a = TropicalMaxMul::new(5.0f64);
        let zero = TropicalMaxMul::tropical_zero();
        let one = TropicalMaxMul::tropical_one();

        // a ⊕ 0 = a
        assert_eq!(a.tropical_add(zero), a);
        // a ⊗ 1 = a
        assert_eq!(a.tropical_mul(one), a);
    }

    #[test]
    fn test_operations() {
        let a = TropicalMaxMul::new(3.0f64);
        let b = TropicalMaxMul::new(5.0f64);

        // max(3, 5) = 5
        assert_eq!(a.tropical_add(b).0, 5.0);
        // 3 * 5 = 15
        assert_eq!(a.tropical_mul(b).0, 15.0);
    }

    #[test]
    fn test_absorbing_zero() {
        let a = TropicalMaxMul::new(5.0f64);
        let zero = TropicalMaxMul::tropical_zero();

        // a ⊗ 0 = 0
        assert_eq!(a.tropical_mul(zero), zero);
    }

    #[test]
    fn test_operator_overloads() {
        let a = TropicalMaxMul::new(3.0f64);
        let b = TropicalMaxMul::new(5.0f64);

        // Add operator (max)
        assert_eq!((a + b).0, 5.0);
        assert_eq!((b + a).0, 5.0);

        // Mul operator (product)
        assert_eq!((a * b).0, 15.0);
        assert_eq!((b * a).0, 15.0);
    }

    #[test]
    fn test_default() {
        let d = TropicalMaxMul::<f64>::default();
        assert_eq!(d.0, 0.0); // Zero is 0 for MaxMul
        assert_eq!(d, TropicalMaxMul::tropical_zero());
    }

    #[test]
    fn test_display_debug() {
        let a = TropicalMaxMul::new(5.0f64);

        assert_eq!(format!("{}", a), "5");
        assert_eq!(format!("{:?}", a), "TropicalMaxMul(5)");
    }

    #[test]
    fn test_from() {
        let a: TropicalMaxMul<f64> = 5.0.into();
        assert_eq!(a.0, 5.0);

        let b = TropicalMaxMul::<f64>::from(3.0);
        assert_eq!(b.0, 3.0);
    }

    #[test]
    fn test_value_and_from_scalar() {
        let a = TropicalMaxMul::new(5.0f64);
        assert_eq!(a.value(), 5.0);

        let b = TropicalMaxMul::<f64>::from_scalar(3.0);
        assert_eq!(b.value(), 3.0);
    }

    #[test]
    fn test_argmax_self_wins() {
        let a = TropicalMaxMul::new(7.0f64);
        let b = TropicalMaxMul::new(3.0f64);

        let (result, idx) = a.tropical_add_argmax(1, b, 2);
        assert_eq!(result.0, 7.0);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_argmax_rhs_wins() {
        let a = TropicalMaxMul::new(3.0f64);
        let b = TropicalMaxMul::new(7.0f64);

        let (result, idx) = a.tropical_add_argmax(1, b, 2);
        assert_eq!(result.0, 7.0);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_argmax_equal_self_wins() {
        // When equal, self wins (>= comparison)
        let a = TropicalMaxMul::new(5.0f64);
        let b = TropicalMaxMul::new(5.0f64);

        let (result, idx) = a.tropical_add_argmax(1, b, 2);
        assert_eq!(result.0, 5.0);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_argmax_chain() {
        let mut acc = TropicalMaxMul::tropical_zero();
        let mut idx = 0u32;

        let values = [3.0, 7.0, 2.0, 5.0]; // Max at index 1
        for (k, &val) in values.iter().enumerate() {
            let candidate = TropicalMaxMul::new(val);
            (acc, idx) = acc.tropical_add_argmax(idx, candidate, k as u32);
        }

        assert_eq!(acc.0, 7.0);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_simd_tropical() {
        assert!(TropicalMaxMul::<f64>::SIMD_AVAILABLE);
        assert_eq!(TropicalMaxMul::<f64>::SIMD_WIDTH, 8);
    }

    #[test]
    fn test_clone_copy() {
        let a = TropicalMaxMul::new(5.0f64);
        let a_copy = a;
        let a_clone = a.clone();

        assert_eq!(a, a_copy);
        assert_eq!(a, a_clone);
    }

    #[test]
    fn test_eq() {
        let a1 = TropicalMaxMul::new(5.0f64);
        let a2 = TropicalMaxMul::new(5.0f64);
        let b = TropicalMaxMul::new(3.0f64);

        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }

    #[test]
    fn test_f32() {
        let a = TropicalMaxMul::new(3.0f32);
        let b = TropicalMaxMul::new(5.0f32);

        assert!((a.tropical_add(b).0 - 5.0).abs() < 1e-6);
        assert!((a.tropical_mul(b).0 - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_fuzzy_logic_example() {
        // Fuzzy AND (product) and OR (max)
        let high = TropicalMaxMul::new(0.9f64);
        let medium = TropicalMaxMul::new(0.5f64);
        let low = TropicalMaxMul::new(0.2f64);

        // Fuzzy OR of high and low
        let or_result = high.tropical_add(low);
        assert_eq!(or_result.0, 0.9);

        // Fuzzy AND of high and medium (product t-norm)
        let and_result = high.tropical_mul(medium);
        assert!((and_result.0 - 0.45).abs() < 1e-10);
    }
}
