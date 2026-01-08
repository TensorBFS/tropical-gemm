use super::scalar::TropicalScalar;
use super::traits::{SimdTropical, TropicalSemiring, TropicalWithArgmax};
use std::fmt;
use std::ops::{Add, Mul};

/// TropicalMinPlus semiring: (ℝ ∪ {+∞}, min, +)
///
/// - Addition (⊕) = min
/// - Multiplication (⊗) = +
/// - Zero = +∞
/// - One = 0
///
/// This is used for:
/// - Shortest path algorithms (Dijkstra, Floyd-Warshall)
/// - Dynamic programming with minimum cost
#[derive(Copy, Clone, PartialEq)]
#[repr(transparent)]
pub struct TropicalMinPlus<T: TropicalScalar>(pub T);

impl<T: TropicalScalar> TropicalMinPlus<T> {
    /// Create a new TropicalMinPlus value.
    #[inline(always)]
    pub fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TropicalScalar> TropicalSemiring for TropicalMinPlus<T> {
    type Scalar = T;

    #[inline(always)]
    fn tropical_zero() -> Self {
        Self(T::pos_infinity())
    }

    #[inline(always)]
    fn tropical_one() -> Self {
        Self(T::scalar_zero())
    }

    #[inline(always)]
    fn tropical_add(self, rhs: Self) -> Self {
        Self(self.0.scalar_min(rhs.0))
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

impl<T: TropicalScalar> TropicalWithArgmax for TropicalMinPlus<T> {
    type Index = u32;

    #[inline(always)]
    fn tropical_add_argmax(self, self_idx: u32, rhs: Self, rhs_idx: u32) -> (Self, u32) {
        // For min, we track argmin
        if self.0 <= rhs.0 {
            (self, self_idx)
        } else {
            (rhs, rhs_idx)
        }
    }
}

impl<T: TropicalScalar> SimdTropical for TropicalMinPlus<T> {
    const SIMD_AVAILABLE: bool = true;
    const SIMD_WIDTH: usize = 8;
}

impl<T: TropicalScalar> Add for TropicalMinPlus<T> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.tropical_add(rhs)
    }
}

impl<T: TropicalScalar> Mul for TropicalMinPlus<T> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.tropical_mul(rhs)
    }
}

impl<T: TropicalScalar> Default for TropicalMinPlus<T> {
    #[inline(always)]
    fn default() -> Self {
        Self::tropical_zero()
    }
}

impl<T: TropicalScalar> fmt::Debug for TropicalMinPlus<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TropicalMinPlus({})", self.0)
    }
}

impl<T: TropicalScalar> fmt::Display for TropicalMinPlus<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: TropicalScalar> From<T> for TropicalMinPlus<T> {
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
        let a = TropicalMinPlus::new(5.0f64);
        let zero = TropicalMinPlus::tropical_zero();
        let one = TropicalMinPlus::tropical_one();

        // a ⊕ 0 = a
        assert_eq!(a.tropical_add(zero), a);
        // a ⊗ 1 = a
        assert_eq!(a.tropical_mul(one), a);
    }

    #[test]
    fn test_operations() {
        let a = TropicalMinPlus::new(3.0f64);
        let b = TropicalMinPlus::new(5.0f64);

        // min(3, 5) = 3
        assert_eq!(a.tropical_add(b).0, 3.0);
        // 3 + 5 = 8
        assert_eq!(a.tropical_mul(b).0, 8.0);
    }

    #[test]
    fn test_shortest_path_scenario() {
        // Simulating: path cost a=10, path cost b=5, combine = min(10,5) = 5
        let a = TropicalMinPlus::new(10.0f64);
        let b = TropicalMinPlus::new(5.0f64);
        assert_eq!(a.tropical_add(b).0, 5.0);

        // Extending a path: cost=5, edge=3, total = 5+3 = 8
        let path = TropicalMinPlus::new(5.0f64);
        let edge = TropicalMinPlus::new(3.0f64);
        assert_eq!(path.tropical_mul(edge).0, 8.0);
    }

    #[test]
    fn test_argmin_right_wins() {
        // For MinPlus, argmax actually tracks argmin
        let a = TropicalMinPlus::new(5.0f64);
        let b = TropicalMinPlus::new(3.0f64);

        let (result, idx) = a.tropical_add_argmax(0, b, 1);
        assert_eq!(result.0, 3.0);
        assert_eq!(idx, 1); // Right has smaller value
    }

    #[test]
    fn test_argmin_left_wins() {
        let a = TropicalMinPlus::new(2.0f64);
        let b = TropicalMinPlus::new(7.0f64);

        let (result, idx) = a.tropical_add_argmax(10, b, 20);
        assert_eq!(result.0, 2.0);
        assert_eq!(idx, 10); // Left has smaller value
    }

    #[test]
    fn test_argmin_equal_values() {
        // When values are equal, left (self) wins (<= comparison)
        let a = TropicalMinPlus::new(5.0f64);
        let b = TropicalMinPlus::new(5.0f64);

        let (result, idx) = a.tropical_add_argmax(1, b, 2);
        assert_eq!(result.0, 5.0);
        assert_eq!(idx, 1); // Equal, so left (self) wins
    }

    #[test]
    fn test_argmin_chain() {
        // Simulate accumulating through k iterations - find minimum
        let mut acc = TropicalMinPlus::tropical_zero(); // +inf
        let mut idx = 0u32;

        let values = [8.0, 3.0, 9.0, 5.0]; // Min is at index 1
        for (k, &val) in values.iter().enumerate() {
            let candidate = TropicalMinPlus::new(val);
            (acc, idx) = acc.tropical_add_argmax(idx, candidate, k as u32);
        }

        assert_eq!(acc.0, 3.0);
        assert_eq!(idx, 1); // Index where min occurred
    }

    #[test]
    fn test_argmin_pos_infinity() {
        let a = TropicalMinPlus::tropical_zero(); // +inf
        let b = TropicalMinPlus::new(100.0f64);

        let (result, idx) = a.tropical_add_argmax(0, b, 1);
        assert_eq!(result.0, 100.0);
        assert_eq!(idx, 1); // 100 < +inf
    }

    #[test]
    fn test_absorbing_zero() {
        let a = TropicalMinPlus::new(5.0f64);
        let zero = TropicalMinPlus::tropical_zero();

        // a ⊗ 0 = a + (+inf) = +inf
        let result = a.tropical_mul(zero);
        assert!(result.0.is_infinite() && result.0 > 0.0);
    }

    #[test]
    fn test_operator_overloads() {
        let a = TropicalMinPlus::new(3.0f64);
        let b = TropicalMinPlus::new(5.0f64);

        // Add operator (min)
        assert_eq!((a + b).0, 3.0);
        assert_eq!((b + a).0, 3.0);

        // Mul operator (add)
        assert_eq!((a * b).0, 8.0);
        assert_eq!((b * a).0, 8.0);
    }

    #[test]
    fn test_default() {
        let d = TropicalMinPlus::<f64>::default();
        assert!(d.0.is_infinite() && d.0 > 0.0); // +inf
        assert_eq!(d, TropicalMinPlus::tropical_zero());
    }

    #[test]
    fn test_display_debug() {
        let a = TropicalMinPlus::new(5.0f64);

        assert_eq!(format!("{}", a), "5");
        assert_eq!(format!("{:?}", a), "TropicalMinPlus(5)");
    }

    #[test]
    fn test_from() {
        let a: TropicalMinPlus<f64> = 5.0.into();
        assert_eq!(a.0, 5.0);

        let b = TropicalMinPlus::<f64>::from(3.0);
        assert_eq!(b.0, 3.0);
    }

    #[test]
    fn test_value_and_from_scalar() {
        let a = TropicalMinPlus::new(5.0f64);
        assert_eq!(a.value(), 5.0);

        let b = TropicalMinPlus::<f64>::from_scalar(3.0);
        assert_eq!(b.value(), 3.0);
    }

    #[test]
    fn test_simd_tropical() {
        assert!(TropicalMinPlus::<f64>::SIMD_AVAILABLE);
        assert_eq!(TropicalMinPlus::<f64>::SIMD_WIDTH, 8);
    }

    #[test]
    fn test_clone_copy() {
        let a = TropicalMinPlus::new(5.0f64);
        let a_copy = a;
        let a_clone = a.clone();

        assert_eq!(a, a_copy);
        assert_eq!(a, a_clone);
    }

    #[test]
    fn test_eq() {
        let a1 = TropicalMinPlus::new(5.0f64);
        let a2 = TropicalMinPlus::new(5.0f64);
        let b = TropicalMinPlus::new(3.0f64);

        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }

    #[test]
    fn test_f32() {
        let a = TropicalMinPlus::new(3.0f32);
        let b = TropicalMinPlus::new(5.0f32);

        assert!((a.tropical_add(b).0 - 3.0).abs() < 1e-6);
        assert!((a.tropical_mul(b).0 - 8.0).abs() < 1e-6);
    }
}
