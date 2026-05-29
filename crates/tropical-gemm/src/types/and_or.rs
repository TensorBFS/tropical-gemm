use super::traits::{SimdTropical, TropicalSemiring, TropicalWithArgmax};
use std::fmt;
use std::ops::{Add, Mul};

/// TropicalAndOr semiring: ({true, false}, OR, AND)
///
/// - Addition (⊕) = OR (logical disjunction)
/// - Multiplication (⊗) = AND (logical conjunction)
/// - Zero = false
/// - One = true
///
/// This is used for:
/// - Boolean matrix multiplication (transitive closure)
/// - Graph reachability
/// - SAT-related computations
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct TropicalAndOr(pub bool);

impl TropicalAndOr {
    /// Create a new TropicalAndOr value.
    #[inline(always)]
    pub fn new(value: bool) -> Self {
        Self(value)
    }
}

impl TropicalSemiring for TropicalAndOr {
    type Scalar = bool;

    #[inline(always)]
    fn tropical_zero() -> Self {
        Self(false)
    }

    #[inline(always)]
    fn tropical_one() -> Self {
        Self(true)
    }

    #[inline(always)]
    fn tropical_add(self, rhs: Self) -> Self {
        Self(self.0 || rhs.0)
    }

    #[inline(always)]
    fn tropical_mul(self, rhs: Self) -> Self {
        Self(self.0 && rhs.0)
    }

    #[inline(always)]
    fn value(&self) -> bool {
        self.0
    }

    #[inline(always)]
    fn from_scalar(s: bool) -> Self {
        Self(s)
    }
}

impl TropicalWithArgmax for TropicalAndOr {
    type Index = u32;

    #[inline(always)]
    fn tropical_add_argmax(self, self_idx: u32, rhs: Self, rhs_idx: u32) -> (Self, u32) {
        // For OR, return index of first true (or last if both false)
        if self.0 {
            (self, self_idx)
        } else if rhs.0 {
            (rhs, rhs_idx)
        } else {
            (self, self_idx)
        }
    }
}

impl SimdTropical for TropicalAndOr {
    // Bool operations can be SIMD'd via bitmasks
    const SIMD_AVAILABLE: bool = true;
    const SIMD_WIDTH: usize = 256; // 256 bits = 256 bools for AVX2
}

impl Add for TropicalAndOr {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.tropical_add(rhs)
    }
}

impl Mul for TropicalAndOr {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.tropical_mul(rhs)
    }
}

impl Default for TropicalAndOr {
    #[inline(always)]
    fn default() -> Self {
        Self::tropical_zero()
    }
}

impl fmt::Debug for TropicalAndOr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TropicalAndOr({})", self.0)
    }
}

impl fmt::Display for TropicalAndOr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<bool> for TropicalAndOr {
    #[inline(always)]
    fn from(value: bool) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semiring_identity() {
        let a = TropicalAndOr::new(true);
        let zero = TropicalAndOr::tropical_zero();
        let one = TropicalAndOr::tropical_one();

        // a ⊕ 0 = a
        assert_eq!(a.tropical_add(zero), a);
        // a ⊗ 1 = a
        assert_eq!(a.tropical_mul(one), a);

        // Test with false value too
        let f = TropicalAndOr::new(false);
        assert_eq!(f.tropical_add(zero), f);
        assert_eq!(f.tropical_mul(one), f);
    }

    #[test]
    fn test_operations() {
        let t = TropicalAndOr::new(true);
        let f = TropicalAndOr::new(false);

        // OR operations
        assert!(t.tropical_add(f).0);
        assert!(!f.tropical_add(f).0);
        assert!(t.tropical_add(t).0);

        // AND operations
        assert!(!t.tropical_mul(f).0);
        assert!(t.tropical_mul(t).0);
        assert!(!f.tropical_mul(f).0);
    }

    #[test]
    fn test_absorbing_zero() {
        let a = TropicalAndOr::new(true);
        let zero = TropicalAndOr::tropical_zero();

        // a ⊗ 0 = 0
        assert_eq!(a.tropical_mul(zero), zero);
    }

    #[test]
    fn test_reachability_example() {
        // Graph adjacency: can we reach node j from node i?
        // A[0,1] = true (0->1), A[1,2] = true (1->2)
        // (A*A)[0,2] = A[0,0]*A[0,2] OR A[0,1]*A[1,2] = false OR true = true
        let a01 = TropicalAndOr::new(true);
        let a12 = TropicalAndOr::new(true);
        let a00 = TropicalAndOr::new(false);
        let a02 = TropicalAndOr::new(false);

        let result = a00.tropical_mul(a02).tropical_add(a01.tropical_mul(a12));
        assert!(result.0);
    }

    #[test]
    fn test_operator_overloads() {
        let t = TropicalAndOr::new(true);
        let f = TropicalAndOr::new(false);

        // Add operator (OR)
        assert!((t + f).0);
        assert!((f + t).0);
        assert!(!(f + f).0);
        assert!((t + t).0);

        // Mul operator (AND)
        assert!(!(t * f).0);
        assert!(!(f * t).0);
        assert!(!(f * f).0);
        assert!((t * t).0);
    }

    #[test]
    fn test_default() {
        let d = TropicalAndOr::default();
        assert!(!d.0); // Default is zero (false)
        assert_eq!(d, TropicalAndOr::tropical_zero());
    }

    #[test]
    fn test_display_debug() {
        let t = TropicalAndOr::new(true);
        let f = TropicalAndOr::new(false);

        assert_eq!(format!("{}", t), "true");
        assert_eq!(format!("{}", f), "false");
        assert_eq!(format!("{:?}", t), "TropicalAndOr(true)");
        assert_eq!(format!("{:?}", f), "TropicalAndOr(false)");
    }

    #[test]
    fn test_from() {
        let t: TropicalAndOr = true.into();
        let f: TropicalAndOr = false.into();

        assert!(t.0);
        assert!(!f.0);

        // Using From trait directly
        let t2 = TropicalAndOr::from(true);
        let f2 = TropicalAndOr::from(false);
        assert!(t2.0);
        assert!(!f2.0);
    }

    #[test]
    fn test_value_and_from_scalar() {
        let t = TropicalAndOr::new(true);
        let f = TropicalAndOr::new(false);

        assert!(t.value());
        assert!(!f.value());

        let t2 = TropicalAndOr::from_scalar(true);
        let f2 = TropicalAndOr::from_scalar(false);
        assert!(t2.value());
        assert!(!f2.value());
    }

    #[test]
    fn test_argmax_self_true() {
        // If self is true, return self
        let t = TropicalAndOr::new(true);
        let f = TropicalAndOr::new(false);

        let (result, idx) = t.tropical_add_argmax(1, f, 2);
        assert!(result.0);
        assert_eq!(idx, 1);

        let (result, idx) = t.tropical_add_argmax(5, t, 10);
        assert!(result.0);
        assert_eq!(idx, 5);
    }

    #[test]
    fn test_argmax_rhs_true() {
        // If self is false but rhs is true, return rhs
        let t = TropicalAndOr::new(true);
        let f = TropicalAndOr::new(false);

        let (result, idx) = f.tropical_add_argmax(1, t, 2);
        assert!(result.0);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_argmax_both_false() {
        // If both are false, return self (first one)
        let f1 = TropicalAndOr::new(false);
        let f2 = TropicalAndOr::new(false);

        let (result, idx) = f1.tropical_add_argmax(10, f2, 20);
        assert!(!result.0);
        assert_eq!(idx, 10);
    }

    #[test]
    fn test_argmax_chain() {
        // Simulate accumulating through k iterations - find first true
        let mut acc = TropicalAndOr::tropical_zero();
        let mut idx = 0u32;

        let values = [false, false, true, false]; // First true at index 2
        for (k, &val) in values.iter().enumerate() {
            let candidate = TropicalAndOr::new(val);
            (acc, idx) = acc.tropical_add_argmax(idx, candidate, k as u32);
        }

        assert!(acc.0);
        assert_eq!(idx, 2); // Index where first true occurred
    }

    #[test]
    fn test_simd_tropical() {
        assert!(TropicalAndOr::SIMD_AVAILABLE);
        assert_eq!(TropicalAndOr::SIMD_WIDTH, 256);
    }

    #[test]
    fn test_clone_copy() {
        let t = TropicalAndOr::new(true);
        let t_copy = t; // Copy
        let t_clone = t.clone(); // Clone

        assert_eq!(t, t_copy);
        assert_eq!(t, t_clone);
    }

    #[test]
    fn test_eq() {
        let t1 = TropicalAndOr::new(true);
        let t2 = TropicalAndOr::new(true);
        let f1 = TropicalAndOr::new(false);
        let f2 = TropicalAndOr::new(false);

        assert_eq!(t1, t2);
        assert_eq!(f1, f2);
        assert_ne!(t1, f1);
    }
}
