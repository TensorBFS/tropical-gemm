use std::fmt::{Debug, Display};

/// Trait for scalar types that can be used as underlying values in tropical numbers.
pub trait TropicalScalar:
    Copy + Clone + Send + Sync + Debug + Display + PartialOrd + 'static + Sized
{
    /// The additive identity (standard arithmetic).
    fn scalar_zero() -> Self;

    /// The multiplicative identity (standard arithmetic).
    fn scalar_one() -> Self;

    /// Standard arithmetic addition.
    fn scalar_add(self, rhs: Self) -> Self;

    /// Standard arithmetic multiplication.
    fn scalar_mul(self, rhs: Self) -> Self;

    /// Positive infinity (for MinPlus zero).
    fn pos_infinity() -> Self;

    /// Negative infinity (for MaxPlus zero).
    fn neg_infinity() -> Self;

    /// Maximum of two values.
    fn scalar_max(self, rhs: Self) -> Self;

    /// Minimum of two values.
    fn scalar_min(self, rhs: Self) -> Self;

    /// Whether `self` is an in-band integer sentinel that has *drifted* off the
    /// canonical negative tropical zero (`-∞`).
    ///
    /// Integer tropical zeros use a finite sentinel plus a guard-free `+`, so a
    /// no-contribution cell's value lands in "infinity territory" (past
    /// `neg_infinity() / 2`) without being exactly the sentinel. This is used at
    /// GEMM write-back to canonicalize the argmax index of such cells.
    ///
    /// Exact-infinity representations (floats) never drift, and narrow / unsigned
    /// integers are out of the headroom-sentinel scheme, so the default is
    /// `false` — only `i32`/`i64` override it, which lets the canonicalization
    /// branch fold away entirely for every other monomorphization.
    #[inline(always)]
    fn is_drifted_neg_zero(self) -> bool {
        false
    }

    /// Positive (`+∞`) counterpart of [`TropicalScalar::is_drifted_neg_zero`].
    #[inline(always)]
    fn is_drifted_pos_zero(self) -> bool {
        false
    }
}

macro_rules! impl_tropical_scalar_float {
    ($($t:ty),*) => {
        $(
            impl TropicalScalar for $t {
                #[inline(always)]
                fn scalar_zero() -> Self {
                    0.0
                }

                #[inline(always)]
                fn scalar_one() -> Self {
                    1.0
                }

                #[inline(always)]
                fn scalar_add(self, rhs: Self) -> Self {
                    self + rhs
                }

                #[inline(always)]
                fn scalar_mul(self, rhs: Self) -> Self {
                    self * rhs
                }

                #[inline(always)]
                fn pos_infinity() -> Self {
                    <$t>::INFINITY
                }

                #[inline(always)]
                fn neg_infinity() -> Self {
                    <$t>::NEG_INFINITY
                }

                #[inline(always)]
                fn scalar_max(self, rhs: Self) -> Self {
                    if self >= rhs { self } else { rhs }
                }

                #[inline(always)]
                fn scalar_min(self, rhs: Self) -> Self {
                    if self <= rhs { self } else { rhs }
                }
            }
        )*
    };
}

/// The arithmetic + ordering methods shared by every integer `TropicalScalar`
/// impl, regardless of how it represents the tropical zero (`MIN`/`MAX` for the
/// narrow types, a headroom sentinel for `i32`/`i64`). The integer literals `0`
/// and `1` infer to `Self` at each concrete instantiation.
macro_rules! tropical_int_common {
    () => {
        #[inline(always)]
        fn scalar_zero() -> Self {
            0
        }

        #[inline(always)]
        fn scalar_one() -> Self {
            1
        }

        #[inline(always)]
        fn scalar_add(self, rhs: Self) -> Self {
            self + rhs
        }

        #[inline(always)]
        fn scalar_mul(self, rhs: Self) -> Self {
            self * rhs
        }

        #[inline(always)]
        fn scalar_max(self, rhs: Self) -> Self {
            if self >= rhs {
                self
            } else {
                rhs
            }
        }

        #[inline(always)]
        fn scalar_min(self, rhs: Self) -> Self {
            if self <= rhs {
                self
            } else {
                rhs
            }
        }
    };
}

/// Narrow / unsigned integers use `MIN`/`MAX` as the tropical zero and keep the
/// default (`false`) drift hooks — they are out of the headroom-sentinel scheme,
/// so a guard-free `zero ⊗ zero` (`MIN + MIN`) can overflow. These types are not
/// wired into `KernelDispatch`, so they are only reachable through the low-level
/// core kernels, not the public `tropical_matmul*` API.
macro_rules! impl_tropical_scalar_int {
    ($($t:ty),*) => {
        $(
            impl TropicalScalar for $t {
                tropical_int_common!();

                #[inline(always)]
                fn pos_infinity() -> Self {
                    <$t>::MAX
                }

                #[inline(always)]
                fn neg_infinity() -> Self {
                    <$t>::MIN
                }
            }
        )*
    };
}

/// Wide signed integers (`i32`/`i64`) use a large *headroom* sentinel instead of
/// `MIN`/`MAX`, so a guard-free `+` neither overflows on `zero ⊗ zero`
/// (`±S + ±S` stays in range) nor collides with realistic data, and a drifted
/// tropical zero is detectable by the `|value| >= |S|/2` threshold. This matches
/// the CUDA backend (`NEG_INF_I32 = -1e9`, `NEG_INF_I64 = -(1 << 60)`).
macro_rules! impl_tropical_scalar_int_wide {
    ($($t:ty => ($neg:expr, $pos:expr)),* $(,)?) => {
        $(
            impl TropicalScalar for $t {
                tropical_int_common!();

                #[inline(always)]
                fn pos_infinity() -> Self {
                    $pos
                }

                #[inline(always)]
                fn neg_infinity() -> Self {
                    $neg
                }

                #[inline(always)]
                fn is_drifted_neg_zero(self) -> bool {
                    self <= $neg / 2
                }

                #[inline(always)]
                fn is_drifted_pos_zero(self) -> bool {
                    self >= $pos / 2
                }
            }
        )*
    };
}

impl_tropical_scalar_float!(f32, f64);
impl_tropical_scalar_int!(i8, i16, u8, u16, u32, u64);
impl_tropical_scalar_int_wide!(
    i32 => (-1_000_000_000, 1_000_000_000),
    i64 => (-(1i64 << 60), 1i64 << 60),
);

impl TropicalScalar for bool {
    #[inline(always)]
    fn scalar_zero() -> Self {
        false
    }

    #[inline(always)]
    fn scalar_one() -> Self {
        true
    }

    #[inline(always)]
    fn scalar_add(self, rhs: Self) -> Self {
        self || rhs
    }

    #[inline(always)]
    fn scalar_mul(self, rhs: Self) -> Self {
        self && rhs
    }

    #[inline(always)]
    fn pos_infinity() -> Self {
        true
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        false
    }

    #[inline(always)]
    fn scalar_max(self, rhs: Self) -> Self {
        self || rhs
    }

    #[inline(always)]
    fn scalar_min(self, rhs: Self) -> Self {
        self && rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_scalar() {
        assert_eq!(f64::scalar_zero(), 0.0);
        assert_eq!(f64::scalar_one(), 1.0);
        assert_eq!(3.0f64.scalar_add(5.0), 8.0);
        assert_eq!(3.0f64.scalar_mul(5.0), 15.0);
        assert!(f64::pos_infinity().is_infinite() && f64::pos_infinity() > 0.0);
        assert!(f64::neg_infinity().is_infinite() && f64::neg_infinity() < 0.0);
        assert_eq!(3.0f64.scalar_max(5.0), 5.0);
        assert_eq!(3.0f64.scalar_min(5.0), 3.0);
    }

    #[test]
    fn test_f32_scalar() {
        assert_eq!(f32::scalar_zero(), 0.0);
        assert_eq!(f32::scalar_one(), 1.0);
        assert!((3.0f32.scalar_add(5.0) - 8.0).abs() < 1e-6);
        assert!((3.0f32.scalar_mul(5.0) - 15.0).abs() < 1e-6);
        assert!(f32::pos_infinity().is_infinite() && f32::pos_infinity() > 0.0);
        assert!(f32::neg_infinity().is_infinite() && f32::neg_infinity() < 0.0);
        assert!((3.0f32.scalar_max(5.0) - 5.0).abs() < 1e-6);
        assert!((3.0f32.scalar_min(5.0) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_i32_scalar() {
        assert_eq!(i32::scalar_zero(), 0);
        assert_eq!(i32::scalar_one(), 1);
        assert_eq!(3i32.scalar_add(5), 8);
        assert_eq!(3i32.scalar_mul(5), 15);
        // Headroom sentinel (not MIN/MAX): guard-free + stays in range, drift is
        // detectable by threshold, and it matches the CUDA backend.
        assert_eq!(i32::pos_infinity(), 1_000_000_000);
        assert_eq!(i32::neg_infinity(), -1_000_000_000);
        assert_eq!(3i32.scalar_max(5), 5);
        assert_eq!(3i32.scalar_min(5), 3);
    }

    #[test]
    fn test_i64_scalar() {
        assert_eq!(i64::scalar_zero(), 0);
        assert_eq!(i64::scalar_one(), 1);
        assert_eq!(3i64.scalar_add(5), 8);
        assert_eq!(3i64.scalar_mul(5), 15);
        assert_eq!(i64::pos_infinity(), 1i64 << 60);
        assert_eq!(i64::neg_infinity(), -(1i64 << 60));
        assert_eq!(3i64.scalar_max(5), 5);
        assert_eq!(3i64.scalar_min(5), 3);
    }

    #[test]
    fn test_i8_scalar() {
        assert_eq!(i8::scalar_zero(), 0);
        assert_eq!(i8::scalar_one(), 1);
        assert_eq!(3i8.scalar_add(5), 8);
        assert_eq!(3i8.scalar_mul(5), 15);
        assert_eq!(i8::pos_infinity(), i8::MAX);
        assert_eq!(i8::neg_infinity(), i8::MIN);
        assert_eq!(3i8.scalar_max(5), 5);
        assert_eq!(3i8.scalar_min(5), 3);
    }

    #[test]
    fn test_i16_scalar() {
        assert_eq!(i16::scalar_zero(), 0);
        assert_eq!(i16::scalar_one(), 1);
        assert_eq!(3i16.scalar_add(5), 8);
        assert_eq!(3i16.scalar_mul(5), 15);
        assert_eq!(i16::pos_infinity(), i16::MAX);
        assert_eq!(i16::neg_infinity(), i16::MIN);
        assert_eq!(3i16.scalar_max(5), 5);
        assert_eq!(3i16.scalar_min(5), 3);
    }

    #[test]
    fn test_u8_scalar() {
        assert_eq!(u8::scalar_zero(), 0);
        assert_eq!(u8::scalar_one(), 1);
        assert_eq!(3u8.scalar_add(5), 8);
        assert_eq!(3u8.scalar_mul(5), 15);
        assert_eq!(u8::pos_infinity(), u8::MAX);
        assert_eq!(u8::neg_infinity(), u8::MIN);
        assert_eq!(3u8.scalar_max(5), 5);
        assert_eq!(3u8.scalar_min(5), 3);
    }

    #[test]
    fn test_u16_scalar() {
        assert_eq!(u16::scalar_zero(), 0);
        assert_eq!(u16::scalar_one(), 1);
        assert_eq!(3u16.scalar_add(5), 8);
        assert_eq!(3u16.scalar_mul(5), 15);
        assert_eq!(u16::pos_infinity(), u16::MAX);
        assert_eq!(u16::neg_infinity(), u16::MIN);
        assert_eq!(3u16.scalar_max(5), 5);
        assert_eq!(3u16.scalar_min(5), 3);
    }

    #[test]
    fn test_u32_scalar() {
        assert_eq!(u32::scalar_zero(), 0);
        assert_eq!(u32::scalar_one(), 1);
        assert_eq!(3u32.scalar_add(5), 8);
        assert_eq!(3u32.scalar_mul(5), 15);
        assert_eq!(u32::pos_infinity(), u32::MAX);
        assert_eq!(u32::neg_infinity(), u32::MIN);
        assert_eq!(3u32.scalar_max(5), 5);
        assert_eq!(3u32.scalar_min(5), 3);
    }

    #[test]
    fn test_u64_scalar() {
        assert_eq!(u64::scalar_zero(), 0);
        assert_eq!(u64::scalar_one(), 1);
        assert_eq!(3u64.scalar_add(5), 8);
        assert_eq!(3u64.scalar_mul(5), 15);
        assert_eq!(u64::pos_infinity(), u64::MAX);
        assert_eq!(u64::neg_infinity(), u64::MIN);
        assert_eq!(3u64.scalar_max(5), 5);
        assert_eq!(3u64.scalar_min(5), 3);
    }

    #[test]
    fn test_bool_scalar() {
        assert!(!bool::scalar_zero());
        assert!(bool::scalar_one());
        // scalar_add is OR
        assert!(true.scalar_add(false));
        assert!(false.scalar_add(true));
        assert!(!false.scalar_add(false));
        assert!(true.scalar_add(true));
        // scalar_mul is AND
        assert!(!true.scalar_mul(false));
        assert!(!false.scalar_mul(true));
        assert!(!false.scalar_mul(false));
        assert!(true.scalar_mul(true));
        // pos_infinity is true, neg_infinity is false
        assert!(bool::pos_infinity());
        assert!(!bool::neg_infinity());
        // scalar_max is OR
        assert!(true.scalar_max(false));
        assert!(!false.scalar_max(false));
        // scalar_min is AND
        assert!(!true.scalar_min(false));
        assert!(true.scalar_min(true));
    }

    #[test]
    fn test_float_edge_cases() {
        // Test max/min with equal values
        assert_eq!(5.0f64.scalar_max(5.0), 5.0);
        assert_eq!(5.0f64.scalar_min(5.0), 5.0);
        assert_eq!(5.0f32.scalar_max(5.0), 5.0);
        assert_eq!(5.0f32.scalar_min(5.0), 5.0);
    }

    #[test]
    fn test_int_edge_cases() {
        // Test max/min with equal values
        assert_eq!(5i32.scalar_max(5), 5);
        assert_eq!(5i32.scalar_min(5), 5);
        // Test with negative numbers
        assert_eq!((-3i32).scalar_max(-5), -3);
        assert_eq!((-3i32).scalar_min(-5), -5);
    }

    #[test]
    fn test_drifted_zero_detection() {
        // The sentinel itself and anything in "infinity territory" (past S/2) is
        // a drifted tropical zero; realistic data and the multiplicative one are
        // not. Threshold = ±5e8 for i32, ±2^59 for i64.
        assert!(i32::neg_infinity().is_drifted_neg_zero());
        assert!((i32::neg_infinity() + 1000).is_drifted_neg_zero()); // drifted, still in territory
        assert!(i32::pos_infinity().is_drifted_pos_zero());
        assert!((i32::pos_infinity() - 1000).is_drifted_pos_zero());
        // Realistic values are never mistaken for the zero.
        assert!(!0i32.is_drifted_neg_zero());
        assert!(!0i32.is_drifted_pos_zero());
        assert!(!123_456i32.is_drifted_neg_zero());
        assert!(!(-123_456i32).is_drifted_neg_zero());

        assert!(i64::neg_infinity().is_drifted_neg_zero());
        assert!(i64::pos_infinity().is_drifted_pos_zero());
        assert!(!0i64.is_drifted_neg_zero());
        assert!(!1_000_000_000_000i64.is_drifted_neg_zero());

        // Floats never "drift" — exact ±∞, so the hook stays false (default).
        assert!(!f64::neg_infinity().is_drifted_neg_zero());
        assert!(!f64::pos_infinity().is_drifted_pos_zero());

        // Narrow / unsigned ints keep MIN/MAX and the default false hook.
        assert!(!i8::neg_infinity().is_drifted_neg_zero());
        assert!(!u32::pos_infinity().is_drifted_pos_zero());
    }
}
