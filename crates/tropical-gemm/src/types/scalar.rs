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

macro_rules! impl_tropical_scalar_int {
    ($($t:ty),*) => {
        $(
            impl TropicalScalar for $t {
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
                fn pos_infinity() -> Self {
                    <$t>::MAX
                }

                #[inline(always)]
                fn neg_infinity() -> Self {
                    <$t>::MIN
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

impl_tropical_scalar_float!(f32, f64);
impl_tropical_scalar_int!(i32, i64, i8, i16, u8, u16, u32, u64);

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
        assert_eq!(i32::pos_infinity(), i32::MAX);
        assert_eq!(i32::neg_infinity(), i32::MIN);
        assert_eq!(3i32.scalar_max(5), 5);
        assert_eq!(3i32.scalar_min(5), 3);
    }

    #[test]
    fn test_i64_scalar() {
        assert_eq!(i64::scalar_zero(), 0);
        assert_eq!(i64::scalar_one(), 1);
        assert_eq!(3i64.scalar_add(5), 8);
        assert_eq!(3i64.scalar_mul(5), 15);
        assert_eq!(i64::pos_infinity(), i64::MAX);
        assert_eq!(i64::neg_infinity(), i64::MIN);
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
}
