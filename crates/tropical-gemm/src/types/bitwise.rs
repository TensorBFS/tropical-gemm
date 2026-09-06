use super::scalar::TropicalScalar;
use super::traits::{SimdTropical, TropicalSemiring};
use std::fmt;
use std::ops::{Add, BitAnd, BitOr, Mul};

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// Unsigned-integer element types valid as a `TropicalBitwise` lane container.
///
/// Sealed: only `u32` (32 lanes) and `u64` (64 lanes) are permitted.
pub trait BitwiseScalar:
    TropicalScalar + sealed::Sealed + BitOr<Output = Self> + BitAnd<Output = Self>
{
    /// All lanes false (tropical zero).
    const ZERO: Self;
    /// All lanes true (tropical one), i.e. `!0`.
    const ONES: Self;
}

impl BitwiseScalar for u32 {
    const ZERO: u32 = 0;
    const ONES: u32 = u32::MAX;
}

impl BitwiseScalar for u64 {
    const ZERO: u64 = 0;
    const ONES: u64 = u64::MAX;
}

/// TropicalBitwise semiring: `(uint, |, &, 0, ~0)` — bit-packed boolean.
///
/// Each bit-lane of the wrapped word is an **independent** boolean problem
/// (bit-slicing): one GEMM computes 32 (`u32`) or 64 (`u64`) boolean matmuls at
/// once. `⊕ = |`, `⊗ = &`, zero = `0`, one = `!0`.
///
/// This is for **many independent dense boolean problems**. For a single large
/// (sparse) boolean graph, use a sparse GraphBLAS tool (GraphBLAST / cuBool /
/// Bit-GraphBLAS) — that is out of scope for this dense library.
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct TropicalBitwise<T: BitwiseScalar>(pub T);

impl<T: BitwiseScalar> TropicalBitwise<T> {
    /// Create a new TropicalBitwise value from a packed word.
    #[inline(always)]
    pub fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: BitwiseScalar> TropicalSemiring for TropicalBitwise<T> {
    type Scalar = T;

    #[inline(always)]
    fn tropical_zero() -> Self {
        Self(T::ZERO)
    }

    #[inline(always)]
    fn tropical_one() -> Self {
        Self(T::ONES)
    }

    #[inline(always)]
    fn tropical_add(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }

    #[inline(always)]
    fn tropical_mul(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
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

impl<T: BitwiseScalar> SimdTropical for TropicalBitwise<T> {
    // No hand-written bitwise microkernel yet; the portable path handles |/&.
    const SIMD_AVAILABLE: bool = false;
    const SIMD_WIDTH: usize = 0;
}

impl<T: BitwiseScalar> Add for TropicalBitwise<T> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.tropical_add(rhs)
    }
}

impl<T: BitwiseScalar> Mul for TropicalBitwise<T> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.tropical_mul(rhs)
    }
}

impl<T: BitwiseScalar> Default for TropicalBitwise<T> {
    #[inline(always)]
    fn default() -> Self {
        Self::tropical_zero()
    }
}

impl<T: BitwiseScalar> fmt::Debug for TropicalBitwise<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TropicalBitwise({})", self.0)
    }
}

impl<T: BitwiseScalar> fmt::Display for TropicalBitwise<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: BitwiseScalar> From<T> for TropicalBitwise<T> {
    #[inline(always)]
    fn from(value: T) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn additive_identity() {
        let a = TropicalBitwise::<u32>(0b1011);
        assert_eq!(a.tropical_add(TropicalBitwise::tropical_zero()), a);
    }

    #[test]
    fn multiplicative_identity() {
        let a = TropicalBitwise::<u32>(0b1011);
        assert_eq!(a.tropical_mul(TropicalBitwise::tropical_one()), a);
    }

    #[test]
    fn absorbing_zero() {
        let a = TropicalBitwise::<u32>(0xDEADBEEF);
        assert_eq!(
            a.tropical_mul(TropicalBitwise::tropical_zero()),
            TropicalBitwise::tropical_zero()
        );
    }

    #[test]
    fn ops_are_bitwise() {
        let a = TropicalBitwise::<u64>(0b1100);
        let b = TropicalBitwise::<u64>(0b1010);
        assert_eq!(a.tropical_add(b).0, 0b1110); // OR
        assert_eq!(a.tropical_mul(b).0, 0b1000); // AND
    }

    #[test]
    fn zero_and_one_values() {
        assert_eq!(TropicalBitwise::<u32>::tropical_zero().0, 0u32);
        assert_eq!(TropicalBitwise::<u32>::tropical_one().0, u32::MAX);
        assert_eq!(TropicalBitwise::<u64>::tropical_one().0, u64::MAX);
    }

    // Bit-lane 0 of a TropicalBitwise<u32> GEMM must equal the same problem run
    // as TropicalAndOr (one lane is one AndOr problem). Uses the public matmul
    // API, which requires KernelDispatch (added in this task).
    #[test]
    fn lane0_matches_andor() {
        use crate::types::TropicalAndOr;
        use crate::tropical_matmul;

        // 2x3 * 3x2 column-major boolean problem (same as the GPU AndOr test).
        let a_bool = [true, false, false, false, true, false];
        let b_bool = [true, true, false, false, true, true];

        let a_u32: Vec<u32> = a_bool.iter().map(|&x| x as u32).collect();
        let b_u32: Vec<u32> = b_bool.iter().map(|&x| x as u32).collect();

        let c_bw = tropical_matmul::<TropicalBitwise<u32>>(&a_u32, 2, 3, &b_u32, 2);

        let mut c_ao = vec![TropicalAndOr(false); 4];
        // Safety: 2x3 * 3x2, all leading dims match, no aliasing.
        unsafe {
            crate::core::tropical_gemm_portable::<TropicalAndOr>(
                2, 2, 3,
                a_bool.as_ptr(), 2, crate::Transpose::NoTrans,
                b_bool.as_ptr(), 3, crate::Transpose::NoTrans,
                c_ao.as_mut_ptr(), 2,
            );
        }

        for i in 0..4 {
            let lane0 = (c_bw[i].0 & 1) == 1;
            assert_eq!(lane0, c_ao[i].0, "cell {i}: bitwise lane0 != andor");
        }
    }
}
