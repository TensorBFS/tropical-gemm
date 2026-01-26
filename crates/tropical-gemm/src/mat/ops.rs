//! Operator implementations for matrix types.

use std::ops::Mul;

use crate::simd::KernelDispatch;
use crate::types::TropicalSemiring;

use super::{Mat, MatRef};

// MatRef * MatRef
impl<'a, 'b, S> Mul<&'b MatRef<'b, S>> for &'a MatRef<'a, S>
where
    S: TropicalSemiring + KernelDispatch,
{
    type Output = Mat<S>;

    fn mul(self, rhs: &'b MatRef<'b, S>) -> Mat<S> {
        self.matmul(rhs)
    }
}

// MatRef * MatRef (by value, since MatRef is Copy)
impl<'a, 'b, S> Mul<MatRef<'b, S>> for MatRef<'a, S>
where
    S: TropicalSemiring + KernelDispatch,
{
    type Output = Mat<S>;

    fn mul(self, rhs: MatRef<'b, S>) -> Mat<S> {
        self.matmul(&rhs)
    }
}

// &Mat * &MatRef
impl<'a, S> Mul<&'a MatRef<'a, S>> for &Mat<S>
where
    S: TropicalSemiring + KernelDispatch,
    S::Scalar: Copy,
{
    type Output = Mat<S>;

    fn mul(self, rhs: &'a MatRef<'a, S>) -> Mat<S> {
        self.as_ref().matmul(rhs)
    }
}

// &Mat * &Mat
impl<S> Mul<&Mat<S>> for &Mat<S>
where
    S: TropicalSemiring + KernelDispatch,
    S::Scalar: Copy,
{
    type Output = Mat<S>;

    fn mul(self, rhs: &Mat<S>) -> Mat<S> {
        self.as_ref().matmul(&rhs.as_ref())
    }
}

// Mat * Mat (consuming)
impl<S> Mul<Mat<S>> for Mat<S>
where
    S: TropicalSemiring + KernelDispatch,
    S::Scalar: Copy,
{
    type Output = Mat<S>;

    fn mul(self, rhs: Mat<S>) -> Mat<S> {
        self.as_ref().matmul(&rhs.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TropicalMaxPlus;

    #[test]
    fn test_matref_mul_matref() {
        // Column-major: 2×2 matrix [[1,2],[3,4]] stored as [1,3,2,4]
        let a_data = [1.0f64, 3.0, 2.0, 4.0];
        let b_data = [1.0f64, 3.0, 2.0, 4.0];

        let a = MatRef::<TropicalMaxPlus<f64>>::from_slice(&a_data, 2, 2);
        let b = MatRef::<TropicalMaxPlus<f64>>::from_slice(&b_data, 2, 2);

        let c = &a * &b;

        // C[0,0] = max(1+1, 2+3) = 5
        assert_eq!(c[(0, 0)].0, 5.0);
    }

    #[test]
    fn test_matref_mul_matref_by_value() {
        // Column-major: 2×2 matrix [[1,2],[3,4]] stored as [1,3,2,4]
        let a_data = [1.0f64, 3.0, 2.0, 4.0];
        let b_data = [1.0f64, 3.0, 2.0, 4.0];

        let a = MatRef::<TropicalMaxPlus<f64>>::from_slice(&a_data, 2, 2);
        let b = MatRef::<TropicalMaxPlus<f64>>::from_slice(&b_data, 2, 2);

        // MatRef is Copy, so this tests the by-value multiplication
        let c = a * b;

        assert_eq!(c[(0, 0)].0, 5.0);
    }

    #[test]
    fn test_mat_ref_mul_matref() {
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        // Column-major: 2×2 matrix [[1,2],[3,4]] stored as [1,3,2,4]
        let b_data = [1.0f64, 3.0, 2.0, 4.0];
        let b = MatRef::<TropicalMaxPlus<f64>>::from_slice(&b_data, 2, 2);

        let c = &a * &b;

        assert_eq!(c[(0, 0)].0, 5.0);
    }

    #[test]
    fn test_mat_mul_mat() {
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);

        let c = &a * &b;

        assert_eq!(c[(0, 0)].0, 5.0);
    }

    #[test]
    fn test_mat_mul_consuming() {
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);

        let c = a * b;

        assert_eq!(c[(0, 0)].0, 5.0);
    }
}
