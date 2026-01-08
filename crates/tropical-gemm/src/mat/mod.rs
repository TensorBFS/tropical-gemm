//! Matrix types for tropical algebra.
//!
//! This module provides faer-inspired matrix types:
//! - [`Mat<S>`]: Owned matrix storing semiring values
//! - [`MatRef<'a, S>`]: Immutable view over scalar data
//! - [`MatMut<'a, S>`]: Mutable view over semiring data
//!
//! # Example
//!
//! ```
//! use tropical_gemm::{Mat, MatRef, MaxPlus};
//!
//! // Create a view from raw data
//! let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let a = MatRef::<MaxPlus<f32>>::from_slice(&data, 2, 3);
//! let b = MatRef::<MaxPlus<f32>>::from_slice(&data, 3, 2);
//!
//! // Matrix multiplication using method
//! let c = a.matmul(&b);
//!
//! // Or using operator syntax
//! let c = &a * &b;
//!
//! // Factory methods
//! let zeros = Mat::<MaxPlus<f32>>::zeros(3, 3);
//! let identity = Mat::<MaxPlus<f32>>::identity(3);
//! ```

mod mut_;
mod ops;
mod owned;
mod ref_;

pub use mut_::MatMut;
pub use owned::Mat;
pub use ref_::MatRef;

/// Result of matrix multiplication with argmax tracking.
pub struct MatWithArgmax<S: crate::TropicalWithArgmax> {
    /// The result matrix values.
    pub values: Mat<S>,
    /// The argmax indices (which k produced each C[i,j]).
    pub argmax: Vec<u32>,
}

impl<S: crate::TropicalWithArgmax<Index = u32>> MatWithArgmax<S> {
    /// Get the value at position (i, j).
    pub fn get(&self, i: usize, j: usize) -> S {
        self.values[(i, j)]
    }

    /// Get the scalar value at position (i, j).
    ///
    /// This is a convenience method that extracts the underlying scalar
    /// without requiring a trait import.
    #[inline]
    pub fn get_value(&self, i: usize, j: usize) -> S::Scalar {
        self.values[(i, j)].value()
    }

    /// Get the argmax index at position (i, j).
    pub fn get_argmax(&self, i: usize, j: usize) -> u32 {
        self.argmax[i * self.values.ncols() + j]
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.values.nrows()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.values.ncols()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TropicalMaxPlus;

    #[test]
    fn test_mat_zeros() {
        let m = Mat::<TropicalMaxPlus<f64>>::zeros(3, 4);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 4);
        assert_eq!(m[(0, 0)].0, f64::NEG_INFINITY);
    }

    #[test]
    fn test_mat_identity() {
        let m = Mat::<TropicalMaxPlus<f64>>::identity(3);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m[(0, 0)].0, 0.0); // tropical one
        assert_eq!(m[(0, 1)].0, f64::NEG_INFINITY); // tropical zero
        assert_eq!(m[(1, 1)].0, 0.0);
        assert_eq!(m[(2, 2)].0, 0.0);
    }

    #[test]
    fn test_mat_from_fn() {
        let m =
            Mat::<TropicalMaxPlus<f64>>::from_fn(2, 3, |i, j| TropicalMaxPlus((i * 3 + j) as f64));
        assert_eq!(m[(0, 0)].0, 0.0);
        assert_eq!(m[(0, 2)].0, 2.0);
        assert_eq!(m[(1, 0)].0, 3.0);
        assert_eq!(m[(1, 2)].0, 5.0);
    }

    #[test]
    fn test_matref_from_slice() {
        let data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = MatRef::<TropicalMaxPlus<f64>>::from_slice(&data, 2, 3);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 6.0);
    }

    #[test]
    fn test_matmul() {
        let a_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = MatRef::<TropicalMaxPlus<f64>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<TropicalMaxPlus<f64>>::from_slice(&b_data, 3, 2);

        let c = a.matmul(&b);

        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert_eq!(c[(0, 0)].0, 8.0);
        // C[0,1] = max(1+2, 2+4, 3+6) = 9
        assert_eq!(c[(0, 1)].0, 9.0);
        // C[1,0] = max(4+1, 5+3, 6+5) = 11
        assert_eq!(c[(1, 0)].0, 11.0);
        // C[1,1] = max(4+2, 5+4, 6+6) = 12
        assert_eq!(c[(1, 1)].0, 12.0);
    }

    #[test]
    fn test_matmul_operator() {
        let a_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = MatRef::<TropicalMaxPlus<f64>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<TropicalMaxPlus<f64>>::from_slice(&b_data, 3, 2);

        let c = &a * &b;

        assert_eq!(c[(0, 0)].0, 8.0);
        assert_eq!(c[(1, 1)].0, 12.0);
    }

    #[test]
    fn test_matmul_argmax() {
        let a_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = MatRef::<TropicalMaxPlus<f64>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<TropicalMaxPlus<f64>>::from_slice(&b_data, 3, 2);

        let result = a.matmul_argmax(&b);

        assert_eq!(result.get(0, 0).0, 8.0);
        assert_eq!(result.get_argmax(0, 0), 2); // k=2 gave max
    }

    #[test]
    fn test_minplus_matmul() {
        use crate::TropicalMinPlus;

        let a_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = MatRef::<TropicalMinPlus<f64>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<TropicalMinPlus<f64>>::from_slice(&b_data, 3, 2);

        let c = a.matmul(&b);

        // C[0,0] = min(1+1, 2+3, 3+5) = 2
        assert_eq!(c[(0, 0)].0, 2.0);
        // C[1,1] = min(4+2, 5+4, 6+6) = 6
        assert_eq!(c[(1, 1)].0, 6.0);
    }

    #[test]
    fn test_mat_as_ref() {
        let m =
            Mat::<TropicalMaxPlus<f64>>::from_fn(2, 3, |i, j| TropicalMaxPlus((i * 3 + j) as f64));

        let r = m.as_ref();
        assert_eq!(r.nrows(), 2);
        assert_eq!(r.ncols(), 3);
        assert_eq!(r.get(0, 0), 0.0);
        assert_eq!(r.get(1, 2), 5.0);
    }

    #[test]
    fn test_mat_matmul_direct() {
        // Test Mat::matmul directly (no as_ref needed)
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        let c = a.matmul(&b);

        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert_eq!(c[(0, 0)].0, 8.0);
        // C[1,1] = max(4+2, 5+4, 6+6) = 12
        assert_eq!(c[(1, 1)].0, 12.0);
    }

    #[test]
    fn test_mat_matmul_argmax_direct() {
        // Test Mat::matmul_argmax directly
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        let result = a.matmul_argmax(&b);

        assert_eq!(result.get(0, 0).0, 8.0);
        assert_eq!(result.get_argmax(0, 0), 2); // k=2 gave max
    }

    #[test]
    fn test_mat_get_value() {
        // Test get_value method - no trait import needed
        let m = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);

        assert_eq!(m.get_value(0, 0), 1.0);
        assert_eq!(m.get_value(0, 1), 2.0);
        assert_eq!(m.get_value(1, 0), 3.0);
        assert_eq!(m.get_value(1, 1), 4.0);
    }

    #[test]
    fn test_minplus_mat_matmul_direct() {
        use crate::TropicalMinPlus;

        let a = Mat::<TropicalMinPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = Mat::<TropicalMinPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        let c = a.matmul(&b);

        // C[0,0] = min(1+1, 2+3, 3+5) = 2
        assert_eq!(c[(0, 0)].0, 2.0);
        // C[1,1] = min(4+2, 5+4, 6+6) = 6
        assert_eq!(c[(1, 1)].0, 6.0);
    }

    #[test]
    fn test_mat_from_vec() {
        let data = vec![
            TropicalMaxPlus(1.0f64),
            TropicalMaxPlus(2.0),
            TropicalMaxPlus(3.0),
            TropicalMaxPlus(4.0),
        ];
        let m = Mat::from_vec(data, 2, 2);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m[(0, 0)].0, 1.0);
        assert_eq!(m[(1, 1)].0, 4.0);
    }

    #[test]
    fn test_mat_as_slice() {
        let m = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let slice = m.as_slice();
        assert_eq!(slice.len(), 4);
        assert_eq!(slice[0].0, 1.0);
        assert_eq!(slice[3].0, 4.0);
    }

    #[test]
    fn test_mat_as_mut_slice() {
        let mut m = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let slice = m.as_mut_slice();
        slice[0] = TropicalMaxPlus(100.0);
        assert_eq!(m[(0, 0)].0, 100.0);
    }

    #[test]
    fn test_mat_as_mut_ptr() {
        let mut m = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let ptr = m.as_mut_ptr();
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_mat_index_mut() {
        let mut m = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        m[(0, 0)] = TropicalMaxPlus(10.0);
        m[(1, 1)] = TropicalMaxPlus(40.0);
        assert_eq!(m[(0, 0)].0, 10.0);
        assert_eq!(m[(1, 1)].0, 40.0);
    }

    #[test]
    fn test_mat_matmul_ref() {
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = MatRef::<TropicalMaxPlus<f64>>::from_slice(&b_data, 3, 2);

        let c = a.matmul_ref(&b);

        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert_eq!(c[(0, 0)].0, 8.0);
        // C[1,1] = max(4+2, 5+4, 6+6) = 12
        assert_eq!(c[(1, 1)].0, 12.0);
    }

    #[test]
    fn test_matref_copy_clone() {
        let data = [1.0f64, 2.0, 3.0, 4.0];
        let a = MatRef::<TropicalMaxPlus<f64>>::from_slice(&data, 2, 2);
        let b = a; // Copy
        let c = a.clone(); // Clone
        assert_eq!(a.get(0, 0), b.get(0, 0));
        assert_eq!(a.get(0, 0), c.get(0, 0));
    }

    #[test]
    fn test_matref_to_owned() {
        let data = [1.0f64, 2.0, 3.0, 4.0];
        let a = MatRef::<TropicalMaxPlus<f64>>::from_slice(&data, 2, 2);
        let owned = a.to_owned();
        assert_eq!(owned.nrows(), 2);
        assert_eq!(owned.ncols(), 2);
        assert_eq!(owned[(0, 0)].0, 1.0);
    }

    #[test]
    fn test_matref_debug() {
        let data = [1.0f64, 2.0];
        let m = MatRef::<TropicalMaxPlus<f64>>::from_slice(&data, 1, 2);
        let debug_str = format!("{:?}", m);
        assert!(debug_str.contains("MatRef"));
    }

    #[test]
    fn test_mat_clone() {
        let m = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let m2 = m.clone();
        assert_eq!(m2[(0, 0)].0, 1.0);
        assert_eq!(m2[(1, 1)].0, 4.0);
    }

    #[test]
    fn test_mat_debug() {
        let m = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0], 1, 2);
        let debug_str = format!("{:?}", m);
        assert!(debug_str.contains("Mat"));
    }

    #[test]
    fn test_matwithargmax_get_value() {
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        let result = a.matmul_argmax(&b);

        // Test get_value (scalar extraction without trait import)
        assert_eq!(result.get_value(0, 0), 8.0);
        assert_eq!(result.get_value(1, 1), 12.0);
    }

    #[test]
    fn test_matwithargmax_nrows_ncols() {
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        let result = a.matmul_argmax(&b);

        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_mat_from_row_major_size_mismatch() {
        let _ = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0], 2, 2);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_mat_from_vec_size_mismatch() {
        let data = vec![TropicalMaxPlus(1.0f64), TropicalMaxPlus(2.0)];
        let _ = Mat::from_vec(data, 2, 2);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_matref_from_slice_size_mismatch() {
        let data = [1.0f64, 2.0];
        let _ = MatRef::<TropicalMaxPlus<f64>>::from_slice(&data, 2, 2);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_matmul_dimension_mismatch() {
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let _ = a.matmul(&b); // Should panic: A is 2x2, B is 3x2
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_matref_matmul_dimension_mismatch() {
        let a_data = [1.0f64, 2.0, 3.0, 4.0];
        let b_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = MatRef::<TropicalMaxPlus<f64>>::from_slice(&a_data, 2, 2);
        let b = MatRef::<TropicalMaxPlus<f64>>::from_slice(&b_data, 3, 2);
        let _ = a.matmul(&b); // Should panic
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_matmul_argmax_dimension_mismatch() {
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let _ = a.matmul_argmax(&b); // Should panic
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_matref_matmul_argmax_dimension_mismatch() {
        let a_data = [1.0f64, 2.0, 3.0, 4.0];
        let b_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = MatRef::<TropicalMaxPlus<f64>>::from_slice(&a_data, 2, 2);
        let b = MatRef::<TropicalMaxPlus<f64>>::from_slice(&b_data, 3, 2);
        let _ = a.matmul_argmax(&b); // Should panic
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_mat_matmul_ref_dimension_mismatch() {
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = MatRef::<TropicalMaxPlus<f64>>::from_slice(&b_data, 3, 2);
        let _ = a.matmul_ref(&b); // Should panic
    }

    // ========================================================================
    // Batched operation tests
    // ========================================================================

    #[test]
    fn test_mat_matmul_batched() {
        let a1 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let a2 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let b1 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let b2 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);

        let results = Mat::matmul_batched(&[a1, a2], &[b1, b2]);
        assert_eq!(results.len(), 2);

        // C[0] = A[0] * B[0] (MaxPlus)
        // C[0,0] = max(1+1, 2+0) = 2
        assert!((results[0][(0, 0)].0 - 2.0).abs() < 1e-5);

        // C[1] = A[1] * B[1] (MaxPlus)
        // C[0,0] = max(5+1, 6+3) = 9
        assert!((results[1][(0, 0)].0 - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_mat_matmul_batched_empty() {
        let a_batch: Vec<Mat<TropicalMaxPlus<f32>>> = vec![];
        let b_batch: Vec<Mat<TropicalMaxPlus<f32>>> = vec![];

        let results = Mat::matmul_batched(&a_batch, &b_batch);
        assert!(results.is_empty());
    }

    #[test]
    #[should_panic(expected = "batch sizes must match")]
    fn test_mat_matmul_batched_size_mismatch() {
        let a1 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b1 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let b2 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);

        let _ = Mat::matmul_batched(&[a1], &[b1, b2]); // Should panic
    }

    #[test]
    #[should_panic(expected = "has dimensions")]
    fn test_mat_matmul_batched_dimension_mismatch() {
        let a1 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let a2 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 2, 3); // Different size
        let b1 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let b2 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);

        let _ = Mat::matmul_batched(&[a1, a2], &[b1, b2]); // Should panic
    }

    #[test]
    fn test_mat_matmul_batched_with_argmax() {
        let a1 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let a2 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[6.0, 5.0, 4.0, 3.0, 2.0, 1.0], 2, 3);
        let b1 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let b2 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        let results = Mat::matmul_batched_with_argmax(&[a1, a2], &[b1, b2]);
        assert_eq!(results.len(), 2);

        // C[0,0] = max(1+1, 2+3, 3+5) = 8, argmax=2
        assert!((results[0].get(0, 0).0 - 8.0).abs() < 1e-5);
        assert_eq!(results[0].get_argmax(0, 0), 2);
    }

    #[test]
    fn test_mat_matmul_batched_with_argmax_empty() {
        let a_batch: Vec<Mat<TropicalMaxPlus<f32>>> = vec![];
        let b_batch: Vec<Mat<TropicalMaxPlus<f32>>> = vec![];

        let results = Mat::matmul_batched_with_argmax(&a_batch, &b_batch);
        assert!(results.is_empty());
    }

    #[test]
    #[should_panic(expected = "batch sizes must match")]
    fn test_mat_matmul_batched_with_argmax_size_mismatch() {
        let a1 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b1 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let b2 = Mat::<TropicalMaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        let _ = Mat::matmul_batched_with_argmax(&[a1], &[b1, b2]); // Should panic
    }
}
