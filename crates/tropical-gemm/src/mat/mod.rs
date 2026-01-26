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
        // Column-major indexing
        self.argmax[j * self.values.nrows() + i]
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.values.nrows()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.values.ncols()
    }

    /// Get the argmax indices as a slice.
    ///
    /// This is useful for backward pass computation.
    #[inline]
    pub fn argmax_slice(&self) -> &[u32] {
        &self.argmax
    }

    /// Compute gradient with respect to matrix A.
    ///
    /// Given the upstream gradient dL/dC, computes dL/dA using the argmax
    /// indices from the forward pass.
    ///
    /// For C = A ⊗ B where C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j]):
    /// dL/dA[i,k] = Σ_j { dL/dC[i,j] if argmax[i,j] == k }
    ///
    /// # Arguments
    ///
    /// * `grad_c` - Gradient of the loss with respect to C, dimensions m×n
    /// * `k` - Number of columns in A (the inner dimension)
    ///
    /// # Returns
    ///
    /// Gradient of the loss with respect to A, dimensions m×k
    ///
    /// # Example
    ///
    /// ```
    /// use tropical_gemm::{Mat, MaxPlus, TropicalMaxPlus};
    ///
    /// let a = Mat::<MaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    /// let b = Mat::<MaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    ///
    /// // Forward pass with argmax
    /// let result = a.matmul_argmax(&b);
    ///
    /// // Backward pass: grad_c is upstream gradient (e.g., all ones)
    /// let grad_c = Mat::<MaxPlus<f64>>::from_fn(2, 2, |_, _| TropicalMaxPlus(1.0));
    /// let grad_a = result.backward_a(&grad_c, 3); // k=3 (columns in A)
    ///
    /// assert_eq!(grad_a.nrows(), 2);
    /// assert_eq!(grad_a.ncols(), 3);
    /// ```
    pub fn backward_a<G>(&self, grad_c: &Mat<G>, k: usize) -> Mat<G>
    where
        G: crate::TropicalSemiring,
        G::Scalar: Copy + Default + std::ops::AddAssign,
    {
        let m = self.nrows();
        let n = self.ncols();
        assert_eq!(grad_c.nrows(), m, "grad_c rows mismatch");
        assert_eq!(grad_c.ncols(), n, "grad_c cols mismatch");

        // Output is m×k in column-major
        let mut grad_a_data = vec![G::Scalar::default(); m * k];

        for j in 0..n {
            for i in 0..m {
                // Column-major indexing for argmax
                let idx = self.argmax[j * m + i] as usize;
                if idx < k {
                    // Column-major indexing for grad_a: element (i, idx) at idx * m + i
                    grad_a_data[idx * m + i] += grad_c[(i, j)].value();
                }
            }
        }

        Mat::from_col_major(&grad_a_data, m, k)
    }

    /// Compute gradient with respect to matrix B.
    ///
    /// Given the upstream gradient dL/dC, computes dL/dB using the argmax
    /// indices from the forward pass.
    ///
    /// For C = A ⊗ B where C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j]):
    /// dL/dB[k,j] = Σ_i { dL/dC[i,j] if argmax[i,j] == k }
    ///
    /// # Arguments
    ///
    /// * `grad_c` - Gradient of the loss with respect to C, dimensions m×n
    /// * `k` - Number of rows in B (the inner dimension)
    ///
    /// # Returns
    ///
    /// Gradient of the loss with respect to B, dimensions k×n
    ///
    /// # Example
    ///
    /// ```
    /// use tropical_gemm::{Mat, MaxPlus, TropicalMaxPlus};
    ///
    /// let a = Mat::<MaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    /// let b = Mat::<MaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    ///
    /// // Forward pass with argmax
    /// let result = a.matmul_argmax(&b);
    ///
    /// // Backward pass: grad_c is upstream gradient
    /// let grad_c = Mat::<MaxPlus<f64>>::from_fn(2, 2, |_, _| TropicalMaxPlus(1.0));
    /// let grad_b = result.backward_b(&grad_c, 3); // k=3 (rows in B)
    ///
    /// assert_eq!(grad_b.nrows(), 3);
    /// assert_eq!(grad_b.ncols(), 2);
    /// ```
    pub fn backward_b<G>(&self, grad_c: &Mat<G>, k: usize) -> Mat<G>
    where
        G: crate::TropicalSemiring,
        G::Scalar: Copy + Default + std::ops::AddAssign,
    {
        let m = self.nrows();
        let n = self.ncols();
        assert_eq!(grad_c.nrows(), m, "grad_c rows mismatch");
        assert_eq!(grad_c.ncols(), n, "grad_c cols mismatch");

        // Output is k×n in column-major
        let mut grad_b_data = vec![G::Scalar::default(); k * n];

        for j in 0..n {
            for i in 0..m {
                // Column-major indexing for argmax
                let idx = self.argmax[j * m + i] as usize;
                if idx < k {
                    // Column-major indexing for grad_b: element (idx, j) at j * k + idx
                    grad_b_data[j * k + idx] += grad_c[(i, j)].value();
                }
            }
        }

        Mat::from_col_major(&grad_b_data, k, n)
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
        // Column-major data: 2×3 matrix [[1,2,3],[4,5,6]] stored as [1,4,2,5,3,6]
        let data = [1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let m = MatRef::<TropicalMaxPlus<f64>>::from_slice(&data, 2, 3);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 6.0);
    }

    #[test]
    fn test_matmul() {
        // Column-major data:
        // A: 2×3 matrix [[1,2,3],[4,5,6]] stored as [1,4,2,5,3,6]
        // B: 3×2 matrix [[1,2],[3,4],[5,6]] stored as [1,3,5,2,4,6]
        let a_data = [1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b_data = [1.0f64, 3.0, 5.0, 2.0, 4.0, 6.0];

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
        // Column-major data
        let a_data = [1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b_data = [1.0f64, 3.0, 5.0, 2.0, 4.0, 6.0];

        let a = MatRef::<TropicalMaxPlus<f64>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<TropicalMaxPlus<f64>>::from_slice(&b_data, 3, 2);

        let c = &a * &b;

        assert_eq!(c[(0, 0)].0, 8.0);
        assert_eq!(c[(1, 1)].0, 12.0);
    }

    #[test]
    fn test_matmul_argmax() {
        // Column-major data
        let a_data = [1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b_data = [1.0f64, 3.0, 5.0, 2.0, 4.0, 6.0];

        let a = MatRef::<TropicalMaxPlus<f64>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<TropicalMaxPlus<f64>>::from_slice(&b_data, 3, 2);

        let result = a.matmul_argmax(&b);

        assert_eq!(result.get(0, 0).0, 8.0);
        assert_eq!(result.get_argmax(0, 0), 2); // k=2 gave max
    }

    #[test]
    fn test_minplus_matmul() {
        use crate::TropicalMinPlus;

        // Column-major data
        let a_data = [1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b_data = [1.0f64, 3.0, 5.0, 2.0, 4.0, 6.0];

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
        // Column-major data for B: 3×2 matrix [[1,2],[3,4],[5,6]] stored as [1,3,5,2,4,6]
        let b_data = [1.0f64, 3.0, 5.0, 2.0, 4.0, 6.0];
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
        let a2 =
            Mat::<TropicalMaxPlus<f32>>::from_row_major(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 2, 3); // Different size
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

    // ========================================================================
    // Backward pass tests
    // ========================================================================

    #[test]
    fn test_matwithargmax_backward_a() {
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        // Forward pass
        let result = a.matmul_argmax(&b);

        // All argmax should be 2 (k=2 wins for all)
        assert_eq!(result.get_argmax(0, 0), 2);
        assert_eq!(result.get_argmax(0, 1), 2);
        assert_eq!(result.get_argmax(1, 0), 2);
        assert_eq!(result.get_argmax(1, 1), 2);

        // Backward pass with unit gradients
        let grad_c = Mat::<TropicalMaxPlus<f64>>::from_fn(2, 2, |_, _| TropicalMaxPlus(1.0));
        let grad_a = result.backward_a(&grad_c, 3);

        // Only column 2 should have gradients
        assert_eq!(grad_a.nrows(), 2);
        assert_eq!(grad_a.ncols(), 3);
        assert_eq!(grad_a[(0, 0)].0, 0.0); // Not selected
        assert_eq!(grad_a[(0, 1)].0, 0.0); // Not selected
        assert_eq!(grad_a[(0, 2)].0, 2.0); // Selected for C[0,0] and C[0,1]
        assert_eq!(grad_a[(1, 0)].0, 0.0); // Not selected
        assert_eq!(grad_a[(1, 1)].0, 0.0); // Not selected
        assert_eq!(grad_a[(1, 2)].0, 2.0); // Selected for C[1,0] and C[1,1]
    }

    #[test]
    fn test_matwithargmax_backward_b() {
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        // Forward pass
        let result = a.matmul_argmax(&b);

        // Backward pass with unit gradients
        let grad_c = Mat::<TropicalMaxPlus<f64>>::from_fn(2, 2, |_, _| TropicalMaxPlus(1.0));
        let grad_b = result.backward_b(&grad_c, 3);

        // Only row 2 should have gradients
        assert_eq!(grad_b.nrows(), 3);
        assert_eq!(grad_b.ncols(), 2);
        assert_eq!(grad_b[(0, 0)].0, 0.0); // Not selected
        assert_eq!(grad_b[(0, 1)].0, 0.0); // Not selected
        assert_eq!(grad_b[(1, 0)].0, 0.0); // Not selected
        assert_eq!(grad_b[(1, 1)].0, 0.0); // Not selected
        assert_eq!(grad_b[(2, 0)].0, 2.0); // Selected for C[0,0] and C[1,0]
        assert_eq!(grad_b[(2, 1)].0, 2.0); // Selected for C[0,1] and C[1,1]
    }

    #[test]
    fn test_matwithargmax_backward_varied_argmax() {
        // Design matrices where different k-indices win
        let a =
            Mat::<TropicalMaxPlus<f64>>::from_row_major(&[10.0, 1.0, 1.0, 1.0, 10.0, 1.0], 2, 3);
        let b =
            Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 1.0, 1.0, 1.0, 10.0, 10.0], 3, 2);

        let result = a.matmul_argmax(&b);

        // Check argmax patterns
        // C[0,0] = max(10+1=11, 1+1=2, 1+10=11), first wins -> k=0
        // C[1,0] = max(1+1=2, 10+1=11, 1+10=11), second wins -> k=1
        assert_eq!(result.get_argmax(0, 0), 0);
        assert_eq!(result.get_argmax(1, 0), 1);

        let grad_c = Mat::<TropicalMaxPlus<f64>>::from_fn(2, 2, |_, _| TropicalMaxPlus(1.0));
        let grad_a = result.backward_a(&grad_c, 3);

        // grad_a[0,0] should get contributions from C[0,*] where argmax == 0
        // grad_a[1,1] should get contributions from C[1,*] where argmax == 1
        assert!(grad_a[(0, 0)].0 > 0.0); // k=0 selected for C[0,0] and C[0,1]
        assert!(grad_a[(1, 1)].0 > 0.0); // k=1 selected for C[1,0] and C[1,1]
    }

    #[test]
    fn test_matwithargmax_argmax_slice() {
        let a = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = Mat::<TropicalMaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        let result = a.matmul_argmax(&b);
        let argmax_slice = result.argmax_slice();

        assert_eq!(argmax_slice.len(), 4); // 2x2 output
        assert_eq!(argmax_slice[0], result.get_argmax(0, 0));
        assert_eq!(argmax_slice[1], result.get_argmax(0, 1));
        assert_eq!(argmax_slice[2], result.get_argmax(1, 0));
        assert_eq!(argmax_slice[3], result.get_argmax(1, 1));
    }
}
