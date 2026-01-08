//! Owned matrix type.

use std::ops::{Index, IndexMut};

use crate::core::Transpose;
use crate::simd::{tropical_gemm_dispatch, KernelDispatch};
use crate::types::{TropicalSemiring, TropicalWithArgmax};

use super::{MatRef, MatWithArgmax};

/// Owned matrix storing semiring values.
///
/// The matrix stores values in row-major order. Use factory methods
/// to create matrices:
///
/// ```
/// use tropical_gemm::{Mat, MaxPlus, TropicalSemiring};
///
/// let zeros = Mat::<MaxPlus<f32>>::zeros(3, 4);
/// let identity = Mat::<MaxPlus<f32>>::identity(3);
/// let custom = Mat::<MaxPlus<f32>>::from_fn(2, 2, |i, j| {
///     MaxPlus::<f32>::from_scalar((i + j) as f32)
/// });
/// ```
#[derive(Debug, Clone)]
pub struct Mat<S: TropicalSemiring> {
    pub(crate) data: Vec<S>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

impl<S: TropicalSemiring> Mat<S> {
    /// Create a matrix filled with tropical zeros.
    ///
    /// For MaxPlus, this fills with -∞.
    /// For MinPlus, this fills with +∞.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            data: vec![S::tropical_zero(); nrows * ncols],
            nrows,
            ncols,
        }
    }

    /// Create a tropical identity matrix.
    ///
    /// Diagonal elements are tropical one (0 for MaxPlus/MinPlus).
    /// Off-diagonal elements are tropical zero (-∞ for MaxPlus, +∞ for MinPlus).
    pub fn identity(n: usize) -> Self {
        let mut mat = Self::zeros(n, n);
        for i in 0..n {
            mat.data[i * n + i] = S::tropical_one();
        }
        mat
    }

    /// Create a matrix from a function.
    ///
    /// The function is called with (row, col) indices.
    pub fn from_fn<F>(nrows: usize, ncols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> S,
    {
        let data = (0..nrows * ncols)
            .map(|idx| f(idx / ncols, idx % ncols))
            .collect();
        Self { data, nrows, ncols }
    }

    /// Create a matrix from row-major scalar data.
    ///
    /// Each scalar is wrapped in the semiring type.
    pub fn from_row_major(data: &[S::Scalar], nrows: usize, ncols: usize) -> Self
    where
        S::Scalar: Copy,
    {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "data length {} != nrows {} * ncols {}",
            data.len(),
            nrows,
            ncols
        );
        let data = data.iter().map(|&s| S::from_scalar(s)).collect();
        Self { data, nrows, ncols }
    }

    /// Create a matrix from a vector of semiring values.
    pub fn from_vec(data: Vec<S>, nrows: usize, ncols: usize) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "data length {} != nrows {} * ncols {}",
            data.len(),
            nrows,
            ncols
        );
        Self { data, nrows, ncols }
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Get the underlying data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        &self.data
    }

    /// Get the underlying data as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        &mut self.data
    }

    /// Get the scalar value at position (i, j).
    ///
    /// This is a convenience method that extracts the underlying scalar
    /// without requiring a trait import.
    ///
    /// # Example
    ///
    /// ```
    /// use tropical_gemm::{Mat, MaxPlus};
    ///
    /// let m = Mat::<MaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    /// assert_eq!(m.get_value(0, 0), 1.0);
    /// assert_eq!(m.get_value(1, 1), 4.0);
    /// ```
    #[inline]
    pub fn get_value(&self, i: usize, j: usize) -> S::Scalar {
        self[(i, j)].value()
    }

    /// Convert to an immutable matrix reference.
    ///
    /// The returned reference views the scalar values.
    pub fn as_ref(&self) -> MatRef<'_, S>
    where
        S::Scalar: Copy,
    {
        // Extract scalars from semiring values
        // This requires that the data is laid out such that we can get scalars
        // For now, we create a view that extracts values on-the-fly
        // This is a limitation - ideally we'd have a separate scalar buffer
        MatRef::from_mat(self)
    }

    /// Get a mutable pointer to the data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        self.data.as_mut_ptr()
    }
}

impl<S: TropicalSemiring> Index<(usize, usize)> for Mat<S> {
    type Output = S;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &S {
        debug_assert!(
            i < self.nrows,
            "row index {} out of bounds {}",
            i,
            self.nrows
        );
        debug_assert!(
            j < self.ncols,
            "col index {} out of bounds {}",
            j,
            self.ncols
        );
        &self.data[i * self.ncols + j]
    }
}

impl<S: TropicalSemiring> IndexMut<(usize, usize)> for Mat<S> {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut S {
        debug_assert!(
            i < self.nrows,
            "row index {} out of bounds {}",
            i,
            self.nrows
        );
        debug_assert!(
            j < self.ncols,
            "col index {} out of bounds {}",
            j,
            self.ncols
        );
        &mut self.data[i * self.ncols + j]
    }
}

// Matrix multiplication methods directly on Mat
impl<S> Mat<S>
where
    S: TropicalSemiring + KernelDispatch,
    S::Scalar: Copy,
{
    /// Perform tropical matrix multiplication: C = A ⊗ B.
    ///
    /// Computes C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])
    ///
    /// # Panics
    ///
    /// Panics if dimensions don't match (self.ncols != b.nrows).
    ///
    /// # Example
    ///
    /// ```
    /// use tropical_gemm::{Mat, MaxPlus, TropicalSemiring};
    ///
    /// let a = Mat::<MaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    /// let b = Mat::<MaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    ///
    /// let c = a.matmul(&b);
    ///
    /// // C[0,0] = max(1+1, 2+3, 3+5) = 8
    /// assert_eq!(c[(0, 0)].value(), 8.0);
    /// ```
    pub fn matmul(&self, b: &Mat<S>) -> Mat<S> {
        assert_eq!(
            self.ncols, b.nrows,
            "dimension mismatch: A is {}x{}, B is {}x{}",
            self.nrows, self.ncols, b.nrows, b.ncols
        );

        let a_ref = self.as_ref();
        let b_ref = b.as_ref();

        let mut c = Mat::<S>::zeros(self.nrows, b.ncols);

        unsafe {
            tropical_gemm_dispatch::<S>(
                self.nrows,
                b.ncols,
                self.ncols,
                a_ref.as_slice().as_ptr(),
                self.ncols,
                Transpose::NoTrans,
                b_ref.as_slice().as_ptr(),
                b.ncols,
                Transpose::NoTrans,
                c.data.as_mut_ptr(),
                b.ncols,
            );
        }

        c
    }

    /// Perform tropical matrix multiplication with a MatRef.
    ///
    /// This allows mixing owned and reference matrices.
    pub fn matmul_ref(&self, b: &MatRef<S>) -> Mat<S> {
        assert_eq!(
            self.ncols,
            b.nrows(),
            "dimension mismatch: A is {}x{}, B is {}x{}",
            self.nrows,
            self.ncols,
            b.nrows(),
            b.ncols()
        );

        let a_ref = self.as_ref();

        let mut c = Mat::<S>::zeros(self.nrows, b.ncols());

        unsafe {
            tropical_gemm_dispatch::<S>(
                self.nrows,
                b.ncols(),
                self.ncols,
                a_ref.as_slice().as_ptr(),
                self.ncols,
                Transpose::NoTrans,
                b.as_slice().as_ptr(),
                b.ncols(),
                Transpose::NoTrans,
                c.data.as_mut_ptr(),
                b.ncols(),
            );
        }

        c
    }
}

// Argmax methods on Mat
impl<S> Mat<S>
where
    S: TropicalWithArgmax<Index = u32> + KernelDispatch,
    S::Scalar: Copy,
{
    /// Perform tropical matrix multiplication with argmax tracking.
    ///
    /// Returns both the result matrix and the argmax indices indicating
    /// which k-index produced each optimal value.
    ///
    /// # Example
    ///
    /// ```
    /// use tropical_gemm::{Mat, MaxPlus, TropicalSemiring};
    ///
    /// let a = Mat::<MaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    /// let b = Mat::<MaxPlus<f64>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    ///
    /// let result = a.matmul_argmax(&b);
    ///
    /// assert_eq!(result.get(0, 0).value(), 8.0);
    /// assert_eq!(result.get_argmax(0, 0), 2); // k=2 gave max
    /// ```
    pub fn matmul_argmax(&self, b: &Mat<S>) -> MatWithArgmax<S> {
        assert_eq!(
            self.ncols, b.nrows,
            "dimension mismatch: A is {}x{}, B is {}x{}",
            self.nrows, self.ncols, b.nrows, b.ncols
        );

        let a_ref = self.as_ref();
        let b_ref = b.as_ref();

        let m = self.nrows;
        let n = b.ncols;
        let k = self.ncols;

        let mut result = crate::core::GemmWithArgmax::<S>::new(m, n);

        unsafe {
            crate::core::tropical_gemm_with_argmax_portable::<S>(
                m,
                n,
                k,
                a_ref.as_slice().as_ptr(),
                self.ncols,
                Transpose::NoTrans,
                b_ref.as_slice().as_ptr(),
                b.ncols,
                Transpose::NoTrans,
                &mut result,
            );
        }

        MatWithArgmax {
            values: Mat::from_vec(result.values, m, n),
            argmax: result.argmax,
        }
    }

    /// Batched tropical matrix multiplication with argmax tracking.
    ///
    /// Computes C[i] = A[i] ⊗ B[i] for each pair of matrices in the batch,
    /// tracking which k-index produced each optimal value.
    ///
    /// All matrices in `a_batch` must have the same dimensions, and all
    /// matrices in `b_batch` must have the same dimensions.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `a_batch` and `b_batch` have different lengths
    /// - Matrices in `a_batch` have different dimensions
    /// - Matrices in `b_batch` have different dimensions
    /// - Inner dimensions don't match (A.ncols != B.nrows)
    ///
    /// # Example
    ///
    /// ```
    /// use tropical_gemm::{Mat, MaxPlus};
    ///
    /// let a1 = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    /// let a2 = Mat::<MaxPlus<f32>>::from_row_major(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    /// let b1 = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 0.0, 0.0, 1.0], 2, 2);
    /// let b2 = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    ///
    /// let results = Mat::matmul_batched_with_argmax(&[a1, a2], &[b1, b2]);
    /// assert_eq!(results.len(), 2);
    /// ```
    pub fn matmul_batched_with_argmax(
        a_batch: &[Mat<S>],
        b_batch: &[Mat<S>],
    ) -> Vec<MatWithArgmax<S>> {
        assert_eq!(
            a_batch.len(),
            b_batch.len(),
            "batch sizes must match: {} != {}",
            a_batch.len(),
            b_batch.len()
        );

        if a_batch.is_empty() {
            return Vec::new();
        }

        // Validate dimensions
        let (m, k) = (a_batch[0].nrows, a_batch[0].ncols);
        let n = b_batch[0].ncols;

        for (i, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
            assert_eq!(
                (a.nrows, a.ncols),
                (m, k),
                "A[{}] has dimensions {}x{}, expected {}x{}",
                i,
                a.nrows,
                a.ncols,
                m,
                k
            );
            assert_eq!(
                (b.nrows, b.ncols),
                (k, n),
                "B[{}] has dimensions {}x{}, expected {}x{}",
                i,
                b.nrows,
                b.ncols,
                k,
                n
            );
        }

        a_batch
            .iter()
            .zip(b_batch.iter())
            .map(|(a, b)| a.matmul_argmax(b))
            .collect()
    }
}

// Batched operations on Mat
impl<S> Mat<S>
where
    S: TropicalSemiring + KernelDispatch,
    S::Scalar: Copy,
{
    /// Batched tropical matrix multiplication.
    ///
    /// Computes C[i] = A[i] ⊗ B[i] for each pair of matrices in the batch.
    /// All matrices in `a_batch` must have the same dimensions, and all
    /// matrices in `b_batch` must have the same dimensions.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `a_batch` and `b_batch` have different lengths
    /// - Matrices in `a_batch` have different dimensions
    /// - Matrices in `b_batch` have different dimensions
    /// - Inner dimensions don't match (A.ncols != B.nrows)
    ///
    /// # Example
    ///
    /// ```
    /// use tropical_gemm::{Mat, MaxPlus};
    ///
    /// let a1 = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    /// let a2 = Mat::<MaxPlus<f32>>::from_row_major(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    /// let b1 = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 0.0, 0.0, 1.0], 2, 2);
    /// let b2 = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    ///
    /// let results = Mat::matmul_batched(&[a1, a2], &[b1, b2]);
    /// assert_eq!(results.len(), 2);
    /// ```
    pub fn matmul_batched(a_batch: &[Mat<S>], b_batch: &[Mat<S>]) -> Vec<Mat<S>> {
        assert_eq!(
            a_batch.len(),
            b_batch.len(),
            "batch sizes must match: {} != {}",
            a_batch.len(),
            b_batch.len()
        );

        if a_batch.is_empty() {
            return Vec::new();
        }

        // Validate dimensions
        let (m, k) = (a_batch[0].nrows, a_batch[0].ncols);
        let n = b_batch[0].ncols;

        for (i, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
            assert_eq!(
                (a.nrows, a.ncols),
                (m, k),
                "A[{}] has dimensions {}x{}, expected {}x{}",
                i,
                a.nrows,
                a.ncols,
                m,
                k
            );
            assert_eq!(
                (b.nrows, b.ncols),
                (k, n),
                "B[{}] has dimensions {}x{}, expected {}x{}",
                i,
                b.nrows,
                b.ncols,
                k,
                n
            );
        }

        a_batch
            .iter()
            .zip(b_batch.iter())
            .map(|(a, b)| a.matmul(b))
            .collect()
    }
}
