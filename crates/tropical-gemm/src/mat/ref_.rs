//! Immutable matrix reference type.

use std::marker::PhantomData;

use crate::core::Transpose;
use crate::simd::{tropical_gemm_dispatch, KernelDispatch};
use crate::types::{TropicalSemiring, TropicalWithArgmax};

use super::{Mat, MatWithArgmax};

/// Immutable view over scalar data interpreted as a tropical matrix.
///
/// This is a lightweight view type that can be copied freely.
/// It references scalar data and interprets operations using the
/// specified semiring type.
///
/// ```
/// use tropical_gemm::{MatRef, MaxPlus};
///
/// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let a = MatRef::<MaxPlus<f32>>::from_slice(&data, 2, 3);
///
/// assert_eq!(a.nrows(), 2);
/// assert_eq!(a.ncols(), 3);
/// assert_eq!(a.get(0, 0), 1.0);
/// ```
#[derive(Debug)]
pub struct MatRef<'a, S: TropicalSemiring> {
    data: &'a [S::Scalar],
    nrows: usize,
    ncols: usize,
    _phantom: PhantomData<S>,
}

impl<'a, S: TropicalSemiring> Copy for MatRef<'a, S> {}

impl<'a, S: TropicalSemiring> Clone for MatRef<'a, S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, S: TropicalSemiring> MatRef<'a, S> {
    /// Create a matrix reference from a slice of scalars.
    ///
    /// The data must be in column-major order with length `nrows * ncols`.
    pub fn from_slice(data: &'a [S::Scalar], nrows: usize, ncols: usize) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "data length {} != nrows {} * ncols {}",
            data.len(),
            nrows,
            ncols
        );
        Self {
            data,
            nrows,
            ncols,
            _phantom: PhantomData,
        }
    }

    /// Create a matrix reference from an owned Mat.
    ///
    /// This extracts the scalar values from the semiring wrapper.
    pub(crate) fn from_mat(mat: &'a Mat<S>) -> Self
    where
        S::Scalar: Copy,
    {
        // We need to get scalars from the Mat<S> which stores S values
        // Since S wraps Scalar, we can use value() to extract
        // But MatRef needs &[Scalar], not &[S]
        // This is a design tension - for now we'll use unsafe transmute
        // since S is repr(transparent) over Scalar
        //
        // Safety: TropicalMaxPlus<T>, TropicalMinPlus<T>, etc. are all
        // repr(transparent) newtype wrappers over T
        let scalar_slice = unsafe {
            std::slice::from_raw_parts(mat.data.as_ptr() as *const S::Scalar, mat.data.len())
        };
        Self {
            data: scalar_slice,
            nrows: mat.nrows,
            ncols: mat.ncols,
            _phantom: PhantomData,
        }
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

    /// Get the underlying scalar data.
    #[inline]
    pub fn as_slice(&self) -> &[S::Scalar] {
        self.data
    }

    /// Get the scalar value at position (i, j).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> S::Scalar
    where
        S::Scalar: Copy,
    {
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
        // Column-major indexing
        self.data[j * self.nrows + i]
    }

    /// Convert to an owned matrix.
    pub fn to_owned(&self) -> Mat<S>
    where
        S::Scalar: Copy,
    {
        Mat::from_col_major(self.data, self.nrows, self.ncols)
    }
}

// Matrix multiplication methods
impl<'a, S: TropicalSemiring + KernelDispatch> MatRef<'a, S> {
    /// Perform tropical matrix multiplication: C = A ⊗ B.
    ///
    /// Computes C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])
    ///
    /// # Panics
    ///
    /// Panics if dimensions don't match (self.ncols != b.nrows).
    pub fn matmul(&self, b: &MatRef<S>) -> Mat<S> {
        assert_eq!(
            self.ncols, b.nrows,
            "dimension mismatch: A is {}x{}, B is {}x{}",
            self.nrows, self.ncols, b.nrows, b.ncols
        );

        let m = self.nrows;
        let n = b.ncols;
        let k = self.ncols;

        let mut c = Mat::<S>::zeros(m, n);

        // Transpose trick for column-major: C = A * B becomes C^T = B^T * A^T
        unsafe {
            tropical_gemm_dispatch::<S>(
                n,
                m,
                k,
                b.data.as_ptr(),
                k,
                Transpose::NoTrans,
                self.data.as_ptr(),
                m,
                Transpose::NoTrans,
                c.data.as_mut_ptr(),
                m,
            );
        }

        c
    }
}

// Argmax methods (separate impl block for different trait bounds)
impl<'a, S> MatRef<'a, S>
where
    S: TropicalWithArgmax<Index = u32> + KernelDispatch,
{
    /// Perform tropical matrix multiplication with argmax tracking.
    ///
    /// Returns both the result matrix and the argmax indices indicating
    /// which k-index produced each optimal value.
    ///
    /// # Panics
    ///
    /// Panics if dimensions don't match (self.ncols != b.nrows).
    pub fn matmul_argmax(&self, b: &MatRef<S>) -> MatWithArgmax<S> {
        assert_eq!(
            self.ncols, b.nrows,
            "dimension mismatch: A is {}x{}, B is {}x{}",
            self.nrows, self.ncols, b.nrows, b.ncols
        );

        let m = self.nrows;
        let n = b.ncols;
        let k = self.ncols;

        // Transpose trick for column-major: output (n×m) row-major = (m×n) col-major
        let mut result = crate::core::GemmWithArgmax::<S>::new(n, m);

        unsafe {
            crate::core::tropical_gemm_with_argmax_portable::<S>(
                n,
                m,
                k,
                b.data.as_ptr(),
                k,
                Transpose::NoTrans,
                self.data.as_ptr(),
                m,
                Transpose::NoTrans,
                &mut result,
            );
        }

        MatWithArgmax {
            values: Mat {
                data: result.values,
                nrows: m,
                ncols: n,
            },
            argmax: result.argmax,
        }
    }
}
