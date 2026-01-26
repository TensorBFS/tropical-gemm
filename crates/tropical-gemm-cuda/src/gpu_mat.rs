//! High-level GPU matrix types with semiring information.
//!
//! This module provides faer-style matrix types for GPU operations:
//! - [`GpuMat<S>`]: GPU matrix with embedded semiring type
//! - [`GpuMatWithArgmax<S>`]: GPU matrix with argmax tracking
//!
//! # Example
//!
//! ```ignore
//! use tropical_gemm::{Mat, MatRef, MaxPlus};
//! use tropical_gemm_cuda::{CudaContext, GpuMat};
//!
//! let ctx = CudaContext::new()?;
//!
//! // Create CPU matrices
//! let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
//!
//! // Upload to GPU
//! let a_gpu = GpuMat::from_matref(&ctx, &a)?;
//!
//! // Compute on GPU
//! let b_gpu = GpuMat::from_matref(&ctx, &b)?;
//! let c_gpu = a_gpu.matmul(&ctx, &b_gpu)?;
//!
//! // Download result
//! let c = c_gpu.to_mat(&ctx)?;
//! ```

use std::marker::PhantomData;

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use tropical_gemm::mat::{Mat, MatRef, MatWithArgmax};
use tropical_gemm::TropicalSemiring;

use crate::context::CudaContext;
use crate::error::Result;
use crate::kernels::{CudaKernel, CudaKernelWithArgmax};
use crate::memory::{ArgmaxIndex, GpuMatrix, GpuMatrixWithArgmax};

/// A GPU matrix with embedded semiring type information.
///
/// This provides a higher-level API compared to [`GpuMatrix`], embedding
/// the semiring type so operations know which algebra to use.
pub struct GpuMat<S: TropicalSemiring>
where
    S::Scalar: DeviceRepr,
{
    inner: GpuMatrix<S::Scalar>,
    _phantom: PhantomData<S>,
}

// Basic methods, construction, and conversion
impl<S> GpuMat<S>
where
    S: TropicalSemiring,
    S::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Create a GPU matrix from a CPU MatRef.
    ///
    /// The MatRef data is expected to be in column-major order.
    pub fn from_matref(ctx: &CudaContext, mat: &MatRef<S>) -> Result<Self> {
        let inner = GpuMatrix::from_host(ctx, mat.as_slice(), mat.nrows(), mat.ncols())?;
        Ok(Self {
            inner,
            _phantom: PhantomData,
        })
    }

    /// Create a GPU matrix from raw scalar data.
    ///
    /// Data should be in column-major order.
    pub fn from_slice(
        ctx: &CudaContext,
        data: &[S::Scalar],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self> {
        let inner = GpuMatrix::from_host(ctx, data, nrows, ncols)?;
        Ok(Self {
            inner,
            _phantom: PhantomData,
        })
    }

    /// Allocate a zeroed GPU matrix.
    pub fn zeros(ctx: &CudaContext, nrows: usize, ncols: usize) -> Result<Self> {
        let inner = GpuMatrix::alloc(ctx, nrows, ncols)?;
        Ok(Self {
            inner,
            _phantom: PhantomData,
        })
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.inner.rows()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.inner.cols()
    }

    /// Get the underlying GpuMatrix.
    pub fn as_gpu_matrix(&self) -> &GpuMatrix<S::Scalar> {
        &self.inner
    }

    /// Get mutable access to the underlying GpuMatrix.
    pub fn as_gpu_matrix_mut(&mut self) -> &mut GpuMatrix<S::Scalar> {
        &mut self.inner
    }

    /// Convert to a CPU Mat.
    ///
    /// Returns data in column-major order.
    pub fn to_mat(&self, ctx: &CudaContext) -> Result<Mat<S>>
    where
        S::Scalar: Copy,
    {
        let data = self.inner.to_host(ctx)?;
        Ok(Mat::from_col_major(&data, self.nrows(), self.ncols()))
    }
}

// Kernel operations
impl<S> GpuMat<S>
where
    S: CudaKernel,
    S::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Perform tropical matrix multiplication on GPU.
    ///
    /// Computes C = A ⊗ B where ⊗ is the tropical matmul defined by the semiring S.
    pub fn matmul(&self, ctx: &CudaContext, b: &GpuMat<S>) -> Result<GpuMat<S>> {
        if self.ncols() != b.nrows() {
            return Err(crate::CudaError::DimensionMismatch(format!(
                "A.ncols ({}) != B.nrows ({})",
                self.ncols(),
                b.nrows()
            )));
        }

        let mut c = GpuMat::zeros(ctx, self.nrows(), b.ncols())?;
        S::launch_gemm(ctx, &self.inner, &b.inner, &mut c.inner)?;
        Ok(c)
    }
}

impl<S> GpuMat<S>
where
    S: CudaKernelWithArgmax,
    S::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Perform tropical matrix multiplication with argmax tracking.
    ///
    /// Returns both the result matrix and argmax indices for backward propagation.
    pub fn matmul_argmax(&self, ctx: &CudaContext, b: &GpuMat<S>) -> Result<GpuMatWithArgmax<S>> {
        if self.ncols() != b.nrows() {
            return Err(crate::CudaError::DimensionMismatch(format!(
                "A.ncols ({}) != B.nrows ({})",
                self.ncols(),
                b.nrows()
            )));
        }

        let mut c = GpuMatrixWithArgmax::alloc(ctx, self.nrows(), b.ncols())?;
        S::launch_gemm_with_argmax(ctx, &self.inner, &b.inner, &mut c)?;
        Ok(GpuMatWithArgmax {
            inner: c,
            _phantom: PhantomData,
        })
    }

    /// Batched tropical matrix multiplication with argmax tracking.
    ///
    /// Computes C[i] = A[i] ⊗ B[i] for each pair of GPU matrices,
    /// tracking which k-index produced each optimal value.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `a_batch` - Slice of GPU matrices A[i]
    /// * `b_batch` - Slice of GPU matrices B[i]
    ///
    /// # Returns
    ///
    /// Vector of GpuMatWithArgmax results, one for each matrix pair.
    pub fn matmul_batched_with_argmax(
        ctx: &CudaContext,
        a_batch: &[GpuMat<S>],
        b_batch: &[GpuMat<S>],
    ) -> Result<Vec<GpuMatWithArgmax<S>>> {
        if a_batch.len() != b_batch.len() {
            return Err(crate::CudaError::DimensionMismatch(format!(
                "Batch sizes must match: {} != {}",
                a_batch.len(),
                b_batch.len()
            )));
        }

        if a_batch.is_empty() {
            return Ok(Vec::new());
        }

        // Validate dimensions
        let (m, k) = (a_batch[0].nrows(), a_batch[0].ncols());
        let n = b_batch[0].ncols();

        for (i, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
            if a.nrows() != m || a.ncols() != k {
                return Err(crate::CudaError::DimensionMismatch(format!(
                    "A[{}] has dimensions {}x{}, expected {}x{}",
                    i,
                    a.nrows(),
                    a.ncols(),
                    m,
                    k
                )));
            }
            if b.nrows() != k || b.ncols() != n {
                return Err(crate::CudaError::DimensionMismatch(format!(
                    "B[{}] has dimensions {}x{}, expected {}x{}",
                    i,
                    b.nrows(),
                    b.ncols(),
                    k,
                    n
                )));
            }
        }

        a_batch
            .iter()
            .zip(b_batch.iter())
            .map(|(a, b)| a.matmul_argmax(ctx, b))
            .collect()
    }
}

// Batched operations
impl<S> GpuMat<S>
where
    S: CudaKernel,
    S::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Upload a batch of CPU matrices to GPU.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tropical_gemm::{Mat, MaxPlus};
    /// use tropical_gemm_cuda::{CudaContext, GpuMat};
    ///
    /// let ctx = CudaContext::new()?;
    /// let mats = vec![
    ///     Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2),
    ///     Mat::<MaxPlus<f32>>::from_row_major(&[5.0, 6.0, 7.0, 8.0], 2, 2),
    /// ];
    /// let gpu_mats = GpuMat::from_mats(&ctx, &mats)?;
    /// ```
    pub fn from_mats(ctx: &CudaContext, mats: &[Mat<S>]) -> Result<Vec<GpuMat<S>>>
    where
        S::Scalar: Copy,
    {
        mats.iter()
            .map(|m| {
                let mat_ref = m.as_ref();
                GpuMat::from_matref(ctx, &mat_ref)
            })
            .collect()
    }

    /// Download a batch of GPU matrices to CPU.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cpu_mats = GpuMat::to_mats(&ctx, &gpu_mats)?;
    /// ```
    pub fn to_mats(ctx: &CudaContext, gpu_mats: &[GpuMat<S>]) -> Result<Vec<Mat<S>>>
    where
        S::Scalar: Copy,
    {
        gpu_mats.iter().map(|m| m.to_mat(ctx)).collect()
    }

    /// Batched tropical matrix multiplication on GPU.
    ///
    /// Computes C[i] = A[i] ⊗ B[i] for each pair of GPU matrices.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `a_batch` - Slice of GPU matrices A[i]
    /// * `b_batch` - Slice of GPU matrices B[i]
    ///
    /// # Returns
    ///
    /// Vector of GPU result matrices C[i].
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tropical_gemm::{Mat, MaxPlus};
    /// use tropical_gemm_cuda::{CudaContext, GpuMat};
    ///
    /// let ctx = CudaContext::new()?;
    ///
    /// // Upload matrices to GPU
    /// let a_gpu = GpuMat::from_mats(&ctx, &a_mats)?;
    /// let b_gpu = GpuMat::from_mats(&ctx, &b_mats)?;
    ///
    /// // Batched multiply on GPU
    /// let c_gpu = GpuMat::<MaxPlus<f32>>::matmul_batched(&ctx, &a_gpu, &b_gpu)?;
    ///
    /// // Download results
    /// let c_mats = GpuMat::to_mats(&ctx, &c_gpu)?;
    /// ```
    pub fn matmul_batched(
        ctx: &CudaContext,
        a_batch: &[GpuMat<S>],
        b_batch: &[GpuMat<S>],
    ) -> Result<Vec<GpuMat<S>>> {
        if a_batch.len() != b_batch.len() {
            return Err(crate::CudaError::DimensionMismatch(format!(
                "Batch sizes must match: {} != {}",
                a_batch.len(),
                b_batch.len()
            )));
        }

        if a_batch.is_empty() {
            return Ok(Vec::new());
        }

        // Validate dimensions
        let (m, k) = (a_batch[0].nrows(), a_batch[0].ncols());
        let n = b_batch[0].ncols();

        for (i, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
            if a.nrows() != m || a.ncols() != k {
                return Err(crate::CudaError::DimensionMismatch(format!(
                    "A[{}] has dimensions {}x{}, expected {}x{}",
                    i,
                    a.nrows(),
                    a.ncols(),
                    m,
                    k
                )));
            }
            if b.nrows() != k || b.ncols() != n {
                return Err(crate::CudaError::DimensionMismatch(format!(
                    "B[{}] has dimensions {}x{}, expected {}x{}",
                    i,
                    b.nrows(),
                    b.ncols(),
                    k,
                    n
                )));
            }
        }

        a_batch
            .iter()
            .zip(b_batch.iter())
            .map(|(a, b)| a.matmul(ctx, b))
            .collect()
    }
}

/// A GPU matrix with argmax tracking for backward propagation.
pub struct GpuMatWithArgmax<S: TropicalSemiring>
where
    S::Scalar: DeviceRepr,
{
    inner: GpuMatrixWithArgmax<S::Scalar>,
    _phantom: PhantomData<S>,
}

impl<S> GpuMatWithArgmax<S>
where
    S: TropicalSemiring,
    S::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.inner.rows()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.inner.cols()
    }

    /// Convert to CPU MatWithArgmax.
    ///
    /// Returns data in column-major order.
    pub fn to_mat_with_argmax(&self, ctx: &CudaContext) -> Result<MatWithArgmax<S>>
    where
        S: tropical_gemm::TropicalWithArgmax<Index = u32>,
        S::Scalar: Copy,
    {
        let values_data = self.inner.matrix_to_host(ctx)?;
        let argmax_data = self.inner.argmax_to_host(ctx)?;

        let values = Mat::from_col_major(&values_data, self.nrows(), self.ncols());
        let argmax: Vec<u32> = argmax_data.into_iter().map(|x| x as u32).collect();

        Ok(MatWithArgmax { values, argmax })
    }

    /// Get just the result matrix as CPU Mat.
    ///
    /// Returns data in column-major order.
    pub fn to_mat(&self, ctx: &CudaContext) -> Result<Mat<S>>
    where
        S::Scalar: Copy,
    {
        let data = self.inner.matrix_to_host(ctx)?;
        Ok(Mat::from_col_major(&data, self.nrows(), self.ncols()))
    }

    /// Get just the argmax indices (column-major order).
    pub fn to_argmax(&self, ctx: &CudaContext) -> Result<Vec<ArgmaxIndex>> {
        self.inner.argmax_to_host(ctx)
    }

    /// Compute gradient with respect to matrix A.
    ///
    /// Given the upstream gradient dL/dC, computes dL/dA using the argmax
    /// indices from the forward pass.
    ///
    /// For C = A ⊗ B where C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j]):
    /// dL/dA[i,k] = Σ_j { dL/dC[i,j] if argmax[i,j] == k }
    ///
    /// All matrices are in column-major order.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `grad_c` - Gradient of the loss with respect to C, dimensions m×n
    /// * `k` - Number of columns in A (the inner dimension)
    ///
    /// # Returns
    ///
    /// Gradient of the loss with respect to A, dimensions m×k
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tropical_gemm::{Mat, MaxPlus};
    /// use tropical_gemm_cuda::{CudaContext, GpuMat};
    ///
    /// let ctx = CudaContext::new()?;
    ///
    /// // Forward pass with argmax
    /// let a_gpu = GpuMat::from_matref(&ctx, &a)?;
    /// let b_gpu = GpuMat::from_matref(&ctx, &b)?;
    /// let result = a_gpu.matmul_argmax(&ctx, &b_gpu)?;
    ///
    /// // Backward pass
    /// let grad_c = Mat::<MaxPlus<f32>>::from_fn(m, n, |_, _| MaxPlus(1.0));
    /// let grad_a = result.backward_a(&ctx, &grad_c, k)?;
    /// ```
    pub fn backward_a<G>(&self, ctx: &CudaContext, grad_c: &Mat<G>, k: usize) -> Result<Mat<G>>
    where
        G: TropicalSemiring,
        G::Scalar: Copy + Default + std::ops::AddAssign,
    {
        let m = self.nrows();
        let n = self.ncols();
        assert_eq!(grad_c.nrows(), m, "grad_c rows mismatch");
        assert_eq!(grad_c.ncols(), n, "grad_c cols mismatch");

        // Download argmax to host (column-major)
        let argmax = self.inner.argmax_to_host(ctx)?;

        let mut grad_a_data = vec![G::Scalar::default(); m * k];

        // Column-major indexing: element (i,j) is at index i + j*m
        for j in 0..n {
            for i in 0..m {
                let col_idx = i + j * m;
                let kk = argmax[col_idx] as usize;
                if kk < k {
                    // grad_a[i, kk] += grad_c[i, j]
                    grad_a_data[i + kk * m] += grad_c[(i, j)].value();
                }
            }
        }

        Ok(Mat::from_col_major(&grad_a_data, m, k))
    }

    /// Compute gradient with respect to matrix B.
    ///
    /// Given the upstream gradient dL/dC, computes dL/dB using the argmax
    /// indices from the forward pass.
    ///
    /// For C = A ⊗ B where C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j]):
    /// dL/dB[k,j] = Σ_i { dL/dC[i,j] if argmax[i,j] == k }
    ///
    /// All matrices are in column-major order.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `grad_c` - Gradient of the loss with respect to C, dimensions m×n
    /// * `k` - Number of rows in B (the inner dimension)
    ///
    /// # Returns
    ///
    /// Gradient of the loss with respect to B, dimensions k×n
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tropical_gemm::{Mat, MaxPlus};
    /// use tropical_gemm_cuda::{CudaContext, GpuMat};
    ///
    /// let ctx = CudaContext::new()?;
    ///
    /// // Forward pass with argmax
    /// let a_gpu = GpuMat::from_matref(&ctx, &a)?;
    /// let b_gpu = GpuMat::from_matref(&ctx, &b)?;
    /// let result = a_gpu.matmul_argmax(&ctx, &b_gpu)?;
    ///
    /// // Backward pass
    /// let grad_c = Mat::<MaxPlus<f32>>::from_fn(m, n, |_, _| MaxPlus(1.0));
    /// let grad_b = result.backward_b(&ctx, &grad_c, k)?;
    /// ```
    pub fn backward_b<G>(&self, ctx: &CudaContext, grad_c: &Mat<G>, k: usize) -> Result<Mat<G>>
    where
        G: TropicalSemiring,
        G::Scalar: Copy + Default + std::ops::AddAssign,
    {
        let m = self.nrows();
        let n = self.ncols();
        assert_eq!(grad_c.nrows(), m, "grad_c rows mismatch");
        assert_eq!(grad_c.ncols(), n, "grad_c cols mismatch");

        // Download argmax to host (column-major)
        let argmax = self.inner.argmax_to_host(ctx)?;

        let mut grad_b_data = vec![G::Scalar::default(); k * n];

        // Column-major indexing: element (i,j) is at index i + j*m
        for j in 0..n {
            for i in 0..m {
                let col_idx = i + j * m;
                let kk = argmax[col_idx] as usize;
                if kk < k {
                    // grad_b[kk, j] += grad_c[i, j]
                    grad_b_data[kk + j * k] += grad_c[(i, j)].value();
                }
            }
        }

        Ok(Mat::from_col_major(&grad_b_data, k, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tropical_gemm::{Mat, MaxPlus, MinPlus, TropicalSemiring};

    #[test]
    fn test_gpu_mat_basic() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = MatRef::<MaxPlus<f32>>::from_slice(&data, 2, 3);

        let a_gpu = GpuMat::from_matref(&ctx, &a).unwrap();
        assert_eq!(a_gpu.nrows(), 2);
        assert_eq!(a_gpu.ncols(), 3);

        let a_back = a_gpu.to_mat(&ctx).unwrap();
        assert_eq!(a_back.nrows(), 2);
        assert_eq!(a_back.ncols(), 3);
        assert_eq!(a_back[(0, 0)].value(), 1.0);
        assert_eq!(a_back[(1, 2)].value(), 6.0);
    }

    #[test]
    fn test_gpu_mat_matmul() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);

        let a_gpu = GpuMat::from_matref(&ctx, &a).unwrap();
        let b_gpu = GpuMat::from_matref(&ctx, &b).unwrap();

        let c_gpu = a_gpu.matmul(&ctx, &b_gpu).unwrap();
        let c = c_gpu.to_mat(&ctx).unwrap();

        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert!((c[(0, 0)].value() - 8.0).abs() < 1e-5);
        // C[1,1] = max(4+2, 5+4, 6+6) = 12
        assert!((c[(1, 1)].value() - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_mat_matmul_argmax() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);

        let a_gpu = GpuMat::from_matref(&ctx, &a).unwrap();
        let b_gpu = GpuMat::from_matref(&ctx, &b).unwrap();

        let result_gpu = a_gpu.matmul_argmax(&ctx, &b_gpu).unwrap();
        let result = result_gpu.to_mat_with_argmax(&ctx).unwrap();

        assert!((result.get(0, 0).value() - 8.0).abs() < 1e-5);
        assert_eq!(result.get_argmax(0, 0), 2); // k=2 gave max
    }

    #[test]
    fn test_gpu_mat_minplus() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = MatRef::<MinPlus<f32>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<MinPlus<f32>>::from_slice(&b_data, 3, 2);

        let a_gpu = GpuMat::from_matref(&ctx, &a).unwrap();
        let b_gpu = GpuMat::from_matref(&ctx, &b).unwrap();

        let c_gpu = a_gpu.matmul(&ctx, &b_gpu).unwrap();
        let c = c_gpu.to_mat(&ctx).unwrap();

        // C[0,0] = min(1+1, 2+3, 3+5) = 2
        assert!((c[(0, 0)].value() - 2.0).abs() < 1e-5);
        // C[1,1] = min(4+2, 5+4, 6+6) = 6
        assert!((c[(1, 1)].value() - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_mat_batched() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        // Create batch of CPU matrices
        let a1 = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let a2 = Mat::<MaxPlus<f32>>::from_row_major(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let b1 = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let b2 = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0], 2, 2);

        // Upload batch to GPU
        let a_gpu = GpuMat::from_mats(&ctx, &[a1, a2]).unwrap();
        let b_gpu = GpuMat::from_mats(&ctx, &[b1, b2]).unwrap();

        // Batched matmul
        let c_gpu = GpuMat::<MaxPlus<f32>>::matmul_batched(&ctx, &a_gpu, &b_gpu).unwrap();
        assert_eq!(c_gpu.len(), 2);

        // Download results
        let c_mats = GpuMat::to_mats(&ctx, &c_gpu).unwrap();
        assert_eq!(c_mats.len(), 2);

        // C[0] = A[0] * B[0] (MaxPlus)
        // C[0,0] = max(1+1, 2+0) = 2
        assert!((c_mats[0][(0, 0)].value() - 2.0).abs() < 1e-5);

        // C[1] = A[1] * B[1] (MaxPlus)
        // C[0,0] = max(5+1, 6+3) = 9
        assert!((c_mats[1][(0, 0)].value() - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_mat_batched_with_argmax() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let a1 = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let a2 = Mat::<MaxPlus<f32>>::from_row_major(&[6.0, 5.0, 4.0, 3.0, 2.0, 1.0], 2, 3);
        let b1 = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let b2 = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        // Upload to GPU
        let a_gpu = GpuMat::from_mats(&ctx, &[a1, a2]).unwrap();
        let b_gpu = GpuMat::from_mats(&ctx, &[b1, b2]).unwrap();

        // Batched matmul with argmax
        let results =
            GpuMat::<MaxPlus<f32>>::matmul_batched_with_argmax(&ctx, &a_gpu, &b_gpu).unwrap();
        assert_eq!(results.len(), 2);

        // Download and verify
        let r0 = results[0].to_mat_with_argmax(&ctx).unwrap();
        assert!((r0.get(0, 0).value() - 8.0).abs() < 1e-5); // max(1+1, 2+3, 3+5) = 8
        assert_eq!(r0.get_argmax(0, 0), 2);
    }

    #[test]
    fn test_gpu_mat_batched_empty() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let a_gpu: Vec<GpuMat<MaxPlus<f32>>> = vec![];
        let b_gpu: Vec<GpuMat<MaxPlus<f32>>> = vec![];

        let c_gpu = GpuMat::<MaxPlus<f32>>::matmul_batched(&ctx, &a_gpu, &b_gpu).unwrap();
        assert!(c_gpu.is_empty());
    }

    #[test]
    fn test_gpu_mat_backward_a() {
        use tropical_gemm::TropicalMaxPlus;

        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);

        let a_gpu = GpuMat::from_matref(&ctx, &a).unwrap();
        let b_gpu = GpuMat::from_matref(&ctx, &b).unwrap();

        // Forward pass
        let result = a_gpu.matmul_argmax(&ctx, &b_gpu).unwrap();

        // Backward pass with unit gradients
        let grad_c = Mat::<MaxPlus<f32>>::from_fn(2, 2, |_, _| TropicalMaxPlus(1.0));
        let grad_a = result.backward_a(&ctx, &grad_c, 3).unwrap();

        // All argmax should be 2 (k=2 wins for all)
        // So only column 2 should have gradients
        assert_eq!(grad_a.nrows(), 2);
        assert_eq!(grad_a.ncols(), 3);
        assert!((grad_a[(0, 0)].value() - 0.0).abs() < 1e-5);
        assert!((grad_a[(0, 1)].value() - 0.0).abs() < 1e-5);
        assert!((grad_a[(0, 2)].value() - 2.0).abs() < 1e-5); // 2 outputs use k=2
        assert!((grad_a[(1, 2)].value() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_mat_backward_b() {
        use tropical_gemm::TropicalMaxPlus;

        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);

        let a_gpu = GpuMat::from_matref(&ctx, &a).unwrap();
        let b_gpu = GpuMat::from_matref(&ctx, &b).unwrap();

        // Forward pass
        let result = a_gpu.matmul_argmax(&ctx, &b_gpu).unwrap();

        // Backward pass with unit gradients
        let grad_c = Mat::<MaxPlus<f32>>::from_fn(2, 2, |_, _| TropicalMaxPlus(1.0));
        let grad_b = result.backward_b(&ctx, &grad_c, 3).unwrap();

        // All argmax should be 2 (k=2 wins for all)
        // So only row 2 should have gradients
        assert_eq!(grad_b.nrows(), 3);
        assert_eq!(grad_b.ncols(), 2);
        assert!((grad_b[(0, 0)].value() - 0.0).abs() < 1e-5);
        assert!((grad_b[(1, 0)].value() - 0.0).abs() < 1e-5);
        assert!((grad_b[(2, 0)].value() - 2.0).abs() < 1e-5); // 2 outputs use k=2
        assert!((grad_b[(2, 1)].value() - 2.0).abs() < 1e-5);
    }
}
