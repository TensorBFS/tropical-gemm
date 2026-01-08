//! CUDA backend for tropical matrix multiplication.
//!
//! This crate provides GPU-accelerated tropical GEMM operations using CUDA.
//!
//! # Quick Start
//!
//! ```ignore
//! use tropical_gemm_cuda::{tropical_matmul_gpu, CudaContext};
//! use tropical_types::TropicalMaxPlus;
//!
//! // Simple one-shot API
//! let a = vec![1.0f32; 1024 * 1024];
//! let b = vec![1.0f32; 1024 * 1024];
//! let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, 1024, 1024, &b, 1024)?;
//! ```
//!
//! # Persistent Context
//!
//! For multiple operations, reuse the CUDA context to avoid repeated kernel compilation:
//!
//! ```ignore
//! use tropical_gemm_cuda::{CudaContext, GpuMatrix, tropical_gemm_gpu};
//! use tropical_types::TropicalMaxPlus;
//!
//! let ctx = CudaContext::new()?;
//!
//! let a_gpu = GpuMatrix::from_host_row_major(&ctx, &a, m, k)?;
//! let b_gpu = GpuMatrix::from_host_row_major(&ctx, &b, k, n)?;
//! let mut c_gpu = GpuMatrix::alloc(&ctx, m, n)?;
//!
//! tropical_gemm_gpu::<TropicalMaxPlus<f32>>(&ctx, &a_gpu, &b_gpu, &mut c_gpu)?;
//!
//! let c = c_gpu.to_host_row_major(&ctx)?;
//! ```

mod context;
mod error;
mod kernels;
mod memory;

pub use context::CudaContext;
pub use error::{CudaError, Result};
pub use kernels::{CudaKernel, CudaKernelWithArgmax};
pub use memory::{ArgmaxIndex, GpuMatrix, GpuMatrixWithArgmax};

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

/// One-shot tropical matrix multiplication on GPU.
///
/// This function handles all GPU memory management automatically.
/// For repeated operations, use `tropical_gemm_gpu` with a persistent context.
///
/// # Arguments
///
/// * `a` - Matrix A in row-major order, dimensions m×k
/// * `m` - Number of rows in A
/// * `k` - Number of columns in A / rows in B
/// * `b` - Matrix B in row-major order, dimensions k×n
/// * `n` - Number of columns in B
///
/// # Returns
///
/// Result matrix C in row-major order, dimensions m×n
///
/// # Example
///
/// ```ignore
/// use tropical_gemm_cuda::tropical_matmul_gpu;
/// use tropical_types::TropicalMaxPlus;
///
/// let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
/// let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
///
/// let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2)?;
/// // c is 2x2, row-major
/// ```
pub fn tropical_matmul_gpu<T>(
    a: &[T::Scalar],
    m: usize,
    k: usize,
    b: &[T::Scalar],
    n: usize,
) -> Result<Vec<T::Scalar>>
where
    T: CudaKernel,
    T::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    if a.len() != m * k {
        return Err(CudaError::DimensionMismatch(format!(
            "A: expected {} elements, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(CudaError::DimensionMismatch(format!(
            "B: expected {} elements, got {}",
            k * n,
            b.len()
        )));
    }

    let ctx = CudaContext::new()?;

    let a_gpu = GpuMatrix::from_host_row_major(&ctx, a, m, k)?;
    let b_gpu = GpuMatrix::from_host_row_major(&ctx, b, k, n)?;
    let mut c_gpu = GpuMatrix::alloc(&ctx, m, n)?;

    T::launch_gemm(&ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

    c_gpu.to_host_row_major(&ctx)
}

/// Tropical matrix multiplication with persistent context.
///
/// Use this function when performing multiple GPU operations to avoid
/// repeated context initialization and kernel compilation.
///
/// # Arguments
///
/// * `ctx` - CUDA context
/// * `a` - Matrix A on GPU
/// * `b` - Matrix B on GPU
/// * `c` - Output matrix C on GPU (will be overwritten)
pub fn tropical_gemm_gpu<T>(
    ctx: &CudaContext,
    a: &GpuMatrix<T::Scalar>,
    b: &GpuMatrix<T::Scalar>,
    c: &mut GpuMatrix<T::Scalar>,
) -> Result<()>
where
    T: CudaKernel,
    T::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    if a.cols() != b.rows() {
        return Err(CudaError::DimensionMismatch(format!(
            "A.cols ({}) != B.rows ({})",
            a.cols(),
            b.rows()
        )));
    }
    if c.rows() != a.rows() || c.cols() != b.cols() {
        return Err(CudaError::DimensionMismatch(format!(
            "C dimensions ({}, {}) don't match A×B ({}, {})",
            c.rows(),
            c.cols(),
            a.rows(),
            b.cols()
        )));
    }

    T::launch_gemm(ctx, a, b, c)
}

/// Tropical matrix multiplication with context, returning a new GPU matrix.
///
/// Allocates the output matrix automatically.
pub fn tropical_matmul_gpu_with_ctx<T>(
    ctx: &CudaContext,
    a: &GpuMatrix<T::Scalar>,
    b: &GpuMatrix<T::Scalar>,
) -> Result<GpuMatrix<T::Scalar>>
where
    T: CudaKernel,
    T::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    if a.cols() != b.rows() {
        return Err(CudaError::DimensionMismatch(format!(
            "A.cols ({}) != B.rows ({})",
            a.cols(),
            b.rows()
        )));
    }

    let mut c = GpuMatrix::alloc(ctx, a.rows(), b.cols())?;
    T::launch_gemm(ctx, a, b, &mut c)?;
    Ok(c)
}

// ============================================================================
// Argmax API - for backward propagation
// ============================================================================

/// One-shot tropical matrix multiplication with argmax tracking on GPU.
///
/// Returns both the result matrix C and the argmax indices that indicate
/// which k-index produced each C[i,j]. This is essential for backward
/// propagation in tropical neural networks.
///
/// # Arguments
///
/// * `a` - Matrix A in row-major order, dimensions m×k
/// * `m` - Number of rows in A
/// * `k` - Number of columns in A / rows in B
/// * `b` - Matrix B in row-major order, dimensions k×n
/// * `n` - Number of columns in B
///
/// # Returns
///
/// A tuple of (C, argmax) where:
/// - C is the result matrix in row-major order, dimensions m×n
/// - argmax[i,j] is the k-index such that C[i,j] = A[i,k] ⊗ B[k,j]
///
/// # Example
///
/// ```ignore
/// use tropical_gemm_cuda::tropical_matmul_gpu_with_argmax;
/// use tropical_types::TropicalMaxPlus;
///
/// let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
/// let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
///
/// let (c, argmax) = tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2)?;
/// // c is 2x2, argmax is 2x2 with k-indices
/// ```
pub fn tropical_matmul_gpu_with_argmax<T>(
    a: &[T::Scalar],
    m: usize,
    k: usize,
    b: &[T::Scalar],
    n: usize,
) -> Result<(Vec<T::Scalar>, Vec<ArgmaxIndex>)>
where
    T: CudaKernelWithArgmax,
    T::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    if a.len() != m * k {
        return Err(CudaError::DimensionMismatch(format!(
            "A: expected {} elements, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(CudaError::DimensionMismatch(format!(
            "B: expected {} elements, got {}",
            k * n,
            b.len()
        )));
    }

    let ctx = CudaContext::new()?;

    let a_gpu = GpuMatrix::from_host_row_major(&ctx, a, m, k)?;
    let b_gpu = GpuMatrix::from_host_row_major(&ctx, b, k, n)?;
    let mut c_gpu = GpuMatrixWithArgmax::alloc(&ctx, m, n)?;

    T::launch_gemm_with_argmax(&ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

    let c = c_gpu.matrix_to_host_row_major(&ctx)?;
    let argmax = c_gpu.argmax_to_host_row_major(&ctx)?;

    Ok((c, argmax))
}

/// Tropical matrix multiplication with argmax using persistent context.
///
/// Use this function when performing multiple GPU operations to avoid
/// repeated context initialization and kernel compilation.
///
/// # Arguments
///
/// * `ctx` - CUDA context
/// * `a` - Matrix A on GPU
/// * `b` - Matrix B on GPU
/// * `c` - Output matrix with argmax on GPU (will be overwritten)
pub fn tropical_gemm_gpu_with_argmax<T>(
    ctx: &CudaContext,
    a: &GpuMatrix<T::Scalar>,
    b: &GpuMatrix<T::Scalar>,
    c: &mut GpuMatrixWithArgmax<T::Scalar>,
) -> Result<()>
where
    T: CudaKernelWithArgmax,
    T::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    if a.cols() != b.rows() {
        return Err(CudaError::DimensionMismatch(format!(
            "A.cols ({}) != B.rows ({})",
            a.cols(),
            b.rows()
        )));
    }
    if c.rows() != a.rows() || c.cols() != b.cols() {
        return Err(CudaError::DimensionMismatch(format!(
            "C dimensions ({}, {}) don't match A×B ({}, {})",
            c.rows(),
            c.cols(),
            a.rows(),
            b.cols()
        )));
    }

    T::launch_gemm_with_argmax(ctx, a, b, c)
}

/// Tropical matrix multiplication with argmax, returning a new GPU matrix.
///
/// Allocates the output matrix and argmax buffer automatically.
pub fn tropical_matmul_gpu_with_ctx_and_argmax<T>(
    ctx: &CudaContext,
    a: &GpuMatrix<T::Scalar>,
    b: &GpuMatrix<T::Scalar>,
) -> Result<GpuMatrixWithArgmax<T::Scalar>>
where
    T: CudaKernelWithArgmax,
    T::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    if a.cols() != b.rows() {
        return Err(CudaError::DimensionMismatch(format!(
            "A.cols ({}) != B.rows ({})",
            a.cols(),
            b.rows()
        )));
    }

    let mut c = GpuMatrixWithArgmax::alloc(ctx, a.rows(), b.cols())?;
    T::launch_gemm_with_argmax(ctx, a, b, &mut c)?;
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tropical_types::{TropicalMaxPlus, TropicalMinPlus};

    /// Helper to check if CUDA is available
    fn cuda_context_or_skip() -> Option<CudaContext> {
        let result = std::panic::catch_unwind(|| CudaContext::new());
        match result {
            Ok(Ok(ctx)) => Some(ctx),
            Ok(Err(e)) => {
                println!("CUDA not available (error: {:?}), skipping test", e);
                None
            }
            Err(_) => {
                println!("CUDA libraries not found, skipping test");
                None
            }
        }
    }

    #[test]
    fn test_tropical_matmul_gpu_small() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // 2x3 matrix A
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // 3x2 matrix B
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2).unwrap();

        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert!((c[0] - 8.0).abs() < 1e-5, "C[0,0] = {}, expected 8", c[0]);
        // C[0,1] = max(1+2, 2+4, 3+6) = 9
        assert!((c[1] - 9.0).abs() < 1e-5, "C[0,1] = {}, expected 9", c[1]);
        // C[1,0] = max(4+1, 5+3, 6+5) = 11
        assert!((c[2] - 11.0).abs() < 1e-5, "C[1,0] = {}, expected 11", c[2]);
        // C[1,1] = max(4+2, 5+4, 6+6) = 12
        assert!((c[3] - 12.0).abs() < 1e-5, "C[1,1] = {}, expected 12", c[3]);
    }

    #[test]
    fn test_tropical_matmul_gpu_with_argmax_maxplus() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // 2x3 matrix A (row-major)
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // 3x2 matrix B (row-major)
        // B = [[1, 2],
        //      [3, 4],
        //      [5, 6]]
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2).unwrap();

        // C[0,0] = max(1+1=2, 2+3=5, 3+5=8) = 8, argmax=2
        assert!((c[0] - 8.0).abs() < 1e-5, "C[0,0] = {}, expected 8", c[0]);
        assert_eq!(argmax[0], 2, "argmax[0,0] = {}, expected 2", argmax[0]);

        // C[0,1] = max(1+2=3, 2+4=6, 3+6=9) = 9, argmax=2
        assert!((c[1] - 9.0).abs() < 1e-5, "C[0,1] = {}, expected 9", c[1]);
        assert_eq!(argmax[1], 2, "argmax[0,1] = {}, expected 2", argmax[1]);

        // C[1,0] = max(4+1=5, 5+3=8, 6+5=11) = 11, argmax=2
        assert!((c[2] - 11.0).abs() < 1e-5, "C[1,0] = {}, expected 11", c[2]);
        assert_eq!(argmax[2], 2, "argmax[1,0] = {}, expected 2", argmax[2]);

        // C[1,1] = max(4+2=6, 5+4=9, 6+6=12) = 12, argmax=2
        assert!((c[3] - 12.0).abs() < 1e-5, "C[1,1] = {}, expected 12", c[3]);
        assert_eq!(argmax[3], 2, "argmax[1,1] = {}, expected 2", argmax[3]);
    }

    #[test]
    fn test_tropical_matmul_gpu_with_argmax_minplus() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // 2x3 matrix A (row-major)
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // 3x2 matrix B (row-major)
        // B = [[1, 2],
        //      [3, 4],
        //      [5, 6]]
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMinPlus<f32>>(&a, 2, 3, &b, 2).unwrap();

        // C[0,0] = min(1+1=2, 2+3=5, 3+5=8) = 2, argmax=0
        assert!((c[0] - 2.0).abs() < 1e-5, "C[0,0] = {}, expected 2", c[0]);
        assert_eq!(argmax[0], 0, "argmax[0,0] = {}, expected 0", argmax[0]);

        // C[0,1] = min(1+2=3, 2+4=6, 3+6=9) = 3, argmax=0
        assert!((c[1] - 3.0).abs() < 1e-5, "C[0,1] = {}, expected 3", c[1]);
        assert_eq!(argmax[1], 0, "argmax[0,1] = {}, expected 0", argmax[1]);

        // C[1,0] = min(4+1=5, 5+3=8, 6+5=11) = 5, argmax=0
        assert!((c[2] - 5.0).abs() < 1e-5, "C[1,0] = {}, expected 5", c[2]);
        assert_eq!(argmax[2], 0, "argmax[1,0] = {}, expected 0", argmax[2]);

        // C[1,1] = min(4+2=6, 5+4=9, 6+6=12) = 6, argmax=0
        assert!((c[3] - 6.0).abs() < 1e-5, "C[1,1] = {}, expected 6", c[3]);
        assert_eq!(argmax[3], 0, "argmax[1,1] = {}, expected 0", argmax[3]);
    }

    #[test]
    fn test_tropical_matmul_gpu_with_argmax_varied_winners() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // Design a matrix where different k-indices win for different output elements
        // A = [[10, 1, 1],
        //      [1, 10, 1]]
        let a = vec![10.0f32, 1.0, 1.0, 1.0, 10.0, 1.0];
        // B = [[1, 1],
        //      [1, 1],
        //      [10, 10]]
        let b = vec![1.0f32, 1.0, 1.0, 1.0, 10.0, 10.0];

        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2).unwrap();

        // C[0,0] = max(10+1=11, 1+1=2, 1+10=11) = 11
        // First occurrence wins (k=0), as we use > not >=
        assert!((c[0] - 11.0).abs() < 1e-5, "C[0,0] = {}, expected 11", c[0]);
        assert_eq!(argmax[0], 0, "argmax[0,0] = {}, expected 0", argmax[0]);

        // C[0,1] = max(10+1=11, 1+1=2, 1+10=11) = 11, first k=0 wins
        assert!((c[1] - 11.0).abs() < 1e-5, "C[0,1] = {}, expected 11", c[1]);
        assert_eq!(argmax[1], 0, "argmax[0,1] = {}, expected 0", argmax[1]);

        // C[1,0] = max(1+1=2, 10+1=11, 1+10=11) = 11, first k=1 wins
        assert!((c[2] - 11.0).abs() < 1e-5, "C[1,0] = {}, expected 11", c[2]);
        assert_eq!(argmax[2], 1, "argmax[1,0] = {}, expected 1", argmax[2]);

        // C[1,1] = max(1+1=2, 10+1=11, 1+10=11) = 11, first k=1 wins
        assert!((c[3] - 11.0).abs() < 1e-5, "C[1,1] = {}, expected 11", c[3]);
        assert_eq!(argmax[3], 1, "argmax[1,1] = {}, expected 1", argmax[3]);
    }

    #[test]
    fn test_tropical_matmul_gpu_with_argmax_f64() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // 2x3 matrix A (row-major)
        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        // 3x2 matrix B (row-major)
        let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f64>>(&a, 2, 3, &b, 2).unwrap();

        // C[0,0] = max(1+1=2, 2+3=5, 3+5=8) = 8, argmax=2
        assert!((c[0] - 8.0).abs() < 1e-10, "C[0,0] = {}, expected 8", c[0]);
        assert_eq!(argmax[0], 2, "argmax[0,0] = {}, expected 2", argmax[0]);

        // C[1,1] = max(4+2=6, 5+4=9, 6+6=12) = 12, argmax=2
        assert!(
            (c[3] - 12.0).abs() < 1e-10,
            "C[1,1] = {}, expected 12",
            c[3]
        );
        assert_eq!(argmax[3], 2, "argmax[1,1] = {}, expected 2", argmax[3]);
    }
}
