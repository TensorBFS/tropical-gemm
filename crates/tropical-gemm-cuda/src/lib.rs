//! CUDA backend for tropical matrix multiplication.
//!
//! This crate provides GPU-accelerated tropical GEMM operations using CUDA.
//!
//! # Quick Start
//!
//! ```ignore
//! use tropical_gemm_cuda::{tropical_matmul_gpu, CudaContext};
//! use tropical_gemm::types::TropicalMaxPlus;
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
//! use tropical_gemm::types::TropicalMaxPlus;
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
mod gpu_mat;
mod kernels;
mod memory;

pub use context::CudaContext;
pub use error::{CudaError, Result};
pub use gpu_mat::{GpuMat, GpuMatWithArgmax};
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
/// use tropical_gemm::types::TropicalMaxPlus;
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
/// use tropical_gemm::types::TropicalMaxPlus;
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

// ============================================================================
// Batched GEMM API
// ============================================================================

/// Batched tropical matrix multiplication on GPU.
///
/// Computes C[i] = A[i] ⊗ B[i] for i = 0..batch_size.
/// All matrices in the batch must have the same dimensions.
///
/// # Arguments
///
/// * `a_batch` - Slice of batch_size matrices A[i], each of size m×k
/// * `b_batch` - Slice of batch_size matrices B[i], each of size k×n
/// * `m` - Number of rows in each A matrix
/// * `k` - Number of columns in A / rows in B
/// * `n` - Number of columns in each B matrix
///
/// # Returns
///
/// Vector of batch_size result matrices C[i], each of size m×n
///
/// # Example
///
/// ```ignore
/// use tropical_gemm_cuda::tropical_matmul_gpu_batched;
/// use tropical_gemm::TropicalMaxPlus;
///
/// let a_batch = vec![
///     vec![1.0f32, 2.0, 3.0, 4.0],  // A[0]: 2x2
///     vec![5.0f32, 6.0, 7.0, 8.0],  // A[1]: 2x2
/// ];
/// let b_batch = vec![
///     vec![1.0f32, 2.0, 3.0, 4.0],  // B[0]: 2x2
///     vec![1.0f32, 2.0, 3.0, 4.0],  // B[1]: 2x2
/// ];
///
/// let c_batch = tropical_matmul_gpu_batched::<TropicalMaxPlus<f32>>(&a_batch, &b_batch, 2, 2, 2)?;
/// assert_eq!(c_batch.len(), 2);
/// ```
pub fn tropical_matmul_gpu_batched<T>(
    a_batch: &[Vec<T::Scalar>],
    b_batch: &[Vec<T::Scalar>],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<Vec<T::Scalar>>>
where
    T: CudaKernel,
    T::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    if a_batch.len() != b_batch.len() {
        return Err(CudaError::DimensionMismatch(format!(
            "Batch sizes must match: A has {} matrices, B has {}",
            a_batch.len(),
            b_batch.len()
        )));
    }

    let batch_size = a_batch.len();
    if batch_size == 0 {
        return Ok(Vec::new());
    }

    // Validate dimensions
    for (i, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
        if a.len() != m * k {
            return Err(CudaError::DimensionMismatch(format!(
                "A[{}] dimensions mismatch: expected {}, got {}",
                i,
                m * k,
                a.len()
            )));
        }
        if b.len() != k * n {
            return Err(CudaError::DimensionMismatch(format!(
                "B[{}] dimensions mismatch: expected {}, got {}",
                i,
                k * n,
                b.len()
            )));
        }
    }

    let ctx = CudaContext::new()?;
    let mut results = Vec::with_capacity(batch_size);

    for (a, b) in a_batch.iter().zip(b_batch.iter()) {
        let a_gpu = GpuMatrix::from_host_row_major(&ctx, a, m, k)?;
        let b_gpu = GpuMatrix::from_host_row_major(&ctx, b, k, n)?;
        let mut c_gpu = GpuMatrix::alloc(&ctx, m, n)?;

        T::launch_gemm(&ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

        results.push(c_gpu.to_host_row_major(&ctx)?);
    }

    Ok(results)
}

/// Strided batched tropical matrix multiplication on GPU.
///
/// Computes C[i] = A[i] ⊗ B[i] from contiguous memory.
/// More efficient than `tropical_matmul_gpu_batched` when matrices are stored contiguously.
///
/// # Arguments
///
/// * `a` - Contiguous array of all A matrices (batch_size × m × k elements)
/// * `b` - Contiguous array of all B matrices (batch_size × k × n elements)
/// * `batch_size` - Number of matrix pairs
/// * `m` - Rows in each A
/// * `k` - Columns in A / rows in B
/// * `n` - Columns in each B
///
/// # Returns
///
/// Contiguous array of all C matrices (batch_size × m × n elements)
pub fn tropical_matmul_gpu_strided_batched<T>(
    a: &[T::Scalar],
    b: &[T::Scalar],
    batch_size: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<T::Scalar>>
where
    T: CudaKernel,
    T::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    if a.len() != batch_size * a_stride {
        return Err(CudaError::DimensionMismatch(format!(
            "A size mismatch: expected {}, got {}",
            batch_size * a_stride,
            a.len()
        )));
    }
    if b.len() != batch_size * b_stride {
        return Err(CudaError::DimensionMismatch(format!(
            "B size mismatch: expected {}, got {}",
            batch_size * b_stride,
            b.len()
        )));
    }

    if batch_size == 0 {
        return Ok(Vec::new());
    }

    let ctx = CudaContext::new()?;
    let mut c = vec![T::Scalar::default(); batch_size * c_stride];

    for i in 0..batch_size {
        let a_slice = &a[i * a_stride..(i + 1) * a_stride];
        let b_slice = &b[i * b_stride..(i + 1) * b_stride];

        let a_gpu = GpuMatrix::from_host_row_major(&ctx, a_slice, m, k)?;
        let b_gpu = GpuMatrix::from_host_row_major(&ctx, b_slice, k, n)?;
        let mut c_gpu = GpuMatrix::alloc(&ctx, m, n)?;

        T::launch_gemm(&ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

        let c_result = c_gpu.to_host_row_major(&ctx)?;
        c[i * c_stride..(i + 1) * c_stride].copy_from_slice(&c_result);
    }

    Ok(c)
}

// ============================================================================
// Backward Pass API
// ============================================================================

/// Compute gradient with respect to matrix A from GPU argmax indices.
///
/// This function takes the argmax indices (typically downloaded from GPU after
/// a forward pass with `tropical_matmul_gpu_with_argmax`) and computes the gradient
/// for backpropagation.
///
/// # Arguments
///
/// * `grad_c` - Gradient of the loss with respect to C, dimensions m×n
/// * `argmax` - Argmax indices from forward pass (k-index that produced each C[i,j])
/// * `m` - Number of rows in A and C
/// * `k` - Number of columns in A / rows in B
/// * `n` - Number of columns in B and C
///
/// # Returns
///
/// Gradient of the loss with respect to A, dimensions m×k
///
/// # Example
///
/// ```ignore
/// use tropical_gemm_cuda::{tropical_matmul_gpu_with_argmax, tropical_backward_a_gpu};
/// use tropical_gemm::TropicalMaxPlus;
///
/// // Forward pass
/// let (c, argmax) = tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)?;
///
/// // Backward pass (given grad_c from upstream)
/// let grad_a = tropical_backward_a_gpu(&grad_c, &argmax, m, k, n);
/// ```
pub fn tropical_backward_a_gpu<T>(
    grad_c: &[T],
    argmax: &[ArgmaxIndex],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T>
where
    T: Copy + Default + std::ops::AddAssign,
{
    assert_eq!(grad_c.len(), m * n, "grad_c size mismatch");
    assert_eq!(argmax.len(), m * n, "argmax size mismatch");

    let mut grad_a = vec![T::default(); m * k];

    for i in 0..m {
        for j in 0..n {
            let idx = argmax[i * n + j] as usize;
            if idx < k {
                grad_a[i * k + idx] += grad_c[i * n + j];
            }
        }
    }

    grad_a
}

/// Compute gradient with respect to matrix B from GPU argmax indices.
///
/// # Arguments
///
/// * `grad_c` - Gradient of the loss with respect to C, dimensions m×n
/// * `argmax` - Argmax indices from forward pass (k-index that produced each C[i,j])
/// * `m` - Number of rows in A and C
/// * `k` - Number of columns in A / rows in B
/// * `n` - Number of columns in B and C
///
/// # Returns
///
/// Gradient of the loss with respect to B, dimensions k×n
///
/// # Example
///
/// ```ignore
/// use tropical_gemm_cuda::{tropical_matmul_gpu_with_argmax, tropical_backward_b_gpu};
/// use tropical_gemm::TropicalMaxPlus;
///
/// // Forward pass
/// let (c, argmax) = tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)?;
///
/// // Backward pass (given grad_c from upstream)
/// let grad_b = tropical_backward_b_gpu(&grad_c, &argmax, m, k, n);
/// ```
pub fn tropical_backward_b_gpu<T>(
    grad_c: &[T],
    argmax: &[ArgmaxIndex],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T>
where
    T: Copy + Default + std::ops::AddAssign,
{
    assert_eq!(grad_c.len(), m * n, "grad_c size mismatch");
    assert_eq!(argmax.len(), m * n, "argmax size mismatch");

    let mut grad_b = vec![T::default(); k * n];

    for i in 0..m {
        for j in 0..n {
            let idx = argmax[i * n + j] as usize;
            if idx < k {
                grad_b[idx * n + j] += grad_c[i * n + j];
            }
        }
    }

    grad_b
}

/// Batched backward pass for gradient with respect to A on GPU.
///
/// Computes gradients for a batch of matrices. Each batch element is processed
/// independently.
///
/// # Arguments
///
/// * `grad_c_batch` - Batch of gradients w.r.t. C, each of dimensions m×n
/// * `argmax_batch` - Batch of argmax indices from forward pass
/// * `m` - Number of rows in each A and C
/// * `k` - Number of columns in A / rows in B
/// * `n` - Number of columns in B and C
///
/// # Returns
///
/// Batch of gradients w.r.t. A, each of dimensions m×k
pub fn tropical_backward_a_gpu_batched<T>(
    grad_c_batch: &[Vec<T>],
    argmax_batch: &[Vec<ArgmaxIndex>],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Vec<T>>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
{
    assert_eq!(
        grad_c_batch.len(),
        argmax_batch.len(),
        "batch sizes must match"
    );

    grad_c_batch
        .iter()
        .zip(argmax_batch.iter())
        .map(|(grad_c, argmax)| tropical_backward_a_gpu(grad_c, argmax, m, k, n))
        .collect()
}

/// Batched backward pass for gradient with respect to B on GPU.
///
/// Computes gradients for a batch of matrices. Each batch element is processed
/// independently.
///
/// # Arguments
///
/// * `grad_c_batch` - Batch of gradients w.r.t. C, each of dimensions m×n
/// * `argmax_batch` - Batch of argmax indices from forward pass
/// * `m` - Number of rows in each A and C
/// * `k` - Number of columns in A / rows in B
/// * `n` - Number of columns in B and C
///
/// # Returns
///
/// Batch of gradients w.r.t. B, each of dimensions k×n
pub fn tropical_backward_b_gpu_batched<T>(
    grad_c_batch: &[Vec<T>],
    argmax_batch: &[Vec<ArgmaxIndex>],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Vec<T>>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
{
    assert_eq!(
        grad_c_batch.len(),
        argmax_batch.len(),
        "batch sizes must match"
    );

    grad_c_batch
        .iter()
        .zip(argmax_batch.iter())
        .map(|(grad_c, argmax)| tropical_backward_b_gpu(grad_c, argmax, m, k, n))
        .collect()
}

// ============================================================================
// True GPU Backward Pass (using CUDA kernels with atomicAdd)
// ============================================================================

/// Compute gradient with respect to matrix A on GPU using CUDA kernel.
///
/// This is a true GPU implementation using atomicAdd for parallel scatter.
/// Much faster than CPU for large matrices.
///
/// # Arguments
///
/// * `ctx` - CUDA context with compiled kernels
/// * `grad_c` - Gradient w.r.t. C on GPU (M x N)
/// * `argmax` - Argmax indices on GPU (M x N)
/// * `m` - Number of rows in A and C
/// * `k` - Number of columns in A / rows in B
/// * `n` - Number of columns in B and C
///
/// # Returns
///
/// Gradient w.r.t. A on GPU (M x K), initialized to zero and accumulated
pub fn tropical_backward_a_gpu_kernel(
    ctx: &CudaContext,
    grad_c: &GpuMatrix<f32>,
    argmax: &cudarc::driver::CudaSlice<i32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<GpuMatrix<f32>> {
    use cudarc::driver::LaunchAsync;

    // Allocate output gradient (initialized to zero)
    let mut grad_a = GpuMatrix::alloc(ctx, m, k)?;

    let kernel = ctx.get_kernel("tropical_backward_a_f32")?;

    let total = m * n;
    let block_size = 256u32;
    let grid_size = ((total as u32) + block_size - 1) / block_size;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        kernel.launch(
            cfg,
            (
                grad_c.as_slice(),
                argmax,
                grad_a.as_slice_mut(),
                m as i32,
                n as i32,
                k as i32,
            ),
        )?;
    }

    ctx.device().synchronize()?;
    Ok(grad_a)
}

/// Compute gradient with respect to matrix B on GPU using CUDA kernel.
///
/// This is a true GPU implementation using atomicAdd for parallel scatter.
/// Much faster than CPU for large matrices.
///
/// # Arguments
///
/// * `ctx` - CUDA context with compiled kernels
/// * `grad_c` - Gradient w.r.t. C on GPU (M x N)
/// * `argmax` - Argmax indices on GPU (M x N)
/// * `m` - Number of rows in A and C
/// * `k` - Number of columns in A / rows in B
/// * `n` - Number of columns in B and C
///
/// # Returns
///
/// Gradient w.r.t. B on GPU (K x N), initialized to zero and accumulated
pub fn tropical_backward_b_gpu_kernel(
    ctx: &CudaContext,
    grad_c: &GpuMatrix<f32>,
    argmax: &cudarc::driver::CudaSlice<i32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<GpuMatrix<f32>> {
    use cudarc::driver::LaunchAsync;

    // Allocate output gradient (initialized to zero)
    let mut grad_b = GpuMatrix::alloc(ctx, k, n)?;

    let kernel = ctx.get_kernel("tropical_backward_b_f32")?;

    let total = m * n;
    let block_size = 256u32;
    let grid_size = ((total as u32) + block_size - 1) / block_size;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        kernel.launch(
            cfg,
            (
                grad_c.as_slice(),
                argmax,
                grad_b.as_slice_mut(),
                m as i32,
                n as i32,
                k as i32,
            ),
        )?;
    }

    ctx.device().synchronize()?;
    Ok(grad_b)
}

/// High-level GPU backward pass for gradient w.r.t. A.
///
/// Uploads grad_c and argmax to GPU, computes gradient, downloads result.
/// For best performance, use `tropical_backward_a_gpu_kernel` with data already on GPU.
pub fn tropical_backward_a_gpu_cuda(
    ctx: &CudaContext,
    grad_c: &[f32],
    argmax: &[ArgmaxIndex],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    assert_eq!(grad_c.len(), m * n, "grad_c size mismatch");
    assert_eq!(argmax.len(), m * n, "argmax size mismatch");

    // Upload to GPU (column-major conversion)
    let grad_c_gpu = GpuMatrix::from_host_row_major(ctx, grad_c, m, n)?;

    // Convert argmax to i32 and upload
    let argmax_i32: Vec<i32> = argmax.iter().map(|&x| x as i32).collect();
    // Convert to column-major for GPU
    let mut argmax_col_major = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            argmax_col_major[i + j * m] = argmax_i32[i * n + j];
        }
    }
    let argmax_gpu = ctx.device().htod_sync_copy(&argmax_col_major)?;

    // Run kernel
    let grad_a_gpu = tropical_backward_a_gpu_kernel(ctx, &grad_c_gpu, &argmax_gpu, m, k, n)?;

    // Download result
    grad_a_gpu.to_host_row_major(ctx)
}

/// High-level GPU backward pass for gradient w.r.t. B.
///
/// Uploads grad_c and argmax to GPU, computes gradient, downloads result.
/// For best performance, use `tropical_backward_b_gpu_kernel` with data already on GPU.
pub fn tropical_backward_b_gpu_cuda(
    ctx: &CudaContext,
    grad_c: &[f32],
    argmax: &[ArgmaxIndex],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    assert_eq!(grad_c.len(), m * n, "grad_c size mismatch");
    assert_eq!(argmax.len(), m * n, "argmax size mismatch");

    // Upload to GPU (column-major conversion)
    let grad_c_gpu = GpuMatrix::from_host_row_major(ctx, grad_c, m, n)?;

    // Convert argmax to i32 and upload
    let argmax_i32: Vec<i32> = argmax.iter().map(|&x| x as i32).collect();
    // Convert to column-major for GPU
    let mut argmax_col_major = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            argmax_col_major[i + j * m] = argmax_i32[i * n + j];
        }
    }
    let argmax_gpu = ctx.device().htod_sync_copy(&argmax_col_major)?;

    // Run kernel
    let grad_b_gpu = tropical_backward_b_gpu_kernel(ctx, &grad_c_gpu, &argmax_gpu, m, k, n)?;

    // Download result
    grad_b_gpu.to_host_row_major(ctx)
}

/// Batched tropical matrix multiplication with argmax tracking on GPU.
///
/// Computes C[i] = A[i] ⊗ B[i] for i = 0..batch_size, with argmax indices.
pub fn tropical_matmul_gpu_batched_with_argmax<T>(
    a_batch: &[Vec<T::Scalar>],
    b_batch: &[Vec<T::Scalar>],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<(Vec<T::Scalar>, Vec<ArgmaxIndex>)>>
where
    T: CudaKernelWithArgmax,
    T::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    if a_batch.len() != b_batch.len() {
        return Err(CudaError::DimensionMismatch(format!(
            "Batch sizes must match: A has {} matrices, B has {}",
            a_batch.len(),
            b_batch.len()
        )));
    }

    let batch_size = a_batch.len();
    if batch_size == 0 {
        return Ok(Vec::new());
    }

    // Validate dimensions
    for (i, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
        if a.len() != m * k {
            return Err(CudaError::DimensionMismatch(format!(
                "A[{}] dimensions mismatch: expected {}, got {}",
                i,
                m * k,
                a.len()
            )));
        }
        if b.len() != k * n {
            return Err(CudaError::DimensionMismatch(format!(
                "B[{}] dimensions mismatch: expected {}, got {}",
                i,
                k * n,
                b.len()
            )));
        }
    }

    let ctx = CudaContext::new()?;
    let mut results = Vec::with_capacity(batch_size);

    for (a, b) in a_batch.iter().zip(b_batch.iter()) {
        let a_gpu = GpuMatrix::from_host_row_major(&ctx, a, m, k)?;
        let b_gpu = GpuMatrix::from_host_row_major(&ctx, b, k, n)?;
        let mut c_gpu = GpuMatrixWithArgmax::alloc(&ctx, m, n)?;

        T::launch_gemm_with_argmax(&ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

        let c_values = c_gpu.matrix_to_host_row_major(&ctx)?;
        let c_argmax = c_gpu.argmax_to_host_row_major(&ctx)?;
        results.push((c_values, c_argmax));
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tropical_gemm::types::{TropicalMaxPlus, TropicalMinPlus};

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

    #[test]
    fn test_argmax_finite_difference_maxplus() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        let m = 4;
        let k = 5;
        let n = 3;
        let epsilon = 1e-3f32; // Larger epsilon for better numerical stability

        // Random-ish matrices with distinct values to avoid ties
        let mut a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.7 - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5 - 2.0).collect();

        // Compute C and argmax
        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n).unwrap();

        // Test finite difference for each element of A
        for i in 0..m {
            for kk in 0..k {
                // Perturb A[i, kk]
                let a_idx = i * k + kk;
                a[a_idx] += epsilon;

                // Recompute C with perturbed A
                let (c_perturbed, _) =
                    tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)
                        .unwrap();

                // Restore A
                a[a_idx] -= epsilon;

                // Check gradient for each C[i, j]
                for j in 0..n {
                    let c_idx = i * n + j;
                    let numerical_grad = (c_perturbed[c_idx] - c[c_idx]) / epsilon;
                    let expected_grad = if argmax[c_idx] == kk as i32 { 1.0 } else { 0.0 };

                    assert!(
                        (numerical_grad - expected_grad).abs() < 0.05,
                        "Finite diff failed at A[{},{}] -> C[{},{}]: \
                         numerical={}, expected={}, argmax={}",
                        i,
                        kk,
                        i,
                        j,
                        numerical_grad,
                        expected_grad,
                        argmax[c_idx]
                    );
                }
            }
        }
        println!(
            "MaxPlus finite difference test passed for {}x{}x{} matrices",
            m, k, n
        );
    }

    #[test]
    fn test_argmax_finite_difference_minplus() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        let m = 4;
        let k = 5;
        let n = 3;
        let epsilon = 1e-3f32; // Larger epsilon for better numerical stability

        // Random-ish matrices with distinct values to avoid ties
        let mut a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.7 - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5 - 2.0).collect();

        // Compute C and argmax (argmin for MinPlus)
        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMinPlus<f32>>(&a, m, k, &b, n).unwrap();

        // Test finite difference for each element of A
        for i in 0..m {
            for kk in 0..k {
                // Perturb A[i, kk]
                let a_idx = i * k + kk;
                a[a_idx] += epsilon;

                // Recompute C with perturbed A
                let (c_perturbed, _) =
                    tropical_matmul_gpu_with_argmax::<TropicalMinPlus<f32>>(&a, m, k, &b, n)
                        .unwrap();

                // Restore A
                a[a_idx] -= epsilon;

                // Check gradient for each C[i, j]
                for j in 0..n {
                    let c_idx = i * n + j;
                    let numerical_grad = (c_perturbed[c_idx] - c[c_idx]) / epsilon;
                    let expected_grad = if argmax[c_idx] == kk as i32 { 1.0 } else { 0.0 };

                    assert!(
                        (numerical_grad - expected_grad).abs() < 0.05,
                        "Finite diff failed at A[{},{}] -> C[{},{}]: \
                         numerical={}, expected={}, argmax={}",
                        i,
                        kk,
                        i,
                        j,
                        numerical_grad,
                        expected_grad,
                        argmax[c_idx]
                    );
                }
            }
        }
        println!(
            "MinPlus finite difference test passed for {}x{}x{} matrices",
            m, k, n
        );
    }

    #[test]
    fn test_argmax_finite_difference_b_matrix() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        let m = 3;
        let k = 4;
        let n = 5;
        let epsilon = 1e-3f32; // Larger epsilon for better numerical stability

        // Random-ish matrices with distinct values
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.6 - 2.0).collect();
        let mut b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.4 - 1.5).collect();

        // Compute C and argmax
        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n).unwrap();

        // Test finite difference for each element of B
        for kk in 0..k {
            for j in 0..n {
                // Perturb B[kk, j]
                let b_idx = kk * n + j;
                b[b_idx] += epsilon;

                // Recompute C with perturbed B
                let (c_perturbed, _) =
                    tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)
                        .unwrap();

                // Restore B
                b[b_idx] -= epsilon;

                // Check gradient for each C[i, j]
                for i in 0..m {
                    let c_idx = i * n + j;
                    let numerical_grad = (c_perturbed[c_idx] - c[c_idx]) / epsilon;
                    let expected_grad = if argmax[c_idx] == kk as i32 { 1.0 } else { 0.0 };

                    assert!(
                        (numerical_grad - expected_grad).abs() < 0.05,
                        "Finite diff failed at B[{},{}] -> C[{},{}]: \
                         numerical={}, expected={}, argmax={}",
                        kk,
                        j,
                        i,
                        j,
                        numerical_grad,
                        expected_grad,
                        argmax[c_idx]
                    );
                }
            }
        }
        println!(
            "MaxPlus B-matrix finite difference test passed for {}x{}x{} matrices",
            m, k, n
        );
    }

    // ========================================================================
    // Batched GEMM tests
    // ========================================================================

    #[test]
    fn test_tropical_matmul_gpu_batched_basic() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // Batch of 2 matrices, each 2x2
        let a_batch = vec![
            vec![1.0f32, 2.0, 3.0, 4.0], // A[0]: [[1,2],[3,4]]
            vec![5.0f32, 6.0, 7.0, 8.0], // A[1]: [[5,6],[7,8]]
        ];
        let b_batch = vec![
            vec![1.0f32, 0.0, 0.0, 1.0], // B[0]: [[1,0],[0,1]] (identity-ish)
            vec![1.0f32, 2.0, 3.0, 4.0], // B[1]: [[1,2],[3,4]]
        ];

        let c_batch =
            tropical_matmul_gpu_batched::<TropicalMaxPlus<f32>>(&a_batch, &b_batch, 2, 2, 2)
                .unwrap();

        assert_eq!(c_batch.len(), 2);

        // C[0] = A[0] * B[0] (MaxPlus)
        // C[0,0] = max(1+1, 2+0) = 2
        // C[0,1] = max(1+0, 2+1) = 3
        // C[1,0] = max(3+1, 4+0) = 4
        // C[1,1] = max(3+0, 4+1) = 5
        assert!((c_batch[0][0] - 2.0).abs() < 1e-5);
        assert!((c_batch[0][1] - 3.0).abs() < 1e-5);
        assert!((c_batch[0][2] - 4.0).abs() < 1e-5);
        assert!((c_batch[0][3] - 5.0).abs() < 1e-5);

        // C[1] = A[1] * B[1] (MaxPlus)
        // C[0,0] = max(5+1, 6+3) = 9
        // C[0,1] = max(5+2, 6+4) = 10
        // C[1,0] = max(7+1, 8+3) = 11
        // C[1,1] = max(7+2, 8+4) = 12
        assert!((c_batch[1][0] - 9.0).abs() < 1e-5);
        assert!((c_batch[1][1] - 10.0).abs() < 1e-5);
        assert!((c_batch[1][2] - 11.0).abs() < 1e-5);
        assert!((c_batch[1][3] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_tropical_matmul_gpu_batched_empty() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        let a_batch: Vec<Vec<f32>> = vec![];
        let b_batch: Vec<Vec<f32>> = vec![];

        let c_batch =
            tropical_matmul_gpu_batched::<TropicalMaxPlus<f32>>(&a_batch, &b_batch, 2, 2, 2)
                .unwrap();

        assert!(c_batch.is_empty());
    }

    #[test]
    fn test_tropical_matmul_gpu_batched_dimension_mismatch() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        let a_batch = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        let b_batch = vec![
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![5.0f32, 6.0, 7.0, 8.0], // Extra matrix
        ];

        let result =
            tropical_matmul_gpu_batched::<TropicalMaxPlus<f32>>(&a_batch, &b_batch, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_tropical_matmul_gpu_strided_batched_basic() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // 2 batches of 2x2 matrices, stored contiguously
        let a = vec![
            // Batch 0: [[1,2],[3,4]]
            1.0f32, 2.0, 3.0, 4.0, // Batch 1: [[5,6],[7,8]]
            5.0, 6.0, 7.0, 8.0,
        ];
        let b = vec![
            // Batch 0: [[1,0],[0,1]]
            1.0f32, 0.0, 0.0, 1.0, // Batch 1: [[1,2],[3,4]]
            1.0, 2.0, 3.0, 4.0,
        ];

        let c = tropical_matmul_gpu_strided_batched::<TropicalMaxPlus<f32>>(&a, &b, 2, 2, 2, 2)
            .unwrap();

        // Should have 2 * 2 * 2 = 8 elements
        assert_eq!(c.len(), 8);

        // Batch 0 results (same as above test)
        assert!((c[0] - 2.0).abs() < 1e-5);
        assert!((c[1] - 3.0).abs() < 1e-5);
        assert!((c[2] - 4.0).abs() < 1e-5);
        assert!((c[3] - 5.0).abs() < 1e-5);

        // Batch 1 results
        assert!((c[4] - 9.0).abs() < 1e-5);
        assert!((c[5] - 10.0).abs() < 1e-5);
        assert!((c[6] - 11.0).abs() < 1e-5);
        assert!((c[7] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_tropical_matmul_gpu_strided_batched_empty() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];

        let c = tropical_matmul_gpu_strided_batched::<TropicalMaxPlus<f32>>(&a, &b, 0, 2, 2, 2)
            .unwrap();

        assert!(c.is_empty());
    }

    #[test]
    fn test_tropical_matmul_gpu_batched_with_argmax_basic() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // Batch of 2 matrices
        let a_batch = vec![
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], // A[0]: 2x3
            vec![6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0], // A[1]: 2x3 (reversed)
        ];
        let b_batch = vec![
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], // B[0]: 3x2
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], // B[1]: 3x2
        ];

        let results = tropical_matmul_gpu_batched_with_argmax::<TropicalMaxPlus<f32>>(
            &a_batch, &b_batch, 2, 3, 2,
        )
        .unwrap();

        assert_eq!(results.len(), 2);

        // Batch 0: same as single matrix test
        let (c0, argmax0) = &results[0];
        assert!((c0[0] - 8.0).abs() < 1e-5); // max(1+1, 2+3, 3+5) = 8
        assert_eq!(argmax0[0], 2);

        // Batch 1: A[1] has reversed values
        // C[0,0] = max(6+1, 5+3, 4+5) = max(7, 8, 9) = 9, argmax=2
        let (c1, argmax1) = &results[1];
        assert!((c1[0] - 9.0).abs() < 1e-5);
        assert_eq!(argmax1[0], 2);
    }

    #[test]
    fn test_tropical_matmul_gpu_batched_minplus() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        let a_batch = vec![
            vec![1.0f32, 2.0, 3.0, 4.0], // 2x2
        ];
        let b_batch = vec![
            vec![1.0f32, 2.0, 3.0, 4.0], // 2x2
        ];

        let c_batch =
            tropical_matmul_gpu_batched::<TropicalMinPlus<f32>>(&a_batch, &b_batch, 2, 2, 2)
                .unwrap();

        // MinPlus: C[i,j] = min_k(A[i,k] + B[k,j])
        // C[0,0] = min(1+1, 2+3) = min(2, 5) = 2
        // C[0,1] = min(1+2, 2+4) = min(3, 6) = 3
        // C[1,0] = min(3+1, 4+3) = min(4, 7) = 4
        // C[1,1] = min(3+2, 4+4) = min(5, 8) = 5
        assert!((c_batch[0][0] - 2.0).abs() < 1e-5);
        assert!((c_batch[0][1] - 3.0).abs() < 1e-5);
        assert!((c_batch[0][2] - 4.0).abs() < 1e-5);
        assert!((c_batch[0][3] - 5.0).abs() < 1e-5);
    }

    // ========================================================================
    // Backward Pass tests
    // ========================================================================

    #[test]
    fn test_tropical_backward_a_gpu() {
        // Test backward pass for A
        // C[i,j] = A[i,argmax[i,j]] + B[argmax[i,j],j]
        // dL/dA[i,k] = sum_j { dL/dC[i,j] if argmax[i,j] == k }

        let m = 2;
        let k = 3;
        let n = 2;

        // Gradient from upstream (all ones for simplicity)
        let grad_c = vec![1.0f32; m * n];

        // Argmax: row-major, for each C[i,j] which k produced it
        // Let's say argmax = [[0, 2], [1, 2]]
        let argmax: Vec<ArgmaxIndex> = vec![0, 2, 1, 2];

        let grad_a = tropical_backward_a_gpu(&grad_c, &argmax, m, k, n);

        // Expected grad_a (2x3):
        // grad_a[0,0] = grad_c[0,0] because argmax[0,0]=0 -> 1.0
        // grad_a[0,1] = 0 (no argmax points here)
        // grad_a[0,2] = grad_c[0,1] because argmax[0,1]=2 -> 1.0
        // grad_a[1,0] = 0
        // grad_a[1,1] = grad_c[1,0] because argmax[1,0]=1 -> 1.0
        // grad_a[1,2] = grad_c[1,1] because argmax[1,1]=2 -> 1.0
        assert_eq!(grad_a.len(), m * k);
        assert!((grad_a[0] - 1.0).abs() < 1e-5); // [0,0]
        assert!((grad_a[1] - 0.0).abs() < 1e-5); // [0,1]
        assert!((grad_a[2] - 1.0).abs() < 1e-5); // [0,2]
        assert!((grad_a[3] - 0.0).abs() < 1e-5); // [1,0]
        assert!((grad_a[4] - 1.0).abs() < 1e-5); // [1,1]
        assert!((grad_a[5] - 1.0).abs() < 1e-5); // [1,2]
    }

    #[test]
    fn test_tropical_backward_b_gpu() {
        // Test backward pass for B
        // dL/dB[k,j] = sum_i { dL/dC[i,j] if argmax[i,j] == k }

        let m = 2;
        let k = 3;
        let n = 2;

        // Gradient from upstream (all ones)
        let grad_c = vec![1.0f32; m * n];

        // Argmax: [[0, 2], [1, 2]]
        let argmax: Vec<ArgmaxIndex> = vec![0, 2, 1, 2];

        let grad_b = tropical_backward_b_gpu(&grad_c, &argmax, m, k, n);

        // Expected grad_b (3x2):
        // grad_b[0,0] = grad_c[0,0] because argmax[0,0]=0 -> 1.0
        // grad_b[0,1] = 0
        // grad_b[1,0] = grad_c[1,0] because argmax[1,0]=1 -> 1.0
        // grad_b[1,1] = 0
        // grad_b[2,0] = 0
        // grad_b[2,1] = grad_c[0,1] + grad_c[1,1] because argmax[0,1]=2 and argmax[1,1]=2 -> 2.0
        assert_eq!(grad_b.len(), k * n);
        assert!((grad_b[0] - 1.0).abs() < 1e-5); // [0,0]
        assert!((grad_b[1] - 0.0).abs() < 1e-5); // [0,1]
        assert!((grad_b[2] - 1.0).abs() < 1e-5); // [1,0]
        assert!((grad_b[3] - 0.0).abs() < 1e-5); // [1,1]
        assert!((grad_b[4] - 0.0).abs() < 1e-5); // [2,0]
        assert!((grad_b[5] - 2.0).abs() < 1e-5); // [2,1]
    }

    #[test]
    fn test_tropical_backward_gpu_integration() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // Full integration test: forward pass on GPU, backward pass
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2

        let m = 2;
        let k = 3;
        let n = 2;

        // Forward pass on GPU
        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n).unwrap();

        // Verify forward pass
        assert!((c[0] - 8.0).abs() < 1e-5); // max(1+1, 2+3, 3+5)

        // Backward pass with unit gradients
        let grad_c = vec![1.0f32; m * n];
        let grad_a = tropical_backward_a_gpu(&grad_c, &argmax, m, k, n);
        let grad_b = tropical_backward_b_gpu(&grad_c, &argmax, m, k, n);

        // For this specific case, argmax should be all 2 (k=2 wins for all)
        // So grad_a[i,2] should be n (sum over j) and others 0
        // grad_a[0,2] = 2 (from C[0,0] and C[0,1])
        // grad_a[1,2] = 2 (from C[1,0] and C[1,1])
        assert_eq!(grad_a.len(), m * k);
        assert!((grad_a[2] - 2.0).abs() < 1e-5); // A[0,2]
        assert!((grad_a[5] - 2.0).abs() < 1e-5); // A[1,2]

        // grad_b[2,j] = m (sum over i) for each j
        assert_eq!(grad_b.len(), k * n);
        assert!((grad_b[4] - 2.0).abs() < 1e-5); // B[2,0]
        assert!((grad_b[5] - 2.0).abs() < 1e-5); // B[2,1]
    }

    #[test]
    fn test_tropical_backward_gpu_batched() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        let m = 2;
        let k = 3;
        let n = 2;

        // Batch of 2 forward passes
        let a_batch = vec![
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], // A[0]
            vec![6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0], // A[1] (reversed)
        ];
        let b_batch = vec![
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], // B[0]
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], // B[1]
        ];

        // Forward pass
        let results = tropical_matmul_gpu_batched_with_argmax::<TropicalMaxPlus<f32>>(
            &a_batch, &b_batch, m, k, n,
        )
        .unwrap();

        // Extract argmax for backward
        let argmax_batch: Vec<Vec<ArgmaxIndex>> =
            results.iter().map(|(_, argmax)| argmax.clone()).collect();

        // Backward pass
        let grad_c_batch = vec![vec![1.0f32; m * n]; 2];
        let grad_a_batch =
            tropical_backward_a_gpu_batched(&grad_c_batch, &argmax_batch, m, k, n);
        let grad_b_batch =
            tropical_backward_b_gpu_batched(&grad_c_batch, &argmax_batch, m, k, n);

        assert_eq!(grad_a_batch.len(), 2);
        assert_eq!(grad_b_batch.len(), 2);

        // Each gradient should have correct dimensions
        for grad_a in &grad_a_batch {
            assert_eq!(grad_a.len(), m * k);
        }
        for grad_b in &grad_b_batch {
            assert_eq!(grad_b.len(), k * n);
        }
    }
}
