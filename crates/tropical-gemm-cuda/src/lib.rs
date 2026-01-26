//! CUDA backend for tropical matrix multiplication.
//!
//! This crate provides GPU-accelerated tropical GEMM operations using CUDA.
//! All matrices use **column-major** storage (matching tropical-gemm's Mat type).
//!
//! # Quick Start
//!
//! ```ignore
//! use tropical_gemm_cuda::{tropical_matmul_gpu, CudaContext};
//! use tropical_gemm::types::TropicalMaxPlus;
//!
//! // Simple one-shot API (uses cached global context for performance)
//! // Data should be in column-major order
//! let a = vec![1.0f32; 1024 * 1024];
//! let b = vec![1.0f32; 1024 * 1024];
//! let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, 1024, 1024, &b, 1024)?;
//! ```
//!
//! # Persistent Context
//!
//! For explicit context management:
//!
//! ```ignore
//! use tropical_gemm_cuda::{CudaContext, GpuMatrix, tropical_gemm_gpu};
//! use tropical_gemm::types::TropicalMaxPlus;
//!
//! let ctx = CudaContext::new()?;
//!
//! // Data in column-major order (zero-copy upload)
//! let a_gpu = GpuMatrix::from_host(&ctx, &a, m, k)?;
//! let b_gpu = GpuMatrix::from_host(&ctx, &b, k, n)?;
//! let mut c_gpu = GpuMatrix::alloc(&ctx, m, n)?;
//!
//! tropical_gemm_gpu::<TropicalMaxPlus<f32>>(&ctx, &a_gpu, &b_gpu, &mut c_gpu)?;
//!
//! let c = c_gpu.to_host(&ctx)?;  // Column-major result
//! ```
//!
//! # Performance
//!
//! The convenience functions (`tropical_matmul_gpu`, etc.) use a lazily-initialized
//! global context that persists across calls. This avoids the ~7 second NVRTC
//! compilation overhead on each call.

mod context;
mod error;
mod gpu_mat;
mod kernels;
mod memory;

use once_cell::sync::OnceCell;
use std::sync::Mutex;

/// Global CUDA context for convenience functions.
/// Lazily initialized on first use, persists for process lifetime.
static GLOBAL_CONTEXT: OnceCell<CudaContext> = OnceCell::new();

/// Mutex to ensure only one thread initializes the context.
static INIT_MUTEX: Mutex<()> = Mutex::new(());

/// Get or initialize the global CUDA context.
///
/// This function is thread-safe and will only initialize the context once.
/// Subsequent calls return the cached context.
///
/// # Errors
///
/// Returns an error if CUDA initialization fails (no device, driver issues, etc.)
pub fn get_global_context() -> Result<&'static CudaContext> {
    // Fast path: already initialized
    if let Some(ctx) = GLOBAL_CONTEXT.get() {
        return Ok(ctx);
    }

    // Slow path: need to initialize
    let _lock = INIT_MUTEX.lock().unwrap();

    // Double-check after acquiring lock
    if let Some(ctx) = GLOBAL_CONTEXT.get() {
        return Ok(ctx);
    }

    // Initialize and store
    let ctx = CudaContext::new()?;
    let _ = GLOBAL_CONTEXT.set(ctx);

    Ok(GLOBAL_CONTEXT.get().unwrap())
}

pub use context::CudaContext;
pub use error::{CudaError, Result};
pub use gpu_mat::{GpuMat, GpuMatWithArgmax};
pub use kernels::{
    launch_gemm_external_f32, launch_gemm_external_with_argmax_f32, CudaKernel,
    CudaKernelWithArgmax,
};
pub use memory::{
    ArgmaxIndex, ExternalGpuMatrix, ExternalGpuMemory, GpuMatrix, GpuMatrixWithArgmax,
};

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

// ============================================================================
// Helper: validate GEMM dimensions
// ============================================================================

fn validate_gemm_input<T>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Result<()> {
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
    Ok(())
}

/// One-shot tropical matrix multiplication on GPU.
///
/// This function handles all GPU memory management automatically.
/// For repeated operations, use `tropical_gemm_gpu` with a persistent context.
///
/// # Arguments
///
/// * `a` - Matrix A in **column-major** order, dimensions m×k
/// * `m` - Number of rows in A
/// * `k` - Number of columns in A / rows in B
/// * `b` - Matrix B in **column-major** order, dimensions k×n
/// * `n` - Number of columns in B
///
/// # Returns
///
/// Result matrix C in **column-major** order, dimensions m×n
///
/// # Example
///
/// ```ignore
/// use tropical_gemm_cuda::tropical_matmul_gpu;
/// use tropical_gemm::types::TropicalMaxPlus;
///
/// // Column-major data
/// let a = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]; // 2x3 col-major
/// let b = vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0]; // 3x2 col-major
///
/// let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2)?;
/// // c is 2x2, column-major
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
    validate_gemm_input(a, b, m, k, n)?;

    let ctx = get_global_context()?;

    let a_gpu = GpuMatrix::from_host(ctx, a, m, k)?;
    let b_gpu = GpuMatrix::from_host(ctx, b, k, n)?;
    let mut c_gpu = GpuMatrix::alloc(ctx, m, n)?;

    T::launch_gemm(ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

    c_gpu.to_host(ctx)
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
/// * `a` - Matrix A in **column-major** order, dimensions m×k
/// * `m` - Number of rows in A
/// * `k` - Number of columns in A / rows in B
/// * `b` - Matrix B in **column-major** order, dimensions k×n
/// * `n` - Number of columns in B
///
/// # Returns
///
/// A tuple of (C, argmax) where:
/// - C is the result matrix in **column-major** order, dimensions m×n
/// - argmax[i,j] is the k-index such that C[i,j] = A[i,k] ⊗ B[k,j]
///
/// # Example
///
/// ```ignore
/// use tropical_gemm_cuda::tropical_matmul_gpu_with_argmax;
/// use tropical_gemm::types::TropicalMaxPlus;
///
/// // Column-major data
/// let a = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]; // 2x3 col-major
/// let b = vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0]; // 3x2 col-major
///
/// let (c, argmax) = tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2)?;
/// // c is 2x2 column-major, argmax is 2x2 column-major with k-indices
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
    validate_gemm_input(a, b, m, k, n)?;

    let ctx = get_global_context()?;

    let a_gpu = GpuMatrix::from_host(ctx, a, m, k)?;
    let b_gpu = GpuMatrix::from_host(ctx, b, k, n)?;
    let mut c_gpu = GpuMatrixWithArgmax::alloc(ctx, m, n)?;

    T::launch_gemm_with_argmax(ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

    let c = c_gpu.matrix_to_host(ctx)?;
    let argmax = c_gpu.argmax_to_host(ctx)?;

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

    let ctx = get_global_context()?;
    let mut results = Vec::with_capacity(batch_size);

    for (a, b) in a_batch.iter().zip(b_batch.iter()) {
        let a_gpu = GpuMatrix::from_host(ctx, a, m, k)?;
        let b_gpu = GpuMatrix::from_host(ctx, b, k, n)?;
        let mut c_gpu = GpuMatrix::alloc(ctx, m, n)?;

        T::launch_gemm(ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

        results.push(c_gpu.to_host(ctx)?);
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

    let ctx = get_global_context()?;
    let mut c = vec![T::Scalar::default(); batch_size * c_stride];

    for i in 0..batch_size {
        let a_slice = &a[i * a_stride..(i + 1) * a_stride];
        let b_slice = &b[i * b_stride..(i + 1) * b_stride];

        let a_gpu = GpuMatrix::from_host(ctx, a_slice, m, k)?;
        let b_gpu = GpuMatrix::from_host(ctx, b_slice, k, n)?;
        let mut c_gpu = GpuMatrix::alloc(ctx, m, n)?;

        T::launch_gemm(ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

        let c_result = c_gpu.to_host(ctx)?;
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
/// All matrices are in **column-major** order.
///
/// # Arguments
///
/// * `grad_c` - Gradient of the loss with respect to C (column-major), dimensions m×n
/// * `argmax` - Argmax indices from forward pass (column-major), k-index that produced each C[i,j]
/// * `m` - Number of rows in A and C
/// * `k` - Number of columns in A / rows in B
/// * `n` - Number of columns in B and C
///
/// # Returns
///
/// Gradient of the loss with respect to A (column-major), dimensions m×k
///
/// # Example
///
/// ```ignore
/// use tropical_gemm_cuda::{tropical_matmul_gpu_with_argmax, tropical_backward_a_gpu};
/// use tropical_gemm::TropicalMaxPlus;
///
/// // Forward pass (column-major data)
/// let (c, argmax) = tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)?;
///
/// // Backward pass (given grad_c from upstream, column-major)
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

    // Column-major indexing: element (i,j) is at index i + j*m
    for j in 0..n {
        for i in 0..m {
            let col_idx = i + j * m;
            let kk = argmax[col_idx] as usize;
            if kk < k {
                // grad_a[i, kk] += grad_c[i, j]
                grad_a[i + kk * m] += grad_c[col_idx];
            }
        }
    }

    grad_a
}

/// Compute gradient with respect to matrix B from GPU argmax indices.
///
/// All matrices are in **column-major** order.
///
/// # Arguments
///
/// * `grad_c` - Gradient of the loss with respect to C (column-major), dimensions m×n
/// * `argmax` - Argmax indices from forward pass (column-major), k-index that produced each C[i,j]
/// * `m` - Number of rows in A and C
/// * `k` - Number of columns in A / rows in B
/// * `n` - Number of columns in B and C
///
/// # Returns
///
/// Gradient of the loss with respect to B (column-major), dimensions k×n
///
/// # Example
///
/// ```ignore
/// use tropical_gemm_cuda::{tropical_matmul_gpu_with_argmax, tropical_backward_b_gpu};
/// use tropical_gemm::TropicalMaxPlus;
///
/// // Forward pass (column-major data)
/// let (c, argmax) = tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)?;
///
/// // Backward pass (given grad_c from upstream, column-major)
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

    // Column-major indexing: element (i,j) is at index i + j*m
    for j in 0..n {
        for i in 0..m {
            let col_idx = i + j * m;
            let kk = argmax[col_idx] as usize;
            if kk < k {
                // grad_b[kk, j] += grad_c[i, j]
                grad_b[kk + j * k] += grad_c[col_idx];
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
///
/// All matrices are in **column-major** order.
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

    // Upload to GPU (already column-major)
    let grad_c_gpu = GpuMatrix::from_host(ctx, grad_c, m, n)?;

    // Upload argmax directly (already column-major)
    let argmax_gpu = ctx.device().htod_sync_copy(argmax)?;

    // Run kernel
    let grad_a_gpu = tropical_backward_a_gpu_kernel(ctx, &grad_c_gpu, &argmax_gpu, m, k, n)?;

    // Download result (column-major)
    grad_a_gpu.to_host(ctx)
}

/// High-level GPU backward pass for gradient w.r.t. B.
///
/// Uploads grad_c and argmax to GPU, computes gradient, downloads result.
/// For best performance, use `tropical_backward_b_gpu_kernel` with data already on GPU.
///
/// All matrices are in **column-major** order.
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

    // Upload to GPU (already column-major)
    let grad_c_gpu = GpuMatrix::from_host(ctx, grad_c, m, n)?;

    // Upload argmax directly (already column-major)
    let argmax_gpu = ctx.device().htod_sync_copy(argmax)?;

    // Run kernel
    let grad_b_gpu = tropical_backward_b_gpu_kernel(ctx, &grad_c_gpu, &argmax_gpu, m, k, n)?;

    // Download result (column-major)
    grad_b_gpu.to_host(ctx)
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

    let ctx = get_global_context()?;
    let mut results = Vec::with_capacity(batch_size);

    for (a, b) in a_batch.iter().zip(b_batch.iter()) {
        let a_gpu = GpuMatrix::from_host(ctx, a, m, k)?;
        let b_gpu = GpuMatrix::from_host(ctx, b, k, n)?;
        let mut c_gpu = GpuMatrixWithArgmax::alloc(ctx, m, n)?;

        T::launch_gemm_with_argmax(ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

        let c_values = c_gpu.matrix_to_host(ctx)?;
        let c_argmax = c_gpu.argmax_to_host(ctx)?;
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

        // 2x3 matrix A (column-major)
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        let a = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        // 3x2 matrix B (column-major)
        // B = [[1, 2],
        //      [3, 4],
        //      [5, 6]]
        let b = vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0];

        let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2).unwrap();

        // Result is column-major: [C[0,0], C[1,0], C[0,1], C[1,1]]
        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert!((c[0] - 8.0).abs() < 1e-5, "C[0,0] = {}, expected 8", c[0]);
        // C[1,0] = max(4+1, 5+3, 6+5) = 11
        assert!((c[1] - 11.0).abs() < 1e-5, "C[1,0] = {}, expected 11", c[1]);
        // C[0,1] = max(1+2, 2+4, 3+6) = 9
        assert!((c[2] - 9.0).abs() < 1e-5, "C[0,1] = {}, expected 9", c[2]);
        // C[1,1] = max(4+2, 5+4, 6+6) = 12
        assert!((c[3] - 12.0).abs() < 1e-5, "C[1,1] = {}, expected 12", c[3]);
    }

    #[test]
    fn test_tropical_matmul_gpu_with_argmax_maxplus() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // 2x3 matrix A (column-major)
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        let a = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        // 3x2 matrix B (column-major)
        // B = [[1, 2],
        //      [3, 4],
        //      [5, 6]]
        let b = vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0];

        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2).unwrap();

        // Result is column-major: [C[0,0], C[1,0], C[0,1], C[1,1]]
        // C[0,0] = max(1+1=2, 2+3=5, 3+5=8) = 8, argmax=2
        assert!((c[0] - 8.0).abs() < 1e-5, "C[0,0] = {}, expected 8", c[0]);
        assert_eq!(argmax[0], 2, "argmax[0,0] = {}, expected 2", argmax[0]);

        // C[1,0] = max(4+1=5, 5+3=8, 6+5=11) = 11, argmax=2
        assert!((c[1] - 11.0).abs() < 1e-5, "C[1,0] = {}, expected 11", c[1]);
        assert_eq!(argmax[1], 2, "argmax[1,0] = {}, expected 2", argmax[1]);

        // C[0,1] = max(1+2=3, 2+4=6, 3+6=9) = 9, argmax=2
        assert!((c[2] - 9.0).abs() < 1e-5, "C[0,1] = {}, expected 9", c[2]);
        assert_eq!(argmax[2], 2, "argmax[0,1] = {}, expected 2", argmax[2]);

        // C[1,1] = max(4+2=6, 5+4=9, 6+6=12) = 12, argmax=2
        assert!((c[3] - 12.0).abs() < 1e-5, "C[1,1] = {}, expected 12", c[3]);
        assert_eq!(argmax[3], 2, "argmax[1,1] = {}, expected 2", argmax[3]);
    }

    #[test]
    fn test_tropical_matmul_gpu_with_argmax_minplus() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // 2x3 matrix A (column-major)
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        let a = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        // 3x2 matrix B (column-major)
        // B = [[1, 2],
        //      [3, 4],
        //      [5, 6]]
        let b = vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0];

        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMinPlus<f32>>(&a, 2, 3, &b, 2).unwrap();

        // Result is column-major: [C[0,0], C[1,0], C[0,1], C[1,1]]
        // C[0,0] = min(1+1=2, 2+3=5, 3+5=8) = 2, argmax=0
        assert!((c[0] - 2.0).abs() < 1e-5, "C[0,0] = {}, expected 2", c[0]);
        assert_eq!(argmax[0], 0, "argmax[0,0] = {}, expected 0", argmax[0]);

        // C[1,0] = min(4+1=5, 5+3=8, 6+5=11) = 5, argmax=0
        assert!((c[1] - 5.0).abs() < 1e-5, "C[1,0] = {}, expected 5", c[1]);
        assert_eq!(argmax[1], 0, "argmax[1,0] = {}, expected 0", argmax[1]);

        // C[0,1] = min(1+2=3, 2+4=6, 3+6=9) = 3, argmax=0
        assert!((c[2] - 3.0).abs() < 1e-5, "C[0,1] = {}, expected 3", c[2]);
        assert_eq!(argmax[2], 0, "argmax[0,1] = {}, expected 0", argmax[2]);

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
        // Column-major: [10, 1, 1, 10, 1, 1]
        let a = vec![10.0f32, 1.0, 1.0, 10.0, 1.0, 1.0];
        // B = [[1, 1],
        //      [1, 1],
        //      [10, 10]]
        // Column-major: [1, 1, 10, 1, 1, 10]
        let b = vec![1.0f32, 1.0, 10.0, 1.0, 1.0, 10.0];

        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2).unwrap();

        // Result is column-major: [C[0,0], C[1,0], C[0,1], C[1,1]]
        // C[0,0] = max(10+1=11, 1+1=2, 1+10=11) = 11
        // First occurrence wins (k=0), as we use > not >=
        assert!((c[0] - 11.0).abs() < 1e-5, "C[0,0] = {}, expected 11", c[0]);
        assert_eq!(argmax[0], 0, "argmax[0,0] = {}, expected 0", argmax[0]);

        // C[1,0] = max(1+1=2, 10+1=11, 1+10=11) = 11, first k=1 wins
        assert!((c[1] - 11.0).abs() < 1e-5, "C[1,0] = {}, expected 11", c[1]);
        assert_eq!(argmax[1], 1, "argmax[1,0] = {}, expected 1", argmax[1]);

        // C[0,1] = max(10+1=11, 1+1=2, 1+10=11) = 11, first k=0 wins
        assert!((c[2] - 11.0).abs() < 1e-5, "C[0,1] = {}, expected 11", c[2]);
        assert_eq!(argmax[2], 0, "argmax[0,1] = {}, expected 0", argmax[2]);

        // C[1,1] = max(1+1=2, 10+1=11, 1+10=11) = 11, first k=1 wins
        assert!((c[3] - 11.0).abs() < 1e-5, "C[1,1] = {}, expected 11", c[3]);
        assert_eq!(argmax[3], 1, "argmax[1,1] = {}, expected 1", argmax[3]);
    }

    #[test]
    fn test_tropical_matmul_gpu_with_argmax_f64() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // 2x3 matrix A (column-major)
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        let a = vec![1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        // 3x2 matrix B (column-major)
        // B = [[1, 2],
        //      [3, 4],
        //      [5, 6]]
        let b = vec![1.0f64, 3.0, 5.0, 2.0, 4.0, 6.0];

        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f64>>(&a, 2, 3, &b, 2).unwrap();

        // Result is column-major: [C[0,0], C[1,0], C[0,1], C[1,1]]
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
        let epsilon = 1e-3f32;

        // Generate column-major matrices with distinct values to avoid ties
        // A is m x k, B is k x n
        let mut a: Vec<f32> = (0..m * k)
            .map(|idx| {
                let i = idx % m;
                let kk = idx / m;
                ((i * k + kk) as f32) * 0.7 - 3.0
            })
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|idx| {
                let kk = idx % k;
                let j = idx / k;
                ((kk * n + j) as f32) * 0.5 - 2.0
            })
            .collect();

        // Compute C and argmax
        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n).unwrap();

        // Test finite difference for each element of A
        for i in 0..m {
            for kk in 0..k {
                // Perturb A[i, kk] (column-major: index = i + kk * m)
                let a_idx = i + kk * m;
                a[a_idx] += epsilon;

                // Recompute C with perturbed A
                let (c_perturbed, _) =
                    tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)
                        .unwrap();

                // Restore A
                a[a_idx] -= epsilon;

                // Check gradient for each C[i, j] (column-major: index = i + j * m)
                for j in 0..n {
                    let c_idx = i + j * m;
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
        let epsilon = 1e-3f32;

        // Generate column-major matrices with distinct values to avoid ties
        let mut a: Vec<f32> = (0..m * k)
            .map(|idx| {
                let i = idx % m;
                let kk = idx / m;
                ((i * k + kk) as f32) * 0.7 - 3.0
            })
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|idx| {
                let kk = idx % k;
                let j = idx / k;
                ((kk * n + j) as f32) * 0.5 - 2.0
            })
            .collect();

        // Compute C and argmax (argmin for MinPlus)
        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMinPlus<f32>>(&a, m, k, &b, n).unwrap();

        // Test finite difference for each element of A
        for i in 0..m {
            for kk in 0..k {
                // Perturb A[i, kk] (column-major: index = i + kk * m)
                let a_idx = i + kk * m;
                a[a_idx] += epsilon;

                // Recompute C with perturbed A
                let (c_perturbed, _) =
                    tropical_matmul_gpu_with_argmax::<TropicalMinPlus<f32>>(&a, m, k, &b, n)
                        .unwrap();

                // Restore A
                a[a_idx] -= epsilon;

                // Check gradient for each C[i, j] (column-major: index = i + j * m)
                for j in 0..n {
                    let c_idx = i + j * m;
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
        let epsilon = 1e-3f32;

        // Generate column-major matrices with distinct values
        let a: Vec<f32> = (0..m * k)
            .map(|idx| {
                let i = idx % m;
                let kk = idx / m;
                ((i * k + kk) as f32) * 0.6 - 2.0
            })
            .collect();
        let mut b: Vec<f32> = (0..k * n)
            .map(|idx| {
                let kk = idx % k;
                let j = idx / k;
                ((kk * n + j) as f32) * 0.4 - 1.5
            })
            .collect();

        // Compute C and argmax
        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n).unwrap();

        // Test finite difference for each element of B
        for kk in 0..k {
            for j in 0..n {
                // Perturb B[kk, j] (column-major: index = kk + j * k)
                let b_idx = kk + j * k;
                b[b_idx] += epsilon;

                // Recompute C with perturbed B
                let (c_perturbed, _) =
                    tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)
                        .unwrap();

                // Restore B
                b[b_idx] -= epsilon;

                // Check gradient for each C[i, j] (column-major: index = i + j * m)
                for i in 0..m {
                    let c_idx = i + j * m;
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

        // Batch of 2 matrices, each 2x2 (column-major)
        let a_batch = vec![
            vec![1.0f32, 3.0, 2.0, 4.0], // A[0]: [[1,2],[3,4]] col-major
            vec![5.0f32, 7.0, 6.0, 8.0], // A[1]: [[5,6],[7,8]] col-major
        ];
        let b_batch = vec![
            vec![1.0f32, 0.0, 0.0, 1.0], // B[0]: [[1,0],[0,1]] col-major (symmetric)
            vec![1.0f32, 3.0, 2.0, 4.0], // B[1]: [[1,2],[3,4]] col-major
        ];

        let c_batch =
            tropical_matmul_gpu_batched::<TropicalMaxPlus<f32>>(&a_batch, &b_batch, 2, 2, 2)
                .unwrap();

        assert_eq!(c_batch.len(), 2);

        // C[0] = A[0] * B[0] (MaxPlus)
        // C[0,0] = max(1+1, 2+0) = 2
        // C[1,0] = max(3+1, 4+0) = 4
        // C[0,1] = max(1+0, 2+1) = 3
        // C[1,1] = max(3+0, 4+1) = 5
        // Column-major result: [C[0,0], C[1,0], C[0,1], C[1,1]] = [2, 4, 3, 5]
        assert!((c_batch[0][0] - 2.0).abs() < 1e-5);
        assert!((c_batch[0][1] - 4.0).abs() < 1e-5);
        assert!((c_batch[0][2] - 3.0).abs() < 1e-5);
        assert!((c_batch[0][3] - 5.0).abs() < 1e-5);

        // C[1] = A[1] * B[1] (MaxPlus)
        // C[0,0] = max(5+1, 6+3) = 9
        // C[1,0] = max(7+1, 8+3) = 11
        // C[0,1] = max(5+2, 6+4) = 10
        // C[1,1] = max(7+2, 8+4) = 12
        // Column-major result: [C[0,0], C[1,0], C[0,1], C[1,1]] = [9, 11, 10, 12]
        assert!((c_batch[1][0] - 9.0).abs() < 1e-5);
        assert!((c_batch[1][1] - 11.0).abs() < 1e-5);
        assert!((c_batch[1][2] - 10.0).abs() < 1e-5);
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

        // 2 batches of 2x2 matrices, stored contiguously (column-major)
        let a = vec![
            // Batch 0: [[1,2],[3,4]] col-major: [1,3,2,4]
            1.0f32, 3.0, 2.0, 4.0,
            // Batch 1: [[5,6],[7,8]] col-major: [5,7,6,8]
            5.0, 7.0, 6.0, 8.0,
        ];
        let b = vec![
            // Batch 0: [[1,0],[0,1]] col-major: [1,0,0,1]
            1.0f32, 0.0, 0.0, 1.0,
            // Batch 1: [[1,2],[3,4]] col-major: [1,3,2,4]
            1.0, 3.0, 2.0, 4.0,
        ];

        let c = tropical_matmul_gpu_strided_batched::<TropicalMaxPlus<f32>>(&a, &b, 2, 2, 2, 2)
            .unwrap();

        // Should have 2 * 2 * 2 = 8 elements
        assert_eq!(c.len(), 8);

        // Batch 0 results (column-major: [C[0,0], C[1,0], C[0,1], C[1,1]] = [2, 4, 3, 5])
        assert!((c[0] - 2.0).abs() < 1e-5);
        assert!((c[1] - 4.0).abs() < 1e-5);
        assert!((c[2] - 3.0).abs() < 1e-5);
        assert!((c[3] - 5.0).abs() < 1e-5);

        // Batch 1 results (column-major: [C[0,0], C[1,0], C[0,1], C[1,1]] = [9, 11, 10, 12])
        assert!((c[4] - 9.0).abs() < 1e-5);
        assert!((c[5] - 11.0).abs() < 1e-5);
        assert!((c[6] - 10.0).abs() < 1e-5);
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

        // Batch of 2 matrices (column-major)
        // A[0]: 2x3 = [[1, 2, 3], [4, 5, 6]] col-major: [1, 4, 2, 5, 3, 6]
        // A[1]: 2x3 = [[6, 5, 4], [3, 2, 1]] col-major: [6, 3, 5, 2, 4, 1]
        let a_batch = vec![
            vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], // A[0]: 2x3 col-major
            vec![6.0f32, 3.0, 5.0, 2.0, 4.0, 1.0], // A[1]: 2x3 col-major (reversed)
        ];
        // B: 3x2 = [[1, 2], [3, 4], [5, 6]] col-major: [1, 3, 5, 2, 4, 6]
        let b_batch = vec![
            vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0], // B[0]: 3x2 col-major
            vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0], // B[1]: 3x2 col-major
        ];

        let results = tropical_matmul_gpu_batched_with_argmax::<TropicalMaxPlus<f32>>(
            &a_batch, &b_batch, 2, 3, 2,
        )
        .unwrap();

        assert_eq!(results.len(), 2);

        // Batch 0: C is column-major [C[0,0], C[1,0], C[0,1], C[1,1]]
        // C[0,0] = max(1+1, 2+3, 3+5) = 8, argmax=2
        let (c0, argmax0) = &results[0];
        assert!((c0[0] - 8.0).abs() < 1e-5);
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

        // A: 2x2 = [[1, 2], [3, 4]] col-major: [1, 3, 2, 4]
        let a_batch = vec![
            vec![1.0f32, 3.0, 2.0, 4.0], // 2x2 col-major
        ];
        // B: 2x2 = [[1, 2], [3, 4]] col-major: [1, 3, 2, 4]
        let b_batch = vec![
            vec![1.0f32, 3.0, 2.0, 4.0], // 2x2 col-major
        ];

        let c_batch =
            tropical_matmul_gpu_batched::<TropicalMinPlus<f32>>(&a_batch, &b_batch, 2, 2, 2)
                .unwrap();

        // MinPlus: C[i,j] = min_k(A[i,k] + B[k,j])
        // C[0,0] = min(1+1, 2+3) = min(2, 5) = 2
        // C[1,0] = min(3+1, 4+3) = min(4, 7) = 4
        // C[0,1] = min(1+2, 2+4) = min(3, 6) = 3
        // C[1,1] = min(3+2, 4+4) = min(5, 8) = 5
        // Column-major result: [2, 4, 3, 5]
        assert!((c_batch[0][0] - 2.0).abs() < 1e-5);
        assert!((c_batch[0][1] - 4.0).abs() < 1e-5);
        assert!((c_batch[0][2] - 3.0).abs() < 1e-5);
        assert!((c_batch[0][3] - 5.0).abs() < 1e-5);
    }

    // ========================================================================
    // Backward Pass tests
    // ========================================================================

    #[test]
    fn test_tropical_backward_a_gpu() {
        // Test backward pass for A (column-major)
        // C[i,j] = A[i,argmax[i,j]] + B[argmax[i,j],j]
        // dL/dA[i,k] = sum_j { dL/dC[i,j] if argmax[i,j] == k }

        let m = 2;
        let k = 3;
        let n = 2;

        // Gradient from upstream (all ones for simplicity), column-major
        let grad_c = vec![1.0f32; m * n];

        // Argmax: column-major, for each C[i,j] which k produced it
        // Logical matrix = [[0, 2], [1, 2]] in row notation
        // Column-major storage: [argmax[0,0], argmax[1,0], argmax[0,1], argmax[1,1]] = [0, 1, 2, 2]
        let argmax: Vec<ArgmaxIndex> = vec![0, 1, 2, 2];

        let grad_a = tropical_backward_a_gpu(&grad_c, &argmax, m, k, n);

        // Expected grad_a (2x3) column-major:
        // grad_a[0,0] = grad_c[0,0] because argmax[0,0]=0 -> 1.0
        // grad_a[1,0] = 0
        // grad_a[0,1] = 0
        // grad_a[1,1] = grad_c[1,0] because argmax[1,0]=1 -> 1.0
        // grad_a[0,2] = grad_c[0,1] because argmax[0,1]=2 -> 1.0
        // grad_a[1,2] = grad_c[1,1] because argmax[1,1]=2 -> 1.0
        // Column-major grad_a: [[1,0,1],[0,1,1]] -> [1, 0, 0, 1, 1, 1]
        assert_eq!(grad_a.len(), m * k);
        assert!((grad_a[0] - 1.0).abs() < 1e-5); // [0,0]
        assert!((grad_a[1] - 0.0).abs() < 1e-5); // [1,0]
        assert!((grad_a[2] - 0.0).abs() < 1e-5); // [0,1]
        assert!((grad_a[3] - 1.0).abs() < 1e-5); // [1,1]
        assert!((grad_a[4] - 1.0).abs() < 1e-5); // [0,2]
        assert!((grad_a[5] - 1.0).abs() < 1e-5); // [1,2]
    }

    #[test]
    fn test_tropical_backward_b_gpu() {
        // Test backward pass for B (column-major)
        // dL/dB[k,j] = sum_i { dL/dC[i,j] if argmax[i,j] == k }

        let m = 2;
        let k = 3;
        let n = 2;

        // Gradient from upstream (all ones), column-major
        let grad_c = vec![1.0f32; m * n];

        // Argmax: column-major [argmax[0,0], argmax[1,0], argmax[0,1], argmax[1,1]] = [0, 1, 2, 2]
        // Logical: [[0, 2], [1, 2]]
        let argmax: Vec<ArgmaxIndex> = vec![0, 1, 2, 2];

        let grad_b = tropical_backward_b_gpu(&grad_c, &argmax, m, k, n);

        // Expected grad_b (3x2) column-major:
        // grad_b[0,0] = grad_c[0,0] because argmax[0,0]=0 -> 1.0
        // grad_b[1,0] = grad_c[1,0] because argmax[1,0]=1 -> 1.0
        // grad_b[2,0] = 0
        // grad_b[0,1] = 0
        // grad_b[1,1] = 0
        // grad_b[2,1] = grad_c[0,1] + grad_c[1,1] because argmax[0,1]=2 and argmax[1,1]=2 -> 2.0
        // Column-major grad_b: [[1,0],[1,0],[0,2]] -> [1, 1, 0, 0, 0, 2]
        assert_eq!(grad_b.len(), k * n);
        assert!((grad_b[0] - 1.0).abs() < 1e-5); // [0,0]
        assert!((grad_b[1] - 1.0).abs() < 1e-5); // [1,0]
        assert!((grad_b[2] - 0.0).abs() < 1e-5); // [2,0]
        assert!((grad_b[3] - 0.0).abs() < 1e-5); // [0,1]
        assert!((grad_b[4] - 0.0).abs() < 1e-5); // [1,1]
        assert!((grad_b[5] - 2.0).abs() < 1e-5); // [2,1]
    }

    #[test]
    fn test_tropical_backward_gpu_integration() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // Full integration test: forward pass on GPU, backward pass (column-major)
        // A: 2x3 = [[1, 2, 3], [4, 5, 6]] col-major: [1, 4, 2, 5, 3, 6]
        let a = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        // B: 3x2 = [[1, 2], [3, 4], [5, 6]] col-major: [1, 3, 5, 2, 4, 6]
        let b = vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0];

        let m = 2;
        let k = 3;
        let n = 2;

        // Forward pass on GPU
        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n).unwrap();

        // Verify forward pass (C is column-major: [C[0,0], C[1,0], C[0,1], C[1,1]])
        assert!((c[0] - 8.0).abs() < 1e-5); // C[0,0] = max(1+1, 2+3, 3+5) = 8

        // Backward pass with unit gradients
        let grad_c = vec![1.0f32; m * n];
        let grad_a = tropical_backward_a_gpu(&grad_c, &argmax, m, k, n);
        let grad_b = tropical_backward_b_gpu(&grad_c, &argmax, m, k, n);

        // For this specific case, argmax should be all 2 (k=2 wins for all)
        // So grad_a[i,2] should be n (sum over j) and others 0
        // grad_a[0,2] = 2 (from C[0,0] and C[0,1])
        // grad_a[1,2] = 2 (from C[1,0] and C[1,1])
        // Column-major: A[0,2] at index 0+2*2=4, A[1,2] at index 1+2*2=5
        assert_eq!(grad_a.len(), m * k);
        assert!((grad_a[4] - 2.0).abs() < 1e-5); // A[0,2]
        assert!((grad_a[5] - 2.0).abs() < 1e-5); // A[1,2]

        // grad_b[2,j] = m (sum over i) for each j
        // Column-major: B[2,0] at index 2+0*3=2, B[2,1] at index 2+1*3=5
        assert_eq!(grad_b.len(), k * n);
        assert!((grad_b[2] - 2.0).abs() < 1e-5); // B[2,0]
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

        // Batch of 2 forward passes (column-major)
        // A: 2x3 = [[1, 2, 3], [4, 5, 6]] col-major: [1, 4, 2, 5, 3, 6]
        let a_batch = vec![
            vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], // A[0] col-major
            vec![6.0f32, 3.0, 5.0, 2.0, 4.0, 1.0], // A[1] col-major (reversed)
        ];
        // B: 3x2 = [[1, 2], [3, 4], [5, 6]] col-major: [1, 3, 5, 2, 4, 6]
        let b_batch = vec![
            vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0], // B[0] col-major
            vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0], // B[1] col-major
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
        let grad_a_batch = tropical_backward_a_gpu_batched(&grad_c_batch, &argmax_batch, m, k, n);
        let grad_b_batch = tropical_backward_b_gpu_batched(&grad_c_batch, &argmax_batch, m, k, n);

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

    // ========================================================================
    // Additional coverage tests
    // ========================================================================

    #[test]
    fn test_context_helper_functions() {
        // Test grid_dims_f32
        let (gx, gy, gz) = CudaContext::grid_dims_f32(128, 256);
        assert!(gx > 0);
        assert_eq!(gy, 1);
        assert_eq!(gz, 1);

        // Test grid_dims_f64
        let (gx, gy, gz) = CudaContext::grid_dims_f64(128, 256);
        assert!(gx > 0);
        assert_eq!(gy, 1);
        assert_eq!(gz, 1);

        // Test block_dims_f32
        let (bx, by, bz) = CudaContext::block_dims_f32();
        assert!(bx > 0);
        assert!(by > 0);
        assert_eq!(bz, 1);

        // Test block_dims_f64
        let (bx, by, bz) = CudaContext::block_dims_f64();
        assert!(bx > 0);
        assert!(by > 0);
        assert_eq!(bz, 1);
    }

    #[test]
    fn test_context_device_name() {
        if let Some(ctx) = cuda_context_or_skip() {
            let name = ctx.device_name();
            assert!(name.starts_with("CUDA Device"));
        }
    }

    #[test]
    fn test_context_get_kernel() {
        if let Some(ctx) = cuda_context_or_skip() {
            // Test getting a valid kernel
            let kernel = ctx.get_kernel("tropical_maxplus_f32_nn");
            assert!(kernel.is_ok());

            // Test getting an invalid kernel
            let kernel = ctx.get_kernel("nonexistent_kernel");
            assert!(kernel.is_err());
        }
    }

    #[test]
    fn test_tropical_matmul_gpu_dimension_mismatch_a() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // Wrong size for A
        let a = vec![1.0f32; 5]; // Should be m*k = 6
        let b = vec![1.0f32; 6];

        let result = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, CudaError::DimensionMismatch(_)));
    }

    #[test]
    fn test_tropical_matmul_gpu_dimension_mismatch_b() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        // Wrong size for B
        let a = vec![1.0f32; 6];
        let b = vec![1.0f32; 5]; // Should be k*n = 6

        let result = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, CudaError::DimensionMismatch(_)));
    }

    #[test]
    fn test_tropical_matmul_gpu_with_argmax_dimension_mismatch() {
        if cuda_context_or_skip().is_none() {
            return;
        }

        let a = vec![1.0f32; 5]; // Wrong size
        let b = vec![1.0f32; 6];

        let result = tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_tropical_gemm_gpu_dimension_mismatch() {
        if let Some(ctx) = cuda_context_or_skip() {
            // Create matrices with incompatible dimensions
            let a = GpuMatrix::from_host_row_major(&ctx, &vec![1.0f32; 6], 2, 3).unwrap();
            let b = GpuMatrix::from_host_row_major(&ctx, &vec![1.0f32; 6], 2, 3).unwrap(); // 2x3, not 3x2
            let mut c = GpuMatrix::alloc(&ctx, 2, 3).unwrap();

            let result = tropical_gemm_gpu::<TropicalMaxPlus<f32>>(&ctx, &a, &b, &mut c);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_tropical_matmul_gpu_with_ctx_dimension_mismatch() {
        if let Some(ctx) = cuda_context_or_skip() {
            let a = GpuMatrix::from_host_row_major(&ctx, &vec![1.0f32; 6], 2, 3).unwrap();
            let b = GpuMatrix::from_host_row_major(&ctx, &vec![1.0f32; 6], 2, 3).unwrap(); // Wrong dimensions

            let result = tropical_matmul_gpu_with_ctx::<TropicalMaxPlus<f32>>(&ctx, &a, &b);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_gpu_memory_dimension_mismatch() {
        if let Some(ctx) = cuda_context_or_skip() {
            // Try to create matrix with wrong data size
            let result = GpuMatrix::<f32>::from_host_row_major(&ctx, &vec![1.0f32; 5], 2, 3);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_gpu_memory_col_major_dimension_mismatch() {
        if let Some(ctx) = cuda_context_or_skip() {
            let result = GpuMatrix::<f32>::from_host_col_major(&ctx, &vec![1.0f32; 5], 2, 3);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_gpu_matrix_accessors() {
        if let Some(ctx) = cuda_context_or_skip() {
            let mat = GpuMatrix::from_host_row_major(&ctx, &vec![1.0f32; 6], 2, 3).unwrap();
            assert_eq!(mat.rows(), 2);
            assert_eq!(mat.cols(), 3);
            assert_eq!(mat.ld(), 2); // Column-major leading dimension = rows
        }
    }

    #[test]
    fn test_gpu_matrix_with_argmax_accessors() {
        if let Some(ctx) = cuda_context_or_skip() {
            let mat = GpuMatrixWithArgmax::<f32>::alloc(&ctx, 2, 3).unwrap();
            assert_eq!(mat.rows(), 2);
            assert_eq!(mat.cols(), 3);
        }
    }

    #[test]
    fn test_gpu_matrix_col_major_roundtrip() {
        if let Some(ctx) = cuda_context_or_skip() {
            let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
            let mat = GpuMatrix::from_host_col_major(&ctx, &data, 2, 3).unwrap();
            let back = mat.to_host_col_major(&ctx).unwrap();
            assert_eq!(data, back);
        }
    }

    #[test]
    fn test_tropical_matmul_gpu_maxmul() {
        use tropical_gemm::types::TropicalMaxMul;

        if cuda_context_or_skip().is_none() {
            return;
        }

        // 2x2 matrices with positive values (MaxMul uses multiplication)
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];

        let c = tropical_matmul_gpu::<TropicalMaxMul<f32>>(&a, 2, 2, &b, 2).unwrap();

        // C[0,0] = max(1*1, 2*3) = max(1, 6) = 6
        // C[0,1] = max(1*2, 2*4) = max(2, 8) = 8
        // C[1,0] = max(3*1, 4*3) = max(3, 12) = 12
        // C[1,1] = max(3*2, 4*4) = max(6, 16) = 16
        assert!((c[0] - 6.0).abs() < 1e-5);
        assert!((c[1] - 8.0).abs() < 1e-5);
        assert!((c[2] - 12.0).abs() < 1e-5);
        assert!((c[3] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_tropical_matmul_gpu_maxmul_with_argmax() {
        use tropical_gemm::types::TropicalMaxMul;

        if cuda_context_or_skip().is_none() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];

        let (c, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxMul<f32>>(&a, 2, 2, &b, 2).unwrap();

        // C[0,0] = max(1*1, 2*3) = 6, argmax=1
        assert!((c[0] - 6.0).abs() < 1e-5);
        assert_eq!(argmax[0], 1);
    }

    #[test]
    fn test_tropical_matmul_gpu_i32() {
        use tropical_gemm::types::TropicalMaxPlus;

        if cuda_context_or_skip().is_none() {
            return;
        }

        let a = vec![1i32, 2, 3, 4, 5, 6];
        let b = vec![1i32, 2, 3, 4, 5, 6];

        let c = tropical_matmul_gpu::<TropicalMaxPlus<i32>>(&a, 2, 3, &b, 2).unwrap();

        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert_eq!(c[0], 8);
        // C[1,1] = max(4+2, 5+4, 6+6) = 12
        assert_eq!(c[3], 12);
    }

    #[test]
    fn test_tropical_matmul_gpu_i64() {
        use tropical_gemm::types::TropicalMaxPlus;

        if cuda_context_or_skip().is_none() {
            return;
        }

        let a = vec![1i64, 2, 3, 4, 5, 6];
        let b = vec![1i64, 2, 3, 4, 5, 6];

        let c = tropical_matmul_gpu::<TropicalMaxPlus<i64>>(&a, 2, 3, &b, 2).unwrap();

        assert_eq!(c[0], 8);
        assert_eq!(c[3], 12);
    }

    #[test]
    fn test_error_display() {
        // Test error message formatting
        let err = CudaError::NoDevice;
        let msg = format!("{}", err);
        assert!(msg.contains("No CUDA device"));

        let err = CudaError::DimensionMismatch("test".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Dimension mismatch"));
        assert!(msg.contains("test"));

        let err = CudaError::KernelNotFound("my_kernel".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Kernel not found"));
        assert!(msg.contains("my_kernel"));
    }

    #[test]
    fn test_backward_empty_matrices() {
        // Test backward with empty inputs
        let grad_c: Vec<f32> = vec![];
        let argmax: Vec<ArgmaxIndex> = vec![];

        let grad_a = tropical_backward_a_gpu(&grad_c, &argmax, 0, 0, 0);
        assert!(grad_a.is_empty());

        let grad_b = tropical_backward_b_gpu(&grad_c, &argmax, 0, 0, 0);
        assert!(grad_b.is_empty());
    }

    #[test]
    fn test_backward_f64() {
        let m = 2;
        let k = 3;
        let n = 2;

        let grad_c = vec![1.0f64; m * n];
        let argmax: Vec<ArgmaxIndex> = vec![0, 2, 1, 2];

        let grad_a = tropical_backward_a_gpu::<f64>(&grad_c, &argmax, m, k, n);
        let grad_b = tropical_backward_b_gpu::<f64>(&grad_c, &argmax, m, k, n);

        assert_eq!(grad_a.len(), m * k);
        assert_eq!(grad_b.len(), k * n);
    }

    #[test]
    fn test_backward_batched_empty() {
        let grad_c_batch: Vec<Vec<f32>> = vec![];
        let argmax_batch: Vec<Vec<ArgmaxIndex>> = vec![];

        let grad_a = tropical_backward_a_gpu_batched(&grad_c_batch, &argmax_batch, 2, 3, 2);
        let grad_b = tropical_backward_b_gpu_batched(&grad_c_batch, &argmax_batch, 2, 3, 2);

        assert!(grad_a.is_empty());
        assert!(grad_b.is_empty());
    }
}
