//! Metal backend for tropical matrix multiplication.
//!
//! This crate provides GPU-accelerated tropical GEMM operations using Metal
//! on Apple Silicon and other macOS GPUs.
//!
//! # Quick Start
//!
//! ```ignore
//! use tropical_gemm_metal::{tropical_matmul_metal, MetalContext};
//! use tropical_gemm::types::TropicalMaxPlus;
//!
//! // Simple one-shot API (uses cached global context for performance)
//! let a = vec![1.0f32; 1024 * 1024];
//! let b = vec![1.0f32; 1024 * 1024];
//! let c = tropical_matmul_metal::<TropicalMaxPlus<f32>>(&a, 1024, 1024, &b, 1024)?;
//! ```
//!
//! # Persistent Context
//!
//! For explicit context management:
//!
//! ```ignore
//! use tropical_gemm_metal::{MetalContext, GpuMatrix, tropical_gemm_metal};
//! use tropical_gemm::types::TropicalMaxPlus;
//!
//! let ctx = MetalContext::new()?;
//!
//! let a_gpu = GpuMatrix::from_host_row_major(&ctx, &a, m, k)?;
//! let b_gpu = GpuMatrix::from_host_row_major(&ctx, &b, k, n)?;
//! let mut c_gpu = GpuMatrix::alloc(&ctx, m, n)?;
//!
//! tropical_gemm_metal::<TropicalMaxPlus<f32>>(&ctx, &a_gpu, &b_gpu, &mut c_gpu)?;
//!
//! let c = c_gpu.to_host_row_major(&ctx)?;
//! ```
//!
//! # Performance
//!
//! The convenience functions (`tropical_matmul_metal`, etc.) use a lazily-initialized
//! global context that persists across calls. This avoids the shader compilation
//! overhead on each call.

mod context;
mod error;
mod gpu_mat;
mod kernels;

use once_cell::sync::OnceCell;
use std::sync::Mutex;

/// Global Metal context for convenience functions.
/// Lazily initialized on first use, persists for process lifetime.
static GLOBAL_CONTEXT: OnceCell<MetalContext> = OnceCell::new();

/// Mutex to ensure only one thread initializes the context.
static INIT_MUTEX: Mutex<()> = Mutex::new(());

/// Get or initialize the global Metal context.
///
/// This function is thread-safe and will only initialize the context once.
/// Subsequent calls return the cached context.
///
/// # Errors
///
/// Returns an error if Metal initialization fails (no device, etc.)
pub fn get_global_context() -> Result<&'static MetalContext> {
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
    let ctx = MetalContext::new()?;
    let _ = GLOBAL_CONTEXT.set(ctx);

    Ok(GLOBAL_CONTEXT.get().unwrap())
}

pub use context::MetalContext;
pub use error::{MetalError, Result};
pub use gpu_mat::{ArgmaxIndex, GpuMatrix, GpuMatrixWithArgmax};
pub use kernels::{MetalKernel, MetalKernelWithArgmax};

/// One-shot tropical matrix multiplication on Metal GPU.
///
/// This function handles all GPU memory management automatically.
/// For repeated operations, use `tropical_gemm_metal` with a persistent context.
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
/// use tropical_gemm_metal::tropical_matmul_metal;
/// use tropical_gemm::types::TropicalMaxPlus;
///
/// let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
/// let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
///
/// let c = tropical_matmul_metal::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2)?;
/// // c is 2x2, row-major
/// ```
pub fn tropical_matmul_metal<T>(
    a: &[T::Scalar],
    m: usize,
    k: usize,
    b: &[T::Scalar],
    n: usize,
) -> Result<Vec<T::Scalar>>
where
    T: MetalKernel,
    T::Scalar: Copy + Default,
{
    if a.len() != m * k {
        return Err(MetalError::DimensionMismatch(format!(
            "A: expected {} elements, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(MetalError::DimensionMismatch(format!(
            "B: expected {} elements, got {}",
            k * n,
            b.len()
        )));
    }

    // Use global cached context to avoid shader recompilation
    let ctx = get_global_context()?;

    let a_gpu = GpuMatrix::from_host_row_major(ctx, a, m, k)?;
    let b_gpu = GpuMatrix::from_host_row_major(ctx, b, k, n)?;
    let mut c_gpu = GpuMatrix::alloc(ctx, m, n)?;

    T::launch_gemm(ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

    c_gpu.to_host_row_major(ctx)
}

/// Tropical matrix multiplication with persistent context.
///
/// Use this function when performing multiple GPU operations to avoid
/// repeated context initialization and kernel compilation.
///
/// # Arguments
///
/// * `ctx` - Metal context
/// * `a` - Matrix A on GPU
/// * `b` - Matrix B on GPU
/// * `c` - Output matrix C on GPU (will be overwritten)
pub fn tropical_gemm_metal<T>(
    ctx: &MetalContext,
    a: &GpuMatrix<T::Scalar>,
    b: &GpuMatrix<T::Scalar>,
    c: &mut GpuMatrix<T::Scalar>,
) -> Result<()>
where
    T: MetalKernel,
    T::Scalar: Copy + Default,
{
    if a.cols() != b.rows() {
        return Err(MetalError::DimensionMismatch(format!(
            "A.cols ({}) != B.rows ({})",
            a.cols(),
            b.rows()
        )));
    }
    if c.rows() != a.rows() || c.cols() != b.cols() {
        return Err(MetalError::DimensionMismatch(format!(
            "C dimensions ({}, {}) don't match A×B ({}, {})",
            c.rows(),
            c.cols(),
            a.rows(),
            b.cols()
        )));
    }

    T::launch_gemm(ctx, a, b, c)
}

/// One-shot tropical matrix multiplication with argmax on Metal GPU.
///
/// This function handles all GPU memory management automatically.
/// Returns both the result matrix and argmax indices for backpropagation.
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
/// Tuple of (result matrix C, argmax indices) in row-major order
pub fn tropical_matmul_metal_with_argmax<T>(
    a: &[T::Scalar],
    m: usize,
    k: usize,
    b: &[T::Scalar],
    n: usize,
) -> Result<(Vec<T::Scalar>, Vec<ArgmaxIndex>)>
where
    T: MetalKernelWithArgmax,
    T::Scalar: Copy + Default,
{
    if a.len() != m * k {
        return Err(MetalError::DimensionMismatch(format!(
            "A: expected {} elements, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(MetalError::DimensionMismatch(format!(
            "B: expected {} elements, got {}",
            k * n,
            b.len()
        )));
    }

    let ctx = get_global_context()?;

    let a_gpu = GpuMatrix::from_host_row_major(ctx, a, m, k)?;
    let b_gpu = GpuMatrix::from_host_row_major(ctx, b, k, n)?;
    let mut c_gpu = GpuMatrixWithArgmax::alloc(ctx, m, n)?;

    T::launch_gemm_with_argmax(ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

    let c_data = c_gpu.matrix_to_host_row_major(ctx)?;
    let argmax_data = c_gpu.argmax_to_host_row_major(ctx)?;

    Ok((c_data, argmax_data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tropical_gemm::types::{TropicalMaxPlus, TropicalMinPlus, TropicalMaxMul};

    #[test]
    fn test_maxplus_basic() {
        let _ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("Metal not available, skipping test");
                return;
            }
        };

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2

        let c = tropical_matmul_metal::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2).unwrap();

        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert!((c[0] - 8.0).abs() < 1e-5);
        // C[0,1] = max(1+2, 2+4, 3+6) = 9
        assert!((c[1] - 9.0).abs() < 1e-5);
        // C[1,0] = max(4+1, 5+3, 6+5) = 11
        assert!((c[2] - 11.0).abs() < 1e-5);
        // C[1,1] = max(4+2, 5+4, 6+6) = 12
        assert!((c[3] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_minplus_basic() {
        let _ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("Metal not available, skipping test");
                return;
            }
        };

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2

        let c = tropical_matmul_metal::<TropicalMinPlus<f32>>(&a, 2, 3, &b, 2).unwrap();

        // C[0,0] = min(1+1, 2+3, 3+5) = 2
        assert!((c[0] - 2.0).abs() < 1e-5);
        // C[0,1] = min(1+2, 2+4, 3+6) = 3
        assert!((c[1] - 3.0).abs() < 1e-5);
        // C[1,0] = min(4+1, 5+3, 6+5) = 5
        assert!((c[2] - 5.0).abs() < 1e-5);
        // C[1,1] = min(4+2, 5+4, 6+6) = 6
        assert!((c[3] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_maxmul_basic() {
        let _ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("Metal not available, skipping test");
                return;
            }
        };

        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![2.0f32, 3.0, 4.0, 5.0]; // 2x2

        let c = tropical_matmul_metal::<TropicalMaxMul<f32>>(&a, 2, 2, &b, 2).unwrap();

        // C[0,0] = max(1*2, 2*4) = 8
        assert!((c[0] - 8.0).abs() < 1e-5);
        // C[0,1] = max(1*3, 2*5) = 10
        assert!((c[1] - 10.0).abs() < 1e-5);
        // C[1,0] = max(3*2, 4*4) = 16
        assert!((c[2] - 16.0).abs() < 1e-5);
        // C[1,1] = max(3*3, 4*5) = 20
        assert!((c[3] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_maxplus_with_argmax() {
        let _ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("Metal not available, skipping test");
                return;
            }
        };

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2

        let (c, argmax) = tropical_matmul_metal_with_argmax::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2).unwrap();

        // C[0,0] = max(1+1, 2+3, 3+5) = 8, argmax = 2
        assert!((c[0] - 8.0).abs() < 1e-5);
        assert_eq!(argmax[0], 2);

        // C[1,1] = max(4+2, 5+4, 6+6) = 12, argmax = 2
        assert!((c[3] - 12.0).abs() < 1e-5);
        assert_eq!(argmax[3], 2);
    }

    #[test]
    fn test_larger_matrix() {
        let _ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("Metal not available, skipping test");
                return;
            }
        };

        let m = 64;
        let k = 128;
        let n = 64;

        let a: Vec<f32> = (0..m*k).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k*n).map(|i| i as f32 * 0.01).collect();

        let c = tropical_matmul_metal::<TropicalMaxPlus<f32>>(&a, m, k, &b, n).unwrap();

        assert_eq!(c.len(), m * n);
    }

    #[test]
    fn test_device_name() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("Metal not available, skipping test");
                return;
            }
        };

        let name = ctx.device_name();
        println!("Metal device: {}", name);
        assert!(!name.is_empty());
    }
}
