//! Metal (Apple Silicon GPU) backend for tropical matrix multiplication.
//!
//! Mirrors the structure and public API of `tropical-gemm-cuda`. All matrices
//! are **column-major** (same convention as the CUDA crate's `GpuMatrix`).
//!
//! f64 is not supported: Metal has no `double` type. Use the CPU backend for f64.
//!
//! Unlike the CUDA crate there is no on-disk kernel cache: `newLibraryWithSource`
//! compiles the whole MSL source in well under a second and macOS keeps a
//! system-level shader cache, so `MetalContext::new()` is cheap on warm starts.
//!
//! ## Quick Start
//!
//! ```ignore
//! use tropical_gemm::prelude::*;
//! use tropical_gemm_metal::tropical_matmul_gpu;
//!
//! // 2×3 * 3×2 MaxPlus GEMM — column-major slices
//! let a = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]; // [[1,2,3],[4,5,6]]
//! let b = vec![1.0f32, 0.0, 2.0, 0.0, 1.0, 3.0]; // [[1,0],[0,1],[2,3]]
//! let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2).unwrap();
//! assert_eq!(c[0], 5.0); // max(1+1, 2+0, 3+2) = 5
//! ```
//!
//! For GEMM with argmax (the winning K-index for each output cell) use
//! [`tropical_matmul_gpu_with_argmax`].
//!
//! ## Integer Sentinel Contract
//!
//! Integer semirings (MaxPlus/MinPlus over i32/i64) require input values to
//! stay well within the sentinel magnitude. See [`SENTINEL_I32`] / [`SENTINEL_I64`]
//! and [`MAX_RELIABLE_DATA_I32`] / [`MAX_RELIABLE_DATA_I64`] for bounds.
//!
//! ## Differences from the CUDA Crate
//!
//! * Single-device only (no explicit device selection); uses the system default GPU.
//! * No separate kernel cache file; MSL is compiled once per process via
//!   `newLibraryWithSource` and macOS caches the result across launches.
//! * f64 / double is absent from MSL — not supported.

mod error;
pub use error::{MetalError, Result};

/// Integer tropical-zero sentinel magnitudes — same names, values and
/// data-range contract as the CUDA backend (`tropical-gemm-cuda`):
/// MaxPlus uses `-SENTINEL`, MinPlus `+SENTINEL`; keep `|data| < |S|/4`.
pub const SENTINEL_I32: i32 = 1_000_000_000;
pub const SENTINEL_I64: i64 = 1 << 60;
pub const MAX_RELIABLE_DATA_I32: i32 = SENTINEL_I32 / 4 - 1;
pub const MAX_RELIABLE_DATA_I64: i64 = SENTINEL_I64 / 4 - 1;

#[cfg(target_os = "macos")]
mod context;
#[cfg(target_os = "macos")]
pub use context::MetalContext;

#[cfg(target_os = "macos")]
mod memory;
#[cfg(target_os = "macos")]
pub use memory::{GpuMatrix, MetalScalar};

#[cfg(target_os = "macos")]
mod kernels;
#[cfg(target_os = "macos")]
pub use kernels::{tropical_gemm_gpu, tropical_gemm_gpu_with_argmax, MetalKernel, MetalKernelWithArgmax};

/// Lazily-initialized process-wide context (all Metal protocol objects used
/// here are Send + Sync — verified against objc2-metal 0.3.2 generated source).
#[cfg(target_os = "macos")]
pub fn get_global_context() -> Result<&'static MetalContext> {
    use std::sync::OnceLock;
    static CTX: OnceLock<MetalContext> = OnceLock::new();
    // Storing OnceLock<Result<...>> would freeze a transient failure forever;
    // constructing outside get_or_init keeps errors retryable. Trade-off: two
    // threads racing here may both run MetalContext::new() and the loser's
    // context is dropped (a harmless ARC release) — acceptable for a cheap,
    // deterministic constructor.
    if let Some(c) = CTX.get() {
        return Ok(c);
    }
    let ctx = MetalContext::new()?;
    Ok(CTX.get_or_init(|| ctx))
}

/// One-shot GEMM on the global context. `a`/`b` are **column-major**
/// (m×k and k×n); returns column-major m×n.
#[cfg(target_os = "macos")]
pub fn tropical_matmul_gpu<T>(
    a: &[T::Scalar],
    m: usize,
    k: usize,
    b: &[T::Scalar],
    n: usize,
) -> Result<Vec<T::Scalar>>
where
    T: MetalKernel,
    T::Scalar: MetalScalar,
{
    let ctx = get_global_context()?;
    tropical_matmul_gpu_with_ctx::<T>(ctx, a, m, k, b, n)
}

/// Same as [`tropical_matmul_gpu`] with an explicit context.
#[cfg(target_os = "macos")]
pub fn tropical_matmul_gpu_with_ctx<T>(
    ctx: &MetalContext,
    a: &[T::Scalar],
    m: usize,
    k: usize,
    b: &[T::Scalar],
    n: usize,
) -> Result<Vec<T::Scalar>>
where
    T: MetalKernel,
    T::Scalar: MetalScalar,
{
    let a = GpuMatrix::from_host(ctx, a, m, k)?;
    let b = GpuMatrix::from_host(ctx, b, k, n)?;
    let mut c = GpuMatrix::alloc(ctx, m, n)?;
    tropical_gemm_gpu::<T>(ctx, &a, &b, &mut c)?;
    Ok(c.to_host())
}

/// One-shot GEMM + argmax on the global context (column-major slices).
#[cfg(target_os = "macos")]
pub fn tropical_matmul_gpu_with_argmax<T>(
    a: &[T::Scalar],
    m: usize,
    k: usize,
    b: &[T::Scalar],
    n: usize,
) -> Result<(Vec<T::Scalar>, Vec<u32>)>
where
    T: MetalKernelWithArgmax,
    T::Scalar: MetalScalar,
{
    tropical_matmul_gpu_with_ctx_and_argmax::<T>(get_global_context()?, a, m, k, b, n)
}

/// Same as [`tropical_matmul_gpu_with_argmax`] with an explicit context.
#[cfg(target_os = "macos")]
pub fn tropical_matmul_gpu_with_ctx_and_argmax<T>(
    ctx: &MetalContext,
    a: &[T::Scalar],
    m: usize,
    k: usize,
    b: &[T::Scalar],
    n: usize,
) -> Result<(Vec<T::Scalar>, Vec<u32>)>
where
    T: MetalKernelWithArgmax,
    T::Scalar: MetalScalar,
{
    let a = GpuMatrix::from_host(ctx, a, m, k)?;
    let b = GpuMatrix::from_host(ctx, b, k, n)?;
    let mut c = GpuMatrix::alloc(ctx, m, n)?;
    let mut am = GpuMatrix::<u32>::alloc(ctx, m, n)?;
    tropical_gemm_gpu_with_argmax::<T>(ctx, &a, &b, &mut c, &mut am)?;
    Ok((c.to_host(), am.to_host()))
}
