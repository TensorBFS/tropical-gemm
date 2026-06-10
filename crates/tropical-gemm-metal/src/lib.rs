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

mod error;
pub use error::{MetalError, Result};

#[cfg(target_os = "macos")]
mod context;
#[cfg(target_os = "macos")]
pub use context::MetalContext;

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
