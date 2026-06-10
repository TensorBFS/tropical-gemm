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
