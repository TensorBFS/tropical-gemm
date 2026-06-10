//! Error types for the Metal backend.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetalError {
    /// No Metal device available (`MTLCreateSystemDefaultDevice` returned nil).
    #[error("no Metal device found")]
    DeviceNotFound,

    /// MSL source failed to compile. Contains the full NSError description
    /// (includes source line numbers).
    #[error("Metal shader compilation failed: {0}")]
    Compile(String),

    /// Compute pipeline creation failed for a kernel function.
    #[error("compute pipeline creation failed for `{kernel}`: {message}")]
    Pipeline {
        kernel: &'static str,
        message: String,
    },

    /// A kernel name was not found in the compiled library.
    #[error("kernel `{0}` not found in Metal library")]
    KernelNotFound(&'static str),

    /// GPU buffer allocation failed.
    #[error("Metal buffer allocation failed ({bytes} bytes)")]
    Alloc { bytes: usize },

    /// Matrix dimensions don't match for the requested operation.
    #[error("dimension mismatch: A is {m}x{ka}, B is {kb}x{n}")]
    DimensionMismatch {
        m: usize,
        ka: usize,
        kb: usize,
        n: usize,
    },

    /// Host slice length doesn't match rows*cols.
    #[error("host data length {len} != rows*cols = {expected}")]
    HostLengthMismatch { len: usize, expected: usize },
}

pub type Result<T> = std::result::Result<T, MetalError>;
