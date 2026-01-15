//! Error types for Metal operations.

use thiserror::Error;

/// Errors that can occur during Metal operations.
#[derive(Debug, Error)]
pub enum MetalError {
    /// No Metal device available.
    #[error("No Metal device available")]
    NoDevice,

    /// Metal shader compilation error.
    #[error("Metal shader compilation error: {0}")]
    ShaderCompile(String),

    /// Metal library creation error.
    #[error("Metal library error: {0}")]
    Library(String),

    /// Kernel function not found.
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// Buffer creation error.
    #[error("Buffer creation error: {0}")]
    BufferCreation(String),

    /// Dimension mismatch.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Command buffer error.
    #[error("Command buffer error: {0}")]
    CommandBuffer(String),

    /// Execution error.
    #[error("Execution error: {0}")]
    Execution(String),
}

/// Result type for Metal operations.
pub type Result<T> = std::result::Result<T, MetalError>;
