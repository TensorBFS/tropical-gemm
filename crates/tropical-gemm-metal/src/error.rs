//! Error types for Metal operations.

use thiserror::Error;

/// Errors that can occur during Metal operations.
#[derive(Debug, Error)]
pub enum MetalError {
    /// No Metal device available.
    #[error("No Metal device available")]
    NoDevice,

    /// Failed to create command queue.
    #[error("Failed to create command queue")]
    CommandQueueCreation,

    /// Failed to create compute pipeline.
    #[error("Failed to create compute pipeline: {0}")]
    PipelineCreation(String),

    /// Failed to create buffer.
    #[error("Failed to create buffer")]
    BufferCreation,

    /// Dimension mismatch.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Kernel not found.
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// Shader compilation error.
    #[error("Shader compilation error: {0}")]
    ShaderCompilation(String),

    /// Command buffer error.
    #[error("Command buffer error: {0}")]
    CommandBuffer(String),
}

/// Result type for Metal operations.
pub type Result<T> = std::result::Result<T, MetalError>;
