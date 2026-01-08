//! SIMD-optimized microkernels for tropical GEMM.
//!
//! This module provides architecture-specific SIMD implementations
//! of the microkernel for tropical matrix multiplication.

mod detect;
pub mod dispatch;
pub mod kernels;

pub use detect::{simd_level, SimdLevel};
pub use dispatch::{tropical_gemm_dispatch, KernelDispatch};
pub use kernels::*;
