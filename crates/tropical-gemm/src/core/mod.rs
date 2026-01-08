//! Core tropical GEMM algorithms.
//!
//! This module provides the portable implementation of tropical
//! matrix multiplication using BLIS-style blocking for cache efficiency.

mod argmax;
mod gemm;
mod kernel;
mod packing;
mod tiling;

pub use argmax::GemmWithArgmax;
pub use gemm::{
    tropical_gemm_inner, tropical_gemm_portable, tropical_gemm_with_argmax_inner,
    tropical_gemm_with_argmax_portable,
};
pub use kernel::{Microkernel, MicrokernelWithArgmax, PortableMicrokernel};
pub use packing::{pack_a, pack_b, packed_a_size, packed_b_size, Layout, Transpose};
pub use tiling::{BlockIterator, TilingParams};
