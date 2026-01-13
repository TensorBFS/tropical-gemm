//! SIMD-optimized microkernels for tropical GEMM.
//!
//! This module provides architecture-specific SIMD implementations of the
//! microkernel, which is the innermost loop of the BLIS-style GEMM algorithm.
//!
//! # Supported Architectures
//!
//! | Architecture | Instruction Set | Register Width | Supported Types |
//! |--------------|-----------------|----------------|-----------------|
//! | x86_64 | AVX-512 | 512-bit | f32, f64 |
//! | x86_64 | AVX2 | 256-bit | f32, f64 |
//! | x86_64 | SSE4.1 | 128-bit | f32, f64 |
//! | aarch64 | NEON | 128-bit | f32 |
//! | Any | Portable | Scalar | All types |
//!
//! # Runtime Dispatch
//!
//! At runtime, [`tropical_gemm_dispatch`] selects the best kernel:
//!
//! ```rust,ignore
//! // Automatically uses AVX2 on supported CPUs
//! tropical_gemm_dispatch::<MaxPlus<f32>>(...);
//! ```
//!
//! The dispatch mechanism:
//! 1. [`simd_level()`] detects CPU features at runtime
//! 2. [`KernelDispatch`] trait routes to the appropriate implementation
//! 3. Falls back to portable kernel if no SIMD available
//!
//! # Microkernel Design
//!
//! For tropical MaxPlus f32 with AVX2 (8-wide vectors):
//!
//! ```text
//! // MR×NR = 8×8 output tile
//! for k in 0..KC:
//!     a_vec = load_8xf32(packed_a)     // 8 elements from A column
//!     for j in 0..8:
//!         b_scalar = broadcast(packed_b[j])  // 1 element from B row
//!         prod = a_vec + b_scalar            // tropical multiply
//!         c[j] = max(c[j], prod)             // tropical accumulate
//! ```
//!
//! # Module Contents
//!
//! - [`detect`](detect): CPU feature detection ([`SimdLevel`])
//! - [`dispatch`](dispatch): Runtime kernel selection ([`KernelDispatch`])
//! - [`kernels`](kernels): Architecture-specific microkernel implementations

mod detect;
pub mod dispatch;
pub mod kernels;

pub use detect::{simd_level, SimdLevel};
pub use dispatch::{tropical_gemm_dispatch, KernelDispatch};
pub use kernels::*;
