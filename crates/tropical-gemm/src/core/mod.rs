//! Core tropical GEMM algorithms using BLIS-style blocking.
//!
//! This module provides the portable implementation of tropical matrix
//! multiplication, optimized for cache efficiency using the BLIS framework.
//!
//! # BLIS Algorithm Overview
//!
//! The BLIS (BLAS-like Library Instantiation Software) approach achieves
//! near-optimal performance through **hierarchical cache blocking**:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ Loop 5: for jc in 0..N step NC    (partition columns of B)     │
//! │   Loop 4: for pc in 0..K step KC  (partition depth)            │
//! │     Pack B[pc:KC, jc:NC] → B̃  (fits in L3 cache)               │
//! │     Loop 3: for ic in 0..M step MC  (partition rows of A)      │
//! │       Pack A[ic:MC, pc:KC] → Ã  (fits in L2 cache)             │
//! │       Loop 2: for jr in 0..NC step NR  (register blocking)     │
//! │         Loop 1: for ir in 0..MC step MR  (microkernel)         │
//! │           microkernel(Ã[ir], B̃[jr], C[ic+ir, jc+jr])           │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Cache Tiling Parameters
//!
//! The [`TilingParams`] struct controls blocking sizes:
//!
//! | Parameter | Purpose | Typical Value (f32 AVX2) |
//! |-----------|---------|--------------------------|
//! | MC | Rows per L2 block | 256 |
//! | NC | Columns per L3 block | 256 |
//! | KC | Depth per block | 512 |
//! | MR | Microkernel rows | 8 |
//! | NR | Microkernel columns | 8 |
//!
//! # Packing
//!
//! Before computation, matrices are **packed** into contiguous buffers:
//!
//! - [`pack_a`]: Packs an `MC×KC` block of A for sequential microkernel access
//! - [`pack_b`]: Packs a `KC×NC` block of B for efficient broadcasting
//!
//! Packing eliminates TLB misses and enables SIMD vectorization.
//!
//! # Microkernel
//!
//! The innermost loop executes an `MR×NR` tile computation:
//!
//! ```text
//! for each k in packed_k:
//!     C[0:MR, 0:NR] = C ⊕ (A_col[0:MR] ⊗ B_row[0:NR])
//! ```
//!
//! The [`Microkernel`] trait abstracts over portable and SIMD implementations.
//!
//! # Module Contents
//!
//! - [`gemm`](gemm): The main GEMM algorithm with blocking loops
//! - [`kernel`](kernel): Microkernel trait and portable implementation
//! - [`packing`](packing): Matrix packing utilities
//! - [`tiling`](tiling): Cache tiling parameters and iterators
//! - [`argmax`](argmax): Argmax tracking for backpropagation

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
