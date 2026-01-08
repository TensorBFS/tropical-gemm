//! High-performance tropical matrix multiplication.
//!
//! This library provides BLAS-level performance for tropical matrix
//! multiplication across multiple semiring types.
//!
//! # GPU Acceleration
//!
//! For GPU-accelerated operations, add the `tropical-gemm-cuda` crate:
//!
//! ```toml
//! [dependencies]
//! tropical-gemm = "0.1"
//! tropical-gemm-cuda = "0.1"
//! ```
//!
//! Then use the GPU API:
//!
//! ```ignore
//! use tropical_gemm::TropicalMaxPlus;
//! use tropical_gemm_cuda::{tropical_matmul_gpu, CudaContext};
//!
//! let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)?;
//! ```
//!
//! # Tropical Semirings
//!
//! Tropical algebra replaces standard arithmetic operations:
//! - Standard addition → tropical addition (typically max or min)
//! - Standard multiplication → tropical multiplication (typically + or ×)
//!
//! | Type | ⊕ (add) | ⊗ (mul) | Zero | One | Use Case |
//! |------|---------|---------|------|-----|----------|
//! | [`TropicalMaxPlus<T>`] | max | + | -∞ | 0 | Viterbi, longest path |
//! | [`TropicalMinPlus<T>`] | min | + | +∞ | 0 | Shortest path |
//! | [`TropicalMaxMul<T>`] | max | × | 0 | 1 | Probability (non-log) |
//! | [`TropicalAndOr`] | OR | AND | false | true | Graph reachability |
//! | [`CountingTropical<T,C>`] | max+count | +,× | (-∞,0) | (0,1) | Path counting |
//!
//! # Quick Start
//!
//! ## Function-based API
//!
//! ```
//! use tropical_gemm::{tropical_matmul, TropicalMaxPlus, TropicalSemiring};
//!
//! // Create 2x3 and 3x2 matrices
//! let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//!
//! // Compute C = A ⊗ B using TropicalMaxPlus semiring
//! let c = tropical_matmul::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2);
//!
//! // C[i,j] = max_k(A[i,k] + B[k,j])
//! assert_eq!(c[0].value(), 8.0); // max(1+1, 2+3, 3+5) = 8
//! ```
//!
//! ## Matrix-based API (faer-style)
//!
//! ```
//! use tropical_gemm::{Mat, MatRef, MaxPlus, TropicalSemiring};
//!
//! // Create matrix views from raw data
//! let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//!
//! let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
//! let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);
//!
//! // Matrix multiplication using operators
//! let c = &a * &b;
//! assert_eq!(c[(0, 0)].value(), 8.0);
//!
//! // Or using methods
//! let c = a.matmul(&b);
//!
//! // Factory methods
//! let zeros = Mat::<MaxPlus<f32>>::zeros(3, 3);
//! let identity = Mat::<MaxPlus<f32>>::identity(3);
//! ```
//!
//! # Argmax Tracking (Backpropagation)
//!
//! For gradient routing in neural networks, you can track which k index
//! produced each optimal value:
//!
//! ```
//! use tropical_gemm::{tropical_matmul_with_argmax, TropicalMaxPlus, TropicalSemiring};
//!
//! let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
//!
//! let result = tropical_matmul_with_argmax::<TropicalMaxPlus<f64>>(&a, 2, 3, &b, 2);
//!
//! // Get the optimal value and which k produced it
//! let value = result.get(0, 0).value(); // 8.0
//! let k_idx = result.get_argmax(0, 0);  // 2 (k=2 gave max)
//! ```
//!
//! # Performance
//!
//! The library uses:
//! - BLIS-style cache blocking for memory efficiency
//! - Runtime CPU feature detection for optimal SIMD kernels
//! - AVX2/AVX-512 on x86-64, NEON on ARM
//!
//! ```
//! use tropical_gemm::Backend;
//!
//! println!("Using: {}", Backend::description());
//! ```
//!
//! # BLAS-style API
//!
//! For fine-grained control:
//!
//! ```
//! use tropical_gemm::{TropicalGemm, TropicalMaxPlus, TropicalSemiring};
//!
//! let a = vec![1.0f32; 64 * 64];
//! let b = vec![1.0f32; 64 * 64];
//! let mut c = vec![TropicalMaxPlus::tropical_zero(); 64 * 64];
//!
//! TropicalGemm::<TropicalMaxPlus<f32>>::new(64, 64, 64)
//!     .execute(&a, 64, &b, 64, &mut c, 64);
//! ```

// Internal modules
pub mod core;
pub mod mat;
pub mod simd;
pub mod types;

mod api;
mod backend;

// Public API
pub use api::{
    tropical_backward_a, tropical_backward_a_batched, tropical_backward_b,
    tropical_backward_b_batched, tropical_gemm, tropical_matmul, tropical_matmul_batched,
    tropical_matmul_batched_with_argmax, tropical_matmul_strided_batched,
    tropical_matmul_with_argmax, TropicalGemm,
};
pub use backend::{version_info, Backend};

// Re-export commonly used types at crate root
pub use core::{GemmWithArgmax, Layout, Transpose};
pub use mat::{Mat, MatMut, MatRef, MatWithArgmax};
pub use simd::{simd_level, KernelDispatch, SimdLevel};
pub use types::{
    CountingTropical, SimdTropical, TropicalAndOr, TropicalMaxMul, TropicalMaxPlus,
    TropicalMinPlus, TropicalScalar, TropicalSemiring, TropicalWithArgmax,
};

// Convenient type aliases
/// Alias for [`TropicalMaxPlus`].
pub type MaxPlus<T> = TropicalMaxPlus<T>;
/// Alias for [`TropicalMinPlus`].
pub type MinPlus<T> = TropicalMinPlus<T>;
/// Alias for [`TropicalMaxMul`].
pub type MaxMul<T> = TropicalMaxMul<T>;
/// Alias for [`TropicalAndOr`].
pub type AndOr = TropicalAndOr;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use super::{
        tropical_backward_a, tropical_backward_a_batched, tropical_backward_b,
        tropical_backward_b_batched, tropical_matmul, tropical_matmul_batched,
        tropical_matmul_batched_with_argmax, tropical_matmul_strided_batched,
        tropical_matmul_with_argmax, AndOr, Backend, CountingTropical, GemmWithArgmax, Mat,
        MatMut, MatRef, MatWithArgmax, MaxMul, MaxPlus, MinPlus, Transpose, TropicalAndOr,
        TropicalGemm, TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring,
        TropicalWithArgmax,
    };
}
