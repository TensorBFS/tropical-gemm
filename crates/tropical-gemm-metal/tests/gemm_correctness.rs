#![cfg(target_os = "macos")]
//! GPU kernels vs the CPU reference implementation.
//! CPU API is row-major; GPU is column-major — the helper transposes.

use tropical_gemm::prelude::*;
use tropical_gemm::KernelDispatch; // NOT in the prelude — must be imported from the root
use tropical_gemm_metal::{tropical_gemm_gpu, GpuMatrix, MetalContext, MetalKernel, MetalScalar};

/// Sizes covering full blocks, edge blocks, K remainders and degenerate cases.
pub const SIZES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (17, 5, 3),
    (64, 64, 64),
    (65, 33, 100),
    (257, 130, 67),
    (128, 128, 128),
];

fn transpose<T: Copy + Default>(data: &[T], rows: usize, cols: usize) -> Vec<T> {
    // row-major rows×cols  ->  column-major rows×cols
    let mut out = vec![T::default(); data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Run one semiring × one size on GPU (col-major) and CPU (row-major), compare.
///
/// Two traps verified during plan review:
/// * the trait-declaration `where Self::Scalar: MetalScalar` is NOT an implied
///   bound — every generic user must repeat `T::Scalar: MetalScalar` or the
///   `T: MetalKernel` bound itself fails to resolve (E0277);
/// * CPU `tropical_matmul::<T>` returns `Vec<T>` (the semiring NEWTYPE, not the
///   scalar) — unwrap with `TropicalSemiring::value()` before comparing.
fn check<T>(m: usize, n: usize, k: usize, a_rm: Vec<T::Scalar>, b_rm: Vec<T::Scalar>)
where
    T: MetalKernel + KernelDispatch,
    T::Scalar: MetalScalar + PartialEq + std::fmt::Debug,
{
    let expected_rm: Vec<T::Scalar> = tropical_matmul::<T>(&a_rm, m, k, &b_rm, n)
        .iter()
        .map(|v| v.value())
        .collect();

    let ctx = MetalContext::new().unwrap();
    let a = GpuMatrix::from_host(&ctx, &transpose(&a_rm, m, k), m, k).unwrap();
    let b = GpuMatrix::from_host(&ctx, &transpose(&b_rm, k, n), k, n).unwrap();
    let mut c = GpuMatrix::alloc(&ctx, m, n).unwrap();
    tropical_gemm_gpu::<T>(&ctx, &a, &b, &mut c).unwrap();

    assert_eq!(
        c.to_host_row_major(),
        expected_rm,
        "mismatch for {m}x{n}x{k}"
    );
}

fn f32_data(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let v = i
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503))
                % 2000;
            (v as f32) * 0.01 - 10.0
        })
        .collect()
}

#[test]
fn maxplus_f32_matches_cpu() {
    for &(m, n, k) in SIZES {
        check::<TropicalMaxPlus<f32>>(m, n, k, f32_data(m * k, 1), f32_data(k * n, 2));
    }
}

#[test]
fn minplus_f32_matches_cpu() {
    for &(m, n, k) in SIZES {
        check::<TropicalMinPlus<f32>>(m, n, k, f32_data(m * k, 3), f32_data(k * n, 4));
    }
}

#[test]
fn maxmul_f32_matches_cpu() {
    for &(m, n, k) in SIZES {
        check::<TropicalMaxMul<f32>>(m, n, k, f32_data(m * k, 5), f32_data(k * n, 6));
    }
}

fn i32_data(len: usize, salt: usize) -> Vec<i32> {
    (0..len)
        .map(|i| {
            (i.wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503))
                % 2001) as i32
                - 1000
        })
        .collect()
}

fn i64_data(len: usize, salt: usize) -> Vec<i64> {
    (0..len)
        .map(|i| {
            (i.wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503))
                % 2_000_001) as i64
                - 1_000_000
        })
        .collect()
}

#[test]
fn maxplus_minplus_maxmul_i32_match_cpu() {
    for &(m, n, k) in SIZES {
        check::<TropicalMaxPlus<i32>>(m, n, k, i32_data(m * k, 1), i32_data(k * n, 2));
        check::<TropicalMinPlus<i32>>(m, n, k, i32_data(m * k, 3), i32_data(k * n, 4));
        check::<TropicalMaxMul<i32>>(m, n, k, i32_data(m * k, 5), i32_data(k * n, 6));
    }
}

#[test]
fn maxplus_minplus_maxmul_i64_match_cpu() {
    for &(m, n, k) in SIZES {
        check::<TropicalMaxPlus<i64>>(m, n, k, i64_data(m * k, 1), i64_data(k * n, 2));
        check::<TropicalMinPlus<i64>>(m, n, k, i64_data(m * k, 3), i64_data(k * n, 4));
        check::<TropicalMaxMul<i64>>(m, n, k, i64_data(m * k, 5), i64_data(k * n, 6));
    }
}

#[test]
fn i32_sentinel_zero_is_detectable() {
    // A row of tropical zeros (-S) stays in sentinel territory after drift:
    // value <= -SENTINEL/2 detects it (same contract as the CUDA crate).
    let m = 4;
    let (n, k) = (4, 8);
    let a_rm = vec![-tropical_gemm_metal::SENTINEL_I32; m * k];
    let b_rm = i32_data(k * n, 7);
    let ctx = MetalContext::new().unwrap();
    let a = GpuMatrix::from_host_row_major(&ctx, &a_rm, m, k).unwrap();
    let b = GpuMatrix::from_host_row_major(&ctx, &b_rm, k, n).unwrap();
    let mut c = GpuMatrix::alloc(&ctx, m, n).unwrap();
    tropical_gemm_gpu::<TropicalMaxPlus<i32>>(&ctx, &a, &b, &mut c).unwrap();
    for v in c.to_host() {
        assert!(v <= -tropical_gemm_metal::SENTINEL_I32 / 2, "got {v}");
    }
}

#[test]
fn empty_dims_are_ok() {
    let ctx = MetalContext::new().unwrap();

    // m == 0: nothing to compute, output is the (empty) 0xN matrix
    let a = GpuMatrix::<f32>::alloc(&ctx, 0, 3).unwrap();
    let b = GpuMatrix::from_host(&ctx, &[1.0f32; 12], 3, 4).unwrap();
    let mut c = GpuMatrix::<f32>::alloc(&ctx, 0, 4).unwrap();
    tropical_gemm_gpu::<TropicalMaxPlus<f32>>(&ctx, &a, &b, &mut c).unwrap();
    assert!(c.to_host().is_empty());

    // k == 0: dispatch still runs; C is filled with the semiring zero (-inf)
    let a = GpuMatrix::<f32>::alloc(&ctx, 4, 0).unwrap();
    let b = GpuMatrix::<f32>::alloc(&ctx, 0, 5).unwrap();
    let mut c = GpuMatrix::from_host(&ctx, &[7.0f32; 20], 4, 5).unwrap();
    tropical_gemm_gpu::<TropicalMaxPlus<f32>>(&ctx, &a, &b, &mut c).unwrap();
    assert!(c.to_host().iter().all(|v| *v == f32::NEG_INFINITY));
}

#[test]
fn i32_minplus_sentinel_zero_is_detectable() {
    // MinPlus zero is +SENTINEL: a row of them keeps outputs >= S/2.
    let m = 4;
    let (n, k) = (4, 8);
    let a_rm = vec![tropical_gemm_metal::SENTINEL_I32; m * k];
    let b_rm = i32_data(k * n, 8);
    let ctx = MetalContext::new().unwrap();
    let a = GpuMatrix::from_host_row_major(&ctx, &a_rm, m, k).unwrap();
    let b = GpuMatrix::from_host_row_major(&ctx, &b_rm, k, n).unwrap();
    let mut c = GpuMatrix::alloc(&ctx, m, n).unwrap();
    tropical_gemm_gpu::<TropicalMinPlus<i32>>(&ctx, &a, &b, &mut c).unwrap();
    for v in c.to_host() {
        assert!(v >= tropical_gemm_metal::SENTINEL_I32 / 2, "got {v}");
    }
}
