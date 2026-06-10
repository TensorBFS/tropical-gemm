#![cfg(target_os = "macos")]
use tropical_gemm::prelude::*;
use tropical_gemm::KernelDispatch; // NOT in the prelude
use tropical_gemm_metal::{
    tropical_gemm_gpu_with_argmax, GpuMatrix, MetalContext, MetalKernelWithArgmax, MetalScalar,
};

const SIZES: &[(usize, usize, usize)] =
    &[(1, 1, 1), (17, 5, 3), (64, 64, 64), (257, 130, 67)];

fn transpose<T: Copy + Default>(data: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut out = vec![T::default(); data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

// Same two traps as gemm_correctness's `check`: repeat the `MetalScalar` bound
// (trait where-clauses are not implied bounds) and unwrap the CPU newtype via
// value().
fn check_argmax<T>(m: usize, n: usize, k: usize, a_rm: Vec<T::Scalar>, b_rm: Vec<T::Scalar>)
where
    T: MetalKernelWithArgmax + KernelDispatch + TropicalWithArgmax<Index = u32>,
    T::Scalar: MetalScalar + PartialEq + std::fmt::Debug,
{
    let expected = tropical_matmul_with_argmax::<T>(&a_rm, m, k, &b_rm, n);
    // GemmWithArgmax.values is Vec<T> (newtype) — unwrap to scalars.
    let expected_values: Vec<T::Scalar> = expected.values.iter().map(|v| v.value()).collect();

    let ctx = MetalContext::new().unwrap();
    let a = GpuMatrix::from_host(&ctx, &transpose(&a_rm, m, k), m, k).unwrap();
    let b = GpuMatrix::from_host(&ctx, &transpose(&b_rm, k, n), k, n).unwrap();
    let mut c = GpuMatrix::alloc(&ctx, m, n).unwrap();
    let mut am = GpuMatrix::<u32>::alloc(&ctx, m, n).unwrap();
    tropical_gemm_gpu_with_argmax::<T>(&ctx, &a, &b, &mut c, &mut am).unwrap();

    // 值必须精确一致。索引在并列时两侧都取首个最优 k(已核实:CPU 的
    // tropical_add_argmax 用 >= 保留累积值即更早的 k,且 with_argmax 只走
    // portable 标量路径;GPU 用严格 COMPARE_OP 只在严格更优时更新 —— 语义
    // 等价),因此索引也必须逐元素一致。
    assert_eq!(c.to_host_row_major(), expected_values, "values {m}x{n}x{k}");
    assert_eq!(am.to_host_row_major(), expected.argmax, "argmax {m}x{n}x{k}");
}

fn f32_data(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i.wrapping_mul(2654435761).wrapping_add(salt.wrapping_mul(40503)) % 2000) as f32) * 0.01 - 10.0)
        .collect()
}

fn i32_data(len: usize, salt: usize) -> Vec<i32> {
    (0..len)
        .map(|i| (i.wrapping_mul(2654435761).wrapping_add(salt.wrapping_mul(40503)) % 2001) as i32 - 1000)
        .collect()
}

fn i64_data(len: usize, salt: usize) -> Vec<i64> {
    (0..len)
        .map(|i| (i.wrapping_mul(2654435761).wrapping_add(salt.wrapping_mul(40503)) % 2_000_001) as i64 - 1_000_000)
        .collect()
}

#[test]
fn argmax_f32_all_semirings() {
    for &(m, n, k) in SIZES {
        check_argmax::<TropicalMaxPlus<f32>>(m, n, k, f32_data(m * k, 1), f32_data(k * n, 2));
        check_argmax::<TropicalMinPlus<f32>>(m, n, k, f32_data(m * k, 3), f32_data(k * n, 4));
        check_argmax::<TropicalMaxMul<f32>>(m, n, k, f32_data(m * k, 5), f32_data(k * n, 6));
    }
}

#[test]
fn argmax_i32_i64_all_semirings() {
    for &(m, n, k) in SIZES {
        check_argmax::<TropicalMaxPlus<i32>>(m, n, k, i32_data(m * k, 1), i32_data(k * n, 2));
        check_argmax::<TropicalMinPlus<i32>>(m, n, k, i32_data(m * k, 3), i32_data(k * n, 4));
        check_argmax::<TropicalMaxMul<i32>>(m, n, k, i32_data(m * k, 5), i32_data(k * n, 6));
        check_argmax::<TropicalMaxPlus<i64>>(m, n, k, i64_data(m * k, 1), i64_data(k * n, 2));
        check_argmax::<TropicalMinPlus<i64>>(m, n, k, i64_data(m * k, 3), i64_data(k * n, 4));
        check_argmax::<TropicalMaxMul<i64>>(m, n, k, i64_data(m * k, 5), i64_data(k * n, 6));
    }
}

#[test]
fn argmax_ties_take_first_k() {
    // all-equal inputs: every k ties, argmax must be 0 everywhere
    let (m, n, k) = (8, 8, 16);
    let ctx = MetalContext::new().unwrap();
    let a = GpuMatrix::from_host(&ctx, &vec![1.0f32; m * k], m, k).unwrap();
    let b = GpuMatrix::from_host(&ctx, &vec![1.0f32; k * n], k, n).unwrap();
    let mut c = GpuMatrix::alloc(&ctx, m, n).unwrap();
    let mut am = GpuMatrix::<u32>::alloc(&ctx, m, n).unwrap();
    tropical_gemm_gpu_with_argmax::<TropicalMaxPlus<f32>>(&ctx, &a, &b, &mut c, &mut am).unwrap();
    assert!(am.to_host().iter().all(|&i| i == 0));
}

#[test]
fn argmax_i32_zero_cell_canonicalized_to_0() {
    // a row of tropical zeros -> drifted-zero cells get argmax reset to 0
    let (m, n, k) = (4, 4, 8);
    let ctx = MetalContext::new().unwrap();
    let a = GpuMatrix::from_host(
        &ctx,
        &vec![-tropical_gemm_metal::SENTINEL_I32; m * k],
        m,
        k,
    )
    .unwrap();
    let b_data: Vec<i32> = (0..k * n).map(|i| (i % 100) as i32).collect();
    let b = GpuMatrix::from_host(&ctx, &b_data, k, n).unwrap();
    let mut c = GpuMatrix::alloc(&ctx, m, n).unwrap();
    let mut am = GpuMatrix::<u32>::alloc(&ctx, m, n).unwrap();
    tropical_gemm_gpu_with_argmax::<TropicalMaxPlus<i32>>(&ctx, &a, &b, &mut c, &mut am).unwrap();
    assert!(am.to_host().iter().all(|&i| i == 0));
}
