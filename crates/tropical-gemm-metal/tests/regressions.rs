#![cfg(target_os = "macos")]
use tropical_gemm::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus};
use tropical_gemm_metal::{
    tropical_matmul_metal_with_argmax, GpuMatrix, MetalContext, MetalKernel, MetalKernelWithArgmax,
};
fn compare<S: MetalKernelWithArgmax<Scalar = f32>>() {
    for (m, k, n) in [(1, 1, 1), (3, 33, 5), (65, 35, 67)] {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 / 7.).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 17) as f32 / 9.).collect();
        let (c, idx) = tropical_matmul_metal_with_argmax::<S>(&a, m, k, &b, n).unwrap();
        for i in 0..m {
            for j in 0..n {
                let mut best = S::tropical_zero();
                let mut win = 0;
                for p in 0..k {
                    let v = S::from_scalar(a[i * k + p]).tropical_mul(S::from_scalar(b[p * n + j]));
                    let next = best.tropical_add(v);
                    if next != best {
                        win = p;
                        best = next;
                    }
                }
                assert!((c[i * n + j] - best.value()).abs() < 1e-6);
                assert_eq!(idx[i * n + j] as usize, win);
            }
        }
    }
}
#[test]
fn rectangular_argmax_all_semirings() {
    compare::<TropicalMaxPlus<f32>>();
    compare::<TropicalMinPlus<f32>>();
    compare::<TropicalMaxMul<f32>>();
}
#[test]
fn invalid_buffers_are_rejected() {
    let ctx = MetalContext::new().unwrap();
    let a = GpuMatrix::from_host_col_major(&ctx, &[1.0f32; 6], 2, 3).unwrap();
    let b = GpuMatrix::from_host_col_major(&ctx, &[1.0f32; 2], 1, 2).unwrap();
    let mut c = GpuMatrix::alloc(&ctx, 2, 2).unwrap();
    assert!(TropicalMaxPlus::<f32>::launch_gemm(&ctx, &a, &b, &mut c).is_err());
    assert!(GpuMatrix::<f32>::alloc(&ctx, usize::MAX, 2).is_err());
}
