#![cfg(target_os = "macos")]
use tropical_gemm::prelude::*;
use tropical_gemm_metal::{tropical_matmul_gpu, tropical_matmul_gpu_with_argmax};

#[test]
fn matmul_gpu_slice_api_col_major() {
    // 2x3 * 3x2 maxplus, column-major slices (same convention as the CUDA crate)
    let a = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]; // [[1,2,3],[4,5,6]] col-major
    let b = vec![1.0f32, 0.0, 2.0, 0.0, 1.0, 3.0]; // [[1,0],[0,1],[2,3]] col-major
    let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2).unwrap();
    // c[i,j] = max_k(a[i,k]+b[k,j]); spot-check c[0,0] = max(1+1, 2+0, 3+2) = 5
    assert_eq!(c[0], 5.0);
    assert_eq!(c.len(), 4);
}

#[test]
fn matmul_gpu_with_argmax_slice_api() {
    let a = vec![0.0f32; 4]; // 2x2 zeros
    let b = vec![1.0f32, 9.0, 1.0, 9.0]; // col-major: k=1 row dominates
    let (c, am) = tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, 2, 2, &b, 2).unwrap();
    assert_eq!(c, vec![9.0; 4]);
    assert_eq!(am, vec![1u32; 4]);
}

#[test]
fn dimension_mismatch_is_error() {
    let r = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&[0.0; 6], 2, 3, &[0.0; 5], 2);
    assert!(r.is_err());
}
