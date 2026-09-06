use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use tropical_gemm::{TropicalAndOr, TropicalBitwise};
use tropical_gemm_cuda::{get_global_context, CudaKernel};
fn check<S: CudaKernel>(value: impl Fn(usize) -> S::Scalar)
where
    S::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    let Ok(Ok(ctx)) = std::panic::catch_unwind(get_global_context) else {
        return;
    };
    let (batch, m, k, n) = (3, 5, 33, 7);
    let a: Vec<_> = (0..batch * m * k).map(&value).collect();
    let b: Vec<_> = (0..batch * k * n).map(|i| value(i + 19)).collect();
    let stream = ctx.stream();
    let da = stream.clone_htod(&a).unwrap();
    let db = stream.clone_htod(&b).unwrap();
    let mut dc = stream.alloc_zeros::<S::Scalar>(batch * m * n).unwrap();
    S::launch_gemm_batched(ctx, &da, &db, &mut dc, batch, m, k, n).unwrap();
    let got = stream.clone_dtoh(&dc).unwrap();
    stream.synchronize().unwrap();
    for z in 0..batch {
        for j in 0..n {
            for i in 0..m {
                let mut expected = S::tropical_zero();
                for p in 0..k {
                    expected = expected.tropical_add(
                        S::from_scalar(a[z * m * k + p * m + i])
                            .tropical_mul(S::from_scalar(b[z * k * n + j * k + p])),
                    );
                }
                assert_eq!(got[z * m * n + j * m + i], expected.value());
            }
        }
    }
}
#[test]
fn bool_and_bitwise_batched_match_cpu() {
    check::<TropicalAndOr>(|i| i % 7 == 0);
    check::<TropicalBitwise<u32>>(|i| 1u32.rotate_left((i % 32) as u32));
    check::<TropicalBitwise<u64>>(|i| 1u64.rotate_left((i % 64) as u32));
}

#[test]
fn direct_launch_rejects_invalid_buffers() {
    use tropical_gemm::TropicalMaxPlus;
    use tropical_gemm_cuda::{CudaKernelWithArgmax, GpuMatrix, GpuMatrixWithArgmax};
    let Ok(Ok(ctx)) = std::panic::catch_unwind(get_global_context) else {
        return;
    };
    let a = GpuMatrix::from_host(ctx, &[1f32; 6], 2, 3).unwrap();
    let b = GpuMatrix::from_host(ctx, &[2f32; 6], 3, 2).unwrap();
    let mut c = GpuMatrix::<f32>::alloc(ctx, 1, 1).unwrap();
    assert!(TropicalMaxPlus::<f32>::launch_gemm(ctx, &a, &b, &mut c).is_err());
    let mut c = GpuMatrix::<f32>::alloc(ctx, 2, 2).unwrap();
    *c.as_slice_mut() = ctx.stream().alloc_zeros::<f32>(1).unwrap();
    assert!(TropicalMaxPlus::<f32>::launch_gemm(ctx, &a, &b, &mut c).is_err());
    let mut c = GpuMatrixWithArgmax::<f32>::alloc(ctx, 2, 2).unwrap();
    c.argmax = GpuMatrix::alloc(ctx, 1, 1).unwrap();
    assert!(TropicalMaxPlus::<f32>::launch_gemm_with_argmax(ctx, &a, &b, &mut c).is_err());
}
