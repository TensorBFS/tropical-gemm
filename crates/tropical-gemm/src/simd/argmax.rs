//! Runtime dispatch for argmax; integer and custom types retain the portable path.
use crate::core::{GemmWithArgmax, GemmWorkspace, Transpose};
use crate::types::TropicalWithArgmax;

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dispatch_f32<
    S: TropicalWithArgmax<Scalar = f32, Index = u32>,
    const MIN: bool,
    const MUL: bool,
>(
    m: usize,
    n: usize,
    k: usize,
    a: *const f32,
    lda: usize,
    trans_a: Transpose,
    b: *const f32,
    ldb: usize,
    trans_b: Transpose,
    result: &mut GemmWithArgmax<S>,
    workspace: &mut GemmWorkspace<S::Scalar>,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        let supported = is_x86_feature_detected!("avx2");
        #[cfg(target_arch = "aarch64")]
        let supported = true;
        if supported {
            use super::kernels::argmax_native::ArgmaxF32;
            use crate::core::{
                tropical_gemm_with_argmax_inner_with_workspace, Microkernel, TilingParams,
            };
            let kernel = ArgmaxF32::<MIN, MUL>;
            let nr = <ArgmaxF32<MIN, MUL> as Microkernel<S>>::NR;
            let params = TilingParams::new(128, 128, 256, 4, nr);
            tropical_gemm_with_argmax_inner_with_workspace(
                m, n, k, a, lda, trans_a, b, ldb, trans_b, result, &params, &kernel, workspace,
            );
            return;
        }
    }
    crate::core::tropical_gemm_with_argmax_inner_with_workspace(
        m,
        n,
        k,
        a,
        lda,
        trans_a,
        b,
        ldb,
        trans_b,
        result,
        &crate::core::TilingParams::PORTABLE,
        &crate::core::PortableMicrokernel,
        workspace,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dispatch_f64<
    S: TropicalWithArgmax<Scalar = f64, Index = u32>,
    const MIN: bool,
    const MUL: bool,
>(
    m: usize,
    n: usize,
    k: usize,
    a: *const f64,
    lda: usize,
    trans_a: Transpose,
    b: *const f64,
    ldb: usize,
    trans_b: Transpose,
    result: &mut GemmWithArgmax<S>,
    workspace: &mut GemmWorkspace<S::Scalar>,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        let supported = is_x86_feature_detected!("avx2");
        #[cfg(target_arch = "aarch64")]
        let supported = true;
        if supported {
            use super::kernels::argmax_native::ArgmaxF64;
            use crate::core::{
                tropical_gemm_with_argmax_inner_with_workspace, Microkernel, TilingParams,
            };
            let kernel = ArgmaxF64::<MIN, MUL>;
            let nr = <ArgmaxF64<MIN, MUL> as Microkernel<S>>::NR;
            let params = TilingParams::new(128, 128, 256, 4, nr);
            tropical_gemm_with_argmax_inner_with_workspace(
                m, n, k, a, lda, trans_a, b, ldb, trans_b, result, &params, &kernel, workspace,
            );
            return;
        }
    }
    crate::core::tropical_gemm_with_argmax_inner_with_workspace(
        m,
        n,
        k,
        a,
        lda,
        trans_a,
        b,
        ldb,
        trans_b,
        result,
        &crate::core::TilingParams::PORTABLE,
        &crate::core::PortableMicrokernel,
        workspace,
    );
}
