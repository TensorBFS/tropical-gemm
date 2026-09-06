use super::detect::{simd_level, SimdLevel};
use super::kernels::*;
use crate::core::{tropical_gemm_inner_with_workspace, GemmWorkspace, TilingParams, Transpose};
use crate::types::{
    TropicalAndOr, TropicalBitwise, TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus,
    TropicalSemiring,
};

/// Runtime-dispatched GEMM that selects the best kernel for the current CPU.
///
/// # Safety
/// Same requirements as `tropical_gemm_inner`
pub unsafe fn tropical_gemm_dispatch<T: TropicalSemiring + KernelDispatch>(
    m: usize,
    n: usize,
    k: usize,
    a: *const T::Scalar,
    lda: usize,
    trans_a: Transpose,
    b: *const T::Scalar,
    ldb: usize,
    trans_b: Transpose,
    c: *mut T,
    ldc: usize,
) {
    T::dispatch_gemm(m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc);
}

/// Runtime-dispatched GEMM with first-winner indices.
///
/// # Safety
/// Same pointer and output storage requirements as `tropical_gemm_with_argmax_portable`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn tropical_gemm_with_argmax_dispatch<
    T: KernelDispatch + crate::TropicalWithArgmax<Index = u32>,
>(
    m: usize,
    n: usize,
    k: usize,
    a: *const T::Scalar,
    lda: usize,
    trans_a: Transpose,
    b: *const T::Scalar,
    ldb: usize,
    trans_b: Transpose,
    result: &mut crate::core::GemmWithArgmax<T>,
) {
    T::dispatch_gemm_with_argmax(m, n, k, a, lda, trans_a, b, ldb, trans_b, result);
}

/// Trait for types that support kernel dispatch.
pub trait KernelDispatch: TropicalSemiring {
    /// Dispatch to the appropriate kernel based on CPU features.
    unsafe fn dispatch_gemm(
        m: usize,
        n: usize,
        k: usize,
        a: *const Self::Scalar,
        lda: usize,
        trans_a: Transpose,
        b: *const Self::Scalar,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
    );
    /// Dispatch argmax, defaulting to the portable kernel for custom types.
    ///
    /// # Safety
    /// Inputs and result must be valid for the requested dimensions and strides.
    #[allow(clippy::too_many_arguments)]
    unsafe fn dispatch_gemm_with_argmax(
        m: usize,
        n: usize,
        k: usize,
        a: *const Self::Scalar,
        lda: usize,
        trans_a: Transpose,
        b: *const Self::Scalar,
        ldb: usize,
        trans_b: Transpose,
        result: &mut crate::core::GemmWithArgmax<Self>,
    ) where
        Self: crate::TropicalWithArgmax<Index = u32>,
    {
        crate::core::tropical_gemm_with_argmax_portable(
            m, n, k, a, lda, trans_a, b, ldb, trans_b, result,
        );
    }
    /// Dispatch using reusable packing storage. Custom implementations retain
    /// their existing dispatch unless they override this method to use workspace.
    ///
    /// # Safety
    /// Same requirements as `dispatch_gemm`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn dispatch_gemm_with_workspace(
        m: usize,
        n: usize,
        k: usize,
        a: *const Self::Scalar,
        lda: usize,
        trans_a: Transpose,
        b: *const Self::Scalar,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
        _workspace: &mut GemmWorkspace<Self::Scalar>,
    ) {
        Self::dispatch_gemm(m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc);
    }

    /// Dispatch argmax using reusable packing storage.
    ///
    /// # Safety
    /// Same requirements as `dispatch_gemm_with_argmax`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn dispatch_gemm_with_argmax_with_workspace(
        m: usize,
        n: usize,
        k: usize,
        a: *const Self::Scalar,
        lda: usize,
        trans_a: Transpose,
        b: *const Self::Scalar,
        ldb: usize,
        trans_b: Transpose,
        result: &mut crate::core::GemmWithArgmax<Self>,
        workspace: &mut GemmWorkspace<Self::Scalar>,
    ) where
        Self: crate::TropicalWithArgmax<Index = u32>,
    {
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
            &TilingParams::PORTABLE,
            &crate::core::PortableMicrokernel,
            workspace,
        );
    }
}

// Preserve the existing dispatch entry point while allowing callers to opt in
// to packing reuse through the workspace-aware variant.
macro_rules! dispatch_with_local_workspace {
    () => {
        unsafe fn dispatch_gemm(
            m: usize,
            n: usize,
            k: usize,
            a: *const Self::Scalar,
            lda: usize,
            trans_a: Transpose,
            b: *const Self::Scalar,
            ldb: usize,
            trans_b: Transpose,
            c: *mut Self,
            ldc: usize,
        ) {
            Self::dispatch_gemm_with_workspace(
                m,
                n,
                k,
                a,
                lda,
                trans_a,
                b,
                ldb,
                trans_b,
                c,
                ldc,
                &mut GemmWorkspace::new(),
            );
        }
    };
}

// Each float semiring supplies its ordered comparison and product operation.
macro_rules! argmax_dispatch {
    ($scalar:ty, $dispatch:ident, $min:expr, $mul:expr) => {
        unsafe fn dispatch_gemm_with_argmax(
            m: usize,
            n: usize,
            k: usize,
            a: *const $scalar,
            lda: usize,
            trans_a: Transpose,
            b: *const $scalar,
            ldb: usize,
            trans_b: Transpose,
            result: &mut crate::core::GemmWithArgmax<Self>,
        ) {
            Self::dispatch_gemm_with_argmax_with_workspace(
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
                &mut GemmWorkspace::new(),
            );
        }
        unsafe fn dispatch_gemm_with_argmax_with_workspace(
            m: usize,
            n: usize,
            k: usize,
            a: *const $scalar,
            lda: usize,
            trans_a: Transpose,
            b: *const $scalar,
            ldb: usize,
            trans_b: Transpose,
            result: &mut crate::core::GemmWithArgmax<Self>,
            workspace: &mut GemmWorkspace<Self::Scalar>,
        ) {
            super::argmax::$dispatch::<Self, $min, $mul>(
                m, n, k, a, lda, trans_a, b, ldb, trans_b, result, workspace,
            );
        }
    };
}

impl KernelDispatch for TropicalMaxPlus<f32> {
    argmax_dispatch!(f32, dispatch_f32, false, false);
    dispatch_with_local_workspace!();
    unsafe fn dispatch_gemm_with_workspace(
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        lda: usize,
        trans_a: Transpose,
        b: *const f32,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
        workspace: &mut GemmWorkspace<Self::Scalar>,
    ) {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                let kernel = Avx2MaxPlusF32Kernel;
                let params = TilingParams::F32_AVX2;
                tropical_gemm_inner_with_workspace::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel, workspace,
                );
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => {
                let kernel = NeonMaxPlusF32Kernel;
                let params = TilingParams::new(128, 128, 256, 4, 4);
                tropical_gemm_inner_with_workspace::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel, workspace,
                );
            }
            _ => {
                let kernel = PortableKernel;
                let params = TilingParams::PORTABLE;
                tropical_gemm_inner_with_workspace::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel, workspace,
                );
            }
        }
    }
}

impl KernelDispatch for TropicalMaxPlus<f64> {
    argmax_dispatch!(f64, dispatch_f64, false, false);
    dispatch_with_local_workspace!();
    unsafe fn dispatch_gemm_with_workspace(
        m: usize,
        n: usize,
        k: usize,
        a: *const f64,
        lda: usize,
        trans_a: Transpose,
        b: *const f64,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
        workspace: &mut GemmWorkspace<Self::Scalar>,
    ) {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                let kernel = Avx2MaxPlusF64Kernel;
                let params = TilingParams::F64_AVX2;
                tropical_gemm_inner_with_workspace::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel, workspace,
                );
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => {
                let kernel = NeonMaxPlusF64Kernel;
                let params = TilingParams::new(64, 64, 128, 2, 2);
                tropical_gemm_inner_with_workspace::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel, workspace,
                );
            }
            _ => {
                let kernel = PortableKernel;
                let params = TilingParams::PORTABLE;
                tropical_gemm_inner_with_workspace::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel, workspace,
                );
            }
        }
    }
}

impl KernelDispatch for TropicalMinPlus<f32> {
    argmax_dispatch!(f32, dispatch_f32, true, false);
    dispatch_with_local_workspace!();
    unsafe fn dispatch_gemm_with_workspace(
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        lda: usize,
        trans_a: Transpose,
        b: *const f32,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
        workspace: &mut GemmWorkspace<Self::Scalar>,
    ) {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                let kernel = Avx2MinPlusF32Kernel;
                let params = TilingParams::F32_AVX2;
                tropical_gemm_inner_with_workspace::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel, workspace,
                );
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => {
                let kernel = NeonMinPlusF32Kernel;
                let params = TilingParams::new(128, 128, 256, 4, 4);
                tropical_gemm_inner_with_workspace::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel, workspace,
                );
            }
            _ => {
                let kernel = PortableKernel;
                let params = TilingParams::PORTABLE;
                tropical_gemm_inner_with_workspace::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel, workspace,
                );
            }
        }
    }
}

impl KernelDispatch for TropicalMaxMul<f32> {
    argmax_dispatch!(f32, dispatch_f32, false, true);
    dispatch_with_local_workspace!();
    unsafe fn dispatch_gemm_with_workspace(
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        lda: usize,
        trans_a: Transpose,
        b: *const f32,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
        workspace: &mut GemmWorkspace<Self::Scalar>,
    ) {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                let kernel = Avx2MaxMulF32Kernel;
                let params = TilingParams::F32_AVX2;
                tropical_gemm_inner_with_workspace::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel, workspace,
                );
            }
            _ => {
                let kernel = PortableKernel;
                let params = TilingParams::PORTABLE;
                tropical_gemm_inner_with_workspace::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel, workspace,
                );
            }
        }
    }
}

impl KernelDispatch for TropicalMinPlus<f64> {
    argmax_dispatch!(f64, dispatch_f64, true, false);
    dispatch_with_local_workspace!();
    unsafe fn dispatch_gemm_with_workspace(
        m: usize,
        n: usize,
        k: usize,
        a: *const f64,
        lda: usize,
        trans_a: Transpose,
        b: *const f64,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
        workspace: &mut GemmWorkspace<Self::Scalar>,
    ) {
        crate::core::tropical_gemm_inner_with_workspace(
            m,
            n,
            k,
            a,
            lda,
            trans_a,
            b,
            ldb,
            trans_b,
            c,
            ldc,
            &TilingParams::PORTABLE,
            &crate::core::PortableMicrokernel,
            workspace,
        );
    }
}

impl KernelDispatch for TropicalMaxMul<f64> {
    argmax_dispatch!(f64, dispatch_f64, false, true);
    dispatch_with_local_workspace!();
    unsafe fn dispatch_gemm_with_workspace(
        m: usize,
        n: usize,
        k: usize,
        a: *const f64,
        lda: usize,
        trans_a: Transpose,
        b: *const f64,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
        workspace: &mut GemmWorkspace<Self::Scalar>,
    ) {
        crate::core::tropical_gemm_inner_with_workspace(
            m,
            n,
            k,
            a,
            lda,
            trans_a,
            b,
            ldb,
            trans_b,
            c,
            ldc,
            &TilingParams::PORTABLE,
            &crate::core::PortableMicrokernel,
            workspace,
        );
    }
}

// Fallback implementations for other types
macro_rules! impl_kernel_dispatch_portable {
    ($($t:ty),*) => {
        $(
            impl KernelDispatch for $t {
                dispatch_with_local_workspace!();
                unsafe fn dispatch_gemm_with_workspace(
                    m: usize,
                    n: usize,
                    k: usize,
                    a: *const Self::Scalar,
                    lda: usize,
                    trans_a: Transpose,
                    b: *const Self::Scalar,
                    ldb: usize,
                    trans_b: Transpose,
                    c: *mut Self,
                    ldc: usize,
                    workspace: &mut GemmWorkspace<Self::Scalar>,
                ) {
                    let kernel = PortableKernel;
                    let params = TilingParams::PORTABLE;
                    tropical_gemm_inner_with_workspace::<Self, _>(
                        m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel, workspace,
                    );
                }
            }
        )*
    };
}

impl_kernel_dispatch_portable!(
    TropicalAndOr,
    TropicalMaxPlus<i32>,
    TropicalMaxPlus<i64>,
    TropicalMinPlus<i32>,
    TropicalMinPlus<i64>,
    TropicalMaxMul<i32>,
    TropicalMaxMul<i64>
);

impl KernelDispatch for TropicalBitwise<u32> {
    dispatch_with_local_workspace!();
    unsafe fn dispatch_gemm_with_workspace(
        m: usize,
        n: usize,
        k: usize,
        a: *const u32,
        lda: usize,
        trans_a: Transpose,
        b: *const u32,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
        workspace: &mut GemmWorkspace<u32>,
    ) {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            #[cfg(target_arch = "x86_64")]
            let supported = is_x86_feature_detected!("avx2");
            #[cfg(target_arch = "aarch64")]
            let supported = true;
            if supported {
                use super::kernels::bitwise_native::Bitwise32;
                use crate::core::Microkernel;
                let params = TilingParams::new(128, 128, 256, 4, Bitwise32::NR);
                tropical_gemm_inner_with_workspace(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &Bitwise32,
                    workspace,
                );
                return;
            }
        }
        tropical_gemm_inner_with_workspace(
            m,
            n,
            k,
            a,
            lda,
            trans_a,
            b,
            ldb,
            trans_b,
            c,
            ldc,
            &TilingParams::PORTABLE,
            &PortableKernel,
            workspace,
        );
    }
}

impl KernelDispatch for TropicalBitwise<u64> {
    dispatch_with_local_workspace!();
    unsafe fn dispatch_gemm_with_workspace(
        m: usize,
        n: usize,
        k: usize,
        a: *const u64,
        lda: usize,
        trans_a: Transpose,
        b: *const u64,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
        workspace: &mut GemmWorkspace<u64>,
    ) {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            #[cfg(target_arch = "x86_64")]
            let supported = is_x86_feature_detected!("avx2");
            #[cfg(target_arch = "aarch64")]
            let supported = true;
            if supported {
                use super::kernels::bitwise_native::Bitwise64;
                use crate::core::Microkernel;
                let params = TilingParams::new(128, 128, 256, 4, Bitwise64::NR);
                tropical_gemm_inner_with_workspace(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &Bitwise64,
                    workspace,
                );
                return;
            }
        }
        tropical_gemm_inner_with_workspace(
            m,
            n,
            k,
            a,
            lda,
            trans_a,
            b,
            ldb,
            trans_b,
            c,
            ldc,
            &TilingParams::PORTABLE,
            &PortableKernel,
            workspace,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test that the dispatch function exists and doesn't panic for small inputs
    #[test]
    fn test_dispatch_maxplus_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![TropicalMaxPlus::tropical_zero(); 4];

        unsafe {
            tropical_gemm_dispatch::<TropicalMaxPlus<f32>>(
                2,
                2,
                2,
                a.as_ptr(),
                2,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        // C[0,0] = max(A[0,0]+B[0,0], A[0,1]+B[1,0]) = max(1+1, 2+3) = 5
        assert_eq!(c[0].0, 5.0);
    }

    #[test]
    fn test_dispatch_maxplus_f64() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0];
        let mut c = vec![TropicalMaxPlus::tropical_zero(); 4];

        unsafe {
            tropical_gemm_dispatch::<TropicalMaxPlus<f64>>(
                2,
                2,
                2,
                a.as_ptr(),
                2,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c[0].0, 5.0);
    }

    #[test]
    fn test_dispatch_minplus_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![TropicalMinPlus::tropical_zero(); 4];

        unsafe {
            tropical_gemm_dispatch::<TropicalMinPlus<f32>>(
                2,
                2,
                2,
                a.as_ptr(),
                2,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        // C[0,0] = min(A[0,0]+B[0,0], A[0,1]+B[1,0]) = min(1+1, 2+3) = 2
        assert_eq!(c[0].0, 2.0);
    }

    #[test]
    fn test_dispatch_minplus_f64() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0];
        let mut c = vec![TropicalMinPlus::tropical_zero(); 4];

        unsafe {
            tropical_gemm_dispatch::<TropicalMinPlus<f64>>(
                2,
                2,
                2,
                a.as_ptr(),
                2,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c[0].0, 2.0);
    }

    #[test]
    fn test_dispatch_maxmul_f32() {
        let a = vec![2.0f32, 3.0, 4.0, 5.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![TropicalMaxMul::tropical_zero(); 4];

        unsafe {
            tropical_gemm_dispatch::<TropicalMaxMul<f32>>(
                2,
                2,
                2,
                a.as_ptr(),
                2,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        // C[0,0] = max(A[0,0]*B[0,0], A[0,1]*B[1,0]) = max(2*1, 3*3) = 9
        assert_eq!(c[0].0, 9.0);
    }

    #[test]
    fn test_dispatch_maxmul_f64() {
        let a = vec![2.0f64, 3.0, 4.0, 5.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0];
        let mut c = vec![TropicalMaxMul::tropical_zero(); 4];

        unsafe {
            tropical_gemm_dispatch::<TropicalMaxMul<f64>>(
                2,
                2,
                2,
                a.as_ptr(),
                2,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c[0].0, 9.0);
    }

    #[test]
    fn test_dispatch_maxplus_i32() {
        let a = vec![1i32, 2, 3, 4];
        let b = vec![1i32, 2, 3, 4];
        let mut c = vec![TropicalMaxPlus::tropical_zero(); 4];

        unsafe {
            tropical_gemm_dispatch::<TropicalMaxPlus<i32>>(
                2,
                2,
                2,
                a.as_ptr(),
                2,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c[0].0, 5);
    }

    #[test]
    fn test_dispatch_maxplus_i64() {
        let a = vec![1i64, 2, 3, 4];
        let b = vec![1i64, 2, 3, 4];
        let mut c = vec![TropicalMaxPlus::tropical_zero(); 4];

        unsafe {
            tropical_gemm_dispatch::<TropicalMaxPlus<i64>>(
                2,
                2,
                2,
                a.as_ptr(),
                2,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c[0].0, 5);
    }

    #[test]
    fn test_dispatch_minplus_i32() {
        let a = vec![1i32, 2, 3, 4];
        let b = vec![1i32, 2, 3, 4];
        let mut c = vec![TropicalMinPlus::tropical_zero(); 4];

        unsafe {
            tropical_gemm_dispatch::<TropicalMinPlus<i32>>(
                2,
                2,
                2,
                a.as_ptr(),
                2,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c[0].0, 2);
    }

    #[test]
    fn test_dispatch_minplus_i64() {
        let a = vec![1i64, 2, 3, 4];
        let b = vec![1i64, 2, 3, 4];
        let mut c = vec![TropicalMinPlus::tropical_zero(); 4];

        unsafe {
            tropical_gemm_dispatch::<TropicalMinPlus<i64>>(
                2,
                2,
                2,
                a.as_ptr(),
                2,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c[0].0, 2);
    }

    #[test]
    fn test_dispatch_maxmul_i32() {
        let a = vec![2i32, 3, 4, 5];
        let b = vec![1i32, 2, 3, 4];
        let mut c = vec![TropicalMaxMul::tropical_zero(); 4];

        unsafe {
            tropical_gemm_dispatch::<TropicalMaxMul<i32>>(
                2,
                2,
                2,
                a.as_ptr(),
                2,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c[0].0, 9);
    }

    #[test]
    fn test_dispatch_maxmul_i64() {
        let a = vec![2i64, 3, 4, 5];
        let b = vec![1i64, 2, 3, 4];
        let mut c = vec![TropicalMaxMul::tropical_zero(); 4];

        unsafe {
            tropical_gemm_dispatch::<TropicalMaxMul<i64>>(
                2,
                2,
                2,
                a.as_ptr(),
                2,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c[0].0, 9);
    }

    #[test]
    fn test_dispatch_larger_matrix() {
        // Test a larger matrix to exercise blocking
        let m = 16;
        let n = 16;
        let k = 16;

        let a: Vec<f32> = (0..m * k).map(|i| (i % 10) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 10) as f32).collect();
        let mut c = vec![TropicalMaxPlus::tropical_zero(); m * n];

        unsafe {
            tropical_gemm_dispatch::<TropicalMaxPlus<f32>>(
                m,
                n,
                k,
                a.as_ptr(),
                k,
                Transpose::NoTrans,
                b.as_ptr(),
                n,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                n,
            );
        }

        // Just verify no panic and result is not all zeros
        let has_non_zero = c.iter().any(|x| x.0 > f32::NEG_INFINITY);
        assert!(has_non_zero);
    }
}
