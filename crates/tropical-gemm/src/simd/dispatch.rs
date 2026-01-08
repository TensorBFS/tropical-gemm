use super::detect::{simd_level, SimdLevel};
use super::kernels::*;
use crate::core::{tropical_gemm_inner, TilingParams, Transpose};
use crate::types::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring};

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
}

impl KernelDispatch for TropicalMaxPlus<f32> {
    unsafe fn dispatch_gemm(
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
    ) {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                let kernel = Avx2MaxPlusF32Kernel;
                let params = TilingParams::F32_AVX2;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => {
                let kernel = NeonMaxPlusF32Kernel;
                let params = TilingParams::new(128, 128, 256, 4, 4);
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            _ => {
                let kernel = PortableKernel;
                let params = TilingParams::PORTABLE;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
        }
    }
}

impl KernelDispatch for TropicalMaxPlus<f64> {
    unsafe fn dispatch_gemm(
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
    ) {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                let kernel = Avx2MaxPlusF64Kernel;
                let params = TilingParams::F64_AVX2;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => {
                let kernel = NeonMaxPlusF64Kernel;
                let params = TilingParams::new(64, 64, 128, 2, 2);
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            _ => {
                let kernel = PortableKernel;
                let params = TilingParams::PORTABLE;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
        }
    }
}

impl KernelDispatch for TropicalMinPlus<f32> {
    unsafe fn dispatch_gemm(
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
    ) {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                let kernel = Avx2MinPlusF32Kernel;
                let params = TilingParams::F32_AVX2;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => {
                let kernel = NeonMinPlusF32Kernel;
                let params = TilingParams::new(128, 128, 256, 4, 4);
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            _ => {
                let kernel = PortableKernel;
                let params = TilingParams::PORTABLE;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
        }
    }
}

impl KernelDispatch for TropicalMaxMul<f32> {
    unsafe fn dispatch_gemm(
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
    ) {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                let kernel = Avx2MaxMulF32Kernel;
                let params = TilingParams::F32_AVX2;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            _ => {
                let kernel = PortableKernel;
                let params = TilingParams::PORTABLE;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
        }
    }
}

// Fallback implementations for other types
macro_rules! impl_kernel_dispatch_portable {
    ($($t:ty),*) => {
        $(
            impl KernelDispatch for $t {
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
                    let kernel = PortableKernel;
                    let params = TilingParams::PORTABLE;
                    tropical_gemm_inner::<Self, _>(
                        m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                    );
                }
            }
        )*
    };
}

impl_kernel_dispatch_portable!(
    TropicalMinPlus<f64>,
    TropicalMaxMul<f64>,
    TropicalMaxPlus<i32>,
    TropicalMaxPlus<i64>,
    TropicalMinPlus<i32>,
    TropicalMinPlus<i64>,
    TropicalMaxMul<i32>,
    TropicalMaxMul<i64>
);

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
