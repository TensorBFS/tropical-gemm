use crate::core::{Microkernel, MicrokernelWithArgmax};
use crate::types::{TropicalSemiring, TropicalWithArgmax};

/// Portable (non-SIMD) microkernel using the `wide` crate.
///
/// This provides a fallback when no SIMD instructions are available,
/// but uses `wide` types which may still auto-vectorize.
#[derive(Default, Clone, Copy)]
pub struct PortableKernel;

impl<T: TropicalSemiring> Microkernel<T> for PortableKernel {
    const MR: usize = 4;
    const NR: usize = 4;

    unsafe fn execute(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        a: *const T::Scalar,
        b: *const T::Scalar,
        c: *mut T,
        ldc: usize,
    ) {
        // Delegate to the core portable implementation
        let core_kernel = crate::core::PortableMicrokernel;
        core_kernel.execute(mr, nr, k, a, b, c, ldc);
    }
}

impl<T: TropicalWithArgmax<Index = u32>> MicrokernelWithArgmax<T> for PortableKernel {
    unsafe fn execute_with_argmax(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        k_offset: usize,
        a: *const T::Scalar,
        b: *const T::Scalar,
        c: *mut T,
        argmax: *mut u32,
        ldc: usize,
    ) {
        let core_kernel = crate::core::PortableMicrokernel;
        core_kernel.execute_with_argmax(mr, nr, k, k_offset, a, b, c, argmax, ldc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TropicalMaxPlus;

    #[test]
    fn test_portable_kernel_execute() {
        let kernel = PortableKernel;
        let mr = 2;
        let nr = 2;
        let k = 2;

        let a: [f64; 8] = [1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0];
        let b: [f64; 8] = [1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0];
        let mut c = [TropicalMaxPlus::tropical_zero(); 4];
        let ldc = 2;

        unsafe {
            kernel.execute(mr, nr, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), ldc);
        }

        // C[0,0] = max(1+1, 3+3) = 6
        assert_eq!(c[0].0, 6.0);
    }

    #[test]
    fn test_portable_kernel_execute_with_argmax() {
        let kernel = PortableKernel;
        let mr = 2;
        let nr = 2;
        let k = 2;

        let a: [f64; 8] = [1.0, 2.0, 0.0, 0.0, 10.0, 20.0, 0.0, 0.0];
        let b: [f64; 8] = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let mut c = [TropicalMaxPlus::tropical_zero(); 4];
        let mut argmax = [0u32; 4];
        let ldc = 2;
        let k_offset = 0;

        unsafe {
            kernel.execute_with_argmax(
                mr,
                nr,
                k,
                k_offset,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                argmax.as_mut_ptr(),
                ldc,
            );
        }

        // C[0,0] = max(1+1, 10+1) = 11 at k=1
        assert_eq!(c[0].0, 11.0);
        assert_eq!(argmax[0], 1);
    }

    #[test]
    fn test_portable_kernel_default() {
        let kernel = PortableKernel::default();
        // Just verify it can be created and constants are accessible
        assert_eq!(<PortableKernel as Microkernel<TropicalMaxPlus<f64>>>::MR, 4);
        assert_eq!(<PortableKernel as Microkernel<TropicalMaxPlus<f64>>>::NR, 4);
        let _ = kernel;
    }
}
