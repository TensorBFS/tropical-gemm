//! SIMD value-and-index accumulation with portable first-winner semantics.
use crate::core::{Microkernel, MicrokernelWithArgmax};
use crate::types::TropicalWithArgmax;
use std::arch::aarch64::*;

pub(crate) struct ArgmaxF32<const MIN: bool, const MUL: bool>;

impl<S: TropicalWithArgmax<Scalar = f32, Index = u32>, const MIN: bool, const MUL: bool>
    Microkernel<S> for ArgmaxF32<MIN, MUL>
{
    const MR: usize = 4;
    const NR: usize = 4;
    // This kernel is dispatched only for argmax. Keep the value-only trait
    // operation correct for the same packed layout as well.
    unsafe fn execute(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        a: *const f32,
        b: *const f32,
        c: *mut S,
        ldc: usize,
    ) {
        for i in 0..mr {
            for j in 0..nr {
                let out = c.add(i * ldc + j);
                for p in 0..k {
                    *out = (*out).tropical_add(
                        S::from_scalar(*a.add(p * 4 + i))
                            .tropical_mul(S::from_scalar(*b.add(p * 4 + j))),
                    );
                }
            }
        }
    }
}

impl<S: TropicalWithArgmax<Scalar = f32, Index = u32>, const MIN: bool, const MUL: bool>
    MicrokernelWithArgmax<S> for ArgmaxF32<MIN, MUL>
{
    unsafe fn execute_with_argmax(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        offset: usize,
        a: *const f32,
        b: *const f32,
        c: *mut S,
        argmax: *mut u32,
        ldc: usize,
    ) {
        let mut acc = [vdupq_n_f32(0.0); 4];
        let mut indices = [vdupq_n_u32(0); 4];
        for i in 0..mr {
            let mut values = [0.0f32; 4];
            let mut idx = [0u32; 4];
            for j in 0..nr {
                values[j] = (*c.add(i * ldc + j)).value();
                idx[j] = *argmax.add(i * ldc + j);
            }
            acc[i] = vld1q_f32(values.as_ptr());
            indices[i] = vld1q_u32(idx.as_ptr());
        }
        for p in 0..k {
            // B is packed and padded to NR even at the right edge.
            let bv = vld1q_f32(b.add(p * 4));
            let current = vdupq_n_u32((offset + p) as u32 as _);
            for i in 0..mr {
                let av = vdupq_n_f32(*a.add(p * 4 + i));
                let product = if MUL {
                    vmulq_f32(av, bv)
                } else {
                    vaddq_f32(av, bv)
                };
                // Keep on equality, including signed zeros. An unordered
                // comparison selects the candidate, matching the scalar API.
                let keep = if MIN {
                    vcleq_f32(acc[i], product)
                } else {
                    vcgeq_f32(acc[i], product)
                };
                acc[i] = vbslq_f32(keep, acc[i], product);
                indices[i] = vbslq_u32(keep, indices[i], current);
            }
        }
        for i in 0..mr {
            let mut values = [0.0f32; 4];
            let mut idx = [0u32; 4];
            vst1q_f32(values.as_mut_ptr(), acc[i]);
            vst1q_u32(idx.as_mut_ptr(), indices[i]);
            // Only store logical cells; output rows need not have SIMD padding.
            for j in 0..nr {
                *c.add(i * ldc + j) = S::from_scalar(values[j]);
                *argmax.add(i * ldc + j) = idx[j];
            }
        }
    }
}

pub(crate) struct ArgmaxF64<const MIN: bool, const MUL: bool>;

impl<S: TropicalWithArgmax<Scalar = f64, Index = u32>, const MIN: bool, const MUL: bool>
    Microkernel<S> for ArgmaxF64<MIN, MUL>
{
    const MR: usize = 4;
    const NR: usize = 2;
    // This kernel is dispatched only for argmax. Keep the value-only trait
    // operation correct for the same packed layout as well.
    unsafe fn execute(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        a: *const f64,
        b: *const f64,
        c: *mut S,
        ldc: usize,
    ) {
        for i in 0..mr {
            for j in 0..nr {
                let out = c.add(i * ldc + j);
                for p in 0..k {
                    *out = (*out).tropical_add(
                        S::from_scalar(*a.add(p * 4 + i))
                            .tropical_mul(S::from_scalar(*b.add(p * 2 + j))),
                    );
                }
            }
        }
    }
}

impl<S: TropicalWithArgmax<Scalar = f64, Index = u32>, const MIN: bool, const MUL: bool>
    MicrokernelWithArgmax<S> for ArgmaxF64<MIN, MUL>
{
    unsafe fn execute_with_argmax(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        offset: usize,
        a: *const f64,
        b: *const f64,
        c: *mut S,
        argmax: *mut u32,
        ldc: usize,
    ) {
        let mut acc = [vdupq_n_f64(0.0); 4];
        let mut indices = [vdupq_n_u64(0); 4];
        for i in 0..mr {
            let mut values = [0.0f64; 2];
            let mut idx = [0u64; 2];
            for j in 0..nr {
                values[j] = (*c.add(i * ldc + j)).value();
                idx[j] = *argmax.add(i * ldc + j) as u64;
            }
            acc[i] = vld1q_f64(values.as_ptr());
            indices[i] = vld1q_u64(idx.as_ptr());
        }
        for p in 0..k {
            // B is packed and padded to NR even at the right edge.
            let bv = vld1q_f64(b.add(p * 2));
            let current = vdupq_n_u64((offset + p) as u32 as _);
            for i in 0..mr {
                let av = vdupq_n_f64(*a.add(p * 4 + i));
                let product = if MUL {
                    vmulq_f64(av, bv)
                } else {
                    vaddq_f64(av, bv)
                };
                // Keep on equality, including signed zeros. An unordered
                // comparison selects the candidate, matching the scalar API.
                let keep = if MIN {
                    vcleq_f64(acc[i], product)
                } else {
                    vcgeq_f64(acc[i], product)
                };
                acc[i] = vbslq_f64(keep, acc[i], product);
                indices[i] = vbslq_u64(keep, indices[i], current);
            }
        }
        for i in 0..mr {
            let mut values = [0.0f64; 2];
            let mut idx = [0u64; 2];
            vst1q_f64(values.as_mut_ptr(), acc[i]);
            vst1q_u64(idx.as_mut_ptr(), indices[i]);
            // Only store logical cells; output rows need not have SIMD padding.
            for j in 0..nr {
                *c.add(i * ldc + j) = S::from_scalar(values[j]);
                *argmax.add(i * ldc + j) = idx[j] as u32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TropicalMaxPlus;
    #[test]
    fn indices_preserve_high_u32_bits() {
        {
            let a: [f32; 8] = [1., 1., 1., 1., 2., 2., 2., 2.];
            let b = [1.; 8];
            let mut c = [TropicalMaxPlus(<f32>::NEG_INFINITY); 16];
            let mut indices = [0; 16];
            unsafe {
                ArgmaxF32::<false, false>.execute_with_argmax(
                    4,
                    4,
                    2,
                    0x8000_0011,
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_mut_ptr(),
                    indices.as_mut_ptr(),
                    4,
                );
            }
            assert!(c.iter().all(|v| v.0 == 3.));
            assert!(indices.iter().all(|&p| p == 0x8000_0012));
        }
        {
            let a: [f64; 8] = [1., 1., 1., 1., 2., 2., 2., 2.];
            let b = [1.; 4];
            let mut c = [TropicalMaxPlus(<f64>::NEG_INFINITY); 8];
            let mut indices = [0; 8];
            unsafe {
                ArgmaxF64::<false, false>.execute_with_argmax(
                    4,
                    2,
                    2,
                    0x8000_0011,
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_mut_ptr(),
                    indices.as_mut_ptr(),
                    2,
                );
            }
            assert!(c.iter().all(|v| v.0 == 3.));
            assert!(indices.iter().all(|&p| p == 0x8000_0012));
        }
    }
}
