//! SIMD value-and-index accumulation with portable first-winner semantics.
use crate::core::{Microkernel, MicrokernelWithArgmax};
use crate::types::TropicalWithArgmax;
use std::arch::x86_64::*;

pub(crate) struct ArgmaxF32<const MIN: bool, const MUL: bool>;

impl<S: TropicalWithArgmax<Scalar = f32, Index = u32>, const MIN: bool, const MUL: bool>
    Microkernel<S> for ArgmaxF32<MIN, MUL>
{
    const MR: usize = 4;
    const NR: usize = 8;
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
                            .tropical_mul(S::from_scalar(*b.add(p * 8 + j))),
                    );
                }
            }
        }
    }
}

impl<S: TropicalWithArgmax<Scalar = f32, Index = u32>, const MIN: bool, const MUL: bool>
    MicrokernelWithArgmax<S> for ArgmaxF32<MIN, MUL>
{
    #[target_feature(enable = "avx2")]
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
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut indices = [_mm256_setzero_si256(); 4];
        for i in 0..mr {
            let mut values = [0.0f32; 8];
            let mut idx = [0u32; 8];
            for j in 0..nr {
                values[j] = (*c.add(i * ldc + j)).value();
                idx[j] = *argmax.add(i * ldc + j);
            }
            acc[i] = _mm256_loadu_ps(values.as_ptr());
            indices[i] = _mm256_loadu_si256(idx.as_ptr().cast());
        }
        for p in 0..k {
            // B is packed and padded to NR even at the right edge.
            let bv = _mm256_loadu_ps(b.add(p * 8));
            let current = _mm256_set1_epi32((offset + p) as u32 as _);
            for i in 0..mr {
                let av = _mm256_set1_ps(*a.add(p * 4 + i));
                let product = if MUL {
                    _mm256_mul_ps(av, bv)
                } else {
                    _mm256_add_ps(av, bv)
                };
                // Keep on equality, including signed zeros. An unordered
                // comparison selects the candidate, matching the scalar API.
                let keep = if MIN {
                    _mm256_cmp_ps::<_CMP_LE_OQ>(acc[i], product)
                } else {
                    _mm256_cmp_ps::<_CMP_GE_OQ>(acc[i], product)
                };
                acc[i] = _mm256_blendv_ps(product, acc[i], keep);
                indices[i] = _mm256_blendv_epi8(current, indices[i], _mm256_castps_si256(keep));
            }
        }
        for i in 0..mr {
            let mut values = [0.0f32; 8];
            let mut idx = [0u32; 8];
            _mm256_storeu_ps(values.as_mut_ptr(), acc[i]);
            _mm256_storeu_si256(idx.as_mut_ptr().cast(), indices[i]);
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
    const NR: usize = 4;
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
                            .tropical_mul(S::from_scalar(*b.add(p * 4 + j))),
                    );
                }
            }
        }
    }
}

impl<S: TropicalWithArgmax<Scalar = f64, Index = u32>, const MIN: bool, const MUL: bool>
    MicrokernelWithArgmax<S> for ArgmaxF64<MIN, MUL>
{
    #[target_feature(enable = "avx2")]
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
        let mut acc = [_mm256_setzero_pd(); 4];
        let mut indices = [_mm256_setzero_si256(); 4];
        for i in 0..mr {
            let mut values = [0.0f64; 4];
            let mut idx = [0u64; 4];
            for j in 0..nr {
                values[j] = (*c.add(i * ldc + j)).value();
                idx[j] = *argmax.add(i * ldc + j) as u64;
            }
            acc[i] = _mm256_loadu_pd(values.as_ptr());
            indices[i] = _mm256_loadu_si256(idx.as_ptr().cast());
        }
        for p in 0..k {
            // B is packed and padded to NR even at the right edge.
            let bv = _mm256_loadu_pd(b.add(p * 4));
            let current = _mm256_set1_epi64x((offset + p) as u32 as _);
            for i in 0..mr {
                let av = _mm256_set1_pd(*a.add(p * 4 + i));
                let product = if MUL {
                    _mm256_mul_pd(av, bv)
                } else {
                    _mm256_add_pd(av, bv)
                };
                // Keep on equality, including signed zeros. An unordered
                // comparison selects the candidate, matching the scalar API.
                let keep = if MIN {
                    _mm256_cmp_pd::<_CMP_LE_OQ>(acc[i], product)
                } else {
                    _mm256_cmp_pd::<_CMP_GE_OQ>(acc[i], product)
                };
                acc[i] = _mm256_blendv_pd(product, acc[i], keep);
                indices[i] = _mm256_blendv_epi8(current, indices[i], _mm256_castpd_si256(keep));
            }
        }
        for i in 0..mr {
            let mut values = [0.0f64; 4];
            let mut idx = [0u64; 4];
            _mm256_storeu_pd(values.as_mut_ptr(), acc[i]);
            _mm256_storeu_si256(idx.as_mut_ptr().cast(), indices[i]);
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
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        {
            let a: [f32; 8] = [1., 1., 1., 1., 2., 2., 2., 2.];
            let b = [1.; 16];
            let mut c = [TropicalMaxPlus(<f32>::NEG_INFINITY); 32];
            let mut indices = [0; 32];
            unsafe {
                ArgmaxF32::<false, false>.execute_with_argmax(
                    4,
                    8,
                    2,
                    0x8000_0011,
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_mut_ptr(),
                    indices.as_mut_ptr(),
                    8,
                );
            }
            assert!(c.iter().all(|v| v.0 == 3.));
            assert!(indices.iter().all(|&p| p == 0x8000_0012));
        }
        {
            let a: [f64; 8] = [1., 1., 1., 1., 2., 2., 2., 2.];
            let b = [1.; 8];
            let mut c = [TropicalMaxPlus(<f64>::NEG_INFINITY); 16];
            let mut indices = [0; 16];
            unsafe {
                ArgmaxF64::<false, false>.execute_with_argmax(
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
    }
}
