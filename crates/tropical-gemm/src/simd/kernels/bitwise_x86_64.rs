//! Bit-sliced Boolean OR/AND microkernels.
// Register row/lane indices also address the packed and strided buffers.
#![allow(clippy::needless_range_loop)]
use crate::core::Microkernel;
use crate::types::TropicalBitwise;
use std::arch::x86_64::*;

pub(crate) struct Bitwise32;
impl Microkernel<TropicalBitwise<u32>> for Bitwise32 {
    const MR: usize = 4;
    const NR: usize = 8;
    #[target_feature(enable = "avx2")]
    unsafe fn execute(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        a: *const u32,
        b: *const u32,
        c: *mut TropicalBitwise<u32>,
        ldc: usize,
    ) {
        let mut acc = [_mm256_setzero_si256(); 4];
        for i in 0..mr {
            let mut values = [0u32; 8];
            for j in 0..nr {
                values[j] = (*c.add(i * ldc + j)).0;
            }
            acc[i] = _mm256_loadu_si256(values.as_ptr().cast());
        }
        for p in 0..k {
            let bv = _mm256_loadu_si256(b.add(p * 8).cast());
            for i in 0..mr {
                let av = _mm256_set1_epi32(*a.add(p * 4 + i) as _);
                acc[i] = _mm256_or_si256(acc[i], _mm256_and_si256(av, bv));
            }
        }
        for i in 0..mr {
            let mut values = [0u32; 8];
            _mm256_storeu_si256(values.as_mut_ptr().cast(), acc[i]);
            for j in 0..nr {
                *c.add(i * ldc + j) = TropicalBitwise(values[j]);
            }
        }
    }
}

pub(crate) struct Bitwise64;
impl Microkernel<TropicalBitwise<u64>> for Bitwise64 {
    const MR: usize = 4;
    const NR: usize = 4;
    #[target_feature(enable = "avx2")]
    unsafe fn execute(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        a: *const u64,
        b: *const u64,
        c: *mut TropicalBitwise<u64>,
        ldc: usize,
    ) {
        let mut acc = [_mm256_setzero_si256(); 4];
        for i in 0..mr {
            let mut values = [0u64; 4];
            for j in 0..nr {
                values[j] = (*c.add(i * ldc + j)).0;
            }
            acc[i] = _mm256_loadu_si256(values.as_ptr().cast());
        }
        for p in 0..k {
            let bv = _mm256_loadu_si256(b.add(p * 4).cast());
            for i in 0..mr {
                let av = _mm256_set1_epi64x(*a.add(p * 4 + i) as _);
                acc[i] = _mm256_or_si256(acc[i], _mm256_and_si256(av, bv));
            }
        }
        for i in 0..mr {
            let mut values = [0u64; 4];
            _mm256_storeu_si256(values.as_mut_ptr().cast(), acc[i]);
            for j in 0..nr {
                *c.add(i * ldc + j) = TropicalBitwise(values[j]);
            }
        }
    }
}
