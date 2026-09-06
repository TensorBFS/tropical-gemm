//! Bit-sliced Boolean OR/AND microkernels.
// Register row/lane indices also address the packed and strided buffers.
#![allow(clippy::needless_range_loop)]
use crate::core::Microkernel;
use crate::types::TropicalBitwise;
use std::arch::aarch64::*;

pub(crate) struct Bitwise32;
impl Microkernel<TropicalBitwise<u32>> for Bitwise32 {
    const MR: usize = 4;
    const NR: usize = 4;
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
        let mut acc = [vdupq_n_u32(0); 4];
        for i in 0..mr {
            let mut values = [0u32; 4];
            for j in 0..nr {
                values[j] = (*c.add(i * ldc + j)).0;
            }
            acc[i] = vld1q_u32(values.as_ptr());
        }
        for p in 0..k {
            let bv = vld1q_u32(b.add(p * 4));
            for i in 0..mr {
                let av = vdupq_n_u32(*a.add(p * 4 + i));
                acc[i] = vorrq_u32(acc[i], vandq_u32(av, bv));
            }
        }
        for i in 0..mr {
            let mut values = [0u32; 4];
            vst1q_u32(values.as_mut_ptr(), acc[i]);
            for j in 0..nr {
                *c.add(i * ldc + j) = TropicalBitwise(values[j]);
            }
        }
    }
}

pub(crate) struct Bitwise64;
impl Microkernel<TropicalBitwise<u64>> for Bitwise64 {
    const MR: usize = 4;
    const NR: usize = 2;
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
        let mut acc = [vdupq_n_u64(0); 4];
        for i in 0..mr {
            let mut values = [0u64; 2];
            for j in 0..nr {
                values[j] = (*c.add(i * ldc + j)).0;
            }
            acc[i] = vld1q_u64(values.as_ptr());
        }
        for p in 0..k {
            let bv = vld1q_u64(b.add(p * 2));
            for i in 0..mr {
                let av = vdupq_n_u64(*a.add(p * 4 + i));
                acc[i] = vorrq_u64(acc[i], vandq_u64(av, bv));
            }
        }
        for i in 0..mr {
            let mut values = [0u64; 2];
            vst1q_u64(values.as_mut_ptr(), acc[i]);
            for j in 0..nr {
                *c.add(i * ldc + j) = TropicalBitwise(values[j]);
            }
        }
    }
}
