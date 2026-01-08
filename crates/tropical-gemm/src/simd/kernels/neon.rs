use crate::core::Microkernel;
use crate::types::{TropicalMaxPlus, TropicalMinPlus};
use wide::{f32x4, f64x2};

/// ARM NEON microkernel for TropicalMaxPlus<f32>.
///
/// Uses 4x4 register blocking with f32x4 vectors.
#[derive(Default, Clone, Copy)]
pub struct NeonMaxPlusF32Kernel;

impl Microkernel<TropicalMaxPlus<f32>> for NeonMaxPlusF32Kernel {
    const MR: usize = 4;
    const NR: usize = 4;

    unsafe fn execute(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        a: *const f32,
        b: *const f32,
        c: *mut TropicalMaxPlus<f32>,
        ldc: usize,
    ) {
        let neg_inf = f32x4::splat(f32::NEG_INFINITY);
        let mut acc = [neg_inf; 4];

        // Load existing C
        for i in 0..mr {
            let mut row_acc = [f32::NEG_INFINITY; 4];
            for j in 0..nr {
                row_acc[j] = (*c.add(i * ldc + j)).0;
            }
            acc[i] = f32x4::from(row_acc);
        }

        // Main loop
        for p in 0..k {
            let mut a_vals = [0.0f32; 4];
            for i in 0..mr {
                a_vals[i] = *a.add(p * Self::MR + i);
            }

            let mut b_vals = [0.0f32; 4];
            for j in 0..nr {
                b_vals[j] = *b.add(p * Self::NR + j);
            }
            let b_vec = f32x4::from(b_vals);

            for i in 0..mr {
                let a_broadcast = f32x4::splat(a_vals[i]);
                let product = a_broadcast + b_vec;
                acc[i] = acc[i].max(product);
            }
        }

        // Write back
        for i in 0..mr {
            let row: [f32; 4] = acc[i].into();
            for j in 0..nr {
                *c.add(i * ldc + j) = TropicalMaxPlus(row[j]);
            }
        }
    }
}

/// ARM NEON microkernel for TropicalMaxPlus<f64>.
#[derive(Default, Clone, Copy)]
pub struct NeonMaxPlusF64Kernel;

impl Microkernel<TropicalMaxPlus<f64>> for NeonMaxPlusF64Kernel {
    const MR: usize = 2;
    const NR: usize = 2;

    unsafe fn execute(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        a: *const f64,
        b: *const f64,
        c: *mut TropicalMaxPlus<f64>,
        ldc: usize,
    ) {
        let neg_inf = f64x2::splat(f64::NEG_INFINITY);
        let mut acc = [neg_inf; 2];

        // Load existing C
        for i in 0..mr {
            let mut row_acc = [f64::NEG_INFINITY; 2];
            for j in 0..nr {
                row_acc[j] = (*c.add(i * ldc + j)).0;
            }
            acc[i] = f64x2::from(row_acc);
        }

        // Main loop
        for p in 0..k {
            let mut a_vals = [0.0f64; 2];
            for i in 0..mr {
                a_vals[i] = *a.add(p * Self::MR + i);
            }

            let mut b_vals = [0.0f64; 2];
            for j in 0..nr {
                b_vals[j] = *b.add(p * Self::NR + j);
            }
            let b_vec = f64x2::from(b_vals);

            for i in 0..mr {
                let a_broadcast = f64x2::splat(a_vals[i]);
                let product = a_broadcast + b_vec;
                acc[i] = acc[i].max(product);
            }
        }

        // Write back
        for i in 0..mr {
            let row: [f64; 2] = acc[i].into();
            for j in 0..nr {
                *c.add(i * ldc + j) = TropicalMaxPlus(row[j]);
            }
        }
    }
}

/// ARM NEON microkernel for TropicalMinPlus<f32>.
#[derive(Default, Clone, Copy)]
pub struct NeonMinPlusF32Kernel;

impl Microkernel<TropicalMinPlus<f32>> for NeonMinPlusF32Kernel {
    const MR: usize = 4;
    const NR: usize = 4;

    unsafe fn execute(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        a: *const f32,
        b: *const f32,
        c: *mut TropicalMinPlus<f32>,
        ldc: usize,
    ) {
        let pos_inf = f32x4::splat(f32::INFINITY);
        let mut acc = [pos_inf; 4];

        for i in 0..mr {
            let mut row_acc = [f32::INFINITY; 4];
            for j in 0..nr {
                row_acc[j] = (*c.add(i * ldc + j)).0;
            }
            acc[i] = f32x4::from(row_acc);
        }

        for p in 0..k {
            let mut a_vals = [0.0f32; 4];
            for i in 0..mr {
                a_vals[i] = *a.add(p * Self::MR + i);
            }

            let mut b_vals = [0.0f32; 4];
            for j in 0..nr {
                b_vals[j] = *b.add(p * Self::NR + j);
            }
            let b_vec = f32x4::from(b_vals);

            for i in 0..mr {
                let a_broadcast = f32x4::splat(a_vals[i]);
                let product = a_broadcast + b_vec;
                acc[i] = acc[i].min(product);
            }
        }

        for i in 0..mr {
            let row: [f32; 4] = acc[i].into();
            for j in 0..nr {
                *c.add(i * ldc + j) = TropicalMinPlus(row[j]);
            }
        }
    }
}
