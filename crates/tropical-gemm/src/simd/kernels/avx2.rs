use crate::core::Microkernel;
use crate::types::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus};
use wide::{f32x8, f64x4};

/// AVX2 microkernel for TropicalMaxPlus<f32>.
///
/// Uses 8x8 register blocking with f32x8 vectors.
/// Total: 8 accumulators × 8 lanes = 64 elements in registers.
#[derive(Default, Clone, Copy)]
pub struct Avx2MaxPlusF32Kernel;

impl Microkernel<TropicalMaxPlus<f32>> for Avx2MaxPlusF32Kernel {
    const MR: usize = 8;
    const NR: usize = 8;

    #[target_feature(enable = "avx2")]
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
        // Initialize accumulators with -inf
        let neg_inf = f32x8::splat(f32::NEG_INFINITY);
        let mut acc = [neg_inf; 8];

        // Load existing C values into accumulators
        for i in 0..mr {
            let mut row_acc = [f32::NEG_INFINITY; 8];
            for j in 0..nr {
                row_acc[j] = (*c.add(i * ldc + j)).0;
            }
            acc[i] = f32x8::from(row_acc);
        }

        // Main computation loop
        for p in 0..k {
            // Load A column (mr elements, padded to 8)
            let mut a_vals = [0.0f32; 8];
            for i in 0..mr {
                a_vals[i] = *a.add(p * Self::MR + i);
            }

            // Load B row (nr elements, padded to 8)
            let mut b_vals = [0.0f32; 8];
            for j in 0..nr {
                b_vals[j] = *b.add(p * Self::NR + j);
            }
            let b_vec = f32x8::from(b_vals);

            // For each row of A
            for i in 0..mr {
                // Tropical mul: a[i] + b[j] for all j
                let a_broadcast = f32x8::splat(a_vals[i]);
                let product = a_broadcast + b_vec;

                // Tropical add: max(acc, product)
                acc[i] = acc[i].max(product);
            }
        }

        // Write back results
        for i in 0..mr {
            let row: [f32; 8] = acc[i].into();
            for j in 0..nr {
                *c.add(i * ldc + j) = TropicalMaxPlus(row[j]);
            }
        }
    }
}

/// AVX2 microkernel for TropicalMaxPlus<f64>.
#[derive(Default, Clone, Copy)]
pub struct Avx2MaxPlusF64Kernel;

impl Microkernel<TropicalMaxPlus<f64>> for Avx2MaxPlusF64Kernel {
    const MR: usize = 4;
    const NR: usize = 4;

    #[target_feature(enable = "avx2")]
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
        let neg_inf = f64x4::splat(f64::NEG_INFINITY);
        let mut acc = [neg_inf; 4];

        // Load existing C
        for i in 0..mr {
            let mut row_acc = [f64::NEG_INFINITY; 4];
            for j in 0..nr {
                row_acc[j] = (*c.add(i * ldc + j)).0;
            }
            acc[i] = f64x4::from(row_acc);
        }

        // Main loop
        for p in 0..k {
            let mut a_vals = [0.0f64; 4];
            for i in 0..mr {
                a_vals[i] = *a.add(p * Self::MR + i);
            }

            let mut b_vals = [0.0f64; 4];
            for j in 0..nr {
                b_vals[j] = *b.add(p * Self::NR + j);
            }
            let b_vec = f64x4::from(b_vals);

            for i in 0..mr {
                let a_broadcast = f64x4::splat(a_vals[i]);
                let product = a_broadcast + b_vec;
                acc[i] = acc[i].max(product);
            }
        }

        // Write back
        for i in 0..mr {
            let row: [f64; 4] = acc[i].into();
            for j in 0..nr {
                *c.add(i * ldc + j) = TropicalMaxPlus(row[j]);
            }
        }
    }
}

/// AVX2 microkernel for TropicalMinPlus<f32>.
#[derive(Default, Clone, Copy)]
pub struct Avx2MinPlusF32Kernel;

impl Microkernel<TropicalMinPlus<f32>> for Avx2MinPlusF32Kernel {
    const MR: usize = 8;
    const NR: usize = 8;

    #[target_feature(enable = "avx2")]
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
        let pos_inf = f32x8::splat(f32::INFINITY);
        let mut acc = [pos_inf; 8];

        // Load existing C
        for i in 0..mr {
            let mut row_acc = [f32::INFINITY; 8];
            for j in 0..nr {
                row_acc[j] = (*c.add(i * ldc + j)).0;
            }
            acc[i] = f32x8::from(row_acc);
        }

        // Main loop
        for p in 0..k {
            let mut a_vals = [0.0f32; 8];
            for i in 0..mr {
                a_vals[i] = *a.add(p * Self::MR + i);
            }

            let mut b_vals = [0.0f32; 8];
            for j in 0..nr {
                b_vals[j] = *b.add(p * Self::NR + j);
            }
            let b_vec = f32x8::from(b_vals);

            for i in 0..mr {
                let a_broadcast = f32x8::splat(a_vals[i]);
                let product = a_broadcast + b_vec;
                // MinPlus: tropical add = min
                acc[i] = acc[i].min(product);
            }
        }

        // Write back
        for i in 0..mr {
            let row: [f32; 8] = acc[i].into();
            for j in 0..nr {
                *c.add(i * ldc + j) = TropicalMinPlus(row[j]);
            }
        }
    }
}

/// AVX2 microkernel for TropicalMaxMul<f32>.
#[derive(Default, Clone, Copy)]
pub struct Avx2MaxMulF32Kernel;

impl Microkernel<TropicalMaxMul<f32>> for Avx2MaxMulF32Kernel {
    const MR: usize = 8;
    const NR: usize = 8;

    #[target_feature(enable = "avx2")]
    unsafe fn execute(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        a: *const f32,
        b: *const f32,
        c: *mut TropicalMaxMul<f32>,
        ldc: usize,
    ) {
        let zero = f32x8::splat(0.0);
        let mut acc = [zero; 8];

        // Load existing C
        for i in 0..mr {
            let mut row_acc = [0.0f32; 8];
            for j in 0..nr {
                row_acc[j] = (*c.add(i * ldc + j)).0;
            }
            acc[i] = f32x8::from(row_acc);
        }

        // Main loop
        for p in 0..k {
            let mut a_vals = [0.0f32; 8];
            for i in 0..mr {
                a_vals[i] = *a.add(p * Self::MR + i);
            }

            let mut b_vals = [0.0f32; 8];
            for j in 0..nr {
                b_vals[j] = *b.add(p * Self::NR + j);
            }
            let b_vec = f32x8::from(b_vals);

            for i in 0..mr {
                let a_broadcast = f32x8::splat(a_vals[i]);
                // MaxMul: tropical mul = standard mul
                let product = a_broadcast * b_vec;
                // tropical add = max
                acc[i] = acc[i].max(product);
            }
        }

        // Write back
        for i in 0..mr {
            let row: [f32; 8] = acc[i].into();
            for j in 0..nr {
                *c.add(i * ldc + j) = TropicalMaxMul(row[j]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TropicalSemiring;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_max_plus_f32() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let kernel = Avx2MaxPlusF32Kernel;
        let mr = 2;
        let nr = 2;
        let k = 3;

        // A: 2x3 packed
        let a: [f32; 24] = [
            1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // col 0
            2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // col 1
            3.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // col 2
        ];

        // B: 3x2 packed
        let b: [f32; 24] = [
            1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 0
            3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 1
            5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 2
        ];

        let mut c = vec![TropicalMaxPlus::tropical_zero(); 4];
        let ldc = 2;

        unsafe {
            kernel.execute(mr, nr, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), ldc);
        }

        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert!((c[0].0 - 8.0).abs() < 1e-6);
        // C[0,1] = max(1+2, 2+4, 3+6) = 9
        assert!((c[1].0 - 9.0).abs() < 1e-6);
        // C[1,0] = max(4+1, 5+3, 6+5) = 11
        assert!((c[2].0 - 11.0).abs() < 1e-6);
        // C[1,1] = max(4+2, 5+4, 6+6) = 12
        assert!((c[3].0 - 12.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_min_plus_f32() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let kernel = Avx2MinPlusF32Kernel;
        let mr = 2;
        let nr = 2;
        let k = 3;

        // A: 2x3 packed
        let a: [f32; 24] = [
            1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0,
            6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        // B: 3x2 packed
        let b: [f32; 24] = [
            1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0,
            6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let mut c = vec![TropicalMinPlus::tropical_zero(); 4];
        let ldc = 2;

        unsafe {
            kernel.execute(mr, nr, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), ldc);
        }

        // C[0,0] = min(1+1, 2+3, 3+5) = 2
        assert!((c[0].0 - 2.0).abs() < 1e-6);
        // C[0,1] = min(1+2, 2+4, 3+6) = 3
        assert!((c[1].0 - 3.0).abs() < 1e-6);
        // C[1,0] = min(4+1, 5+3, 6+5) = 5
        assert!((c[2].0 - 5.0).abs() < 1e-6);
        // C[1,1] = min(4+2, 5+4, 6+6) = 6
        assert!((c[3].0 - 6.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_max_mul_f32() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let kernel = Avx2MaxMulF32Kernel;
        let mr = 2;
        let nr = 2;
        let k = 2;

        // A: 2x2 packed
        let a: [f32; 16] = [
            2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        // B: 2x2 packed
        let b: [f32; 16] = [
            1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let mut c = vec![TropicalMaxMul::tropical_zero(); 4];
        let ldc = 2;

        unsafe {
            kernel.execute(mr, nr, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), ldc);
        }

        // C[0,0] = max(2*1, 4*3) = max(2, 12) = 12
        assert!((c[0].0 - 12.0).abs() < 1e-6);
        // C[0,1] = max(2*2, 4*4) = max(4, 16) = 16
        assert!((c[1].0 - 16.0).abs() < 1e-6);
        // C[1,0] = max(3*1, 5*3) = max(3, 15) = 15
        assert!((c[2].0 - 15.0).abs() < 1e-6);
        // C[1,1] = max(3*2, 5*4) = max(6, 20) = 20
        assert!((c[3].0 - 20.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_max_plus_f64() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let kernel = Avx2MaxPlusF64Kernel;
        let mr = 2;
        let nr = 2;
        let k = 2;

        // A: 2x2 packed (4 f64 per column for mr=4 padding)
        let a: [f64; 8] = [
            1.0, 2.0, 0.0, 0.0, // col 0
            3.0, 4.0, 0.0, 0.0, // col 1
        ];

        // B: 2x2 packed (4 f64 per row for nr=4 padding)
        let b: [f64; 8] = [
            1.0, 2.0, 0.0, 0.0, // row 0
            3.0, 4.0, 0.0, 0.0, // row 1
        ];

        let mut c = vec![TropicalMaxPlus::tropical_zero(); 4];
        let ldc = 2;

        unsafe {
            kernel.execute(mr, nr, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), ldc);
        }

        // C[0,0] = max(1+1, 3+3) = 6
        assert!((c[0].0 - 6.0).abs() < 1e-10);
        // C[0,1] = max(1+2, 3+4) = 7
        assert!((c[1].0 - 7.0).abs() < 1e-10);
        // C[1,0] = max(2+1, 4+3) = 7
        assert!((c[2].0 - 7.0).abs() < 1e-10);
        // C[1,1] = max(2+2, 4+4) = 8
        assert!((c[3].0 - 8.0).abs() < 1e-10);
    }
}
