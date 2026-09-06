#![cfg(feature = "parallel")]

use rayon::{ThreadPool, ThreadPoolBuilder};
use tropical_gemm::{
    core::{tropical_gemm_with_argmax_portable, GemmWithArgmax, Transpose},
    simd::KernelDispatch,
    tropical_matmul, tropical_matmul_batched, tropical_matmul_strided_batched, Mat,
    TropicalBitwise, TropicalGemm, TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus,
    TropicalSemiring, TropicalWithArgmax,
};

fn pool(threads: usize) -> ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

// Build physically transposed, padded inputs from logical row-major values.
fn input<T: Copy + Default>(
    rows: usize,
    cols: usize,
    trans: bool,
    value: impl Fn(usize, usize) -> T,
) -> (Vec<T>, usize) {
    let (r, c) = if trans { (cols, rows) } else { (rows, cols) };
    let ld = c + 3;
    let mut data = vec![T::default(); r * ld];
    for i in 0..rows {
        for j in 0..cols {
            data[if trans { j * ld + i } else { i * ld + j }] = value(i, j);
        }
    }
    (data, ld)
}

fn values<T: TropicalSemiring<Scalar = f32> + KernelDispatch>(ta: bool, tb: bool) {
    let (one, four) = (pool(1), pool(4));
    // Exercise both splitting axes, non-tile-aligned tails, and several K panels.
    for (m, n) in [(257, 33), (33, 257)] {
        let k = 513;
        let (a, lda) = input(m, k, ta, |i, p| ((i + p * 3) % 13) as f32 * 0.25);
        let (b, ldb) = input(k, n, tb, |p, j| ((p * 7 + j) % 17) as f32 * 0.5);
        let ldc = n + 5;
        let run = || {
            let mut c = vec![T::from_scalar(12345.); m * ldc];
            let mut gemm = TropicalGemm::<T>::new(m, n, k);
            if ta {
                gemm = gemm.trans_a();
            }
            if tb {
                gemm = gemm.trans_b();
            }
            gemm.execute(&a, lda, &b, ldb, &mut c, ldc);
            for row in c.chunks(ldc) {
                assert!(row[n..].iter().all(|v| v.value() == 12345.));
            }
            c.iter().map(|v| v.value().to_bits()).collect::<Vec<_>>()
        };
        assert_eq!(one.install(run), four.install(run));
    }
}

#[test]
fn parallel_values_match_serial_with_transposes_and_padding() {
    values::<TropicalMaxPlus<f32>>(false, false);
    values::<TropicalMaxPlus<f32>>(true, true);
    values::<TropicalMinPlus<f32>>(true, false);
    values::<TropicalMaxMul<f32>>(false, true);
}

fn argmax<T: TropicalWithArgmax<Index = u32, Scalar = i32>>(ta: bool, tb: bool) {
    let (one, four) = (pool(1), pool(4));
    for (m, n) in [(257, 33), (33, 257)] {
        let k = 513;
        // Repeated input values force ties across K-panel boundaries. The first
        // row has no contribution, exercising integer argmax canonicalization.
        let (a, lda) = input(m, k, ta, |i, p| {
            if i == 0 {
                T::tropical_zero().value()
            } else {
                (p % 7) as i32
            }
        });
        let (b, ldb) = input(k, n, tb, |p, j| ((p + j) % 5) as i32);
        let run = || {
            let mut c = GemmWithArgmax::<T>::with_ld(m, n, n + 5);
            c.values.fill(T::from_scalar(12345));
            c.argmax.fill(u32::MAX);
            unsafe {
                tropical_gemm_with_argmax_portable(
                    m,
                    n,
                    k,
                    a.as_ptr(),
                    lda,
                    if ta {
                        Transpose::Trans
                    } else {
                        Transpose::NoTrans
                    },
                    b.as_ptr(),
                    ldb,
                    if tb {
                        Transpose::Trans
                    } else {
                        Transpose::NoTrans
                    },
                    &mut c,
                );
            }
            for i in 0..m {
                for j in n..c.ld {
                    assert_eq!(c.values[i * c.ld + j].value(), 12345);
                    assert_eq!(c.argmax[i * c.ld + j], u32::MAX);
                }
            }
            if T::tropical_zero().value() != 0 {
                assert!(c.argmax[..n].iter().all(|&p| p == 0));
            }
            (c.values, c.argmax)
        };
        assert_eq!(one.install(run), four.install(run));
    }
}

#[test]
fn parallel_argmax_preserves_ties_sentinels_transposes_and_padding() {
    argmax::<TropicalMaxPlus<i32>>(false, false);
    argmax::<TropicalMaxPlus<i32>>(true, true);
    argmax::<TropicalMinPlus<i32>>(true, false);
    argmax::<TropicalMaxMul<i32>>(false, true);
}

#[test]
fn parallel_float_argmax_and_column_major_api_match_serial() {
    let (one, four) = (pool(1), pool(4));
    let (m, n, k) = (257, 33, 513);
    let a =
        Mat::<TropicalMaxPlus<f64>>::from_fn(m, k, |i, p| TropicalMaxPlus(((i + p) % 11) as f64));
    let b =
        Mat::<TropicalMaxPlus<f64>>::from_fn(k, n, |p, j| TropicalMaxPlus(((p + j) % 7) as f64));
    let run = || {
        let c = a.matmul_argmax(&b);
        (c.values.as_slice().to_vec(), c.argmax)
    };
    assert_eq!(one.install(run), four.install(run));
    assert_eq!(
        one.install(|| &a * &b).as_slice(),
        four.install(|| &a * &b).as_slice()
    );
}

#[test]
fn parallel_batches_and_bitwise_match_serial() {
    let (one, four) = (pool(1), pool(4));
    let (m, n, k) = (257, 33, 513);
    let a: Vec<u32> = (0..m * k).map(|i| (i as u32).wrapping_mul(34567)).collect();
    let b: Vec<u32> = (0..k * n).map(|i| !(i as u32)).collect();
    let expected = one.install(|| tropical_matmul::<TropicalBitwise<u32>>(&a, m, k, &b, n));
    let a_batch = vec![a.clone(); 4];
    let b_batch = vec![b.clone(); 4];
    let run = || tropical_matmul_batched::<TropicalBitwise<u32>>(&a_batch, &b_batch, m, k, n);
    let batch = four.install(run);
    assert!(batch.iter().all(|v| *v == expected));
    let flat = four.install(|| {
        tropical_matmul_strided_batched::<TropicalBitwise<u32>>(
            &a.repeat(4),
            &b.repeat(4),
            4,
            m,
            k,
            n,
        )
    });
    assert_eq!(flat, expected.repeat(4));
}

#[test]
fn a_single_gemm_uses_multiple_workers_in_both_paths() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tropical_gemm::core::{
        tropical_gemm_inner, tropical_gemm_with_argmax_inner, Microkernel, MicrokernelWithArgmax,
        PortableMicrokernel, TilingParams,
    };

    struct ObservedKernel(AtomicUsize, std::sync::Mutex<()>, std::sync::Condvar);
    impl ObservedKernel {
        fn record(&self) {
            let bit = 1 << rayon::current_thread_index().unwrap();
            if self.0.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
                // Give another worker a deterministic opportunity to steal a
                // tile. A timeout reports broken splitting without hanging CI.
                let guard = self.1.lock().unwrap();
                let (_guard, _) = self
                    .2
                    .wait_timeout_while(guard, std::time::Duration::from_secs(5), |_| {
                        self.0.load(Ordering::Relaxed).count_ones() < 2
                    })
                    .unwrap();
                self.2.notify_all();
                assert!(self.0.load(Ordering::Relaxed).count_ones() > 1);
            }
        }
    }
    impl Microkernel<TropicalMaxPlus<f32>> for ObservedKernel {
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
            self.record();
            PortableMicrokernel.execute(mr, nr, k, a, b, c, ldc);
        }
    }
    impl MicrokernelWithArgmax<TropicalMaxPlus<f32>> for ObservedKernel {
        unsafe fn execute_with_argmax(
            &self,
            mr: usize,
            nr: usize,
            k: usize,
            offset: usize,
            a: *const f32,
            b: *const f32,
            c: *mut TropicalMaxPlus<f32>,
            argmax: *mut u32,
            ldc: usize,
        ) {
            self.record();
            PortableMicrokernel.execute_with_argmax(mr, nr, k, offset, a, b, c, argmax, ldc);
        }
    }
    let a = vec![1.; 128 * 512];
    let b = vec![2.; 512 * 128];
    let kernel = ObservedKernel(
        AtomicUsize::new(0),
        std::sync::Mutex::new(()),
        std::sync::Condvar::new(),
    );
    let four = pool(4);
    for argmax in [false, true] {
        kernel.0.store(0, Ordering::Relaxed);
        let c = four.install(|| {
            let mut result = GemmWithArgmax::<TropicalMaxPlus<f32>>::new(128, 128);
            unsafe {
                if argmax {
                    tropical_gemm_with_argmax_inner(
                        128,
                        128,
                        512,
                        a.as_ptr(),
                        512,
                        Transpose::NoTrans,
                        b.as_ptr(),
                        128,
                        Transpose::NoTrans,
                        &mut result,
                        &TilingParams::PORTABLE,
                        &kernel,
                    );
                } else {
                    tropical_gemm_inner(
                        128,
                        128,
                        512,
                        a.as_ptr(),
                        512,
                        Transpose::NoTrans,
                        b.as_ptr(),
                        128,
                        Transpose::NoTrans,
                        result.values.as_mut_ptr(),
                        128,
                        &TilingParams::PORTABLE,
                        &kernel,
                    );
                }
            }
            result
        });
        assert!(c.values.iter().all(|v| v.0 == 3.));
        assert!(c.argmax.iter().all(|&p| p == 0));
        assert!(
            kernel.0.load(Ordering::Relaxed).count_ones() > 1,
            "single GEMM ran on only one worker (argmax={argmax})"
        );
    }
}
