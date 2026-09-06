//! Reproducible CPU semiring benchmark: N threads samples (default: 256 1 7).
//! Copy this unchanged into an older checkout for a baseline comparison.
#[cfg(feature = "parallel")]
fn main() {
    use std::{hint::black_box, time::Instant};
    use tropical_gemm::{
        tropical_matmul, tropical_matmul_with_argmax, TropicalBitwise, TropicalMaxMul,
        TropicalMaxPlus, TropicalMinPlus,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let number = |i: usize, default| args.get(i).map_or(default, |s| s.parse::<usize>().unwrap());
    let (n, threads, samples) = (number(0, 256), number(1, 1), number(2, 7));
    assert!(n > 0 && threads > 0 && samples > 0);
    println!("n,threads,semiring,scalar,mode,median_ms");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let measure = |semiring, scalar, mode, run: &mut dyn FnMut()| {
            for _ in 0..3 {
                run();
            }
            let mut times = Vec::with_capacity(samples);
            for _ in 0..samples {
                let start = Instant::now();
                run();
                times.push(start.elapsed().as_secs_f64() * 1000.);
            }
            times.sort_by(f64::total_cmp);
            println!(
                "{n},{threads},{semiring},{scalar},{mode},{:.6}",
                times[samples / 2]
            );
        };
        macro_rules! floats {
            ($s:ident, $t:ty) => {{
                let a: Vec<$t> = (0..n * n).map(|i| (i % 101) as $t * 0.125).collect();
                let b: Vec<$t> = (0..n * n).map(|i| (i % 97) as $t * 0.25).collect();
                measure(stringify!($s), stringify!($t), "values", &mut || {
                    black_box(tropical_matmul::<$s<$t>>(&a, n, n, &b, n));
                });
                measure(stringify!($s), stringify!($t), "argmax", &mut || {
                    black_box(tropical_matmul_with_argmax::<$s<$t>>(&a, n, n, &b, n));
                });
            }};
        }
        floats!(TropicalMaxPlus, f32);
        floats!(TropicalMinPlus, f32);
        floats!(TropicalMaxMul, f32);
        floats!(TropicalMaxPlus, f64);
        floats!(TropicalMinPlus, f64);
        floats!(TropicalMaxMul, f64);
        macro_rules! bits {
            ($t:ty) => {{
                let a: Vec<$t> = (0..n * n)
                    .map(|i| (1 as $t).rotate_left((i % <$t>::BITS as usize) as u32))
                    .collect();
                let b: Vec<$t> = (0..n * n)
                    .map(|i| (1 as $t).rotate_left(((i * 7) % <$t>::BITS as usize) as u32))
                    .collect();
                measure("TropicalBitwise", stringify!($t), "values", &mut || {
                    black_box(tropical_matmul::<TropicalBitwise<$t>>(&a, n, n, &b, n));
                });
            }};
        }
        bits!(u32);
        bits!(u64);
    });
}
#[cfg(not(feature = "parallel"))]
fn main() {
    eprintln!("bench_cpu requires the parallel feature");
}
