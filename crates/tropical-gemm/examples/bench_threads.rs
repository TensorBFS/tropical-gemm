//! Measure single-GEMM thread scaling (including allocations and packing).
//!
//! cargo run --release -p tropical-gemm --example bench_threads -- 512 512 512 7 1,2,4,8
//! Arguments: M N K samples thread-counts. Reports median wall time in CSV format.

#[cfg(feature = "parallel")]
fn main() {
    use std::{hint::black_box, time::Instant};
    use tropical_gemm::{
        tropical_matmul, tropical_matmul_batched, tropical_matmul_with_argmax, TropicalMaxPlus,
    };

    let args: Vec<String> = std::env::args().skip(1).collect();
    let number = |i: usize, default| args.get(i).map_or(default, |s| s.parse::<usize>().unwrap());
    let (m, n, k, samples) = (number(0, 512), number(1, 512), number(2, 512), number(3, 7));
    assert!(samples > 0);
    let threads = args.get(4).map_or("1,2,4,8", String::as_str);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 101) as f32 * 0.125).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 97) as f32 * 0.25).collect();
    let a_batch = vec![a.clone(); 4];
    let b_batch = vec![b.clone(); 4];
    println!("m,n,k,threads,mode,median_ms");
    for threads in threads.split(',').map(|s| s.parse::<usize>().unwrap()) {
        assert!(threads > 0);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for mode in ["values", "argmax", "batch4"] {
            let run = || {
                if mode == "argmax" {
                    black_box(tropical_matmul_with_argmax::<TropicalMaxPlus<f32>>(
                        &a, m, k, &b, n,
                    ));
                } else if mode == "batch4" {
                    black_box(tropical_matmul_batched::<TropicalMaxPlus<f32>>(
                        &a_batch, &b_batch, m, k, n,
                    ));
                } else {
                    black_box(tropical_matmul::<TropicalMaxPlus<f32>>(&a, m, k, &b, n));
                }
            };
            for _ in 0..3 {
                pool.install(run);
            }
            let mut times = Vec::with_capacity(samples);
            for _ in 0..samples {
                let start = Instant::now();
                pool.install(run);
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(f64::total_cmp);
            println!("{m},{n},{k},{threads},{},{:.4}", mode, times[samples / 2]);
        }
    }
}

#[cfg(not(feature = "parallel"))]
fn main() {
    eprintln!("bench_threads requires the parallel feature");
}
