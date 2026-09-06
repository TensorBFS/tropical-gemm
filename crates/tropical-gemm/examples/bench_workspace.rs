//! Packing reuse benchmark: N threads samples (default: 128 1 21).
#[cfg(feature = "parallel")]
fn main() {
    use std::{hint::black_box, time::Instant};
    use tropical_gemm::{
        tropical_matmul_with_argmax, tropical_matmul_with_argmax_with_workspace, GemmWorkspace,
        TropicalGemm, TropicalMaxPlus,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let number = |i: usize, default| args.get(i).map_or(default, |s| s.parse::<usize>().unwrap());
    let (n, threads, samples) = (number(0, 128), number(1, 1), number(2, 21));
    assert!(n > 0 && threads > 0 && samples > 0);
    let a: Vec<f32> = (0..n * n).map(|i| (i % 101) as f32 * 0.125).collect();
    let b: Vec<f32> = (0..n * n).map(|i| (i % 97) as f32 * 0.25).collect();
    let mut c = vec![TropicalMaxPlus(0.); n * n];
    let mut workspace = GemmWorkspace::new();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    println!("n,threads,mode,reuse,median_ms,packing_capacity_bytes");
    pool.install(|| {
        for mode in ["values", "argmax"] {
            for reuse in [false, true] {
                let mut run = || {
                    if mode == "values" {
                        if reuse {
                            TropicalGemm::new(n, n, n).execute_with_workspace(
                                &a,
                                n,
                                &b,
                                n,
                                &mut c,
                                n,
                                &mut workspace,
                            );
                        } else {
                            TropicalGemm::new(n, n, n).execute(&a, n, &b, n, &mut c, n);
                        }
                        black_box(&c);
                    } else if reuse {
                        black_box(tropical_matmul_with_argmax_with_workspace::<
                            TropicalMaxPlus<f32>,
                        >(&a, n, n, &b, n, &mut workspace));
                    } else {
                        black_box(tropical_matmul_with_argmax::<TropicalMaxPlus<f32>>(
                            &a, n, n, &b, n,
                        ));
                    }
                };
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
                    "{n},{threads},{mode},{reuse},{:.6},{}",
                    times[samples / 2],
                    if reuse { workspace.capacity_bytes() } else { 0 }
                );
            }
        }
    });
}
#[cfg(not(feature = "parallel"))]
fn main() {
    eprintln!("bench_workspace requires the parallel feature");
}
