//! Quick benchmark: Metal GPU vs sizes table.
//! Run: cargo run --release -p tropical-gemm-metal --example bench_metal

use std::time::Instant;
use tropical_gemm::{TropicalAndOr, TropicalMaxPlus, TropicalMinPlus};
use tropical_gemm_metal::{tropical_matmul_gpu_with_ctx, MetalContext, MetalKernel, MetalScalar};

const SIZES: &[usize] = &[128, 256, 512, 1024, 2048, 4096];
const WARMUP: usize = 2;
const ITERS: usize = 5;

fn bench<T>(ctx: &MetalContext, n: usize, mk_a: impl Fn(usize) -> Vec<T::Scalar>) -> (f64, f64)
where
    T: MetalKernel,
    T::Scalar: MetalScalar,
{
    let a = mk_a(n * n);
    let b = mk_a(n * n);
    for _ in 0..WARMUP {
        let _ = tropical_matmul_gpu_with_ctx::<T>(ctx, &a, n, n, &b, n).unwrap();
    }
    let mut times: Vec<f64> = (0..ITERS)
        .map(|_| {
            let t = Instant::now();
            let _ = tropical_matmul_gpu_with_ctx::<T>(ctx, &a, n, n, &b, n).unwrap();
            t.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], times[0])
}

fn main() {
    let ctx = MetalContext::new().expect("Metal context");
    println!("device: {}", ctx.device_name());
    println!(
        "{:<24} {:>6} {:>12} {:>12} {:>10}",
        "semiring", "n", "median(ms)", "min(ms)", "GOps/s"
    );

    let print_row = |label: &str, med: f64, min: f64, n: usize| {
        let gops = 2.0 * (n as f64).powi(3) / min / 1e9;
        println!(
            "{:<24} {:>6} {:>12.3} {:>12.3} {:>10.1}",
            label,
            n,
            med * 1e3,
            min * 1e3,
            gops
        );
    };

    for &n in SIZES {
        let (med, min) = bench::<TropicalMaxPlus<f32>>(&ctx, n, |l| {
            (0..l).map(|i| (i % 1000) as f32 * 0.01).collect()
        });
        print_row("TropicalMaxPlus<f32>", med, min, n);

        let (med, min) = bench::<TropicalMinPlus<f32>>(&ctx, n, |l| {
            (0..l).map(|i| (i % 1000) as f32 * 0.01).collect()
        });
        print_row("TropicalMinPlus<f32>", med, min, n);

        let (med, min) = bench::<TropicalAndOr>(&ctx, n, |l| (0..l).map(|i| i % 3 == 0).collect());
        print_row("TropicalAndOr", med, min, n);
    }
}
