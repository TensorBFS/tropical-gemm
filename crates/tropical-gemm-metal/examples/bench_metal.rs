//! Benchmark comparing Metal GPU vs CPU performance.

use std::time::Instant;
use tropical_gemm_metal::{tropical_matmul_gpu, MetalContext};
use tropical_types::TropicalMaxPlus;

fn main() {
    println!("Tropical GEMM Metal Benchmark");
    println!("==============================\n");

    // Check Metal availability
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Metal not available: {:?}", e);
            return;
        }
    };
    println!("Device: {}\n", ctx.device_name());

    let sizes = [256, 512, 1024, 2048];

    println!("{:>6} {:>12} {:>12}", "Size", "GPU (ms)", "GFLOPS");
    println!("{:-<6} {:-<12} {:-<12}", "", "", "");

    for &n in &sizes {
        let a: Vec<f32> = (0..n * n).map(|i| (i % 100) as f32).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i + 50) % 100) as f32).collect();

        // Warmup
        let _ = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, n, n, &b, n);

        // Benchmark
        let iterations = if n <= 512 { 10 } else { 3 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, n, n, &b, n);
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        // Calculate GFLOPS (2 ops per element: add + max)
        let ops = 2.0 * (n as f64).powi(3);
        let gflops = ops / (avg_ms / 1000.0) / 1e9;

        println!("{:>6} {:>12.3} {:>12.1}", n, avg_ms, gflops);
    }
}
