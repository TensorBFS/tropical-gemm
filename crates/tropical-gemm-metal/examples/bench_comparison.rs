//! Comprehensive benchmark for Metal GPU vs CPU comparison.

use std::time::Instant;
use tropical_gemm_metal::{tropical_matmul_gpu, MetalContext};
use tropical_types::{TropicalMaxPlus, TropicalMinPlus, TropicalMaxMul};

fn bench_cpu(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    // Simple CPU implementation for comparison
    let mut c = vec![f32::NEG_INFINITY; m * n];
    for i in 0..m {
        for j in 0..n {
            for kk in 0..k {
                let val = a[i * k + kk] + b[kk * n + j];
                if val > c[i * n + j] {
                    c[i * n + j] = val;
                }
            }
        }
    }
    c
}

fn main() {
    println!("Tropical GEMM: Metal vs CPU Benchmark");
    println!("======================================\n");

    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Metal not available: {:?}", e);
            return;
        }
    };
    println!("GPU: {}", ctx.device_name());
    println!();

    // GPU vs CPU comparison
    println!("### GPU vs CPU Performance\n");
    println!("| Size | CPU (ms) | Metal GPU (ms) | Speedup |");
    println!("|------|----------|----------------|---------|");

    let sizes = [256, 512, 1024, 2048];

    for &n in &sizes {
        let a: Vec<f32> = (0..n * n).map(|i| (i % 100) as f32).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i + 50) % 100) as f32).collect();

        // Warmup GPU
        let _ = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, n, n, &b, n);

        // Benchmark GPU
        let gpu_iters = if n <= 512 { 10 } else { 3 };
        let start = Instant::now();
        for _ in 0..gpu_iters {
            let _ = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, n, n, &b, n);
        }
        let gpu_ms = start.elapsed().as_secs_f64() * 1000.0 / gpu_iters as f64;

        // Benchmark CPU (fewer iterations for large sizes)
        let cpu_iters = if n <= 512 { 3 } else { 1 };
        let start = Instant::now();
        for _ in 0..cpu_iters {
            let _ = bench_cpu(&a, n, n, &b, n);
        }
        let cpu_ms = start.elapsed().as_secs_f64() * 1000.0 / cpu_iters as f64;

        let speedup = cpu_ms / gpu_ms;
        println!("| {:>4} | {:>8.1} | {:>14.3} | **{:.0}x** |",
                 n, cpu_ms, gpu_ms, speedup);
    }

    // All semirings benchmark
    println!("\n### All Semirings (Metal GPU Kernel Time)\n");
    println!("| Size | MaxPlus (ms) | MinPlus (ms) | MaxMul (ms) |");
    println!("|------|--------------|--------------|-------------|");

    for &n in &sizes {
        let a: Vec<f32> = (0..n * n).map(|i| (i % 100) as f32 + 1.0).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i + 50) % 100) as f32 + 1.0).collect();

        let iters = if n <= 512 { 10 } else { 3 };

        // MaxPlus
        let _ = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, n, n, &b, n);
        let start = Instant::now();
        for _ in 0..iters {
            let _ = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, n, n, &b, n);
        }
        let maxplus_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // MinPlus
        let _ = tropical_matmul_gpu::<TropicalMinPlus<f32>>(&a, n, n, &b, n);
        let start = Instant::now();
        for _ in 0..iters {
            let _ = tropical_matmul_gpu::<TropicalMinPlus<f32>>(&a, n, n, &b, n);
        }
        let minplus_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // MaxMul
        let _ = tropical_matmul_gpu::<TropicalMaxMul<f32>>(&a, n, n, &b, n);
        let start = Instant::now();
        for _ in 0..iters {
            let _ = tropical_matmul_gpu::<TropicalMaxMul<f32>>(&a, n, n, &b, n);
        }
        let maxmul_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        println!("| {:>4} | {:>12.3} | {:>12.3} | {:>11.3} |",
                 n, maxplus_ms, minplus_ms, maxmul_ms);
    }
}
