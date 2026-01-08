//! Benchmark comparing CUDA vs CPU backends
//!
//! Run with: cargo run --release --example bench_cuda_vs_cpu -p tropical-gemm-cuda

use std::time::Instant;
use tropical_gemm::{tropical_matmul, TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus};
use tropical_gemm_cuda::{tropical_matmul_gpu, CudaContext, CudaError, GpuMatrix};

const SIZES: &[usize] = &[256, 512, 1024, 2048];
const WARMUP_ITERS: usize = 1;
const BENCH_ITERS: usize = 3;

fn bench_cpu_maxplus_f32(a: &[f32], b: &[f32], n: usize) -> f64 {
    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = tropical_matmul::<TropicalMaxPlus<f32>>(a, n, n, b, n);
    }

    // Benchmark
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = tropical_matmul::<TropicalMaxPlus<f32>>(a, n, n, b, n);
        times.push(start.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[0]
}

fn bench_cpu_minplus_f32(a: &[f32], b: &[f32], n: usize) -> f64 {
    for _ in 0..WARMUP_ITERS {
        let _ = tropical_matmul::<TropicalMinPlus<f32>>(a, n, n, b, n);
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = tropical_matmul::<TropicalMinPlus<f32>>(a, n, n, b, n);
        times.push(start.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[0]
}

fn bench_cpu_maxmul_f32(a: &[f32], b: &[f32], n: usize) -> f64 {
    for _ in 0..WARMUP_ITERS {
        let _ = tropical_matmul::<TropicalMaxMul<f32>>(a, n, n, b, n);
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = tropical_matmul::<TropicalMaxMul<f32>>(a, n, n, b, n);
        times.push(start.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[0]
}

fn bench_gpu_oneshot<T: tropical_gemm_cuda::CudaKernel>(
    a: &[T::Scalar],
    b: &[T::Scalar],
    n: usize,
) -> Result<f64, CudaError>
where
    T::Scalar: cudarc::driver::DeviceRepr + Default + Clone + cudarc::driver::ValidAsZeroBits,
{
    // Warmup (includes context creation overhead)
    for _ in 0..WARMUP_ITERS {
        let _ = tropical_matmul_gpu::<T>(a, n, n, b, n)?;
    }

    // Benchmark
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = tropical_matmul_gpu::<T>(a, n, n, b, n)?;
        times.push(start.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[0])
}

fn bench_gpu_persistent<T: tropical_gemm_cuda::CudaKernel>(
    ctx: &CudaContext,
    a: &[T::Scalar],
    b: &[T::Scalar],
    n: usize,
) -> Result<f64, CudaError>
where
    T::Scalar: cudarc::driver::DeviceRepr + Default + Clone + cudarc::driver::ValidAsZeroBits,
{
    // Pre-allocate GPU memory
    let a_gpu = GpuMatrix::from_host_row_major(ctx, a, n, n)?;
    let b_gpu = GpuMatrix::from_host_row_major(ctx, b, n, n)?;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = tropical_gemm_cuda::tropical_matmul_gpu_with_ctx::<T>(ctx, &a_gpu, &b_gpu)?;
    }

    // Benchmark (kernel only, no transfer)
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = tropical_gemm_cuda::tropical_matmul_gpu_with_ctx::<T>(ctx, &a_gpu, &b_gpu)?;
        times.push(start.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[0])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("Tropical GEMM: CUDA vs CPU Benchmark");
    println!("{}", "=".repeat(80));
    println!();

    // Initialize CUDA context
    println!("Initializing CUDA...");
    let ctx = match CudaContext::new() {
        Ok(c) => {
            println!("CUDA initialized: {}", c.device_name());
            c
        }
        Err(e) => {
            eprintln!("Failed to initialize CUDA: {:?}", e);
            eprintln!("Make sure CUDA drivers and toolkit are installed.");
            return Ok(());
        }
    };
    println!();

    // Generate test data
    let max_n = *SIZES.last().unwrap();
    let a_f32: Vec<f32> = (0..max_n * max_n)
        .map(|i| ((i % 1000) as f32) * 0.01)
        .collect();
    let b_f32: Vec<f32> = (0..max_n * max_n)
        .map(|i| (((i + 500) % 1000) as f32) * 0.01)
        .collect();

    // Benchmark TropicalMaxPlus<f32>
    println!("{}", "-".repeat(80));
    println!("TropicalMaxPlus<f32>");
    println!("{}", "-".repeat(80));
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>10}  {:>10}",
        "Size", "CPU (ms)", "GPU (ms)", "GPU-Pers", "Speedup", "Speedup-P"
    );

    for &n in SIZES {
        let a = &a_f32[..n * n];
        let b = &b_f32[..n * n];

        let cpu_time = bench_cpu_maxplus_f32(a, b, n);
        let gpu_time = bench_gpu_oneshot::<TropicalMaxPlus<f32>>(a, b, n)?;
        let gpu_pers_time = bench_gpu_persistent::<TropicalMaxPlus<f32>>(&ctx, a, b, n)?;

        let speedup = cpu_time / gpu_time;
        let speedup_pers = cpu_time / gpu_pers_time;

        println!(
            "{:>6}  {:>12.3}  {:>12.3}  {:>12.3}  {:>10.1}x  {:>10.1}x",
            n,
            cpu_time * 1000.0,
            gpu_time * 1000.0,
            gpu_pers_time * 1000.0,
            speedup,
            speedup_pers
        );
    }
    println!();

    // Benchmark TropicalMinPlus<f32>
    println!("{}", "-".repeat(80));
    println!("TropicalMinPlus<f32>");
    println!("{}", "-".repeat(80));
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>10}  {:>10}",
        "Size", "CPU (ms)", "GPU (ms)", "GPU-Pers", "Speedup", "Speedup-P"
    );

    for &n in SIZES {
        let a = &a_f32[..n * n];
        let b = &b_f32[..n * n];

        let cpu_time = bench_cpu_minplus_f32(a, b, n);
        let gpu_time = bench_gpu_oneshot::<TropicalMinPlus<f32>>(a, b, n)?;
        let gpu_pers_time = bench_gpu_persistent::<TropicalMinPlus<f32>>(&ctx, a, b, n)?;

        let speedup = cpu_time / gpu_time;
        let speedup_pers = cpu_time / gpu_pers_time;

        println!(
            "{:>6}  {:>12.3}  {:>12.3}  {:>12.3}  {:>10.1}x  {:>10.1}x",
            n,
            cpu_time * 1000.0,
            gpu_time * 1000.0,
            gpu_pers_time * 1000.0,
            speedup,
            speedup_pers
        );
    }
    println!();

    // Benchmark TropicalMaxMul<f32>
    println!("{}", "-".repeat(80));
    println!("TropicalMaxMul<f32>");
    println!("{}", "-".repeat(80));
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>10}  {:>10}",
        "Size", "CPU (ms)", "GPU (ms)", "GPU-Pers", "Speedup", "Speedup-P"
    );

    for &n in SIZES {
        let a = &a_f32[..n * n];
        let b = &b_f32[..n * n];

        let cpu_time = bench_cpu_maxmul_f32(a, b, n);
        let gpu_time = bench_gpu_oneshot::<TropicalMaxMul<f32>>(a, b, n)?;
        let gpu_pers_time = bench_gpu_persistent::<TropicalMaxMul<f32>>(&ctx, a, b, n)?;

        let speedup = cpu_time / gpu_time;
        let speedup_pers = cpu_time / gpu_pers_time;

        println!(
            "{:>6}  {:>12.3}  {:>12.3}  {:>12.3}  {:>10.1}x  {:>10.1}x",
            n,
            cpu_time * 1000.0,
            gpu_time * 1000.0,
            gpu_pers_time * 1000.0,
            speedup,
            speedup_pers
        );
    }
    println!();

    // Summary
    println!("{}", "=".repeat(80));
    println!("Legend:");
    println!("  CPU       - CPU backend (SIMD-optimized)");
    println!("  GPU       - GPU with one-shot API (includes context + transfer overhead)");
    println!("  GPU-Pers  - GPU with persistent context (kernel only, data pre-loaded)");
    println!("  Speedup   - CPU time / GPU time");
    println!("  Speedup-P - CPU time / GPU-Pers time");
    println!("{}", "=".repeat(80));

    Ok(())
}
