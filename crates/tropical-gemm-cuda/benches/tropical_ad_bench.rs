//! Benchmark for Tropical Automatic Differentiation (forward vs backward pass).
//!
//! This benchmark compares:
//! - Forward pass only (tropical_matmul_gpu)
//! - Forward pass with argmax tracking (tropical_matmul_gpu_with_argmax)
//! - Backward pass (gradient computation)
//!
//! GPU only, MaxPlus algebra only.
//! Uses persistent CudaContext to exclude JIT compilation overhead.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tropical_gemm::TropicalMaxPlus;
use tropical_gemm_cuda::{
    tropical_backward_a_gpu, tropical_backward_b_gpu, tropical_matmul_gpu_with_ctx,
    tropical_matmul_gpu_with_ctx_and_argmax, CudaContext, GpuMatrix,
};

/// Check if CUDA is available
fn cuda_available() -> bool {
    CudaContext::new().is_ok()
}

/// Benchmark forward pass without argmax (kernel only, no JIT)
fn bench_forward_no_argmax(c: &mut Criterion) {
    if !cuda_available() {
        println!("CUDA not available, skipping GPU benchmarks");
        return;
    }

    let ctx = CudaContext::new().unwrap();

    let mut group = c.benchmark_group("GPU_Forward_NoArgmax");
    group.sample_size(50);

    for size in [256, 512, 1024, 2048].iter() {
        let n = *size;
        let m = n;
        let k = n;
        let elements = (m * n) as u64;

        // Prepare data on GPU (outside benchmark)
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 1000) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| (((i + 500) % 1000) as f32) * 0.01)
            .collect();

        let a_gpu = GpuMatrix::from_host_row_major(&ctx, &a, m, k).unwrap();
        let b_gpu = GpuMatrix::from_host_row_major(&ctx, &b, k, n).unwrap();

        // Warm up - compile kernel
        let _ = tropical_matmul_gpu_with_ctx::<TropicalMaxPlus<f32>>(&ctx, &a_gpu, &b_gpu).unwrap();

        group.throughput(Throughput::Elements(elements));

        group.bench_with_input(BenchmarkId::new("MaxPlus", n), &n, |bench, _| {
            bench.iter(|| {
                black_box(
                    tropical_matmul_gpu_with_ctx::<TropicalMaxPlus<f32>>(&ctx, &a_gpu, &b_gpu)
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark forward pass with argmax tracking (kernel only, no JIT)
fn bench_forward_with_argmax(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new().unwrap();

    let mut group = c.benchmark_group("GPU_Forward_WithArgmax");
    group.sample_size(50);

    for size in [256, 512, 1024, 2048].iter() {
        let n = *size;
        let m = n;
        let k = n;
        let elements = (m * n) as u64;

        let a: Vec<f32> = (0..m * k).map(|i| ((i % 1000) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| (((i + 500) % 1000) as f32) * 0.01)
            .collect();

        let a_gpu = GpuMatrix::from_host_row_major(&ctx, &a, m, k).unwrap();
        let b_gpu = GpuMatrix::from_host_row_major(&ctx, &b, k, n).unwrap();

        // Warm up - compile kernel
        let _ = tropical_matmul_gpu_with_ctx_and_argmax::<TropicalMaxPlus<f32>>(&ctx, &a_gpu, &b_gpu)
            .unwrap();

        group.throughput(Throughput::Elements(elements));

        group.bench_with_input(BenchmarkId::new("MaxPlus", n), &n, |bench, _| {
            bench.iter(|| {
                black_box(
                    tropical_matmul_gpu_with_ctx_and_argmax::<TropicalMaxPlus<f32>>(
                        &ctx, &a_gpu, &b_gpu,
                    )
                    .unwrap(),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark backward pass only (CPU, given pre-computed argmax)
fn bench_backward(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new().unwrap();

    let mut group = c.benchmark_group("GPU_Backward");
    group.sample_size(50);

    for size in [256, 512, 1024, 2048].iter() {
        let n = *size;
        let m = n;
        let k = n;
        let elements = (m * k + k * n) as u64; // grad_a + grad_b sizes

        let a: Vec<f32> = (0..m * k).map(|i| ((i % 1000) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| (((i + 500) % 1000) as f32) * 0.01)
            .collect();

        let a_gpu = GpuMatrix::from_host_row_major(&ctx, &a, m, k).unwrap();
        let b_gpu = GpuMatrix::from_host_row_major(&ctx, &b, k, n).unwrap();

        // Pre-compute forward pass with argmax
        let c_gpu =
            tropical_matmul_gpu_with_ctx_and_argmax::<TropicalMaxPlus<f32>>(&ctx, &a_gpu, &b_gpu)
                .unwrap();
        let argmax = c_gpu.argmax_to_host_row_major(&ctx).unwrap();

        // Upstream gradient (all ones)
        let grad_c: Vec<f32> = vec![1.0; m * n];

        group.throughput(Throughput::Elements(elements));

        group.bench_with_input(BenchmarkId::new("backward_a", n), &n, |bench, _| {
            bench.iter(|| black_box(tropical_backward_a_gpu(&grad_c, &argmax, m, k, n)));
        });

        group.bench_with_input(BenchmarkId::new("backward_b", n), &n, |bench, _| {
            bench.iter(|| black_box(tropical_backward_b_gpu(&grad_c, &argmax, m, k, n)));
        });

        group.bench_with_input(BenchmarkId::new("backward_both", n), &n, |bench, _| {
            bench.iter(|| {
                let grad_a = tropical_backward_a_gpu(&grad_c, &argmax, m, k, n);
                let grad_b = tropical_backward_b_gpu(&grad_c, &argmax, m, k, n);
                black_box((grad_a, grad_b))
            });
        });
    }

    group.finish();
}

/// Compare forward vs forward+argmax vs backward at same sizes
fn bench_comparison(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new().unwrap();

    let mut group = c.benchmark_group("GPU_Comparison");
    group.sample_size(50);

    for size in [256, 512, 1024, 2048].iter() {
        let n = *size;
        let m = n;
        let k = n;
        let elements = (m * n) as u64;

        let a: Vec<f32> = (0..m * k).map(|i| ((i % 1000) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| (((i + 500) % 1000) as f32) * 0.01)
            .collect();

        let a_gpu = GpuMatrix::from_host_row_major(&ctx, &a, m, k).unwrap();
        let b_gpu = GpuMatrix::from_host_row_major(&ctx, &b, k, n).unwrap();

        // Warm up both kernels
        let _ = tropical_matmul_gpu_with_ctx::<TropicalMaxPlus<f32>>(&ctx, &a_gpu, &b_gpu).unwrap();
        let c_argmax =
            tropical_matmul_gpu_with_ctx_and_argmax::<TropicalMaxPlus<f32>>(&ctx, &a_gpu, &b_gpu)
                .unwrap();
        let argmax = c_argmax.argmax_to_host_row_major(&ctx).unwrap();
        let grad_c: Vec<f32> = vec![1.0; m * n];

        group.throughput(Throughput::Elements(elements));

        // Forward without argmax
        group.bench_with_input(
            BenchmarkId::new("forward_no_argmax", n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    black_box(
                        tropical_matmul_gpu_with_ctx::<TropicalMaxPlus<f32>>(&ctx, &a_gpu, &b_gpu)
                            .unwrap(),
                    )
                });
            },
        );

        // Forward with argmax
        group.bench_with_input(
            BenchmarkId::new("forward_with_argmax", n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    black_box(
                        tropical_matmul_gpu_with_ctx_and_argmax::<TropicalMaxPlus<f32>>(
                            &ctx, &a_gpu, &b_gpu,
                        )
                        .unwrap(),
                    )
                });
            },
        );

        // Backward (both gradients)
        group.bench_with_input(BenchmarkId::new("backward_both", n), &n, |bench, _| {
            bench.iter(|| {
                let grad_a = tropical_backward_a_gpu(&grad_c, &argmax, m, k, n);
                let grad_b = tropical_backward_b_gpu(&grad_c, &argmax, m, k, n);
                black_box((grad_a, grad_b))
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_forward_no_argmax,
    bench_forward_with_argmax,
    bench_backward,
    bench_comparison,
);
criterion_main!(benches);
