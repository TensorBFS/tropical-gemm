//! Benchmark for Tropical Automatic Differentiation (forward vs backward pass).
//!
//! This benchmark compares:
//! - Forward pass only (tropical_matmul_gpu)
//! - Forward pass with argmax tracking (tropical_matmul_gpu_with_argmax)
//! - Backward pass (gradient computation)
//! - Full forward + backward pass
//!
//! GPU only, MaxPlus algebra only.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tropical_gemm::TropicalMaxPlus;
use tropical_gemm_cuda::{
    tropical_backward_a_gpu, tropical_backward_b_gpu, tropical_matmul_gpu,
    tropical_matmul_gpu_batched, tropical_matmul_gpu_batched_with_argmax,
    tropical_matmul_gpu_with_argmax, CudaContext,
};

/// Check if CUDA is available
fn cuda_available() -> bool {
    CudaContext::new().is_ok()
}

/// Benchmark forward pass only (no argmax)
fn bench_forward(c: &mut Criterion) {
    if !cuda_available() {
        println!("CUDA not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("GPU_Forward");
    group.sample_size(20);

    for size in [256, 512, 1024, 2048].iter() {
        let n = *size;
        let elements = (n * n) as u64;

        let a: Vec<f32> = (0..n * n).map(|i| ((i % 1000) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n * n)
            .map(|i| (((i + 500) % 1000) as f32) * 0.01)
            .collect();

        group.throughput(Throughput::Elements(elements * 2));

        group.bench_with_input(BenchmarkId::new("MaxPlus", n), &n, |bench, &n| {
            bench.iter(|| {
                black_box(
                    tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, n, n, &b, n).unwrap(),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark forward pass with argmax tracking
fn bench_forward_with_argmax(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }

    let mut group = c.benchmark_group("GPU_Forward_Argmax");
    group.sample_size(20);

    for size in [256, 512, 1024, 2048].iter() {
        let n = *size;
        let elements = (n * n) as u64;

        let a: Vec<f32> = (0..n * n).map(|i| ((i % 1000) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n * n)
            .map(|i| (((i + 500) % 1000) as f32) * 0.01)
            .collect();

        group.throughput(Throughput::Elements(elements * 2));

        group.bench_with_input(BenchmarkId::new("MaxPlus", n), &n, |bench, &n| {
            bench.iter(|| {
                black_box(
                    tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, n, n, &b, n)
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark backward pass only (given pre-computed argmax)
fn bench_backward(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }

    let mut group = c.benchmark_group("GPU_Backward");
    group.sample_size(20);

    for size in [256, 512, 1024, 2048].iter() {
        let n = *size;
        let m = n;
        let k = n;
        let elements = (m * k + k * n) as u64; // grad_a + grad_b sizes

        let a: Vec<f32> = (0..m * k).map(|i| ((i % 1000) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| (((i + 500) % 1000) as f32) * 0.01)
            .collect();

        // Pre-compute forward pass with argmax
        let (_, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n).unwrap();

        // Upstream gradient (all ones)
        let grad_c: Vec<f32> = vec![1.0; m * n];

        group.throughput(Throughput::Elements(elements));

        group.bench_with_input(BenchmarkId::new("backward_a", n), &n, |bench, _| {
            bench.iter(|| {
                black_box(tropical_backward_a_gpu(&grad_c, &argmax, m, k, n))
            });
        });

        group.bench_with_input(BenchmarkId::new("backward_b", n), &n, |bench, _| {
            bench.iter(|| {
                black_box(tropical_backward_b_gpu(&grad_c, &argmax, m, k, n))
            });
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

/// Benchmark full forward + backward pass (tropical AD)
fn bench_full_ad(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }

    let mut group = c.benchmark_group("GPU_Full_AD");
    group.sample_size(20);

    for size in [256, 512, 1024, 2048].iter() {
        let n = *size;
        let m = n;
        let k = n;

        let a: Vec<f32> = (0..m * k).map(|i| ((i % 1000) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| (((i + 500) % 1000) as f32) * 0.01)
            .collect();
        let grad_c: Vec<f32> = vec![1.0; m * n];

        group.bench_with_input(BenchmarkId::new("forward+backward", n), &n, |bench, _| {
            bench.iter(|| {
                // Forward pass with argmax
                let (c, argmax) =
                    tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)
                        .unwrap();

                // Backward pass
                let grad_a = tropical_backward_a_gpu(&grad_c, &argmax, m, k, n);
                let grad_b = tropical_backward_b_gpu(&grad_c, &argmax, m, k, n);

                black_box((c, grad_a, grad_b))
            });
        });
    }

    group.finish();
}

/// Benchmark batched forward pass
fn bench_batched_forward(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }

    let mut group = c.benchmark_group("GPU_Batched_Forward");
    group.sample_size(20);

    let n = 512; // Fixed matrix size for batched tests
    let m = n;
    let k = n;

    for batch_size in [4, 8, 16, 32].iter() {
        let bs = *batch_size;

        let a_batch: Vec<Vec<f32>> = (0..bs)
            .map(|b| {
                (0..m * k)
                    .map(|i| (((i + b * 100) % 1000) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let b_batch: Vec<Vec<f32>> = (0..bs)
            .map(|b| {
                (0..k * n)
                    .map(|i| (((i + b * 200 + 500) % 1000) as f32) * 0.01)
                    .collect()
            })
            .collect();

        let total_elements = (bs * m * n) as u64;
        group.throughput(Throughput::Elements(total_elements));

        group.bench_with_input(
            BenchmarkId::new("MaxPlus", format!("{}x{}x{}", bs, n, n)),
            &bs,
            |bench, _| {
                bench.iter(|| {
                    black_box(
                        tropical_matmul_gpu_batched::<TropicalMaxPlus<f32>>(
                            &a_batch, &b_batch, m, k, n,
                        )
                        .unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batched forward with argmax
fn bench_batched_forward_argmax(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }

    let mut group = c.benchmark_group("GPU_Batched_Forward_Argmax");
    group.sample_size(20);

    let n = 512;
    let m = n;
    let k = n;

    for batch_size in [4, 8, 16, 32].iter() {
        let bs = *batch_size;

        let a_batch: Vec<Vec<f32>> = (0..bs)
            .map(|b| {
                (0..m * k)
                    .map(|i| (((i + b * 100) % 1000) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let b_batch: Vec<Vec<f32>> = (0..bs)
            .map(|b| {
                (0..k * n)
                    .map(|i| (((i + b * 200 + 500) % 1000) as f32) * 0.01)
                    .collect()
            })
            .collect();

        let total_elements = (bs * m * n) as u64;
        group.throughput(Throughput::Elements(total_elements));

        group.bench_with_input(
            BenchmarkId::new("MaxPlus", format!("{}x{}x{}", bs, n, n)),
            &bs,
            |bench, _| {
                bench.iter(|| {
                    black_box(
                        tropical_matmul_gpu_batched_with_argmax::<TropicalMaxPlus<f32>>(
                            &a_batch, &b_batch, m, k, n,
                        )
                        .unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batched full AD (forward + backward)
fn bench_batched_full_ad(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }

    let mut group = c.benchmark_group("GPU_Batched_Full_AD");
    group.sample_size(20);

    let n = 512;
    let m = n;
    let k = n;

    for batch_size in [4, 8, 16, 32].iter() {
        let bs = *batch_size;

        let a_batch: Vec<Vec<f32>> = (0..bs)
            .map(|b| {
                (0..m * k)
                    .map(|i| (((i + b * 100) % 1000) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let b_batch: Vec<Vec<f32>> = (0..bs)
            .map(|b| {
                (0..k * n)
                    .map(|i| (((i + b * 200 + 500) % 1000) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let grad_c_batch: Vec<Vec<f32>> = (0..bs).map(|_| vec![1.0f32; m * n]).collect();

        group.bench_with_input(
            BenchmarkId::new("forward+backward", format!("{}x{}x{}", bs, n, n)),
            &bs,
            |bench, _| {
                bench.iter(|| {
                    // Forward pass with argmax
                    let results = tropical_matmul_gpu_batched_with_argmax::<TropicalMaxPlus<f32>>(
                        &a_batch, &b_batch, m, k, n,
                    )
                    .unwrap();

                    // Extract argmax for backward
                    let argmax_batch: Vec<_> =
                        results.iter().map(|(_, argmax)| argmax.clone()).collect();

                    // Backward pass
                    let grad_a_batch: Vec<_> = grad_c_batch
                        .iter()
                        .zip(argmax_batch.iter())
                        .map(|(grad_c, argmax)| tropical_backward_a_gpu(grad_c, argmax, m, k, n))
                        .collect();
                    let grad_b_batch: Vec<_> = grad_c_batch
                        .iter()
                        .zip(argmax_batch.iter())
                        .map(|(grad_c, argmax)| tropical_backward_b_gpu(grad_c, argmax, m, k, n))
                        .collect();

                    black_box((results, grad_a_batch, grad_b_batch))
                });
            },
        );
    }

    group.finish();
}

/// Compare forward vs forward+argmax overhead
fn bench_argmax_overhead(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }

    let mut group = c.benchmark_group("GPU_Argmax_Overhead");
    group.sample_size(20);

    for size in [256, 512, 1024, 2048].iter() {
        let n = *size;

        let a: Vec<f32> = (0..n * n).map(|i| ((i % 1000) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n * n)
            .map(|i| (((i + 500) % 1000) as f32) * 0.01)
            .collect();

        group.bench_with_input(BenchmarkId::new("forward_only", n), &n, |bench, &n| {
            bench.iter(|| {
                black_box(
                    tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, n, n, &b, n).unwrap(),
                )
            });
        });

        group.bench_with_input(BenchmarkId::new("forward_with_argmax", n), &n, |bench, &n| {
            bench.iter(|| {
                black_box(
                    tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, n, n, &b, n)
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_forward,
    bench_forward_with_argmax,
    bench_backward,
    bench_full_ad,
    bench_batched_forward,
    bench_batched_forward_argmax,
    bench_batched_full_ad,
    bench_argmax_overhead,
);
criterion_main!(benches);
