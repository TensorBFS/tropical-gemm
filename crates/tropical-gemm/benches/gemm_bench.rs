use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tropical_gemm::prelude::*;
use tropical_gemm::TropicalScalar; // for `neg_infinity()` (not in the prelude)

fn bench_tropical_gemm_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("TropicalGemm_f32");
    group.sample_size(20);

    // Matrix sizes to benchmark (matching Julia comparison)
    for size in [128, 256, 512, 1024, 2048, 4096].iter() {
        let n = *size;
        let elements = (n * n) as u64;

        // Create test matrices
        let a: Vec<f32> = (0..n * n).map(|i| ((i % 1000) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n * n)
            .map(|i| (((i + 500) % 1000) as f32) * 0.01)
            .collect();

        group.throughput(Throughput::Elements(elements * 2)); // 2 matrices read

        group.bench_with_input(BenchmarkId::new("MaxPlus", n), &n, |bench, &n| {
            bench.iter(|| black_box(tropical_matmul::<TropicalMaxPlus<f32>>(&a, n, n, &b, n)));
        });

        group.bench_with_input(BenchmarkId::new("MinPlus", n), &n, |bench, &n| {
            bench.iter(|| black_box(tropical_matmul::<TropicalMinPlus<f32>>(&a, n, n, &b, n)));
        });

        group.bench_with_input(BenchmarkId::new("MaxMul", n), &n, |bench, &n| {
            bench.iter(|| black_box(tropical_matmul::<TropicalMaxMul<f32>>(&a, n, n, &b, n)));
        });
    }

    group.finish();
}

fn bench_tropical_gemm_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("TropicalGemm_f64");
    group.sample_size(20);

    for size in [128, 256, 512, 1024, 2048].iter() {
        let n = *size;
        let elements = (n * n) as u64;

        let a: Vec<f64> = (0..n * n).map(|i| ((i % 1000) as f64) * 0.01).collect();
        let b: Vec<f64> = (0..n * n)
            .map(|i| (((i + 500) % 1000) as f64) * 0.01)
            .collect();

        group.throughput(Throughput::Elements(elements * 2));

        group.bench_with_input(BenchmarkId::new("MaxPlus", n), &n, |bench, &n| {
            bench.iter(|| black_box(tropical_matmul::<TropicalMaxPlus<f64>>(&a, n, n, &b, n)));
        });
    }

    group.finish();
}

fn bench_with_argmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("TropicalGemm_Argmax");
    group.sample_size(20);

    for size in [128, 256, 512, 1024].iter() {
        let n = *size;
        let a: Vec<f64> = (0..n * n).map(|i| ((i % 100) as f64) * 0.1).collect();
        let b: Vec<f64> = (0..n * n)
            .map(|i| (((i + 50) % 100) as f64) * 0.1)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("MaxPlus_with_argmax", n),
            &n,
            |bench, &n| {
                bench.iter(|| {
                    black_box(tropical_matmul_with_argmax::<TropicalMaxPlus<f64>>(
                        &a, n, n, &b, n,
                    ))
                });
            },
        );

        // Integer (i32) argmax — the path the write-back canonicalization touches.
        // Make every 8th row of A the tropical zero (`-∞`) so its whole output row
        // is a "no contribution" cell; otherwise the canonicalization branch never
        // fires and the bench would only measure the guard-free add.
        let neg = <i32 as TropicalScalar>::neg_infinity();
        let ai: Vec<i32> = (0..n * n)
            .map(|i| if (i / n) % 8 == 0 { neg } else { (i % 100) as i32 })
            .collect();
        let bi: Vec<i32> = (0..n * n).map(|i| ((i + 50) % 100) as i32).collect();

        group.bench_with_input(
            BenchmarkId::new("MaxPlus_i32_with_argmax", n),
            &n,
            |bench, &n| {
                bench.iter(|| {
                    black_box(tropical_matmul_with_argmax::<TropicalMaxPlus<i32>>(
                        &ai, n, n, &bi, n,
                    ))
                });
            },
        );
    }

    group.finish();
}

/// Quick benchmark for command-line timing comparison
fn bench_quick(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quick");
    group.sample_size(10);

    // Standard comparison sizes
    for size in [1024, 2048, 4096].iter() {
        let n = *size;
        let a: Vec<f32> = (0..n * n).map(|i| ((i % 1000) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n * n)
            .map(|i| (((i + 500) % 1000) as f32) * 0.01)
            .collect();

        group.bench_with_input(BenchmarkId::new("MaxPlus_f32", n), &n, |bench, &n| {
            bench.iter(|| black_box(tropical_matmul::<TropicalMaxPlus<f32>>(&a, n, n, &b, n)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tropical_gemm_f32,
    bench_tropical_gemm_f64,
    bench_with_argmax,
    bench_quick
);
criterion_main!(benches);
