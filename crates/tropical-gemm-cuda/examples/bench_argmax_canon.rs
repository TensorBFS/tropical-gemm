//! Correctness + throughput for the integer argmax **zero-cell canonicalization**
//! (step 2: unify a tropical-zero output cell's argmax on the seed `0`).
//!
//! Background: the guard-free integer kernels let a tropical-zero cell's value
//! drift to `S + data`, and its argmax (updated by `prod > accum`) then adopts a
//! data-dependent `k` instead of a fixed value. The shipped `*_with_argmax`
//! I32/I64 kernels now canonicalize that index at write-out: if the accumulated
//! value is in sentinel territory (`<= NEG_INF/2` for maxplus, `>= INF/2` for
//! minplus) the argmax is reset to `0`. The canonicalization is a single
//! predicated select per output element in the O(M*N) epilogue — NOT in the
//! O(M*N*K) inner loop — so it should be free.
//!
//! This example proves both, head-to-head in ONE build (same GPU, no drift):
//!   * the SHIPPED kernel (`zero_maxplus_i32` -> canon), and
//!   * a bench-only `*_nocanon` variant instantiated with `zero_never_i32`,
//!     which reproduces the pre-step-2 behavior (write the drifted index).
//!
//! Run on a CUDA machine:
//!     cargo run --release --example bench_argmax_canon -p tropical-gemm-cuda

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use tropical_gemm_cuda::SENTINEL_I32;

const KERNEL_SRC: &str = include_str!("../kernels/tropical_gemm.cu");

/// maxplus i32 tropical zero (-inf) = -SENTINEL_I32 = -1e9.
const S: i32 = -SENTINEL_I32;

/// Bench-only no-canon variant: same kernel, `zero_never_i32` => write the
/// drifted accum_idx verbatim (the pre-canonicalization behavior). Defined via
/// the SAME shipped macro so the only difference is the ZERO_FN argument.
const NOCANON_SRC: &str = r#"
TROPICAL_GEMM_ARGMAX_I32(tropical_maxplus_i32_nn_with_argmax_nocanon, NEG_INF_I32, >, add_i32, zero_never_i32)
"#;

const CANON: &str = "tropical_maxplus_i32_nn_with_argmax";
const NOCANON: &str = "tropical_maxplus_i32_nn_with_argmax_nocanon";

const WARMUP: usize = 10;
const ITERS: usize = 50;

type R<T> = Result<T, Box<dyn std::error::Error>>;

/// i32 argmax kernels use a 64x64 tile with a 16x16 thread block.
fn gemm_cfg(m: usize, n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((m as u32).div_ceil(64) * (n as u32).div_ceil(64), 1, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    }
}

/// One i32 NN GEMM with argmax (column-major), returning (C, argmax).
fn argmax_gemm(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    name: &str,
    a: &[i32],
    b: &[i32],
    m: usize,
    k: usize,
    n: usize,
) -> R<(Vec<i32>, Vec<i32>)> {
    let da = stream.clone_htod(a)?;
    let db = stream.clone_htod(b)?;
    let mut dc = stream.alloc_zeros::<i32>(m * n)?;
    let mut dam = stream.alloc_zeros::<i32>(m * n)?;
    let f = module.load_function(name).unwrap();
    let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
    let mut launch_args = stream.launch_builder(&f);
    launch_args
        .arg(&da)
        .arg(&db)
        .arg(&mut dc)
        .arg(&mut dam)
        .arg(&m_i)
        .arg(&n_i)
        .arg(&k_i);
    unsafe {
        launch_args.launch(gemm_cfg(m, n))?;
    }
    Ok((stream.clone_dtoh(&dc)?, stream.clone_dtoh(&dam)?))
}

/// Time `name`'s argmax GEMM (timing only, zero-filled inputs). The canon is a
/// branchless `ZERO_FN(accum) ? 0 : accum_idx` predicated select, so its compare
/// runs on every output cell regardless of outcome — this measures its full cost
/// even though zero-filled cells are finite (the select just keeps accum_idx).
fn bench(stream: &Arc<CudaStream>, module: &Arc<CudaModule>, name: &str, n: usize) -> R<f64> {
    let a = stream.alloc_zeros::<i32>(n * n)?;
    let b = stream.alloc_zeros::<i32>(n * n)?;
    let mut c = stream.alloc_zeros::<i32>(n * n)?;
    let mut am = stream.alloc_zeros::<i32>(n * n)?;
    let cfg = gemm_cfg(n, n);
    let f = module.load_function(name).unwrap();
    let n_i = n as i32;
    let run = |c: &mut cudarc::driver::CudaSlice<i32>,
               am: &mut cudarc::driver::CudaSlice<i32>|
     -> R<()> {
        let mut launch_args = stream.launch_builder(&f);
        launch_args
            .arg(&a)
            .arg(&b)
            .arg(&mut *c)
            .arg(&mut *am)
            .arg(&n_i)
            .arg(&n_i)
            .arg(&n_i);
        unsafe {
            launch_args.launch(cfg)?;
        }
        Ok(())
    };
    for _ in 0..WARMUP {
        run(&mut c, &mut am)?;
    }
    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..ITERS {
        run(&mut c, &mut am)?;
    }
    stream.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / ITERS as f64)
}

fn main() -> R<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(format!("{KERNEL_SRC}\n{NOCANON_SRC}"))?;
    let module = ctx.load_module(ptx)?;

    println!("=== correctness: tropical-zero cell argmax canonicalization (maxplus i32) ===");

    // Drift case: A row is all -inf (no edges), B finite. C[0,0] = max(S+5, S+7)
    // = S+7 (drifts, k=1). The cell is mathematically -inf, so its k is meaningless.
    let (c_c, am_c) = argmax_gemm(&stream, &module,CANON, &[S, S], &[5, 7], 1, 2, 1)?;
    let (c_n, am_n) = argmax_gemm(&stream, &module,NOCANON, &[S, S], &[5, 7], 1, 2, 1)?;
    println!("  A=[-inf,-inf] B=[5,7]  (cell is tropical zero):");
    println!("    canon   : C={:>12} argmax={}", c_c[0], am_c[0]);
    println!("    nocanon : C={:>12} argmax={}", c_n[0], am_n[0]);
    assert!(c_c[0] <= S / 2, "canon value must stay in sentinel territory");
    assert_eq!(am_c[0], 0, "canon: zero-cell argmax must be 0");
    assert_eq!(am_n[0], 1, "nocanon: drifts to the data-dependent k=1");
    println!("    -> canon pins it to 0; nocanon drifts to k=1 (data-dependent).");

    // Order-independence: swapping B must not change the canon index (still 0).
    let (_, am_57) = argmax_gemm(&stream, &module,CANON, &[S, S], &[5, 7], 1, 2, 1)?;
    let (_, am_75) = argmax_gemm(&stream, &module,CANON, &[S, S], &[7, 5], 1, 2, 1)?;
    assert_eq!((am_57[0], am_75[0]), (0, 0), "canon index must be order-independent");
    println!("  B order swap [5,7] vs [7,5]: canon argmax = {} / {} (stable)", am_57[0], am_75[0]);

    // Sanity: a genuinely finite cell keeps the CORRECT argmax under canon.
    // A=[1,5,3], B=[2,4,0]: k=1 gives 5+4=9 (max).
    let (c_f, am_f) = argmax_gemm(&stream, &module,CANON, &[1, 5, 3], &[2, 4, 0], 1, 3, 1)?;
    assert_eq!((c_f[0], am_f[0]), (9, 1), "finite-cell argmax must stay correct");
    println!("  finite A=[1,5,3] B=[2,4,0]: C={} argmax={} (expect 9,1) -> correct\n", c_f[0], am_f[0]);

    println!("=== throughput: canon vs nocanon argmax (64x64 tile), ms/iter ===");
    println!("  (canon compare+select runs on every output cell regardless of outcome)");
    println!("{:>6} | {:>12} {:>12} | {:>10}", "size", "nocanon", "canon", "overhead");
    println!("{}", "-".repeat(50));
    for &sz in &[1024usize, 2048, 4096] {
        let nocanon = bench(&stream, &module,NOCANON, sz)?;
        let canon = bench(&stream, &module,CANON, sz)?;
        let overhead = (canon / nocanon - 1.0) * 100.0;
        println!("{sz:>6} | {nocanon:>12.4} {canon:>12.4} | {overhead:>+9.2}%");
    }
    println!("\n(overhead = canon/nocanon - 1; expected ~0: epilogue O(M*N), not the inner loop)");
    Ok(())
}
