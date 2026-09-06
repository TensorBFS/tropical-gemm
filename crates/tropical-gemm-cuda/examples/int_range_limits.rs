//! Empirically map the safe integer data range for the guard-free tropical
//! kernels (issue #48) and show exactly where results stop being reliable —
//! i.e. when you must switch i32 -> i64.
//!
//! The tropical zero (-inf) is a large sentinel S (i32 -1e9, i64 -2^60); multiply
//! is a bare add with no guard/clamp. The failure mode is a real path being
//! masked by a "no-edge times data" term:
//!
//!   C[i,j] = max_k (A[i,k] + B[k,j])
//!   k=0 (real edge):   a0 = -D,  b0 = -D   -> term = -2D   (the TRUE answer)
//!   k=1 (no edge):     a1 = S,   b1 = +D   -> term = S + D (sentinel-derived)
//!
//! Tropically, k=1 contributes S(+)b1 = -inf, so the true result is -2D. The
//! kernel computes max(-2D, S+D). They diverge once S+D > -2D, i.e. D > |S|/3:
//! the bogus sentinel-derived term wins and a real path is silently lost.
//!
//! Two boundaries this sweep confirms on the GPU:
//!   * VALUE correct          while  D <  |S|/3   (~3.33e8 for i32, ~2^60/3 for i64)
//!   * "<= S/2" zero-detection while  D <  |S|/4   (the finite result -2D stays
//!                                                  above the S/2 threshold)
//! So the documented safe range is the tighter one: |data| < |S|/4.
//!
//! Run on a CUDA machine:
//!     cargo run --release --example int_range_limits -p tropical-gemm-cuda

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use tropical_gemm_cuda::{SENTINEL_I32, SENTINEL_I64};

const KERNEL_SRC: &str = include_str!("../kernels/tropical_gemm.cu");

// Sourced from the crate constants (which mirror NEG_INF_I32 / NEG_INF_I64).
const S_I32: i64 = -(SENTINEL_I32 as i64); // NEG_INF_I32
const S_I64: i64 = -SENTINEL_I64; // NEG_INF_I64

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn cfg_i32() -> LaunchConfig {
    LaunchConfig { grid_dim: (1, 1, 1), block_dim: (16, 16, 1), shared_mem_bytes: 0 }
}
fn cfg_i64() -> LaunchConfig {
    LaunchConfig { grid_dim: (1, 1, 1), block_dim: (8, 8, 1), shared_mem_bytes: 0 }
}

/// Run the 1x2x1 masking GEMM for the given data magnitude D, return C[0,0].
fn mask_i32(stream: &Arc<CudaStream>, module: &Arc<CudaModule>, d: i32) -> R<i32> {
    let s = S_I32 as i32;
    // col-major: A(1x2)=[a0,a1], B(2x1)=[b0,b1]
    let a = stream.clone_htod(&[-d, s])?; // a0=-D (real), a1=S (no edge)
    let b = stream.clone_htod(&[-d, d])?; // b0=-D (real), b1=+D (data)
    let mut c = stream.alloc_zeros::<i32>(1)?;
    let f = module.load_function("tropical_maxplus_i32_nn").unwrap();
    let (m, n, k) = (1i32, 1i32, 2i32);
    let mut launch_args = stream.launch_builder(&f);
    launch_args.arg(&a).arg(&b).arg(&mut c).arg(&m).arg(&n).arg(&k);
    unsafe { launch_args.launch(cfg_i32())?; }
    Ok(stream.clone_dtoh(&c)?[0])
}

fn mask_i64(stream: &Arc<CudaStream>, module: &Arc<CudaModule>, d: i64) -> R<i64> {
    let a = stream.clone_htod(&[-d, S_I64])?;
    let b = stream.clone_htod(&[-d, d])?;
    let mut c = stream.alloc_zeros::<i64>(1)?;
    let f = module.load_function("tropical_maxplus_i64_nn").unwrap();
    let (m, n, k) = (1i32, 1i32, 2i32);
    let mut launch_args = stream.launch_builder(&f);
    launch_args.arg(&a).arg(&b).arg(&mut c).arg(&m).arg(&n).arg(&k);
    unsafe { launch_args.launch(cfg_i64())?; }
    Ok(stream.clone_dtoh(&c)?[0])
}

/// Classify and print one row. `gpu` is the measured C[0,0]; the true tropical
/// answer is -2D; the kernel is expected to compute max(-2D, S+D).
fn report(label: &str, s: i64, d: i64, gpu: i64) {
    let truth = -2 * d; // the real path weight
    let expected = truth.max(s + d); // what bare-add deterministically computes
    assert_eq!(gpu, expected, "{label} D={d}: kernel output {gpu} != max(-2D,S+D)={expected}");

    let value_ok = gpu == truth; // D < |S|/3
    let detect_ok = value_ok && truth > s / 2; // finite result stays above S/2 threshold  (D < |S|/4)
    let verdict = if !value_ok {
        "VALUE WRONG (real path masked -> switch type)"
    } else if !detect_ok {
        "value ok, but <=S/2 detect MISFIRES"
    } else {
        "OK"
    };
    // ratio D/|S| as a percentage, integer math
    let pct = (d as f64) / (s.unsigned_abs() as f64) * 100.0;
    println!(
        "  D={:>20}  ({:>5.1}% of |S|)  gpu C={:>20}  true={:>20}  {}",
        d, pct, gpu, truth, verdict
    );
}

fn main() -> R<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(KERNEL_SRC)?;
    let module = ctx.load_module(ptx)?;

    let s32 = S_I32.unsigned_abs() as i64; // 1e9
    println!("=== i32 maxplus, |S| = {s32} (markers: |S|/4 = {}, |S|/3 = {}) ===", s32 / 4, s32 / 3);
    for &d in &[100_000_000i64, 200_000_000, 250_000_000, 300_000_000, 333_333_333, 400_000_000, 500_000_000] {
        let gpu = mask_i32(&stream, &module, d as i32)?;
        report("i32", S_I32, d, gpu as i64);
    }

    let s64 = S_I64.unsigned_abs() as i64; // 2^60
    println!("\n=== i64 maxplus, |S| = 2^60 = {s64} (markers: |S|/4 = {}, |S|/3 = {}) ===", s64 / 4, s64 / 3);
    for &d in &[1i64 << 54, 1 << 56, 1 << 58, s64 / 3, 1 << 59] {
        let gpu = mask_i64(&stream, &module, d)?;
        report("i64", S_I64, d, gpu);
    }

    println!("\nConclusion: results are fully reliable while |data| < |S|/4");
    println!("  i32: |data| < {:>20}  (~2.5e8)", s32 / 4);
    println!("  i64: |data| < {:>20}  (~2.9e17)", s64 / 4);
    println!("Between |S|/4 and |S|/3 the value is still correct but the <=S/2 zero-test misfires;");
    println!("beyond |S|/3 a real path can be masked -> use i64 (or scale your data down).");
    Ok(())
}
