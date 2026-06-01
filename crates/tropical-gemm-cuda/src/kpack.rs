//! K-packed Boolean (AndOr) GPU GEMM.
//!
//! Packs the *contraction* dimension K of a single boolean matmul into 32-bit
//! words so the inner loop runs `acc |= (A_word & B_word)` over `Kw = ceil(K/32)`
//! words instead of one `bool` byte per K-element. This is distinct from the byte
//! `tropical_andor_bool_nn` kernel (kept as the default) and from `TropicalBitwise`
//! (which packs the problem axis). Column-major throughout; bit `b` of word `w`
//! holds K-element `32*w + b` (LSB-first); tail bits are zero (0 absorbs AND).
//!
//! See `docs/superpowers/specs/2026-06-01-andor-kpack-design.md`.

use crate::context::CudaContext;
use crate::error::{CudaError, Result};
use crate::memory::GpuMatrix;
use cudarc::driver::{LaunchConfig, PushKernelArg};

/// Square thread-block edge for the pack and direct kernels (256 threads/block).
const TILE: usize = 16;

#[inline]
fn ceil_div(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

/// Convert a dimension to the `int` extent the CUDA kernels take, rejecting values
/// that don't fit `i32` rather than silently truncating to a negative/garbage extent.
/// Once every dimension is `<= i32::MAX`, the `m*k` products in `validate_gemm_input`
/// also cannot overflow `usize` on a 64-bit target.
fn dim_i32(label: &str, v: usize) -> Result<i32> {
    i32::try_from(v).map_err(|_| {
        CudaError::DimensionMismatch(format!("{label} ({v}) exceeds i32::MAX for CUDA kernels"))
    })
}

/// Bit-packed left operand: logical M×Kw column-major `u32`, `Kw = ceil(K/32)`.
/// Opaque so a row-packed operand can't be passed where a column-packed one is
/// expected. Reusable across many GEMMs to amortize packing cost.
pub struct AndOrPackedRows {
    words: GpuMatrix<u32>, // M × Kw, column-major
    m: usize,
    k: usize,
}

/// Bit-packed right operand: logical Kw×N column-major `u32`, `Kw = ceil(K/32)`.
pub struct AndOrPackedCols {
    words: GpuMatrix<u32>, // Kw × N, column-major
    k: usize,
    n: usize,
}

impl AndOrPackedRows {
    /// Logical rows (M) of the original `bool` operand.
    pub fn rows(&self) -> usize {
        self.m
    }
    /// Logical contraction dimension (K) of the original `bool` operand.
    pub fn k(&self) -> usize {
        self.k
    }
}

impl AndOrPackedCols {
    /// Logical contraction dimension (K) of the original `bool` operand.
    pub fn k(&self) -> usize {
        self.k
    }
    /// Logical columns (N) of the original `bool` operand.
    pub fn cols(&self) -> usize {
        self.n
    }
}

/// Launch `pack_rows_u32` into a caller-provided `words` buffer (M×Kw u32). Split
/// out from `pack_andor_rows_gpu` so a test can pack into a pre-dirtied buffer and
/// confirm the tail bits are cleared (not merely left zero by `alloc`).
fn launch_pack_rows(
    ctx: &CudaContext,
    a: &GpuMatrix<bool>,
    words: &mut GpuMatrix<u32>,
    m: usize,
    k: usize,
) -> Result<()> {
    let kw = k.div_ceil(32);
    let kernel = ctx.get_kernel("pack_rows_u32")?;
    let cfg = LaunchConfig {
        grid_dim: (ceil_div(m, TILE), ceil_div(kw, TILE), 1),
        block_dim: (TILE as u32, TILE as u32, 1),
        shared_mem_bytes: 0,
    };
    let m_i32 = dim_i32("M", m)?;
    let k_i32 = dim_i32("K", k)?;
    let stream = ctx.stream();
    let mut builder = stream.launch_builder(&kernel);
    builder
        .arg(a.as_slice())
        .arg(words.as_slice_mut())
        .arg(&m_i32)
        .arg(&k_i32);
    unsafe {
        builder.launch(cfg)?;
    }
    stream.synchronize()?;
    Ok(())
}

/// Launch `pack_cols_u32` into a caller-provided `words` buffer (Kw×N u32).
fn launch_pack_cols(
    ctx: &CudaContext,
    b: &GpuMatrix<bool>,
    words: &mut GpuMatrix<u32>,
    n: usize,
    k: usize,
) -> Result<()> {
    let kw = k.div_ceil(32);
    let kernel = ctx.get_kernel("pack_cols_u32")?;
    let cfg = LaunchConfig {
        grid_dim: (ceil_div(n, TILE), ceil_div(kw, TILE), 1),
        block_dim: (TILE as u32, TILE as u32, 1),
        shared_mem_bytes: 0,
    };
    let n_i32 = dim_i32("N", n)?;
    let k_i32 = dim_i32("K", k)?;
    let stream = ctx.stream();
    let mut builder = stream.launch_builder(&kernel);
    builder
        .arg(b.as_slice())
        .arg(words.as_slice_mut())
        .arg(&n_i32)
        .arg(&k_i32);
    unsafe {
        builder.launch(cfg)?;
    }
    stream.synchronize()?;
    Ok(())
}

/// Pack a column-major `bool` matrix A (M×K) into `AndOrPackedRows` (M×Kw u32) on GPU.
pub fn pack_andor_rows_gpu(ctx: &CudaContext, a: &GpuMatrix<bool>) -> Result<AndOrPackedRows> {
    let m = a.rows();
    let k = a.cols();
    if m == 0 || k == 0 {
        return Err(CudaError::DimensionMismatch(format!(
            "pack_andor_rows_gpu: zero dimension (M={m}, K={k})"
        )));
    }
    let kw = k.div_ceil(32);
    let mut words = GpuMatrix::<u32>::alloc(ctx, m, kw)?;
    launch_pack_rows(ctx, a, &mut words, m, k)?;
    Ok(AndOrPackedRows { words, m, k })
}

/// Pack a column-major `bool` matrix B (K×N) into `AndOrPackedCols` (Kw×N u32) on GPU.
pub fn pack_andor_cols_gpu(ctx: &CudaContext, b: &GpuMatrix<bool>) -> Result<AndOrPackedCols> {
    let k = b.rows();
    let n = b.cols();
    if k == 0 || n == 0 {
        return Err(CudaError::DimensionMismatch(format!(
            "pack_andor_cols_gpu: zero dimension (K={k}, N={n})"
        )));
    }
    let kw = k.div_ceil(32);
    let mut words = GpuMatrix::<u32>::alloc(ctx, kw, n)?;
    launch_pack_cols(ctx, b, &mut words, n, k)?;
    Ok(AndOrPackedCols { words, k, n })
}

/// GPU-resident K-packed AndOr GEMM: `C[i,j] = any_w(A_word(i,w) & B_word(j,w))`.
/// Reuse the packed operands across many calls to amortize packing.
pub fn tropical_gemm_gpu_andor_packed(
    ctx: &CudaContext,
    a: &AndOrPackedRows,
    b: &AndOrPackedCols,
    c: &mut GpuMatrix<bool>,
) -> Result<()> {
    if a.k != b.k {
        return Err(CudaError::DimensionMismatch(format!(
            "A.k ({}) != B.k ({})",
            a.k, b.k
        )));
    }
    if c.rows() != a.m || c.cols() != b.n {
        return Err(CudaError::DimensionMismatch(format!(
            "C dimensions ({}, {}) don't match A×B ({}, {})",
            c.rows(),
            c.cols(),
            a.m,
            b.n
        )));
    }
    let m = a.m;
    let n = b.n;
    let kw = a.k.div_ceil(32);

    let kernel = ctx.get_kernel("tropical_andor_kpack_direct_u32")?;
    let cfg = LaunchConfig {
        grid_dim: (ceil_div(m, TILE), ceil_div(n, TILE), 1),
        block_dim: (TILE as u32, TILE as u32, 1),
        shared_mem_bytes: 0,
    };
    let m_i32 = dim_i32("M", m)?;
    let n_i32 = dim_i32("N", n)?;
    let kw_i32 = dim_i32("Kw", kw)?;
    let stream = ctx.stream();
    let mut builder = stream.launch_builder(&kernel);
    builder
        .arg(a.words.as_slice())
        .arg(b.words.as_slice())
        .arg(c.as_slice_mut())
        .arg(&m_i32)
        .arg(&n_i32)
        .arg(&kw_i32);
    unsafe {
        builder.launch(cfg)?;
    }
    stream.synchronize()?;
    Ok(())
}

/// One-shot K-packed AndOr GEMM: upload → pack → gemm → download. bool in / bool out.
///
/// `a` is column-major M×K, `b` is column-major K×N, result is column-major M×N.
/// Unlike `validate_gemm_input` (slice-length only), this **rejects zero dimensions**:
/// `Kw = 0` / empty operands are degenerate for packing.
pub fn tropical_matmul_gpu_andor_packed(
    a: &[bool],
    m: usize,
    k: usize,
    b: &[bool],
    n: usize,
) -> Result<Vec<bool>> {
    // Reject zero and over-`i32` dims BEFORE `validate_gemm_input` (which multiplies
    // `m*k` / `k*n` unchecked): once each dim fits `i32`, those products fit `usize`.
    if m == 0 || k == 0 || n == 0 {
        return Err(CudaError::DimensionMismatch(format!(
            "K-packed AndOr GEMM requires non-zero dimensions (M={m}, K={k}, N={n})"
        )));
    }
    dim_i32("M", m)?;
    dim_i32("K", k)?;
    dim_i32("N", n)?;
    crate::validate_gemm_input(a, b, m, k, n)?;

    let ctx = crate::get_global_context()?;
    let a_gpu = GpuMatrix::from_host(ctx, a, m, k)?;
    let b_gpu = GpuMatrix::from_host(ctx, b, k, n)?;

    let packed_a = pack_andor_rows_gpu(ctx, &a_gpu)?;
    let packed_b = pack_andor_cols_gpu(ctx, &b_gpu)?;

    let mut c_gpu = GpuMatrix::<bool>::alloc(ctx, m, n)?;
    tropical_gemm_gpu_andor_packed(ctx, &packed_a, &packed_b, &mut c_gpu)?;
    c_gpu.to_host(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaContext;

    /// Skip helper (kpack.rs can't see lib.rs's `cuda_context_or_skip`).
    fn ctx_or_skip() -> Option<&'static CudaContext> {
        match std::panic::catch_unwind(crate::get_global_context) {
            Ok(Ok(ctx)) => Some(ctx),
            _ => {
                println!("CUDA not available, skipping test");
                None
            }
        }
    }

    /// Reference row packer (oracle). A is M×K column-major bool; returns Apacked
    /// M×Kw column-major u32 with `Apacked[w*M + i]`, bit `b` = `A[(32*w+b)*M + i]`.
    fn reference_pack_rows(a: &[bool], m: usize, k: usize) -> Vec<u32> {
        let kw = k.div_ceil(32);
        let mut out = vec![0u32; m * kw];
        for i in 0..m {
            for w in 0..kw {
                let mut word = 0u32;
                for b in 0..32 {
                    let kk = 32 * w + b;
                    if kk < k && a[kk * m + i] {
                        word |= 1u32 << b;
                    }
                }
                out[w * m + i] = word;
            }
        }
        out
    }

    /// Reference column packer (oracle). B is K×N column-major bool; returns Bpacked
    /// Kw×N column-major u32 with `Bpacked[j*Kw + w]`, bit `b` = `B[j*K + (32*w+b)]`.
    fn reference_pack_cols(b: &[bool], n: usize, k: usize) -> Vec<u32> {
        let kw = k.div_ceil(32);
        let mut out = vec![0u32; kw * n];
        for j in 0..n {
            for w in 0..kw {
                let mut word = 0u32;
                for bit in 0..32 {
                    let kk = 32 * w + bit;
                    if kk < k && b[j * k + kk] {
                        word |= 1u32 << bit;
                    }
                }
                out[j * kw + w] = word;
            }
        }
        out
    }

    #[test]
    fn reference_pack_rows_lsb_first_and_tail_zero() {
        // M=1, K=33: one column of 33 bools. Set k=0 (bit 0 of word 0),
        // k=31 (bit 31 of word 0), k=32 (bit 0 of word 1). Kw=2.
        let mut a = vec![false; 33];
        a[0] = true;
        a[31] = true;
        a[32] = true;
        let packed = reference_pack_rows(&a, 1, 33);
        assert_eq!(packed.len(), 2, "Kw=ceil(33/32)=2 words for M=1");
        assert_eq!(packed[0], (1u32 << 0) | (1u32 << 31), "word 0: bits 0 and 31");
        // Word 1 holds only k=32 in bit 0; bits 1..32 are tail (k>=33) and must be 0.
        assert_eq!(packed[1], 1u32 << 0, "word 1: bit 0 set, tail bits zero");
    }

    #[test]
    fn reference_pack_cols_lsb_first_and_tail_zero() {
        // N=1, K=33 mirror of the rows test. B column is contiguous (stride 1).
        let mut b = vec![false; 33];
        b[0] = true;
        b[31] = true;
        b[32] = true;
        let packed = reference_pack_cols(&b, 1, 33);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], (1u32 << 0) | (1u32 << 31));
        assert_eq!(packed[1], 1u32 << 0);
    }

    #[test]
    fn gpu_pack_rows_matches_reference() {
        let Some(ctx) = ctx_or_skip() else { return };
        // Spec K-set; M chosen non-tile-aligned to exercise the bounds guard.
        for &k in &[1usize, 31, 32, 33, 63, 64, 65] {
            let m = 5;
            let a: Vec<bool> = (0..m * k).map(|i| (i * 7 + 1) % 3 == 0).collect();
            let a_gpu = GpuMatrix::from_host(ctx, &a, m, k).unwrap();
            let packed = pack_andor_rows_gpu(ctx, &a_gpu).unwrap();
            let got = packed.words.to_host(ctx).unwrap();
            let expected = reference_pack_rows(&a, m, k);
            assert_eq!(got, expected, "GPU pack_rows != reference at K={k}");
        }
    }

    #[test]
    fn gpu_pack_cols_matches_reference() {
        let Some(ctx) = ctx_or_skip() else { return };
        for &k in &[1usize, 31, 32, 33, 63, 64, 65] {
            let n = 6;
            let b: Vec<bool> = (0..k * n).map(|i| (i * 5 + 2) % 4 == 0).collect();
            let b_gpu = GpuMatrix::from_host(ctx, &b, k, n).unwrap();
            let packed = pack_andor_cols_gpu(ctx, &b_gpu).unwrap();
            let got = packed.words.to_host(ctx).unwrap();
            let expected = reference_pack_cols(&b, n, k);
            assert_eq!(got, expected, "GPU pack_cols != reference at K={k}");
        }
    }

    #[test]
    fn gpu_pack_rows_clears_tail_in_dirty_buffer() {
        let Some(ctx) = ctx_or_skip() else { return };
        // K=33 -> Kw=2; word 1 has only bit 0 valid, bits 1..32 are tail. Pack into a
        // buffer pre-filled with all-ones: a packer that wrote only set bits (|=) would
        // leave the tail dirty. Ours writes each word in full, so the tail must be 0.
        let (m, k) = (3usize, 33usize);
        let kw = k.div_ceil(32);
        let a: Vec<bool> = (0..m * k).map(|i| i % 2 == 0).collect();
        let a_gpu = GpuMatrix::from_host(ctx, &a, m, k).unwrap();
        // Pre-dirty the words buffer with 0xFFFF_FFFF.
        let dirty = vec![u32::MAX; m * kw];
        let mut words = GpuMatrix::from_host(ctx, &dirty, m, kw).unwrap();
        launch_pack_rows(ctx, &a_gpu, &mut words, m, k).unwrap();
        let got = words.to_host(ctx).unwrap();
        let expected = reference_pack_rows(&a, m, k);
        assert_eq!(got, expected, "row pack: tail not cleared in dirty buffer");
    }

    #[test]
    fn gpu_pack_cols_clears_tail_in_dirty_buffer() {
        let Some(ctx) = ctx_or_skip() else { return };
        // pack_cols_u32 is independent code from pack_rows_u32, so it needs its own
        // dirty-buffer check. K=33 -> Kw=2; word 1's bits 1..32 are tail.
        let (n, k) = (4usize, 33usize);
        let kw = k.div_ceil(32);
        let b: Vec<bool> = (0..k * n).map(|i| i % 3 == 0).collect();
        let b_gpu = GpuMatrix::from_host(ctx, &b, k, n).unwrap();
        let dirty = vec![u32::MAX; kw * n];
        let mut words = GpuMatrix::from_host(ctx, &dirty, kw, n).unwrap();
        launch_pack_cols(ctx, &b_gpu, &mut words, n, k).unwrap();
        let got = words.to_host(ctx).unwrap();
        let expected = reference_pack_cols(&b, n, k);
        assert_eq!(got, expected, "col pack: tail not cleared in dirty buffer");
    }
}
