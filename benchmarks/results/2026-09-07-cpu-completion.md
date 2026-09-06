# CPU SIMD, packing reuse, and PyTorch comparison — 2026-09-07

This completes the measured CPU work in #38 and the remaining CPU coverage in
#44, and refreshes benchmarks for #36. Float argmax now has AVX2/NEON kernels for
MaxPlus, MinPlus, and MaxMul at both precisions. Bitwise u32/u64 has native CPU
SIMD, and AndOr is registered for the public CPU matrix/slice APIs. Repeated
calls can retain packing buffers in `GemmWorkspace`.

## Method

- Baseline: `e01d58f05eefcd78a280187fd1f35a698d2c0df7` (already includes CPU
  threading and shape-bounded packing from #66). Current: the implementation
  accompanying this report. Copy the exact `bench_cpu.rs` into the baseline;
  build the two checkouts in separate target directories.
- Release builds, default features, no custom `RUSTFLAGS`. Rust 1.98.0 on Apple
  M4/macOS; Rust 1.93.0 on Linux, dual Intel Xeon Platinum 8378A, 128 logical
  CPUs. The Xeon uses AVX2 kernels even though it supports AVX-512.
- CPU runs: square N=128/512, 1/8 Rayon threads, three warmups and median of
  seven samples. Pool construction and input generation are excluded. Each
  sample includes safe API result/packing allocation and result destruction.
  Baseline/current pairs run sequentially. All semirings/precisions, including
  unchanged value-only paths, are retained in the raw CSVs.
- Xeon processes are pinned to eight physical cores on socket 1:
  `32,44,46,51,53,54,57,59`. The host is shared: cores and SMT siblings are not
  reserved, and unrelated jobs remain active. Results are snapshots, without
  confidence intervals or claims of exclusive-machine peak throughput.
- Workspace runs use the same current binary, one thread, three warmups and
  21 samples, at N=32/128/512. Value output is preallocated for both variants;
  argmax allocates its returned values/indices for both. Only packing reuse
  differs. Capacity is retained across modes and can exceed a later kernel's
  needs; zero in the fresh column means no caller-owned storage is retained.
- PyTorch runs use Python 3.10, PyTorch 2.2.2+cu121, NumPy 1.26.4, and the
  release CUDA extension. Inputs are random positive f32, seed 42, with
  `torch.no_grad()`, three warmups and seven samples. Both thread pools use
  eight threads. Values are checked exactly against broadcast/reduce before
  timing all three semirings.
- CUDA runs use an NVIDIA A800 80GB PCIe (physical GPU 2), CUDA 12.1, with
  `CUDA_VISIBLE_DEVICES=2`. Other processes retain about 59 GB on this device;
  sampled utilization is zero outside our run. Inputs start on the GPU.
  Timings include the explicit `*_matmul_gpu` Python API, DLPack ownership,
  allocations, index conversion, and synchronization before/after each call.
  They exclude input upload, compilation, and result destruction. The CPU
  PyTorch comparison uses the ordinary `*_matmul` functions with CPU inputs.
- The PyTorch reference materializes an N³ candidate tensor and reduces along
  K using `max/min(dim=1)`, which also computes indices. It is an explicit
  baseline, not a claim about every possible PyTorch implementation. At
  N=1024 its f32 candidate tensor alone needs 4 GiB.

Raw data: [all CSV files](cpu-completion-2026-09-07/).

## CPU before/after

Selected **512×512, eight-thread** medians; speedup is baseline/current.

| Machine | Operation | Baseline ms | Current ms | Speedup |
|---|---|---:|---:|---:|
| M4 | MaxPlus f32 argmax | 6.745 | 2.683 | 2.51× |
| M4 | MinPlus f32 argmax | 6.924 | 2.550 | 2.72× |
| M4 | MaxMul f64 argmax | 7.751 | 4.552 | 1.70× |
| M4 | Bitwise u32 | 3.557 | 1.698 | 2.10× |
| M4 | Bitwise u64 | 3.609 | 3.636 | 0.99× |
| Xeon | MaxPlus f32 argmax | 45.779 | 5.430 | 8.43× |
| Xeon | MinPlus f32 argmax | 35.156 | 5.026 | 6.99× |
| Xeon | MaxMul f64 argmax | 33.041 | 8.112 | 4.07× |
| Xeon | Bitwise u32 | 12.570 | 1.628 | 7.72× |
| Xeon | Bitwise u64 | 12.090 | 3.018 | 4.01× |

M4 Bitwise u64 is effectively unchanged at this eight-thread size. Its
single-thread N=512 result improves from 16.282 to 12.797 ms (1.27×). SIMD does
not guarantee a win at every shape/thread count. Shared-host scheduling also
causes variation and occasional non-monotonic scaling; use the raw results
and rerun on the intended deployment hardware.

## Packing reuse

At N=128, one thread:

| Machine | Mode | Fresh ms | Reused ms | Retained bytes |
|---|---|---:|---:|---:|
| M4 | MaxPlus f32 values | 0.486750 | 0.426625 | 131072 |
| M4 | MaxPlus f32 argmax | 0.296083 | 0.268041 | 131072 |
| Xeon | MaxPlus f32 values | 0.288794 | 0.284412 | 131072 |
| Xeon | MaxPlus f32 argmax | 0.303207 | 0.301416 | 131072 |

Reuse removes packing allocation from warmed serial calls, verified with a
tracking allocator test. Its measured time benefit is modest here, especially
on Xeon; the old issue's predicted 2–3× gain is not supported by these runs.
Parallel calls still allocate task metadata, and argmax still allocates output.
`clear()` releases packing buffers. Packed matrix contents are recomputed on
every call, so callers can change inputs freely.

## End-to-end PyTorch comparison

MaxPlus f32, eight CPU threads; times are milliseconds:

| Device | N | tropical-gemm | PyTorch broadcast/reduce | Reference/native |
|---|---:|---:|---:|---:|
| Xeon CPU | 64 | 0.067538 | 0.096442 | 1.43× |
| Xeon CPU | 256 | 1.388982 | 13.776540 | 9.92× |
| Xeon CPU | 512 | 11.249947 | 102.555921 | 9.12× |
| A800 CUDA | 64 | 0.981556 | 0.047972 | 0.05× |
| A800 CUDA | 256 | 1.014561 | 0.150594 | 0.15× |
| A800 CUDA | 512 | 1.059812 | 0.866391 | 0.82× |
| A800 CUDA | 1024 | 1.479561 | 6.716553 | 4.54× |

The CUDA interface has substantial fixed overhead at small sizes in this
environment. At N=1024 it wins for all three semirings (4.2–4.5×); this does
not establish a universal crossover. These are synchronized Python interface
measurements, not device-resident kernel-only timings. CUDA kernels are
unchanged by this CPU work.

An initial draft of the harness incorrectly called the CPU convenience API
with CUDA tensors. Profiling showed host copies and CPU execution; those
measurements were discarded and the committed harness/results use the
explicit GPU APIs. Merely putting tensors on CUDA does not select the
single-matrix GPU convenience function.

## Reproduce

```bash
cargo run --release -p tropical-gemm --example bench_cpu -- 512 8 7
cargo run --release -p tropical-gemm --example bench_workspace -- 128 1 21
python benchmarks/bench_pytorch.py --device cpu --sizes 64 256 512 --threads 8
CUDA_VISIBLE_DEVICES=2 python benchmarks/bench_pytorch.py \
  --device cuda --sizes 64 256 512 1024 --threads 8
```

Build/install the Python extension in release mode with CUDA enabled first.
On the measured host, preload `/usr/local/cuda/lib64/libnvrtc.so` using the
harness's `--nvrtc` option and set `LD_LIBRARY_PATH=/usr/local/cuda/lib64` to
avoid an older NVRTC bundled with another Python environment. Use the CPU
affinity above with `taskset -c` when reproducing the Linux runs.

## Correctness and scope

Native kernels are compared against the explicit portable path for transposes,
strides, partial tiles, multiple K panels, ties, signed zeros, NaNs/infinities,
and sparse high/low bit lanes. Private argmax tests check indices above 2³¹.
Workspace tests cover changed shapes, semirings, pool sizes, zero dimensions,
clearing, and allocation-free warmed serial execution. Public AndOr is checked
against Boolean multiplication and a MaxPlus encoding.

Dedicated AVX-512 kernels, per-bit argmax, JAX (#27), and general einsum (#21)
remain outside this agreed scope. The earlier #38 performance estimates were
hypotheses, not measurements to multiply together; this report replaces them
with reproducible observations.
