# Single CPU GEMM threading — 2026-09-07

Large single GEMMs now partition rows or columns across the active Rayon pool.
The reduction over K stays serial within each output cell. Packing allocations
also use each submatrix's actual panel dimensions instead of the maximum tiling
capacities. This addresses #58 and the excess allocation portion of #38.

## Method

- Baseline: main commit `3f7452644e44ac1a87114016263a94979505076f`.
- Changed code: the single-GEMM threading implementation accompanying this report.
- Both checkouts used the same `bench_threads` example and their own release build
  directories. No `target-cpu=native` or other custom compiler flags were set.
- Each row is the median of seven timed calls after three warmups, with inputs
  prepared in advance. Timings include the safe API's allocation, initialization,
  packing, result disposal, and `ThreadPool::install` overhead. Pool construction
  is excluded.
- Inputs are deterministic f32 matrices. `values` calls `tropical_matmul`,
  `argmax` calls `tropical_matmul_with_argmax`, and `batch4` calls
  `tropical_matmul_batched` on four matrices.
- A separate caller-owned Rayon pool is constructed for each thread count.
  The baseline receives the same pool settings even though its single GEMM
  remains serial.

Example reproduction:

```bash
cargo run --release -p tropical-gemm --example bench_threads -- 512 512 512 7 1,2,4,8
```

## Intel Xeon Platinum 8378A, Linux

Measured via `ssh gpu`, using the host CPUs, not its A800 GPUs. Rust 1.93.0;
AVX2 value dispatch and portable argmax kernel. Both executables were pinned
with `taskset -c 33,34,36,37,47,48,49,59` to eight physical cores on one socket.
For each size, the baseline ran immediately before the changed code.

The server was shared with other workloads. Pinning controls placement but does
not reserve those cores or eliminate contention on their SMT siblings. The raw
results show some nonmonotonic scaling; these observations are not a promise of
linear scaling or a controlled comparison against PyTorch.

| Size | Mode | Baseline, 8-thread pool (ms) | Changed, 1 thread (ms) | Changed, 8 threads (ms) | Baseline / changed at 8 threads |
|---|---|---:|---:|---:|---:|
| 64 | values | 0.3381 | 0.0436 | 0.0457 | 7.40× |
| 128 | values | 0.6191 | 0.3810 | 0.3174 | 1.95× |
| 256 | values | 2.5857 | 2.8755 | 0.6761 | 3.82× |
| 512 | values | 17.8613 | 17.7408 | 4.6520 | 3.84× |
| 512 | argmax | 132.2822 | 127.2942 | 32.4318 | 4.08× |
| 512 | batch4 | 33.2108 | 73.2155 | 16.7441 | 1.98× |
| 1024 | values | 143.0939 | 142.9180 | 35.6812 | 4.01× |
| 1024 | argmax | 921.8149 | 882.6758 | 245.7955 | 3.75× |
| 1024 | batch4 | 222.9293 | 499.2319 | 105.2391 | 2.12× |

64 and 128 remain below the parallelization threshold, so their improvements
come from bounded packing allocation, not splitting a single GEMM. At 64×64,
f32 AVX2 packing storage drops from 1 MiB to 32 KiB across the two panels.
The difference between thread counts for these serial calls reflects timing
variation and pool scheduling.

## Apple M4, macOS

Rust 1.98.0, NEON value dispatch and portable argmax kernel. Baseline and changed
executables ran sequentially without another task from this review competing
for CPU time; threads were not pinned. The eight-thread results use the M4's
mix of performance and efficiency cores as scheduled by macOS.

| Mode, 512×512 | Baseline, 8-thread pool (ms) | Changed, 1 thread (ms) | Changed, 8 threads (ms) | Baseline / changed at 8 threads |
|---|---:|---:|---:|---:|
| values | 14.4729 | 14.5457 | 3.2475 | 4.46× |
| argmax | 32.4333 | 32.2668 | 6.9696 | 4.65× |
| batch4 | 16.9817 | 58.2799 | 12.0102 | 1.41× |

## Raw results and correctness

[Raw CSV directory](cpu-threading-2026-09-07/) contains all 1/2/4/8-thread
measurements. `thread-*` files are the Xeon runs; `m4-*` files are the M4 runs.

The accompanying integration tests compare one- and four-thread outputs for
both split axes, transposed and padded inputs/outputs, K-panel boundaries,
integer sentinels, first-winner ties, floating-point values, column-major matrix
APIs, Bitwise, and nested batches. A custom observed microkernel confirms that
both single-GEMM paths actually execute on multiple workers.

Broader work remains in #38: dedicated AVX-512 kernels, SIMD coverage for argmax
and additional types, and reusable workspace. These measurements do not resolve
all of that issue or replace the GPU/cross-language benchmark project in #36.
