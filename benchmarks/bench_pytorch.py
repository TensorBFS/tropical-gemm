"""Compare tropical-gemm's PyTorch forward interface with broadcast/reduce.

python benchmarks/bench_pytorch.py --device cpu --sizes 64 256 512 --threads 8
python benchmarks/bench_pytorch.py --device cuda --sizes 64 256 512 1024

Times include Python/binding overhead; GPU calls synchronize before/after timing.
The reference materializes an M*K*N tensor. Both paths compute argmax internally.
"""

import argparse
import csv
import ctypes
import os
import statistics
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 512])
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--samples", type=int, default=7)
parser.add_argument(
    "--nvrtc", help="Optional absolute libnvrtc.so path to load before PyTorch"
)
args = parser.parse_args()
if args.samples < 1 or args.threads < 1 or any(n < 1 for n in args.sizes):
    parser.error("sizes, threads, and samples must be positive")
os.environ["RAYON_NUM_THREADS"] = str(args.threads)
if args.nvrtc:
    # Some environments bundle an older NVRTC with PyTorch. Explicitly bind both
    # names before import so cudarc resolves the requested toolkit consistently.
    nvrtc = ctypes.CDLL(args.nvrtc)
    nvrtc_alias = ctypes.CDLL("libnvrtc.so")

import torch
import tropical_gemm
from tropical_gemm import pytorch as tg

torch.set_num_threads(args.threads)
torch.manual_seed(42)
if args.device == "cuda" and not (
    torch.cuda.is_available() and tropical_gemm.cuda_available()
):
    parser.error("both PyTorch and tropical-gemm must have CUDA support")
print(
    f"torch={torch.__version__}, tropical-gemm={tropical_gemm.__version__}, threads={args.threads}",
    file=sys.stderr,
)
if args.device == "cuda":
    print(f"device={torch.cuda.get_device_name()}", file=sys.stderr)


def sync():
    if args.device == "cuda":
        torch.cuda.synchronize()


writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(["device", "n", "semiring", "backend", "threads", "median_ms"])
with torch.no_grad():
    for n in args.sizes:
        a = torch.rand((n, n), device=args.device, dtype=torch.float32)
        b = torch.rand((n, n), device=args.device, dtype=torch.float32)
        for op in ["maxplus", "minplus", "maxmul"]:
            # Single-matrix convenience functions use CPU unless the explicit
            # GPU entry point is selected, even when given CUDA tensors.
            suffix = "_gpu" if args.device == "cuda" else ""
            native = getattr(tg, f"tropical_{op}_matmul{suffix}")

            def reference():
                candidates = (
                    a[:, :, None] * b[None, :, :]
                    if op == "maxmul"
                    else a[:, :, None] + b[None, :, :]
                )
                return (
                    candidates.min(dim=1).values
                    if op == "minplus"
                    else candidates.max(dim=1).values
                )

            torch.testing.assert_close(native(a, b), reference(), rtol=0, atol=0)
            for backend, fn in [
                ("tropical-gemm", lambda: native(a, b)),
                ("torch-broadcast", reference),
            ]:
                for _ in range(3):
                    fn()
                sync()
                times = []
                for _ in range(args.samples):
                    sync()
                    start = time.perf_counter()
                    result = fn()
                    sync()
                    times.append((time.perf_counter() - start) * 1000)
                    del result
                writer.writerow(
                    [
                        args.device,
                        n,
                        op,
                        backend,
                        args.threads,
                        f"{statistics.median(times):.6f}",
                    ]
                )
                sys.stdout.flush()
