"""
Benchmark: tropical_gemm PyTorch wrapper vs naive PyTorch implementation.

Covers all operations:
- 2D: tropical_maxplus_matmul, tropical_minplus_matmul, tropical_maxmul_matmul
- 3D batched: tropical_maxplus_matmul_batched, etc.
"""

import argparse
import torch
import time
import numpy as np
from typing import Tuple, Callable

import tropical_gemm
from tropical_gemm.pytorch import (
    tropical_maxplus_matmul,
    tropical_minplus_matmul,
    tropical_maxmul_matmul,
    tropical_maxplus_matmul_batched,
    tropical_minplus_matmul_batched,
    tropical_maxmul_matmul_batched,
)

CUDA_AVAILABLE = tropical_gemm.cuda_available() and torch.cuda.is_available()

if CUDA_AVAILABLE:
    from tropical_gemm.pytorch import (
        tropical_maxplus_matmul_gpu,
        tropical_minplus_matmul_gpu,
        tropical_maxmul_matmul_gpu,
        tropical_maxplus_matmul_batched_gpu,
        tropical_minplus_matmul_batched_gpu,
        tropical_maxmul_matmul_batched_gpu,
    )


# ============================================================================
# Naive PyTorch Implementations
# ============================================================================

# 2D operations
def naive_maxplus_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.unsqueeze(-1) + b.unsqueeze(0)).max(dim=1).values

def naive_minplus_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.unsqueeze(-1) + b.unsqueeze(0)).min(dim=1).values

def naive_maxmul_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.unsqueeze(-1) * b.unsqueeze(0)).max(dim=1).values

# 3D batched operations
def naive_maxplus_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.unsqueeze(-1) + b.unsqueeze(-3)).max(dim=-2).values

def naive_minplus_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.unsqueeze(-1) + b.unsqueeze(-3)).min(dim=-2).values

def naive_maxmul_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).max(dim=-2).values


# ============================================================================
# Benchmark Utilities
# ============================================================================

def bench_forward(fn: Callable, a: torch.Tensor, b: torch.Tensor,
                  warmup: int = 5, iters: int = 20, sync: bool = False) -> Tuple[float, float]:
    for _ in range(warmup):
        _ = fn(a, b)
        if sync: torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if sync: torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn(a, b)
        if sync: torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def bench_backward(fn: Callable, a: torch.Tensor, b: torch.Tensor,
                   warmup: int = 3, iters: int = 10, sync: bool = False) -> Tuple[float, float]:
    for _ in range(warmup):
        ac, bc = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        fn(ac, bc).sum().backward()
        if sync: torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        ac, bc = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        if sync: torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(ac, bc).sum().backward()
        if sync: torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def verify(opt_fn, naive_fn, a, b, atol=1e-5) -> bool:
    return torch.allclose(opt_fn(a, b), naive_fn(a, b), atol=atol)


def fmt_time(mean, std):
    return f"{mean:>6.2f} ± {std:>4.2f}"


# ============================================================================
# CPU Benchmarks
# ============================================================================

def run_cpu_benchmarks():
    print("=" * 85)
    print("CPU Benchmark: Optimized Rust vs Naive PyTorch")
    print("=" * 85)

    # 2D configs: (M, K, N)
    configs_2d = [(128, 128, 128), (256, 256, 256), (512, 512, 512)]

    # 3D configs: (B, M, K, N)
    configs_3d = [(4, 64, 64, 64), (8, 128, 128, 128), (16, 128, 128, 128)]

    ops_2d = [
        ("MaxPlus 2D", tropical_maxplus_matmul, naive_maxplus_2d, False),
        ("MinPlus 2D", tropical_minplus_matmul, naive_minplus_2d, False),
        ("MaxMul 2D", tropical_maxmul_matmul, naive_maxmul_2d, True),
    ]

    ops_3d = [
        ("MaxPlus Batched", tropical_maxplus_matmul_batched, naive_maxplus_3d, False),
        ("MinPlus Batched", tropical_minplus_matmul_batched, naive_minplus_3d, False),
        ("MaxMul Batched", tropical_maxmul_matmul_batched, naive_maxmul_3d, True),
    ]

    # Forward 2D
    print("\n2D Forward Pass")
    print("-" * 85)
    print(f"{'Op':<16} {'Shape':<18} {'Optimized (ms)':<16} {'Naive (ms)':<16} {'Speedup':<10} {'OK':<4}")
    print("-" * 85)

    for name, opt_fn, naive_fn, pos in ops_2d:
        for m, k, n in configs_2d:
            a = torch.randn(m, k).abs() + 0.1 if pos else torch.randn(m, k)
            b = torch.randn(k, n).abs() + 0.1 if pos else torch.randn(k, n)

            opt_m, opt_s = bench_forward(opt_fn, a, b)
            naive_m, naive_s = bench_forward(naive_fn, a, b)
            ok = verify(opt_fn, naive_fn, a, b)

            print(f"{name:<16} ({m},{k},{n}){'':<9} {fmt_time(opt_m, opt_s):<16} {fmt_time(naive_m, naive_s):<16} {naive_m/opt_m:>5.2f}x     {'✓' if ok else '✗'}")

    # Forward 3D
    print("\n3D Batched Forward Pass")
    print("-" * 85)
    print(f"{'Op':<16} {'Shape':<18} {'Optimized (ms)':<16} {'Naive (ms)':<16} {'Speedup':<10} {'OK':<4}")
    print("-" * 85)

    for name, opt_fn, naive_fn, pos in ops_3d:
        for b, m, k, n in configs_3d:
            a = torch.randn(b, m, k).abs() + 0.1 if pos else torch.randn(b, m, k)
            bb = torch.randn(b, k, n).abs() + 0.1 if pos else torch.randn(b, k, n)

            opt_m, opt_s = bench_forward(opt_fn, a, bb)
            naive_m, naive_s = bench_forward(naive_fn, a, bb)
            ok = verify(opt_fn, naive_fn, a, bb)

            print(f"{name:<16} ({b},{m},{k},{n}){'':<6} {fmt_time(opt_m, opt_s):<16} {fmt_time(naive_m, naive_s):<16} {naive_m/opt_m:>5.2f}x     {'✓' if ok else '✗'}")

    # Backward
    print("\nForward + Backward Pass")
    print("-" * 85)
    print(f"{'Op':<16} {'Shape':<18} {'Optimized (ms)':<16} {'Naive (ms)':<16} {'Speedup':<10}")
    print("-" * 85)

    # 2D backward
    for m, k, n in [(128, 128, 128), (256, 256, 256)]:
        a, b = torch.randn(m, k), torch.randn(k, n)
        opt_m, _ = bench_backward(tropical_maxplus_matmul, a, b)
        naive_m, _ = bench_backward(naive_maxplus_2d, a, b)
        print(f"{'MaxPlus 2D':<16} ({m},{k},{n}){'':<9} {opt_m:>6.2f}            {naive_m:>6.2f}            {naive_m/opt_m:>5.2f}x")

    # 3D backward
    for batch, m, k, n in [(4, 64, 64, 64), (8, 128, 128, 128)]:
        a, b = torch.randn(batch, m, k), torch.randn(batch, k, n)
        opt_m, _ = bench_backward(tropical_maxplus_matmul_batched, a, b)
        naive_m, _ = bench_backward(naive_maxplus_3d, a, b)
        print(f"{'MaxPlus Batched':<16} ({batch},{m},{k},{n}){'':<6} {opt_m:>6.2f}            {naive_m:>6.2f}            {naive_m/opt_m:>5.2f}x")

    print()


# ============================================================================
# GPU Benchmarks
# ============================================================================

def run_gpu_benchmarks():
    if not CUDA_AVAILABLE:
        print("CUDA not available, skipping GPU benchmarks")
        return

    print("=" * 95)
    print(f"GPU Benchmark: Optimized CUDA vs Naive PyTorch | Device: {torch.cuda.get_device_name(0)}")
    print("=" * 95)

    device = torch.device("cuda")

    configs_2d = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    configs_3d = [(16, 256, 256, 256), (32, 256, 256, 256), (8, 512, 512, 512)]

    ops_2d = [
        ("MaxPlus 2D", tropical_maxplus_matmul_gpu, tropical_maxplus_matmul, naive_maxplus_2d, False),
        ("MinPlus 2D", tropical_minplus_matmul_gpu, tropical_minplus_matmul, naive_minplus_2d, False),
        ("MaxMul 2D", tropical_maxmul_matmul_gpu, tropical_maxmul_matmul, naive_maxmul_2d, True),
    ]

    ops_3d = [
        ("MaxPlus Batched", tropical_maxplus_matmul_batched_gpu, tropical_maxplus_matmul_batched, naive_maxplus_3d, False),
        ("MinPlus Batched", tropical_minplus_matmul_batched_gpu, tropical_minplus_matmul_batched, naive_minplus_3d, False),
        ("MaxMul Batched", tropical_maxmul_matmul_batched_gpu, tropical_maxmul_matmul_batched, naive_maxmul_3d, True),
    ]

    # 2D Forward
    print("\n2D Forward Pass")
    print("-" * 95)
    print(f"{'Op':<14} {'Shape':<18} {'GPU Opt (ms)':<14} {'GPU Naive (ms)':<16} {'CPU Opt (ms)':<14} {'vs Naive':<10}")
    print("-" * 95)

    for name, gpu_fn, cpu_fn, naive_fn, pos in ops_2d:
        for m, k, n in configs_2d:
            a_gpu = (torch.randn(m, k, device=device).abs() + 0.1) if pos else torch.randn(m, k, device=device)
            b_gpu = (torch.randn(k, n, device=device).abs() + 0.1) if pos else torch.randn(k, n, device=device)
            a_cpu, b_cpu = a_gpu.cpu(), b_gpu.cpu()

            gpu_m, _ = bench_forward(gpu_fn, a_gpu, b_gpu, sync=True)
            naive_m, _ = bench_forward(naive_fn, a_gpu, b_gpu, sync=True)
            cpu_m, _ = bench_forward(cpu_fn, a_cpu, b_cpu)

            print(f"{name:<14} ({m},{k},{n}){'':<7} {gpu_m:>8.2f}       {naive_m:>8.2f}         {cpu_m:>8.2f}       {naive_m/gpu_m:>5.2f}x")

    # 3D Forward
    print("\n3D Batched Forward Pass")
    print("-" * 95)
    print(f"{'Op':<14} {'Shape':<18} {'GPU Opt (ms)':<14} {'GPU Naive (ms)':<16} {'CPU Opt (ms)':<14} {'vs Naive':<10}")
    print("-" * 95)

    for name, gpu_fn, cpu_fn, naive_fn, pos in ops_3d:
        for batch, m, k, n in configs_3d:
            a_gpu = (torch.randn(batch, m, k, device=device).abs() + 0.1) if pos else torch.randn(batch, m, k, device=device)
            b_gpu = (torch.randn(batch, k, n, device=device).abs() + 0.1) if pos else torch.randn(batch, k, n, device=device)
            a_cpu, b_cpu = a_gpu.cpu(), b_gpu.cpu()

            gpu_m, _ = bench_forward(gpu_fn, a_gpu, b_gpu, sync=True)
            naive_m, _ = bench_forward(naive_fn, a_gpu, b_gpu, sync=True)
            cpu_m, _ = bench_forward(cpu_fn, a_cpu, b_cpu)

            print(f"{name:<14} ({batch},{m},{k},{n}){'':<4} {gpu_m:>8.2f}       {naive_m:>8.2f}         {cpu_m:>8.2f}       {naive_m/gpu_m:>5.2f}x")

    # Backward
    print("\nForward + Backward Pass")
    print("-" * 80)
    print(f"{'Op':<16} {'Shape':<18} {'GPU (ms)':<14} {'CPU (ms)':<14} {'Speedup':<10}")
    print("-" * 80)

    for m, k, n in [(512, 512, 512), (1024, 1024, 1024)]:
        a_gpu = torch.randn(m, k, device=device)
        b_gpu = torch.randn(k, n, device=device)
        a_cpu, b_cpu = a_gpu.cpu(), b_gpu.cpu()

        gpu_m, _ = bench_backward(tropical_maxplus_matmul_gpu, a_gpu, b_gpu, sync=True)
        cpu_m, _ = bench_backward(tropical_maxplus_matmul, a_cpu, b_cpu)
        print(f"{'MaxPlus 2D':<16} ({m},{k},{n}){'':<7} {gpu_m:>8.2f}       {cpu_m:>8.2f}       {cpu_m/gpu_m:>5.2f}x")

    for batch, m, k, n in [(16, 256, 256, 256), (32, 256, 256, 256)]:
        a_gpu = torch.randn(batch, m, k, device=device)
        b_gpu = torch.randn(batch, k, n, device=device)
        a_cpu, b_cpu = a_gpu.cpu(), b_gpu.cpu()

        gpu_m, _ = bench_backward(tropical_maxplus_matmul_batched_gpu, a_gpu, b_gpu, sync=True)
        cpu_m, _ = bench_backward(tropical_maxplus_matmul_batched, a_cpu, b_cpu)
        print(f"{'MaxPlus Batched':<16} ({batch},{m},{k},{n}){'':<4} {gpu_m:>8.2f}       {cpu_m:>8.2f}       {cpu_m/gpu_m:>5.2f}x")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark tropical_gemm")
    parser.add_argument("--gpu", action="store_true", help="Run GPU benchmarks")
    parser.add_argument("--cpu", action="store_true", help="Run CPU benchmarks")
    args = parser.parse_args()

    if not args.gpu and not args.cpu:
        args.cpu = True

    if args.cpu:
        run_cpu_benchmarks()
    if args.gpu:
        run_gpu_benchmarks()
