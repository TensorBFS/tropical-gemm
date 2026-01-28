"""
Benchmark: tropical_gemm wrappers vs naive implementations.

Covers:
- PyTorch: CPU and GPU benchmarks
- JAX: CPU benchmarks with autodiff

Operations:
- 2D: tropical_maxplus_matmul, tropical_minplus_matmul, tropical_maxmul_matmul
- 3D batched: tropical_maxplus_matmul_batched, etc.
"""

import argparse
import time
import numpy as np
from typing import Tuple, Callable

import tropical_gemm

# PyTorch imports
try:
    import torch
    from tropical_gemm.pytorch import (
        tropical_maxplus_matmul as torch_maxplus,
        tropical_minplus_matmul as torch_minplus,
        tropical_maxmul_matmul as torch_maxmul,
        tropical_maxplus_matmul_batched as torch_maxplus_batched,
        tropical_minplus_matmul_batched as torch_minplus_batched,
        tropical_maxmul_matmul_batched as torch_maxmul_batched,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

CUDA_AVAILABLE = TORCH_AVAILABLE and tropical_gemm.cuda_available() and torch.cuda.is_available()

if CUDA_AVAILABLE:
    from tropical_gemm.pytorch import (
        tropical_maxplus_matmul_gpu,
        tropical_minplus_matmul_gpu,
        tropical_maxmul_matmul_gpu,
        tropical_maxplus_matmul_batched_gpu,
        tropical_minplus_matmul_batched_gpu,
        tropical_maxmul_matmul_batched_gpu,
    )

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import grad
    from tropical_gemm.jax import (
        tropical_maxplus_matmul as jax_maxplus,
        tropical_minplus_matmul as jax_minplus,
        tropical_maxmul_matmul as jax_maxmul,
        tropical_maxplus_matmul_batched as jax_maxplus_batched,
        tropical_minplus_matmul_batched as jax_minplus_batched,
        tropical_maxmul_matmul_batched as jax_maxmul_batched,
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ============================================================================
# Naive Implementations
# ============================================================================

# PyTorch naive
def torch_naive_maxplus_2d(a, b):
    return (a.unsqueeze(-1) + b.unsqueeze(0)).max(dim=1).values

def torch_naive_minplus_2d(a, b):
    return (a.unsqueeze(-1) + b.unsqueeze(0)).min(dim=1).values

def torch_naive_maxmul_2d(a, b):
    return (a.unsqueeze(-1) * b.unsqueeze(0)).max(dim=1).values

def torch_naive_maxplus_3d(a, b):
    return (a.unsqueeze(-1) + b.unsqueeze(-3)).max(dim=-2).values

def torch_naive_minplus_3d(a, b):
    return (a.unsqueeze(-1) + b.unsqueeze(-3)).min(dim=-2).values

def torch_naive_maxmul_3d(a, b):
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).max(dim=-2).values

# JAX naive
def jax_naive_maxplus_2d(a, b):
    return (a[:, :, None] + b[None, :, :]).max(axis=1)

def jax_naive_minplus_2d(a, b):
    return (a[:, :, None] + b[None, :, :]).min(axis=1)

def jax_naive_maxmul_2d(a, b):
    return (a[:, :, None] * b[None, :, :]).max(axis=1)

def jax_naive_maxplus_3d(a, b):
    return (a[:, :, :, None] + b[:, None, :, :]).max(axis=2)

def jax_naive_minplus_3d(a, b):
    return (a[:, :, :, None] + b[:, None, :, :]).min(axis=2)

def jax_naive_maxmul_3d(a, b):
    return (a[:, :, :, None] * b[:, None, :, :]).max(axis=2)


# ============================================================================
# Benchmark Utilities
# ============================================================================

def bench_torch_forward(fn, a, b, warmup=5, iters=20, sync=False):
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


def bench_torch_backward(fn, a, b, warmup=3, iters=10, sync=False):
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


def bench_jax_forward(fn, a, b, warmup=5, iters=20):
    # Block until computation completes
    for _ in range(warmup):
        _ = fn(a, b).block_until_ready()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn(a, b).block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def bench_jax_backward(fn, a, b, warmup=3, iters=10):
    def loss_fn(a, b):
        return fn(a, b).sum()

    grad_fn = grad(loss_fn, argnums=(0, 1))

    for _ in range(warmup):
        ga, gb = grad_fn(a, b)
        ga.block_until_ready()
        gb.block_until_ready()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ga, gb = grad_fn(a, b)
        ga.block_until_ready()
        gb.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def fmt_time(mean, std):
    return f"{mean:>6.2f} ± {std:>4.2f}"


# ============================================================================
# PyTorch CPU Benchmarks
# ============================================================================

def run_pytorch_cpu_benchmarks():
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping PyTorch benchmarks")
        return

    print("=" * 85)
    print("PyTorch CPU Benchmark: Optimized Rust vs Naive PyTorch")
    print("=" * 85)

    configs_2d = [(128, 128, 128), (256, 256, 256), (512, 512, 512)]
    configs_3d = [(4, 64, 64, 64), (8, 128, 128, 128), (16, 128, 128, 128)]

    ops_2d = [
        ("MaxPlus 2D", torch_maxplus, torch_naive_maxplus_2d, False),
        ("MinPlus 2D", torch_minplus, torch_naive_minplus_2d, False),
        ("MaxMul 2D", torch_maxmul, torch_naive_maxmul_2d, True),
    ]

    ops_3d = [
        ("MaxPlus Batched", torch_maxplus_batched, torch_naive_maxplus_3d, False),
        ("MinPlus Batched", torch_minplus_batched, torch_naive_minplus_3d, False),
        ("MaxMul Batched", torch_maxmul_batched, torch_naive_maxmul_3d, True),
    ]

    print("\n2D Forward Pass")
    print("-" * 85)
    print(f"{'Op':<16} {'Shape':<18} {'Optimized (ms)':<16} {'Naive (ms)':<16} {'Speedup':<10}")
    print("-" * 85)

    for name, opt_fn, naive_fn, pos in ops_2d:
        for m, k, n in configs_2d:
            a = torch.randn(m, k).abs() + 0.1 if pos else torch.randn(m, k)
            b = torch.randn(k, n).abs() + 0.1 if pos else torch.randn(k, n)

            opt_m, opt_s = bench_torch_forward(opt_fn, a, b)
            naive_m, naive_s = bench_torch_forward(naive_fn, a, b)

            print(f"{name:<16} ({m},{k},{n}){'':<9} {fmt_time(opt_m, opt_s):<16} {fmt_time(naive_m, naive_s):<16} {naive_m/opt_m:>5.2f}x")

    print("\n3D Batched Forward Pass")
    print("-" * 85)
    print(f"{'Op':<16} {'Shape':<18} {'Optimized (ms)':<16} {'Naive (ms)':<16} {'Speedup':<10}")
    print("-" * 85)

    for name, opt_fn, naive_fn, pos in ops_3d:
        for batch, m, k, n in configs_3d:
            a = torch.randn(batch, m, k).abs() + 0.1 if pos else torch.randn(batch, m, k)
            bb = torch.randn(batch, k, n).abs() + 0.1 if pos else torch.randn(batch, k, n)

            opt_m, opt_s = bench_torch_forward(opt_fn, a, bb)
            naive_m, naive_s = bench_torch_forward(naive_fn, a, bb)

            print(f"{name:<16} ({batch},{m},{k},{n}){'':<6} {fmt_time(opt_m, opt_s):<16} {fmt_time(naive_m, naive_s):<16} {naive_m/opt_m:>5.2f}x")

    print("\nForward + Backward Pass")
    print("-" * 85)
    print(f"{'Op':<16} {'Shape':<18} {'Optimized (ms)':<16} {'Naive (ms)':<16} {'Speedup':<10}")
    print("-" * 85)

    for m, k, n in [(128, 128, 128), (256, 256, 256)]:
        a, b = torch.randn(m, k), torch.randn(k, n)
        opt_m, _ = bench_torch_backward(torch_maxplus, a, b)
        naive_m, _ = bench_torch_backward(torch_naive_maxplus_2d, a, b)
        print(f"{'MaxPlus 2D':<16} ({m},{k},{n}){'':<9} {opt_m:>6.2f}            {naive_m:>6.2f}            {naive_m/opt_m:>5.2f}x")

    for batch, m, k, n in [(4, 64, 64, 64), (8, 128, 128, 128)]:
        a, b = torch.randn(batch, m, k), torch.randn(batch, k, n)
        opt_m, _ = bench_torch_backward(torch_maxplus_batched, a, b)
        naive_m, _ = bench_torch_backward(torch_naive_maxplus_3d, a, b)
        print(f"{'MaxPlus Batched':<16} ({batch},{m},{k},{n}){'':<6} {opt_m:>6.2f}            {naive_m:>6.2f}            {naive_m/opt_m:>5.2f}x")

    print()


# ============================================================================
# PyTorch GPU Benchmarks
# ============================================================================

def run_pytorch_gpu_benchmarks():
    if not CUDA_AVAILABLE:
        print("CUDA not available, skipping GPU benchmarks")
        return

    print("=" * 95)
    print(f"PyTorch GPU Benchmark: Optimized CUDA vs Naive | Device: {torch.cuda.get_device_name(0)}")
    print("=" * 95)

    device = torch.device("cuda")

    configs_2d = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    configs_3d = [(16, 256, 256, 256), (32, 256, 256, 256), (8, 512, 512, 512)]

    ops_2d = [
        ("MaxPlus 2D", tropical_maxplus_matmul_gpu, torch_maxplus, torch_naive_maxplus_2d, False),
        ("MinPlus 2D", tropical_minplus_matmul_gpu, torch_minplus, torch_naive_minplus_2d, False),
        ("MaxMul 2D", tropical_maxmul_matmul_gpu, torch_maxmul, torch_naive_maxmul_2d, True),
    ]

    ops_3d = [
        ("MaxPlus Batched", tropical_maxplus_matmul_batched_gpu, torch_maxplus_batched, torch_naive_maxplus_3d, False),
        ("MinPlus Batched", tropical_minplus_matmul_batched_gpu, torch_minplus_batched, torch_naive_minplus_3d, False),
        ("MaxMul Batched", tropical_maxmul_matmul_batched_gpu, torch_maxmul_batched, torch_naive_maxmul_3d, True),
    ]

    print("\n2D Forward Pass")
    print("-" * 95)
    print(f"{'Op':<14} {'Shape':<18} {'GPU Opt (ms)':<14} {'GPU Naive (ms)':<16} {'CPU Opt (ms)':<14} {'vs Naive':<10}")
    print("-" * 95)

    for name, gpu_fn, cpu_fn, naive_fn, pos in ops_2d:
        for m, k, n in configs_2d:
            a_gpu = (torch.randn(m, k, device=device).abs() + 0.1) if pos else torch.randn(m, k, device=device)
            b_gpu = (torch.randn(k, n, device=device).abs() + 0.1) if pos else torch.randn(k, n, device=device)
            a_cpu, b_cpu = a_gpu.cpu(), b_gpu.cpu()

            gpu_m, _ = bench_torch_forward(gpu_fn, a_gpu, b_gpu, sync=True)
            naive_m, _ = bench_torch_forward(naive_fn, a_gpu, b_gpu, sync=True)
            cpu_m, _ = bench_torch_forward(cpu_fn, a_cpu, b_cpu)

            print(f"{name:<14} ({m},{k},{n}){'':<7} {gpu_m:>8.2f}       {naive_m:>8.2f}         {cpu_m:>8.2f}       {naive_m/gpu_m:>5.2f}x")

    print("\n3D Batched Forward Pass")
    print("-" * 95)
    print(f"{'Op':<14} {'Shape':<18} {'GPU Opt (ms)':<14} {'GPU Naive (ms)':<16} {'CPU Opt (ms)':<14} {'vs Naive':<10}")
    print("-" * 95)

    for name, gpu_fn, cpu_fn, naive_fn, pos in ops_3d:
        for batch, m, k, n in configs_3d:
            a_gpu = (torch.randn(batch, m, k, device=device).abs() + 0.1) if pos else torch.randn(batch, m, k, device=device)
            b_gpu = (torch.randn(batch, k, n, device=device).abs() + 0.1) if pos else torch.randn(batch, k, n, device=device)
            a_cpu, b_cpu = a_gpu.cpu(), b_gpu.cpu()

            gpu_m, _ = bench_torch_forward(gpu_fn, a_gpu, b_gpu, sync=True)
            naive_m, _ = bench_torch_forward(naive_fn, a_gpu, b_gpu, sync=True)
            cpu_m, _ = bench_torch_forward(cpu_fn, a_cpu, b_cpu)

            print(f"{name:<14} ({batch},{m},{k},{n}){'':<4} {gpu_m:>8.2f}       {naive_m:>8.2f}         {cpu_m:>8.2f}       {naive_m/gpu_m:>5.2f}x")

    print("\nForward + Backward Pass")
    print("-" * 80)
    print(f"{'Op':<16} {'Shape':<18} {'GPU (ms)':<14} {'CPU (ms)':<14} {'Speedup':<10}")
    print("-" * 80)

    for m, k, n in [(512, 512, 512), (1024, 1024, 1024)]:
        a_gpu = torch.randn(m, k, device=device)
        b_gpu = torch.randn(k, n, device=device)
        a_cpu, b_cpu = a_gpu.cpu(), b_gpu.cpu()

        gpu_m, _ = bench_torch_backward(tropical_maxplus_matmul_gpu, a_gpu, b_gpu, sync=True)
        cpu_m, _ = bench_torch_backward(torch_maxplus, a_cpu, b_cpu)
        print(f"{'MaxPlus 2D':<16} ({m},{k},{n}){'':<7} {gpu_m:>8.2f}       {cpu_m:>8.2f}       {cpu_m/gpu_m:>5.2f}x")

    for batch, m, k, n in [(16, 256, 256, 256), (32, 256, 256, 256)]:
        a_gpu = torch.randn(batch, m, k, device=device)
        b_gpu = torch.randn(batch, k, n, device=device)
        a_cpu, b_cpu = a_gpu.cpu(), b_gpu.cpu()

        gpu_m, _ = bench_torch_backward(tropical_maxplus_matmul_batched_gpu, a_gpu, b_gpu, sync=True)
        cpu_m, _ = bench_torch_backward(torch_maxplus_batched, a_cpu, b_cpu)
        print(f"{'MaxPlus Batched':<16} ({batch},{m},{k},{n}){'':<4} {gpu_m:>8.2f}       {cpu_m:>8.2f}       {cpu_m/gpu_m:>5.2f}x")

    print()


# ============================================================================
# JAX Benchmarks
# ============================================================================

def run_jax_benchmarks():
    if not JAX_AVAILABLE:
        print("JAX not available, skipping JAX benchmarks")
        return

    print("=" * 85)
    print("JAX Benchmark: Optimized Rust vs Naive JAX")
    print("=" * 85)

    configs_2d = [(128, 128, 128), (256, 256, 256), (512, 512, 512)]
    configs_3d = [(4, 64, 64, 64), (8, 128, 128, 128), (16, 128, 128, 128)]

    ops_2d = [
        ("MaxPlus 2D", jax_maxplus, jax_naive_maxplus_2d, False),
        ("MinPlus 2D", jax_minplus, jax_naive_minplus_2d, False),
        ("MaxMul 2D", jax_maxmul, jax_naive_maxmul_2d, True),
    ]

    ops_3d = [
        ("MaxPlus Batched", jax_maxplus_batched, jax_naive_maxplus_3d, False),
        ("MinPlus Batched", jax_minplus_batched, jax_naive_minplus_3d, False),
        ("MaxMul Batched", jax_maxmul_batched, jax_naive_maxmul_3d, True),
    ]

    print("\n2D Forward Pass")
    print("-" * 85)
    print(f"{'Op':<16} {'Shape':<18} {'Optimized (ms)':<16} {'Naive (ms)':<16} {'Speedup':<10}")
    print("-" * 85)

    for name, opt_fn, naive_fn, pos in ops_2d:
        for m, k, n in configs_2d:
            key = jax.random.PRNGKey(42)
            a = jnp.abs(jax.random.normal(key, (m, k))) + 0.1 if pos else jax.random.normal(key, (m, k))
            b = jnp.abs(jax.random.normal(jax.random.PRNGKey(43), (k, n))) + 0.1 if pos else jax.random.normal(jax.random.PRNGKey(43), (k, n))

            opt_m, opt_s = bench_jax_forward(opt_fn, a, b)
            naive_m, naive_s = bench_jax_forward(naive_fn, a, b)

            print(f"{name:<16} ({m},{k},{n}){'':<9} {fmt_time(opt_m, opt_s):<16} {fmt_time(naive_m, naive_s):<16} {naive_m/opt_m:>5.2f}x")

    print("\n3D Batched Forward Pass")
    print("-" * 85)
    print(f"{'Op':<16} {'Shape':<18} {'Optimized (ms)':<16} {'Naive (ms)':<16} {'Speedup':<10}")
    print("-" * 85)

    for name, opt_fn, naive_fn, pos in ops_3d:
        for batch, m, k, n in configs_3d:
            key = jax.random.PRNGKey(42)
            a = jnp.abs(jax.random.normal(key, (batch, m, k))) + 0.1 if pos else jax.random.normal(key, (batch, m, k))
            b = jnp.abs(jax.random.normal(jax.random.PRNGKey(43), (batch, k, n))) + 0.1 if pos else jax.random.normal(jax.random.PRNGKey(43), (batch, k, n))

            opt_m, opt_s = bench_jax_forward(opt_fn, a, b)
            naive_m, naive_s = bench_jax_forward(naive_fn, a, b)

            print(f"{name:<16} ({batch},{m},{k},{n}){'':<6} {fmt_time(opt_m, opt_s):<16} {fmt_time(naive_m, naive_s):<16} {naive_m/opt_m:>5.2f}x")

    print("\nForward + Backward Pass")
    print("-" * 85)
    print(f"{'Op':<16} {'Shape':<18} {'Optimized (ms)':<16} {'Naive (ms)':<16} {'Speedup':<10}")
    print("-" * 85)

    for m, k, n in [(128, 128, 128), (256, 256, 256)]:
        key = jax.random.PRNGKey(42)
        a = jax.random.normal(key, (m, k))
        b = jax.random.normal(jax.random.PRNGKey(43), (k, n))

        opt_m, _ = bench_jax_backward(jax_maxplus, a, b)
        naive_m, _ = bench_jax_backward(jax_naive_maxplus_2d, a, b)
        print(f"{'MaxPlus 2D':<16} ({m},{k},{n}){'':<9} {opt_m:>6.2f}            {naive_m:>6.2f}            {naive_m/opt_m:>5.2f}x")

    for batch, m, k, n in [(4, 64, 64, 64), (8, 128, 128, 128)]:
        key = jax.random.PRNGKey(42)
        a = jax.random.normal(key, (batch, m, k))
        b = jax.random.normal(jax.random.PRNGKey(43), (batch, k, n))

        opt_m, _ = bench_jax_backward(jax_maxplus_batched, a, b)
        naive_m, _ = bench_jax_backward(jax_naive_maxplus_3d, a, b)
        print(f"{'MaxPlus Batched':<16} ({batch},{m},{k},{n}){'':<6} {opt_m:>6.2f}            {naive_m:>6.2f}            {naive_m/opt_m:>5.2f}x")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark tropical_gemm")
    parser.add_argument("--gpu", action="store_true", help="Run PyTorch GPU benchmarks")
    parser.add_argument("--cpu", action="store_true", help="Run PyTorch CPU benchmarks")
    parser.add_argument("--jax", action="store_true", help="Run JAX benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if args.all:
        args.cpu = args.gpu = args.jax = True

    if not (args.cpu or args.gpu or args.jax):
        args.cpu = True  # Default to PyTorch CPU

    if args.cpu:
        run_pytorch_cpu_benchmarks()
    if args.gpu:
        run_pytorch_gpu_benchmarks()
    if args.jax:
        run_jax_benchmarks()
