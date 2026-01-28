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


# ============================================================================
# Cross-Validation
# ============================================================================

def run_cross_validation():
    """Cross-validate results between PyTorch, JAX, and naive implementations."""
    print("=" * 90)
    print("Cross-Validation: Comparing PyTorch, JAX, and Naive Implementations")
    print("=" * 90)
    print()

    if not TORCH_AVAILABLE and not JAX_AVAILABLE:
        print("Neither PyTorch nor JAX available. Skipping validation.")
        return True

    # Test configurations
    configs_2d = [(32, 64, 48), (64, 128, 96), (128, 256, 192)]
    configs_3d = [(4, 32, 64, 48), (8, 64, 128, 96)]

    all_passed = True

    def check(name, a, b, rtol=1e-5, atol=1e-5):
        nonlocal all_passed
        if np.allclose(a, b, rtol=rtol, atol=atol):
            return "✓"
        else:
            all_passed = False
            max_diff = np.max(np.abs(a - b))
            return f"✗ (max_diff={max_diff:.2e})"

    # 2D Operations
    print("2D Operations")
    print("-" * 90)
    print(f"{'Op':<14} {'Shape':<16} {'Torch vs Naive':<16} {'JAX vs Naive':<16} {'Torch vs JAX':<16}")
    print("-" * 90)

    ops_2d = [
        ("MaxPlus", torch_maxplus if TORCH_AVAILABLE else None, jax_maxplus if JAX_AVAILABLE else None,
         torch_naive_maxplus_2d, jax_naive_maxplus_2d, False),
        ("MinPlus", torch_minplus if TORCH_AVAILABLE else None, jax_minplus if JAX_AVAILABLE else None,
         torch_naive_minplus_2d, jax_naive_minplus_2d, False),
        ("MaxMul", torch_maxmul if TORCH_AVAILABLE else None, jax_maxmul if JAX_AVAILABLE else None,
         torch_naive_maxmul_2d, jax_naive_maxmul_2d, True),
    ]

    for name, torch_fn, jax_fn, torch_naive, jax_naive, pos in ops_2d:
        for m, k, n in configs_2d:
            # Create test data
            np.random.seed(42)
            a_np = np.random.randn(m, k).astype(np.float32)
            b_np = np.random.randn(k, n).astype(np.float32)
            if pos:
                a_np, b_np = np.abs(a_np) + 0.1, np.abs(b_np) + 0.1

            # PyTorch
            if TORCH_AVAILABLE and torch_fn is not None:
                a_torch = torch.from_numpy(a_np)
                b_torch = torch.from_numpy(b_np)
                torch_opt = torch_fn(a_torch, b_torch).numpy()
                torch_ref = torch_naive(a_torch, b_torch).numpy()
                torch_vs_naive = check(f"{name} torch", torch_opt, torch_ref)
            else:
                torch_opt = None
                torch_vs_naive = "N/A"

            # JAX
            if JAX_AVAILABLE and jax_fn is not None:
                a_jax = jnp.array(a_np)
                b_jax = jnp.array(b_np)
                jax_opt = np.array(jax_fn(a_jax, b_jax))
                jax_ref = np.array(jax_naive(a_jax, b_jax))
                jax_vs_naive = check(f"{name} jax", jax_opt, jax_ref)
            else:
                jax_opt = None
                jax_vs_naive = "N/A"

            # Cross-framework
            if torch_opt is not None and jax_opt is not None:
                torch_vs_jax = check(f"{name} cross", torch_opt, jax_opt)
            else:
                torch_vs_jax = "N/A"

            cfg = f"({m},{k},{n})"
            print(f"{name:<14} {cfg:<16} {torch_vs_naive:<16} {jax_vs_naive:<16} {torch_vs_jax:<16}")

    # 3D Batched Operations
    print()
    print("3D Batched Operations")
    print("-" * 90)
    print(f"{'Op':<14} {'Shape':<16} {'Torch vs Naive':<16} {'JAX vs Naive':<16} {'Torch vs JAX':<16}")
    print("-" * 90)

    ops_3d = [
        ("MaxPlus", torch_maxplus_batched if TORCH_AVAILABLE else None, jax_maxplus_batched if JAX_AVAILABLE else None,
         torch_naive_maxplus_3d, jax_naive_maxplus_3d, False),
        ("MinPlus", torch_minplus_batched if TORCH_AVAILABLE else None, jax_minplus_batched if JAX_AVAILABLE else None,
         torch_naive_minplus_3d, jax_naive_minplus_3d, False),
        ("MaxMul", torch_maxmul_batched if TORCH_AVAILABLE else None, jax_maxmul_batched if JAX_AVAILABLE else None,
         torch_naive_maxmul_3d, jax_naive_maxmul_3d, True),
    ]

    for name, torch_fn, jax_fn, torch_naive, jax_naive, pos in ops_3d:
        for batch, m, k, n in configs_3d:
            np.random.seed(42)
            a_np = np.random.randn(batch, m, k).astype(np.float32)
            b_np = np.random.randn(batch, k, n).astype(np.float32)
            if pos:
                a_np, b_np = np.abs(a_np) + 0.1, np.abs(b_np) + 0.1

            if TORCH_AVAILABLE and torch_fn is not None:
                a_torch = torch.from_numpy(a_np)
                b_torch = torch.from_numpy(b_np)
                torch_opt = torch_fn(a_torch, b_torch).numpy()
                torch_ref = torch_naive(a_torch, b_torch).numpy()
                torch_vs_naive = check(f"{name} torch", torch_opt, torch_ref)
            else:
                torch_opt = None
                torch_vs_naive = "N/A"

            if JAX_AVAILABLE and jax_fn is not None:
                a_jax = jnp.array(a_np)
                b_jax = jnp.array(b_np)
                jax_opt = np.array(jax_fn(a_jax, b_jax))
                jax_ref = np.array(jax_naive(a_jax, b_jax))
                jax_vs_naive = check(f"{name} jax", jax_opt, jax_ref)
            else:
                jax_opt = None
                jax_vs_naive = "N/A"

            if torch_opt is not None and jax_opt is not None:
                torch_vs_jax = check(f"{name} cross", torch_opt, jax_opt)
            else:
                torch_vs_jax = "N/A"

            cfg = f"({batch},{m},{k},{n})"
            print(f"{name:<14} {cfg:<16} {torch_vs_naive:<16} {jax_vs_naive:<16} {torch_vs_jax:<16}")

    # Gradient Cross-Validation
    print()
    print("Gradient Cross-Validation (MaxPlus)")
    print("-" * 90)
    print(f"{'Shape':<20} {'Torch grad_a vs JAX':<24} {'Torch grad_b vs JAX':<24}")
    print("-" * 90)

    if TORCH_AVAILABLE and JAX_AVAILABLE:
        for m, k, n in [(32, 64, 48), (64, 128, 96)]:
            np.random.seed(42)
            a_np = np.random.randn(m, k).astype(np.float32)
            b_np = np.random.randn(k, n).astype(np.float32)

            # PyTorch gradients
            a_torch = torch.from_numpy(a_np).requires_grad_(True)
            b_torch = torch.from_numpy(b_np).requires_grad_(True)
            c_torch = torch_maxplus(a_torch, b_torch)
            c_torch.sum().backward()
            torch_grad_a = a_torch.grad.numpy()
            torch_grad_b = b_torch.grad.numpy()

            # JAX gradients
            from jax import grad as jax_grad
            def jax_loss(a, b):
                return jax_maxplus(a, b).sum()
            jax_grad_fn = jax_grad(jax_loss, argnums=(0, 1))
            a_jax, b_jax = jnp.array(a_np), jnp.array(b_np)
            jax_grad_a, jax_grad_b = jax_grad_fn(a_jax, b_jax)
            jax_grad_a, jax_grad_b = np.array(jax_grad_a), np.array(jax_grad_b)

            grad_a_check = check("grad_a", torch_grad_a, jax_grad_a)
            grad_b_check = check("grad_b", torch_grad_b, jax_grad_b)

            cfg = f"({m},{k},{n})"
            print(f"{cfg:<20} {grad_a_check:<24} {grad_b_check:<24}")

        # Batched gradient
        for batch, m, k, n in [(4, 32, 64, 48)]:
            np.random.seed(42)
            a_np = np.random.randn(batch, m, k).astype(np.float32)
            b_np = np.random.randn(batch, k, n).astype(np.float32)

            a_torch = torch.from_numpy(a_np).requires_grad_(True)
            b_torch = torch.from_numpy(b_np).requires_grad_(True)
            c_torch = torch_maxplus_batched(a_torch, b_torch)
            c_torch.sum().backward()
            torch_grad_a = a_torch.grad.numpy()
            torch_grad_b = b_torch.grad.numpy()

            def jax_loss_batched(a, b):
                return jax_maxplus_batched(a, b).sum()
            jax_grad_fn = jax_grad(jax_loss_batched, argnums=(0, 1))
            a_jax, b_jax = jnp.array(a_np), jnp.array(b_np)
            jax_grad_a, jax_grad_b = jax_grad_fn(a_jax, b_jax)
            jax_grad_a, jax_grad_b = np.array(jax_grad_a), np.array(jax_grad_b)

            grad_a_check = check("grad_a batched", torch_grad_a, jax_grad_a)
            grad_b_check = check("grad_b batched", torch_grad_b, jax_grad_b)

            cfg = f"({batch},{m},{k},{n})"
            print(f"{cfg:<20} {grad_a_check:<24} {grad_b_check:<24}")
    else:
        print("Requires both PyTorch and JAX to be available")

    print()
    if all_passed:
        print("All cross-validation checks PASSED ✓")
    else:
        print("Some cross-validation checks FAILED ✗")
    print()

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark tropical_gemm")
    parser.add_argument("--gpu", action="store_true", help="Run PyTorch GPU benchmarks")
    parser.add_argument("--cpu", action="store_true", help="Run PyTorch CPU benchmarks")
    parser.add_argument("--jax", action="store_true", help="Run JAX benchmarks")
    parser.add_argument("--validate", action="store_true", help="Run cross-validation")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if args.all:
        args.cpu = args.gpu = args.jax = args.validate = True

    if not (args.cpu or args.gpu or args.jax or args.validate):
        args.cpu = True  # Default to PyTorch CPU

    if args.validate:
        run_cross_validation()
    if args.cpu:
        run_pytorch_cpu_benchmarks()
    if args.gpu:
        run_pytorch_gpu_benchmarks()
    if args.jax:
        run_jax_benchmarks()
