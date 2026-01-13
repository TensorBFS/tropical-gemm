#!/usr/bin/env python3
"""
PyTorch Custom Autograd Function for Tropical Matrix Multiplication.

This example demonstrates how to integrate tropical-gemm with PyTorch's
automatic differentiation system using custom autograd functions.

Tropical semirings are useful for:
- Shortest path problems (MinPlus)
- Longest path problems (MaxPlus)
- Dynamic programming on graphs
- Probabilistic inference (log-space operations)

Usage:
    pip install torch numpy
    cd crates/tropical-gemm-python
    maturin develop
    python examples/pytorch_tropical.py
"""

import torch
import numpy as np

# Import the Rust-backed tropical_gemm module
import tropical_gemm


class TropicalMaxPlusMatmul(torch.autograd.Function):
    """
    Custom autograd function for MaxPlus tropical matrix multiplication.

    Forward: C[i,j] = max_k(A[i,k] + B[k,j])

    The gradient is sparse: only the argmax index contributes to each output.
    For each output C[i,j], the gradient flows back to:
    - A[i, argmax[i,j]]
    - B[argmax[i,j], j]
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute tropical MaxPlus matmul.

        Args:
            a: Input tensor of shape (M, K)
            b: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        m, k = a.shape
        n = b.shape[1]

        # Convert to contiguous numpy arrays
        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        # Ensure contiguous layout
        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        # Call the optimized Rust implementation (returns flattened arrays)
        c_flat, argmax_flat = tropical_gemm.maxplus_matmul_with_argmax(a_np, b_np)

        # Reshape to 2D
        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        # Save argmax for backward pass
        ctx.save_for_backward(torch.from_numpy(argmax_np))
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        """
        Backward pass: compute gradients w.r.t. A and B.

        The tropical matmul gradient is sparse because only the argmax
        index contributes to each output element.
        """
        argmax, = ctx.saved_tensors
        k = ctx.k
        m = ctx.m
        n = ctx.n

        # Ensure contiguous numpy arrays
        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        # Compute gradients using the Rust backend (returns flattened arrays)
        grad_a_flat = tropical_gemm.backward_a(grad_c_np, argmax_np, k)
        grad_b_flat = tropical_gemm.backward_b(grad_c_np, argmax_np, k)

        # Reshape to 2D
        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k)).to(grad_c.device)
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k, n)).to(grad_c.device)

        return grad_a, grad_b


class TropicalMinPlusMatmul(torch.autograd.Function):
    """
    Custom autograd function for MinPlus tropical matrix multiplication.

    Forward: C[i,j] = min_k(A[i,k] + B[k,j])

    Useful for shortest path computations in graphs.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.shape
        n = b.shape[1]

        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        c_flat, argmax_flat = tropical_gemm.minplus_matmul_with_argmax(a_np, b_np)

        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        ctx.save_for_backward(torch.from_numpy(argmax_np))
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        argmax, = ctx.saved_tensors
        k = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        grad_a_flat = tropical_gemm.backward_a(grad_c_np, argmax_np, k)
        grad_b_flat = tropical_gemm.backward_b(grad_c_np, argmax_np, k)

        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k)).to(grad_c.device)
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k, n)).to(grad_c.device)

        return grad_a, grad_b


class TropicalMaxMulMatmul(torch.autograd.Function):
    """
    Custom autograd function for MaxMul tropical matrix multiplication.

    Forward: C[i,j] = max_k(A[i,k] * B[k,j])

    The backward pass differs from MaxPlus/MinPlus because the operation
    is multiplication, not addition. The chain rule gives:
    - grad_A[i,k] = grad_C[i,j] * B[k,j] if k == argmax[i,j]
    - grad_B[k,j] = grad_C[i,j] * A[i,k] if k == argmax[i,j]

    Useful for max-probability computations (non-log space).
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.shape
        n = b.shape[1]

        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        c_flat, argmax_flat = tropical_gemm.maxmul_matmul_with_argmax(a_np, b_np)

        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        # Save inputs and argmax for backward (needed for multiplicative gradient)
        ctx.save_for_backward(
            torch.from_numpy(a_np),
            torch.from_numpy(b_np),
            torch.from_numpy(argmax_np),
        )
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        a, b, argmax = ctx.saved_tensors
        k_dim = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)
        a_np = a.numpy().astype(np.float32)
        b_np = b.numpy().astype(np.float32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        # Use multiplicative backward rule
        grad_a_flat = tropical_gemm.maxmul_backward_a(grad_c_np, argmax_np, b_np)
        grad_b_flat = tropical_gemm.maxmul_backward_b(grad_c_np, argmax_np, a_np)

        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k_dim)).to(grad_c.device)
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k_dim, n)).to(grad_c.device)

        return grad_a, grad_b


# Convenience functions
def tropical_maxplus_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Functional interface for MaxPlus tropical matmul."""
    return TropicalMaxPlusMatmul.apply(a, b)


def tropical_minplus_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Functional interface for MinPlus tropical matmul."""
    return TropicalMinPlusMatmul.apply(a, b)


def tropical_maxmul_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Functional interface for MaxMul tropical matmul."""
    return TropicalMaxMulMatmul.apply(a, b)


def verify_gradients():
    """Verify gradients using manual check."""
    print("=" * 60)
    print("Gradient Verification")
    print("=" * 60)

    print("\nManual gradient check:")

    a = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    b = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

    # Forward pass
    c = tropical_maxplus_matmul(a, b)
    loss = c.sum()
    loss.backward()

    print(f"  Input A shape: {a.shape}")
    print(f"  Input B shape: {b.shape}")
    print(f"  Output C shape: {c.shape}")
    print(f"  grad_A shape: {a.grad.shape}")
    print(f"  grad_B shape: {b.grad.shape}")
    print(f"  grad_A sum: {a.grad.sum().item():.4f}")
    print(f"  grad_B sum: {b.grad.sum().item():.4f}")

    # The sum of gradients should equal the number of elements in C
    # because each C[i,j] contributes exactly 1 to both grad_A and grad_B
    expected_sum = c.numel()
    actual_sum_a = a.grad.sum().item()
    actual_sum_b = b.grad.sum().item()

    print(f"\n  Expected grad sum (num elements in C): {expected_sum}")
    print(f"  Actual grad_A sum: {actual_sum_a:.1f}")
    print(f"  Actual grad_B sum: {actual_sum_b:.1f}")

    if abs(actual_sum_a - expected_sum) < 0.1 and abs(actual_sum_b - expected_sum) < 0.1:
        print("  ✓ Gradient sum check passed!")
    else:
        print("  ✗ Gradient sum check failed!")


def demo_shortest_path():
    """
    Demonstrate using tropical MinPlus for shortest path computation.

    In graph algorithms, the adjacency matrix A contains edge weights,
    and A^n (tropical power) gives shortest paths of length n.
    """
    print("\n" + "=" * 60)
    print("Shortest Path Example (MinPlus)")
    print("=" * 60)

    # Adjacency matrix for a simple graph (inf = no edge)
    inf = float("inf")
    adj = torch.tensor(
        [
            [0.0, 1.0, inf, 4.0],
            [inf, 0.0, 2.0, inf],
            [inf, inf, 0.0, 1.0],
            [inf, inf, inf, 0.0],
        ],
        dtype=torch.float32,
    )

    print("\nAdjacency matrix (edge weights, inf = no edge):")
    print(adj.numpy())

    # Compute 2-hop shortest paths
    two_hop = tropical_minplus_matmul(adj, adj)
    print("\n2-hop shortest paths:")
    print(two_hop.numpy())

    # Compute 3-hop shortest paths
    three_hop = tropical_minplus_matmul(two_hop, adj)
    print("\n3-hop shortest paths:")
    print(three_hop.numpy())


def demo_longest_path():
    """
    Demonstrate using tropical MaxPlus for longest path computation.

    Useful in critical path analysis (project scheduling).
    """
    print("\n" + "=" * 60)
    print("Longest Path Example (MaxPlus)")
    print("=" * 60)

    # Edge weights for a DAG (task durations)
    neg_inf = float("-inf")
    adj = torch.tensor(
        [
            [0.0, 3.0, 2.0, neg_inf],
            [neg_inf, 0.0, neg_inf, 4.0],
            [neg_inf, neg_inf, 0.0, 5.0],
            [neg_inf, neg_inf, neg_inf, 0.0],
        ],
        dtype=torch.float32,
    )

    print("\nAdjacency matrix (edge weights, -inf = no edge):")
    print(adj.numpy())

    # Compute 2-hop longest paths
    two_hop = tropical_maxplus_matmul(adj, adj)
    print("\n2-hop longest paths:")
    print(two_hop.numpy())


def demo_optimization():
    """
    Demonstrate using tropical matmul in an optimization loop.

    This shows that gradients flow correctly through the tropical operations.
    """
    print("\n" + "=" * 60)
    print("Optimization Example")
    print("=" * 60)

    torch.manual_seed(42)

    # Create learnable parameters
    a = torch.randn(4, 5, requires_grad=True)
    b = torch.randn(5, 3, requires_grad=True)

    # Target output
    target = torch.randn(4, 3)

    optimizer = torch.optim.Adam([a, b], lr=0.1)

    print("\nOptimizing to match target using tropical MaxPlus matmul:")
    for step in range(5):
        optimizer.zero_grad()

        # Forward pass through tropical matmul
        c = tropical_maxplus_matmul(a, b)

        # MSE loss
        loss = ((c - target) ** 2).mean()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        print(f"  Step {step + 1}: loss = {loss.item():.6f}")


def benchmark():
    """Simple performance comparison."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    import time

    sizes = [64, 128, 256, 512]

    for n in sizes:
        a = torch.randn(n, n, dtype=torch.float32)
        b = torch.randn(n, n, dtype=torch.float32)

        # Warm up
        _ = tropical_maxplus_matmul(a, b)

        # Benchmark
        start = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            c = tropical_maxplus_matmul(a, b)
        elapsed = (time.perf_counter() - start) / iterations * 1000

        print(f"  {n}x{n}: {elapsed:.3f} ms per matmul")


if __name__ == "__main__":
    print("Tropical GEMM PyTorch Integration Demo")
    print("=" * 60)

    verify_gradients()
    demo_shortest_path()
    demo_longest_path()
    demo_optimization()
    benchmark()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
