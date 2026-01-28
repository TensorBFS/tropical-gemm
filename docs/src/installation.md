# Installation

## Rust Crate

Add to your `Cargo.toml`:

```toml
[dependencies]
tropical-gemm = "0.1"

# For GPU acceleration (optional):
tropical-gemm-cuda = "0.1"
```

## Python Package

### From PyPI (Recommended)

```bash
# Basic installation
pip install tropical-gemm

# With PyTorch support for automatic differentiation
pip install tropical-gemm[torch]

# With JAX support (requires Python >= 3.10)
pip install tropical-gemm[jax]

# For development
pip install tropical-gemm[dev]
```

### Optional Dependencies

The Python package has optional extras:

| Extra | Command | Description |
|-------|---------|-------------|
| `torch` | `pip install tropical-gemm[torch]` | PyTorch integration with autograd support |
| `jax` | `pip install tropical-gemm[jax]` | JAX integration with custom_vjp support (Python >= 3.10) |
| `dev` | `pip install tropical-gemm[dev]` | Development dependencies (pytest, torch, jax) |

### From Source

```bash
# Clone the repository
git clone https://github.com/TensorBFS/tropical-gemm
cd tropical-gemm/crates/tropical-gemm-python

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install maturin and build
pip install maturin
maturin develop --release

# With CUDA support
maturin develop --release --features cuda
```

### Verify Installation

```python
import tropical_gemm
import numpy as np

a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

c = tropical_gemm.maxplus_matmul(a, b)
print(c)  # [[5. 6.] [7. 8.]]

# Check GPU availability
print(f"CUDA available: {tropical_gemm.cuda_available()}")
```

### Verify PyTorch Integration

```python
import torch
from tropical_gemm.pytorch import tropical_maxplus_matmul, GPU_AVAILABLE

print(f"GPU available: {GPU_AVAILABLE}")

a = torch.randn(3, 4, requires_grad=True)
b = torch.randn(4, 5, requires_grad=True)

c = tropical_maxplus_matmul(a, b)
c.sum().backward()

print(f"grad_a: {a.grad.shape}")  # (3, 4)
print(f"grad_b: {b.grad.shape}")  # (4, 5)
```

### Verify JAX Integration

```python
import jax
import jax.numpy as jnp
from tropical_gemm.jax import tropical_maxplus_matmul, GPU_AVAILABLE

print(f"GPU available: {GPU_AVAILABLE}")

a = jax.random.normal(jax.random.PRNGKey(0), (3, 4))
b = jax.random.normal(jax.random.PRNGKey(1), (4, 5))

c = tropical_maxplus_matmul(a, b)

# Compute gradients
def loss_fn(a, b):
    return tropical_maxplus_matmul(a, b).sum()

grad_a = jax.grad(loss_fn, argnums=0)(a, b)
grad_b = jax.grad(loss_fn, argnums=1)(a, b)

print(f"grad_a: {grad_a.shape}")  # (3, 4)
print(f"grad_b: {grad_b.shape}")  # (4, 5)
```

## CUDA Setup

For GPU acceleration, ensure CUDA is properly installed:

```bash
# Check CUDA installation
nvcc --version

# If not found, install CUDA toolkit
# Ubuntu:
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA:
# https://developer.nvidia.com/cuda-downloads
```

The CUDA kernels are compiled at runtime using NVRTC, so you don't need to
compile the library with a specific CUDA version.

### Building Python Package with CUDA

```bash
cd crates/tropical-gemm-python

# Build with CUDA feature
maturin develop --features cuda

# Or for release
maturin build --release --features cuda
```

## Building from Source

```bash
# Clone
git clone https://github.com/TensorBFS/tropical-gemm
cd tropical-gemm

# Build all crates
cargo build --release --workspace

# Run tests
cargo test --workspace

# Build documentation
cargo doc --workspace --no-deps --open
```

## Using the Makefile

A Makefile is provided for common tasks:

```bash
make help          # Show all targets
make setup         # Setup development environment
make build         # Build in release mode
make test          # Run all tests
make docs          # Build documentation
make bench         # Run benchmarks
```
