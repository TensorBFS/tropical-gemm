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

# Optional: with PyTorch support
pip install torch
```

### Verify Installation

```python
import tropical_gemm
import numpy as np

a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

c = tropical_gemm.maxplus_matmul(a, b)
print(c)  # [[5. 6.] [7. 8.]]
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
