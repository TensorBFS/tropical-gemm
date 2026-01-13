# Getting Started

This section covers how to install and start using tropical-gemm.

## Overview

tropical-gemm is organized as a Cargo workspace with three crates:

| Crate | Description |
|-------|-------------|
| `tropical-gemm` | Core library with CPU implementation |
| `tropical-gemm-cuda` | Optional GPU acceleration via CUDA |
| `tropical-gemm-python` | Python bindings for NumPy/PyTorch |

## System Requirements

### CPU
- Rust 1.70 or later
- x86-64 (AVX2/AVX-512) or ARM64 (NEON) for best performance

### GPU (optional)
- NVIDIA GPU with compute capability 3.5+
- CUDA Toolkit 11.0 or later
- `nvcc` in PATH

### Python (optional)
- Python 3.8+
- NumPy 1.20+
- PyTorch 2.0+ (for autograd integration)

## Next Steps

- [Installation](./installation.md) - Detailed installation instructions
- [Quick Start](./quick-start.md) - Your first tropical matrix multiplication
