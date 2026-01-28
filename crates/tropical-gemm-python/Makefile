.PHONY: help setup setup-gpu build build-gpu build-release build-release-gpu test test-gpu validate benchmark benchmark-gpu benchmark-jax benchmark-all clean

# Default CUDA version (override with: make setup-gpu CUDA=cu118)
CUDA ?= cu121

help:
	@echo "tropical-gemm-python"
	@echo ""
	@echo "Setup (run once):"
	@echo "  make setup          - Install dependencies (CPU only)"
	@echo "  make setup-gpu      - Install dependencies with CUDA (default: cu121)"
	@echo "                        Override CUDA version: make setup-gpu CUDA=cu118"
	@echo ""
	@echo "Build:"
	@echo "  make build          - Build CPU-only (debug)"
	@echo "  make build-release  - Build CPU-only (release)"
	@echo "  make build-gpu      - Build with CUDA support (debug)"
	@echo "  make build-release-gpu - Build with CUDA support (release)"
	@echo ""
	@echo "Test:"
	@echo "  make test           - Run all tests (GPU tests skip if no CUDA)"
	@echo "  make test-gpu       - Build with CUDA and run all tests"
	@echo "  make validate       - Cross-validate PyTorch/JAX results"
	@echo ""
	@echo "Benchmark:"
	@echo "  make benchmark      - Run CPU benchmarks"
	@echo "  make benchmark-gpu  - Run GPU benchmarks (requires CUDA build)"
	@echo "  make benchmark-jax  - Run JAX benchmarks"
	@echo "  make benchmark-all  - Run all benchmarks"
	@echo ""
	@echo "Clean:"
	@echo "  make clean          - Remove build artifacts"
	@echo ""
	@echo "Quick start (GPU):"
	@echo "  make setup-gpu      # Install PyTorch with CUDA"
	@echo "  make build-release-gpu  # Build extension with CUDA"
	@echo "  make test           # Run tests (GPU tests will run)"

# =============================================================================
# Setup
# =============================================================================

setup:
	uv pip install -e ".[dev]"

setup-gpu:
	uv pip uninstall torch -q 2>/dev/null || true
	uv pip install torch --index-url https://download.pytorch.org/whl/$(CUDA)
	uv pip install -e ".[dev]"
	@echo ""
	@echo "Verifying CUDA..."
	@uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# =============================================================================
# Build
# =============================================================================

build:
	uv run maturin develop

build-release:
	uv run maturin develop --release

build-gpu:
	uv run maturin develop --features cuda

build-release-gpu:
	uv run maturin develop --release --features cuda

# =============================================================================
# Test
# =============================================================================

test:
	uv run pytest tests/ -v

test-gpu: build-release-gpu
	uv run pytest tests/ -v

validate: build-release
	uv run python benchmarks/benchmark.py --validate

# =============================================================================
# Benchmark
# =============================================================================

benchmark: build-release
	uv run python benchmarks/benchmark.py --cpu

benchmark-gpu: build-release-gpu
	uv run python benchmarks/benchmark.py --gpu

benchmark-jax: build-release
	uv run python benchmarks/benchmark.py --jax

benchmark-all: build-release-gpu
	uv run python benchmarks/benchmark.py --all

# =============================================================================
# Clean
# =============================================================================

clean:
	rm -rf target/ *.egg-info/ dist/ build/
	find . -name "*.so" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
