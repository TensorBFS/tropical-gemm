# Makefile for tropical-gemm
# Automates environment setup, benchmarking, testing, documentation, and examples

.PHONY: all build build-debug check clean help
.PHONY: setup setup-rust setup-python setup-python-gpu setup-cuda
.PHONY: test test-rust test-python test-python-gpu
.PHONY: bench bench-cpu bench-cuda bench-python bench-python-gpu
.PHONY: validate
.PHONY: example-rust example-python example-mnist example-mnist-gpu
.PHONY: docs docs-build docs-serve docs-deploy docs-book docs-book-serve
.PHONY: fmt fmt-check clippy lint coverage

# Python package directory
PYTHON_PKG := crates/tropical-gemm-python

# Default CUDA version for PyTorch (override with: make setup-python-gpu CUDA=cu118)
CUDA ?= cu121

# Default target
all: build test

#==============================================================================
# Help
#==============================================================================

help:
	@echo "tropical-gemm Makefile"
	@echo ""
	@echo "Setup targets:"
	@echo "  setup              - Setup complete dev environment (Rust + Python CPU)"
	@echo "  setup-rust         - Install Rust toolchain and components"
	@echo "  setup-python       - Setup Python with uv (CPU only)"
	@echo "  setup-python-gpu   - Setup Python with CUDA PyTorch (default: cu121)"
	@echo "                       Override: make setup-python-gpu CUDA=cu118"
	@echo "  setup-cuda         - Verify CUDA installation"
	@echo ""
	@echo "Build targets:"
	@echo "  build              - Build all Rust crates (release)"
	@echo "  build-debug        - Build all Rust crates (debug)"
	@echo "  build-python       - Build Python extension (CPU)"
	@echo "  build-python-gpu   - Build Python extension (with CUDA)"
	@echo "  check              - Check all crates for errors"
	@echo ""
	@echo "Test targets:"
	@echo "  test               - Run all tests (Rust + Python)"
	@echo "  test-rust          - Run Rust tests only"
	@echo "  test-python        - Run Python tests (CPU build)"
	@echo "  test-python-gpu    - Run Python tests (CUDA build)"
	@echo ""
	@echo "Benchmark targets:"
	@echo "  bench              - Run all benchmarks"
	@echo "  bench-cpu          - Run Rust CPU benchmarks"
	@echo "  bench-cuda         - Run Rust CUDA benchmarks"
	@echo "  bench-python       - Run Python CPU benchmarks (PyTorch + JAX)"
	@echo "  bench-python-gpu   - Run Python GPU benchmarks (PyTorch + JAX)"
	@echo "  validate           - Cross-validate PyTorch vs JAX results"
	@echo ""
	@echo "Example targets:"
	@echo "  example-rust       - Run Rust examples"
	@echo "  example-python     - Run Python PyTorch example"
	@echo "  example-mnist      - Run MNIST tropical example (CPU)"
	@echo "  example-mnist-gpu  - Run MNIST tropical example (GPU)"
	@echo ""
	@echo "Documentation targets:"
	@echo "  docs               - Build all documentation"
	@echo "  docs-build         - Build Rust API documentation"
	@echo "  docs-book          - Build mdBook user guide"
	@echo "  docs-book-serve    - Serve mdBook locally (port 3000)"
	@echo "  docs-serve         - Serve API docs locally (port 8000)"
	@echo ""
	@echo "Code quality targets:"
	@echo "  fmt                - Format code"
	@echo "  clippy             - Run clippy lints"
	@echo "  lint               - Run all lints"
	@echo "  coverage           - Generate test coverage report"
	@echo ""
	@echo "Quick start (GPU):"
	@echo "  make setup-python-gpu   # Install PyTorch with CUDA"
	@echo "  make test-python-gpu    # Build with CUDA and run tests"

#==============================================================================
# Environment Setup
#==============================================================================

setup: setup-rust setup-python setup-cuda
	@echo "Development environment setup complete!"

setup-rust:
	@echo "Setting up Rust toolchain..."
	rustup update stable
	rustup component add rustfmt clippy
	@echo "Rust setup complete."

setup-python:
	@echo "Setting up Python environment with uv..."
	cd $(PYTHON_PKG) && uv pip install -e ".[dev]"
	@echo "Python setup complete."

setup-python-gpu:
	@echo "Setting up Python environment with CUDA..."
	cd $(PYTHON_PKG) && uv pip uninstall torch -q 2>/dev/null || true
	cd $(PYTHON_PKG) && uv pip install torch --index-url https://download.pytorch.org/whl/$(CUDA)
	cd $(PYTHON_PKG) && uv pip install -e ".[dev]"
	@echo ""
	@echo "Verifying CUDA..."
	@cd $(PYTHON_PKG) && uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

setup-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc > /dev/null 2>&1 || (echo "Warning: nvcc not found. CUDA features may not work." && exit 0)
	@nvcc --version 2>/dev/null || echo "CUDA not available"
	@echo "CUDA check complete."

#==============================================================================
# Build
#==============================================================================

build:
	cargo build --release --workspace

build-debug:
	cargo build --workspace

build-python:
	cd $(PYTHON_PKG) && uv run maturin develop --release

build-python-gpu:
	cd $(PYTHON_PKG) && uv run maturin develop --release --features cuda

check:
	cargo check --workspace

#==============================================================================
# Testing
#==============================================================================

test: test-rust test-python

test-rust:
	@echo "Running Rust tests..."
	cargo test --workspace --release
	@echo "Rust tests complete."

test-python: build-python
	@echo "Running Python tests..."
	cd $(PYTHON_PKG) && uv run pytest tests/ -v
	@echo "Python tests complete."

test-python-gpu: build-python-gpu
	@echo "Running Python tests (with CUDA)..."
	cd $(PYTHON_PKG) && uv run pytest tests/ -v
	@echo "Python tests complete."

#==============================================================================
# Benchmarks
#==============================================================================

bench: bench-cpu bench-cuda

bench-cpu:
	@echo "Running Rust CPU benchmarks..."
	cargo run --release --example bench_rust -p tropical-gemm
	@echo "CPU benchmarks complete."

bench-cuda:
	@echo "Running Rust CUDA benchmarks..."
	@if which nvcc > /dev/null 2>&1; then \
		LD_LIBRARY_PATH=/usr/local/cuda/lib64:$$LD_LIBRARY_PATH \
		cargo run --release --example bench_cuda_vs_cpu -p tropical-gemm-cuda; \
	else \
		echo "CUDA not available, skipping CUDA benchmarks."; \
	fi
	@echo "CUDA benchmarks complete."

bench-python: build-python
	@echo "Running Python CPU benchmarks (PyTorch + JAX)..."
	cd $(PYTHON_PKG) && uv run python benchmarks/benchmark.py --cpu
	@echo "Python CPU benchmarks complete."

bench-python-gpu: build-python-gpu
	@echo "Running Python GPU benchmarks (PyTorch + JAX)..."
	cd $(PYTHON_PKG) && uv run python benchmarks/benchmark.py --gpu
	@echo "Python GPU benchmarks complete."

validate: build-python
	@echo "Cross-validating PyTorch vs JAX..."
	cd $(PYTHON_PKG) && uv run python benchmarks/benchmark.py --validate
	@echo "Validation complete."

#==============================================================================
# Examples
#==============================================================================

example-rust:
	@echo "Running Rust examples..."
	cargo run --release --example basic -p tropical-gemm
	cargo run --release --example shortest_path -p tropical-gemm
	@echo "Rust examples complete."

example-python: build-python
	@echo "Running Python examples..."
	cd $(PYTHON_PKG) && uv run python examples/pytorch_tropical.py
	@echo "Python examples complete."

example-mnist: build-python
	@echo "Running MNIST tropical example (CPU)..."
	cd $(PYTHON_PKG) && uv run python examples/mnist_tropical.py
	@echo "MNIST example complete."

example-mnist-gpu: build-python-gpu
	@echo "Running MNIST tropical example (GPU)..."
	cd $(PYTHON_PKG) && uv run python examples/mnist_tropical.py --gpu
	@echo "MNIST example complete."

#==============================================================================
# Documentation
#==============================================================================

docs: docs-build docs-book

docs-build:
	@echo "Building Rust API documentation..."
	cargo doc --workspace --no-deps
	@echo "API documentation built at target/doc/"

docs-book:
	@echo "Building mdBook user guide..."
	@which mdbook > /dev/null 2>&1 || (echo "Install mdbook: cargo install mdbook" && exit 1)
	mdbook build docs/
	@echo "User guide built at docs/book/"

docs-book-serve:
	@echo "Serving mdBook at http://localhost:3000"
	@which mdbook > /dev/null 2>&1 || (echo "Install mdbook: cargo install mdbook" && exit 1)
	mdbook serve docs/

docs-serve: docs-build
	@echo "Serving API documentation at http://localhost:8000"
	@cd target/doc && python -m http.server 8000

docs-deploy: docs-build docs-book
	@echo "Deploying documentation to GitHub Pages..."
	@which ghp-import > /dev/null 2>&1 || (echo "Install ghp-import: pip install ghp-import" && exit 1)
	ghp-import -n -p -f target/doc

#==============================================================================
# Code Quality
#==============================================================================

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace -- -D warnings

lint: fmt-check clippy

coverage:
	@echo "Generating test coverage..."
	cargo tarpaulin --workspace --out Html --output-dir coverage/
	@echo "Coverage report at coverage/tarpaulin-report.html"

#==============================================================================
# Cleanup
#==============================================================================

clean:
	cargo clean
	rm -rf coverage/
	rm -rf docs/book/
	rm -rf $(PYTHON_PKG)/target
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.so" -delete 2>/dev/null || true
	@echo "Clean complete."
