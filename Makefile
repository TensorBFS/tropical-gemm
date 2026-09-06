# Makefile for tropical-gemm
# Automates environment setup, benchmarking, testing, documentation, and examples

.PHONY: all build check test bench docs clean help
.PHONY: setup setup-rust setup-python setup-cuda
.PHONY: test-rust test-python test-all
.PHONY: bench-cpu bench-cuda bench-all
.PHONY: example-rust example-python example-mnist example-mnist-gpu
.PHONY: docs-build docs-serve docs-deploy docs-book docs-book-serve
.PHONY: fmt clippy lint coverage

# Default target
all: build test

#==============================================================================
# Help
#==============================================================================

help:
	@echo "tropical-gemm Makefile"
	@echo ""
	@echo "Setup targets:"
	@echo "  setup          - Setup complete development environment"
	@echo "  setup-rust     - Install Rust toolchain and components"
	@echo "  setup-python   - Setup Python virtual environment"
	@echo "  setup-cuda     - Verify CUDA installation"
	@echo ""
	@echo "Build targets:"
	@echo "  build          - Build all crates in release mode"
	@echo "  build-debug    - Build all crates in debug mode"
	@echo "  check          - Check all crates for errors"
	@echo ""
	@echo "Test targets:"
	@echo "  test           - Run all tests (Rust + Python)"
	@echo "  test-rust      - Run Rust tests only"
	@echo "  test-python    - Run Python tests only"
	@echo ""
	@echo "Benchmark targets:"
	@echo "  bench          - Run all benchmarks"
	@echo "  bench-cpu      - Run CPU benchmarks"
	@echo "  bench-cuda     - Run CUDA benchmarks"
	@echo ""
	@echo "Example targets:"
	@echo "  example-rust      - Run Rust examples"
	@echo "  example-python    - Run Python PyTorch example"
	@echo "  example-mnist     - Run MNIST tropical example (CPU)"
	@echo "  example-mnist-gpu - Run MNIST tropical example (GPU)"
	@echo ""
	@echo "Documentation targets:"
	@echo "  docs           - Build all documentation (API + user guide)"
	@echo "  docs-build     - Build Rust API documentation"
	@echo "  docs-book      - Build mdBook user guide"
	@echo "  docs-book-serve- Serve mdBook locally (port 3000)"
	@echo "  docs-serve     - Serve API docs locally (port 8000)"
	@echo "  docs-deploy    - Deploy documentation to GitHub Pages"
	@echo ""
	@echo "Code quality targets:"
	@echo "  fmt            - Format code"
	@echo "  clippy         - Run clippy lints"
	@echo "  lint           - Run all lints (fmt check + clippy)"
	@echo "  coverage       - Generate test coverage report"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean          - Clean build artifacts"

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
	@echo "Setting up Python environment..."
	cd crates/tropical-gemm-python && \
		python -m venv .venv && \
		unset CONDA_PREFIX && \
		. .venv/bin/activate && \
		pip install --upgrade pip && \
		pip install maturin pytest numpy && \
		maturin develop
	@echo "Python setup complete."

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

test-python:
	@echo "Running Python tests..."
	cd crates/tropical-gemm-python && \
		unset CONDA_PREFIX && \
		. .venv/bin/activate && \
		pytest tests/ -v
	@echo "Python tests complete."

test-all: test

#==============================================================================
# Benchmarks
#==============================================================================

bench: bench-cpu bench-cuda

bench-cpu:
	@echo "Running CPU benchmarks..."
	cargo run --release --example bench_rust -p tropical-gemm
	@echo "CPU benchmarks complete."

bench-cuda:
	@echo "Running CUDA benchmarks..."
	@if which nvcc > /dev/null 2>&1; then \
		LD_LIBRARY_PATH=/usr/local/cuda/lib64:$$LD_LIBRARY_PATH \
		cargo run --release --example bench_cuda_vs_cpu -p tropical-gemm-cuda; \
	else \
		echo "CUDA not available, skipping CUDA benchmarks."; \
	fi
	@echo "CUDA benchmarks complete."

bench-all: bench

#==============================================================================
# Examples
#==============================================================================

example-rust:
	@echo "Running Rust examples..."
	cargo run --release --example basic -p tropical-gemm
	cargo run --release --example shortest_path -p tropical-gemm
	@echo "Rust examples complete."

example-python:
	@echo "Running Python examples..."
	cd crates/tropical-gemm-python && \
		unset CONDA_PREFIX && \
		. .venv/bin/activate && \
		pip install torch --quiet 2>/dev/null || true && \
		maturin develop && \
		python examples/pytorch_tropical.py
	@echo "Python examples complete."

example-mnist:
	@echo "Running MNIST tropical example (CPU)..."
	cd crates/tropical-gemm-python && \
		unset CONDA_PREFIX && \
		. .venv/bin/activate && \
		pip install torch torchvision --quiet 2>/dev/null || true && \
		maturin develop && \
		python examples/mnist_tropical.py
	@echo "MNIST example complete."

example-mnist-gpu:
	@echo "Running MNIST tropical example (GPU)..."
	cd crates/tropical-gemm-python && \
		unset CONDA_PREFIX && \
		. .venv/bin/activate && \
		pip install torch torchvision --quiet 2>/dev/null || true && \
		maturin develop --features cuda && \
		python examples/mnist_tropical.py --gpu
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
	@echo "Note: This requires gh-pages branch setup."
	@echo "Run: ghp-import -n -p -f target/doc"
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
	rm -rf crates/tropical-gemm-python/.venv
	rm -rf crates/tropical-gemm-python/target
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."
