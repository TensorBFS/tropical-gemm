.PHONY: help build build-release test benchmark benchmark-gpu benchmark-jax benchmark-all validate clean

help:
	@echo "tropical-gemm-python"
	@echo ""
	@echo "Build:"
	@echo "  make build          - Build debug mode"
	@echo "  make build-release  - Build release mode"
	@echo ""
	@echo "Test:"
	@echo "  make test           - Run all tests"
	@echo "  make validate       - Cross-validate PyTorch/JAX results"
	@echo ""
	@echo "Benchmark:"
	@echo "  make benchmark      - Run PyTorch CPU benchmarks"
	@echo "  make benchmark-gpu  - Run PyTorch GPU benchmarks"
	@echo "  make benchmark-jax  - Run JAX benchmarks"
	@echo "  make benchmark-all  - Run all benchmarks"
	@echo ""
	@echo "Misc:"
	@echo "  make clean          - Clean build artifacts"

build:
	maturin develop

build-release:
	maturin develop --release

test:
	pytest tests/ -v

validate: build-release
	python benchmarks/benchmark.py --validate

benchmark: build-release
	python benchmarks/benchmark.py --cpu

benchmark-gpu: build-release
	python benchmarks/benchmark.py --gpu

benchmark-jax: build-release
	python benchmarks/benchmark.py --jax

benchmark-all: build-release
	python benchmarks/benchmark.py --all

clean:
	rm -rf target/ *.egg-info/ dist/ build/
	find . -name "*.so" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
