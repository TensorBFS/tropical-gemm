.PHONY: help build build-release test benchmark benchmark-gpu benchmark-all clean

help:
	@echo "tropical-gemm-python"
	@echo ""
	@echo "Build:"
	@echo "  make build          - Build debug mode"
	@echo "  make build-release  - Build release mode"
	@echo ""
	@echo "Test:"
	@echo "  make test           - Run all tests"
	@echo ""
	@echo "Benchmark:"
	@echo "  make benchmark      - Run CPU benchmarks"
	@echo "  make benchmark-gpu  - Run GPU benchmarks"
	@echo "  make benchmark-all  - Run CPU + GPU benchmarks"
	@echo ""
	@echo "Misc:"
	@echo "  make clean          - Clean build artifacts"

build:
	maturin develop

build-release:
	maturin develop --release

test:
	pytest tests/ -v

benchmark: build-release
	python benchmarks/benchmark.py --cpu

benchmark-gpu: build-release
	python benchmarks/benchmark.py --gpu

benchmark-all: build-release
	python benchmarks/benchmark.py --cpu --gpu

clean:
	rm -rf target/ *.egg-info/ dist/ build/
	find . -name "*.so" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
