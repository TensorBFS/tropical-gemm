#!/bin/bash
# Metal GPU Benchmark Script for macOS
#
# Usage: ./bench_metal.sh [options]
#   --quick    Run quick benchmarks (smaller sizes)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================================================"
echo "Tropical GEMM Metal Benchmark (macOS)"
echo "========================================================================"
echo ""
echo "Project: $PROJECT_DIR"
echo "Date: $(date)"
echo ""

# Create results directory
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_DIR"

# Build release
echo "Building release binary..."
cargo build --release --example bench_comparison -p tropical-gemm-metal

# Run benchmark
echo ""
echo "Running Metal GPU benchmark..."
OUTPUT="$RESULTS_DIR/metal_${TIMESTAMP}.txt"
cargo run --release --example bench_comparison -p tropical-gemm-metal 2>&1 | tee "$OUTPUT"

echo ""
echo "========================================================================"
echo "Benchmark Complete"
echo "========================================================================"
echo "Results saved to: $OUTPUT"
