# Julia Reference Tests Design

**Issue:** #30 - Improve the test of gemm and batched gemm
**Date:** 2026-01-28
**Status:** Approved

## Overview

Generate ground-truth test datasets using TropicalNumbers.jl and validate the Rust/Python implementation against them. This provides cross-validation with an independent implementation.

## Test Matrix

### Dimension 1: Algebra Types (5)

| Type | Julia Equivalent | Notes |
|------|------------------|-------|
| MaxPlus | `Tropical{T}` | Most common |
| MinPlus | `TropicalMinPlus{T}` | Shortest path |
| MaxMul | `TropicalMaxMul{T}` | Probability |
| AndOr | `TropicalAndOr` | Boolean/reachability |
| CountingTropical | `CountingTropical{T}` | Path counting |

### Dimension 2: Scalar Types (3)

| Type | Applies To |
|------|------------|
| Float32 | MaxPlus, MinPlus, MaxMul, CountingTropical |
| Float64 | MaxPlus, MinPlus, MaxMul, CountingTropical |
| Bool | AndOr only |

### Dimension 3: Matrix Shapes (5)

| Shape | (M, K, N) | Purpose |
|-------|-----------|---------|
| Tiny square | (4, 4, 4) | Edge case, manual verification |
| Small square | (16, 16, 16) | Quick regression |
| Medium square | (64, 64, 64) | Typical use |
| Large square | (256, 256, 256) | Performance regime |
| Rectangular | (32, 64, 48) | Non-square coverage |

### Dimension 4: Operation Variants (4)

| Variant | Tests |
|---------|-------|
| Basic matmul | Forward pass only |
| With argmax | Forward + argmax indices |
| Batched (batch=3) | Batched forward |
| Batched with argmax | Batched forward + argmax |

### Dimension 5: Special Cases (4)

| Case | Values | Applies To |
|------|--------|------------|
| Normal random | rand(-10, 10) | All |
| With zeros | sparse zeros | MaxMul (tricky edge case) |
| With infinity | ±Inf values | MaxPlus, MinPlus |
| All negative | rand(-10, -1) | MaxPlus, MaxMul |

## Test Case Count

- **Core matrix:** 5 algebra configs × 5 shapes × 4 operation variants = ~100
- **Special cases:** 3 special cases × 3 key algebras × 2 shapes = ~18
- **Total:** ~118 test cases

## File Structure

```
tests/
├── fixtures/
│   ├── julia_generator.jl       # Run once to generate all fixtures
│   ├── README.md                # Instructions for regenerating
│   ├── maxplus_f32/
│   │   ├── square_4.json
│   │   ├── square_16.json
│   │   ├── square_64.json
│   │   ├── square_256.json
│   │   ├── rect_32x64x48.json
│   │   ├── batched_16_b3.json
│   │   ├── batched_64_b3.json
│   │   └── special_zeros.json
│   ├── maxplus_f64/
│   │   └── ... (same structure)
│   ├── minplus_f32/
│   ├── minplus_f64/
│   ├── maxmul_f32/
│   ├── maxmul_f64/
│   ├── andor_bool/
│   └── counting_f32/
└── test_julia_reference.py      # Parametrized tests loading fixtures
```

## JSON Fixture Format

```json
{
  "algebra": "maxplus",
  "scalar": "f32",
  "m": 16,
  "k": 16,
  "n": 16,
  "a": [[...], ...],
  "b": [[...], ...],
  "c_expected": [[...], ...],
  "argmax_expected": [[...], ...],
  "batch_size": null
}
```

For batched operations:
```json
{
  "algebra": "maxplus",
  "scalar": "f32",
  "m": 16,
  "k": 16,
  "n": 16,
  "batch_size": 3,
  "a": [[[...], ...], ...],
  "b": [[[...], ...], ...],
  "c_expected": [[[...], ...], ...],
  "argmax_expected": [[[...], ...], ...]
}
```

## Julia Generator Script

```julia
# tests/fixtures/julia_generator.jl

using TropicalNumbers
using JSON3
using Random

Random.seed!(42)  # Reproducibility

ALGEBRA_CONFIGS = [
    ("maxplus", "f32", Tropical{Float32}),
    ("maxplus", "f64", Tropical{Float64}),
    ("minplus", "f32", TropicalMinPlus{Float32}),
    ("minplus", "f64", TropicalMinPlus{Float64}),
    ("maxmul", "f32", TropicalMaxMul{Float32}),
    ("maxmul", "f64", TropicalMaxMul{Float64}),
    ("andor", "bool", TropicalAndOr),
    ("counting", "f32", CountingTropical{Float32}),
]

SHAPES = [
    (4, 4, 4),
    (16, 16, 16),
    (64, 64, 64),
    (256, 256, 256),
    (32, 64, 48),
]

BATCH_SIZE = 3
```

## Python Test Structure

```python
# tests/test_julia_reference.py

import pytest
import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

def load_fixture(algebra, scalar, shape_name):
    path = FIXTURES / f"{algebra}_{scalar}" / f"{shape_name}.json"
    return json.loads(path.read_text())

class TestNonBatchedMatmul:
    """Test basic matmul against Julia reference (~40 cases)."""
    pass

class TestNonBatchedWithArgmax:
    """Test matmul with argmax tracking (~40 cases)."""
    pass

class TestBatchedMatmul:
    """Test batched matmul against Julia reference (~20 cases)."""
    pass

class TestBatchedWithArgmax:
    """Test batched matmul with argmax (~20 cases)."""
    pass

class TestSpecialCases:
    """Test edge cases: zeros, infinity, negatives (~18 cases)."""
    pass
```

## Implementation Tasks

1. [ ] Create `tests/fixtures/julia_generator.jl` script
2. [ ] Run generator to create all JSON fixtures
3. [ ] Create `tests/fixtures/README.md` with regeneration instructions
4. [ ] Create `tests/test_julia_reference.py` with parametrized tests
5. [ ] Verify all tests pass
6. [ ] Commit fixtures to git (~3-5 MB)

## Summary

| Component | Count |
|-----------|-------|
| Algebra types | 5 |
| Scalar types | 3 (f32, f64, bool) |
| Shapes | 5 |
| Operation variants | 4 |
| Special cases | 18 |
| **Total test cases** | **~118** |
| **Fixture files** | **~45** |
| **Estimated size** | **~3-5 MB** |
