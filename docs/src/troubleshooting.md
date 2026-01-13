# Troubleshooting

Common issues and solutions for tropical-gemm.

## Installation Issues

### Rust Compilation Errors

**Error: "missing SIMD intrinsics"**

```
error[E0433]: failed to resolve: use of undeclared crate or module `core_arch`
```

**Solution**: Update Rust to latest stable:
```bash
rustup update stable
```

**Error: "target feature `avx2` is not enabled"**

This is expected on non-x86 platforms. The portable fallback will be used automatically.

### CUDA Issues

**Error: "CUDA driver not found"**

```
CudaError: CUDA driver version is insufficient
```

**Solution**:
1. Install/update NVIDIA drivers
2. Verify with `nvidia-smi`
3. Install CUDA Toolkit

**Error: "nvcc not found"**

```
CudaError: Failed to compile kernel: nvcc not found
```

**Solution**:
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH

# Verify
nvcc --version
```

**Error: "Kernel compilation failed"**

```
CudaError: CompilationFailed: ...
```

**Solution**:
1. Check CUDA version compatibility (requires 11.0+)
2. Ensure CUDA headers are installed
3. Try reinstalling CUDA Toolkit

### Python Binding Issues

**Error: "module 'tropical_gemm' not found"**

```python
>>> import tropical_gemm
ModuleNotFoundError: No module named 'tropical_gemm'
```

**Solution**:
```bash
cd crates/tropical-gemm-python
pip install maturin
maturin develop --release
```

**Error: "symbol not found in flat namespace"** (macOS)

```
ImportError: dlopen(...): symbol not found in flat namespace
```

**Solution**: Rebuild with correct Python version:
```bash
# Ensure using correct Python
which python
python --version

# Rebuild
maturin develop --release
```

**Error: "dtype mismatch"**

```
TypeError: Expected float32 array, got float64
```

**Solution**: Explicitly cast to float32:
```python
import numpy as np
a = a.astype(np.float32)
b = b.astype(np.float32)
c = tropical_gemm.maxplus_matmul(a, b)
```

## Runtime Issues

### Incorrect Results

**Symptom: All outputs are -inf or inf**

This typically means input contains NaN or inf values:

```rust
// Check for invalid values
for &x in data.iter() {
    if x.is_nan() || x.is_infinite() {
        panic!("Invalid input value: {}", x);
    }
}
```

**Symptom: Results differ between CPU and GPU**

Small numerical differences are expected due to floating-point associativity. For MaxPlus/MinPlus, results should be identical (only comparisons).

For MaxMul, small differences may occur:
```rust
// Allow small tolerance
let diff = (cpu_result - gpu_result).abs();
assert!(diff < 1e-5, "Results differ by {}", diff);
```

### Performance Issues

**Symptom: GPU slower than CPU**

For small matrices, transfer overhead dominates:

```rust
// Rule of thumb: GPU beneficial for N > 256
if n < 256 {
    // Use CPU
    tropical_matmul::<MaxPlus<f32>>(&a, m, k, &b, n)
} else {
    // Use GPU
    tropical_matmul_gpu::<MaxPlus<f32>>(&a, m, k, &b, n)?
}
```

**Symptom: CPU slower than expected**

Check SIMD detection:
```rust
use tropical_gemm::simd_level;
println!("SIMD level: {:?}", simd_level());
// Should be Avx2 or Avx512 on modern x86
```

### Memory Issues

**Error: "out of memory" (GPU)**

```
CudaError: Out of memory
```

**Solution**:
1. Use smaller batch sizes
2. Process matrices sequentially
3. Free unused GPU memory

```rust
// Process in chunks
for chunk in matrices.chunks(batch_size) {
    let result = process_batch(&ctx, chunk)?;
    // Results are downloaded, GPU memory freed
}
```

**Error: "allocation failed" (CPU)**

Large matrices may exceed available RAM:

```rust
// Estimate memory needed
let bytes = m * n * std::mem::size_of::<f32>();
println!("Matrix requires {} MB", bytes / 1024 / 1024);
```

## PyTorch Issues

### Gradient Issues

**Symptom: Gradients are all zeros**

Check that tensors require gradients:
```python
a = torch.randn(4, 5, requires_grad=True)  # Must be True
b = torch.randn(5, 3, requires_grad=True)

c = TropicalMaxPlusMatmul.apply(a, b)
loss = c.sum()
loss.backward()

print(a.grad)  # Should not be None
```

**Symptom: "RuntimeError: element 0 of tensors does not require grad"**

Ensure input tensors have `requires_grad=True`:
```python
a = torch.tensor([[1.0, 2.0]], requires_grad=True)
# Not: a = torch.tensor([[1.0, 2.0]])  # No gradients!
```

### Device Mismatch

**Error: "Expected all tensors on same device"**

```python
# Ensure both inputs on same device
a = a.to('cuda')
b = b.to('cuda')
c = TropicalMaxPlusMatmul.apply(a, b)
```

## Getting Help

If you encounter issues not covered here:

1. Check GitHub issues: https://github.com/TensorBFS/tropical-gemm/issues
2. Open a new issue with:
   - Error message
   - Rust/Python version
   - OS and hardware
   - Minimal reproduction code

### Diagnostic Information

Include this in bug reports:

```bash
# Rust version
rustc --version
cargo --version

# CUDA (if applicable)
nvcc --version
nvidia-smi

# Python (if applicable)
python --version
pip show tropical_gemm
```
