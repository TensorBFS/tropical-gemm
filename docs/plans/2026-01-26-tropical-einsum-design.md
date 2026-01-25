# Tropical Einsum Library Design

**Date:** 2026-01-26
**Status:** Draft
**Related Issue:** [#21 - Proposal: Extend to general tensor einsum operations](https://github.com/TensorBFS/tropical-gemm/issues/21)

## Summary

Design for a new Rust crate (`tropical-einsum`) that provides algebra-agnostic tensor network contraction with optimized ordering. Supports both tropical semirings and standard arithmetic, with CPU (SIMD) and CUDA backends.

## Goals

1. **Unified tensor abstraction** - Works for tropical and standard arithmetic
2. **Optimized contraction ordering** - Integration with [omeco](https://github.com/GiggleLiu/omeco)
3. **High performance** - Reuses existing `tropical-gemm` kernels
4. **CPU + CUDA support** - Same API for both backends

## Non-Goals

- Full deep learning framework (no autograd graph, optimizers, etc.)
- Replacing Burn/PyTorch for standard ML workloads
- Supporting every possible tensor operation (focus on einsum)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User API                                │
│   einsum(tensors, ixs, iy, sizes) → Tensor                     │
│   Tensor::contract_binary(other, ia, ib, iy) → Tensor          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Einsum Engine                              │
│   1. EinCode from (ixs, iy)                                    │
│   2. omeco::optimize_code() → NestedEinsum (contraction tree)  │
│   3. Execute tree recursively via contract_binary()            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   contract_binary()                             │
│   1. Classify indices (batch, left, right, contracted)         │
│   2. Permute tensors (view - zero cost)                        │
│   3. Reshape to matrices (view - zero cost)                    │
│   4. gemm() → calls contiguous() then GEMM kernel              │
│   5. Reshape + permute result (view - zero cost)               │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│      CPU Backend        │     │      CUDA Backend       │
│  Permute: loop copy     │     │  Permute: cuTENSOR      │
│  GEMM: tropical-gemm    │     │  GEMM: tropical-gemm-   │
│        (SIMD)           │     │        cuda             │
└─────────────────────────┘     └─────────────────────────┘
```

## Core Traits

### Algebra Trait (Semiring Abstraction)

Abstracts over different algebraic structures:

```rust
pub trait Algebra: Copy + Clone + Send + Sync + 'static {
    type Scalar: Copy + Send + Sync + bytemuck::Pod;

    /// Additive identity (0 for standard, -∞ for MaxPlus, +∞ for MinPlus)
    fn zero() -> Self;

    /// Multiplicative identity (1 for standard, 0 for tropical)
    fn one() -> Self;

    /// Addition operation (+ for standard, max/min for tropical)
    fn add(self, rhs: Self) -> Self;

    /// Multiplication operation (× for standard, + for tropical)
    fn mul(self, rhs: Self) -> Self;

    /// Addition with argmax tracking (for tropical backpropagation)
    fn add_with_argmax(self, self_idx: u32, rhs: Self, rhs_idx: u32) -> (Self, u32) {
        (self.add(rhs), 0) // Default: no tracking
    }
}
```

**Implementations:**

| Type | ⊕ (add) | ⊗ (mul) | Zero | One | Use Case |
|------|---------|---------|------|-----|----------|
| `Standard<f32>` | + | × | 0.0 | 1.0 | Regular linear algebra |
| `MaxPlus<f32>` | max | + | -∞ | 0.0 | Longest path, Viterbi |
| `MinPlus<f32>` | min | + | +∞ | 0.0 | Shortest path |
| `MaxMul<f32>` | max | × | 0.0 | 1.0 | Max probability |

### Backend Trait (CPU vs CUDA)

```rust
pub trait Backend: Clone + Send + Sync + 'static {
    type Storage<T: Scalar>: Storage<T>;

    fn name() -> &'static str;
    fn synchronize(&self);

    /// Allocate storage
    fn alloc<T: Scalar>(&self, len: usize) -> Self::Storage<T>;

    /// GEMM dispatch - routes to appropriate kernel based on Algebra
    fn gemm<A: Algebra>(
        a: &Self::Storage<A::Scalar>, m: usize, k: usize,
        b: &Self::Storage<A::Scalar>, n: usize,
    ) -> Self::Storage<A::Scalar>;

    /// Make tensor contiguous (permute/copy as needed)
    fn make_contiguous<T: Scalar>(
        src: &Self::Storage<T>,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Self::Storage<T>;
}
```

**Implementations:**

- `Cpu`: Uses `tropical-gemm` for GEMM, loop-based permute
- `Cuda`: Uses `tropical-gemm-cuda` for GEMM, cuTENSOR for permute

### Storage Trait

```rust
pub trait Storage<T>: Clone + Send + Sync {
    fn len(&self) -> usize;
    fn as_slice(&self) -> &[T];      // CPU only
    fn as_mut_slice(&mut self) -> &mut [T];  // CPU only
    fn to_vec(&self) -> Vec<T>;
    fn from_slice(data: &[T], backend: &impl Backend) -> Self;
}
```

## Tensor Type

Stride-based tensor with zero-copy views:

```rust
pub struct Tensor<A: Algebra, B: Backend> {
    /// Raw data storage (reference counted, may be shared)
    storage: Arc<B::Storage<A::Scalar>>,

    /// Shape of this view
    shape: Vec<usize>,

    /// Strides for each dimension (in elements)
    strides: Vec<usize>,

    /// Offset into storage
    offset: usize,

    /// Backend reference
    backend: B,

    /// Phantom for algebra type
    _algebra: PhantomData<A>,
}
```

### Key Methods

```rust
impl<A: Algebra, B: Backend> Tensor<A, B> {
    // Creation
    pub fn from_data(data: &[A::Scalar], shape: &[usize], backend: &B) -> Self;
    pub fn zeros(shape: &[usize], backend: &B) -> Self;
    pub fn ones(shape: &[usize], backend: &B) -> Self;

    // Metadata
    pub fn shape(&self) -> &[usize];
    pub fn strides(&self) -> &[usize];
    pub fn ndim(&self) -> usize;
    pub fn numel(&self) -> usize;
    pub fn is_contiguous(&self) -> bool;

    // Data access
    pub fn to_vec(&self) -> Vec<A::Scalar>;
    pub fn backend(&self) -> &B;

    // View operations (zero cost)
    pub fn permute(&self, axes: &[usize]) -> Self;
    pub fn transpose(&self) -> Self;  // 2D shorthand
    pub fn reshape(&self, shape: &[usize]) -> Self;

    // Force contiguous layout
    pub fn contiguous(&self) -> Self;

    // Core operations
    pub fn gemm(&self, other: &Self) -> Self;
    pub fn gemm_with_argmax(&self, other: &Self) -> (Self, ArgmaxTensor<B>);

    // Binary contraction (reshape-to-GEMM strategy)
    pub fn contract_binary(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
    ) -> Self;

    pub fn contract_binary_with_argmax(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
    ) -> (Self, ArgmaxTensor<B>);
}
```

### View Operations

`permute` and `reshape` are zero-cost operations that only modify metadata:

```rust
pub fn permute(&self, axes: &[usize]) -> Self {
    assert_eq!(axes.len(), self.shape.len());

    let new_shape: Vec<usize> = axes.iter().map(|&i| self.shape[i]).collect();
    let new_strides: Vec<usize> = axes.iter().map(|&i| self.strides[i]).collect();

    Self {
        storage: Arc::clone(&self.storage),  // Share storage
        shape: new_shape,
        strides: new_strides,
        offset: self.offset,
        backend: self.backend.clone(),
        _algebra: PhantomData,
    }
}

pub fn reshape(&self, new_shape: &[usize]) -> Self {
    let old_numel: usize = self.shape.iter().product();
    let new_numel: usize = new_shape.iter().product();
    assert_eq!(old_numel, new_numel, "reshape: element count mismatch");

    if self.is_contiguous() {
        // Fast path: just update metadata
        Self {
            storage: Arc::clone(&self.storage),
            shape: new_shape.to_vec(),
            strides: compute_contiguous_strides(new_shape),
            offset: self.offset,
            backend: self.backend.clone(),
            _algebra: PhantomData,
        }
    } else {
        // Must copy to contiguous first
        self.contiguous().reshape(new_shape)
    }
}
```

## Contract Binary: Reshape-to-GEMM

Any binary tensor contraction can be reduced to matrix multiplication:

```rust
/// Binary contraction via reshape to GEMM
///
/// Example: A[i,j,k] × B[k,l,m] → C[i,j,l,m]  (contracting over k)
///
/// Steps:
/// 1. Classify indices: batch, left-only, right-only, contracted
/// 2. Permute A to [batch, left, contracted] → reshape to [B*L, K]
/// 3. Permute B to [batch, contracted, right] → reshape to [K, B*R]
/// 4. GEMM: [B*L, K] × [K, B*R] → [B*L, B*R]
/// 5. Reshape result to [batch, left, right] → permute to output order
pub fn contract_binary(
    &self,
    other: &Self,
    ia: &[usize],
    ib: &[usize],
    iy: &[usize],
) -> Self {
    // Step 1: Classify indices
    let contracted: Vec<usize> = ia.iter()
        .filter(|i| ib.contains(i) && !iy.contains(i))
        .copied().collect();

    let left_only: Vec<usize> = ia.iter()
        .filter(|i| !contracted.contains(i))
        .copied().collect();

    let right_only: Vec<usize> = ib.iter()
        .filter(|i| !contracted.contains(i))
        .copied().collect();

    let batch: Vec<usize> = ia.iter()
        .filter(|i| ib.contains(i) && iy.contains(i))
        .copied().collect();

    // Step 2-3: Permute and reshape (these are FREE - just views)
    let a_perm = self.permute(&perm_for(ia, &batch, &left_only, &contracted));
    let a_matrix = a_perm.reshape(&[batch_size * left_size, contract_size]);

    let b_perm = other.permute(&perm_for(ib, &batch, &contracted, &right_only));
    let b_matrix = b_perm.reshape(&[batch_size * contract_size, right_size]);

    // Step 4: GEMM (this is where actual work happens)
    let c_matrix = a_matrix.gemm(&b_matrix);

    // Step 5: Reshape and permute to output (FREE - just views)
    let c_shaped = c_matrix.reshape(&output_shape);
    c_shaped.permute(&output_perm)
}
```

### Visual Example

```
Contraction: A[i,j,k] × B[j,k,l] → C[i,l]

ia = [i, j, k]    shape = [2, 3, 4]
ib = [j, k, l]    shape = [3, 4, 5]
iy = [i, l]

Index classification:
  contracted = [j, k]     (in both inputs, not in output)
  left_only  = [i]        (only in A)
  right_only = [l]        (only in B)
  batch      = []         (in both inputs AND output)

Reshape:
  A[2,3,4] → permute[i,j,k] → reshape[2, 12]   = [L, K]
  B[3,4,5] → permute[j,k,l] → reshape[12, 5]   = [K, R]

GEMM:
  [2, 12] × [12, 5] → [2, 5]

Output:
  [2, 5] = C[i, l] ✓
```

## Einsum Engine with omeco

Integration with omeco for contraction order optimization:

```rust
use omeco::{EinCode, NestedEinsum, GreedyMethod, TreeSA, optimize_code, Label};

pub struct Einsum<L: Label> {
    /// Einsum specification
    pub code: EinCode<L>,

    /// Dimension sizes for each index
    pub size_dict: HashMap<L, usize>,

    /// Optimized contraction tree (after optimization)
    optimized: Option<NestedEinsum<L>>,
}

impl<L: Label> Einsum<L> {
    /// Create from einsum specification
    pub fn new(ixs: Vec<Vec<L>>, iy: Vec<L>, size_dict: HashMap<L, usize>) -> Self {
        Self {
            code: EinCode::new(ixs, iy),
            size_dict,
            optimized: None,
        }
    }

    /// Optimize with greedy algorithm (fast, O(n²))
    pub fn optimize_greedy(&mut self) -> &mut Self {
        let optimizer = GreedyMethod::new(0.0, 0.0);
        self.optimized = optimize_code(&self.code, &self.size_dict, &optimizer);
        self
    }

    /// Optimize with simulated annealing (better quality)
    pub fn optimize_treesa(&mut self) -> &mut Self {
        let optimizer = TreeSA::default();
        self.optimized = optimize_code(&self.code, &self.size_dict, &optimizer);
        self
    }

    /// Execute the contraction
    pub fn execute<A, B, T>(&self, tensors: &[T]) -> T
    where
        A: Algebra,
        B: Backend,
        T: TensorOps<A, B>,
        L: Into<usize> + Copy,
    {
        match &self.optimized {
            Some(tree) => self.execute_tree(tree, tensors),
            None => self.execute_pairwise(tensors),
        }
    }

    /// Execute optimized contraction tree (recursive)
    fn execute_tree<A, B, T>(&self, tree: &NestedEinsum<L>, tensors: &[T]) -> T
    where
        A: Algebra,
        B: Backend,
        T: TensorOps<A, B>,
        L: Into<usize> + Copy,
    {
        match tree {
            NestedEinsum::Leaf { tensor_index } => {
                tensors[*tensor_index].clone()
            }
            NestedEinsum::Node { args, eins } => {
                assert_eq!(args.len(), 2, "omeco produces binary trees");

                let left = self.execute_tree(&args[0], tensors);
                let right = self.execute_tree(&args[1], tensors);

                let ia: Vec<usize> = eins.ixs[0].iter().map(|l| (*l).into()).collect();
                let ib: Vec<usize> = eins.ixs[1].iter().map(|l| (*l).into()).collect();
                let iy: Vec<usize> = eins.iy.iter().map(|l| (*l).into()).collect();

                left.contract_binary(&right, &ia, &ib, &iy)
            }
        }
    }
}
```

## Kernel Dispatch

| Operation | CPU | CUDA | Tropical-Aware? |
|-----------|-----|------|-----------------|
| Permute | Loop-based copy | cuTENSOR | No (data movement) |
| Reshape | Metadata only | Metadata only | N/A |
| Contiguous | Loop-based copy | cuTENSOR | No (data movement) |
| **GEMM** | tropical-gemm (SIMD) | tropical-gemm-cuda | **Yes** |
| **Reduction** | Custom kernel | Custom kernel | **Yes** |

cuTENSOR is safe for tropical data because permutation is pure data movement - no arithmetic operations.

## Crate Structure

```
tropical-einsum/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API exports
│   ├── algebra.rs          # Algebra trait + Standard, MaxPlus, MinPlus, MaxMul
│   ├── backend/
│   │   ├── mod.rs          # Backend trait
│   │   ├── cpu.rs          # CPU backend implementation
│   │   └── cuda.rs         # CUDA backend (optional feature)
│   ├── storage/
│   │   ├── mod.rs          # Storage trait
│   │   ├── cpu.rs          # Vec-based storage
│   │   └── cuda.rs         # CudaSlice-based storage
│   ├── tensor.rs           # Stride-based Tensor type
│   ├── einsum.rs           # Einsum engine + omeco integration
│   ├── contract.rs         # contract_binary implementation
│   └── utils.rs            # Stride computation, index helpers
├── tests/
│   ├── tensor_tests.rs
│   ├── contract_tests.rs
│   └── einsum_tests.rs
└── examples/
    ├── basic_einsum.rs
    └── tensor_network.rs
```

## Dependencies

```toml
[dependencies]
tropical-gemm = { path = "../tropical-gemm" }
omeco = { path = "../../omeco" }  # Or git/crates.io
num-traits = "0.2"
bytemuck = "1.14"

[dependencies.tropical-gemm-cuda]
path = "../tropical-gemm-cuda"
optional = true

[dependencies.cudarc]
version = "0.12"
optional = true

[features]
default = []
cuda = ["tropical-gemm-cuda", "cudarc"]
```

## Example Usage

```rust
use tropical_einsum::{Tensor, Einsum, MaxPlus, MinPlus, Standard, Cpu};

fn main() {
    // Create tropical tensors
    let a = Tensor::<MaxPlus<f32>, Cpu>::from_data(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        &Cpu,
    );
    let b = Tensor::<MaxPlus<f32>, Cpu>::from_data(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 2],
        &Cpu,
    );

    // Direct matrix multiplication
    let c = a.gemm(&b);  // MaxPlus: C[i,j] = max_k(A[i,k] + B[k,j])

    // Einsum with optimization
    let a = Tensor::<MaxPlus<f32>, Cpu>::from_data(&data_a, &[10, 20, 30], &Cpu);
    let b = Tensor::<MaxPlus<f32>, Cpu>::from_data(&data_b, &[20, 30, 40], &Cpu);
    let c = Tensor::<MaxPlus<f32>, Cpu>::from_data(&data_c, &[40, 50], &Cpu);

    // A[i,j,k] × B[j,k,l] × C[l,m] → D[i,m]
    let sizes: HashMap<usize, usize> = [
        (0, 10), (1, 20), (2, 30), (3, 40), (4, 50)
    ].into();

    let mut ein = Einsum::new(
        vec![vec![0,1,2], vec![1,2,3], vec![3,4]],
        vec![0, 4],
        sizes,
    );

    ein.optimize_greedy();  // or .optimize_treesa() for better quality
    let result = ein.execute(&[a, b, c]);

    // Same code works with standard arithmetic
    let x = Tensor::<Standard<f32>, Cpu>::from_data(&data, &[10, 20], &Cpu);
    let y = Tensor::<Standard<f32>, Cpu>::from_data(&data, &[20, 30], &Cpu);
    let z = x.gemm(&y);  // Standard: C[i,j] = Σ_k A[i,k] × B[k,j]
}
```

## Implementation Phases

### Phase 1: Core Tensor Type
- [ ] `Algebra` trait with `Standard`, `MaxPlus`, `MinPlus`, `MaxMul`
- [ ] `Backend` trait with `Cpu` implementation
- [ ] `Storage` trait with `Vec<T>` implementation
- [ ] `Tensor` type with `permute`, `reshape`, `contiguous`
- [ ] Loop-based `contiguous()` for CPU
- [ ] GEMM via existing `tropical-gemm`

### Phase 2: Binary Contraction
- [ ] Index classification (batch, left, right, contracted)
- [ ] `contract_binary` with reshape-to-GEMM strategy
- [ ] Argmax tracking for tropical backpropagation
- [ ] Unit tests for various contraction patterns

### Phase 3: Einsum Engine
- [ ] `Einsum` struct with `EinCode`
- [ ] Integration with `omeco::optimize_code`
- [ ] Tree execution via recursive `contract_binary`
- [ ] Pairwise fallback (no optimization)

### Phase 4: CUDA Backend
- [ ] `Cuda` backend implementation
- [ ] `CudaSlice` storage wrapper
- [ ] cuTENSOR integration for permute/contiguous
- [ ] GEMM via existing `tropical-gemm-cuda`

### Phase 5: Python Bindings
- [ ] PyO3 bindings for `Tensor`
- [ ] NumPy array interop
- [ ] PyTorch tensor interop
- [ ] Python `einsum()` function

## Open Questions

1. **Batch dimensions in GEMM**: Current `tropical-gemm` has `tropical_matmul_batched`. Should batch dims in `contract_binary` use this, or handle via loops?

2. **Standard arithmetic GEMM**: For `Standard<f32>`, should we dispatch to BLAS/cuBLAS instead of tropical-gemm? (tropical-gemm with `(+,×)` semiring would work but BLAS is more optimized)

3. **Memory management**: Should `Tensor` own its backend reference, or should backend be passed to operations? Current design has `Tensor` own it.

4. **Slicing support**: omeco supports sliced einsum for memory-constrained execution. Should this be in scope for v1?

## References

- [Issue #21: Proposal: Extend to general tensor einsum operations](https://github.com/TensorBFS/tropical-gemm/issues/21)
- [omeco: One More Einsum Contraction Order](https://github.com/GiggleLiu/omeco)
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) - Julia reference implementation
- [opt_einsum](https://github.com/dgasmith/opt_einsum) - Python einsum optimization
- [cuTENSOR](https://developer.nvidia.com/cutensor) - NVIDIA tensor operations library
