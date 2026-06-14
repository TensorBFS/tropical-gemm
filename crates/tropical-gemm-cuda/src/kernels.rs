//! CUDA kernel trait and implementations.
//!
//! ## Asynchronous launch contract
//!
//! Kernel launches in this crate are enqueued on the context's single CUDA
//! stream and return *without* a device synchronize. Correctness relies on
//! stream ordering: every operand upload, kernel, and device-to-device copy
//! runs on that one stream (see `CudaContext::stream`), so each launch
//! observes its predecessors' results, and any host read goes through
//! `GpuMatrix::to_host`, which synchronizes. A per-launch
//! `stream.synchronize()` was previously issued after every kernel; on
//! contractions built from many small matrices that blocking host↔device
//! round-trip dominated the wall (one full sync per node) while the GPU sat
//! idle between tiny kernels, so it has been removed. Callers that need a
//! barrier (e.g. before timing) can call `ctx.stream().synchronize()`.

use crate::context::CudaContext;
use crate::error::CudaError;
use crate::error::Result;
use crate::memory::{
    ExternalGpuMatrix, ExternalGpuTensor3, GpuMatrix, GpuMatrixWithArgmax, GpuTensor3WithArgmax,
};
use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};
use tropical_gemm::types::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring};

/// Maximum extent of `gridDim.z` on all CUDA compute capabilities (the batch
/// dimension of the strided-batched kernels maps to `blockIdx.z`). A single
/// launch with more than this many batch elements fails at launch time with
/// `CUDA_ERROR_INVALID_VALUE`, so batched launches are split into chunks of at
/// most this size (see `launch_kernel_batched_impl`).
const MAX_GRID_DIM_Z: usize = 65535;

/// Convert a per-batch element stride to the `i32` the kernels take, failing
/// instead of truncating. The chunked launches derive operand base offsets from
/// the `usize` stride while the kernel reads the stride as `i32`; an `as i32`
/// cast that wrapped would desynchronise the two and silently corrupt
/// addressing. The kernel signature is `i32`, so a stride past `i32::MAX` is
/// unrepresentable on the device regardless — reject it at the boundary.
fn stride_to_i32(stride: usize, what: &str) -> Result<i32> {
    i32::try_from(stride).map_err(|_| {
        CudaError::DimensionMismatch(format!(
            "batched GEMM {what} stride {stride} exceeds i32::MAX; \
             matrix too large for the strided-batched kernel"
        ))
    })
}

/// Trait for types that can be computed on GPU.
pub trait CudaKernel: TropicalSemiring
where
    Self::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Kernel function name.
    const KERNEL_NAME: &'static str;

    /// Forward batched kernel function name (one launch, `blockIdx.z` = batch).
    const BATCHED_KERNEL_NAME: &'static str;

    /// Execute the tropical GEMM kernel.
    ///
    /// Computes C = A ⊗ B where ⊗ is tropical matrix multiplication.
    fn launch_gemm(
        ctx: &CudaContext,
        a: &GpuMatrix<Self::Scalar>,
        b: &GpuMatrix<Self::Scalar>,
        c: &mut GpuMatrix<Self::Scalar>,
    ) -> Result<()>;

    /// Execute `batch` independent tropical GEMMs in a single launch.
    ///
    /// Computes `C[i] = A[i] ⊗ B[i]` for `i in 0..batch` over contiguous,
    /// already-device-resident operands: `a` is `batch × m × k`, `b` is
    /// `batch × k × n`, `c` is `batch × m × n`, each column-major per matrix
    /// with contiguous per-batch stride (`m*k`, `k*n`, `m*n`). This is the
    /// strided-batched replacement for a host-side per-slice loop: no per-slice
    /// `clone_dtod`, no per-slice allocation, no reassembly copy. The kernel
    /// fully writes every element of `c`, so `c` may be uninitialized on entry.
    fn launch_gemm_batched(
        ctx: &CudaContext,
        a: &CudaSlice<Self::Scalar>,
        b: &CudaSlice<Self::Scalar>,
        c: &mut CudaSlice<Self::Scalar>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()>;
}

/// Helper function to launch a CUDA kernel with given grid/block dimensions.
fn launch_kernel_impl<T: DeviceRepr + ValidAsZeroBits + Default + Clone>(
    ctx: &CudaContext,
    kernel_name: &'static str,
    a: &GpuMatrix<T>,
    b: &GpuMatrix<T>,
    c: &mut GpuMatrix<T>,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
) -> Result<()> {
    let m = a.rows();
    let k = a.cols();
    let n = b.cols();

    let kernel = ctx.get_kernel(kernel_name)?;
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    // Bind scalar kernel args to locals so they outlive the launch builder.
    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let k_i32 = k as i32;

    let stream = ctx.stream();
    let mut builder = stream.launch_builder(&kernel);
    builder
        .arg(a.as_slice())
        .arg(b.as_slice())
        .arg(c.as_slice_mut())
        .arg(&m_i32)
        .arg(&n_i32)
        .arg(&k_i32);
    unsafe {
        builder.launch(cfg)?;
    }

    // Async: no per-launch device sync (stream-ordered; host reads sync in
    // `to_host`). See the module-level "Asynchronous launch contract".
    Ok(())
}

/// Helper to launch a forward batched kernel over contiguous device buffers.
///
/// `grid.z` selects the batch element; the per-batch strides are the contiguous
/// extents `m*k` / `k*n` / `m*n`. Operands are passed as borrowed `CudaSlice`s
/// — no copy, no allocation. Follows the same asynchronous launch contract as
/// [`launch_kernel_impl`].
#[allow(clippy::too_many_arguments)]
fn launch_kernel_batched_impl<T: DeviceRepr + ValidAsZeroBits + Default + Clone>(
    ctx: &CudaContext,
    kernel_name: &'static str,
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    c: &mut CudaSlice<T>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    grid_xy: u32,
    block: (u32, u32, u32),
) -> Result<()> {
    // The raw-slice API loses GpuMatrix's dimension invariant, so validate the
    // contiguous batched extents before handing pointers to the kernel.
    let want = |dim: usize, len: usize, what: &str| -> Result<()> {
        if len != dim {
            return Err(CudaError::DimensionMismatch(format!(
                "batched GEMM {what}: expected {dim} elements (batch={batch}, m={m}, k={k}, n={n}), got {len}"
            )));
        }
        Ok(())
    };
    want(batch * m * k, a.len(), "operand A")?;
    want(batch * k * n, b.len(), "operand B")?;
    want(batch * m * n, c.len(), "output C")?;

    let kernel = ctx.get_kernel(kernel_name)?;

    // Bind scalar kernel args to locals so they outlive the launch builder.
    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let k_i32 = k as i32;
    // Per-batch element extents, used both to offset each chunk's operand base
    // (as `usize`) and as the kernel's `i32` stride. Derive the `i32` form
    // fallibly so a stride past `i32::MAX` can't truncate out of sync with the
    // `usize` offsets below (a.len()/b.len()/c.len() already fit in `usize`).
    let (sa, sb, sc) = (m * k, k * n, m * n);
    let stride_a = stride_to_i32(sa, "operand A")?;
    let stride_b = stride_to_i32(sb, "operand B")?;
    let stride_c = stride_to_i32(sc, "output C")?;

    let stream = ctx.stream();
    // The batch maps to `blockIdx.z`, which CUDA caps at `MAX_GRID_DIM_Z`. Launch
    // the batch in chunks of at most that size, offsetting every operand's base
    // by the chunk start so `blockIdx.z ∈ [0, chunk)` indexes the correct slice.
    // `grid_xy` from the caller is the x tiling; y is 1 and z is the chunk size.
    let mut start = 0usize;
    while start < batch {
        let chunk = (batch - start).min(MAX_GRID_DIM_Z);
        let a_view = a.slice(start * sa..(start + chunk) * sa);
        let b_view = b.slice(start * sb..(start + chunk) * sb);
        let mut c_view = c.slice_mut(start * sc..(start + chunk) * sc);
        let cfg = LaunchConfig {
            grid_dim: (grid_xy, 1, chunk as u32),
            block_dim: block,
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernel);
        builder
            .arg(&a_view)
            .arg(&b_view)
            .arg(&mut c_view)
            .arg(&m_i32)
            .arg(&n_i32)
            .arg(&k_i32)
            .arg(&stride_a)
            .arg(&stride_b)
            .arg(&stride_c);
        unsafe {
            builder.launch(cfg)?;
        }
        start += chunk;
    }

    // Async: no per-launch device sync (stream-ordered; host reads sync in
    // `to_host`). See the module-level "Asynchronous launch contract".
    Ok(())
}

/// Macro to implement CudaKernel for f32 types.
macro_rules! impl_cuda_kernel_f32 {
    ($($semiring:ty => $kernel_name:literal, $batched_name:literal);* $(;)?) => {
        $(
            impl CudaKernel for $semiring {
                const KERNEL_NAME: &'static str = $kernel_name;
                const BATCHED_KERNEL_NAME: &'static str = $batched_name;

                fn launch_gemm(
                    ctx: &CudaContext,
                    a: &GpuMatrix<f32>,
                    b: &GpuMatrix<f32>,
                    c: &mut GpuMatrix<f32>,
                ) -> Result<()> {
                    let grid = CudaContext::grid_dims_f32(a.rows(), b.cols());
                    let block = CudaContext::block_dims_f32();
                    launch_kernel_impl(ctx, Self::KERNEL_NAME, a, b, c, grid, block)
                }

                fn launch_gemm_batched(
                    ctx: &CudaContext,
                    a: &CudaSlice<f32>,
                    b: &CudaSlice<f32>,
                    c: &mut CudaSlice<f32>,
                    batch: usize,
                    m: usize,
                    k: usize,
                    n: usize,
                ) -> Result<()> {
                    let grid_xy = CudaContext::grid_dims_f32(m, n).0;
                    let block = CudaContext::block_dims_f32();
                    launch_kernel_batched_impl(
                        ctx, Self::BATCHED_KERNEL_NAME, a, b, c, batch, m, k, n, grid_xy, block,
                    )
                }
            }
        )*
    };
}

/// Macro to implement CudaKernel for f64 types.
macro_rules! impl_cuda_kernel_f64 {
    ($($semiring:ty => $kernel_name:literal, $batched_name:literal);* $(;)?) => {
        $(
            impl CudaKernel for $semiring {
                const KERNEL_NAME: &'static str = $kernel_name;
                const BATCHED_KERNEL_NAME: &'static str = $batched_name;

                fn launch_gemm(
                    ctx: &CudaContext,
                    a: &GpuMatrix<f64>,
                    b: &GpuMatrix<f64>,
                    c: &mut GpuMatrix<f64>,
                ) -> Result<()> {
                    let grid = CudaContext::grid_dims_f64(a.rows(), b.cols());
                    let block = CudaContext::block_dims_f64();
                    launch_kernel_impl(ctx, Self::KERNEL_NAME, a, b, c, grid, block)
                }

                fn launch_gemm_batched(
                    ctx: &CudaContext,
                    a: &CudaSlice<f64>,
                    b: &CudaSlice<f64>,
                    c: &mut CudaSlice<f64>,
                    batch: usize,
                    m: usize,
                    k: usize,
                    n: usize,
                ) -> Result<()> {
                    let grid_xy = CudaContext::grid_dims_f64(m, n).0;
                    let block = CudaContext::block_dims_f64();
                    launch_kernel_batched_impl(
                        ctx, Self::BATCHED_KERNEL_NAME, a, b, c, batch, m, k, n, grid_xy, block,
                    )
                }
            }
        )*
    };
}

impl_cuda_kernel_f32! {
    TropicalMaxPlus<f32> => "tropical_maxplus_f32_nn", "tropical_maxplus_f32_nn_batched";
    TropicalMinPlus<f32> => "tropical_minplus_f32_nn", "tropical_minplus_f32_nn_batched";
    TropicalMaxMul<f32> => "tropical_maxmul_f32_nn", "tropical_maxmul_f32_nn_batched";
}

impl_cuda_kernel_f64! {
    TropicalMaxPlus<f64> => "tropical_maxplus_f64_nn", "tropical_maxplus_f64_nn_batched";
    TropicalMinPlus<f64> => "tropical_minplus_f64_nn", "tropical_minplus_f64_nn_batched";
    TropicalMaxMul<f64> => "tropical_maxmul_f64_nn", "tropical_maxmul_f64_nn_batched";
}

/// Macro to implement CudaKernel for i32 types.
/// Uses same block sizes as f32 (64x32x64) since int is 4 bytes.
macro_rules! impl_cuda_kernel_i32 {
    ($($semiring:ty => $kernel_name:literal, $batched_name:literal);* $(;)?) => {
        $(
            impl CudaKernel for $semiring {
                const KERNEL_NAME: &'static str = $kernel_name;
                const BATCHED_KERNEL_NAME: &'static str = $batched_name;

                fn launch_gemm(
                    ctx: &CudaContext,
                    a: &GpuMatrix<i32>,
                    b: &GpuMatrix<i32>,
                    c: &mut GpuMatrix<i32>,
                ) -> Result<()> {
                    let grid = CudaContext::grid_dims_f32(a.rows(), b.cols());
                    let block = CudaContext::block_dims_f32();
                    launch_kernel_impl(ctx, Self::KERNEL_NAME, a, b, c, grid, block)
                }

                fn launch_gemm_batched(
                    ctx: &CudaContext,
                    a: &CudaSlice<i32>,
                    b: &CudaSlice<i32>,
                    c: &mut CudaSlice<i32>,
                    batch: usize,
                    m: usize,
                    k: usize,
                    n: usize,
                ) -> Result<()> {
                    let grid_xy = CudaContext::grid_dims_f32(m, n).0;
                    let block = CudaContext::block_dims_f32();
                    launch_kernel_batched_impl(
                        ctx, Self::BATCHED_KERNEL_NAME, a, b, c, batch, m, k, n, grid_xy, block,
                    )
                }
            }
        )*
    };
}

/// Macro to implement CudaKernel for i64 types.
/// Uses same block sizes as f64 (32x16x32) since long long is 8 bytes.
macro_rules! impl_cuda_kernel_i64 {
    ($($semiring:ty => $kernel_name:literal, $batched_name:literal);* $(;)?) => {
        $(
            impl CudaKernel for $semiring {
                const KERNEL_NAME: &'static str = $kernel_name;
                const BATCHED_KERNEL_NAME: &'static str = $batched_name;

                fn launch_gemm(
                    ctx: &CudaContext,
                    a: &GpuMatrix<i64>,
                    b: &GpuMatrix<i64>,
                    c: &mut GpuMatrix<i64>,
                ) -> Result<()> {
                    let grid = CudaContext::grid_dims_f64(a.rows(), b.cols());
                    let block = CudaContext::block_dims_f64();
                    launch_kernel_impl(ctx, Self::KERNEL_NAME, a, b, c, grid, block)
                }

                fn launch_gemm_batched(
                    ctx: &CudaContext,
                    a: &CudaSlice<i64>,
                    b: &CudaSlice<i64>,
                    c: &mut CudaSlice<i64>,
                    batch: usize,
                    m: usize,
                    k: usize,
                    n: usize,
                ) -> Result<()> {
                    let grid_xy = CudaContext::grid_dims_f64(m, n).0;
                    let block = CudaContext::block_dims_f64();
                    launch_kernel_batched_impl(
                        ctx, Self::BATCHED_KERNEL_NAME, a, b, c, batch, m, k, n, grid_xy, block,
                    )
                }
            }
        )*
    };
}

impl_cuda_kernel_i32! {
    TropicalMaxPlus<i32> => "tropical_maxplus_i32_nn", "tropical_maxplus_i32_nn_batched";
    TropicalMinPlus<i32> => "tropical_minplus_i32_nn", "tropical_minplus_i32_nn_batched";
    TropicalMaxMul<i32> => "tropical_maxmul_i32_nn", "tropical_maxmul_i32_nn_batched";
}

impl_cuda_kernel_i64! {
    TropicalMaxPlus<i64> => "tropical_maxplus_i64_nn", "tropical_maxplus_i64_nn_batched";
    TropicalMinPlus<i64> => "tropical_minplus_i64_nn", "tropical_minplus_i64_nn_batched";
    TropicalMaxMul<i64> => "tropical_maxmul_i64_nn", "tropical_maxmul_i64_nn_batched";
}

// ============================================================================
// CudaKernelWithArgmax - for path reconstruction (integers don't have gradients)
// ============================================================================

/// Trait for tropical GEMM with argmax tracking (for backward propagation).
///
/// This computes both C[i,j] and the k-index that produced each C[i,j],
/// which is needed for gradient computation in tropical neural networks.
pub trait CudaKernelWithArgmax: TropicalSemiring
where
    Self::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Kernel function name for the argmax variant.
    const ARGMAX_KERNEL_NAME: &'static str;

    /// Execute the tropical GEMM kernel with argmax tracking.
    ///
    /// Computes C = A ⊗ B and also records argmax[i,j] = k such that
    /// C[i,j] = A[i,k] ⊗ B[k,j] was the winning value.
    fn launch_gemm_with_argmax(
        ctx: &CudaContext,
        a: &GpuMatrix<Self::Scalar>,
        b: &GpuMatrix<Self::Scalar>,
        c: &mut GpuMatrixWithArgmax<Self::Scalar>,
    ) -> Result<()>;
}

/// Helper function to launch an argmax CUDA kernel.
fn launch_kernel_with_argmax_impl<T: DeviceRepr + ValidAsZeroBits + Default + Clone>(
    ctx: &CudaContext,
    kernel_name: &'static str,
    a: &GpuMatrix<T>,
    b: &GpuMatrix<T>,
    c: &mut GpuMatrixWithArgmax<T>,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
) -> Result<()> {
    let m = a.rows();
    let k = a.cols();
    let n = b.cols();

    let kernel = ctx.get_kernel(kernel_name)?;
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    // Bind scalar kernel args to locals so they outlive the launch builder.
    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let k_i32 = k as i32;

    let stream = ctx.stream();
    let mut builder = stream.launch_builder(&kernel);
    builder
        .arg(a.as_slice())
        .arg(b.as_slice())
        .arg(c.matrix.as_slice_mut())
        .arg(c.argmax.as_slice_mut())
        .arg(&m_i32)
        .arg(&n_i32)
        .arg(&k_i32);
    unsafe {
        builder.launch(cfg)?;
    }

    // Async: no per-launch device sync (stream-ordered; host reads sync in
    // `to_host`). See the module-level "Asynchronous launch contract".
    Ok(())
}

/// Macro to implement CudaKernelWithArgmax for f32 types.
macro_rules! impl_cuda_kernel_with_argmax_f32 {
    ($($semiring:ty => $kernel_name:literal),* $(,)?) => {
        $(
            impl CudaKernelWithArgmax for $semiring {
                const ARGMAX_KERNEL_NAME: &'static str = $kernel_name;

                fn launch_gemm_with_argmax(
                    ctx: &CudaContext,
                    a: &GpuMatrix<f32>,
                    b: &GpuMatrix<f32>,
                    c: &mut GpuMatrixWithArgmax<f32>,
                ) -> Result<()> {
                    let grid = CudaContext::grid_dims_f32(a.rows(), b.cols());
                    let block = CudaContext::block_dims_f32();
                    launch_kernel_with_argmax_impl(ctx, Self::ARGMAX_KERNEL_NAME, a, b, c, grid, block)
                }
            }
        )*
    };
}

/// Macro to implement CudaKernelWithArgmax for f64 types.
macro_rules! impl_cuda_kernel_with_argmax_f64 {
    ($($semiring:ty => $kernel_name:literal),* $(,)?) => {
        $(
            impl CudaKernelWithArgmax for $semiring {
                const ARGMAX_KERNEL_NAME: &'static str = $kernel_name;

                fn launch_gemm_with_argmax(
                    ctx: &CudaContext,
                    a: &GpuMatrix<f64>,
                    b: &GpuMatrix<f64>,
                    c: &mut GpuMatrixWithArgmax<f64>,
                ) -> Result<()> {
                    let grid = CudaContext::grid_dims_f64(a.rows(), b.cols());
                    let block = CudaContext::block_dims_f64();
                    launch_kernel_with_argmax_impl(ctx, Self::ARGMAX_KERNEL_NAME, a, b, c, grid, block)
                }
            }
        )*
    };
}

impl_cuda_kernel_with_argmax_f32! {
    TropicalMaxPlus<f32> => "tropical_maxplus_f32_nn_with_argmax",
    TropicalMinPlus<f32> => "tropical_minplus_f32_nn_with_argmax",
    TropicalMaxMul<f32> => "tropical_maxmul_f32_nn_with_argmax",
}

impl_cuda_kernel_with_argmax_f64! {
    TropicalMaxPlus<f64> => "tropical_maxplus_f64_nn_with_argmax",
    TropicalMinPlus<f64> => "tropical_minplus_f64_nn_with_argmax",
    TropicalMaxMul<f64> => "tropical_maxmul_f64_nn_with_argmax",
}

/// Macro to implement CudaKernelWithArgmax for i32 types.
macro_rules! impl_cuda_kernel_with_argmax_i32 {
    ($($semiring:ty => $kernel_name:literal),* $(,)?) => {
        $(
            impl CudaKernelWithArgmax for $semiring {
                const ARGMAX_KERNEL_NAME: &'static str = $kernel_name;

                fn launch_gemm_with_argmax(
                    ctx: &CudaContext,
                    a: &GpuMatrix<i32>,
                    b: &GpuMatrix<i32>,
                    c: &mut GpuMatrixWithArgmax<i32>,
                ) -> Result<()> {
                    let grid = CudaContext::grid_dims_f32(a.rows(), b.cols());
                    let block = CudaContext::block_dims_f32();
                    launch_kernel_with_argmax_impl(ctx, Self::ARGMAX_KERNEL_NAME, a, b, c, grid, block)
                }
            }
        )*
    };
}

/// Macro to implement CudaKernelWithArgmax for i64 types.
macro_rules! impl_cuda_kernel_with_argmax_i64 {
    ($($semiring:ty => $kernel_name:literal),* $(,)?) => {
        $(
            impl CudaKernelWithArgmax for $semiring {
                const ARGMAX_KERNEL_NAME: &'static str = $kernel_name;

                fn launch_gemm_with_argmax(
                    ctx: &CudaContext,
                    a: &GpuMatrix<i64>,
                    b: &GpuMatrix<i64>,
                    c: &mut GpuMatrixWithArgmax<i64>,
                ) -> Result<()> {
                    let grid = CudaContext::grid_dims_f64(a.rows(), b.cols());
                    let block = CudaContext::block_dims_f64();
                    launch_kernel_with_argmax_impl(ctx, Self::ARGMAX_KERNEL_NAME, a, b, c, grid, block)
                }
            }
        )*
    };
}

impl_cuda_kernel_with_argmax_i32! {
    TropicalMaxPlus<i32> => "tropical_maxplus_i32_nn_with_argmax",
    TropicalMinPlus<i32> => "tropical_minplus_i32_nn_with_argmax",
    TropicalMaxMul<i32> => "tropical_maxmul_i32_nn_with_argmax",
}

impl_cuda_kernel_with_argmax_i64! {
    TropicalMaxPlus<i64> => "tropical_maxplus_i64_nn_with_argmax",
    TropicalMinPlus<i64> => "tropical_minplus_i64_nn_with_argmax",
    TropicalMaxMul<i64> => "tropical_maxmul_i64_nn_with_argmax",
}

// ============================================================================
// External Pointer Kernel Launch (for DLPack zero-copy)
// ============================================================================

/// Launch a tropical GEMM kernel using raw external pointers (e.g., from DLPack).
///
/// This function enables zero-copy kernel execution with PyTorch GPU tensors.
///
/// # Row-Major to Column-Major Trick
///
/// PyTorch uses row-major (C-order), while our CUDA kernels use column-major.
/// Instead of copying/transposing, we use the BLAS trick:
///
/// For `C = A ⊗ B` where `C[i,j] = max_k(A[i,k] + B[k,j])`:
/// - Row-major A (M×K) viewed as column-major = A^T (K×M)
/// - Row-major B (K×N) viewed as column-major = B^T (N×K)
/// - Compute `C^T = B^T ⊗ A^T` using existing column-major kernels
/// - Result C^T column-major (N×M) = C row-major (M×N)
///
/// **Implementation: We swap A↔B and M↔N in the kernel call, no kernel changes needed.**
///
/// # Safety
///
/// - The input pointers must point to valid GPU memory with the specified dimensions
/// - The memory must remain valid for the duration of the kernel execution
/// - The pointers must be properly aligned for the element type
pub unsafe fn launch_gemm_external_with_argmax_f32(
    ctx: &CudaContext,
    kernel_name: &'static str,
    a: &ExternalGpuMatrix<f32>,
    b: &ExternalGpuMatrix<f32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<GpuMatrixWithArgmax<f32>> {
    // Apply row-major → column-major trick: swap inputs and swap M↔N
    // Original: C[i,j] = A[i,k] ⊗ B[k,j]  with A(M,K), B(K,N), C(M,N)
    // Swapped:  C^T = B^T ⊗ A^T  which gives us C in row-major

    // Allocate output: kernel computes C^T (N×M col-major) = C (M×N row-major)
    // But we allocate as (M, N) in col-major and the kernel fills it correctly
    // when we swap the order and dimensions
    let mut c = GpuMatrixWithArgmax::<f32>::alloc(ctx, m, n)?;

    let grid = CudaContext::grid_dims_f32(n, m); // Swapped: (n, m) instead of (m, n)
    let block = CudaContext::block_dims_f32();

    let kernel = ctx.get_kernel(kernel_name)?;
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    // Bind raw external pointers and scalar args to locals so they outlive the
    // launch builder. The pointers are passed as `u64` (== `sys::CUdeviceptr`),
    // which is `DeviceRepr`, so the kernel receives them as device pointers.
    let b_ptr: u64 = b.device_ptr(); // B becomes "A" in kernel
    let a_ptr: u64 = a.device_ptr(); // A becomes "B" in kernel
    let n_i32 = n as i32; // Swapped: N becomes "M"
    let m_i32 = m as i32; // Swapped: M becomes "N"
    let k_i32 = k as i32;

    // Swap order: pass B first, then A, and swap M↔N
    // Kernel signature: (A_ptr, B_ptr, C_ptr, argmax_ptr, M, N, K)
    // We pass:          (B_ptr, A_ptr, C_ptr, argmax_ptr, N, M, K)
    let stream = ctx.stream();
    let mut builder = stream.launch_builder(&kernel);
    builder
        .arg(&b_ptr)
        .arg(&a_ptr)
        .arg(c.matrix.as_slice_mut())
        .arg(c.argmax.as_slice_mut())
        .arg(&n_i32)
        .arg(&m_i32)
        .arg(&k_i32);
    builder.launch(cfg)?;

    // Async: no per-launch device sync (stream-ordered; host reads sync in
    // `to_host`). See the module-level "Asynchronous launch contract".
    Ok(c)
}

/// Launch a tropical GEMM kernel (without argmax) using raw external pointers.
pub unsafe fn launch_gemm_external_f32(
    ctx: &CudaContext,
    kernel_name: &'static str,
    a: &ExternalGpuMatrix<f32>,
    b: &ExternalGpuMatrix<f32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<GpuMatrix<f32>> {
    let mut c = GpuMatrix::<f32>::alloc(ctx, m, n)?;

    let grid = CudaContext::grid_dims_f32(n, m); // Swapped
    let block = CudaContext::block_dims_f32();

    let kernel = ctx.get_kernel(kernel_name)?;
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    // Bind raw external pointers and scalar args to locals so they outlive the
    // launch builder.
    let b_ptr: u64 = b.device_ptr();
    let a_ptr: u64 = a.device_ptr();
    let n_i32 = n as i32;
    let m_i32 = m as i32;
    let k_i32 = k as i32;

    // Swap order and dimensions
    let stream = ctx.stream();
    let mut builder = stream.launch_builder(&kernel);
    builder
        .arg(&b_ptr)
        .arg(&a_ptr)
        .arg(c.as_slice_mut())
        .arg(&n_i32)
        .arg(&m_i32)
        .arg(&k_i32);
    builder.launch(cfg)?;

    // Async: no per-launch device sync (stream-ordered; host reads sync in
    // `to_host`). See the module-level "Asynchronous launch contract".
    Ok(c)
}

// ============================================================================
// Batched External Kernels (for DLPack 3D tensors)
// ============================================================================

/// Launch a batched tropical GEMM kernel with argmax using external (DLPack) 3D tensors.
///
/// Computes C[b] = A[b] ⊗ B[b] for each batch b, where ⊗ is tropical matrix multiplication.
///
/// # Arguments
///
/// * `ctx` - CUDA context
/// * `kernel_name` - Name of the batched kernel to launch
/// * `a` - External 3D tensor A (batch, M, K) in row-major per batch
/// * `b` - External 3D tensor B (batch, K, N) in row-major per batch
/// * `batch` - Batch size
/// * `m` - Rows per matrix in A/C
/// * `k` - Columns in A / rows in B
/// * `n` - Columns per matrix in B/C
///
/// # Safety
///
/// - The input pointers must point to valid GPU memory with the specified dimensions
/// - The memory must remain valid for the duration of the kernel execution
pub unsafe fn launch_gemm_external_batched_with_argmax_f32(
    ctx: &CudaContext,
    kernel_name: &'static str,
    a: &ExternalGpuTensor3<f32>,
    b: &ExternalGpuTensor3<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<GpuTensor3WithArgmax<f32>> {
    // Apply row-major → column-major trick: swap inputs and swap M↔N
    // Same as non-batched version, but for each batch
    let c = GpuTensor3WithArgmax::<f32>::alloc(ctx, batch, m, n)?;

    // Grid: (ceil(N/64) * ceil(M/64), 1, batch) with swapped M↔N. Reuse the
    // shared tile-count helper (keyed off the f32 block constants) rather than
    // hardcoding 64, which would silently desync if the block size changed.
    let grid_xy = CudaContext::grid_dims_f32(n, m).0;
    let block = CudaContext::block_dims_f32();
    let kernel = ctx.get_kernel(kernel_name)?;

    // Per-batch element strides. A/B carry the (possibly padded) stride declared
    // by the external DLPack tensor; C/argmax are freshly allocated contiguous,
    // so their stride is rows*cols. Derive the kernel's `i32` strides fallibly
    // so a stride past `i32::MAX` can't truncate out of sync with the `usize`
    // base-pointer offsets below.
    let stride_a = a.stride();
    let stride_b = b.stride();
    let stride_c = c.tensor.stride();
    let stride_a_i32 = stride_to_i32(stride_a, "operand A")?;
    let stride_b_i32 = stride_to_i32(stride_b, "operand B")?;
    let stride_c_i32 = stride_to_i32(stride_c, "output C")?;
    let n_i32 = n as i32; // Swapped: N becomes "M"
    let m_i32 = m as i32; // Swapped: M becomes "N"
    let k_i32 = k as i32;

    // Operand base device addresses (byte pointers). C and argmax are owned by
    // `c`, which outlives every (async, stream-ordered) launch below.
    let a_base: u64 = a.device_ptr();
    let b_base: u64 = b.device_ptr();
    let c_base: u64 = c.tensor.device_ptr();
    let am_base: u64 = c.argmax.device_ptr();
    let f32_bytes = std::mem::size_of::<f32>() as u64;
    let idx_bytes = std::mem::size_of::<u32>() as u64; // ArgmaxIndex

    // The batch maps to `blockIdx.z` (CUDA cap `MAX_GRID_DIM_Z`). Launch in
    // chunks of at most that size, advancing each operand base pointer by the
    // chunk start so `blockIdx.z ∈ [0, chunk)` indexes the correct batch slice.
    // Swap order: pass B first, then A, and swap M↔N (row-major→col-major).
    // Kernel signature: (A, B, C, argmax, M, N, K, strideA, strideB, strideC)
    // We pass:          (B, A, C, argmax, N, M, K, strideB, strideA, strideC)
    let stream = ctx.stream();
    let mut start = 0usize;
    while start < batch {
        let chunk = (batch - start).min(MAX_GRID_DIM_Z);
        let a_ptr = a_base + (start * stride_a) as u64 * f32_bytes;
        let b_ptr = b_base + (start * stride_b) as u64 * f32_bytes;
        let c_ptr = c_base + (start * stride_c) as u64 * f32_bytes;
        let am_ptr = am_base + (start * stride_c) as u64 * idx_bytes;
        let cfg = LaunchConfig {
            grid_dim: (grid_xy, 1, chunk as u32),
            block_dim: block,
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernel);
        builder
            .arg(&b_ptr)
            .arg(&a_ptr)
            .arg(&c_ptr)
            .arg(&am_ptr)
            .arg(&n_i32)
            .arg(&m_i32)
            .arg(&k_i32)
            .arg(&stride_b_i32) // strideA (B's stride in our swap)
            .arg(&stride_a_i32) // strideB (A's stride in our swap)
            .arg(&stride_c_i32); // strideC
        builder.launch(cfg)?;
        start += chunk;
    }

    // Async: no per-launch device sync (stream-ordered; host reads sync in
    // `to_host`). See the module-level "Asynchronous launch contract".
    Ok(c)
}
