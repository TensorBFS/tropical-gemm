//! CUDA kernel trait and implementations.

use crate::context::CudaContext;
use crate::error::Result;
use crate::memory::{ExternalGpuMatrix, GpuMatrix, GpuMatrixWithArgmax};
use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use tropical_gemm::types::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring};

/// Trait for types that can be computed on GPU.
pub trait CudaKernel: TropicalSemiring
where
    Self::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Kernel function name.
    const KERNEL_NAME: &'static str;

    /// Execute the tropical GEMM kernel.
    ///
    /// Computes C = A ⊗ B where ⊗ is tropical matrix multiplication.
    fn launch_gemm(
        ctx: &CudaContext,
        a: &GpuMatrix<Self::Scalar>,
        b: &GpuMatrix<Self::Scalar>,
        c: &mut GpuMatrix<Self::Scalar>,
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

    unsafe {
        kernel.launch(
            cfg,
            (
                a.as_slice(),
                b.as_slice(),
                c.as_slice_mut(),
                m as i32,
                n as i32,
                k as i32,
            ),
        )?;
    }

    ctx.device().synchronize()?;
    Ok(())
}

/// Macro to implement CudaKernel for f32 types.
macro_rules! impl_cuda_kernel_f32 {
    ($($semiring:ty => $kernel_name:literal),* $(,)?) => {
        $(
            impl CudaKernel for $semiring {
                const KERNEL_NAME: &'static str = $kernel_name;

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
            }
        )*
    };
}

/// Macro to implement CudaKernel for f64 types.
macro_rules! impl_cuda_kernel_f64 {
    ($($semiring:ty => $kernel_name:literal),* $(,)?) => {
        $(
            impl CudaKernel for $semiring {
                const KERNEL_NAME: &'static str = $kernel_name;

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
            }
        )*
    };
}

impl_cuda_kernel_f32! {
    TropicalMaxPlus<f32> => "tropical_maxplus_f32_nn",
    TropicalMinPlus<f32> => "tropical_minplus_f32_nn",
    TropicalMaxMul<f32> => "tropical_maxmul_f32_nn",
}

impl_cuda_kernel_f64! {
    TropicalMaxPlus<f64> => "tropical_maxplus_f64_nn",
    TropicalMinPlus<f64> => "tropical_minplus_f64_nn",
    TropicalMaxMul<f64> => "tropical_maxmul_f64_nn",
}

/// Macro to implement CudaKernel for i32 types.
/// Uses same block sizes as f32 (64x32x64) since int is 4 bytes.
macro_rules! impl_cuda_kernel_i32 {
    ($($semiring:ty => $kernel_name:literal),* $(,)?) => {
        $(
            impl CudaKernel for $semiring {
                const KERNEL_NAME: &'static str = $kernel_name;

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
            }
        )*
    };
}

/// Macro to implement CudaKernel for i64 types.
/// Uses same block sizes as f64 (32x16x32) since long long is 8 bytes.
macro_rules! impl_cuda_kernel_i64 {
    ($($semiring:ty => $kernel_name:literal),* $(,)?) => {
        $(
            impl CudaKernel for $semiring {
                const KERNEL_NAME: &'static str = $kernel_name;

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
            }
        )*
    };
}

impl_cuda_kernel_i32! {
    TropicalMaxPlus<i32> => "tropical_maxplus_i32_nn",
    TropicalMinPlus<i32> => "tropical_minplus_i32_nn",
    TropicalMaxMul<i32> => "tropical_maxmul_i32_nn",
}

impl_cuda_kernel_i64! {
    TropicalMaxPlus<i64> => "tropical_maxplus_i64_nn",
    TropicalMinPlus<i64> => "tropical_minplus_i64_nn",
    TropicalMaxMul<i64> => "tropical_maxmul_i64_nn",
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

    unsafe {
        kernel.launch(
            cfg,
            (
                a.as_slice(),
                b.as_slice(),
                c.matrix.as_slice_mut(),
                c.argmax.as_slice_mut(),
                m as i32,
                n as i32,
                k as i32,
            ),
        )?;
    }

    ctx.device().synchronize()?;
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

    // Swap order: pass B first, then A, and swap M↔N
    // Kernel signature: (A_ptr, B_ptr, C_ptr, argmax_ptr, M, N, K)
    // We pass:          (B_ptr, A_ptr, C_ptr, argmax_ptr, N, M, K)
    kernel.launch(
        cfg,
        (
            b.device_ptr(), // B becomes "A" in kernel
            a.device_ptr(), // A becomes "B" in kernel
            c.matrix.as_slice_mut(),
            c.argmax.as_slice_mut(),
            n as i32, // Swapped: N becomes "M"
            m as i32, // Swapped: M becomes "N"
            k as i32,
        ),
    )?;

    ctx.device().synchronize()?;
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

    // Swap order and dimensions
    kernel.launch(
        cfg,
        (
            b.device_ptr(),
            a.device_ptr(),
            c.as_slice_mut(),
            n as i32,
            m as i32,
            k as i32,
        ),
    )?;

    ctx.device().synchronize()?;
    Ok(c)
}
