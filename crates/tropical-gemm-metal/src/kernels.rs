//! Metal kernel trait and implementations.

use crate::context::MetalContext;
use crate::error::Result;
use crate::gpu_mat::{GpuMatrix, GpuMatrixWithArgmax};
use metal::MTLSize;
use tropical_gemm::types::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring};

/// Trait for types that can be computed on Metal GPU.
pub trait MetalKernel: TropicalSemiring
where
    Self::Scalar: Copy + Default,
{
    /// Kernel function name.
    const KERNEL_NAME: &'static str;

    /// Execute the tropical GEMM kernel.
    ///
    /// Computes C = A ⊗ B where ⊗ is tropical matrix multiplication.
    fn launch_gemm(
        ctx: &MetalContext,
        a: &GpuMatrix<Self::Scalar>,
        b: &GpuMatrix<Self::Scalar>,
        c: &mut GpuMatrix<Self::Scalar>,
    ) -> Result<()>;
}

/// Helper function to launch a Metal kernel with given grid/threadgroup dimensions.
fn launch_kernel_impl(
    ctx: &MetalContext,
    kernel_name: &'static str,
    a: &GpuMatrix<f32>,
    b: &GpuMatrix<f32>,
    c: &mut GpuMatrix<f32>,
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
) -> Result<()> {
    let m = a.rows() as u32;
    let k = a.cols() as u32;
    let n = b.cols() as u32;

    let pipeline = ctx.get_pipeline(kernel_name)?;
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(a.as_buffer()), 0);
    encoder.set_buffer(1, Some(b.as_buffer()), 0);
    encoder.set_buffer(2, Some(c.as_buffer()), 0);
    encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &m as *const u32 as *const _);
    encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &n as *const u32 as *const _);
    encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &k as *const u32 as *const _);

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

/// Macro to implement MetalKernel for f32 types.
macro_rules! impl_metal_kernel_f32 {
    ($($semiring:ty => $kernel_name:literal),* $(,)?) => {
        $(
            impl MetalKernel for $semiring {
                const KERNEL_NAME: &'static str = $kernel_name;

                fn launch_gemm(
                    ctx: &MetalContext,
                    a: &GpuMatrix<f32>,
                    b: &GpuMatrix<f32>,
                    c: &mut GpuMatrix<f32>,
                ) -> Result<()> {
                    let grid_size = MetalContext::grid_size_f32(a.rows(), b.cols());
                    let threadgroup_size = MetalContext::threadgroup_size_f32();
                    launch_kernel_impl(ctx, Self::KERNEL_NAME, a, b, c, grid_size, threadgroup_size)
                }
            }
        )*
    };
}

impl_metal_kernel_f32! {
    TropicalMaxPlus<f32> => "tropical_maxplus_f32_nn",
    TropicalMinPlus<f32> => "tropical_minplus_f32_nn",
    TropicalMaxMul<f32> => "tropical_maxmul_f32_nn",
}

// ============================================================================
// MetalKernelWithArgmax - for path reconstruction
// ============================================================================

/// Trait for tropical GEMM with argmax tracking (for backward propagation).
///
/// This computes both C[i,j] and the k-index that produced each C[i,j],
/// which is needed for gradient computation in tropical neural networks.
pub trait MetalKernelWithArgmax: TropicalSemiring
where
    Self::Scalar: Copy + Default,
{
    /// Kernel function name for the argmax variant.
    const ARGMAX_KERNEL_NAME: &'static str;

    /// Execute the tropical GEMM kernel with argmax tracking.
    ///
    /// Computes C = A ⊗ B and also records argmax[i,j] = k such that
    /// C[i,j] = A[i,k] ⊗ B[k,j] was the winning value.
    fn launch_gemm_with_argmax(
        ctx: &MetalContext,
        a: &GpuMatrix<Self::Scalar>,
        b: &GpuMatrix<Self::Scalar>,
        c: &mut GpuMatrixWithArgmax<Self::Scalar>,
    ) -> Result<()>;
}

/// Helper function to launch an argmax Metal kernel.
fn launch_kernel_with_argmax_impl(
    ctx: &MetalContext,
    kernel_name: &'static str,
    a: &GpuMatrix<f32>,
    b: &GpuMatrix<f32>,
    c: &mut GpuMatrixWithArgmax<f32>,
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
) -> Result<()> {
    let m = a.rows() as u32;
    let k = a.cols() as u32;
    let n = b.cols() as u32;

    let pipeline = ctx.get_pipeline(kernel_name)?;
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(a.as_buffer()), 0);
    encoder.set_buffer(1, Some(b.as_buffer()), 0);
    encoder.set_buffer(2, Some(c.matrix.as_buffer()), 0);
    encoder.set_buffer(3, Some(c.argmax.as_buffer()), 0);
    encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &m as *const u32 as *const _);
    encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &n as *const u32 as *const _);
    encoder.set_bytes(6, std::mem::size_of::<u32>() as u64, &k as *const u32 as *const _);

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

/// Macro to implement MetalKernelWithArgmax for f32 types.
macro_rules! impl_metal_kernel_with_argmax_f32 {
    ($($semiring:ty => $kernel_name:literal),* $(,)?) => {
        $(
            impl MetalKernelWithArgmax for $semiring {
                const ARGMAX_KERNEL_NAME: &'static str = $kernel_name;

                fn launch_gemm_with_argmax(
                    ctx: &MetalContext,
                    a: &GpuMatrix<f32>,
                    b: &GpuMatrix<f32>,
                    c: &mut GpuMatrixWithArgmax<f32>,
                ) -> Result<()> {
                    let grid_size = MetalContext::grid_size_f32(a.rows(), b.cols());
                    let threadgroup_size = MetalContext::threadgroup_size_f32();
                    launch_kernel_with_argmax_impl(ctx, Self::ARGMAX_KERNEL_NAME, a, b, c, grid_size, threadgroup_size)
                }
            }
        )*
    };
}

impl_metal_kernel_with_argmax_f32! {
    TropicalMaxPlus<f32> => "tropical_maxplus_f32_nn_with_argmax",
    TropicalMinPlus<f32> => "tropical_minplus_f32_nn_with_argmax",
    TropicalMaxMul<f32> => "tropical_maxmul_f32_nn_with_argmax",
}

// ============================================================================
// Backward pass kernels
// ============================================================================

/// Launch the backward pass kernel for gradient w.r.t. A.
#[allow(dead_code)]
pub fn launch_backward_a(
    ctx: &MetalContext,
    grad_c: &GpuMatrix<f32>,
    argmax: &GpuMatrix<i32>,
    grad_a: &mut GpuMatrix<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let total = m * n;
    let pipeline = ctx.get_pipeline("tropical_backward_a_f32")?;
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(grad_c.as_buffer()), 0);
    encoder.set_buffer(1, Some(argmax.as_buffer()), 0);
    encoder.set_buffer(2, Some(grad_a.as_buffer()), 0);
    encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &m_u32 as *const u32 as *const _);
    encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &n_u32 as *const u32 as *const _);
    encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &k_u32 as *const u32 as *const _);

    let threads_per_group = 256u64;
    let num_groups = ((total as u64) + threads_per_group - 1) / threads_per_group;
    let grid_size = MTLSize::new(num_groups, 1, 1);
    let threadgroup_size = MTLSize::new(threads_per_group, 1, 1);

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

/// Launch the backward pass kernel for gradient w.r.t. B.
#[allow(dead_code)]
pub fn launch_backward_b(
    ctx: &MetalContext,
    grad_c: &GpuMatrix<f32>,
    argmax: &GpuMatrix<i32>,
    grad_b: &mut GpuMatrix<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let total = m * n;
    let pipeline = ctx.get_pipeline("tropical_backward_b_f32")?;
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(grad_c.as_buffer()), 0);
    encoder.set_buffer(1, Some(argmax.as_buffer()), 0);
    encoder.set_buffer(2, Some(grad_b.as_buffer()), 0);
    encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &m_u32 as *const u32 as *const _);
    encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &n_u32 as *const u32 as *const _);
    encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &k_u32 as *const u32 as *const _);

    let threads_per_group = 256u64;
    let num_groups = ((total as u64) + threads_per_group - 1) / threads_per_group;
    let grid_size = MTLSize::new(num_groups, 1, 1);
    let threadgroup_size = MTLSize::new(threads_per_group, 1, 1);

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}
