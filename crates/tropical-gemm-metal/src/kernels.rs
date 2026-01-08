//! Metal kernel trait and implementations.

use crate::context::MetalContext;
use crate::error::Result;
use crate::memory::GpuMatrix;
use tropical_types::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring};

/// Trait for types that can be computed on Metal GPU.
pub trait MetalKernel: TropicalSemiring
where
    Self::Scalar: Clone + Default + Copy,
{
    /// Kernel function name.
    const KERNEL_NAME: &'static str;

    /// Execute the tropical GEMM kernel.
    fn launch_gemm(
        ctx: &MetalContext,
        a: &GpuMatrix<Self::Scalar>,
        b: &GpuMatrix<Self::Scalar>,
        c: &mut GpuMatrix<Self::Scalar>,
    ) -> Result<()>;
}

/// Helper function to launch a Metal compute kernel.
fn launch_kernel_impl(
    ctx: &MetalContext,
    kernel_name: &'static str,
    a: &GpuMatrix<f32>,
    b: &GpuMatrix<f32>,
    c: &mut GpuMatrix<f32>,
) -> Result<()> {
    let m = a.rows() as i32;
    let k = a.cols() as i32;
    let n = b.cols() as i32;

    let pipeline = ctx.get_pipeline(kernel_name)?;
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(a.buffer()), 0);
    encoder.set_buffer(1, Some(b.buffer()), 0);
    encoder.set_buffer(2, Some(c.buffer()), 0);
    encoder.set_bytes(3, std::mem::size_of::<i32>() as u64, &m as *const i32 as *const _);
    encoder.set_bytes(4, std::mem::size_of::<i32>() as u64, &n as *const i32 as *const _);
    encoder.set_bytes(5, std::mem::size_of::<i32>() as u64, &k as *const i32 as *const _);

    let grid_size = ctx.grid_size(a.rows(), b.cols());
    let threadgroup_size = ctx.threadgroup_size();

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

impl MetalKernel for TropicalMaxPlus<f32> {
    const KERNEL_NAME: &'static str = "tropical_maxplus_f32";

    fn launch_gemm(
        ctx: &MetalContext,
        a: &GpuMatrix<f32>,
        b: &GpuMatrix<f32>,
        c: &mut GpuMatrix<f32>,
    ) -> Result<()> {
        launch_kernel_impl(ctx, Self::KERNEL_NAME, a, b, c)
    }
}

impl MetalKernel for TropicalMinPlus<f32> {
    const KERNEL_NAME: &'static str = "tropical_minplus_f32";

    fn launch_gemm(
        ctx: &MetalContext,
        a: &GpuMatrix<f32>,
        b: &GpuMatrix<f32>,
        c: &mut GpuMatrix<f32>,
    ) -> Result<()> {
        launch_kernel_impl(ctx, Self::KERNEL_NAME, a, b, c)
    }
}

impl MetalKernel for TropicalMaxMul<f32> {
    const KERNEL_NAME: &'static str = "tropical_maxmul_f32";

    fn launch_gemm(
        ctx: &MetalContext,
        a: &GpuMatrix<f32>,
        b: &GpuMatrix<f32>,
        c: &mut GpuMatrix<f32>,
    ) -> Result<()> {
        launch_kernel_impl(ctx, Self::KERNEL_NAME, a, b, c)
    }
}
