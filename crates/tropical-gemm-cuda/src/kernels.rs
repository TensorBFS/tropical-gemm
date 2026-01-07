//! CUDA kernel trait and implementations.

use crate::context::CudaContext;
use crate::error::Result;
use crate::memory::GpuMatrix;
use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use tropical_types::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring};

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
}
