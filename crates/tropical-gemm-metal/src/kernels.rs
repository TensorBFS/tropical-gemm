//! Kernel dispatch: semiring type -> kernel name -> pipeline launch.

use crate::context::MetalContext;
use crate::error::{MetalError, Result};
use crate::memory::{GpuMatrix, MetalScalar};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;
use tropical_gemm::types::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring};

// Fields are read by the GPU (copied via setBytes), never on the CPU side.
#[allow(dead_code)]
#[repr(C)]
struct GemmParams {
    m: i32,
    n: i32,
    k: i32,
}

/// Block sizes per scalar width (must match the .metal instantiations).
/// `BLOCK_8B` (32, 32) is introduced in Task 5 alongside the i64 kernels.
// tile (BM, BN) shared by the 4-byte scalars (f32/i32/u32) and 1-byte bool
pub(crate) const BLOCK_4B: (usize, usize) = (64, 64);

/// Tropical semirings with a plain-GEMM Metal kernel.
pub trait MetalKernel: TropicalSemiring
where
    Self::Scalar: MetalScalar,
{
    const KERNEL_NAME: &'static str;
    /// (BM, BN) of the kernel's tile; threadgroup is (BM/4)x(BN/4).
    const BLOCK: (usize, usize);
}

fn check_dims<T: MetalScalar>(
    a: &GpuMatrix<T>,
    b: &GpuMatrix<T>,
    c: &GpuMatrix<T>,
) -> Result<()> {
    if a.cols() != b.rows() || c.rows() != a.rows() || c.cols() != b.cols() {
        return Err(MetalError::DimensionMismatch {
            m: a.rows(),
            ka: a.cols(),
            kb: b.rows(),
            n: b.cols(),
        });
    }
    Ok(())
}

/// Shared launch path: bind A,B,C(+argmax) and params, dispatch, wait.
///
/// Buffer-index contract: plain kernels take params at buffer(3); argmax
/// kernels take argmax at buffer(3) and params at buffer(4). `kernel_name`
/// and `argmax` must therefore be paired consistently — the two public
/// wrappers (`tropical_gemm_gpu`, `tropical_gemm_gpu_with_argmax`) each do so.
pub(crate) fn launch_gemm_impl<T: MetalScalar>(
    ctx: &MetalContext,
    kernel_name: &'static str,
    block: (usize, usize),
    a: &GpuMatrix<T>,
    b: &GpuMatrix<T>,
    c: &mut GpuMatrix<T>,
    argmax: Option<&mut GpuMatrix<u32>>,
) -> Result<()> {
    check_dims(a, b, c)?;
    let (m, n, k) = (a.rows(), b.cols(), a.cols());
    debug_assert!(
        m <= i32::MAX as usize && n <= i32::MAX as usize && k <= i32::MAX as usize,
        "matrix dimensions exceed i32 kernel params"
    );

    // Metal requires every dispatch dimension > 0. An empty output (m or n
    // zero) has nothing to compute; K == 0 still dispatches and fills C with
    // the semiring zero (the kernel's accum init), matching the CPU semantics.
    if m == 0 || n == 0 {
        return Ok(());
    }

    let pipeline = ctx.get_pipeline(kernel_name)?;

    let cmd = ctx
        .queue()
        .commandBuffer()
        .ok_or(MetalError::DeviceNotFound)?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or(MetalError::DeviceNotFound)?;
    enc.setComputePipelineState(pipeline);

    let params = GemmParams { m: m as i32, n: n as i32, k: k as i32 };
    // SAFETY: buffers outlive the synchronous launch (waitUntilCompleted below);
    // params is a live local copied by setBytes.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(a.buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(b.buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(c.buffer()), 0, 2);
        let params_index = if let Some(am) = argmax {
            enc.setBuffer_offset_atIndex(Some(am.buffer()), 0, 3);
            4
        } else {
            3
        };
        enc.setBytes_length_atIndex(
            NonNull::new(&params as *const GemmParams as *mut c_void).unwrap(),
            std::mem::size_of::<GemmParams>(),
            params_index,
        );
    }

    // 1D grid encoding (gx*gy), decoded in-kernel — mirrors CUDA's blockIdx.x.
    let (bm, bn) = block;
    let gx = m.div_ceil(bm);
    let gy = n.div_ceil(bn);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: gx * gy, height: 1, depth: 1 },
        MTLSize { width: bm / 4, height: bn / 4, depth: 1 },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}

/// Launch `T`'s GEMM kernel: C = A ⊗ B under the semiring.
pub fn tropical_gemm_gpu<T>(
    ctx: &MetalContext,
    a: &GpuMatrix<T::Scalar>,
    b: &GpuMatrix<T::Scalar>,
    c: &mut GpuMatrix<T::Scalar>,
) -> Result<()>
where
    T: MetalKernel,
    T::Scalar: MetalScalar,
{
    launch_gemm_impl(ctx, T::KERNEL_NAME, T::BLOCK, a, b, c, None)
}

macro_rules! impl_metal_kernel {
    ($($semiring:ty => ($name:literal, $block:expr)),* $(,)?) => {
        $(
            impl MetalKernel for $semiring {
                const KERNEL_NAME: &'static str = $name;
                const BLOCK: (usize, usize) = $block;
            }
        )*
    };
}

impl_metal_kernel!(
    TropicalMaxPlus<f32> => ("tropical_maxplus_f32_nn", BLOCK_4B),
    TropicalMinPlus<f32> => ("tropical_minplus_f32_nn", BLOCK_4B),
    TropicalMaxMul<f32>  => ("tropical_maxmul_f32_nn",  BLOCK_4B),
);
