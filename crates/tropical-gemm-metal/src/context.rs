//! Metal device context and pipeline management.

use crate::error::{MetalError, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
};
use std::collections::HashMap;

// `MTLCreateSystemDefaultDevice` lives behind CoreGraphics (objc2-metal docs).
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// MSL kernel source, compiled at context creation.
const KERNEL_SOURCE: &str = include_str!("../kernels/tropical_gemm.metal");

/// All kernel entry points; `MetalContext::new` builds a pipeline for each.
/// Extended task-by-task as kernels land (mirrors the CUDA crate's KERNEL_NAMES).
pub(crate) const KERNEL_NAMES: &[&str] = &[
    "tropical_maxplus_f32_nn",
    "tropical_minplus_f32_nn",
    "tropical_maxmul_f32_nn",
];

pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    // Retained for lifetime; used by dispatch code landing in Task 4+.
    #[allow(dead_code)]
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl MetalContext {
    /// Create a context: default device, command queue, compile the MSL source,
    /// and eagerly build a pipeline for every kernel in [`KERNEL_NAMES`].
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceNotFound)?;
        let queue = device.newCommandQueue().ok_or(MetalError::DeviceNotFound)?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(KERNEL_SOURCE), None)
            .map_err(|e| MetalError::Compile(e.localizedDescription().to_string()))?;

        let mut pipelines = HashMap::with_capacity(KERNEL_NAMES.len());
        for &name in KERNEL_NAMES {
            let func = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(MetalError::KernelNotFound(name))?;
            let pso = device
                .newComputePipelineStateWithFunction_error(&func)
                .map_err(|e| MetalError::Pipeline {
                    kernel: name,
                    message: e.localizedDescription().to_string(),
                })?;
            // Largest threadgroup we ever dispatch is 256 (4-byte tier). The
            // limit is per-pipeline (register pressure can lower it); these
            // kernels are register-light, so catch regressions in debug builds.
            debug_assert!(
                pso.maxTotalThreadsPerThreadgroup() >= 256,
                "pipeline {name} clamps threadgroup below 256"
            );
            pipelines.insert(name, pso);
        }
        Ok(Self { device, queue, pipelines })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    // Used by Task 4+ dispatch code.
    #[allow(dead_code)]
    pub(crate) fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    // Used by Task 4+ dispatch code.
    pub(crate) fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    // Used by Task 4+ dispatch code and the test module below.
    pub(crate) fn get_pipeline(
        &self,
        name: &'static str,
    ) -> Result<&ProtocolObject<dyn MTLComputePipelineState>> {
        self.pipelines
            .get(name)
            .map(|p| &**p)
            .ok_or(MetalError::KernelNotFound(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_new_compiles_and_builds_pipelines() {
        let ctx = MetalContext::new().expect("MetalContext::new");
        assert!(!ctx.device_name().is_empty());
        // every registered kernel has a pipeline
        for name in KERNEL_NAMES {
            assert!(ctx.get_pipeline(name).is_ok(), "missing pipeline {name}");
        }
    }

    #[test]
    fn unknown_kernel_is_a_clean_error() {
        let ctx = MetalContext::new().unwrap();
        assert!(matches!(
            ctx.get_pipeline("no_such_kernel"),
            Err(crate::MetalError::KernelNotFound(_))
        ));
    }

    #[test]
    fn global_context_is_shared() {
        let a = crate::get_global_context().unwrap();
        let b = crate::get_global_context().unwrap();
        assert!(std::ptr::eq(a, b));
    }
}
