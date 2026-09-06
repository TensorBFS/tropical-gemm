//! Metal context and kernel management.

use crate::error::{MetalError, Result};
use metal::{CommandQueue, ComputePipelineState, Device, MTLSize};
use std::collections::HashMap;

/// Metal kernel source code.
const KERNEL_SOURCE: &str = include_str!("../shaders/tropical_gemm.metal");

/// Blocking parameters for f32 kernels.
pub const BLOCK_SIZE_M_F32: u32 = 64;
pub const BLOCK_SIZE_N_F32: u32 = 64;
pub const THREAD_SIZE_M: u32 = 4;
pub const THREAD_SIZE_N: u32 = 4;

/// Kernel function names.
const KERNEL_NAMES: &[&str] = &[
    // Standard GEMM kernels (f32)
    "tropical_maxplus_f32_nn",
    "tropical_minplus_f32_nn",
    "tropical_maxmul_f32_nn",
    // GEMM with argmax kernels (f32)
    "tropical_maxplus_f32_nn_with_argmax",
    "tropical_minplus_f32_nn_with_argmax",
    "tropical_maxmul_f32_nn_with_argmax",
    // Backward pass kernels (f32)
    "tropical_backward_a_f32",
    "tropical_backward_b_f32",
];

/// Metal context for tropical GEMM operations.
///
/// Manages device selection, shader compilation, and caching.
pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
}

impl MetalContext {
    /// Create a new Metal context on the default system device.
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        Self::from_device(device)
    }

    /// Create a context from an existing device.
    pub fn from_device(device: Device) -> Result<Self> {
        // Create command queue
        let command_queue = device.new_command_queue();

        // Compile shaders from source
        let options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(KERNEL_SOURCE, &options)
            .map_err(|e| MetalError::ShaderCompile(e.to_string()))?;

        // Create compute pipelines for each kernel
        let mut pipelines = HashMap::new();
        for name in KERNEL_NAMES {
            let function = library
                .get_function(name, None)
                .map_err(|e| MetalError::KernelNotFound(format!("{}: {}", name, e)))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MetalError::Library(format!("Pipeline for {}: {}", name, e)))?;
            pipelines.insert(*name, pipeline);
        }

        Ok(Self {
            device,
            command_queue,
            pipelines,
        })
    }

    /// Get the underlying Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the command queue.
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Get a compute pipeline by kernel name.
    pub fn get_pipeline(&self, name: &'static str) -> Result<&ComputePipelineState> {
        self.pipelines
            .get(name)
            .ok_or_else(|| MetalError::KernelNotFound(name.to_string()))
    }

    /// Get GPU device name.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Calculate grid dimensions for a given matrix size.
    pub fn grid_size_f32(m: usize, n: usize) -> MTLSize {
        let grid_x = ((m as u64) + BLOCK_SIZE_M_F32 as u64 - 1) / BLOCK_SIZE_M_F32 as u64;
        let grid_y = ((n as u64) + BLOCK_SIZE_N_F32 as u64 - 1) / BLOCK_SIZE_N_F32 as u64;
        MTLSize::new(grid_x, grid_y, 1)
    }

    /// Threadgroup size for f32 kernels.
    pub fn threadgroup_size_f32() -> MTLSize {
        let bszm = BLOCK_SIZE_M_F32 as u64 / THREAD_SIZE_M as u64;
        let bszn = BLOCK_SIZE_N_F32 as u64 / THREAD_SIZE_N as u64;
        MTLSize::new(bszm, bszn, 1)
    }

    /// Calculate grid dimensions for backward pass kernels.
    pub fn grid_size_backward(total_elements: usize) -> MTLSize {
        let threads_per_group = 256u64;
        let num_groups = ((total_elements as u64) + threads_per_group - 1) / threads_per_group;
        MTLSize::new(num_groups * threads_per_group, 1, 1)
    }

    /// Threadgroup size for backward kernels.
    pub fn threadgroup_size_backward() -> MTLSize {
        MTLSize::new(256, 1, 1)
    }
}
