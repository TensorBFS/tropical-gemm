//! Metal context and device management.

use crate::error::{MetalError, Result};
use metal::{Device, ComputePipelineState, CommandQueue};
use std::collections::HashMap;

/// Metal shader source code.
const SHADER_SOURCE: &str = include_str!("../shaders/tropical_gemm.metal");

/// Blocking parameters for f32 kernels.
pub const BLOCK_SIZE_M_F32: u32 = 32;
pub const BLOCK_SIZE_N_F32: u32 = 32;
pub const THREAD_SIZE_M: u32 = 4;
pub const THREAD_SIZE_N: u32 = 4;

/// Kernel function names.
const KERNEL_NAMES: &[&str] = &[
    "tropical_maxplus_f32",
    "tropical_minplus_f32",
    "tropical_maxmul_f32",
];

/// Metal context for tropical GEMM operations.
///
/// Manages device selection, shader compilation, and pipeline caching.
pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
}

impl MetalContext {
    /// Create a new Metal context on the default device.
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        Self::from_device(device)
    }

    /// Create a context from an existing device.
    pub fn from_device(device: Device) -> Result<Self> {
        let command_queue = device
            .new_command_queue();

        // Compile shaders
        let options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

        // Create compute pipelines for each kernel
        let mut pipelines = HashMap::new();
        for name in KERNEL_NAMES {
            let function = library
                .get_function(name, None)
                .map_err(|_| MetalError::KernelNotFound(name.to_string()))?;

            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MetalError::PipelineCreation(e.to_string()))?;

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

    /// Calculate threadgroup size for a kernel.
    pub fn threadgroup_size(&self) -> metal::MTLSize {
        let threads_per_group_m = BLOCK_SIZE_M_F32 / THREAD_SIZE_M;
        let threads_per_group_n = BLOCK_SIZE_N_F32 / THREAD_SIZE_N;
        metal::MTLSize::new(
            threads_per_group_m as u64,
            threads_per_group_n as u64,
            1,
        )
    }

    /// Calculate grid size for given matrix dimensions.
    pub fn grid_size(&self, m: usize, n: usize) -> metal::MTLSize {
        let grid_x = (m as u64 + BLOCK_SIZE_M_F32 as u64 - 1) / BLOCK_SIZE_M_F32 as u64;
        let grid_y = (n as u64 + BLOCK_SIZE_N_F32 as u64 - 1) / BLOCK_SIZE_N_F32 as u64;
        metal::MTLSize::new(grid_x, grid_y, 1)
    }
}
