//! CUDA context and kernel management.

use crate::error::{CudaError, Result};
use cudarc::driver::{CudaDevice, CudaFunction};
use std::collections::HashMap;
use std::sync::Arc;

/// CUDA kernel source code.
const KERNEL_SOURCE: &str = include_str!("../kernels/tropical_gemm.cu");

/// Blocking parameters for f32 kernels.
pub const BLOCK_SIZE_M_F32: u32 = 64;
pub const BLOCK_SIZE_N_F32: u32 = 64;
pub const THREAD_SIZE_M: u32 = 4;
pub const THREAD_SIZE_N: u32 = 4;

/// Blocking parameters for f64 kernels.
pub const BLOCK_SIZE_M_F64: u32 = 32;
pub const BLOCK_SIZE_N_F64: u32 = 32;

/// Kernel function names.
const KERNEL_NAMES: &[&str] = &[
    // Standard GEMM kernels
    "tropical_maxplus_f32_nn",
    "tropical_minplus_f32_nn",
    "tropical_maxmul_f32_nn",
    "tropical_maxplus_f64_nn",
    // GEMM with argmax kernels (for backward propagation)
    "tropical_maxplus_f32_nn_with_argmax",
    "tropical_minplus_f32_nn_with_argmax",
    "tropical_maxplus_f64_nn_with_argmax",
    "tropical_minplus_f64_nn_with_argmax",
];

/// CUDA context for tropical GEMM operations.
///
/// Manages device selection, kernel compilation, and caching.
pub struct CudaContext {
    device: Arc<CudaDevice>,
    kernels: HashMap<&'static str, CudaFunction>,
}

impl CudaContext {
    /// Create a new CUDA context on the default device (device 0).
    pub fn new() -> Result<Self> {
        Self::new_on_device(0)
    }

    /// Create a new CUDA context on a specific device.
    pub fn new_on_device(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id)?;
        Self::from_device(device)
    }

    /// Create a context from an existing device.
    pub fn from_device(device: Arc<CudaDevice>) -> Result<Self> {
        // Compile kernels using NVRTC
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SOURCE)?;

        // Load PTX module
        device.load_ptx(ptx, "tropical_gemm", KERNEL_NAMES)?;

        // Cache kernel functions
        let mut kernels = HashMap::new();
        for name in KERNEL_NAMES {
            let func = device
                .get_func("tropical_gemm", name)
                .ok_or_else(|| CudaError::KernelNotFound(name.to_string()))?;
            kernels.insert(*name, func);
        }

        Ok(Self { device, kernels })
    }

    /// Get the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get a kernel function by name.
    pub fn get_kernel(&self, name: &'static str) -> Result<CudaFunction> {
        self.kernels
            .get(name)
            .cloned()
            .ok_or_else(|| CudaError::KernelNotFound(name.to_string()))
    }

    /// Get GPU device name.
    pub fn device_name(&self) -> String {
        format!("CUDA Device {}", self.device.ordinal())
    }

    /// Calculate grid dimensions for a given matrix size.
    pub fn grid_dims_f32(m: usize, n: usize) -> (u32, u32, u32) {
        let grid_x = ((m as u32) + BLOCK_SIZE_M_F32 - 1) / BLOCK_SIZE_M_F32;
        let grid_y = ((n as u32) + BLOCK_SIZE_N_F32 - 1) / BLOCK_SIZE_N_F32;
        (grid_x * grid_y, 1, 1)
    }

    /// Calculate grid dimensions for f64 kernels.
    pub fn grid_dims_f64(m: usize, n: usize) -> (u32, u32, u32) {
        let grid_x = ((m as u32) + BLOCK_SIZE_M_F64 - 1) / BLOCK_SIZE_M_F64;
        let grid_y = ((n as u32) + BLOCK_SIZE_N_F64 - 1) / BLOCK_SIZE_N_F64;
        (grid_x * grid_y, 1, 1)
    }

    /// Block dimensions for f32 kernels.
    pub fn block_dims_f32() -> (u32, u32, u32) {
        let bszm = BLOCK_SIZE_M_F32 / THREAD_SIZE_M;
        let bszn = BLOCK_SIZE_N_F32 / THREAD_SIZE_N;
        (bszm, bszn, 1)
    }

    /// Block dimensions for f64 kernels.
    pub fn block_dims_f64() -> (u32, u32, u32) {
        let bszm = BLOCK_SIZE_M_F64 / THREAD_SIZE_M;
        let bszn = BLOCK_SIZE_N_F64 / THREAD_SIZE_N;
        (bszm, bszn, 1)
    }
}
