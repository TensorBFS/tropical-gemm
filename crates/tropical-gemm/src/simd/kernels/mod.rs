//! SIMD microkernel implementations.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod portable;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use avx2::{
    Avx2MaxMulF32Kernel, Avx2MaxPlusF32Kernel, Avx2MaxPlusF64Kernel, Avx2MinPlusF32Kernel,
};
#[cfg(target_arch = "aarch64")]
pub use neon::{NeonMaxPlusF32Kernel, NeonMaxPlusF64Kernel, NeonMinPlusF32Kernel};
pub use portable::PortableKernel;
