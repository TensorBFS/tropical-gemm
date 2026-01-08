use crate::simd::SimdLevel;

/// Available backends for tropical GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Pure Rust portable implementation.
    Portable,
    /// SIMD-accelerated (AVX2, NEON, etc.).
    Simd,
}

impl Backend {
    /// Get the currently active backend based on CPU features.
    pub fn current() -> Self {
        match crate::simd::simd_level() {
            SimdLevel::Scalar => Backend::Portable,
            _ => Backend::Simd,
        }
    }

    /// Get a description of the current SIMD capabilities.
    pub fn description() -> String {
        let level = crate::simd::simd_level();
        match level {
            SimdLevel::Scalar => "Portable (no SIMD)".to_string(),
            SimdLevel::Sse2 => "x86-64 SSE2 (128-bit)".to_string(),
            SimdLevel::Avx => "x86-64 AVX (256-bit float)".to_string(),
            SimdLevel::Avx2 => "x86-64 AVX2 (256-bit)".to_string(),
            SimdLevel::Avx512 => "x86-64 AVX-512 (512-bit)".to_string(),
            SimdLevel::Neon => "ARM NEON (128-bit)".to_string(),
        }
    }
}

/// Get information about the library configuration.
pub fn version_info() -> String {
    format!(
        "tropical-gemm v{}\nBackend: {}\nSIMD Level: {:?}",
        env!("CARGO_PKG_VERSION"),
        Backend::description(),
        crate::simd::simd_level()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_detection() {
        let backend = Backend::current();
        println!("Current backend: {:?}", backend);
        println!("Description: {}", Backend::description());
        println!("Version info:\n{}", version_info());

        // Verify backend is one of the expected values
        assert!(backend == Backend::Portable || backend == Backend::Simd);
    }

    #[test]
    fn test_backend_description_not_empty() {
        let desc = Backend::description();
        assert!(!desc.is_empty());
        // Description should mention SIMD type or portable
        assert!(
            desc.contains("Portable")
                || desc.contains("SSE2")
                || desc.contains("AVX")
                || desc.contains("NEON")
        );
    }

    #[test]
    fn test_version_info_format() {
        let info = version_info();
        assert!(info.contains("tropical-gemm v"));
        assert!(info.contains("Backend:"));
        assert!(info.contains("SIMD Level:"));
    }

    #[test]
    fn test_backend_debug() {
        let backend = Backend::current();
        let debug_str = format!("{:?}", backend);
        assert!(debug_str == "Portable" || debug_str == "Simd");
    }

    #[test]
    fn test_backend_clone() {
        let backend = Backend::current();
        let cloned = backend;
        assert_eq!(backend, cloned);
    }

    #[test]
    fn test_backend_eq() {
        assert_eq!(Backend::Portable, Backend::Portable);
        assert_eq!(Backend::Simd, Backend::Simd);
        assert_ne!(Backend::Portable, Backend::Simd);
    }
}
