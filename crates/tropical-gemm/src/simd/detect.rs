/// CPU feature detection for runtime SIMD dispatch.
/// Available SIMD instruction sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// No SIMD, use scalar code.
    Scalar,
    /// SSE2 (128-bit, available on all x86-64).
    Sse2,
    /// AVX (256-bit float).
    Avx,
    /// AVX2 (256-bit integer + FMA).
    Avx2,
    /// AVX-512 (512-bit).
    Avx512,
    /// ARM NEON (128-bit).
    Neon,
}

impl SimdLevel {
    /// Detect the best available SIMD level at runtime.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdLevel::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdLevel::Avx2;
            }
            if is_x86_feature_detected!("avx") {
                return SimdLevel::Avx;
            }
            // SSE2 is always available on x86-64
            SimdLevel::Sse2
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on AArch64
            SimdLevel::Neon
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdLevel::Scalar
        }
    }

    /// Get the SIMD width in bytes.
    pub fn width_bytes(&self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse2 | SimdLevel::Neon => 16,
            SimdLevel::Avx | SimdLevel::Avx2 => 32,
            SimdLevel::Avx512 => 64,
        }
    }

    /// Get the number of f32 elements that fit in one SIMD register.
    pub fn f32_width(&self) -> usize {
        self.width_bytes() / 4
    }

    /// Get the number of f64 elements that fit in one SIMD register.
    pub fn f64_width(&self) -> usize {
        self.width_bytes() / 8
    }
}

/// Global cached SIMD level.
static SIMD_LEVEL: std::sync::OnceLock<SimdLevel> = std::sync::OnceLock::new();

/// Get the detected SIMD level (cached).
pub fn simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(SimdLevel::detect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect() {
        let level = SimdLevel::detect();
        println!("Detected SIMD level: {:?}", level);

        // Should detect at least Scalar
        assert!(level >= SimdLevel::Scalar);

        // On x86-64, should detect at least SSE2
        #[cfg(target_arch = "x86_64")]
        assert!(level >= SimdLevel::Sse2);

        // On AArch64, should detect NEON
        #[cfg(target_arch = "aarch64")]
        assert_eq!(level, SimdLevel::Neon);
    }

    #[test]
    fn test_width() {
        assert_eq!(SimdLevel::Avx2.f32_width(), 8);
        assert_eq!(SimdLevel::Avx2.f64_width(), 4);
        assert_eq!(SimdLevel::Sse2.f32_width(), 4);
    }

    #[test]
    fn test_width_bytes() {
        assert_eq!(SimdLevel::Scalar.width_bytes(), 1);
        assert_eq!(SimdLevel::Sse2.width_bytes(), 16);
        assert_eq!(SimdLevel::Neon.width_bytes(), 16);
        assert_eq!(SimdLevel::Avx.width_bytes(), 32);
        assert_eq!(SimdLevel::Avx2.width_bytes(), 32);
        assert_eq!(SimdLevel::Avx512.width_bytes(), 64);
    }

    #[test]
    fn test_all_widths() {
        // f32 widths
        assert_eq!(SimdLevel::Scalar.f32_width(), 0); // 1/4 = 0
        assert_eq!(SimdLevel::Sse2.f32_width(), 4); // 16/4
        assert_eq!(SimdLevel::Neon.f32_width(), 4); // 16/4
        assert_eq!(SimdLevel::Avx.f32_width(), 8); // 32/4
        assert_eq!(SimdLevel::Avx512.f32_width(), 16); // 64/4

        // f64 widths
        assert_eq!(SimdLevel::Scalar.f64_width(), 0); // 1/8 = 0
        assert_eq!(SimdLevel::Sse2.f64_width(), 2); // 16/8
        assert_eq!(SimdLevel::Neon.f64_width(), 2); // 16/8
        assert_eq!(SimdLevel::Avx.f64_width(), 4); // 32/8
        assert_eq!(SimdLevel::Avx512.f64_width(), 8); // 64/8
    }

    #[test]
    fn test_simd_level_cached() {
        // Calling simd_level() multiple times should return same value
        let level1 = simd_level();
        let level2 = simd_level();
        assert_eq!(level1, level2);
    }
}
