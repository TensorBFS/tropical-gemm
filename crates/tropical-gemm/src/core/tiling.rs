/// Tiling parameters for BLIS-style GEMM blocking.
///
/// These parameters control how matrices are partitioned to fit in
/// various levels of the cache hierarchy.
#[derive(Debug, Clone, Copy)]
pub struct TilingParams {
    /// Block size for M dimension (L2 cache).
    pub mc: usize,
    /// Block size for N dimension (L2 cache).
    pub nc: usize,
    /// Block size for K dimension (L1 cache).
    pub kc: usize,
    /// Microkernel M dimension (registers).
    pub mr: usize,
    /// Microkernel N dimension (registers).
    pub nr: usize,
}

impl TilingParams {
    /// Default parameters for f32 with AVX2.
    pub const F32_AVX2: Self = Self {
        mc: 256,
        nc: 256,
        kc: 512,
        mr: 8,
        nr: 8,
    };

    /// Default parameters for f64 with AVX2.
    pub const F64_AVX2: Self = Self {
        mc: 128,
        nc: 128,
        kc: 256,
        mr: 4,
        nr: 4,
    };

    /// Default parameters for portable (non-SIMD) execution.
    pub const PORTABLE: Self = Self {
        mc: 64,
        nc: 64,
        kc: 256,
        mr: 4,
        nr: 4,
    };

    /// Create custom tiling parameters.
    pub const fn new(mc: usize, nc: usize, kc: usize, mr: usize, nr: usize) -> Self {
        Self { mc, nc, kc, mr, nr }
    }

    /// Validate that tiling parameters are consistent.
    pub fn validate(&self) -> Result<(), &'static str> {
        // Check for zero values first (before divisibility checks)
        if self.mr == 0 || self.nr == 0 {
            return Err("mr and nr must be non-zero");
        }
        if self.mc == 0 || self.nc == 0 || self.kc == 0 {
            return Err("mc, nc, and kc must be non-zero");
        }
        // Now check divisibility
        if !self.mc.is_multiple_of(self.mr) {
            return Err("mc must be divisible by mr");
        }
        if !self.nc.is_multiple_of(self.nr) {
            return Err("nc must be divisible by nr");
        }
        Ok(())
    }
}

impl Default for TilingParams {
    fn default() -> Self {
        Self::PORTABLE
    }
}

/// Iterator over blocks for the outer loop.
pub struct BlockIterator {
    total: usize,
    block_size: usize,
    current: usize,
}

impl BlockIterator {
    pub fn new(total: usize, block_size: usize) -> Self {
        Self {
            total,
            block_size,
            current: 0,
        }
    }
}

impl Iterator for BlockIterator {
    /// (start, length) of each block
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total {
            return None;
        }

        let start = self.current;
        let len = (self.total - start).min(self.block_size);
        self.current += len;

        Some((start, len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_iterator() {
        let iter = BlockIterator::new(10, 3);
        let blocks: Vec<_> = iter.collect();

        assert_eq!(blocks, vec![(0, 3), (3, 3), (6, 3), (9, 1)]);
    }

    #[test]
    fn test_block_iterator_exact() {
        // When total is exactly divisible by block_size
        let iter = BlockIterator::new(9, 3);
        let blocks: Vec<_> = iter.collect();
        assert_eq!(blocks, vec![(0, 3), (3, 3), (6, 3)]);
    }

    #[test]
    fn test_block_iterator_empty() {
        let iter = BlockIterator::new(0, 3);
        let blocks: Vec<_> = iter.collect();
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_validate_params() {
        assert!(TilingParams::F32_AVX2.validate().is_ok());
        assert!(TilingParams::F64_AVX2.validate().is_ok());
        assert!(TilingParams::PORTABLE.validate().is_ok());

        let bad = TilingParams::new(100, 64, 256, 8, 8);
        assert!(bad.validate().is_err()); // 100 % 8 != 0
    }

    #[test]
    fn test_validate_nc_not_divisible() {
        // nc not divisible by nr
        let bad = TilingParams::new(64, 100, 256, 8, 8);
        assert_eq!(bad.validate(), Err("nc must be divisible by nr"));
    }

    #[test]
    fn test_validate_mr_zero() {
        let bad = TilingParams::new(64, 64, 256, 0, 8);
        assert_eq!(bad.validate(), Err("mr and nr must be non-zero"));
    }

    #[test]
    fn test_validate_nr_zero() {
        let bad = TilingParams::new(64, 64, 256, 8, 0);
        assert_eq!(bad.validate(), Err("mr and nr must be non-zero"));
    }

    #[test]
    fn test_validate_mc_zero() {
        let bad = TilingParams::new(0, 64, 256, 8, 8);
        assert_eq!(bad.validate(), Err("mc, nc, and kc must be non-zero"));
    }

    #[test]
    fn test_validate_nc_zero() {
        let bad = TilingParams::new(64, 0, 256, 8, 8);
        assert_eq!(bad.validate(), Err("mc, nc, and kc must be non-zero"));
    }

    #[test]
    fn test_validate_kc_zero() {
        let bad = TilingParams::new(64, 64, 0, 8, 8);
        assert_eq!(bad.validate(), Err("mc, nc, and kc must be non-zero"));
    }

    #[test]
    fn test_default() {
        let default = TilingParams::default();
        assert_eq!(default.mc, TilingParams::PORTABLE.mc);
        assert_eq!(default.nc, TilingParams::PORTABLE.nc);
        assert_eq!(default.kc, TilingParams::PORTABLE.kc);
        assert_eq!(default.mr, TilingParams::PORTABLE.mr);
        assert_eq!(default.nr, TilingParams::PORTABLE.nr);
    }

    #[test]
    fn test_debug() {
        let params = TilingParams::PORTABLE;
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("TilingParams"));
    }

    #[test]
    fn test_clone() {
        let params = TilingParams::F32_AVX2;
        let cloned = params;
        assert_eq!(params.mc, cloned.mc);
    }
}
