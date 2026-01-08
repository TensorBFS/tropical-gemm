//! Tropical number types and semiring traits.
//!
//! This module provides the core type system for tropical algebra operations.

mod and_or;
mod counting;
mod max_mul;
mod max_plus;
mod min_plus;
mod scalar;
mod traits;

pub use and_or::TropicalAndOr;
pub use counting::CountingTropical;
pub use max_mul::TropicalMaxMul;
pub use max_plus::TropicalMaxPlus;
pub use min_plus::TropicalMinPlus;
pub use scalar::TropicalScalar;
pub use traits::{SimdTropical, TropicalSemiring, TropicalWithArgmax};
