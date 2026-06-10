//! GPU matrix storage (column-major, shared memory).

use crate::context::MetalContext;
use crate::error::{MetalError, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Scalar types that may live in a [`GpuMatrix`] buffer.
///
/// # Safety
/// Implementors must be plain-old-data with a layout identical to the MSL-side
/// type named in the kernel source (f32↔float, i32↔int, i64↔long, u32↔uint,
/// u64↔ulong, bool↔uchar). For `bool`, [`GpuMatrix::to_host`] normalizes any
/// nonzero byte to `true` on read-back, so kernels are not required to emit
/// strictly 0/1 bytes (ours do anyway: 0/1 operands under `&`/`|`).
pub unsafe trait MetalScalar: private::Sealed + Copy + Default + 'static {}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for bool {}
}

unsafe impl MetalScalar for f32 {}
unsafe impl MetalScalar for i32 {}
unsafe impl MetalScalar for i64 {}
unsafe impl MetalScalar for u32 {}
unsafe impl MetalScalar for u64 {}
unsafe impl MetalScalar for bool {}

/// Column-major matrix in GPU-visible shared memory.
pub struct GpuMatrix<T: MetalScalar> {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    rows: usize,
    cols: usize,
    _marker: PhantomData<T>,
}

/// `rows * cols` with overflow reported as an error (mirrors the CUDA crate's
/// `checked_mul` policy).
fn checked_len(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or(MetalError::DimensionOverflow { rows, cols })
}

impl<T: MetalScalar> GpuMatrix<T> {
    /// Upload column-major host data (one memcpy into shared memory).
    pub fn from_host(ctx: &MetalContext, data: &[T], rows: usize, cols: usize) -> Result<Self> {
        let expected = checked_len(rows, cols)?;
        if data.len() != expected {
            return Err(MetalError::HostLengthMismatch {
                len: data.len(),
                expected,
            });
        }
        if data.is_empty() {
            // avoid newBufferWithBytes reading from a dangling empty-slice ptr
            return Self::alloc(ctx, rows, cols);
        }
        let bytes = std::mem::size_of_val(data);
        // SAFETY: `data` is a live non-empty slice of POD T; Metal copies
        // `bytes` from it before returning.
        let buffer = unsafe {
            ctx.device().newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
                bytes,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::Alloc { bytes })?;
        Ok(Self { buffer, rows, cols, _marker: PhantomData })
    }

    /// Upload row-major host data, transposing into column-major storage.
    pub fn from_host_row_major(
        ctx: &MetalContext,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let expected = checked_len(rows, cols)?;
        if data.len() != expected {
            return Err(MetalError::HostLengthMismatch {
                len: data.len(),
                expected,
            });
        }
        let mut cm = vec![T::default(); data.len()];
        for r in 0..rows {
            for c in 0..cols {
                cm[c * rows + r] = data[r * cols + c];
            }
        }
        Self::from_host(ctx, &cm, rows, cols)
    }

    /// Allocate a zero-filled rows×cols matrix.
    pub fn alloc(ctx: &MetalContext, rows: usize, cols: usize) -> Result<Self> {
        let len = checked_len(rows, cols)?;
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(MetalError::DimensionOverflow { rows, cols })?
            .max(1);
        let buffer = ctx
            .device()
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::Alloc { bytes })?;
        Ok(Self { buffer, rows, cols, _marker: PhantomData })
    }

    /// Copy back to host, column-major.
    pub fn to_host(&self) -> Vec<T> {
        let n = self.rows * self.cols;

        // For T = bool a bitwise copy of any byte other than 0x00/0x01 would be
        // instant UB. Our kernels only ever combine 0/1 bytes with &/|, but the
        // safe API must not depend on kernel discipline: read raw bytes and
        // normalize instead. The TypeId check proves T == bool, so the Vec
        // transmute below converts between identical types (size 1, align 1).
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
            let mut bytes = vec![0u8; n];
            // SAFETY: buffer holds at least n bytes; contents() stays valid
            // while `self.buffer` is retained.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buffer.contents().as_ptr() as *const u8,
                    bytes.as_mut_ptr(),
                    n,
                );
            }
            let bools: Vec<bool> = bytes.into_iter().map(|b| b != 0).collect();
            // SAFETY: TypeId equality above guarantees T is bool.
            return unsafe { std::mem::transmute::<Vec<bool>, Vec<T>>(bools) };
        }

        let mut out = vec![T::default(); n];
        // SAFETY: buffer holds at least n POD T values (any bit pattern of the
        // remaining MetalScalar types is a valid value); contents() stays valid
        // while `self.buffer` is retained. Callers must not race the GPU: every
        // kernel launch in this crate calls waitUntilCompleted before
        // returning, which upholds this.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buffer.contents().as_ptr() as *const T,
                out.as_mut_ptr(),
                n,
            );
        }
        out
    }

    /// Copy back to host, row-major.
    pub fn to_host_row_major(&self) -> Vec<T> {
        let cm = self.to_host();
        let mut rm = vec![T::default(); cm.len()];
        for r in 0..self.rows {
            for c in 0..self.cols {
                rm[r * self.cols + c] = cm[c * self.rows + r];
            }
        }
        rm
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    /// Leading dimension (== rows; matrices are always packed).
    pub fn ld(&self) -> usize { self.rows }

    // Used by Task 4+ dispatch code.
    pub(crate) fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalContext;

    #[test]
    fn col_major_roundtrip() {
        let ctx = MetalContext::new().unwrap();
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect(); // 3x2 col-major
        let m = GpuMatrix::from_host(&ctx, &data, 3, 2).unwrap();
        assert_eq!((m.rows(), m.cols(), m.ld()), (3, 2, 3));
        assert_eq!(m.to_host(), data);
    }

    #[test]
    fn row_major_roundtrip_transposes() {
        let ctx = MetalContext::new().unwrap();
        // 2x3 row-major: [[1,2,3],[4,5,6]]
        let rm = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = GpuMatrix::from_host_row_major(&ctx, &rm, 2, 3).unwrap();
        // col-major storage: [1,4, 2,5, 3,6]
        assert_eq!(m.to_host(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(m.to_host_row_major(), rm);
    }

    #[test]
    fn alloc_is_zeroed_len() {
        let ctx = MetalContext::new().unwrap();
        let m = GpuMatrix::<i64>::alloc(&ctx, 5, 7).unwrap();
        assert_eq!(m.to_host().len(), 35);
    }

    #[test]
    fn bool_roundtrip_normalized() {
        let ctx = MetalContext::new().unwrap();
        let data = vec![true, false, true, true, false, false];
        let m = GpuMatrix::from_host(&ctx, &data, 3, 2).unwrap();
        assert_eq!(m.to_host(), data);
    }

    #[test]
    fn length_mismatch_is_error() {
        let ctx = MetalContext::new().unwrap();
        let r = GpuMatrix::from_host(&ctx, &[1.0f32; 5], 2, 3);
        assert!(matches!(r, Err(crate::MetalError::HostLengthMismatch { .. })));
    }
}
