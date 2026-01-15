//! GPU memory management for matrices on Metal.

use crate::context::MetalContext;
use crate::error::{MetalError, Result};
use metal::{Buffer, MTLResourceOptions};
use std::mem;

/// Type alias for argmax indices (k-index that produced each C[i,j]).
pub type ArgmaxIndex = i32;

/// A matrix stored in GPU memory.
///
/// Data is stored in column-major order (Fortran order) for compatibility
/// with BLAS conventions.
pub struct GpuMatrix<T: Copy + Default> {
    buffer: Buffer,
    rows: usize,
    cols: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Copy + Default> GpuMatrix<T> {
    /// Create a GPU matrix from host data.
    ///
    /// The input data should be in row-major order. It will be transposed
    /// to column-major for GPU storage.
    pub fn from_host_row_major(
        ctx: &MetalContext,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(MetalError::DimensionMismatch(format!(
                "Expected {} elements, got {}",
                rows * cols,
                data.len()
            )));
        }

        // Transpose to column-major
        let mut col_major = vec![T::default(); rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                col_major[j * rows + i] = data[i * cols + j];
            }
        }

        let byte_len = col_major.len() * mem::size_of::<T>();
        let buffer = ctx.device().new_buffer_with_data(
            col_major.as_ptr() as *const _,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer,
            rows,
            cols,
            _marker: std::marker::PhantomData,
        })
    }

    /// Create a GPU matrix from column-major host data (no transpose).
    pub fn from_host_col_major(
        ctx: &MetalContext,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(MetalError::DimensionMismatch(format!(
                "Expected {} elements, got {}",
                rows * cols,
                data.len()
            )));
        }

        let byte_len = data.len() * mem::size_of::<T>();
        let buffer = ctx.device().new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer,
            rows,
            cols,
            _marker: std::marker::PhantomData,
        })
    }

    /// Allocate a zeroed GPU matrix.
    pub fn alloc(ctx: &MetalContext, rows: usize, cols: usize) -> Result<Self> {
        let byte_len = (rows * cols * mem::size_of::<T>()) as u64;
        let buffer = ctx.device().new_buffer(
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );

        // Zero the buffer
        unsafe {
            std::ptr::write_bytes(buffer.contents() as *mut u8, 0, byte_len as usize);
        }

        Ok(Self {
            buffer,
            rows,
            cols,
            _marker: std::marker::PhantomData,
        })
    }

    /// Copy GPU data back to host in row-major order.
    pub fn to_host_row_major(&self, _ctx: &MetalContext) -> Result<Vec<T>> {
        // Read column-major data from buffer
        let col_major = self.read_buffer();

        // Transpose from column-major to row-major
        let mut row_major = vec![T::default(); self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                row_major[i * self.cols + j] = col_major[j * self.rows + i];
            }
        }

        Ok(row_major)
    }

    /// Copy GPU data back to host in column-major order.
    pub fn to_host_col_major(&self, _ctx: &MetalContext) -> Result<Vec<T>> {
        Ok(self.read_buffer())
    }

    /// Read buffer contents into a Vec.
    fn read_buffer(&self) -> Vec<T> {
        let len = self.rows * self.cols;
        let mut data = vec![T::default(); len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buffer.contents() as *const T,
                data.as_mut_ptr(),
                len,
            );
        }
        data
    }

    /// Get the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the leading dimension (number of rows for column-major).
    pub fn ld(&self) -> usize {
        self.rows
    }

    /// Get the underlying Metal buffer (for kernel launches).
    pub fn as_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get a mutable reference to the underlying Metal buffer.
    pub fn as_buffer_mut(&mut self) -> &mut Buffer {
        &mut self.buffer
    }
}

/// A GPU matrix paired with argmax indices (for backward propagation).
///
/// This stores both the result of a tropical GEMM and the k-indices
/// that produced each optimal value in C[i,j]. Used for gradient computation.
pub struct GpuMatrixWithArgmax<T: Copy + Default> {
    /// The result matrix C.
    pub matrix: GpuMatrix<T>,
    /// The argmax indices: argmax[i,j] = k such that C[i,j] = A[i,k] ⊗ B[k,j].
    pub argmax: GpuMatrix<ArgmaxIndex>,
}

impl<T: Copy + Default> GpuMatrixWithArgmax<T> {
    /// Allocate a zeroed GPU matrix with argmax indices.
    pub fn alloc(ctx: &MetalContext, rows: usize, cols: usize) -> Result<Self> {
        let matrix = GpuMatrix::alloc(ctx, rows, cols)?;
        let argmax = GpuMatrix::alloc(ctx, rows, cols)?;

        Ok(Self { matrix, argmax })
    }

    /// Get the number of rows.
    pub fn rows(&self) -> usize {
        self.matrix.rows()
    }

    /// Get the number of columns.
    pub fn cols(&self) -> usize {
        self.matrix.cols()
    }

    /// Copy the result matrix back to host in row-major order.
    pub fn matrix_to_host_row_major(&self, ctx: &MetalContext) -> Result<Vec<T>> {
        self.matrix.to_host_row_major(ctx)
    }

    /// Copy the argmax indices back to host in row-major order.
    pub fn argmax_to_host_row_major(&self, ctx: &MetalContext) -> Result<Vec<ArgmaxIndex>> {
        self.argmax.to_host_row_major(ctx)
    }

    /// Copy the result matrix back to host in column-major order.
    pub fn matrix_to_host_col_major(&self, ctx: &MetalContext) -> Result<Vec<T>> {
        self.matrix.to_host_col_major(ctx)
    }

    /// Copy the argmax indices back to host in column-major order.
    pub fn argmax_to_host_col_major(&self, ctx: &MetalContext) -> Result<Vec<ArgmaxIndex>> {
        self.argmax.to_host_col_major(ctx)
    }
}
