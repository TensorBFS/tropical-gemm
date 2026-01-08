//! GPU memory management for matrices.

use crate::context::MetalContext;
use crate::error::{MetalError, Result};
use metal::{Buffer, MTLResourceOptions};
use std::marker::PhantomData;

/// A matrix stored in GPU memory.
///
/// Data is stored in column-major order for compatibility with BLAS conventions.
pub struct GpuMatrix<T> {
    buffer: Buffer,
    rows: usize,
    cols: usize,
    _marker: PhantomData<T>,
}

impl<T: Clone + Default + Copy> GpuMatrix<T> {
    /// Create a GPU matrix from host data in row-major order.
    ///
    /// The input data will be transposed to column-major for GPU storage.
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

        let byte_size = (rows * cols * std::mem::size_of::<T>()) as u64;
        let buffer = ctx.device().new_buffer_with_data(
            col_major.as_ptr() as *const _,
            byte_size,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer,
            rows,
            cols,
            _marker: PhantomData,
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

        let byte_size = (rows * cols * std::mem::size_of::<T>()) as u64;
        let buffer = ctx.device().new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_size,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer,
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    /// Allocate a zeroed GPU matrix.
    pub fn alloc(ctx: &MetalContext, rows: usize, cols: usize) -> Result<Self> {
        let byte_size = (rows * cols * std::mem::size_of::<T>()) as u64;
        let buffer = ctx.device().new_buffer(
            byte_size,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer,
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    /// Copy GPU data back to host in row-major order.
    pub fn to_host_row_major(&self) -> Vec<T> {
        let ptr = self.buffer.contents() as *const T;
        let col_major: Vec<T> = unsafe {
            std::slice::from_raw_parts(ptr, self.rows * self.cols).to_vec()
        };

        // Transpose from column-major to row-major
        let mut row_major = vec![T::default(); self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                row_major[i * self.cols + j] = col_major[j * self.rows + i];
            }
        }

        row_major
    }

    /// Copy GPU data back to host in column-major order.
    pub fn to_host_col_major(&self) -> Vec<T> {
        let ptr = self.buffer.contents() as *const T;
        unsafe {
            std::slice::from_raw_parts(ptr, self.rows * self.cols).to_vec()
        }
    }

    /// Get the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the underlying Metal buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}
