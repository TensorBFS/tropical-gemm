//! GPU memory management for matrices.

use crate::context::CudaContext;
use crate::error::{CudaError, Result};
use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
use std::marker::PhantomData;

/// Type alias for argmax indices (k-index that produced each C[i,j]).
pub type ArgmaxIndex = i32;

/// A matrix stored in GPU memory.
///
/// Data is stored in column-major order (Fortran order) for compatibility
/// with BLAS conventions.
pub struct GpuMatrix<T: DeviceRepr> {
    data: CudaSlice<T>,
    rows: usize,
    cols: usize,
    _marker: PhantomData<T>,
}

impl<T: DeviceRepr + Default + Clone + ValidAsZeroBits> GpuMatrix<T> {
    /// Create a GPU matrix from host data.
    ///
    /// The input data should be in row-major order. It will be transposed
    /// to column-major for GPU storage.
    pub fn from_host_row_major(
        ctx: &CudaContext,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(CudaError::DimensionMismatch(format!(
                "Expected {} elements, got {}",
                rows * cols,
                data.len()
            )));
        }

        // Transpose to column-major
        let mut col_major = vec![T::default(); rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                col_major[j * rows + i] = data[i * cols + j].clone();
            }
        }

        let gpu_data = ctx.device().htod_sync_copy(&col_major)?;

        Ok(Self {
            data: gpu_data,
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    /// Create a GPU matrix from column-major host data (no transpose).
    pub fn from_host_col_major(
        ctx: &CudaContext,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(CudaError::DimensionMismatch(format!(
                "Expected {} elements, got {}",
                rows * cols,
                data.len()
            )));
        }

        let gpu_data = ctx.device().htod_sync_copy(data)?;

        Ok(Self {
            data: gpu_data,
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    /// Allocate a zeroed GPU matrix.
    pub fn alloc(ctx: &CudaContext, rows: usize, cols: usize) -> Result<Self> {
        let gpu_data = ctx.device().alloc_zeros::<T>(rows * cols)?;

        Ok(Self {
            data: gpu_data,
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    /// Copy GPU data back to host in row-major order.
    pub fn to_host_row_major(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        let col_major = ctx.device().dtoh_sync_copy(&self.data)?;

        // Transpose from column-major to row-major
        let mut row_major = vec![T::default(); self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                row_major[i * self.cols + j] = col_major[j * self.rows + i].clone();
            }
        }

        Ok(row_major)
    }

    /// Copy GPU data back to host in column-major order.
    pub fn to_host_col_major(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        Ok(ctx.device().dtoh_sync_copy(&self.data)?)
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

    /// Get the underlying CUDA slice (for kernel launches).
    pub fn as_slice(&self) -> &CudaSlice<T> {
        &self.data
    }

    /// Get a mutable reference to the underlying CUDA slice.
    pub fn as_slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.data
    }
}

/// A GPU matrix paired with argmax indices (for backward propagation).
///
/// This stores both the result of a tropical GEMM and the k-indices
/// that produced each optimal value in C[i,j]. Used for gradient computation.
pub struct GpuMatrixWithArgmax<T: DeviceRepr> {
    /// The result matrix C.
    pub matrix: GpuMatrix<T>,
    /// The argmax indices: argmax[i,j] = k such that C[i,j] = A[i,k] ⊗ B[k,j].
    pub argmax: GpuMatrix<ArgmaxIndex>,
}

impl<T: DeviceRepr + Default + Clone + ValidAsZeroBits> GpuMatrixWithArgmax<T> {
    /// Allocate a zeroed GPU matrix with argmax indices.
    pub fn alloc(ctx: &CudaContext, rows: usize, cols: usize) -> Result<Self> {
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
    pub fn matrix_to_host_row_major(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        self.matrix.to_host_row_major(ctx)
    }

    /// Copy the argmax indices back to host in row-major order.
    pub fn argmax_to_host_row_major(&self, ctx: &CudaContext) -> Result<Vec<ArgmaxIndex>> {
        self.argmax.to_host_row_major(ctx)
    }

    /// Copy the result matrix back to host in column-major order.
    pub fn matrix_to_host_col_major(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        self.matrix.to_host_col_major(ctx)
    }

    /// Copy the argmax indices back to host in column-major order.
    pub fn argmax_to_host_col_major(&self, ctx: &CudaContext) -> Result<Vec<ArgmaxIndex>> {
        self.argmax.to_host_col_major(ctx)
    }
}
