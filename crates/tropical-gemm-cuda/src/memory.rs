//! GPU memory management for matrices.

use crate::context::CudaContext;
use crate::error::{CudaError, Result};
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
use std::marker::PhantomData;

/// Type alias for argmax indices (k-index that produced each C[i,j]).
pub type ArgmaxIndex = i32;

// ============================================================================
// Helper: validate dimensions
// ============================================================================

fn validate_dims<T>(data: &[T], rows: usize, cols: usize) -> Result<()> {
    if data.len() != rows * cols {
        return Err(CudaError::DimensionMismatch(format!(
            "Expected {} elements, got {}",
            rows * cols,
            data.len()
        )));
    }
    Ok(())
}

/// A matrix stored in GPU memory.
///
/// Data is stored in column-major order (Fortran/BLAS convention).
/// This matches the tropical-gemm crate's Mat type.
pub struct GpuMatrix<T: DeviceRepr> {
    data: CudaSlice<T>,
    rows: usize,
    cols: usize,
    _marker: PhantomData<T>,
}

impl<T: DeviceRepr + Default + Clone + ValidAsZeroBits> GpuMatrix<T> {
    /// Create a GPU matrix from column-major host data (zero-copy upload).
    ///
    /// This is the primary upload method since tropical-gemm uses column-major storage.
    pub fn from_host(ctx: &CudaContext, data: &[T], rows: usize, cols: usize) -> Result<Self> {
        validate_dims(data, rows, cols)?;
        let gpu_data = ctx.device().htod_sync_copy(data)?;
        Ok(Self {
            data: gpu_data,
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    /// Create a GPU matrix from row-major host data (transposes during upload).
    ///
    /// Use this when interfacing with row-major data sources (e.g., C arrays).
    /// For column-major data, use `from_host` instead for better performance.
    ///
    /// # Performance Warning
    ///
    /// This method performs an O(rows×cols) transpose on the CPU before uploading to GPU.
    /// For performance-critical code, provide data in column-major order and use
    /// [`from_host`] instead.
    #[deprecated(since = "0.4.0", note = "use from_host with column-major data instead; this method has O(m×n) transpose overhead")]
    pub fn from_host_row_major(
        ctx: &CudaContext,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        validate_dims(data, rows, cols)?;
        // Transpose to column-major
        let col_major: Vec<T> = (0..rows * cols)
            .map(|idx| {
                let i = idx % rows;
                let j = idx / rows;
                data[i * cols + j].clone()
            })
            .collect();
        let gpu_data = ctx.device().htod_sync_copy(&col_major)?;
        Ok(Self {
            data: gpu_data,
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    /// Alias for `from_host` (column-major).
    #[inline]
    pub fn from_host_col_major(
        ctx: &CudaContext,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        Self::from_host(ctx, data, rows, cols)
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

    /// Copy GPU data back to host in column-major order (zero-copy download).
    ///
    /// This is the primary download method since tropical-gemm uses column-major storage.
    pub fn to_host(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        Ok(ctx.device().dtoh_sync_copy(&self.data)?)
    }

    /// Copy GPU data back to host in row-major order (transposes during download).
    ///
    /// Use this when interfacing with row-major data consumers.
    /// For column-major data, use `to_host` instead for better performance.
    ///
    /// # Performance Warning
    ///
    /// This method performs an O(rows×cols) transpose on the CPU after downloading from GPU.
    /// For performance-critical code, use [`to_host`] and handle the column-major layout
    /// in your application.
    #[deprecated(since = "0.4.0", note = "use to_host for column-major data instead; this method has O(m×n) transpose overhead")]
    pub fn to_host_row_major(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        let col_major = ctx.device().dtoh_sync_copy(&self.data)?;
        // Transpose from column-major to row-major
        let row_major: Vec<T> = (0..self.rows * self.cols)
            .map(|idx| {
                let i = idx / self.cols;
                let j = idx % self.cols;
                col_major[j * self.rows + i].clone()
            })
            .collect();
        Ok(row_major)
    }

    /// Alias for `to_host` (column-major).
    #[inline]
    pub fn to_host_col_major(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        self.to_host(ctx)
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

    /// Get the raw device pointer (for DLPack export).
    pub fn device_ptr(&self) -> CUdeviceptr {
        use cudarc::driver::DevicePtr;
        *self.data.device_ptr()
    }

    /// Consume self and return the inner CudaSlice (for ownership transfer).
    pub fn into_inner(self) -> CudaSlice<T> {
        self.data
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

    /// Copy the result matrix back to host (column-major).
    pub fn matrix_to_host(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        self.matrix.to_host(ctx)
    }

    /// Copy the argmax indices back to host (column-major).
    pub fn argmax_to_host(&self, ctx: &CudaContext) -> Result<Vec<ArgmaxIndex>> {
        self.argmax.to_host(ctx)
    }

    /// Copy the result matrix back to host in row-major order (deprecated).
    #[deprecated(since = "0.4.0", note = "use matrix_to_host for column-major data instead")]
    pub fn matrix_to_host_row_major(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        #[allow(deprecated)]
        self.matrix.to_host_row_major(ctx)
    }

    /// Copy the argmax indices back to host in row-major order (deprecated).
    #[deprecated(since = "0.4.0", note = "use argmax_to_host for column-major data instead")]
    pub fn argmax_to_host_row_major(&self, ctx: &CudaContext) -> Result<Vec<ArgmaxIndex>> {
        #[allow(deprecated)]
        self.argmax.to_host_row_major(ctx)
    }

    /// Alias for `matrix_to_host` (column-major).
    #[inline]
    pub fn matrix_to_host_col_major(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        self.matrix_to_host(ctx)
    }

    /// Alias for `argmax_to_host` (column-major).
    #[inline]
    pub fn argmax_to_host_col_major(&self, ctx: &CudaContext) -> Result<Vec<ArgmaxIndex>> {
        self.argmax_to_host(ctx)
    }

    /// Consume self and return the matrix and argmax separately.
    ///
    /// This is useful for DLPack export where each tensor needs to be wrapped
    /// independently for ownership transfer.
    pub fn into_parts(self) -> (GpuMatrix<T>, GpuMatrix<ArgmaxIndex>) {
        (self.matrix, self.argmax)
    }
}

// ============================================================================
// External GPU Memory (DLPack integration)
// ============================================================================

/// A non-owning reference to GPU memory from an external source (e.g., PyTorch via DLPack).
///
/// This struct holds a raw device pointer without ownership. It does NOT free
/// the memory on drop - the original owner (e.g., PyTorch) remains responsible
/// for memory management.
///
/// # Safety
///
/// The caller must ensure that the underlying memory remains valid for the
/// lifetime of this struct. This is typically guaranteed by holding a reference
/// to the original tensor (e.g., via DLManagedTensor).
pub struct ExternalGpuMemory<T> {
    device_ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> ExternalGpuMemory<T> {
    /// Create a new ExternalGpuMemory from a raw device pointer.
    ///
    /// # Safety
    ///
    /// - `device_ptr` must point to valid GPU memory containing at least `len` elements of type T
    /// - The memory must remain valid for the lifetime of this struct
    /// - The memory must be properly aligned for type T
    pub unsafe fn from_raw(device_ptr: CUdeviceptr, len: usize) -> Self {
        Self {
            device_ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Get the raw device pointer.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.device_ptr
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A 2D matrix view into external GPU memory.
///
/// This represents a matrix stored in row-major order (as PyTorch tensors are).
/// The actual data is not copied - we just store metadata and a pointer.
pub struct ExternalGpuMatrix<T> {
    memory: ExternalGpuMemory<T>,
    rows: usize,
    cols: usize,
}

impl<T> ExternalGpuMatrix<T> {
    /// Create a new ExternalGpuMatrix from a raw device pointer.
    ///
    /// # Safety
    ///
    /// - `device_ptr` must point to valid GPU memory containing at least `rows * cols` elements
    /// - The memory must be in row-major (C-contiguous) order
    /// - The memory must remain valid for the lifetime of this struct
    pub unsafe fn from_raw(device_ptr: CUdeviceptr, rows: usize, cols: usize) -> Self {
        let memory = ExternalGpuMemory::from_raw(device_ptr, rows * cols);
        Self { memory, rows, cols }
    }

    /// Get the raw device pointer.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.memory.device_ptr()
    }

    /// Get the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// Check if the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }
}

/// A 3D tensor view into external GPU memory for batched operations.
///
/// This represents a batch of matrices stored contiguously in row-major order
/// (as PyTorch tensors are). Shape is (batch, rows, cols) with stride between batches.
///
/// The actual data is not copied - we just store metadata and a pointer.
pub struct ExternalGpuTensor3<T> {
    device_ptr: CUdeviceptr,
    batch: usize,
    rows: usize,
    cols: usize,
    stride: usize, // elements per batch (typically rows * cols for contiguous)
    _marker: PhantomData<T>,
}

impl<T> ExternalGpuTensor3<T> {
    /// Create a new ExternalGpuTensor3 from a raw device pointer.
    ///
    /// # Safety
    ///
    /// - `device_ptr` must point to valid GPU memory containing at least `batch * stride` elements
    /// - The memory must be in row-major (C-contiguous) order per batch
    /// - The memory must remain valid for the lifetime of this struct
    pub unsafe fn from_raw(
        device_ptr: CUdeviceptr,
        batch: usize,
        rows: usize,
        cols: usize,
        stride: usize,
    ) -> Self {
        Self {
            device_ptr,
            batch,
            rows,
            cols,
            stride,
            _marker: PhantomData,
        }
    }

    /// Create from contiguous 3D tensor (stride = rows * cols).
    ///
    /// # Safety
    ///
    /// Same requirements as `from_raw`.
    pub unsafe fn from_raw_contiguous(
        device_ptr: CUdeviceptr,
        batch: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self::from_raw(device_ptr, batch, rows, cols, rows * cols)
    }

    /// Get the raw device pointer.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.device_ptr
    }

    /// Get the batch size.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Get the number of rows per matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns per matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the stride (elements between batches).
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.batch * self.stride
    }

    /// Check if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.batch == 0 || self.rows == 0 || self.cols == 0
    }

    /// Check if the tensor is contiguous (stride == rows * cols).
    pub fn is_contiguous(&self) -> bool {
        self.stride == self.rows * self.cols
    }
}

/// A batched GPU matrix result with owned memory.
///
/// Stores batch_size matrices of shape (rows, cols) contiguously on GPU.
pub struct GpuTensor3<T: DeviceRepr> {
    data: CudaSlice<T>,
    batch: usize,
    rows: usize,
    cols: usize,
}

impl<T: DeviceRepr + Default + Clone + ValidAsZeroBits> GpuTensor3<T> {
    /// Allocate a zeroed batched GPU tensor.
    pub fn alloc(ctx: &CudaContext, batch: usize, rows: usize, cols: usize) -> Result<Self> {
        let gpu_data = ctx.device().alloc_zeros::<T>(batch * rows * cols)?;
        Ok(Self {
            data: gpu_data,
            batch,
            rows,
            cols,
        })
    }

    /// Copy GPU data back to host as a flat vector (batch × rows × cols elements).
    pub fn to_host(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        Ok(ctx.device().dtoh_sync_copy(&self.data)?)
    }

    /// Get the batch size.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Get rows per matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get columns per matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get stride (elements per batch).
    pub fn stride(&self) -> usize {
        self.rows * self.cols
    }

    /// Get the underlying CUDA slice.
    pub fn as_slice(&self) -> &CudaSlice<T> {
        &self.data
    }

    /// Get a mutable reference to the underlying CUDA slice.
    pub fn as_slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.data
    }

    /// Get the raw device pointer (for DLPack export).
    pub fn device_ptr(&self) -> CUdeviceptr {
        use cudarc::driver::DevicePtr;
        *self.data.device_ptr()
    }

    /// Consume self and return the inner CudaSlice (for ownership transfer).
    pub fn into_inner(self) -> CudaSlice<T> {
        self.data
    }
}

/// A batched GPU tensor with argmax indices for backward propagation.
pub struct GpuTensor3WithArgmax<T: DeviceRepr> {
    /// The result tensor C (batch × rows × cols).
    pub tensor: GpuTensor3<T>,
    /// The argmax indices (batch × rows × cols).
    pub argmax: GpuTensor3<ArgmaxIndex>,
}

impl<T: DeviceRepr + Default + Clone + ValidAsZeroBits> GpuTensor3WithArgmax<T> {
    /// Allocate a zeroed batched GPU tensor with argmax.
    pub fn alloc(ctx: &CudaContext, batch: usize, rows: usize, cols: usize) -> Result<Self> {
        let tensor = GpuTensor3::alloc(ctx, batch, rows, cols)?;
        let argmax = GpuTensor3::alloc(ctx, batch, rows, cols)?;
        Ok(Self { tensor, argmax })
    }

    /// Get batch size.
    pub fn batch(&self) -> usize {
        self.tensor.batch()
    }

    /// Get rows per matrix.
    pub fn rows(&self) -> usize {
        self.tensor.rows()
    }

    /// Get cols per matrix.
    pub fn cols(&self) -> usize {
        self.tensor.cols()
    }

    /// Copy the result tensor back to host.
    pub fn tensor_to_host(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        self.tensor.to_host(ctx)
    }

    /// Copy the argmax indices back to host.
    pub fn argmax_to_host(&self, ctx: &CudaContext) -> Result<Vec<ArgmaxIndex>> {
        self.argmax.to_host(ctx)
    }

    /// Consume self and return the tensor and argmax components separately.
    ///
    /// This is useful for DLPack export where each tensor needs to be wrapped
    /// independently for ownership transfer.
    pub fn into_parts(self) -> (GpuTensor3<T>, GpuTensor3<ArgmaxIndex>) {
        (self.tensor, self.argmax)
    }
}
