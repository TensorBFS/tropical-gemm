//! GPU memory management for matrices.

use crate::context::CudaContext;
use crate::error::{CudaError, Result};
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits};
use std::marker::PhantomData;

/// Type alias for argmax indices (k-index that produced each C[i,j]).
///
/// Unsigned to match the CPU core's `u32` argmax index and the downstream
/// consumer (omeinsum-rs). k-indices are non-negative; a no-contribution cell
/// is canonicalized to the seed `0` in the argmax-kernel epilogue.
pub type ArgmaxIndex = u32;

// ============================================================================
// Helper: validate dimensions
// ============================================================================

/// Validate that a buffer of `len` elements matches a `rows × cols` matrix.
///
/// Returns the matrix's total element count (`rows * cols`) on success so callers
/// who need it (e.g. for downstream allocation sizing) can avoid re-computing it.
/// Uses `checked_mul` so adversarial `(rows, cols)` cannot overflow `usize` —
/// without this, debug builds would panic inside the very `format!` that builds
/// the error message, and release builds would wrap and silently accept invalid
/// shapes whose product happens to equal `len`.
fn validate_dims_len(len: usize, rows: usize, cols: usize) -> Result<usize> {
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        CudaError::DimensionMismatch(format!("rows * cols overflows usize: {rows} * {cols}"))
    })?;
    if len != expected {
        return Err(CudaError::DimensionMismatch(format!(
            "Expected {expected} elements, got {len}"
        )));
    }
    Ok(expected)
}

fn validate_dims<T>(data: &[T], rows: usize, cols: usize) -> Result<()> {
    validate_dims_len(data.len(), rows, cols).map(|_| ())
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
    #[deprecated(
        since = "0.4.0",
        note = "use `from_host` with column-major host data, or `from_cuda_slice` if your data is already on GPU; this method has O(m×n) transpose overhead"
    )]
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
        let len = rows.checked_mul(cols).ok_or_else(|| {
            CudaError::DimensionMismatch(format!("rows * cols overflows usize: {rows} * {cols}"))
        })?;
        let gpu_data = ctx.device().alloc_zeros::<T>(len)?;
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
    #[deprecated(
        since = "0.4.0",
        note = "if you need host data, use `to_host` (column-major) and transpose at the consumer; if you only wanted a GPU handle you should not have been calling a `to_host_*` method — see `into_inner` / `from_cuda_slice` for the zero-copy path. This method has O(m×n) transpose overhead."
    )]
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

// Zero-copy ownership-transfer constructors live in this separate impl block
// for organization (they're paired with their argmax variant below), but share
// the same `Default + Clone + ValidAsZeroBits` bound as `from_host` / `alloc`
// — loosening it would build `GpuMatrix<T>` values whose accessors (`to_host`,
// `into_inner`, etc.) are then unreachable. If a future PR moves the
// type-erased accessors into a `T: DeviceRepr` block, the bound here can be
// relaxed in lockstep.
impl<T: DeviceRepr + Default + Clone + ValidAsZeroBits> GpuMatrix<T> {
    /// Wrap an already-resident GPU `CudaSlice` as a column-major matrix.
    ///
    /// This performs no copy: the caller transfers ownership of the slice and
    /// gets back a `GpuMatrix` view with the same underlying allocation. It is
    /// the symmetric counterpart to [`GpuMatrix::into_inner`] and intended for
    /// downstream Rust crates that already hold GPU-resident column-major data
    /// and want to feed it into the tropical GEMM API without round-tripping
    /// through host memory.
    ///
    /// The slice is interpreted as column-major (BLAS / Fortran convention),
    /// matching the rest of [`GpuMatrix`]. For row-major external data
    /// (e.g. PyTorch / DLPack) — which `tropical-gemm-cuda` only ingests via a
    /// **non-owning** view — see [`ExternalGpuMatrix`]; there is currently no
    /// owned row-major zero-copy path.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::DimensionMismatch`] if `rows * cols` overflows
    /// `usize`, if `slice.len() != rows * cols`, or if the slice was allocated
    /// on a different CUDA device than `ctx` — the latter would otherwise
    /// surface as a `CUDA_ERROR_INVALID_DEVICE_POINTER` at the next kernel
    /// launch, much further from the actual mistake.
    ///
    /// # Ownership on error
    ///
    /// `slice` is taken by value; on any error it is dropped at function exit,
    /// which calls `cuMemFree` on the underlying allocation. There is no path
    /// to recover the slice on failure — validate `rows`, `cols`, and the
    /// slice's device before calling if you need to retain it on rejection.
    pub fn from_cuda_slice(
        ctx: &CudaContext,
        slice: CudaSlice<T>,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if slice.device().ordinal() != ctx.device().ordinal() {
            return Err(CudaError::DimensionMismatch(format!(
                "slice belongs to CUDA device {} but context is on device {}",
                slice.device().ordinal(),
                ctx.device().ordinal(),
            )));
        }
        validate_dims_len(slice.len(), rows, cols)?;
        Ok(Self {
            data: slice,
            rows,
            cols,
            _marker: PhantomData,
        })
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
    #[deprecated(
        since = "0.4.0",
        note = "if you need host data, use `matrix_to_host` (column-major) and transpose at the consumer; for staying on GPU see `into_parts` + `GpuMatrix::into_inner` (the zero-copy path doesn't produce host data)"
    )]
    pub fn matrix_to_host_row_major(&self, ctx: &CudaContext) -> Result<Vec<T>> {
        #[allow(deprecated)]
        self.matrix.to_host_row_major(ctx)
    }

    /// Copy the argmax indices back to host in row-major order (deprecated).
    #[deprecated(
        since = "0.4.0",
        note = "if you need host data, use `argmax_to_host` (column-major) and transpose at the consumer; for staying on GPU see `into_parts` + `GpuMatrix::into_inner` (the zero-copy path doesn't produce host data)"
    )]
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

// Zero-copy ownership-transfer constructor (symmetric to `into_parts`).
impl<T: DeviceRepr + Default + Clone + ValidAsZeroBits> GpuMatrixWithArgmax<T> {
    /// Wrap two already-resident GPU `CudaSlice`s as a `(matrix, argmax)` pair.
    ///
    /// Zero-copy counterpart to [`GpuMatrixWithArgmax::into_parts`]. Both slices
    /// are interpreted as column-major with shape `rows × cols` and must live
    /// on the same CUDA device as `ctx`.
    ///
    /// All validation runs **before** either slice is wrapped: both raw
    /// `CudaSlice`s remain live as plain locals until the final `Ok(Self {…})`,
    /// so a failure path does not leave a half-wrapped `GpuMatrix` to be torn
    /// down by `?`-unwinding. (A naïve `from_cuda_slice(matrix)? +
    /// from_cuda_slice(argmax)?` would consume `matrix` into a `GpuMatrix` on
    /// the first call; if the second call failed, the wrapped `matrix` would
    /// be dropped during unwind, freeing the caller's allocation alongside a
    /// destructor on a wrapper that should never have existed.)
    ///
    /// # Ownership on error
    ///
    /// Both `matrix` and `argmax` are taken by value; on any error they are
    /// dropped at function exit, which calls `cuMemFree` on both underlying
    /// allocations. The "no half-wrapped state" guarantee above means there is
    /// only one destructor per slice (not one for the slice + one for a
    /// partial wrap), but it does **not** mean the slices survive the call —
    /// validate `rows`, `cols`, and both slices' devices before calling if you
    /// need to retain them on rejection.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::DimensionMismatch`] if `rows * cols` overflows
    /// `usize`, if either slice has the wrong length (the message distinguishes
    /// `matrix` vs `argmax`), or if either slice was allocated on a different
    /// CUDA device than `ctx`.
    ///
    /// **Type hazard.** When `T == ArgmaxIndex` (`u32`) — which is the case for
    /// the `u32`-scalar argmax variants — `matrix` and `argmax` are both
    /// `CudaSlice<u32>` and a positional swap at the call site type-checks.
    /// Pass them in order.
    pub fn from_cuda_slices(
        ctx: &CudaContext,
        matrix: CudaSlice<T>,
        argmax: CudaSlice<ArgmaxIndex>,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let ctx_ord = ctx.device().ordinal();
        if matrix.device().ordinal() != ctx_ord {
            return Err(CudaError::DimensionMismatch(format!(
                "matrix: slice belongs to CUDA device {} but context is on device {ctx_ord}",
                matrix.device().ordinal(),
            )));
        }
        if argmax.device().ordinal() != ctx_ord {
            return Err(CudaError::DimensionMismatch(format!(
                "argmax: slice belongs to CUDA device {} but context is on device {ctx_ord}",
                argmax.device().ordinal(),
            )));
        }
        validate_dims_len(matrix.len(), rows, cols).map_err(|e| match e {
            CudaError::DimensionMismatch(m) => CudaError::DimensionMismatch(format!("matrix: {m}")),
            other => other,
        })?;
        validate_dims_len(argmax.len(), rows, cols).map_err(|e| match e {
            CudaError::DimensionMismatch(m) => CudaError::DimensionMismatch(format!("argmax: {m}")),
            other => other,
        })?;
        Ok(Self {
            matrix: GpuMatrix {
                data: matrix,
                rows,
                cols,
                _marker: PhantomData,
            },
            argmax: GpuMatrix {
                data: argmax,
                rows,
                cols,
                _marker: PhantomData,
            },
        })
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
        let len = batch
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(cols))
            .ok_or_else(|| {
                CudaError::DimensionMismatch(format!(
                    "batch * rows * cols overflows usize: {batch} * {rows} * {cols}"
                ))
            })?;
        let gpu_data = ctx.device().alloc_zeros::<T>(len)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Skip test gracefully when no CUDA device is present.
    fn cuda_context_or_skip() -> Option<&'static CudaContext> {
        // One context shared across the whole test binary: NVRTC compiles once
        // instead of per-test (~7s each). catch_unwind because cudarc panics
        // rather than returning Err when libcuda is absent.
        match std::panic::catch_unwind(crate::get_global_context) {
            Ok(Ok(ctx)) => Some(ctx),
            Ok(Err(e)) => {
                println!("CUDA not available ({e:?}), skipping test");
                None
            }
            Err(_) => {
                println!("CUDA libraries not found, skipping test");
                None
            }
        }
    }

    #[test]
    fn from_cuda_slice_roundtrip_preserves_data() {
        let Some(ctx) = cuda_context_or_skip() else {
            return;
        };

        let host: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let slice = ctx.device().htod_sync_copy(&host).unwrap();

        let mat = GpuMatrix::<f32>::from_cuda_slice(ctx, slice, 3, 4).unwrap();
        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 4);

        let downloaded = mat.to_host(ctx).unwrap();
        assert_eq!(downloaded, host);
    }

    #[test]
    fn from_cuda_slice_into_inner_is_symmetric() {
        let Some(ctx) = cuda_context_or_skip() else {
            return;
        };

        let host = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let original = GpuMatrix::from_host(ctx, &host, 2, 3).unwrap();

        // into_inner → from_cuda_slice round-trip with no data motion
        let inner = original.into_inner();
        let rebuilt = GpuMatrix::<f64>::from_cuda_slice(ctx, inner, 2, 3).unwrap();

        let downloaded = rebuilt.to_host(ctx).unwrap();
        assert_eq!(downloaded, host);
    }

    #[test]
    fn from_cuda_slice_rejects_dimension_mismatch() {
        let Some(ctx) = cuda_context_or_skip() else {
            return;
        };

        let slice = ctx.device().alloc_zeros::<f32>(10).unwrap();
        match GpuMatrix::<f32>::from_cuda_slice(ctx, slice, 3, 4) {
            Err(CudaError::DimensionMismatch(_)) => {}
            Err(e) => panic!("expected DimensionMismatch, got {e:?}"),
            Ok(_) => panic!("expected error for slice length 10 vs 3*4=12"),
        }
    }

    #[test]
    fn alloc_rejects_rows_cols_overflow() {
        let Some(ctx) = cuda_context_or_skip() else {
            return;
        };

        // `rows * cols` overflows usize. Before the fix, debug builds would
        // panic inside alloc_zeros and release builds would wrap to a tiny
        // length and silently accept the shape.
        match GpuMatrix::<f32>::alloc(ctx, usize::MAX, 2) {
            Err(CudaError::DimensionMismatch(msg)) => assert!(
                msg.contains("overflow"),
                "expected overflow error, got: {msg}"
            ),
            Err(e) => panic!("expected DimensionMismatch(overflow), got {e:?}"),
            Ok(_) => panic!("expected overflow rejection"),
        }
    }

    #[test]
    fn from_cuda_slice_rejects_rows_cols_overflow() {
        let Some(ctx) = cuda_context_or_skip() else {
            return;
        };

        // Allocate any nonzero slice. rows*cols overflows usize, so the error
        // must come from validate_dims_len's checked_mul — NOT from the length
        // comparison (which would otherwise read a wrapped product).
        let slice = ctx.device().alloc_zeros::<f32>(4).unwrap();
        match GpuMatrix::<f32>::from_cuda_slice(ctx, slice, usize::MAX, 2) {
            Err(CudaError::DimensionMismatch(msg)) => {
                assert!(
                    msg.contains("overflow"),
                    "expected overflow error, got: {msg}"
                );
            }
            Err(e) => panic!("expected DimensionMismatch(overflow), got {e:?}"),
            Ok(_) => panic!("expected overflow rejection"),
        }
    }

    #[test]
    fn from_cuda_slices_pairs_matrix_and_argmax() {
        let Some(ctx) = cuda_context_or_skip() else {
            return;
        };

        let mat_host = vec![10.0f32, 20.0, 30.0, 40.0];
        let arg_host: Vec<ArgmaxIndex> = vec![0, 1, 2, 3];

        let mat_slice = ctx.device().htod_sync_copy(&mat_host).unwrap();
        let arg_slice = ctx.device().htod_sync_copy(&arg_host).unwrap();

        let pair =
            GpuMatrixWithArgmax::<f32>::from_cuda_slices(ctx, mat_slice, arg_slice, 2, 2).unwrap();

        assert_eq!(pair.matrix_to_host(ctx).unwrap(), mat_host);
        assert_eq!(pair.argmax_to_host(ctx).unwrap(), arg_host);
    }

    #[test]
    fn from_cuda_slices_rejects_dimension_mismatch_on_either_buffer() {
        let Some(ctx) = cuda_context_or_skip() else {
            return;
        };

        // Matrix slice has wrong length. Error message must attribute the
        // failure to the matrix buffer specifically — a future refactor that
        // collapses both checks or reverses their order would silently break
        // call-site diagnosis.
        let bad_mat = ctx.device().alloc_zeros::<f32>(3).unwrap();
        let good_arg = ctx.device().alloc_zeros::<ArgmaxIndex>(4).unwrap();
        match GpuMatrixWithArgmax::<f32>::from_cuda_slices(ctx, bad_mat, good_arg, 2, 2) {
            Err(CudaError::DimensionMismatch(msg)) => assert!(
                msg.starts_with("matrix:"),
                "expected 'matrix:' attribution, got: {msg}"
            ),
            Err(e) => panic!("expected DimensionMismatch (bad matrix), got {e:?}"),
            Ok(_) => panic!("expected error for matrix len 3 vs 2*2=4"),
        }

        // Argmax slice has wrong length.
        let good_mat = ctx.device().alloc_zeros::<f32>(4).unwrap();
        let bad_arg = ctx.device().alloc_zeros::<ArgmaxIndex>(3).unwrap();
        match GpuMatrixWithArgmax::<f32>::from_cuda_slices(ctx, good_mat, bad_arg, 2, 2) {
            Err(CudaError::DimensionMismatch(msg)) => assert!(
                msg.starts_with("argmax:"),
                "expected 'argmax:' attribution, got: {msg}"
            ),
            Err(e) => panic!("expected DimensionMismatch (bad argmax), got {e:?}"),
            Ok(_) => panic!("expected error for argmax len 3 vs 2*2=4"),
        }
    }
}
