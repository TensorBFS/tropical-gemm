use crate::core::{GemmWithArgmax, Transpose};
use crate::simd::{tropical_gemm_dispatch, KernelDispatch};
use crate::types::{TropicalSemiring, TropicalWithArgmax};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Simple tropical matrix multiplication: C = A ⊗ B
///
/// Computes C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])
///
/// # Arguments
/// - `a`: Matrix A data in row-major order
/// - `m`: Number of rows in A
/// - `k`: Number of columns in A / rows in B
/// - `b`: Matrix B data in row-major order
/// - `n`: Number of columns in B
///
/// # Returns
/// Result matrix C of size m×n in row-major order
///
/// # Example
///
/// ```
/// use tropical_gemm::{tropical_matmul, TropicalMaxPlus};
///
/// let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
/// let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
///
/// let c = tropical_matmul::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2);
/// assert_eq!(c.len(), 4); // 2x2 result
/// ```
pub fn tropical_matmul<T: TropicalSemiring + KernelDispatch>(
    a: &[T::Scalar],
    m: usize,
    k: usize,
    b: &[T::Scalar],
    n: usize,
) -> Vec<T> {
    assert_eq!(a.len(), m * k, "A dimensions mismatch");
    assert_eq!(b.len(), k * n, "B dimensions mismatch");

    let mut c = vec![T::tropical_zero(); m * n];

    unsafe {
        tropical_gemm_dispatch::<T>(
            m,
            n,
            k,
            a.as_ptr(),
            k,
            Transpose::NoTrans,
            b.as_ptr(),
            n,
            Transpose::NoTrans,
            c.as_mut_ptr(),
            n,
        );
    }

    c
}

/// Tropical matrix multiplication with argmax tracking.
///
/// Returns both the result matrix and the argmax indices indicating
/// which k produced each optimal C[i,j].
///
/// # Example
///
/// ```
/// use tropical_gemm::{tropical_matmul_with_argmax, TropicalMaxPlus};
///
/// let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
/// let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
///
/// let result = tropical_matmul_with_argmax::<TropicalMaxPlus<f64>>(&a, 2, 3, &b, 2);
/// assert_eq!(result.m, 2);
/// assert_eq!(result.n, 2);
/// ```
pub fn tropical_matmul_with_argmax<T: TropicalWithArgmax<Index = u32> + KernelDispatch>(
    a: &[T::Scalar],
    m: usize,
    k: usize,
    b: &[T::Scalar],
    n: usize,
) -> GemmWithArgmax<T> {
    assert_eq!(a.len(), m * k, "A dimensions mismatch");
    assert_eq!(b.len(), k * n, "B dimensions mismatch");

    let mut result = GemmWithArgmax::new(m, n);

    unsafe {
        crate::core::tropical_gemm_with_argmax_portable::<T>(
            m,
            n,
            k,
            a.as_ptr(),
            k,
            Transpose::NoTrans,
            b.as_ptr(),
            n,
            Transpose::NoTrans,
            &mut result,
        );
    }

    result
}

/// Builder for configuring tropical GEMM operations.
///
/// Provides a fluent API for setting options like transposition,
/// alpha/beta scaling, and output preferences.
///
/// # Example
///
/// ```
/// use tropical_gemm::{TropicalGemm, TropicalMaxPlus, TropicalSemiring};
///
/// let a = vec![1.0f32; 6]; // 2x3
/// let b = vec![1.0f32; 6]; // 3x2
/// let mut c = vec![TropicalMaxPlus::tropical_zero(); 4]; // 2x2
///
/// TropicalGemm::<TropicalMaxPlus<f32>>::new(2, 2, 3)
///     .execute(&a, 3, &b, 2, &mut c, 2);
/// ```
pub struct TropicalGemm<T: TropicalSemiring> {
    m: usize,
    n: usize,
    k: usize,
    trans_a: Transpose,
    trans_b: Transpose,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: TropicalSemiring + KernelDispatch> TropicalGemm<T> {
    /// Create a new GEMM builder.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            trans_a: Transpose::NoTrans,
            trans_b: Transpose::NoTrans,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Transpose matrix A.
    pub fn trans_a(mut self) -> Self {
        self.trans_a = Transpose::Trans;
        self
    }

    /// Transpose matrix B.
    pub fn trans_b(mut self) -> Self {
        self.trans_b = Transpose::Trans;
        self
    }

    /// Execute the GEMM operation.
    ///
    /// # Arguments
    /// - `a`: Matrix A data
    /// - `lda`: Leading dimension of A
    /// - `b`: Matrix B data
    /// - `ldb`: Leading dimension of B
    /// - `c`: Output matrix C (must be pre-allocated)
    /// - `ldc`: Leading dimension of C
    pub fn execute(
        self,
        a: &[T::Scalar],
        lda: usize,
        b: &[T::Scalar],
        ldb: usize,
        c: &mut [T],
        ldc: usize,
    ) {
        unsafe {
            tropical_gemm_dispatch::<T>(
                self.m,
                self.n,
                self.k,
                a.as_ptr(),
                lda,
                self.trans_a,
                b.as_ptr(),
                ldb,
                self.trans_b,
                c.as_mut_ptr(),
                ldc,
            );
        }
    }
}

/// BLAS-style GEMM interface.
///
/// C = A ⊗ B
///
/// # Safety
/// All pointers must be valid for the specified dimensions.
pub unsafe fn tropical_gemm<T: TropicalSemiring + KernelDispatch>(
    m: usize,
    n: usize,
    k: usize,
    a: *const T::Scalar,
    lda: usize,
    trans_a: Transpose,
    b: *const T::Scalar,
    ldb: usize,
    trans_b: Transpose,
    c: *mut T,
    ldc: usize,
) {
    tropical_gemm_dispatch::<T>(m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc);
}

/// Batched tropical matrix multiplication: C[i] = A[i] ⊗ B[i] for i = 0..batch_size
///
/// All matrices in the batch must have the same dimensions:
/// - Each A[i] is m × k
/// - Each B[i] is k × n
/// - Each C[i] is m × n
///
/// # Arguments
/// - `a_batch`: Slice of batch_size matrices, each of size m×k in row-major order
/// - `b_batch`: Slice of batch_size matrices, each of size k×n in row-major order
/// - `m`: Number of rows in each A matrix
/// - `k`: Number of columns in A / rows in B
/// - `n`: Number of columns in each B matrix
///
/// # Returns
/// Vector of batch_size result matrices, each of size m×n
///
/// # Example
///
/// ```
/// use tropical_gemm::{tropical_matmul_batched, TropicalMaxPlus};
///
/// // Two 2x2 matrix multiplications
/// let a_batch = vec![
///     vec![1.0f32, 2.0, 3.0, 4.0],  // A[0]: 2x2
///     vec![5.0f32, 6.0, 7.0, 8.0],  // A[1]: 2x2
/// ];
/// let b_batch = vec![
///     vec![1.0f32, 2.0, 3.0, 4.0],  // B[0]: 2x2
///     vec![1.0f32, 2.0, 3.0, 4.0],  // B[1]: 2x2
/// ];
///
/// let c_batch = tropical_matmul_batched::<TropicalMaxPlus<f32>>(&a_batch, &b_batch, 2, 2, 2);
/// assert_eq!(c_batch.len(), 2);
/// ```
pub fn tropical_matmul_batched<T: TropicalSemiring + KernelDispatch>(
    a_batch: &[Vec<T::Scalar>],
    b_batch: &[Vec<T::Scalar>],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Vec<T>>
where
    T::Scalar: Send + Sync,
    T: Send + Sync,
{
    assert_eq!(
        a_batch.len(),
        b_batch.len(),
        "Batch sizes must match: A has {} matrices, B has {}",
        a_batch.len(),
        b_batch.len()
    );

    let batch_size = a_batch.len();
    if batch_size == 0 {
        return Vec::new();
    }

    // Validate dimensions
    for (i, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
        assert_eq!(
            a.len(),
            m * k,
            "A[{}] dimensions mismatch: expected {}, got {}",
            i,
            m * k,
            a.len()
        );
        assert_eq!(
            b.len(),
            k * n,
            "B[{}] dimensions mismatch: expected {}, got {}",
            i,
            k * n,
            b.len()
        );
    }

    #[cfg(feature = "parallel")]
    {
        a_batch
            .par_iter()
            .zip(b_batch.par_iter())
            .map(|(a, b)| tropical_matmul::<T>(a, m, k, b, n))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        a_batch
            .iter()
            .zip(b_batch.iter())
            .map(|(a, b)| tropical_matmul::<T>(a, m, k, b, n))
            .collect()
    }
}

/// Batched tropical matrix multiplication with argmax tracking.
///
/// C[i] = A[i] ⊗ B[i] for i = 0..batch_size, with argmax indices.
///
/// # Arguments
/// - `a_batch`: Slice of batch_size matrices, each of size m×k
/// - `b_batch`: Slice of batch_size matrices, each of size k×n
/// - `m`: Number of rows in each A matrix
/// - `k`: Number of columns in A / rows in B
/// - `n`: Number of columns in each B matrix
///
/// # Returns
/// Vector of batch_size GemmWithArgmax results
pub fn tropical_matmul_batched_with_argmax<T: TropicalWithArgmax<Index = u32> + KernelDispatch>(
    a_batch: &[Vec<T::Scalar>],
    b_batch: &[Vec<T::Scalar>],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<GemmWithArgmax<T>>
where
    T::Scalar: Send + Sync,
    T: Send + Sync,
{
    assert_eq!(
        a_batch.len(),
        b_batch.len(),
        "Batch sizes must match: A has {} matrices, B has {}",
        a_batch.len(),
        b_batch.len()
    );

    let batch_size = a_batch.len();
    if batch_size == 0 {
        return Vec::new();
    }

    // Validate dimensions
    for (i, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
        assert_eq!(
            a.len(),
            m * k,
            "A[{}] dimensions mismatch: expected {}, got {}",
            i,
            m * k,
            a.len()
        );
        assert_eq!(
            b.len(),
            k * n,
            "B[{}] dimensions mismatch: expected {}, got {}",
            i,
            k * n,
            b.len()
        );
    }

    #[cfg(feature = "parallel")]
    {
        a_batch
            .par_iter()
            .zip(b_batch.par_iter())
            .map(|(a, b)| tropical_matmul_with_argmax::<T>(a, m, k, b, n))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        a_batch
            .iter()
            .zip(b_batch.iter())
            .map(|(a, b)| tropical_matmul_with_argmax::<T>(a, m, k, b, n))
            .collect()
    }
}

/// Strided batched GEMM: computes C[i] = A[i] ⊗ B[i] from contiguous memory.
///
/// This is more efficient than `tropical_matmul_batched` when all matrices
/// are stored contiguously in memory with fixed strides.
///
/// # Arguments
/// - `a`: Contiguous array of all A matrices (batch_size × m × k elements)
/// - `b`: Contiguous array of all B matrices (batch_size × k × n elements)
/// - `batch_size`: Number of matrix pairs
/// - `m`: Rows in each A
/// - `k`: Columns in A / rows in B
/// - `n`: Columns in each B
///
/// # Returns
/// Contiguous array of all C matrices (batch_size × m × n elements)
///
/// # Example
///
/// ```
/// use tropical_gemm::{tropical_matmul_strided_batched, TropicalMaxPlus};
///
/// // Two 2x2 matrix pairs stored contiguously
/// let a = vec![
///     1.0f32, 2.0, 3.0, 4.0,  // A[0]
///     5.0, 6.0, 7.0, 8.0,      // A[1]
/// ];
/// let b = vec![
///     1.0f32, 2.0, 3.0, 4.0,  // B[0]
///     1.0, 2.0, 3.0, 4.0,      // B[1]
/// ];
///
/// let c = tropical_matmul_strided_batched::<TropicalMaxPlus<f32>>(&a, &b, 2, 2, 2, 2);
/// assert_eq!(c.len(), 8); // 2 batches × 2×2 results
/// ```
pub fn tropical_matmul_strided_batched<T: TropicalSemiring + KernelDispatch>(
    a: &[T::Scalar],
    b: &[T::Scalar],
    batch_size: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T>
where
    T::Scalar: Send + Sync + Copy,
    T: Send + Sync,
{
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    assert_eq!(
        a.len(),
        batch_size * a_stride,
        "A size mismatch: expected {}, got {}",
        batch_size * a_stride,
        a.len()
    );
    assert_eq!(
        b.len(),
        batch_size * b_stride,
        "B size mismatch: expected {}, got {}",
        batch_size * b_stride,
        b.len()
    );

    if batch_size == 0 {
        return Vec::new();
    }

    let mut c = vec![T::tropical_zero(); batch_size * c_stride];

    #[cfg(feature = "parallel")]
    {
        c.par_chunks_mut(c_stride)
            .enumerate()
            .for_each(|(i, c_chunk)| {
                let a_slice = &a[i * a_stride..(i + 1) * a_stride];
                let b_slice = &b[i * b_stride..(i + 1) * b_stride];

                unsafe {
                    tropical_gemm_dispatch::<T>(
                        m,
                        n,
                        k,
                        a_slice.as_ptr(),
                        k,
                        Transpose::NoTrans,
                        b_slice.as_ptr(),
                        n,
                        Transpose::NoTrans,
                        c_chunk.as_mut_ptr(),
                        n,
                    );
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..batch_size {
            let a_slice = &a[i * a_stride..(i + 1) * a_stride];
            let b_slice = &b[i * b_stride..(i + 1) * b_stride];
            let c_slice = &mut c[i * c_stride..(i + 1) * c_stride];

            unsafe {
                tropical_gemm_dispatch::<T>(
                    m,
                    n,
                    k,
                    a_slice.as_ptr(),
                    k,
                    Transpose::NoTrans,
                    b_slice.as_ptr(),
                    n,
                    Transpose::NoTrans,
                    c_slice.as_mut_ptr(),
                    n,
                );
            }
        }
    }

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TropicalMaxPlus;

    #[test]
    fn test_tropical_matmul() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];

        let c = tropical_matmul::<TropicalMaxPlus<f64>>(&a, 2, 3, &b, 2);

        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert_eq!(c[0].0, 8.0);
        // C[0,1] = max(1+2, 2+4, 3+6) = 9
        assert_eq!(c[1].0, 9.0);
        // C[1,0] = max(4+1, 5+3, 6+5) = 11
        assert_eq!(c[2].0, 11.0);
        // C[1,1] = max(4+2, 5+4, 6+6) = 12
        assert_eq!(c[3].0, 12.0);
    }

    #[test]
    fn test_tropical_matmul_with_argmax() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];

        let result = tropical_matmul_with_argmax::<TropicalMaxPlus<f64>>(&a, 2, 3, &b, 2);

        assert_eq!(result.get(0, 0).0, 8.0);
        assert_eq!(result.get_argmax(0, 0), 2); // k=2 produced max

        assert_eq!(result.get(1, 1).0, 12.0);
        assert_eq!(result.get_argmax(1, 1), 2); // k=2 produced max
    }

    #[test]
    fn test_builder_api() {
        let a = vec![1.0f32; 6];
        let b = vec![1.0f32; 6];
        let mut c = vec![TropicalMaxPlus::tropical_zero(); 4];

        TropicalGemm::<TropicalMaxPlus<f32>>::new(2, 2, 3).execute(&a, 3, &b, 2, &mut c, 2);

        // C[0,0] = max(1+1, 1+1, 1+1) = 2 (tropical mul is addition, tropical add is max)
        assert_eq!(c[0].0, 2.0);
    }

    #[test]
    fn test_builder_api_trans_a() {
        // A is 3x2 stored as column-major (actually 2x3 in row-major transposed)
        // A^T is 2x3, B is 3x2, result is 2x2
        let a = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]; // col-major 3x2
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // row-major 3x2
        let mut c = vec![TropicalMaxPlus::tropical_zero(); 4];

        TropicalGemm::<TropicalMaxPlus<f32>>::new(2, 2, 3)
            .trans_a()
            .execute(&a, 2, &b, 2, &mut c, 2);

        // A^T = [[1, 2, 3], [4, 5, 6]]
        // B = [[1, 2], [3, 4], [5, 6]]
        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert_eq!(c[0].0, 8.0);
    }

    #[test]
    fn test_builder_api_trans_b() {
        // A is 2x3, B^T is 2x3 stored as column-major, result is 2x2
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // row-major 2x3
        let b = vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0]; // col-major 2x3
        let mut c = vec![TropicalMaxPlus::tropical_zero(); 4];

        TropicalGemm::<TropicalMaxPlus<f32>>::new(2, 2, 3)
            .trans_b()
            .execute(&a, 3, &b, 3, &mut c, 2);

        // A = [[1, 2, 3], [4, 5, 6]]
        // B^T = [[1, 2], [3, 4], [5, 6]]
        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert_eq!(c[0].0, 8.0);
    }

    #[test]
    fn test_tropical_matmul_min_plus() {
        use crate::types::TropicalMinPlus;

        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];

        let c = tropical_matmul::<TropicalMinPlus<f64>>(&a, 2, 3, &b, 2);

        // C[0,0] = min(1+1, 2+3, 3+5) = 2
        assert_eq!(c[0].0, 2.0);
        // C[0,1] = min(1+2, 2+4, 3+6) = 3
        assert_eq!(c[1].0, 3.0);
        // C[1,0] = min(4+1, 5+3, 6+5) = 5
        assert_eq!(c[2].0, 5.0);
        // C[1,1] = min(4+2, 5+4, 6+6) = 6
        assert_eq!(c[3].0, 6.0);
    }

    #[test]
    fn test_tropical_matmul_max_mul() {
        use crate::types::TropicalMaxMul;

        let a = vec![2.0f64, 3.0, 4.0, 5.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0];

        let c = tropical_matmul::<TropicalMaxMul<f64>>(&a, 2, 2, &b, 2);

        // C[0,0] = max(2*1, 3*3) = max(2, 9) = 9
        assert_eq!(c[0].0, 9.0);
        // C[0,1] = max(2*2, 3*4) = max(4, 12) = 12
        assert_eq!(c[1].0, 12.0);
        // C[1,0] = max(4*1, 5*3) = max(4, 15) = 15
        assert_eq!(c[2].0, 15.0);
        // C[1,1] = max(4*2, 5*4) = max(8, 20) = 20
        assert_eq!(c[3].0, 20.0);
    }

    #[test]
    fn test_tropical_matmul_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let c = tropical_matmul::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2);

        assert!((c[0].0 - 8.0).abs() < 1e-6);
        assert!((c[1].0 - 9.0).abs() < 1e-6);
        assert!((c[2].0 - 11.0).abs() < 1e-6);
        assert!((c[3].0 - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_non_square_matrices() {
        // 3x2 * 2x4 = 3x4
        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let c = tropical_matmul::<TropicalMaxPlus<f64>>(&a, 3, 2, &b, 4);

        assert_eq!(c.len(), 12);
        // C[0,0] = max(1+1, 2+5) = 7
        assert_eq!(c[0].0, 7.0);
    }

    #[test]
    fn test_single_element() {
        let a = vec![5.0f64];
        let b = vec![3.0f64];

        let c = tropical_matmul::<TropicalMaxPlus<f64>>(&a, 1, 1, &b, 1);

        assert_eq!(c.len(), 1);
        assert_eq!(c[0].0, 8.0); // 5 + 3 = 8
    }

    #[test]
    fn test_larger_matrix() {
        let n = 16;
        let a: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..n * n).map(|i| (n * n - 1 - i) as f64).collect();

        let c = tropical_matmul::<TropicalMaxPlus<f64>>(&a, n, n, &b, n);

        assert_eq!(c.len(), n * n);
        // Just verify it doesn't panic and produces reasonable results
        for val in &c {
            assert!(val.0.is_finite());
        }
    }

    #[test]
    fn test_tropical_matmul_i32() {
        let a = vec![1i32, 2, 3, 4, 5, 6];
        let b = vec![1i32, 2, 3, 4, 5, 6];

        let c = tropical_matmul::<TropicalMaxPlus<i32>>(&a, 2, 3, &b, 2);

        assert_eq!(c[0].0, 8);
        assert_eq!(c[1].0, 9);
        assert_eq!(c[2].0, 11);
        assert_eq!(c[3].0, 12);
    }

    #[test]
    fn test_tropical_matmul_i64() {
        let a = vec![1i64, 2, 3, 4, 5, 6];
        let b = vec![1i64, 2, 3, 4, 5, 6];

        let c = tropical_matmul::<TropicalMaxPlus<i64>>(&a, 2, 3, &b, 2);

        assert_eq!(c[0].0, 8);
        assert_eq!(c[1].0, 9);
        assert_eq!(c[2].0, 11);
        assert_eq!(c[3].0, 12);
    }

    #[test]
    fn test_tropical_matmul_minplus_i32() {
        use crate::types::TropicalMinPlus;

        let a = vec![1i32, 2, 3, 4, 5, 6];
        let b = vec![1i32, 2, 3, 4, 5, 6];

        let c = tropical_matmul::<TropicalMinPlus<i32>>(&a, 2, 3, &b, 2);

        assert_eq!(c[0].0, 2);
        assert_eq!(c[1].0, 3);
        assert_eq!(c[2].0, 5);
        assert_eq!(c[3].0, 6);
    }

    #[test]
    fn test_unsafe_tropical_gemm() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![TropicalMaxPlus::tropical_zero(); 4];

        unsafe {
            tropical_gemm::<TropicalMaxPlus<f64>>(
                2,
                2,
                3,
                a.as_ptr(),
                3,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c[0].0, 8.0);
        assert_eq!(c[1].0, 9.0);
        assert_eq!(c[2].0, 11.0);
        assert_eq!(c[3].0, 12.0);
    }

    #[test]
    fn test_minplus_with_argmax() {
        use crate::types::TropicalMinPlus;

        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];

        let result = tropical_matmul_with_argmax::<TropicalMinPlus<f64>>(&a, 2, 3, &b, 2);

        // C[0,0] = min(1+1, 2+3, 3+5) = 2 at k=0
        assert_eq!(result.get(0, 0).0, 2.0);
        assert_eq!(result.get_argmax(0, 0), 0);

        // C[1,1] = min(4+2, 5+4, 6+6) = 6 at k=0
        assert_eq!(result.get(1, 1).0, 6.0);
        assert_eq!(result.get_argmax(1, 1), 0);
    }

    #[test]
    fn test_maxmul_with_argmax() {
        use crate::types::TropicalMaxMul;

        let a = vec![2.0f64, 3.0, 4.0, 5.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0];

        let result = tropical_matmul_with_argmax::<TropicalMaxMul<f64>>(&a, 2, 2, &b, 2);

        // C[0,0] = max(2*1, 3*3) = 9 at k=1
        assert_eq!(result.get(0, 0).0, 9.0);
        assert_eq!(result.get_argmax(0, 0), 1);
    }

    #[test]
    fn test_gemmwithargmax_dimensions() {
        let a = vec![1.0f64; 12]; // 3x4
        let b = vec![1.0f64; 20]; // 4x5

        let result = tropical_matmul_with_argmax::<TropicalMaxPlus<f64>>(&a, 3, 4, &b, 5);

        assert_eq!(result.m, 3);
        assert_eq!(result.n, 5);
        assert_eq!(result.values.len(), 15);
        assert_eq!(result.argmax.len(), 15);
    }

    #[test]
    fn test_identity_like_matrix() {
        // Matrix with -inf everywhere except diagonal has 0
        let a = vec![0.0f64, f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0];

        let c = tropical_matmul::<TropicalMaxPlus<f64>>(&a, 2, 2, &b, 2);

        // With "identity" A, C should equal B
        assert_eq!(c[0].0, 1.0);
        assert_eq!(c[1].0, 2.0);
        assert_eq!(c[2].0, 3.0);
        assert_eq!(c[3].0, 4.0);
    }

    #[test]
    fn test_tropical_matmul_batched() {
        let a_batch = vec![
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], // 2x3
            vec![2.0f64, 3.0, 4.0, 5.0, 6.0, 7.0], // 2x3
        ];
        let b_batch = vec![
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], // 3x2
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], // 3x2
        ];

        let c_batch = tropical_matmul_batched::<TropicalMaxPlus<f64>>(&a_batch, &b_batch, 2, 3, 2);

        assert_eq!(c_batch.len(), 2);

        // C[0][0,0] = max(1+1, 2+3, 3+5) = 8
        assert_eq!(c_batch[0][0].0, 8.0);
        // C[0][1,1] = max(4+2, 5+4, 6+6) = 12
        assert_eq!(c_batch[0][3].0, 12.0);

        // C[1][0,0] = max(2+1, 3+3, 4+5) = 9
        assert_eq!(c_batch[1][0].0, 9.0);
        // C[1][1,1] = max(5+2, 6+4, 7+6) = 13
        assert_eq!(c_batch[1][3].0, 13.0);
    }

    #[test]
    fn test_tropical_matmul_batched_empty() {
        let a_batch: Vec<Vec<f64>> = vec![];
        let b_batch: Vec<Vec<f64>> = vec![];

        let c_batch = tropical_matmul_batched::<TropicalMaxPlus<f64>>(&a_batch, &b_batch, 2, 2, 2);

        assert!(c_batch.is_empty());
    }

    #[test]
    fn test_tropical_matmul_batched_with_argmax() {
        let a_batch = vec![
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], // 2x3
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], // 2x3
        ];
        let b_batch = vec![
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],  // 3x2
            vec![10.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], // 3x2 (different first element)
        ];

        let results = tropical_matmul_batched_with_argmax::<TropicalMaxPlus<f64>>(
            &a_batch, &b_batch, 2, 3, 2,
        );

        assert_eq!(results.len(), 2);

        // First batch: C[0,0] = max(1+1, 2+3, 3+5) = 8 at k=2
        assert_eq!(results[0].get(0, 0).0, 8.0);
        assert_eq!(results[0].get_argmax(0, 0), 2);

        // Second batch: C[0,0] = max(1+10, 2+3, 3+5) = 11 at k=0
        assert_eq!(results[1].get(0, 0).0, 11.0);
        assert_eq!(results[1].get_argmax(0, 0), 0);
    }

    #[test]
    fn test_tropical_matmul_batched_with_argmax_empty() {
        let a_batch: Vec<Vec<f64>> = vec![];
        let b_batch: Vec<Vec<f64>> = vec![];

        let results = tropical_matmul_batched_with_argmax::<TropicalMaxPlus<f64>>(
            &a_batch, &b_batch, 2, 2, 2,
        );

        assert!(results.is_empty());
    }

    #[test]
    fn test_tropical_matmul_strided_batched() {
        // Two 2x2 matrices stored contiguously
        let a = vec![
            1.0f64, 2.0, 3.0, 4.0, // A[0]
            5.0, 6.0, 7.0, 8.0, // A[1]
        ];
        let b = vec![
            1.0f64, 2.0, 3.0, 4.0, // B[0]
            1.0, 2.0, 3.0, 4.0, // B[1]
        ];

        let c = tropical_matmul_strided_batched::<TropicalMaxPlus<f64>>(&a, &b, 2, 2, 2, 2);

        assert_eq!(c.len(), 8);

        // C[0][0,0] = max(1+1, 2+3) = 5
        assert_eq!(c[0].0, 5.0);
        // C[0][1,1] = max(3+2, 4+4) = 8
        assert_eq!(c[3].0, 8.0);

        // C[1][0,0] = max(5+1, 6+3) = 9
        assert_eq!(c[4].0, 9.0);
        // C[1][1,1] = max(7+2, 8+4) = 12
        assert_eq!(c[7].0, 12.0);
    }

    #[test]
    fn test_tropical_matmul_strided_batched_empty() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];

        let c = tropical_matmul_strided_batched::<TropicalMaxPlus<f64>>(&a, &b, 0, 2, 2, 2);

        assert!(c.is_empty());
    }

    #[test]
    fn test_tropical_matmul_strided_batched_minplus() {
        use crate::types::TropicalMinPlus;

        let a = vec![
            1.0f64, 2.0, 3.0, 4.0, // A[0]
            5.0, 6.0, 7.0, 8.0, // A[1]
        ];
        let b = vec![
            1.0f64, 2.0, 3.0, 4.0, // B[0]
            1.0, 2.0, 3.0, 4.0, // B[1]
        ];

        let c = tropical_matmul_strided_batched::<TropicalMinPlus<f64>>(&a, &b, 2, 2, 2, 2);

        assert_eq!(c.len(), 8);

        // C[0][0,0] = min(1+1, 2+3) = 2
        assert_eq!(c[0].0, 2.0);
        // C[0][1,1] = min(3+2, 4+4) = 5
        assert_eq!(c[3].0, 5.0);
    }

    #[test]
    fn test_tropical_matmul_batched_larger() {
        let batch_size = 10;
        let m = 8;
        let k = 6;
        let n = 4;

        let a_batch: Vec<Vec<f64>> = (0..batch_size)
            .map(|i| (0..m * k).map(|j| (i * m * k + j) as f64).collect())
            .collect();
        let b_batch: Vec<Vec<f64>> = (0..batch_size)
            .map(|_| (0..k * n).map(|j| j as f64).collect())
            .collect();

        let c_batch = tropical_matmul_batched::<TropicalMaxPlus<f64>>(&a_batch, &b_batch, m, k, n);

        assert_eq!(c_batch.len(), batch_size);
        for c in &c_batch {
            assert_eq!(c.len(), m * n);
            // Just verify all values are finite
            for val in c {
                assert!(val.0.is_finite());
            }
        }
    }
}
