//! Python bindings for tropical matrix multiplication.
//!
//! This module provides Python/NumPy bindings for tropical GEMM operations,
//! enabling integration with PyTorch custom autograd functions.
//!
//! ## Features
//!
//! - `cuda`: Enable GPU acceleration via CUDA

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;

// Use fully qualified path to avoid naming conflict with the pymodule
use ::tropical_gemm::{
    tropical_matmul, tropical_matmul_with_argmax, GemmWithArgmax, TropicalMaxMul, TropicalMaxPlus,
    TropicalMinPlus, TropicalSemiring,
};

/// Tropical MaxPlus matrix multiplication: C[i,j] = max_k(A[i,k] + B[k,j])
///
/// Args:
///     a: Input matrix A of shape (M, K)
///     b: Input matrix B of shape (K, N)
///
/// Returns:
///     Result matrix C of shape (M, N) as a flattened array
#[pyfunction]
fn maxplus_matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    // Get contiguous data
    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    // Perform tropical matmul
    let c_data = tropical_matmul::<TropicalMaxPlus<f32>>(a_data, m, k, b_data, n);

    // Extract scalar values from semiring wrapper
    let c_scalars: Vec<f32> = c_data.iter().map(|x| x.value()).collect();

    // Create output array
    Ok(c_scalars.into_pyarray(py))
}

/// Tropical MinPlus matrix multiplication: C[i,j] = min_k(A[i,k] + B[k,j])
///
/// Args:
///     a: Input matrix A of shape (M, K)
///     b: Input matrix B of shape (K, N)
///
/// Returns:
///     Result matrix C of shape (M, N) as a flattened array
#[pyfunction]
fn minplus_matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let c_data = tropical_matmul::<TropicalMinPlus<f32>>(a_data, m, k, b_data, n);

    let c_scalars: Vec<f32> = c_data.iter().map(|x| x.value()).collect();

    Ok(c_scalars.into_pyarray(py))
}

/// Tropical MaxPlus matrix multiplication with argmax tracking for backpropagation.
///
/// Args:
///     a: Input matrix A of shape (M, K)
///     b: Input matrix B of shape (K, N)
///
/// Returns:
///     Tuple of (C, argmax) where:
///     - C: Result matrix of shape (M, N) as flattened array
///     - argmax: Indices of shape (M, N) as flattened array where argmax[i*N+j] = k
#[pyfunction]
fn maxplus_matmul_with_argmax<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let result: GemmWithArgmax<TropicalMaxPlus<f32>> =
        tropical_matmul_with_argmax::<TropicalMaxPlus<f32>>(a_data, m, k, b_data, n);

    let c_scalars: Vec<f32> = result.values.iter().map(|x| x.value()).collect();
    let argmax_i32: Vec<i32> = result.argmax.iter().map(|&x| x as i32).collect();

    let c_result = c_scalars.into_pyarray(py);
    let argmax_result = argmax_i32.into_pyarray(py);

    Ok((c_result, argmax_result))
}

/// Tropical MinPlus matrix multiplication with argmax tracking for backpropagation.
///
/// Args:
///     a: Input matrix A of shape (M, K)
///     b: Input matrix B of shape (K, N)
///
/// Returns:
///     Tuple of (C, argmax) where:
///     - C: Result matrix of shape (M, N) as flattened array
///     - argmax: Indices of shape (M, N) as flattened array
#[pyfunction]
fn minplus_matmul_with_argmax<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let result: GemmWithArgmax<TropicalMinPlus<f32>> =
        tropical_matmul_with_argmax::<TropicalMinPlus<f32>>(a_data, m, k, b_data, n);

    let c_scalars: Vec<f32> = result.values.iter().map(|x| x.value()).collect();
    let argmax_i32: Vec<i32> = result.argmax.iter().map(|&x| x as i32).collect();

    let c_result = c_scalars.into_pyarray(py);
    let argmax_result = argmax_i32.into_pyarray(py);

    Ok((c_result, argmax_result))
}

/// Compute gradient with respect to matrix A for tropical matmul backward pass.
///
/// Given grad_c (gradient w.r.t. output C) and argmax indices from forward pass,
/// computes grad_a where: grad_a[i,k] = sum_j { grad_c[i,j] if argmax[i,j] == k }
///
/// Args:
///     grad_c: Gradient w.r.t. C of shape (M, N) as flattened array
///     argmax: Argmax indices from forward pass of shape (M, N) as flattened array
///     m: Number of rows in C
///     n: Number of columns in C
///     k: Number of columns in A (inner dimension)
///
/// Returns:
///     Gradient w.r.t. A of shape (M, K) as flattened array
#[pyfunction]
fn backward_a<'py>(
    py: Python<'py>,
    grad_c: PyReadonlyArray2<'py, f32>,
    argmax: PyReadonlyArray2<'py, i32>,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = grad_c.shape();
    let m = shape[0];
    let n = shape[1];

    let grad_c_data = grad_c.as_slice()?;
    let argmax_data = argmax.as_slice()?;

    // Compute gradient w.r.t. A
    let mut grad_a = vec![0.0f32; m * k];

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let k_idx = argmax_data[idx] as usize;
            if k_idx < k {
                grad_a[i * k + k_idx] += grad_c_data[idx];
            }
        }
    }

    Ok(grad_a.into_pyarray(py))
}

/// Compute gradient with respect to matrix B for tropical matmul backward pass.
///
/// Given grad_c (gradient w.r.t. output C) and argmax indices from forward pass,
/// computes grad_b where: grad_b[k,j] = sum_i { grad_c[i,j] if argmax[i,j] == k }
///
/// Args:
///     grad_c: Gradient w.r.t. C of shape (M, N) as flattened array
///     argmax: Argmax indices from forward pass of shape (M, N) as flattened array
///     m: Number of rows in C
///     n: Number of columns in C
///     k: Number of rows in B (inner dimension)
///
/// Returns:
///     Gradient w.r.t. B of shape (K, N) as flattened array
#[pyfunction]
fn backward_b<'py>(
    py: Python<'py>,
    grad_c: PyReadonlyArray2<'py, f32>,
    argmax: PyReadonlyArray2<'py, i32>,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = grad_c.shape();
    let m = shape[0];
    let n = shape[1];

    let grad_c_data = grad_c.as_slice()?;
    let argmax_data = argmax.as_slice()?;

    // Compute gradient w.r.t. B
    let mut grad_b = vec![0.0f32; k * n];

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let k_idx = argmax_data[idx] as usize;
            if k_idx < k {
                grad_b[k_idx * n + j] += grad_c_data[idx];
            }
        }
    }

    Ok(grad_b.into_pyarray(py))
}

// ============================================================================
// MaxMul operations (f32)
// ============================================================================

/// Tropical MaxMul matrix multiplication: C[i,j] = max_k(A[i,k] * B[k,j])
#[pyfunction]
fn maxmul_matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let c_data = tropical_matmul::<TropicalMaxMul<f32>>(a_data, m, k, b_data, n);
    let c_scalars: Vec<f32> = c_data.iter().map(|x| x.value()).collect();

    Ok(c_scalars.into_pyarray(py))
}

/// Tropical MaxMul matrix multiplication with argmax tracking.
#[pyfunction]
fn maxmul_matmul_with_argmax<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let result: GemmWithArgmax<TropicalMaxMul<f32>> =
        tropical_matmul_with_argmax::<TropicalMaxMul<f32>>(a_data, m, k, b_data, n);

    let c_scalars: Vec<f32> = result.values.iter().map(|x| x.value()).collect();
    let argmax_i32: Vec<i32> = result.argmax.iter().map(|&x| x as i32).collect();

    Ok((c_scalars.into_pyarray(py), argmax_i32.into_pyarray(py)))
}

// ============================================================================
// f64 operations
// ============================================================================

/// Tropical MaxPlus matrix multiplication (f64): C[i,j] = max_k(A[i,k] + B[k,j])
#[pyfunction]
fn maxplus_matmul_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let c_data = tropical_matmul::<TropicalMaxPlus<f64>>(a_data, m, k, b_data, n);
    let c_scalars: Vec<f64> = c_data.iter().map(|x| x.value()).collect();

    Ok(c_scalars.into_pyarray(py))
}

/// Tropical MinPlus matrix multiplication (f64): C[i,j] = min_k(A[i,k] + B[k,j])
#[pyfunction]
fn minplus_matmul_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let c_data = tropical_matmul::<TropicalMinPlus<f64>>(a_data, m, k, b_data, n);
    let c_scalars: Vec<f64> = c_data.iter().map(|x| x.value()).collect();

    Ok(c_scalars.into_pyarray(py))
}

/// Tropical MaxMul matrix multiplication (f64): C[i,j] = max_k(A[i,k] * B[k,j])
#[pyfunction]
fn maxmul_matmul_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let c_data = tropical_matmul::<TropicalMaxMul<f64>>(a_data, m, k, b_data, n);
    let c_scalars: Vec<f64> = c_data.iter().map(|x| x.value()).collect();

    Ok(c_scalars.into_pyarray(py))
}

/// Tropical MaxPlus matrix multiplication with argmax tracking (f64).
#[pyfunction]
fn maxplus_matmul_with_argmax_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let result: GemmWithArgmax<TropicalMaxPlus<f64>> =
        tropical_matmul_with_argmax::<TropicalMaxPlus<f64>>(a_data, m, k, b_data, n);

    let c_scalars: Vec<f64> = result.values.iter().map(|x| x.value()).collect();
    let argmax_i32: Vec<i32> = result.argmax.iter().map(|&x| x as i32).collect();

    Ok((c_scalars.into_pyarray(py), argmax_i32.into_pyarray(py)))
}

/// Tropical MinPlus matrix multiplication with argmax tracking (f64).
#[pyfunction]
fn minplus_matmul_with_argmax_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let result: GemmWithArgmax<TropicalMinPlus<f64>> =
        tropical_matmul_with_argmax::<TropicalMinPlus<f64>>(a_data, m, k, b_data, n);

    let c_scalars: Vec<f64> = result.values.iter().map(|x| x.value()).collect();
    let argmax_i32: Vec<i32> = result.argmax.iter().map(|&x| x as i32).collect();

    Ok((c_scalars.into_pyarray(py), argmax_i32.into_pyarray(py)))
}

/// Tropical MaxMul matrix multiplication with argmax tracking (f64).
#[pyfunction]
fn maxmul_matmul_with_argmax_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let result: GemmWithArgmax<TropicalMaxMul<f64>> =
        tropical_matmul_with_argmax::<TropicalMaxMul<f64>>(a_data, m, k, b_data, n);

    let c_scalars: Vec<f64> = result.values.iter().map(|x| x.value()).collect();
    let argmax_i32: Vec<i32> = result.argmax.iter().map(|&x| x as i32).collect();

    Ok((c_scalars.into_pyarray(py), argmax_i32.into_pyarray(py)))
}

/// Compute gradient with respect to matrix A (f64).
#[pyfunction]
fn backward_a_f64<'py>(
    py: Python<'py>,
    grad_c: PyReadonlyArray2<'py, f64>,
    argmax: PyReadonlyArray2<'py, i32>,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = grad_c.shape();
    let m = shape[0];
    let n = shape[1];

    let grad_c_data = grad_c.as_slice()?;
    let argmax_data = argmax.as_slice()?;

    let mut grad_a = vec![0.0f64; m * k];

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let k_idx = argmax_data[idx] as usize;
            if k_idx < k {
                grad_a[i * k + k_idx] += grad_c_data[idx];
            }
        }
    }

    Ok(grad_a.into_pyarray(py))
}

/// Compute gradient with respect to matrix B (f64).
#[pyfunction]
fn backward_b_f64<'py>(
    py: Python<'py>,
    grad_c: PyReadonlyArray2<'py, f64>,
    argmax: PyReadonlyArray2<'py, i32>,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = grad_c.shape();
    let m = shape[0];
    let n = shape[1];

    let grad_c_data = grad_c.as_slice()?;
    let argmax_data = argmax.as_slice()?;

    let mut grad_b = vec![0.0f64; k * n];

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let k_idx = argmax_data[idx] as usize;
            if k_idx < k {
                grad_b[k_idx * n + j] += grad_c_data[idx];
            }
        }
    }

    Ok(grad_b.into_pyarray(py))
}

// ============================================================================
// MaxMul backward (different from MaxPlus/MinPlus because multiplication)
// For MaxMul: C[i,j] = max_k(A[i,k] * B[k,j])
// grad_A[i,k] = sum_j { grad_C[i,j] * B[k,j] if argmax[i,j] == k }
// grad_B[k,j] = sum_i { grad_C[i,j] * A[i,k] if argmax[i,j] == k }
// ============================================================================

/// Compute MaxMul gradient with respect to matrix A (f32).
///
/// For MaxMul: C[i,j] = max_k(A[i,k] * B[k,j])
/// grad_A[i,k] = sum_j { grad_C[i,j] * B[k,j] if argmax[i,j] == k }
#[pyfunction]
fn maxmul_backward_a<'py>(
    py: Python<'py>,
    grad_c: PyReadonlyArray2<'py, f32>,
    argmax: PyReadonlyArray2<'py, i32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = grad_c.shape();
    let m = shape[0];
    let n = shape[1];
    let k = b.shape()[0];

    let grad_c_data = grad_c.as_slice()?;
    let argmax_data = argmax.as_slice()?;
    let b_data = b.as_slice()?;

    let mut grad_a = vec![0.0f32; m * k];

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let k_idx = argmax_data[idx] as usize;
            if k_idx < k {
                // grad_A[i,k] += grad_C[i,j] * B[k,j]
                grad_a[i * k + k_idx] += grad_c_data[idx] * b_data[k_idx * n + j];
            }
        }
    }

    Ok(grad_a.into_pyarray(py))
}

/// Compute MaxMul gradient with respect to matrix B (f32).
///
/// For MaxMul: C[i,j] = max_k(A[i,k] * B[k,j])
/// grad_B[k,j] = sum_i { grad_C[i,j] * A[i,k] if argmax[i,j] == k }
#[pyfunction]
fn maxmul_backward_b<'py>(
    py: Python<'py>,
    grad_c: PyReadonlyArray2<'py, f32>,
    argmax: PyReadonlyArray2<'py, i32>,
    a: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = grad_c.shape();
    let m = shape[0];
    let n = shape[1];
    let k = a.shape()[1];

    let grad_c_data = grad_c.as_slice()?;
    let argmax_data = argmax.as_slice()?;
    let a_data = a.as_slice()?;

    let mut grad_b = vec![0.0f32; k * n];

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let k_idx = argmax_data[idx] as usize;
            if k_idx < k {
                // grad_B[k,j] += grad_C[i,j] * A[i,k]
                grad_b[k_idx * n + j] += grad_c_data[idx] * a_data[i * k + k_idx];
            }
        }
    }

    Ok(grad_b.into_pyarray(py))
}

/// Compute MaxMul gradient with respect to matrix A (f64).
#[pyfunction]
fn maxmul_backward_a_f64<'py>(
    py: Python<'py>,
    grad_c: PyReadonlyArray2<'py, f64>,
    argmax: PyReadonlyArray2<'py, i32>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = grad_c.shape();
    let m = shape[0];
    let n = shape[1];
    let k = b.shape()[0];

    let grad_c_data = grad_c.as_slice()?;
    let argmax_data = argmax.as_slice()?;
    let b_data = b.as_slice()?;

    let mut grad_a = vec![0.0f64; m * k];

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let k_idx = argmax_data[idx] as usize;
            if k_idx < k {
                grad_a[i * k + k_idx] += grad_c_data[idx] * b_data[k_idx * n + j];
            }
        }
    }

    Ok(grad_a.into_pyarray(py))
}

/// Compute MaxMul gradient with respect to matrix B (f64).
#[pyfunction]
fn maxmul_backward_b_f64<'py>(
    py: Python<'py>,
    grad_c: PyReadonlyArray2<'py, f64>,
    argmax: PyReadonlyArray2<'py, i32>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = grad_c.shape();
    let m = shape[0];
    let n = shape[1];
    let k = a.shape()[1];

    let grad_c_data = grad_c.as_slice()?;
    let argmax_data = argmax.as_slice()?;
    let a_data = a.as_slice()?;

    let mut grad_b = vec![0.0f64; k * n];

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let k_idx = argmax_data[idx] as usize;
            if k_idx < k {
                grad_b[k_idx * n + j] += grad_c_data[idx] * a_data[i * k + k_idx];
            }
        }
    }

    Ok(grad_b.into_pyarray(py))
}

// ============================================================================
// i32 operations
// ============================================================================

/// Tropical MaxPlus matrix multiplication (i32): C[i,j] = max_k(A[i,k] + B[k,j])
#[pyfunction]
fn maxplus_matmul_i32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, i32>,
    b: PyReadonlyArray2<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let c_data = tropical_matmul::<TropicalMaxPlus<i32>>(a_data, m, k, b_data, n);
    let c_scalars: Vec<i32> = c_data.iter().map(|x| x.value()).collect();

    Ok(c_scalars.into_pyarray(py))
}

/// Tropical MinPlus matrix multiplication (i32): C[i,j] = min_k(A[i,k] + B[k,j])
#[pyfunction]
fn minplus_matmul_i32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, i32>,
    b: PyReadonlyArray2<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let c_data = tropical_matmul::<TropicalMinPlus<i32>>(a_data, m, k, b_data, n);
    let c_scalars: Vec<i32> = c_data.iter().map(|x| x.value()).collect();

    Ok(c_scalars.into_pyarray(py))
}

/// Tropical MaxMul matrix multiplication (i32): C[i,j] = max_k(A[i,k] * B[k,j])
#[pyfunction]
fn maxmul_matmul_i32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, i32>,
    b: PyReadonlyArray2<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let c_data = tropical_matmul::<TropicalMaxMul<i32>>(a_data, m, k, b_data, n);
    let c_scalars: Vec<i32> = c_data.iter().map(|x| x.value()).collect();

    Ok(c_scalars.into_pyarray(py))
}

// ============================================================================
// i64 operations
// ============================================================================

/// Tropical MaxPlus matrix multiplication (i64): C[i,j] = max_k(A[i,k] + B[k,j])
#[pyfunction]
fn maxplus_matmul_i64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, i64>,
    b: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let c_data = tropical_matmul::<TropicalMaxPlus<i64>>(a_data, m, k, b_data, n);
    let c_scalars: Vec<i64> = c_data.iter().map(|x| x.value()).collect();

    Ok(c_scalars.into_pyarray(py))
}

/// Tropical MinPlus matrix multiplication (i64): C[i,j] = min_k(A[i,k] + B[k,j])
#[pyfunction]
fn minplus_matmul_i64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, i64>,
    b: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let c_data = tropical_matmul::<TropicalMinPlus<i64>>(a_data, m, k, b_data, n);
    let c_scalars: Vec<i64> = c_data.iter().map(|x| x.value()).collect();

    Ok(c_scalars.into_pyarray(py))
}

/// Tropical MaxMul matrix multiplication (i64): C[i,j] = max_k(A[i,k] * B[k,j])
#[pyfunction]
fn maxmul_matmul_i64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, i64>,
    b: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, b_shape[0], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let c_data = tropical_matmul::<TropicalMaxMul<i64>>(a_data, m, k, b_data, n);
    let c_scalars: Vec<i64> = c_data.iter().map(|x| x.value()).collect();

    Ok(c_scalars.into_pyarray(py))
}

// ============================================================================
// Batched operations (3D arrays: batch × rows × cols)
// ============================================================================

/// Batched MaxPlus tropical matrix multiplication with argmax tracking.
///
/// Args:
///     a: Input tensor of shape (batch, M, K)
///     b: Input tensor of shape (batch, K, N)
///
/// Returns:
///     Tuple of (C, argmax) where:
///     - C: Result tensor of shape (batch × M × N) as flattened array
///     - argmax: Indices of shape (batch × M × N) as flattened array
#[pyfunction]
fn maxplus_matmul_batched_with_argmax<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];

    if batch != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Batch size mismatch: A has batch {}, B has batch {}",
            batch, b_shape[0]
        )));
    }

    if k != b_shape[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is (batch, {}, {}), B is (batch, {}, {})",
            m, k, b_shape[1], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let stride_a = m * k;
    let stride_b = k * n;
    let stride_c = m * n;

    let mut c_result = vec![0.0f32; batch * stride_c];
    let mut argmax_result = vec![0i32; batch * stride_c];

    for i in 0..batch {
        let a_slice = &a_data[i * stride_a..(i + 1) * stride_a];
        let b_slice = &b_data[i * stride_b..(i + 1) * stride_b];

        let result: GemmWithArgmax<TropicalMaxPlus<f32>> =
            tropical_matmul_with_argmax::<TropicalMaxPlus<f32>>(a_slice, m, k, b_slice, n);

        for (j, val) in result.values.iter().enumerate() {
            c_result[i * stride_c + j] = val.value();
        }
        for (j, &idx) in result.argmax.iter().enumerate() {
            argmax_result[i * stride_c + j] = idx as i32;
        }
    }

    Ok((c_result.into_pyarray(py), argmax_result.into_pyarray(py)))
}

/// Batched MinPlus tropical matrix multiplication with argmax tracking.
///
/// Args:
///     a: Input tensor of shape (batch, M, K)
///     b: Input tensor of shape (batch, K, N)
///
/// Returns:
///     Tuple of (C, argmax) where:
///     - C: Result tensor of shape (batch × M × N) as flattened array
///     - argmax: Indices of shape (batch × M × N) as flattened array
#[pyfunction]
fn minplus_matmul_batched_with_argmax<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];

    if batch != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Batch size mismatch: A has batch {}, B has batch {}",
            batch, b_shape[0]
        )));
    }

    if k != b_shape[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is (batch, {}, {}), B is (batch, {}, {})",
            m, k, b_shape[1], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let stride_a = m * k;
    let stride_b = k * n;
    let stride_c = m * n;

    let mut c_result = vec![0.0f32; batch * stride_c];
    let mut argmax_result = vec![0i32; batch * stride_c];

    for i in 0..batch {
        let a_slice = &a_data[i * stride_a..(i + 1) * stride_a];
        let b_slice = &b_data[i * stride_b..(i + 1) * stride_b];

        let result: GemmWithArgmax<TropicalMinPlus<f32>> =
            tropical_matmul_with_argmax::<TropicalMinPlus<f32>>(a_slice, m, k, b_slice, n);

        for (j, val) in result.values.iter().enumerate() {
            c_result[i * stride_c + j] = val.value();
        }
        for (j, &idx) in result.argmax.iter().enumerate() {
            argmax_result[i * stride_c + j] = idx as i32;
        }
    }

    Ok((c_result.into_pyarray(py), argmax_result.into_pyarray(py)))
}

/// Batched MaxMul tropical matrix multiplication with argmax tracking.
///
/// Args:
///     a: Input tensor of shape (batch, M, K)
///     b: Input tensor of shape (batch, K, N)
///
/// Returns:
///     Tuple of (C, argmax) where:
///     - C: Result tensor of shape (batch × M × N) as flattened array
///     - argmax: Indices of shape (batch × M × N) as flattened array
#[pyfunction]
fn maxmul_matmul_batched_with_argmax<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];

    if batch != b_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Batch size mismatch: A has batch {}, B has batch {}",
            batch, b_shape[0]
        )));
    }

    if k != b_shape[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is (batch, {}, {}), B is (batch, {}, {})",
            m, k, b_shape[1], n
        )));
    }

    let a_data = a.as_slice()?;
    let b_data = b.as_slice()?;

    let stride_a = m * k;
    let stride_b = k * n;
    let stride_c = m * n;

    let mut c_result = vec![0.0f32; batch * stride_c];
    let mut argmax_result = vec![0i32; batch * stride_c];

    for i in 0..batch {
        let a_slice = &a_data[i * stride_a..(i + 1) * stride_a];
        let b_slice = &b_data[i * stride_b..(i + 1) * stride_b];

        let result: GemmWithArgmax<TropicalMaxMul<f32>> =
            tropical_matmul_with_argmax::<TropicalMaxMul<f32>>(a_slice, m, k, b_slice, n);

        for (j, val) in result.values.iter().enumerate() {
            c_result[i * stride_c + j] = val.value();
        }
        for (j, &idx) in result.argmax.iter().enumerate() {
            argmax_result[i * stride_c + j] = idx as i32;
        }
    }

    Ok((c_result.into_pyarray(py), argmax_result.into_pyarray(py)))
}

// ============================================================================
// CUDA GPU operations (optional, requires "cuda" feature)
// ============================================================================

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use dlpark::ffi::{DataType, DataTypeCode, Device, DeviceType};
    use dlpark::{ManagerCtx, ManagedTensor, ShapeAndStrides, ToTensor, TensorView};
    #[allow(deprecated)]
    use pyo3::IntoPy;
    use std::ffi::c_void;
    use tropical_gemm_cuda::{
        get_context_for_device, launch_gemm_external_batched_with_argmax_f32,
        launch_gemm_external_with_argmax_f32, tropical_matmul_gpu,
        tropical_matmul_gpu_with_argmax, ExternalGpuMatrix, ExternalGpuTensor3,
        GpuMatrix, GpuTensor3,
    };

    // ========================================================================
    // DLPack wrapper types for GPU tensor export
    // ========================================================================

    /// Wrapper for GpuTensor3<f32> that implements ToTensor for DLPack export.
    ///
    /// This wrapper owns the GPU tensor and provides the necessary metadata
    /// for DLPack tensor exchange. The tensor data remains on GPU.
    struct DLPackGpuTensor3F32 {
        tensor: GpuTensor3<f32>,
        shape: [i64; 3],
        device_id: i32,
    }

    impl DLPackGpuTensor3F32 {
        fn new(tensor: GpuTensor3<f32>, device_id: i32) -> Self {
            let shape = [
                tensor.batch() as i64,
                tensor.rows() as i64,
                tensor.cols() as i64,
            ];
            Self { tensor, shape, device_id }
        }
    }

    impl ToTensor for DLPackGpuTensor3F32 {
        fn data_ptr(&self) -> *mut c_void {
            self.tensor.device_ptr() as *mut c_void
        }

        fn shape_and_strides(&self) -> ShapeAndStrides {
            // Row-major (C-contiguous) strides for 3D tensor
            ShapeAndStrides::new_contiguous(&self.shape)
        }

        fn device(&self) -> Device {
            Device {
                device_type: DeviceType::Cuda,
                device_id: self.device_id,
            }
        }

        fn dtype(&self) -> DataType {
            DataType {
                code: DataTypeCode::Float,
                bits: 32,
                lanes: 1,
            }
        }

        fn byte_offset(&self) -> u64 {
            0
        }
    }

    /// Wrapper for GpuTensor3<i32> (argmax indices) that implements ToTensor.
    struct DLPackGpuTensor3I32 {
        tensor: GpuTensor3<i32>,
        shape: [i64; 3],
        device_id: i32,
    }

    impl DLPackGpuTensor3I32 {
        fn new(tensor: GpuTensor3<i32>, device_id: i32) -> Self {
            let shape = [
                tensor.batch() as i64,
                tensor.rows() as i64,
                tensor.cols() as i64,
            ];
            Self { tensor, shape, device_id }
        }
    }

    impl ToTensor for DLPackGpuTensor3I32 {
        fn data_ptr(&self) -> *mut c_void {
            self.tensor.device_ptr() as *mut c_void
        }

        fn shape_and_strides(&self) -> ShapeAndStrides {
            ShapeAndStrides::new_contiguous(&self.shape)
        }

        fn device(&self) -> Device {
            Device {
                device_type: DeviceType::Cuda,
                device_id: self.device_id,
            }
        }

        fn dtype(&self) -> DataType {
            DataType {
                code: DataTypeCode::Int,
                bits: 32,
                lanes: 1,
            }
        }

        fn byte_offset(&self) -> u64 {
            0
        }
    }

    /// Wrapper for GpuMatrix<f32> (2D) that implements ToTensor for DLPack export.
    struct DLPackGpuMatrixF32 {
        matrix: GpuMatrix<f32>,
        shape: [i64; 2],
        device_id: i32,
    }

    impl DLPackGpuMatrixF32 {
        fn new(matrix: GpuMatrix<f32>, device_id: i32) -> Self {
            let shape = [matrix.rows() as i64, matrix.cols() as i64];
            Self { matrix, shape, device_id }
        }
    }

    impl ToTensor for DLPackGpuMatrixF32 {
        fn data_ptr(&self) -> *mut c_void {
            self.matrix.device_ptr() as *mut c_void
        }

        fn shape_and_strides(&self) -> ShapeAndStrides {
            // Row-major (C-contiguous) strides for 2D matrix
            ShapeAndStrides::new_contiguous(&self.shape)
        }

        fn device(&self) -> Device {
            Device {
                device_type: DeviceType::Cuda,
                device_id: self.device_id,
            }
        }

        fn dtype(&self) -> DataType {
            DataType {
                code: DataTypeCode::Float,
                bits: 32,
                lanes: 1,
            }
        }

        fn byte_offset(&self) -> u64 {
            0
        }
    }

    /// Wrapper for GpuMatrix<i32> (2D argmax) that implements ToTensor for DLPack export.
    struct DLPackGpuMatrixI32 {
        matrix: GpuMatrix<i32>,
        shape: [i64; 2],
        device_id: i32,
    }

    impl DLPackGpuMatrixI32 {
        fn new(matrix: GpuMatrix<i32>, device_id: i32) -> Self {
            let shape = [matrix.rows() as i64, matrix.cols() as i64];
            Self { matrix, shape, device_id }
        }
    }

    impl ToTensor for DLPackGpuMatrixI32 {
        fn data_ptr(&self) -> *mut c_void {
            self.matrix.device_ptr() as *mut c_void
        }

        fn shape_and_strides(&self) -> ShapeAndStrides {
            ShapeAndStrides::new_contiguous(&self.shape)
        }

        fn device(&self) -> Device {
            Device {
                device_type: DeviceType::Cuda,
                device_id: self.device_id,
            }
        }

        fn dtype(&self) -> DataType {
            DataType {
                code: DataTypeCode::Int,
                bits: 32,
                lanes: 1,
            }
        }

        fn byte_offset(&self) -> u64 {
            0
        }
    }

    /// Helper function to extract ManagedTensor from a Python object.
    /// Calls __dlpack__() if available, otherwise tries direct extraction.
    fn extract_dlpack_tensor(_py: Python, obj: &Bound<'_, pyo3::PyAny>) -> PyResult<ManagedTensor> {
        // Try to call __dlpack__() method to get the capsule
        if let Ok(capsule) = obj.call_method0("__dlpack__") {
            // Extract ManagedTensor from the returned capsule
            capsule.extract::<ManagedTensor>()
        } else {
            // Fallback: try direct extraction (for objects that are already capsules)
            obj.extract::<ManagedTensor>()
        }
    }

    /// GPU-accelerated MaxPlus matrix multiplication: C[i,j] = max_k(A[i,k] + B[k,j])
    ///
    /// Note: This creates a new CUDA context for each call. For repeated operations,
    /// consider batching your computations.
    ///
    /// Args:
    ///     a: Input matrix A of shape (M, K)
    ///     b: Input matrix B of shape (K, N)
    ///
    /// Returns:
    ///     Result matrix C of shape (M, N) as a flattened array
    #[pyfunction]
    pub fn maxplus_matmul_gpu<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'py, f32>,
        b: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if k != b_shape[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: A is {}x{}, B is {}x{}",
                m, k, b_shape[0], n
            )));
        }

        let a_data = a.as_slice()?;
        let b_data = b.as_slice()?;

        let c_data = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(a_data, m, k, b_data, n)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {}", e)))?;

        Ok(c_data.into_pyarray(py))
    }

    /// GPU-accelerated MinPlus matrix multiplication: C[i,j] = min_k(A[i,k] + B[k,j])
    #[pyfunction]
    pub fn minplus_matmul_gpu<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'py, f32>,
        b: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if k != b_shape[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: A is {}x{}, B is {}x{}",
                m, k, b_shape[0], n
            )));
        }

        let a_data = a.as_slice()?;
        let b_data = b.as_slice()?;

        let c_data = tropical_matmul_gpu::<TropicalMinPlus<f32>>(a_data, m, k, b_data, n)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {}", e)))?;

        Ok(c_data.into_pyarray(py))
    }

    /// GPU-accelerated MaxPlus with argmax tracking for backpropagation.
    ///
    /// Args:
    ///     a: Input matrix A of shape (M, K)
    ///     b: Input matrix B of shape (K, N)
    ///
    /// Returns:
    ///     Tuple of (C, argmax) where:
    ///     - C: Result matrix of shape (M, N) as flattened array
    ///     - argmax: Indices of shape (M, N) as flattened array
    #[pyfunction]
    pub fn maxplus_matmul_gpu_with_argmax<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'py, f32>,
        b: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if k != b_shape[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: A is {}x{}, B is {}x{}",
                m, k, b_shape[0], n
            )));
        }

        let a_data = a.as_slice()?;
        let b_data = b.as_slice()?;

        let (c_data, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(a_data, m, k, b_data, n)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {}", e))
                })?;

        let argmax_i32: Vec<i32> = argmax.into_iter().map(|x| x as i32).collect();

        Ok((c_data.into_pyarray(py), argmax_i32.into_pyarray(py)))
    }

    /// GPU-accelerated MinPlus with argmax tracking for backpropagation.
    #[pyfunction]
    pub fn minplus_matmul_gpu_with_argmax<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'py, f32>,
        b: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if k != b_shape[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: A is {}x{}, B is {}x{}",
                m, k, b_shape[0], n
            )));
        }

        let a_data = a.as_slice()?;
        let b_data = b.as_slice()?;

        let (c_data, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMinPlus<f32>>(a_data, m, k, b_data, n)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {}", e))
                })?;

        let argmax_i32: Vec<i32> = argmax.into_iter().map(|x| x as i32).collect();

        Ok((c_data.into_pyarray(py), argmax_i32.into_pyarray(py)))
    }

    /// GPU-accelerated MaxMul matrix multiplication: C[i,j] = max_k(A[i,k] * B[k,j])
    #[pyfunction]
    pub fn maxmul_matmul_gpu<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'py, f32>,
        b: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if k != b_shape[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: A is {}x{}, B is {}x{}",
                m, k, b_shape[0], n
            )));
        }

        let a_data = a.as_slice()?;
        let b_data = b.as_slice()?;

        let c_data = tropical_matmul_gpu::<TropicalMaxMul<f32>>(a_data, m, k, b_data, n)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {}", e)))?;

        Ok(c_data.into_pyarray(py))
    }

    /// GPU-accelerated MaxMul with argmax tracking for backpropagation.
    #[pyfunction]
    pub fn maxmul_matmul_gpu_with_argmax<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'py, f32>,
        b: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if k != b_shape[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: A is {}x{}, B is {}x{}",
                m, k, b_shape[0], n
            )));
        }

        let a_data = a.as_slice()?;
        let b_data = b.as_slice()?;

        let (c_data, argmax) =
            tropical_matmul_gpu_with_argmax::<TropicalMaxMul<f32>>(a_data, m, k, b_data, n)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {}", e))
                })?;

        let argmax_i32: Vec<i32> = argmax.into_iter().map(|x| x as i32).collect();

        Ok((c_data.into_pyarray(py), argmax_i32.into_pyarray(py)))
    }

    // ========================================================================
    // DLPack zero-copy functions (2D tensors)
    // ========================================================================

    /// MaxPlus matrix multiplication using DLPack for zero-copy GPU tensor exchange.
    ///
    /// This function accepts PyTorch tensors (or any DLPack-compatible tensor) directly
    /// and performs the computation without copying data for GPU tensors.
    ///
    /// Args:
    ///     a: Input tensor A of shape (M, K) - must support __dlpack__(), f32
    ///     b: Input tensor B of shape (K, N) - must support __dlpack__(), f32
    ///
    /// Returns:
    ///     Tuple of (C, argmax) where the type depends on input device:
    ///
    ///     For CUDA tensors: Returns DLPack capsules (data stays on GPU)
    ///     - C: Result of shape (M*N,) as f32 - use `torch.from_dlpack(c).reshape(m, n)`
    ///     - argmax: Indices of shape (M*N,) as i32 - use `torch.from_dlpack(argmax).reshape(m, n)`
    ///
    ///     For CPU tensors: Returns numpy arrays (flattened)
    ///     - C: Result of shape (M*N,) as f32 - use `torch.from_numpy(c).reshape(m, n)`
    ///     - argmax: Indices of shape (M*N,) as i32 - use `torch.from_numpy(argmax).reshape(m, n)`
    #[pyfunction]
    pub fn maxplus_matmul_dlpack(
        py: Python<'_>,
        a: Bound<'_, pyo3::PyAny>,
        b: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        dlpack_2d_impl(py, a, b, "tropical_maxplus_f32_nn_with_argmax", Algebra::MaxPlus)
    }

    /// MinPlus matrix multiplication using DLPack for zero-copy GPU tensor exchange.
    ///
    /// Returns DLPack capsules - use `torch.from_dlpack(capsule)` to convert.
    #[pyfunction]
    pub fn minplus_matmul_dlpack(
        py: Python<'_>,
        a: Bound<'_, pyo3::PyAny>,
        b: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        dlpack_2d_impl(py, a, b, "tropical_minplus_f32_nn_with_argmax", Algebra::MinPlus)
    }

    /// MaxMul matrix multiplication using DLPack for zero-copy GPU tensor exchange.
    ///
    /// Returns DLPack capsules - use `torch.from_dlpack(capsule)` to convert.
    #[pyfunction]
    pub fn maxmul_matmul_dlpack(
        py: Python<'_>,
        a: Bound<'_, pyo3::PyAny>,
        b: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        dlpack_2d_impl(py, a, b, "tropical_maxmul_f32_nn_with_argmax", Algebra::MaxMul)
    }

    /// Algebra type for CPU dispatch in 2D DLPack functions.
    enum Algebra {
        MaxPlus,
        MinPlus,
        MaxMul,
    }

    /// Implementation for 2D DLPack operations.
    ///
    /// Returns DLPack capsules that keep data on GPU - use `torch.from_dlpack()`
    /// to convert to PyTorch tensors.
    fn dlpack_2d_impl(
        py: Python<'_>,
        a: Bound<'_, pyo3::PyAny>,
        b: Bound<'_, pyo3::PyAny>,
        kernel_name: &'static str,
        algebra: Algebra,
    ) -> PyResult<(PyObject, PyObject)> {
        // Extract tensor info from DLPack
        let a_tensor = extract_dlpack_tensor(py, &a)?;
        let b_tensor = extract_dlpack_tensor(py, &b)?;

        // Get device info
        let a_device = TensorView::device(&a_tensor);
        let b_device = TensorView::device(&b_tensor);

        // Validate: both tensors must be on the same device type
        if a_device.device_type != b_device.device_type {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Tensors must be on the same device type: A is on {:?}, B is on {:?}",
                a_device.device_type, b_device.device_type
            )));
        }

        // Validate: both tensors must be on the same device ID (for multi-GPU)
        if a_device.device_id != b_device.device_id {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Tensors must be on the same device: A is on device {}, B is on device {}",
                a_device.device_id, b_device.device_id
            )));
        }

        let device_id = a_device.device_id;

        // Get dtype and validate
        let a_dtype = TensorView::dtype(&a_tensor);
        let b_dtype = TensorView::dtype(&b_tensor);
        if a_dtype != b_dtype {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Tensors must have the same dtype: A is {:?}, B is {:?}",
                a_dtype, b_dtype
            )));
        }

        // Validate dtype is f32
        if a_dtype.code != DataTypeCode::Float || a_dtype.bits != 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Only f32 tensors are supported for DLPack interface",
            ));
        }

        // Get shapes
        let a_shape = TensorView::shape(&a_tensor);
        let b_shape = TensorView::shape(&b_tensor);

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected 2D tensors, got A with {} dims, B with {} dims",
                a_shape.len(),
                b_shape.len()
            )));
        }

        let m = a_shape[0] as usize;
        let k = a_shape[1] as usize;
        let k2 = b_shape[0] as usize;
        let n = b_shape[1] as usize;

        if k != k2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: A is {}x{}, B is {}x{}",
                m, k, k2, n
            )));
        }

        // Guard against zero-sized dimensions
        if m == 0 || k == 0 || n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Zero-sized dimensions not supported: m={}, k={}, n={}",
                m, k, n
            )));
        }

        // Check strides for contiguity
        let a_strides = TensorView::strides(&a_tensor);
        let b_strides = TensorView::strides(&b_tensor);

        // For row-major (C-contiguous): strides should be [cols, 1]
        let a_contiguous = a_strides.is_none()
            || a_strides.map_or(false, |s| s.len() == 2 && s[1] == 1 && s[0] == k as i64);
        let b_contiguous = b_strides.is_none()
            || b_strides.map_or(false, |s| s.len() == 2 && s[1] == 1 && s[0] == n as i64);

        if !a_contiguous || !b_contiguous {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Tensors must be contiguous (call .contiguous() on PyTorch tensors)",
            ));
        }

        match a_device.device_type {
            DeviceType::Cuda => {
                // GPU path: zero-copy using DLPack
                let a_ptr = TensorView::data_ptr(&a_tensor) as u64;
                let b_ptr = TensorView::data_ptr(&b_tensor) as u64;

                // Create external matrix views
                let a_ext = unsafe { ExternalGpuMatrix::from_raw(a_ptr, m, k) };
                let b_ext = unsafe { ExternalGpuMatrix::from_raw(b_ptr, k, n) };

                // Get CUDA context for the input device
                let ctx = get_context_for_device(device_id as usize).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {}", e))
                })?;

                // Launch kernel
                let result = unsafe {
                    launch_gemm_external_with_argmax_f32(ctx, kernel_name, &a_ext, &b_ext, m, k, n)
                }
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA kernel error: {}", e))
                })?;

                // Split result into matrix and argmax, then wrap for DLPack export
                let (c_matrix, argmax_matrix) = result.into_parts();

                // Wrap in DLPack-compatible wrappers with correct device_id
                let c_dlpack = DLPackGpuMatrixF32::new(c_matrix, device_id);
                let argmax_dlpack = DLPackGpuMatrixI32::new(argmax_matrix, device_id);

                // Convert to DLPack capsules using ManagerCtx
                let c_capsule = ManagerCtx::new(c_dlpack).into_py(py);
                let argmax_capsule = ManagerCtx::new(argmax_dlpack).into_py(py);

                Ok((c_capsule, argmax_capsule))
            }
            DeviceType::CudaHost => {
                // CudaHost (pinned memory) is not supported - the pointer is a host pointer
                // that cannot be used directly by CUDA kernels without explicit handling
                Err(pyo3::exceptions::PyValueError::new_err(
                    "CudaHost (pinned memory) tensors are not supported. Use regular CUDA tensors.",
                ))
            }
            DeviceType::Cpu => {
                // CPU path: use existing CPU backend, return numpy arrays as PyObject
                let a_ptr = TensorView::data_ptr(&a_tensor) as *const f32;
                let b_ptr = TensorView::data_ptr(&b_tensor) as *const f32;

                let a_data = unsafe { std::slice::from_raw_parts(a_ptr, m * k) };
                let b_data = unsafe { std::slice::from_raw_parts(b_ptr, k * n) };

                let (c_scalars, argmax_i32) = match algebra {
                    Algebra::MaxPlus => {
                        let result: ::tropical_gemm::GemmWithArgmax<TropicalMaxPlus<f32>> =
                            ::tropical_gemm::tropical_matmul_with_argmax::<TropicalMaxPlus<f32>>(
                                a_data, m, k, b_data, n,
                            );
                        let c: Vec<f32> = result.values.iter().map(|x| x.value()).collect();
                        let argmax: Vec<i32> = result.argmax.iter().map(|&x| x as i32).collect();
                        (c, argmax)
                    }
                    Algebra::MinPlus => {
                        let result: ::tropical_gemm::GemmWithArgmax<TropicalMinPlus<f32>> =
                            ::tropical_gemm::tropical_matmul_with_argmax::<TropicalMinPlus<f32>>(
                                a_data, m, k, b_data, n,
                            );
                        let c: Vec<f32> = result.values.iter().map(|x| x.value()).collect();
                        let argmax: Vec<i32> = result.argmax.iter().map(|&x| x as i32).collect();
                        (c, argmax)
                    }
                    Algebra::MaxMul => {
                        let result: ::tropical_gemm::GemmWithArgmax<TropicalMaxMul<f32>> =
                            ::tropical_gemm::tropical_matmul_with_argmax::<TropicalMaxMul<f32>>(
                                a_data, m, k, b_data, n,
                            );
                        let c: Vec<f32> = result.values.iter().map(|x| x.value()).collect();
                        let argmax: Vec<i32> = result.argmax.iter().map(|&x| x as i32).collect();
                        (c, argmax)
                    }
                };

                Ok((
                    c_scalars.into_pyarray(py).into_any().unbind(),
                    argmax_i32.into_pyarray(py).into_any().unbind(),
                ))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported device type: {:?}",
                a_device.device_type
            ))),
        }
    }

    /// Check if CUDA is available.
    #[pyfunction]
    pub fn cuda_available() -> bool {
        true
    }

    // ========================================================================
    // Batched DLPack functions (3D tensors)
    // ========================================================================
    //
    // Note: Only f32 with argmax is supported for batched GPU operations.
    // f64/i32/i64 and non-argmax variants are not implemented.
    // For other dtypes, use the CPU batched API.

    /// Batched MaxPlus matrix multiplication using DLPack for zero-copy GPU tensor exchange.
    ///
    /// Computes C[b,i,j] = max_k(A[b,i,k] + B[b,k,j]) for each batch b.
    ///
    /// # Limitations
    ///
    /// - Only f32 dtype is supported (GPU batched)
    /// - Always returns argmax (no non-argmax variant)
    ///
    /// Args:
    ///     a: Input tensor A of shape (batch, M, K) - must support __dlpack__(), f32, CUDA
    ///     b: Input tensor B of shape (batch, K, N) - must support __dlpack__(), f32, CUDA
    ///
    /// Returns:
    ///     Tuple of (C, argmax) as DLPack capsules where:
    ///     - C: Result tensor of shape (batch, M, N) as f32 CUDA tensor
    ///     - argmax: Indices of shape (batch, M, N) as i32 CUDA tensor
    ///
    ///     Use `torch.from_dlpack(capsule)` to convert to PyTorch tensors.
    ///
    /// Raises:
    ///     RuntimeError: If tensors are not on CUDA or DLPack extraction fails
    ///     ValueError: If tensors are not f32 or not 3D
    #[pyfunction]
    pub fn maxplus_matmul_batched_dlpack(
        py: Python<'_>,
        a: Bound<'_, pyo3::PyAny>,
        b: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        batched_dlpack_impl(py, a, b, "tropical_maxplus_f32_nn_batched_with_argmax")
    }

    /// Batched MinPlus matrix multiplication using DLPack for zero-copy GPU tensor exchange.
    ///
    /// Returns DLPack capsules - use `torch.from_dlpack(capsule)` to convert.
    #[pyfunction]
    pub fn minplus_matmul_batched_dlpack(
        py: Python<'_>,
        a: Bound<'_, pyo3::PyAny>,
        b: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        batched_dlpack_impl(py, a, b, "tropical_minplus_f32_nn_batched_with_argmax")
    }

    /// Batched MaxMul matrix multiplication using DLPack for zero-copy GPU tensor exchange.
    ///
    /// Returns DLPack capsules - use `torch.from_dlpack(capsule)` to convert.
    #[pyfunction]
    pub fn maxmul_matmul_batched_dlpack(
        py: Python<'_>,
        a: Bound<'_, pyo3::PyAny>,
        b: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        batched_dlpack_impl(py, a, b, "tropical_maxmul_f32_nn_batched_with_argmax")
    }

    /// Implementation for batched DLPack operations.
    ///
    /// Returns DLPack capsules that keep data on GPU - use `torch.from_dlpack()`
    /// to convert to PyTorch tensors.
    fn batched_dlpack_impl(
        py: Python<'_>,
        a: Bound<'_, pyo3::PyAny>,
        b: Bound<'_, pyo3::PyAny>,
        kernel_name: &'static str,
    ) -> PyResult<(PyObject, PyObject)> {
        // Extract tensor info from DLPack
        let a_tensor = extract_dlpack_tensor(py, &a)?;
        let b_tensor = extract_dlpack_tensor(py, &b)?;

        // Get device info
        let a_device = TensorView::device(&a_tensor);
        let b_device = TensorView::device(&b_tensor);

        // Validate: must be on CUDA device (not CudaHost/pinned memory)
        if a_device.device_type != DeviceType::Cuda {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Tensor A must be on CUDA device, got {:?}. Use CPU batched functions for CPU tensors.",
                a_device.device_type
            )));
        }

        if b_device.device_type != DeviceType::Cuda {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Tensor B must be on CUDA device, got {:?}. Use CPU batched functions for CPU tensors.",
                b_device.device_type
            )));
        }

        if a_device.device_id != b_device.device_id {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Tensors must be on the same CUDA device: A is on cuda:{}, B is on cuda:{}",
                a_device.device_id, b_device.device_id
            )));
        }

        let device_id = a_device.device_id;

        // Get dtype and validate
        let a_dtype = TensorView::dtype(&a_tensor);
        let b_dtype = TensorView::dtype(&b_tensor);
        if a_dtype != b_dtype {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Tensors must have the same dtype: A is {:?}, B is {:?}",
                a_dtype, b_dtype
            )));
        }

        if a_dtype.code != DataTypeCode::Float || a_dtype.bits != 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Only f32 tensors are supported for batched DLPack interface",
            ));
        }

        // Get shapes - must be 3D
        let a_shape = TensorView::shape(&a_tensor);
        let b_shape = TensorView::shape(&b_tensor);

        if a_shape.len() != 3 || b_shape.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected 3D tensors, got A with {} dims, B with {} dims",
                a_shape.len(),
                b_shape.len()
            )));
        }

        let batch = a_shape[0] as usize;
        let m = a_shape[1] as usize;
        let k = a_shape[2] as usize;
        let batch_b = b_shape[0] as usize;
        let k2 = b_shape[1] as usize;
        let n = b_shape[2] as usize;

        if batch != batch_b {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Batch size mismatch: A has batch {}, B has batch {}",
                batch, batch_b
            )));
        }

        if k != k2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: A is (batch, {}, {}), B is (batch, {}, {})",
                m, k, k2, n
            )));
        }

        // Guard against zero-sized dimensions (would cause invalid CUDA launch)
        if batch == 0 || m == 0 || k == 0 || n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Zero-sized dimensions not supported: batch={}, m={}, k={}, n={}",
                batch, m, k, n
            )));
        }

        // Check strides for contiguity (row-major per batch)
        let a_strides = TensorView::strides(&a_tensor);
        let b_strides = TensorView::strides(&b_tensor);

        // For 3D row-major (C-contiguous): strides should be [m*k, k, 1]
        let a_contiguous = a_strides.is_none()
            || a_strides.map_or(false, |s| {
                s.len() == 3 && s[2] == 1 && s[1] == k as i64 && s[0] == (m * k) as i64
            });
        let b_contiguous = b_strides.is_none()
            || b_strides.map_or(false, |s| {
                s.len() == 3 && s[2] == 1 && s[1] == n as i64 && s[0] == (k * n) as i64
            });

        if !a_contiguous || !b_contiguous {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Tensors must be contiguous (call .contiguous() on PyTorch tensors)",
            ));
        }

        // GPU path: zero-copy using DLPack
        let a_ptr = TensorView::data_ptr(&a_tensor) as u64;
        let b_ptr = TensorView::data_ptr(&b_tensor) as u64;

        // Create external 3D tensor views
        let a_ext = unsafe { ExternalGpuTensor3::from_raw_contiguous(a_ptr, batch, m, k) };
        let b_ext = unsafe { ExternalGpuTensor3::from_raw_contiguous(b_ptr, batch, k, n) };

        // Get CUDA context for the input device
        let ctx = get_context_for_device(device_id as usize).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {}", e))
        })?;

        // Launch batched kernel
        let result = unsafe {
            launch_gemm_external_batched_with_argmax_f32(ctx, kernel_name, &a_ext, &b_ext, batch, m, k, n)
        }
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA kernel error: {}", e))
        })?;

        // Split result into tensor and argmax, then wrap for DLPack export
        let (c_tensor, argmax_tensor) = result.into_parts();

        // Wrap in DLPack-compatible wrappers with correct device_id
        let c_dlpack = DLPackGpuTensor3F32::new(c_tensor, device_id);
        let argmax_dlpack = DLPackGpuTensor3I32::new(argmax_tensor, device_id);

        // Convert to DLPack capsules using ManagerCtx
        // ManagerCtx owns the tensor and exports it as a DLPack capsule
        let c_capsule = ManagerCtx::new(c_dlpack).into_py(py);
        let argmax_capsule = ManagerCtx::new(argmax_dlpack).into_py(py);

        Ok((c_capsule, argmax_capsule))
    }

    /// Register GPU functions in the module.
    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(maxplus_matmul_gpu, m)?)?;
        m.add_function(wrap_pyfunction!(minplus_matmul_gpu, m)?)?;
        m.add_function(wrap_pyfunction!(maxmul_matmul_gpu, m)?)?;
        m.add_function(wrap_pyfunction!(maxplus_matmul_gpu_with_argmax, m)?)?;
        m.add_function(wrap_pyfunction!(minplus_matmul_gpu_with_argmax, m)?)?;
        m.add_function(wrap_pyfunction!(maxmul_matmul_gpu_with_argmax, m)?)?;
        // DLPack zero-copy functions
        m.add_function(wrap_pyfunction!(maxplus_matmul_dlpack, m)?)?;
        m.add_function(wrap_pyfunction!(minplus_matmul_dlpack, m)?)?;
        m.add_function(wrap_pyfunction!(maxmul_matmul_dlpack, m)?)?;
        // Batched DLPack functions
        m.add_function(wrap_pyfunction!(maxplus_matmul_batched_dlpack, m)?)?;
        m.add_function(wrap_pyfunction!(minplus_matmul_batched_dlpack, m)?)?;
        m.add_function(wrap_pyfunction!(maxmul_matmul_batched_dlpack, m)?)?;
        m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
mod gpu {
    use super::*;

    /// Check if CUDA is available (stub when not compiled with CUDA).
    #[pyfunction]
    pub fn cuda_available() -> bool {
        false
    }

    /// Register GPU functions in the module (stub).
    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
        Ok(())
    }
}

/// Tropical GEMM Python module (native Rust extension).
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // f32 operations
    m.add_function(wrap_pyfunction!(maxplus_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(minplus_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(maxplus_matmul_with_argmax, m)?)?;
    m.add_function(wrap_pyfunction!(minplus_matmul_with_argmax, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_matmul_with_argmax, m)?)?;
    m.add_function(wrap_pyfunction!(backward_a, m)?)?;
    m.add_function(wrap_pyfunction!(backward_b, m)?)?;

    // f64 operations
    m.add_function(wrap_pyfunction!(maxplus_matmul_f64, m)?)?;
    m.add_function(wrap_pyfunction!(minplus_matmul_f64, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_matmul_f64, m)?)?;
    m.add_function(wrap_pyfunction!(maxplus_matmul_with_argmax_f64, m)?)?;
    m.add_function(wrap_pyfunction!(minplus_matmul_with_argmax_f64, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_matmul_with_argmax_f64, m)?)?;
    m.add_function(wrap_pyfunction!(backward_a_f64, m)?)?;
    m.add_function(wrap_pyfunction!(backward_b_f64, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_backward_a, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_backward_b, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_backward_a_f64, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_backward_b_f64, m)?)?;

    // Batched operations (3D arrays)
    m.add_function(wrap_pyfunction!(maxplus_matmul_batched_with_argmax, m)?)?;
    m.add_function(wrap_pyfunction!(minplus_matmul_batched_with_argmax, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_matmul_batched_with_argmax, m)?)?;

    // i32 operations
    m.add_function(wrap_pyfunction!(maxplus_matmul_i32, m)?)?;
    m.add_function(wrap_pyfunction!(minplus_matmul_i32, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_matmul_i32, m)?)?;

    // i64 operations
    m.add_function(wrap_pyfunction!(maxplus_matmul_i64, m)?)?;
    m.add_function(wrap_pyfunction!(minplus_matmul_i64, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_matmul_i64, m)?)?;

    // GPU operations (if available)
    gpu::register(m)?;

    Ok(())
}
