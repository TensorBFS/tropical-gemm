//! Python bindings for tropical matrix multiplication.
//!
//! This module provides Python/NumPy bindings for tropical GEMM operations,
//! enabling integration with PyTorch custom autograd functions.
//!
//! ## Features
//!
//! - `cuda`: Enable GPU acceleration via CUDA

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
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
// CUDA GPU operations (optional, requires "cuda" feature)
// ============================================================================

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use dlpark::ffi::{DataTypeCode, DeviceType};
    use dlpark::ManagedTensor;
    use dlpark::TensorView;
    use tropical_gemm_cuda::{
        get_global_context, launch_gemm_external_with_argmax_f32, tropical_matmul_gpu,
        tropical_matmul_gpu_with_argmax, ExternalGpuMatrix,
    };

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
    // DLPack zero-copy functions
    // ========================================================================

    /// MaxPlus matrix multiplication using DLPack for zero-copy GPU tensor exchange.
    ///
    /// This function accepts PyTorch tensors (or any DLPack-compatible tensor) directly
    /// and performs the computation without copying input data for GPU tensors.
    ///
    /// Args:
    ///     a: Input tensor A of shape (M, K) - must support __dlpack__()
    ///     b: Input tensor B of shape (K, N) - must support __dlpack__()
    ///
    /// Returns:
    ///     Tuple of (C, argmax) as numpy arrays where:
    ///     - C: Result matrix of shape (M, N) as flattened f32 array
    ///     - argmax: Indices of shape (M, N) as flattened i32 array
    ///
    /// Note:
    ///     - For GPU tensors: Uses zero-copy DLPack interface with Rust CUDA backend
    ///     - For CPU tensors: Falls back to optimized Rust CPU backend
    #[pyfunction]
    pub fn maxplus_matmul_dlpack<'py>(
        py: Python<'py>,
        a: Bound<'py, pyo3::PyAny>,
        b: Bound<'py, pyo3::PyAny>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
        // Extract tensor info from DLPack
        let a_tensor = extract_dlpack_tensor(py, &a)?;
        let b_tensor = extract_dlpack_tensor(py, &b)?;

        // Get device info (ManagedTensor implements TensorView trait)
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
            DeviceType::Cuda | DeviceType::CudaHost => {
                // GPU path: zero-copy using DLPack
                let a_ptr = TensorView::data_ptr(&a_tensor) as u64;
                let b_ptr = TensorView::data_ptr(&b_tensor) as u64;

                // Create external matrix views
                let a_ext = unsafe { ExternalGpuMatrix::from_raw(a_ptr, m, k) };
                let b_ext = unsafe { ExternalGpuMatrix::from_raw(b_ptr, k, n) };

                // Get the global CUDA context
                let ctx = get_global_context().map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {}", e))
                })?;

                // Launch kernel
                let result = unsafe {
                    launch_gemm_external_with_argmax_f32(
                        ctx,
                        "tropical_maxplus_f32_nn_with_argmax",
                        &a_ext,
                        &b_ext,
                        m,
                        k,
                        n,
                    )
                }
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA kernel error: {}", e))
                })?;

                // Download results to host (Phase 1: output is numpy array)
                let c_data = result.matrix_to_host_row_major(ctx).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA D2H error: {}", e))
                })?;
                let argmax_data = result.argmax_to_host_row_major(ctx).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA D2H error: {}", e))
                })?;

                Ok((c_data.into_pyarray(py), argmax_data.into_pyarray(py)))
            }
            DeviceType::Cpu => {
                // CPU path: use existing CPU backend
                let a_ptr = TensorView::data_ptr(&a_tensor) as *const f32;
                let b_ptr = TensorView::data_ptr(&b_tensor) as *const f32;

                let a_data = unsafe { std::slice::from_raw_parts(a_ptr, m * k) };
                let b_data = unsafe { std::slice::from_raw_parts(b_ptr, k * n) };

                let result: ::tropical_gemm::GemmWithArgmax<TropicalMaxPlus<f32>> =
                    ::tropical_gemm::tropical_matmul_with_argmax::<TropicalMaxPlus<f32>>(
                        a_data, m, k, b_data, n,
                    );

                let c_scalars: Vec<f32> = result.values.iter().map(|x| x.value()).collect();
                let argmax_i32: Vec<i32> = result.argmax.iter().map(|&x| x as i32).collect();

                Ok((c_scalars.into_pyarray(py), argmax_i32.into_pyarray(py)))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported device type: {:?}",
                a_device.device_type
            ))),
        }
    }

    /// MinPlus matrix multiplication using DLPack for zero-copy GPU tensor exchange.
    #[pyfunction]
    pub fn minplus_matmul_dlpack<'py>(
        py: Python<'py>,
        a: Bound<'py, pyo3::PyAny>,
        b: Bound<'py, pyo3::PyAny>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
        // Extract tensor info from DLPack
        let a_tensor = extract_dlpack_tensor(py, &a)?;
        let b_tensor = extract_dlpack_tensor(py, &b)?;

        // Get device info
        let a_device = TensorView::device(&a_tensor);
        let b_device = TensorView::device(&b_tensor);

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
                "Only f32 tensors are supported for DLPack interface",
            ));
        }

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

        let a_strides = TensorView::strides(&a_tensor);
        let b_strides = TensorView::strides(&b_tensor);

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
            DeviceType::Cuda | DeviceType::CudaHost => {
                let a_ptr = TensorView::data_ptr(&a_tensor) as u64;
                let b_ptr = TensorView::data_ptr(&b_tensor) as u64;

                let a_ext = unsafe { ExternalGpuMatrix::from_raw(a_ptr, m, k) };
                let b_ext = unsafe { ExternalGpuMatrix::from_raw(b_ptr, k, n) };

                let ctx = get_global_context().map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {}", e))
                })?;

                let result = unsafe {
                    launch_gemm_external_with_argmax_f32(
                        ctx,
                        "tropical_minplus_f32_nn_with_argmax",
                        &a_ext,
                        &b_ext,
                        m,
                        k,
                        n,
                    )
                }
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA kernel error: {}", e))
                })?;

                let c_data = result.matrix_to_host_row_major(ctx).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA D2H error: {}", e))
                })?;
                let argmax_data = result.argmax_to_host_row_major(ctx).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA D2H error: {}", e))
                })?;

                Ok((c_data.into_pyarray(py), argmax_data.into_pyarray(py)))
            }
            DeviceType::Cpu => {
                let a_ptr = TensorView::data_ptr(&a_tensor) as *const f32;
                let b_ptr = TensorView::data_ptr(&b_tensor) as *const f32;

                let a_data = unsafe { std::slice::from_raw_parts(a_ptr, m * k) };
                let b_data = unsafe { std::slice::from_raw_parts(b_ptr, k * n) };

                let result: ::tropical_gemm::GemmWithArgmax<TropicalMinPlus<f32>> =
                    ::tropical_gemm::tropical_matmul_with_argmax::<TropicalMinPlus<f32>>(
                        a_data, m, k, b_data, n,
                    );

                let c_scalars: Vec<f32> = result.values.iter().map(|x| x.value()).collect();
                let argmax_i32: Vec<i32> = result.argmax.iter().map(|&x| x as i32).collect();

                Ok((c_scalars.into_pyarray(py), argmax_i32.into_pyarray(py)))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported device type: {:?}",
                a_device.device_type
            ))),
        }
    }

    /// MaxMul matrix multiplication using DLPack for zero-copy GPU tensor exchange.
    #[pyfunction]
    pub fn maxmul_matmul_dlpack<'py>(
        py: Python<'py>,
        a: Bound<'py, pyo3::PyAny>,
        b: Bound<'py, pyo3::PyAny>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
        let a_tensor = extract_dlpack_tensor(py, &a)?;
        let b_tensor = extract_dlpack_tensor(py, &b)?;

        let a_device = TensorView::device(&a_tensor);
        let b_device = TensorView::device(&b_tensor);

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
                "Only f32 tensors are supported for DLPack interface",
            ));
        }

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

        let a_strides = TensorView::strides(&a_tensor);
        let b_strides = TensorView::strides(&b_tensor);

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
            DeviceType::Cuda | DeviceType::CudaHost => {
                let a_ptr = TensorView::data_ptr(&a_tensor) as u64;
                let b_ptr = TensorView::data_ptr(&b_tensor) as u64;

                let a_ext = unsafe { ExternalGpuMatrix::from_raw(a_ptr, m, k) };
                let b_ext = unsafe { ExternalGpuMatrix::from_raw(b_ptr, k, n) };

                let ctx = get_global_context().map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {}", e))
                })?;

                let result = unsafe {
                    launch_gemm_external_with_argmax_f32(
                        ctx,
                        "tropical_maxmul_f32_nn_with_argmax",
                        &a_ext,
                        &b_ext,
                        m,
                        k,
                        n,
                    )
                }
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA kernel error: {}", e))
                })?;

                let c_data = result.matrix_to_host_row_major(ctx).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA D2H error: {}", e))
                })?;
                let argmax_data = result.argmax_to_host_row_major(ctx).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA D2H error: {}", e))
                })?;

                Ok((c_data.into_pyarray(py), argmax_data.into_pyarray(py)))
            }
            DeviceType::Cpu => {
                let a_ptr = TensorView::data_ptr(&a_tensor) as *const f32;
                let b_ptr = TensorView::data_ptr(&b_tensor) as *const f32;

                let a_data = unsafe { std::slice::from_raw_parts(a_ptr, m * k) };
                let b_data = unsafe { std::slice::from_raw_parts(b_ptr, k * n) };

                let result: ::tropical_gemm::GemmWithArgmax<TropicalMaxMul<f32>> =
                    ::tropical_gemm::tropical_matmul_with_argmax::<TropicalMaxMul<f32>>(
                        a_data, m, k, b_data, n,
                    );

                let c_scalars: Vec<f32> = result.values.iter().map(|x| x.value()).collect();
                let argmax_i32: Vec<i32> = result.argmax.iter().map(|&x| x as i32).collect();

                Ok((c_scalars.into_pyarray(py), argmax_i32.into_pyarray(py)))
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
