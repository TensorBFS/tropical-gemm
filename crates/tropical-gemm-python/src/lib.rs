//! Python bindings for tropical matrix multiplication.
//!
//! This module provides Python/NumPy bindings for tropical GEMM operations,
//! enabling integration with PyTorch custom autograd functions.

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

/// Tropical GEMM Python module.
#[pymodule]
fn tropical_gemm(m: &Bound<'_, PyModule>) -> PyResult<()> {
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

    // i32 operations
    m.add_function(wrap_pyfunction!(maxplus_matmul_i32, m)?)?;
    m.add_function(wrap_pyfunction!(minplus_matmul_i32, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_matmul_i32, m)?)?;

    // i64 operations
    m.add_function(wrap_pyfunction!(maxplus_matmul_i64, m)?)?;
    m.add_function(wrap_pyfunction!(minplus_matmul_i64, m)?)?;
    m.add_function(wrap_pyfunction!(maxmul_matmul_i64, m)?)?;

    Ok(())
}
