//! Mutable matrix reference type.

use crate::types::TropicalSemiring;

/// Mutable view over semiring data.
///
/// Unlike `MatRef`, this holds mutable references to semiring values,
/// not scalars. This is used for in-place operations.
#[derive(Debug)]
pub struct MatMut<'a, S: TropicalSemiring> {
    data: &'a mut [S],
    nrows: usize,
    ncols: usize,
}

impl<'a, S: TropicalSemiring> MatMut<'a, S> {
    /// Create a mutable matrix reference from a slice.
    pub fn from_slice(data: &'a mut [S], nrows: usize, ncols: usize) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "data length {} != nrows {} * ncols {}",
            data.len(),
            nrows,
            ncols
        );
        Self { data, nrows, ncols }
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Get the underlying data as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        self.data
    }

    /// Get a mutable pointer to the data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        self.data.as_mut_ptr()
    }

    /// Get a reference to the value at position (i, j).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &S {
        debug_assert!(
            i < self.nrows,
            "row index {} out of bounds {}",
            i,
            self.nrows
        );
        debug_assert!(
            j < self.ncols,
            "col index {} out of bounds {}",
            j,
            self.ncols
        );
        // Column-major indexing
        &self.data[j * self.nrows + i]
    }

    /// Get a mutable reference to the value at position (i, j).
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut S {
        debug_assert!(
            i < self.nrows,
            "row index {} out of bounds {}",
            i,
            self.nrows
        );
        debug_assert!(
            j < self.ncols,
            "col index {} out of bounds {}",
            j,
            self.ncols
        );
        // Column-major indexing
        &mut self.data[j * self.nrows + i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TropicalMaxPlus;

    #[test]
    fn test_matmut_from_slice() {
        let mut data = vec![
            TropicalMaxPlus(1.0f64),
            TropicalMaxPlus(2.0),
            TropicalMaxPlus(3.0),
            TropicalMaxPlus(4.0),
        ];
        let m = MatMut::from_slice(&mut data, 2, 2);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
    }

    #[test]
    fn test_matmut_get() {
        // Column-major: data stored column-by-column
        // For 2×2 matrix [[1,2],[3,4]], col-major is [1,3,2,4]
        let mut data = vec![
            TropicalMaxPlus(1.0f64),
            TropicalMaxPlus(3.0),
            TropicalMaxPlus(2.0),
            TropicalMaxPlus(4.0),
        ];
        let m = MatMut::from_slice(&mut data, 2, 2);
        assert_eq!(m.get(0, 0).0, 1.0);
        assert_eq!(m.get(0, 1).0, 2.0);
        assert_eq!(m.get(1, 0).0, 3.0);
        assert_eq!(m.get(1, 1).0, 4.0);
    }

    #[test]
    fn test_matmut_get_mut() {
        let mut data = vec![
            TropicalMaxPlus(1.0f64),
            TropicalMaxPlus(2.0),
            TropicalMaxPlus(3.0),
            TropicalMaxPlus(4.0),
        ];
        let mut m = MatMut::from_slice(&mut data, 2, 2);
        *m.get_mut(0, 0) = TropicalMaxPlus(10.0);
        assert_eq!(m.get(0, 0).0, 10.0);
    }

    #[test]
    fn test_matmut_as_mut_slice() {
        let mut data = vec![
            TropicalMaxPlus(1.0f64),
            TropicalMaxPlus(2.0),
            TropicalMaxPlus(3.0),
            TropicalMaxPlus(4.0),
        ];
        let mut m = MatMut::from_slice(&mut data, 2, 2);
        let slice = m.as_mut_slice();
        slice[0] = TropicalMaxPlus(100.0);
        assert_eq!(data[0].0, 100.0);
    }

    #[test]
    fn test_matmut_as_mut_ptr() {
        let mut data = vec![
            TropicalMaxPlus(1.0f64),
            TropicalMaxPlus(2.0),
            TropicalMaxPlus(3.0),
            TropicalMaxPlus(4.0),
        ];
        let mut m = MatMut::from_slice(&mut data, 2, 2);
        let ptr = m.as_mut_ptr();
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_matmut_debug() {
        let mut data = vec![TropicalMaxPlus(1.0f64), TropicalMaxPlus(2.0)];
        let m = MatMut::from_slice(&mut data, 1, 2);
        let debug_str = format!("{:?}", m);
        assert!(debug_str.contains("MatMut"));
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_matmut_size_mismatch() {
        let mut data = vec![TropicalMaxPlus(1.0f64), TropicalMaxPlus(2.0)];
        let _ = MatMut::from_slice(&mut data, 2, 2); // Should panic
    }
}
