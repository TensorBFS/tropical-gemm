use crate::types::TropicalScalar;

/// Matrix layout enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Row-major layout (C-style).
    RowMajor,
    /// Column-major layout (Fortran-style).
    ColMajor,
}

/// Transpose specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transpose {
    /// No transpose.
    NoTrans,
    /// Transpose the matrix.
    Trans,
}

/// Pack a panel of matrix A into a contiguous buffer.
///
/// The packed format stores `mc` rows in column-major order within
/// blocks of `mr` rows. This improves cache locality during the
/// microkernel computation.
///
/// # Layout
/// For A with dimensions m×k:
/// ```text
/// Original A (row-major, m=6, k=4, mr=4):
/// [ a00 a01 a02 a03 ]
/// [ a10 a11 a12 a13 ]
/// [ a20 a21 a22 a23 ]
/// [ a30 a31 a32 a33 ]
/// [ a40 a41 a42 a43 ]
/// [ a50 a51 a52 a53 ]
///
/// Packed (column-major within mr×k blocks):
/// Block 0 (rows 0-3): a00 a10 a20 a30 | a01 a11 a21 a31 | a02 a12 a22 a32 | a03 a13 a23 a33
/// Block 1 (rows 4-5): a40 a50 0   0   | a41 a51 0   0   | a42 a52 0   0   | a43 a53 0   0
/// ```
///
/// # Safety
/// - `a` must point to valid memory for at least `m * lda` elements
/// - `packed` must have capacity for at least `((m + mr - 1) / mr) * mr * k` elements
pub unsafe fn pack_a<T: TropicalScalar>(
    m: usize,
    k: usize,
    a: *const T,
    lda: usize,
    layout: Layout,
    trans: Transpose,
    packed: *mut T,
    mr: usize,
) {
    let zero = T::scalar_zero();

    let mut packed_idx = 0;

    // Process full mr×k blocks
    let m_blocks = m / mr;
    let m_rem = m % mr;

    for block in 0..m_blocks {
        let row_start = block * mr;
        for col in 0..k {
            for row_offset in 0..mr {
                let row = row_start + row_offset;
                let val = get_element(a, row, col, lda, layout, trans);
                *packed.add(packed_idx) = val;
                packed_idx += 1;
            }
        }
    }

    // Process remaining rows (if any)
    if m_rem > 0 {
        let row_start = m_blocks * mr;
        for col in 0..k {
            for row_offset in 0..mr {
                let row = row_start + row_offset;
                let val = if row < m {
                    get_element(a, row, col, lda, layout, trans)
                } else {
                    zero
                };
                *packed.add(packed_idx) = val;
                packed_idx += 1;
            }
        }
    }
}

/// Pack a panel of matrix B into a contiguous buffer.
///
/// The packed format stores `nc` columns in row-major order within
/// blocks of `nr` columns.
///
/// # Layout
/// For B with dimensions k×n:
/// ```text
/// Original B (row-major, k=3, n=6, nr=4):
/// [ b00 b01 b02 b03 b04 b05 ]
/// [ b10 b11 b12 b13 b14 b15 ]
/// [ b20 b21 b22 b23 b24 b25 ]
///
/// Packed (row-major within k×nr blocks):
/// Block 0 (cols 0-3): b00 b01 b02 b03 | b10 b11 b12 b13 | b20 b21 b22 b23
/// Block 1 (cols 4-5): b04 b05 0   0   | b14 b15 0   0   | b24 b25 0   0
/// ```
///
/// # Safety
/// - `b` must point to valid memory for at least `k * ldb` or `ldb * n` elements
/// - `packed` must have capacity for at least `((n + nr - 1) / nr) * nr * k` elements
pub unsafe fn pack_b<T: TropicalScalar>(
    k: usize,
    n: usize,
    b: *const T,
    ldb: usize,
    layout: Layout,
    trans: Transpose,
    packed: *mut T,
    nr: usize,
) {
    let zero = T::scalar_zero();

    let mut packed_idx = 0;

    // Process full k×nr blocks
    let n_blocks = n / nr;
    let n_rem = n % nr;

    for block in 0..n_blocks {
        let col_start = block * nr;
        for row in 0..k {
            for col_offset in 0..nr {
                let col = col_start + col_offset;
                let val = get_element(b, row, col, ldb, layout, trans);
                *packed.add(packed_idx) = val;
                packed_idx += 1;
            }
        }
    }

    // Process remaining columns (if any)
    if n_rem > 0 {
        let col_start = n_blocks * nr;
        for row in 0..k {
            for col_offset in 0..nr {
                let col = col_start + col_offset;
                let val = if col < n {
                    get_element(b, row, col, ldb, layout, trans)
                } else {
                    zero
                };
                *packed.add(packed_idx) = val;
                packed_idx += 1;
            }
        }
    }
}

/// Get element from matrix considering layout and transpose.
#[inline(always)]
unsafe fn get_element<T: Copy>(
    ptr: *const T,
    row: usize,
    col: usize,
    ld: usize,
    layout: Layout,
    trans: Transpose,
) -> T {
    let (actual_row, actual_col) = match trans {
        Transpose::NoTrans => (row, col),
        Transpose::Trans => (col, row),
    };

    let idx = match layout {
        Layout::RowMajor => actual_row * ld + actual_col,
        Layout::ColMajor => actual_col * ld + actual_row,
    };

    *ptr.add(idx)
}

/// Calculate packed buffer size for A.
#[inline]
pub fn packed_a_size(m: usize, k: usize, mr: usize) -> usize {
    let m_padded = m.div_ceil(mr) * mr;
    m_padded * k
}

/// Calculate packed buffer size for B.
#[inline]
pub fn packed_b_size(k: usize, n: usize, nr: usize) -> usize {
    let n_padded = n.div_ceil(nr) * nr;
    k * n_padded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_a_row_major() {
        let a: [f64; 6] = [
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];
        let m = 2;
        let k = 3;
        let mr = 4;
        let lda = 3;

        let mut packed = vec![0.0f64; packed_a_size(m, k, mr)];

        unsafe {
            pack_a(
                m,
                k,
                a.as_ptr(),
                lda,
                Layout::RowMajor,
                Transpose::NoTrans,
                packed.as_mut_ptr(),
                mr,
            );
        }

        // Expected: column 0: [1,4,0,0], column 1: [2,5,0,0], column 2: [3,6,0,0]
        assert_eq!(packed[0], 1.0); // a[0,0]
        assert_eq!(packed[1], 4.0); // a[1,0]
        assert_eq!(packed[2], 0.0); // padding
        assert_eq!(packed[3], 0.0); // padding
        assert_eq!(packed[4], 2.0); // a[0,1]
        assert_eq!(packed[5], 5.0); // a[1,1]
    }

    #[test]
    fn test_pack_a_col_major() {
        // Column-major: columns are stored contiguously
        // Matrix: [[1, 2, 3], [4, 5, 6]]
        // Col-major storage: [1, 4, 2, 5, 3, 6]
        let a: [f64; 6] = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let m = 2;
        let k = 3;
        let mr = 4;
        let lda = 2; // Leading dimension for col-major

        let mut packed = vec![0.0f64; packed_a_size(m, k, mr)];

        unsafe {
            pack_a(
                m,
                k,
                a.as_ptr(),
                lda,
                Layout::ColMajor,
                Transpose::NoTrans,
                packed.as_mut_ptr(),
                mr,
            );
        }

        // Same result as row-major since we're extracting the same logical matrix
        assert_eq!(packed[0], 1.0); // a[0,0]
        assert_eq!(packed[1], 4.0); // a[1,0]
        assert_eq!(packed[4], 2.0); // a[0,1]
        assert_eq!(packed[5], 5.0); // a[1,1]
    }

    #[test]
    fn test_pack_b_row_major() {
        let b: [f64; 6] = [
            1.0, 2.0, // row 0
            3.0, 4.0, // row 1
            5.0, 6.0, // row 2
        ];
        let k = 3;
        let n = 2;
        let nr = 4;
        let ldb = 2;

        let mut packed = vec![0.0f64; packed_b_size(k, n, nr)];

        unsafe {
            pack_b(
                k,
                n,
                b.as_ptr(),
                ldb,
                Layout::RowMajor,
                Transpose::NoTrans,
                packed.as_mut_ptr(),
                nr,
            );
        }

        // Expected: row 0: [1,2,0,0], row 1: [3,4,0,0], row 2: [5,6,0,0]
        assert_eq!(packed[0], 1.0); // b[0,0]
        assert_eq!(packed[1], 2.0); // b[0,1]
        assert_eq!(packed[2], 0.0); // padding
        assert_eq!(packed[3], 0.0); // padding
        assert_eq!(packed[4], 3.0); // b[1,0]
        assert_eq!(packed[5], 4.0); // b[1,1]
    }

    #[test]
    fn test_pack_b_col_major() {
        // Column-major: columns are stored contiguously
        // Matrix B (k=3, n=2): [[1, 2], [3, 4], [5, 6]]
        // Col-major storage: [1, 3, 5, 2, 4, 6]
        let b: [f64; 6] = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let k = 3;
        let n = 2;
        let nr = 4;
        let ldb = 3; // Leading dimension for col-major (number of rows)

        let mut packed = vec![0.0f64; packed_b_size(k, n, nr)];

        unsafe {
            pack_b(
                k,
                n,
                b.as_ptr(),
                ldb,
                Layout::ColMajor,
                Transpose::NoTrans,
                packed.as_mut_ptr(),
                nr,
            );
        }

        // Expected: same logical values as row-major
        assert_eq!(packed[0], 1.0); // b[0,0]
        assert_eq!(packed[1], 2.0); // b[0,1]
        assert_eq!(packed[4], 3.0); // b[1,0]
        assert_eq!(packed[5], 4.0); // b[1,1]
    }

    #[test]
    fn test_pack_a_with_transpose() {
        // Test packing with transpose
        let a: [f64; 6] = [
            1.0, 2.0, // row 0 (becomes col 0 after trans)
            3.0, 4.0, // row 1
            5.0, 6.0, // row 2
        ];
        let m = 2; // After transpose: original 2 columns become 2 rows
        let k = 3; // After transpose: original 3 rows become 3 cols
        let mr = 4;
        let lda = 2;

        let mut packed = vec![0.0f64; packed_a_size(m, k, mr)];

        unsafe {
            pack_a(
                m,
                k,
                a.as_ptr(),
                lda,
                Layout::RowMajor,
                Transpose::Trans,
                packed.as_mut_ptr(),
                mr,
            );
        }

        // A^T = [[1, 3, 5], [2, 4, 6]]
        assert_eq!(packed[0], 1.0); // a^T[0,0]
        assert_eq!(packed[1], 2.0); // a^T[1,0]
        assert_eq!(packed[4], 3.0); // a^T[0,1]
        assert_eq!(packed[5], 4.0); // a^T[1,1]
    }

    #[test]
    fn test_pack_b_with_transpose() {
        // Test packing B with transpose
        let b: [f64; 6] = [
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];
        let k = 3; // After transpose: original 3 cols become 3 rows
        let n = 2; // After transpose: original 2 rows become 2 cols
        let nr = 4;
        let ldb = 3;

        let mut packed = vec![0.0f64; packed_b_size(k, n, nr)];

        unsafe {
            pack_b(
                k,
                n,
                b.as_ptr(),
                ldb,
                Layout::RowMajor,
                Transpose::Trans,
                packed.as_mut_ptr(),
                nr,
            );
        }

        // B^T = [[1, 4], [2, 5], [3, 6]]
        assert_eq!(packed[0], 1.0); // b^T[0,0]
        assert_eq!(packed[1], 4.0); // b^T[0,1]
        assert_eq!(packed[4], 2.0); // b^T[1,0]
        assert_eq!(packed[5], 5.0); // b^T[1,1]
    }

    #[test]
    fn test_pack_a_exact_mr() {
        // Test when m is exactly divisible by mr (no remainder path)
        let a: [f64; 12] = [
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
            10.0, 11.0, 12.0, // row 3
        ];
        let m = 4;
        let k = 3;
        let mr = 4;
        let lda = 3;

        let mut packed = vec![0.0f64; packed_a_size(m, k, mr)];

        unsafe {
            pack_a(
                m,
                k,
                a.as_ptr(),
                lda,
                Layout::RowMajor,
                Transpose::NoTrans,
                packed.as_mut_ptr(),
                mr,
            );
        }

        // No padding needed
        assert_eq!(packed[0], 1.0);
        assert_eq!(packed[1], 4.0);
        assert_eq!(packed[2], 7.0);
        assert_eq!(packed[3], 10.0);
    }

    #[test]
    fn test_pack_b_exact_nr() {
        // Test when n is exactly divisible by nr (no remainder path)
        let b: [f64; 12] = [
            1.0, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
            9.0, 10.0, 11.0, 12.0, // row 2
        ];
        let k = 3;
        let n = 4;
        let nr = 4;
        let ldb = 4;

        let mut packed = vec![0.0f64; packed_b_size(k, n, nr)];

        unsafe {
            pack_b(
                k,
                n,
                b.as_ptr(),
                ldb,
                Layout::RowMajor,
                Transpose::NoTrans,
                packed.as_mut_ptr(),
                nr,
            );
        }

        // No padding needed
        assert_eq!(packed[0], 1.0);
        assert_eq!(packed[1], 2.0);
        assert_eq!(packed[2], 3.0);
        assert_eq!(packed[3], 4.0);
    }

    #[test]
    fn test_packed_a_size() {
        // Exact multiple of mr
        assert_eq!(packed_a_size(8, 10, 4), 8 * 10);
        // Needs padding: m=5, mr=4 -> m_padded=8
        assert_eq!(packed_a_size(5, 10, 4), 8 * 10);
        // m=1, mr=4 -> m_padded=4
        assert_eq!(packed_a_size(1, 10, 4), 4 * 10);
    }

    #[test]
    fn test_packed_b_size() {
        // Exact multiple of nr
        assert_eq!(packed_b_size(10, 8, 4), 10 * 8);
        // Needs padding: n=5, nr=4 -> n_padded=8
        assert_eq!(packed_b_size(10, 5, 4), 10 * 8);
        // n=1, nr=4 -> n_padded=4
        assert_eq!(packed_b_size(10, 1, 4), 10 * 4);
    }

    #[test]
    fn test_layout_debug() {
        assert_eq!(format!("{:?}", Layout::RowMajor), "RowMajor");
        assert_eq!(format!("{:?}", Layout::ColMajor), "ColMajor");
    }

    #[test]
    fn test_layout_clone_eq() {
        let l1 = Layout::RowMajor;
        let l2 = l1;
        assert_eq!(l1, l2);
        assert_ne!(l1, Layout::ColMajor);
    }

    #[test]
    fn test_transpose_debug() {
        assert_eq!(format!("{:?}", Transpose::NoTrans), "NoTrans");
        assert_eq!(format!("{:?}", Transpose::Trans), "Trans");
    }

    #[test]
    fn test_transpose_clone_eq() {
        let t1 = Transpose::Trans;
        let t2 = t1;
        assert_eq!(t1, t2);
        assert_ne!(t1, Transpose::NoTrans);
    }
}
