use tropical_gemm::{
    tropical_matmul_strided_batched, CountingTropical, Mat, TropicalGemm, TropicalMaxMul,
    TropicalMaxPlus, TropicalSemiring,
};
#[test]
fn builder_rejects_short_buffers_and_strides() {
    for (alen, blen, clen, lda, ldb, ldc, trans) in [
        (4, 4, 0, 2, 2, 2, false),
        (0, 4, 4, 2, 2, 2, false),
        (4, 0, 4, 2, 2, 2, false),
        (4, 4, 4, 1, 2, 2, false),
        (4, 4, 4, 2, 1, 2, false),
        (4, 4, 4, 2, 2, 1, false),
        (4, 4, 4, usize::MAX, 2, 2, true),
        (1, 4, 4, 2, 2, 2, true),
    ] {
        let result = std::panic::catch_unwind(|| {
            let mut c = vec![TropicalMaxPlus(0f32); clen];
            let mut g = TropicalGemm::<TropicalMaxPlus<f32>>::new(2, 2, 2);
            if trans {
                g = g.trans_a();
            }
            g.execute(&vec![1.; alen], lda, &vec![2.; blen], ldb, &mut c, ldc);
        });
        assert!(result.is_err());
    }
}
#[test]
fn builder_overwrites_only_logical_output_and_accumulates_k_panels() {
    let k = 1025;
    let mut a = vec![-5f32; k];
    a[0] = 10.;
    let b = vec![0f32; k * 2];
    let mut c = vec![TropicalMaxPlus(100f32); 3];
    TropicalGemm::new(1, 2, k).execute(&a, k, &b, 2, &mut c, 3);
    assert_eq!(
        c,
        [
            TropicalMaxPlus(10.),
            TropicalMaxPlus(10.),
            TropicalMaxPlus(100.)
        ]
    );
    TropicalGemm::new(1, 2, 0).execute(&[], 0, &[], 2, &mut c, 3);
    assert_eq!(c[0].0, f32::NEG_INFINITY);
    assert_eq!(c[2].0, 100.);
}
#[test]
fn zero_extent_batches_return_empty() {
    assert!(
        tropical_matmul_strided_batched::<TropicalMaxPlus<f32>>(&[], &[0.; 6], 1, 0, 2, 3)
            .is_empty()
    );
    assert!(
        tropical_matmul_strided_batched::<TropicalMaxPlus<f32>>(&[0.; 6], &[], 1, 3, 2, 0)
            .is_empty()
    );
}
#[test]
fn counting_view_projects_values_and_invalidates_cache_on_mutation() {
    let mut m = Mat::from_vec(
        vec![
            CountingTropical::new(2f64, 100f64),
            CountingTropical::new(3., 200.),
        ],
        1,
        2,
    );
    assert_eq!(m.as_ref().as_slice(), &[2., 3.]);
    m[(0, 1)].value = 4.;
    assert_eq!(m.as_ref().as_slice(), &[2., 4.]);
    m.as_mut_slice()[0].value = 5.;
    assert_eq!(m.as_ref().as_slice(), &[5., 4.]);
    assert_eq!(m.clone().as_ref().as_slice(), &[5., 4.]);
}
#[test]
fn transparent_views_still_borrow_without_copying() {
    let m = Mat::<TropicalMaxPlus<f64>>::from_col_major(&[2., 3.], 1, 2);
    assert_eq!(m.as_ref().as_slice().as_ptr(), m.as_slice().as_ptr().cast());
}
#[test]
fn maxmul_gradients_include_winning_operand() {
    let a = Mat::<TropicalMaxMul<f64>>::from_col_major(&[2.], 1, 1);
    let b = Mat::<TropicalMaxMul<f64>>::from_col_major(&[3.], 1, 1);
    let g = Mat::<TropicalMaxPlus<f64>>::from_col_major(&[1.], 1, 1);
    let result = a.matmul_argmax(&b);
    assert_eq!(result.backward_a_maxmul(&g, &b).get_value(0, 0), 3.);
    assert_eq!(result.backward_b_maxmul(&g, &a).get_value(0, 0), 2.);
}
#[derive(Copy, Clone, Debug, PartialEq)]
struct Byte(u8);
impl TropicalSemiring for Byte {
    type Scalar = f64;
    fn tropical_zero() -> Self {
        Self(0)
    }
    fn tropical_one() -> Self {
        Self(1)
    }
    fn tropical_add(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0))
    }
    fn tropical_mul(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
    fn from_scalar(v: f64) -> Self {
        Self(v as u8)
    }
    fn value(&self) -> f64 {
        self.0 as f64
    }
}
#[test]
fn custom_semiring_with_smaller_storage_has_safe_view() {
    let m = Mat::from_vec(vec![Byte(2), Byte(7)], 1, 2);
    assert_eq!(m.as_ref().as_slice(), &[2., 7.]);
}

#[test]
fn dimensions_cannot_wrap_in_release_builds() {
    use tropical_gemm::{tropical_matmul, MatRef};
    assert!(
        std::panic::catch_unwind(|| MatRef::<TropicalMaxPlus<f32>>::from_slice(
            &[],
            1usize << (usize::BITS - 1),
            2
        ))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| tropical_matmul::<TropicalMaxPlus<f32>>(
            &[],
            1usize << (usize::BITS - 1),
            2,
            &[],
            0
        ))
        .is_err()
    );
}
