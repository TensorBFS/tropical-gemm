use tropical_gemm::{
    core::{tropical_gemm_portable, tropical_gemm_with_argmax_portable, GemmWithArgmax, Transpose},
    simd::{tropical_gemm_dispatch, tropical_gemm_with_argmax_dispatch},
    TropicalBitwise, TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus,
};

macro_rules! float_parity {
    ($test:ident, $semiring:ident, $scalar:ty) => {
        #[test]
        fn $test() {
            type S = $semiring<$scalar>;
            for (m, n, k) in [
                (1, 1, 0),
                (1, 9, 1),
                (7, 11, 17),
                (9, 5, 513),
                (17, 129, 257),
            ] {
                for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
                    for special in [false, true] {
                        let av = |i: usize, p: usize| -> $scalar {
                            if special {
                                let v = [
                                    0.0,
                                    -0.0,
                                    <$scalar>::INFINITY,
                                    <$scalar>::NEG_INFINITY,
                                    <$scalar>::NAN,
                                    1.0,
                                    2.0,
                                ];
                                v[(i + p) % v.len()]
                            } else {
                                ((i * 11 + p * 7) % 19) as $scalar * 0.125
                            }
                        };
                        let bv = |p: usize, j: usize| -> $scalar {
                            if special {
                                let v = [
                                    -0.0,
                                    0.0,
                                    <$scalar>::NEG_INFINITY,
                                    <$scalar>::INFINITY,
                                    2.0,
                                    1.0,
                                    <$scalar>::NAN,
                                ];
                                v[(p + j * 3) % v.len()]
                            } else {
                                ((p * 3 + j) % 13) as $scalar * 0.25
                            }
                        };
                        let lda = if ta { m + 3 } else { k + 3 };
                        let ldb = if tb { k + 5 } else { n + 5 };
                        let mut a = vec![0.0; (if ta { k } else { m }) * lda];
                        let mut b = vec![0.0; (if tb { n } else { k }) * ldb];
                        for i in 0..m {
                            for p in 0..k {
                                a[if ta { p * lda + i } else { i * lda + p }] = av(i, p);
                            }
                        }
                        for p in 0..k {
                            for j in 0..n {
                                b[if tb { j * ldb + p } else { p * ldb + j }] = bv(p, j);
                            }
                        }
                        let mut expected = GemmWithArgmax::<S>::with_ld(m, n, n + 7);
                        expected.values.fill($semiring(1234.0));
                        expected.argmax.fill(u32::MAX);
                        let mut actual = expected.clone();
                        let trans_a = if ta {
                            Transpose::Trans
                        } else {
                            Transpose::NoTrans
                        };
                        let trans_b = if tb {
                            Transpose::Trans
                        } else {
                            Transpose::NoTrans
                        };
                        unsafe {
                            tropical_gemm_with_argmax_portable(
                                m,
                                n,
                                k,
                                a.as_ptr(),
                                lda,
                                trans_a,
                                b.as_ptr(),
                                ldb,
                                trans_b,
                                &mut expected,
                            );
                            tropical_gemm_with_argmax_dispatch(
                                m,
                                n,
                                k,
                                a.as_ptr(),
                                lda,
                                trans_a,
                                b.as_ptr(),
                                ldb,
                                trans_b,
                                &mut actual,
                            );
                        }
                        assert_eq!(
                            actual.argmax, expected.argmax,
                            "indices: {m}x{n}x{k}, ta={ta}, tb={tb}, special={special}"
                        );
                        for (x, y) in actual.values.iter().zip(&expected.values) {
                            assert_eq!(
                                x.0.to_bits(),
                                y.0.to_bits(),
                                "values: {m}x{n}x{k}, ta={ta}, tb={tb}, special={special}"
                            );
                        }
                    }
                }
            }
        }
    };
}
float_parity!(maxplus_f32, TropicalMaxPlus, f32);
float_parity!(minplus_f32, TropicalMinPlus, f32);
float_parity!(maxmul_f32, TropicalMaxMul, f32);
float_parity!(maxplus_f64, TropicalMaxPlus, f64);
float_parity!(minplus_f64, TropicalMinPlus, f64);
float_parity!(maxmul_f64, TropicalMaxMul, f64);

macro_rules! bitwise_parity {
    ($test:ident, $scalar:ty) => {
        #[test]
        fn $test() {
            for (m, n, k) in [(1, 9, 0), (7, 13, 3), (9, 11, 513), (257, 33, 513)] {
                for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
                    let lda = if ta { m + 3 } else { k + 3 };
                    let ldb = if tb { k + 5 } else { n + 5 };
                    let mut a = vec![0; (if ta { k } else { m }) * lda];
                    let mut b = vec![0; (if tb { n } else { k }) * ldb];
                    for i in 0..m {
                        for p in 0..k {
                            // Individual high/low bits ensure lane order and width matter;
                            // random dense words would saturate OR to all ones too quickly.
                            a[if ta { p * lda + i } else { i * lda + p }] = (1 as $scalar)
                                .rotate_left(((i + p) % <$scalar>::BITS as usize) as u32);
                        }
                    }
                    for p in 0..k {
                        for j in 0..n {
                            b[if tb { j * ldb + p } else { p * ldb + j }] =
                                (1 as $scalar).rotate_left((j % <$scalar>::BITS as usize) as u32);
                        }
                    }
                    let mut expected = vec![TropicalBitwise(<$scalar>::MAX); m * (n + 7)];
                    let mut actual = expected.clone();
                    let trans_a = if ta {
                        Transpose::Trans
                    } else {
                        Transpose::NoTrans
                    };
                    let trans_b = if tb {
                        Transpose::Trans
                    } else {
                        Transpose::NoTrans
                    };
                    unsafe {
                        tropical_gemm_portable(
                            m,
                            n,
                            k,
                            a.as_ptr(),
                            lda,
                            trans_a,
                            b.as_ptr(),
                            ldb,
                            trans_b,
                            expected.as_mut_ptr(),
                            n + 7,
                        );
                        tropical_gemm_dispatch(
                            m,
                            n,
                            k,
                            a.as_ptr(),
                            lda,
                            trans_a,
                            b.as_ptr(),
                            ldb,
                            trans_b,
                            actual.as_mut_ptr(),
                            n + 7,
                        );
                    }
                    assert_eq!(actual, expected, "{m}x{n}x{k}, ta={ta}, tb={tb}");
                }
            }
        }
    };
}
bitwise_parity!(bitwise_u32, u32);
bitwise_parity!(bitwise_u64, u64);

#[test]
fn andor_public_api_matches_boolean_reference_and_maxplus_encoding() {
    use tropical_gemm::{tropical_matmul, tropical_matmul_with_argmax, TropicalAndOr};
    let (m, n, k) = (7, 11, 19);
    let a: Vec<bool> = (0..m * k).map(|p| p / k == p % k).collect();
    let b: Vec<bool> = (0..k * n).map(|p| p / n == (p % n) * 3 % k).collect();
    let c = tropical_matmul::<TropicalAndOr>(&a, m, k, &b, n);
    let indices = tropical_matmul_with_argmax::<TropicalAndOr>(&a, m, k, &b, n);
    let encode = |v: &[bool]| {
        v.iter()
            .map(|&x| if x { 0.0f32 } else { f32::NEG_INFINITY })
            .collect::<Vec<_>>()
    };
    let encoded = tropical_matmul::<TropicalMaxPlus<f32>>(&encode(&a), m, k, &encode(&b), n);
    for i in 0..m {
        for j in 0..n {
            let winner = (0..k).find(|&p| a[i * k + p] && b[p * n + j]);
            assert_eq!(c[i * n + j].0, winner.is_some());
            assert_eq!(indices.values[i * n + j], c[i * n + j]);
            assert_eq!(indices.argmax[i * n + j], winner.unwrap_or(0) as u32);
            assert_eq!(c[i * n + j].0, encoded[i * n + j].0 == 0.0);
        }
    }
}
