//! Shared deterministic test-data generators for the Metal integration tests.
//! Each integration test is its own crate, so this lives in `tests/common/`
//! and is pulled in via `mod common;`. Not every binary uses every helper.
#![allow(dead_code)]

pub fn f32_data(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let v = i
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503))
                % 2000;
            (v as f32) * 0.01 - 10.0
        })
        .collect()
}

pub fn i32_data(len: usize, salt: usize) -> Vec<i32> {
    (0..len)
        .map(|i| {
            (i.wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503))
                % 2001) as i32
                - 1000
        })
        .collect()
}

pub fn i64_data(len: usize, salt: usize) -> Vec<i64> {
    (0..len)
        .map(|i| {
            (i.wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503))
                % 2_000_001) as i64
                - 1_000_000
        })
        .collect()
}
