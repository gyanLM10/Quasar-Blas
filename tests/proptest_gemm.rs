//! Property-based tests for GEMM correctness using proptest.
//!
//! These tests validate ALL GEMM engines against ndarray::Array2::dot() as the
//! ground-truth oracle. They generate random matrix dimensions (including non-power-of-2
//! and edge cases) and random element values, proving mathematical correctness
//! across thousands of randomized inputs.
//!
//! Properties tested:
//! 1. **Oracle correctness**: our result matches ndarray's dot product within ε
//! 2. **Identity**: A × I = A
//! 3. **Transpose**: (A × B)ᵀ = Bᵀ × Aᵀ

use proptest::prelude::*;
use ndarray::Array2;

use quasar_blas::GemmEngine;
use quasar_blas::cpu::{NaiveGemm, TiledGemm, SimdGemm};

/// Maximum dimension for generated matrices (kept small for fast test execution).
const MAX_DIM: usize = 64;

/// Absolute tolerance for f32 comparisons.
/// f32 has ~7 decimal digits of precision; with accumulation over MAX_DIM
/// multiply-adds, we need a bit of slack.
const EPSILON: f32 = 1e-3;

/// Strategy to generate a random (rows, cols) pair.
fn dim_strategy() -> impl Strategy<Value = (usize, usize)> {
    (1..=MAX_DIM, 1..=MAX_DIM)
}

/// Strategy to generate a flat f32 vector of given length with bounded values.
fn matrix_data_strategy(len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..=10.0f32, len)
}

/// Helper: run GEMM on a given engine and compare against ndarray oracle.
fn validate_against_oracle<E: GemmEngine<f32>>(
    engine: &E,
    m: usize,
    k: usize,
    n: usize,
    a_data: &[f32],
    b_data: &[f32],
) -> Result<(), TestCaseError> {
    // Compute reference result using ndarray
    let a_nd = Array2::from_shape_vec((m, k), a_data.to_vec()).unwrap();
    let b_nd = Array2::from_shape_vec((k, n), b_data.to_vec()).unwrap();
    let c_ref = a_nd.dot(&b_nd);

    // Compute our result
    let mut c_ours = vec![0.0f32; m * n];
    engine
        .gemm(m, k, n, a_data, k, b_data, n, &mut c_ours, n)
        .map_err(|e| TestCaseError::fail(format!("GEMM failed: {:?}", e)))?;

    // Compare element-by-element
    for i in 0..m {
        for j in 0..n {
            let expected = c_ref[[i, j]];
            let actual = c_ours[i * n + j];
            let diff = (expected - actual).abs();
            prop_assert!(
                diff < EPSILON,
                "Mismatch at [{}, {}]: expected={}, actual={}, diff={} (m={}, k={}, n={})",
                i, j, expected, actual, diff, m, k, n
            );
        }
    }
    Ok(())
}

// ===========================================================================
// Property Tests — Naive Engine
// ===========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn naive_matches_oracle(
        (m, k) in dim_strategy(),
        n in 1..=MAX_DIM,
        a_seed in 0u64..1000,
        b_seed in 0u64..1000,
    ) {
        // Use seeds to deterministically generate matrix data of the right size
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((a_seed.wrapping_mul(i as u64 + 1).wrapping_add(7)) % 2000) as f32 / 100.0 - 10.0)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((b_seed.wrapping_mul(i as u64 + 1).wrapping_add(13)) % 2000) as f32 / 100.0 - 10.0)
            .collect();

        validate_against_oracle(&NaiveGemm, m, k, n, &a_data, &b_data)?;
    }
}

// ===========================================================================
// Property Tests — Tiled Engine (TILE=8, small for fast testing)
// ===========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn tiled_matches_oracle(
        (m, k) in dim_strategy(),
        n in 1..=MAX_DIM,
        a_seed in 0u64..1000,
        b_seed in 0u64..1000,
    ) {
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((a_seed.wrapping_mul(i as u64 + 1).wrapping_add(7)) % 2000) as f32 / 100.0 - 10.0)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((b_seed.wrapping_mul(i as u64 + 1).wrapping_add(13)) % 2000) as f32 / 100.0 - 10.0)
            .collect();

        validate_against_oracle(&TiledGemm::<8>, m, k, n, &a_data, &b_data)?;
    }
}

// ===========================================================================
// Property Tests — SIMD Engine
// ===========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn simd_matches_oracle(
        (m, k) in dim_strategy(),
        n in 1..=MAX_DIM,
        a_seed in 0u64..1000,
        b_seed in 0u64..1000,
    ) {
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((a_seed.wrapping_mul(i as u64 + 1).wrapping_add(7)) % 2000) as f32 / 100.0 - 10.0)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((b_seed.wrapping_mul(i as u64 + 1).wrapping_add(13)) % 2000) as f32 / 100.0 - 10.0)
            .collect();

        validate_against_oracle(&SimdGemm::<64>, m, k, n, &a_data, &b_data)?;
    }
}

// ===========================================================================
// Algebraic Property Tests
// ===========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// A × I = A (identity property)
    #[test]
    fn identity_property(
        m in 1..=32usize,
        n in 1..=32usize,
        a_seed in 0u64..500,
    ) {
        let a_data: Vec<f32> = (0..m * n)
            .map(|i| ((a_seed.wrapping_mul(i as u64 + 1).wrapping_add(7)) % 2000) as f32 / 100.0 - 10.0)
            .collect();

        // Build n×n identity matrix
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }

        let mut c = vec![0.0f32; m * n];
        NaiveGemm
            .gemm(m, n, n, &a_data, n, &identity, n, &mut c, n)
            .unwrap();

        for idx in 0..m * n {
            prop_assert!(
                (a_data[idx] - c[idx]).abs() < EPSILON,
                "Identity failed at index {}: expected={}, got={}",
                idx, a_data[idx], c[idx]
            );
        }
    }

    /// (A × B)ᵀ = Bᵀ × Aᵀ (transpose property)
    #[test]
    fn transpose_property(
        m in 1..=16usize,
        k in 1..=16usize,
        n in 1..=16usize,
        a_seed in 0u64..500,
        b_seed in 0u64..500,
    ) {
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((a_seed.wrapping_mul(i as u64 + 1).wrapping_add(7)) % 2000) as f32 / 100.0 - 10.0)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((b_seed.wrapping_mul(i as u64 + 1).wrapping_add(13)) % 2000) as f32 / 100.0 - 10.0)
            .collect();

        // Compute C = A × B
        let mut c = vec![0.0f32; m * n];
        NaiveGemm.gemm(m, k, n, &a_data, k, &b_data, n, &mut c, n).unwrap();

        // Transpose A → Aᵀ (k×m)
        let mut a_t = vec![0.0f32; k * m];
        for i in 0..m {
            for j in 0..k {
                a_t[j * m + i] = a_data[i * k + j];
            }
        }

        // Transpose B → Bᵀ (n×k)
        let mut b_t = vec![0.0f32; n * k];
        for i in 0..k {
            for j in 0..n {
                b_t[j * k + i] = b_data[i * n + j];
            }
        }

        // Compute D = Bᵀ × Aᵀ (n×k × k×m = n×m)
        let mut d = vec![0.0f32; n * m];
        NaiveGemm.gemm(n, k, m, &b_t, k, &a_t, m, &mut d, m).unwrap();

        // (A×B)ᵀ should equal Bᵀ×Aᵀ
        // C is m×n, so Cᵀ[j,i] = C[i,j]
        for i in 0..m {
            for j in 0..n {
                let c_t_ji = c[i * n + j]; // C[i,j] = Cᵀ[j,i]
                let d_ji = d[j * m + i];   // D[j,i]
                prop_assert!(
                    (c_t_ji - d_ji).abs() < EPSILON,
                    "Transpose failed at [{},{}]: (AB)^T={}, B^T A^T={}",
                    j, i, c_t_ji, d_ji
                );
            }
        }
    }
}
