//! Portable SIMD-accelerated tiled GEMM engine.
//!
//! This is the highest-performance CPU kernel. It combines the cache-tiling strategy
//! from [`super::tiled`] with portable SIMD (`core::simd`) to process 4 `f32` elements
//! per instruction cycle using vector registers.
//!
//! ## Key SIMD Operations
//!
//! - `Simd::from_slice` — load 4×f32 from memory into a vector register
//! - `Simd::splat` — broadcast a single f32 across all 4 lanes
//! - `vfmaq_f32` equivalent — fused multiply-accumulate via `c_vec + a_vec * b_vec`
//!   (which compiles to FMA if the target supports it, e.g. Neon or FMA3)
//! - `copy_to_slice` — store 4×f32 from register back to memory
//!
//! ## Portability
//!
//! By using `core::simd`, this module automatically compiles to the optimal
//! vector instructions for the target platform (e.g., Neon on ARM, AVX2/AVX-512 on x86).

use crate::types::{GemmError, validate_gemm_dims};
use crate::GemmEngine;

/// SIMD-accelerated tiled GEMM engine.
///
/// Uses ARM Neon 128-bit vector registers to process 4 f32 elements simultaneously
/// within a cache-tiled framework for maximum throughput.
///
/// On non-aarch64 targets, this transparently falls back to scalar tiled code.
pub struct SimdGemm<const TILE: usize>;

/// Default SIMD engine with 64×64 tiles for Apple Silicon.
pub type SimdGemm64 = SimdGemm<64>;

// ==========================================================================
// Portable SIMD implementation
// ==========================================================================
mod portable {
    use super::*;
    use core::simd::f32x4;
    use std::simd::StdFloat;

    /// The number of f32 elements that fit in a 128-bit vector register.
    const SIMD_LANES: usize = 4;

    impl<const TILE: usize> GemmEngine<f32> for SimdGemm<TILE> {
        type Error = GemmError;

        fn gemm(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a: &[f32],
            lda: usize,
            b: &[f32],
            ldb: usize,
            c: &mut [f32],
            ldc: usize,
        ) -> Result<(), Self::Error> {
            validate_gemm_dims(m, k, n, lda, ldb, ldc, a.len(), b.len(), c.len())?;

            // Zero the output matrix
            for i in 0..m {
                for j in 0..n {
                    c[i * ldc + j] = 0.0;
                }
            }

            // Block loop: iterate over TILE-sized blocks
            let mut ii = 0;
            while ii < m {
                let i_end = core::cmp::min(ii + TILE, m);

                let mut kk = 0;
                while kk < k {
                    let k_end = core::cmp::min(kk + TILE, k);

                    let mut jj = 0;
                    while jj < n {
                        let j_end = core::cmp::min(jj + TILE, n);
                        let j_len = j_end - jj;

                        // How many full SIMD vectors fit in this j-strip?
                        let j_simd_end = jj + (j_len / SIMD_LANES) * SIMD_LANES;

                        // Micro-kernel with portable SIMD on the innermost j-loop
                        for i in ii..i_end {
                            for p in kk..k_end {
                                let a_val = a[i * lda + p];

                                // Broadcast A[i,p] to all 4 lanes
                                let a_vec = f32x4::splat(a_val);

                                let mut j = jj;
                                while j < j_simd_end {
                                    // Load 4 elements of B[p, j..j+4]
                                    let b_vec = f32x4::from_slice(&b[p * ldb + j..]);

                                    // Load 4 elements of C[i, j..j+4]
                                    let c_vec = f32x4::from_slice(&c[i * ldc + j..]);

                                    // FMA: c_vec = c_vec + a_vec * b_vec
                                    // `mul_add` intrinsic ensures hardware FMA if available
                                    let result = a_vec.mul_add(b_vec, c_vec);

                                    // Store back
                                    result.copy_to_slice(&mut c[i * ldc + j..i * ldc + j + SIMD_LANES]);

                                    j += SIMD_LANES;
                                }

                                // Scalar remainder for j values not aligned to SIMD width
                                for j in j_simd_end..j_end {
                                    c[i * ldc + j] = a_val.mul_add(b[p * ldb + j], c[i * ldc + j]);
                                }
                            }
                        }

                        jj += TILE;
                    }
                    kk += TILE;
                }
                ii += TILE;
            }

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::NaiveGemm;

    #[test]
    fn test_simd_matches_naive_small() {
        let m = 7;
        let k = 5;
        let n = 9; // Non-multiple of 4 to test scalar remainder path

        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.3 - 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.2 - 0.5).collect();
        let mut c_naive = vec![0.0f32; m * n];
        let mut c_simd = vec![0.0f32; m * n];

        NaiveGemm.gemm(m, k, n, &a, k, &b, n, &mut c_naive, n).unwrap();
        SimdGemm::<4>.gemm(m, k, n, &a, k, &b, n, &mut c_simd, n).unwrap();

        for idx in 0..m * n {
            assert!(
                (c_naive[idx] - c_simd[idx]).abs() < 1e-4,
                "Mismatch at [{}, {}]: naive={}, simd={}",
                idx / n,
                idx % n,
                c_naive[idx],
                c_simd[idx]
            );
        }
    }

    #[test]
    fn test_simd_matches_naive_large() {
        let m = 65; // One past a tile boundary
        let k = 33;
        let n = 66;

        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 100) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 13 + 5) % 100) as f32 * 0.01).collect();
        let mut c_naive = vec![0.0f32; m * n];
        let mut c_simd = vec![0.0f32; m * n];

        NaiveGemm.gemm(m, k, n, &a, k, &b, n, &mut c_naive, n).unwrap();
        SimdGemm::<64>.gemm(m, k, n, &a, k, &b, n, &mut c_simd, n).unwrap();

        for idx in 0..m * n {
            assert!(
                (c_naive[idx] - c_simd[idx]).abs() < 1e-3,
                "Mismatch at [{}, {}]: naive={}, simd={}",
                idx / n,
                idx % n,
                c_naive[idx],
                c_simd[idx]
            );
        }
    }

    #[test]
    fn test_simd_with_strides() {
        let engine = SimdGemm::<8>;
        // A (2×2) with lda=4
        let a = [1.0f32, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 5]; // ldc=3

        engine.gemm(2, 2, 2, &a, 4, &b, 2, &mut c, 3).unwrap();

        assert_eq!(c[0], 19.0);
        assert_eq!(c[1], 22.0);
        assert_eq!(c[3], 43.0);
        assert_eq!(c[4], 50.0);
    }
}
