//! Cache-tiled GEMM with compile-time const-generic tile size.
//!
//! This implementation breaks the large matrix multiplication into small
//! `TILE × TILE` blocks that fit entirely inside the L1 data cache, dramatically
//! reducing cache misses compared to the naive implementation.
//!
//! ## Const Generics
//!
//! The tile size is a compile-time constant (`const TILE: usize`), meaning:
//! - The compiler unrolls tile loops into raw assembly with zero runtime overhead
//! - Different instantiations can be compiled for different cache architectures:
//!   - `TiledGemm::<64>` for Apple M-series (128KB L1d)
//!   - `TiledGemm::<128>` for AMD EPYC (32KB L1d per core, large L2)
//!
//! ## Loop Order
//!
//! The critical optimization is the loop order within each tile:
//! `block_i → block_k → block_j → i → k → j`
//!
//! The innermost `j` loop writes sequentially to C and reads sequentially from B,
//! both of which are cache-friendly (spatial locality). The `k` loop is second-innermost,
//! broadcasting a single value of A[i,k] across the j-loop (temporal locality).

use crate::types::{GemmElement, GemmError, validate_gemm_dims};
use crate::GemmEngine;

/// Cache-tiled GEMM engine with compile-time tile size.
///
/// # Type Parameter
/// - `TILE`: The side length of the square tile. Must be > 0.
///   Recommended values: 32, 64, 128 depending on L1 cache size.
///
/// # Cache Math (Apple M-series example)
///
/// With `TILE = 64` and `f32` elements:
/// - One tile = 64 × 64 × 4 bytes = 16 KB
/// - We hold 3 tiles simultaneously (A_tile, B_tile, C_tile) = 48 KB
/// - Apple M-series L1d = 128 KB → all 3 tiles fit with room to spare
pub struct TiledGemm<const TILE: usize>;

impl<const TILE: usize, T: GemmElement> GemmEngine<T> for TiledGemm<TILE> {
    type Error = GemmError;

    fn gemm(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[T],
        lda: usize,
        b: &[T],
        ldb: usize,
        c: &mut [T],
        ldc: usize,
    ) -> Result<(), Self::Error> {
        validate_gemm_dims(m, k, n, lda, ldb, ldc, a.len(), b.len(), c.len())?;

        // Zero the output matrix
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] = T::zero();
            }
        }

        // Block loop: iterate over TILE-sized blocks
        // Order: block_i → block_k → block_j
        //
        // Why block_k is before block_j:
        // This means we fully accumulate A_block × B_block contributions to C_block
        // before moving to the next C_block column, maximizing C_block reuse in cache.
        let mut ii = 0;
        while ii < m {
            let i_end = core::cmp::min(ii + TILE, m);

            let mut kk = 0;
            while kk < k {
                let k_end = core::cmp::min(kk + TILE, k);

                let mut jj = 0;
                while jj < n {
                    let j_end = core::cmp::min(jj + TILE, n);

                    // Micro-kernel: multiply the tile intersection
                    // Inner loop order: i → k → j (cache-optimal)
                    for i in ii..i_end {
                        for p in kk..k_end {
                            let a_ip = a[i * lda + p];
                            for j in jj..j_end {
                                // C[i,j] += A[i,p] * B[p,j]
                                c[i * ldc + j] = a_ip.mul_add(b[p * ldb + j], c[i * ldc + j]);
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

/// Type alias for the default tile size optimized for Apple Silicon L1d cache.
pub type TiledGemm64 = TiledGemm<64>;

/// Type alias for a larger tile size suitable for server CPUs with bigger caches.
pub type TiledGemm128 = TiledGemm<128>;

/// Type alias for a smaller tile size suitable for experimentation.
pub type TiledGemm32 = TiledGemm<32>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiled_matches_naive() {
        use crate::cpu::NaiveGemm;

        let m = 17; // Non-power-of-2 to test remainder handling
        let k = 13;
        let n = 11;

        // Fill with deterministic values
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let mut c_naive = vec![0.0f32; m * n];
        let mut c_tiled = vec![0.0f32; m * n];

        NaiveGemm.gemm(m, k, n, &a, k, &b, n, &mut c_naive, n).unwrap();
        TiledGemm::<4>.gemm(m, k, n, &a, k, &b, n, &mut c_tiled, n).unwrap();

        for idx in 0..m * n {
            assert!(
                (c_naive[idx] - c_tiled[idx]).abs() < 1e-4,
                "Mismatch at index {idx}: naive={}, tiled={}",
                c_naive[idx],
                c_tiled[idx]
            );
        }
    }

    #[test]
    fn test_tiled_64_identity() {
        let engine = TiledGemm::<64>;
        let n = 5;
        let a: Vec<f32> = (1..=25).map(|x| x as f32).collect();
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        let mut c = vec![0.0f32; n * n];
        engine.gemm(n, n, n, &a, n, &identity, n, &mut c, n).unwrap();
        assert_eq!(c, a);
    }

    #[test]
    fn test_tiled_with_strides() {
        let engine = TiledGemm::<4>;
        // A (2×2) with lda=5
        let a = [1.0f32, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 5]; // ldc=3

        engine.gemm(2, 2, 2, &a, 5, &b, 2, &mut c, 3).unwrap();

        assert_eq!(c[0], 19.0);
        assert_eq!(c[1], 22.0);
        assert_eq!(c[3], 43.0);
        assert_eq!(c[4], 50.0);
    }
}
