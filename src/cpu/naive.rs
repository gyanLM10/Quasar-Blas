//! Naive O(N³) GEMM implementation — the performance floor.
//!
//! This is the textbook triple-nested loop with stride-aware indexing.
//! It serves two purposes:
//! 1. **Correctness oracle** — property tests validate optimized engines against this
//! 2. **Performance baseline** — benchmarks measure speedup relative to this
//!
//! No cache optimization, no SIMD, no parallelism.

use crate::types::{GemmElement, GemmError, validate_gemm_dims};
use crate::GemmEngine;

/// Naive GEMM engine — standard O(N³) triple-nested loop.
///
/// Loop order is `i → j → k` which is intentionally sub-optimal for cache
/// utilization (it accesses B with stride `ldb` in the innermost loop,
/// causing cache line thrashing). This is by design — it's the baseline.
pub struct NaiveGemm;

impl<T: GemmElement> GemmEngine<T> for NaiveGemm {
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

        // C[i][j] += A[i][p] * B[p][j]  for all p in 0..k
        // Loop order: i → j → k (naive, cache-unfriendly for B)
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for p in 0..k {
                    // sum = A[i,p] * B[p,j] + sum
                    sum = a[i * lda + p].mul_add(b[p * ldb + j], sum);
                }
                c[i * ldc + j] = sum;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_1x1() {
        let engine = NaiveGemm;
        let a = [3.0f32];
        let b = [7.0f32];
        let mut c = [0.0f32];
        engine.gemm(1, 1, 1, &a, 1, &b, 1, &mut c, 1).unwrap();
        assert_eq!(c[0], 21.0);
    }

    #[test]
    fn test_naive_with_stride() {
        let engine = NaiveGemm;
        // A (2×2) embedded in a buffer with lda=4:
        // Row 0: [1, 2, _, _]
        // Row 1: [3, 4, _, _]
        let a = [1.0f32, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0];
        // B (2×2) with ldb=2 (tight)
        let b = [5.0f32, 6.0, 7.0, 8.0];
        // C with ldc=3 (extra padding)
        let mut c = [0.0f32; 5]; // (2-1)*3 + 2 = 5

        engine.gemm(2, 2, 2, &a, 4, &b, 2, &mut c, 3).unwrap();

        // C[0,0] = 1*5 + 2*7 = 19
        // C[0,1] = 1*6 + 2*8 = 22
        // C[1,0] = 3*5 + 4*7 = 43
        // C[1,1] = 3*6 + 4*8 = 50
        assert_eq!(c[0], 19.0); // [0*3 + 0]
        assert_eq!(c[1], 22.0); // [0*3 + 1]
        assert_eq!(c[3], 43.0); // [1*3 + 0]
        assert_eq!(c[4], 50.0); // [1*3 + 1]
    }
}
