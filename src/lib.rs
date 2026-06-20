//! # Quasar-BLAS
//!
//! A heterogeneous, high-performance linear algebra engine providing GEMM
//! (General Matrix Multiply) across three hardware tiers:
//!
//! - **Tier 1 (CPU)**: Cache-tiled, ARM Neon SIMD-accelerated kernels
//! - **Tier 2 (GPU)**: wgpu compute shaders targeting Metal (macOS) and Vulkan (Linux)
//! - **Tier 3 (Embedded)**: `no_std` bare-metal with INT8 quantized fixed-point arithmetic
//!
//! ## Architecture
//!
//! All engines implement the [`GemmEngine<T>`] trait, which accepts matrix dimensions,
//! slice references, and leading dimension (stride) parameters. This allows callers to
//! pass row-major or column-major data transparently.
//!
//! The trait is generic over [`GemmElement`], a sealed trait implemented for `f32`.
//! The embedded tier extends this with a separate `i8` path.

#![feature(portable_simd)]

// When the `embedded` feature is active and `std` is not, compile as no_std.
#![cfg_attr(all(feature = "embedded", not(feature = "std")), no_std)]

// Re-export core types at crate root
pub mod types;

// CPU tier — always available
pub mod cpu;

// GPU tier — requires wgpu + pollster + bytemuck
#[cfg(feature = "gpu")]
pub mod gpu;

// Embedded tier — no_std, no alloc, static allocation only
#[cfg(feature = "embedded")]
pub mod embedded;

use types::GemmElement;

/// Universal GEMM engine trait.
///
/// All hardware backends implement this trait, providing a unified API for
/// General Matrix Multiply: `C = A × B` where:
/// - `A` is an `M × K` matrix
/// - `B` is a `K × N` matrix  
/// - `C` is an `M × N` matrix (output)
///
/// ## Leading Dimensions (Strides)
///
/// The `lda`, `ldb`, `ldc` parameters specify the *leading dimension* of each matrix,
/// i.e., the number of elements between consecutive rows in memory.
///
/// For standard row-major layout:
/// - `lda = K` (A is M×K, rows are K elements apart)
/// - `ldb = N` (B is K×N, rows are N elements apart)
/// - `ldc = N` (C is M×N, rows are N elements apart)
///
/// For column-major data, the caller simply transposes the leading dimensions.
/// This is how the original Fortran BLAS API works, giving maximum flexibility
/// with zero abstraction cost.
///
/// ## Indexing
///
/// Element `(i, j)` of matrix `A` with leading dimension `lda` is accessed as:
/// ```text
/// A[i * lda + j]
/// ```
pub trait GemmEngine<T: GemmElement> {
    /// Error type for this engine (e.g., dimension mismatch, GPU errors).
    type Error: core::fmt::Debug;

    /// Perform General Matrix Multiply: `C = A × B`
    ///
    /// # Arguments
    /// - `m`, `k`, `n` — matrix dimensions (A is M×K, B is K×N, C is M×N)
    /// - `a` — input matrix A data (must contain at least `(m-1)*lda + k` elements)
    /// - `lda` — leading dimension of A (stride between rows)
    /// - `b` — input matrix B data (must contain at least `(k-1)*ldb + n` elements)
    /// - `ldb` — leading dimension of B (stride between rows)
    /// - `c` — output matrix C data (must contain at least `(m-1)*ldc + n` elements)
    /// - `ldc` — leading dimension of C (stride between rows)
    ///
    /// # Errors
    /// Returns an error if slice lengths are insufficient for the given dimensions
    /// and leading dimensions.
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
    ) -> Result<(), Self::Error>;
}

/// Convenience method for square matrices with default (tight) leading dimensions.
///
/// This avoids forcing callers to specify `lda=K, ldb=N, ldc=N` for the common
/// case of contiguous row-major matrices.
pub fn gemm_simple<T: GemmElement, E: GemmEngine<T>>(
    engine: &E,
    m: usize,
    k: usize,
    n: usize,
    a: &[T],
    b: &[T],
    c: &mut [T],
) -> Result<(), E::Error> {
    engine.gemm(m, k, n, a, k, b, n, c, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::NaiveGemm;

    #[test]
    fn test_identity_2x2() {
        let engine = NaiveGemm;
        // A = [[1, 2], [3, 4]]
        let a = [1.0f32, 2.0, 3.0, 4.0];
        // I = [[1, 0], [0, 1]]
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];

        engine.gemm(2, 2, 2, &a, 2, &b, 2, &mut c, 2).unwrap();

        assert_eq!(c, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_simple_3x3() {
        let engine = NaiveGemm;
        // A = [[1,2,3],[4,5,6],[7,8,9]]
        let a: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        // B = identity
        let b = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0f32; 9];

        engine.gemm(3, 3, 3, &a, 3, &b, 3, &mut c, 3).unwrap();

        assert_eq!(c, a);
    }

    #[test]
    fn test_non_square() {
        let engine = NaiveGemm;
        // A (2x3) = [[1,2,3],[4,5,6]]
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // B (3x2) = [[1,2],[3,4],[5,6]]
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = [0.0f32; 4];

        engine.gemm(2, 3, 2, &a, 3, &b, 2, &mut c, 2).unwrap();

        // C = [[22, 28], [49, 64]]
        assert_eq!(c, [22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let engine = NaiveGemm;
        let a = [1.0f32; 4]; // too small for 3x3
        let b = [1.0f32; 9];
        let mut c = [0.0f32; 9];

        let result = engine.gemm(3, 3, 3, &a, 3, &b, 3, &mut c, 3);
        assert!(result.is_err());
    }
}
