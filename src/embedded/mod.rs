//! Embedded bare-metal GEMM engine — `no_std`, no heap, no OS.
//!
//! This module provides GEMM operations for constrained edge devices
//! using INT8 quantized fixed-point arithmetic and static arena allocation.
//! Compiles for targets like `thumbv7em-none-eabihf` (Cortex-M4F).

pub mod fixed_point;
pub mod arena;

use crate::types::GemmError;
use fixed_point::{QuantParams, quantize_f32_to_i8};
use arena::StaticArena;

/// Embedded GEMM engine using INT8 quantized fixed-point arithmetic.
///
/// This engine:
/// 1. Quantizes f32 inputs to i8 using calibrated scale/zero-point parameters
/// 2. Performs MAC operations in i32 to prevent overflow
/// 3. Dequantizes the i32 accumulator back to f32 output
///
/// All intermediate buffers are allocated from a static arena — no heap.
pub struct EmbeddedGemm {
    pub a_params: QuantParams,
    pub b_params: QuantParams,
}

impl EmbeddedGemm {
    /// Create a new embedded engine with quantization parameters.
    pub const fn new(a_params: QuantParams, b_params: QuantParams) -> Self {
        Self { a_params, b_params }
    }

    /// Create with default quantization parameters (scale=0.1, zero_point=0).
    ///
    /// Suitable for matrices with values roughly in [-12.7, 12.7].
    pub const fn default_params() -> Self {
        Self {
            a_params: QuantParams { scale: 0.1, zero_point: 0 },
            b_params: QuantParams { scale: 0.1, zero_point: 0 },
        }
    }

    /// Perform quantized INT8 GEMM.
    ///
    /// # Process
    /// 1. Quantize A (f32 → i8) and B (f32 → i8)
    /// 2. Compute C_int32 = A_int8 × B_int8 (accumulate in i32)
    /// 3. Dequantize C_int32 → f32
    ///
    /// The arena provides scratch space for the quantized matrices.
    pub fn gemm_quantized<const ARENA_SIZE: usize>(
        &self,
        arena: &StaticArena<ARENA_SIZE>,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        c: &mut [f32],
        ldc: usize,
    ) -> Result<(), GemmError> {
        crate::types::validate_gemm_dims(m, k, n, lda, ldb, ldc, a.len(), b.len(), c.len())?;

        // Reset arena for this computation
        arena.reset();

        // Allocate quantized buffers from the arena
        let a_q = arena
            .alloc::<i8>(m * k)
            .ok_or(GemmError::InsufficientA { required: m * k, actual: 0 })?;
        let b_q = arena
            .alloc::<i8>(k * n)
            .ok_or(GemmError::InsufficientB { required: k * n, actual: 0 })?;

        // Quantize A
        for i in 0..m {
            for j in 0..k {
                a_q[i * k + j] = quantize_f32_to_i8(a[i * lda + j], &self.a_params);
            }
        }

        // Quantize B
        for i in 0..k {
            for j in 0..n {
                b_q[i * n + j] = quantize_f32_to_i8(b[i * ldb + j], &self.b_params);
            }
        }

        // INT8 GEMM with i32 accumulation
        // The combined output scale is a_scale * b_scale
        let output_scale = self.a_params.scale * self.b_params.scale;

        for i in 0..m {
            for j in 0..n {
                let mut acc: i32 = 0;
                for p in 0..k {
                    // Multiply two i8 values, accumulate in i32
                    // This prevents overflow: max |i8 * i8| = 127 * 127 = 16129
                    // max accumulation over k: 16129 * k (fits in i32 for k < 132,000)
                    let a_val = (a_q[i * k + p] as i32) - (self.a_params.zero_point as i32);
                    let b_val = (b_q[p * n + j] as i32) - (self.b_params.zero_point as i32);
                    acc += a_val * b_val;
                }
                // Dequantize: convert i32 accumulator back to f32
                c[i * ldc + j] = (acc as f32) * output_scale;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_gemm_identity() {
        let engine = EmbeddedGemm::new(
            QuantParams { scale: 0.05, zero_point: 0 },
            QuantParams { scale: 0.05, zero_point: 0 },
        );

        let arena = StaticArena::<4096>::new();

        // A = [[1, 0], [0, 1]] (identity)
        let a = [1.0f32, 0.0, 0.0, 1.0];
        // B = [[3, 4], [5, 6]]
        let b = [3.0f32, 4.0, 5.0, 6.0];
        let mut c = [0.0f32; 4];

        engine.gemm_quantized(&arena, 2, 2, 2, &a, 2, &b, 2, &mut c, 2).unwrap();

        // With quantization error, results should be approximately equal
        for i in 0..4 {
            assert!(
                (c[i] - b[i]).abs() < 0.5,
                "Mismatch at {}: expected ~{}, got {}",
                i, b[i], c[i]
            );
        }
    }

    #[test]
    fn test_embedded_gemm_small() {
        let engine = EmbeddedGemm::default_params();
        let arena = StaticArena::<4096>::new();

        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];

        engine.gemm_quantized(&arena, 2, 2, 2, &a, 2, &b, 2, &mut c, 2).unwrap();

        // Expected: [[19, 22], [43, 50]]
        // With INT8 quantization (scale=0.1), expect ~5-10% error
        assert!((c[0] - 19.0).abs() < 2.0, "c[0]={}", c[0]);
        assert!((c[1] - 22.0).abs() < 2.0, "c[1]={}", c[1]);
        assert!((c[2] - 43.0).abs() < 4.0, "c[2]={}", c[2]);
        assert!((c[3] - 50.0).abs() < 5.0, "c[3]={}", c[3]);
    }
}
