//! Fixed-point INT8 quantization and dequantization.
//!
//! Implements the standard quantization formula used in AI inference:
//!
//! ```text
//! q = clamp(round(r / scale) + zero_point, -128, 127)
//! r = (q - zero_point) * scale
//! ```
//!
//! This module is `no_std` compatible — uses only `core` primitives.

/// Round f32 to nearest integer (no_std compatible).
/// In `core`, f32 doesn't have `.round()`, so we implement it manually.
#[inline(always)]
fn round_f32(x: f32) -> f32 {
    // Truncate toward zero and adjust
    let t = x as i32 as f32;
    if x >= 0.0 {
        if x - t >= 0.5 { t + 1.0 } else { t }
    } else {
        if t - x >= 0.5 { t - 1.0 } else { t }
    }
}

/// Clamp f32 to a range (no_std compatible).
#[inline(always)]
fn clamp_f32(x: f32, min: f32, max: f32) -> f32 {
    if x < min { min } else if x > max { max } else { x }
}

/// Quantization parameters for mapping f32 ↔ i8.
#[derive(Debug, Clone, Copy)]
pub struct QuantParams {
    /// Scale factor: the f32 range per integer step.
    /// Smaller scale = higher precision but narrower representable range.
    pub scale: f32,

    /// Zero point: the i8 value that represents f32 zero.
    /// For symmetric quantization, this is 0.
    pub zero_point: i8,
}

impl QuantParams {
    /// Calculate quantization parameters for a given data range.
    ///
    /// This calibrates scale and zero_point so that the full [-128, 127]
    /// i8 range maps to [min_val, max_val].
    pub fn calibrate(min_val: f32, max_val: f32) -> Self {
        let scale = (max_val - min_val) / 255.0;
        let zero_point = if scale == 0.0 {
            0
        } else {
            let zp = -128.0 - min_val / scale;
            clamp_f32(round_f32(zp), -128.0, 127.0) as i8
        };
        Self { scale, zero_point }
    }
}

/// Quantize a single f32 value to i8.
///
/// ```text
/// q = clamp(round(value / scale) + zero_point, -128, 127)
/// ```
#[inline(always)]
pub fn quantize_f32_to_i8(value: f32, params: &QuantParams) -> i8 {
    if params.scale == 0.0 {
        return params.zero_point;
    }
    let q = round_f32(value / params.scale) + params.zero_point as f32;
    clamp_f32(q, -128.0, 127.0) as i8
}

/// Dequantize a single i8 value back to f32.
///
/// ```text
/// r = (q - zero_point) * scale
/// ```
#[inline(always)]
pub fn dequantize_i8_to_f32(value: i8, params: &QuantParams) -> f32 {
    ((value as f32) - (params.zero_point as f32)) * params.scale
}

/// Quantize an entire f32 slice to i8 in-place into a pre-allocated output buffer.
///
/// This is the batch version — avoids per-element function call overhead.
pub fn quantize_slice(input: &[f32], output: &mut [i8], params: &QuantParams) {
    assert_eq!(input.len(), output.len());
    let inv_scale = if params.scale != 0.0 { 1.0 / params.scale } else { 0.0 };
    let zp = params.zero_point as f32;

    for (i, &val) in input.iter().enumerate() {
        let q = round_f32(val * inv_scale) + zp;
        output[i] = clamp_f32(q, -128.0, 127.0) as i8;
    }
}

/// Dequantize an entire i8 slice back to f32.
pub fn dequantize_slice(input: &[i8], output: &mut [f32], params: &QuantParams) {
    assert_eq!(input.len(), output.len());
    let zp = params.zero_point as f32;

    for (i, &val) in input.iter().enumerate() {
        output[i] = ((val as f32) - zp) * params.scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_symmetric() {
        let params = QuantParams { scale: 0.1, zero_point: 0 };

        // Values within representable range
        for &val in &[0.0f32, 1.0, -1.0, 5.5, -12.7, 12.7] {
            let q = quantize_f32_to_i8(val, &params);
            let r = dequantize_i8_to_f32(q, &params);
            assert!(
                (val - r).abs() < params.scale,
                "Roundtrip failed for {}: q={}, r={}",
                val, q, r
            );
        }
    }

    #[test]
    fn test_saturation() {
        let params = QuantParams { scale: 0.1, zero_point: 0 };

        // Value too large → saturates to 127
        let q = quantize_f32_to_i8(100.0, &params);
        assert_eq!(q, 127);

        // Value too negative → saturates to -128
        let q = quantize_f32_to_i8(-100.0, &params);
        assert_eq!(q, -128);
    }

    #[test]
    fn test_calibrate() {
        let params = QuantParams::calibrate(-10.0, 10.0);
        assert!(params.scale > 0.0);

        // Check that calibrated params can represent the range
        let q_min = quantize_f32_to_i8(-10.0, &params);
        let q_max = quantize_f32_to_i8(10.0, &params);
        assert!(q_min < q_max);
    }

    #[test]
    fn test_batch_quantize() {
        let params = QuantParams { scale: 0.1, zero_point: 0 };
        let input = [1.0f32, 2.0, 3.0, -1.0, -2.0];
        let mut output = [0i8; 5];
        quantize_slice(&input, &mut output, &params);

        assert_eq!(output, [10, 20, 30, -10, -20]);
    }
}
