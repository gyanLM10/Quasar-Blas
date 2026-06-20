//! Core types, traits, and memory alignment primitives for Quasar-BLAS.
//!
//! This module provides:
//! - [`GemmElement`] — the sealed trait bound for matrix element types
//! - [`AlignedVec`] — a cache-line-aligned vector wrapper for optimal SIMD/cache performance
//! - [`GemmError`] — common error type for GEMM operations
//! - Dimension validation utilities

use core::fmt;

// ---------------------------------------------------------------------------
// GemmElement — sealed generic trait for matrix element types
// ---------------------------------------------------------------------------

/// Trait bound for types that can participate in GEMM operations.
///
/// This is a sealed trait — only types within this crate can implement it.
/// Currently implemented for `f32`. The embedded tier extends GEMM to `i8`
/// via a separate trait to keep the type constraints clean.
///
/// # Zero-Cost Abstraction
///
/// All methods are `#[inline(always)]` so the compiler monomorphizes them
/// away entirely. There is zero runtime dispatch cost.
pub trait GemmElement: Copy + Default + fmt::Debug + PartialEq + Send + Sync + 'static + private::Sealed {
    /// The additive identity (0).
    fn zero() -> Self;

    /// The multiplicative identity (1).
    fn one() -> Self;

    /// Fused multiply-add: `self * a + b`
    ///
    /// On hardware with FMA instructions (ARM Neon, x86 FMA3), this compiles
    /// to a single instruction with no intermediate rounding.
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Convert from f64 (used by test harnesses and initialization).
    fn from_f64(val: f64) -> Self;
}

// Seal the trait so external crates cannot implement it.
mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
}

impl GemmElement for f32 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn one() -> Self {
        1.0
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // In std builds, f32::mul_add uses the hardware FMA instruction.
        // In no_std builds, we fall back to manual multiply-add.
        #[cfg(feature = "std")]
        {
            f32::mul_add(self, a, b)
        }
        #[cfg(not(feature = "std"))]
        {
            (self * a) + b
        }
    }

    #[inline(always)]
    fn from_f64(val: f64) -> Self {
        val as f32
    }
}

// ---------------------------------------------------------------------------
// GemmError — common error type
// ---------------------------------------------------------------------------

/// Errors that can occur during GEMM operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemmError {
    /// Input slice `a` is too small for the given dimensions and leading dimension.
    /// Contains (required_len, actual_len).
    InsufficientA { required: usize, actual: usize },

    /// Input slice `b` is too small for the given dimensions and leading dimension.
    /// Contains (required_len, actual_len).
    InsufficientB { required: usize, actual: usize },

    /// Output slice `c` is too small for the given dimensions and leading dimension.
    /// Contains (required_len, actual_len).
    InsufficientC { required: usize, actual: usize },

    /// Leading dimension is smaller than the corresponding matrix dimension.
    /// e.g., `lda < k` for matrix A(M×K).
    InvalidLeadingDimension {
        matrix: &'static str,
        ld: usize,
        min_required: usize,
    },

    /// A matrix dimension is zero.
    ZeroDimension { which: &'static str },
}

impl fmt::Display for GemmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GemmError::InsufficientA { required, actual } => {
                write!(f, "Matrix A: need {required} elements, got {actual}")
            }
            GemmError::InsufficientB { required, actual } => {
                write!(f, "Matrix B: need {required} elements, got {actual}")
            }
            GemmError::InsufficientC { required, actual } => {
                write!(f, "Matrix C: need {required} elements, got {actual}")
            }
            GemmError::InvalidLeadingDimension { matrix, ld, min_required } => {
                write!(
                    f,
                    "Matrix {matrix}: leading dimension {ld} < minimum {min_required}"
                )
            }
            GemmError::ZeroDimension { which } => {
                write!(f, "Dimension '{which}' is zero")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dimension validation
// ---------------------------------------------------------------------------

/// Validate GEMM dimensions and leading dimensions.
///
/// Returns the minimum required lengths for slices A, B, C.
///
/// # Rules
/// - `lda >= k` (A is M×K, each row has K elements, stride must be >= K)
/// - `ldb >= n` (B is K×N, each row has N elements, stride must be >= N)
/// - `ldc >= n` (C is M×N, each row has N elements, stride must be >= N)
/// - All dimensions must be > 0
pub fn validate_gemm_dims(
    m: usize,
    k: usize,
    n: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    a_len: usize,
    b_len: usize,
    c_len: usize,
) -> Result<(), GemmError> {
    // Check for zero dimensions
    if m == 0 {
        return Err(GemmError::ZeroDimension { which: "m" });
    }
    if k == 0 {
        return Err(GemmError::ZeroDimension { which: "k" });
    }
    if n == 0 {
        return Err(GemmError::ZeroDimension { which: "n" });
    }

    // Validate leading dimensions
    if lda < k {
        return Err(GemmError::InvalidLeadingDimension {
            matrix: "A",
            ld: lda,
            min_required: k,
        });
    }
    if ldb < n {
        return Err(GemmError::InvalidLeadingDimension {
            matrix: "B",
            ld: ldb,
            min_required: n,
        });
    }
    if ldc < n {
        return Err(GemmError::InvalidLeadingDimension {
            matrix: "C",
            ld: ldc,
            min_required: n,
        });
    }

    // Calculate required lengths:
    // A: (m - 1) * lda + k  (last row only needs k elements, not full lda)
    // B: (k - 1) * ldb + n
    // C: (m - 1) * ldc + n
    let required_a = (m - 1) * lda + k;
    let required_b = (k - 1) * ldb + n;
    let required_c = (m - 1) * ldc + n;

    if a_len < required_a {
        return Err(GemmError::InsufficientA {
            required: required_a,
            actual: a_len,
        });
    }
    if b_len < required_b {
        return Err(GemmError::InsufficientB {
            required: required_b,
            actual: b_len,
        });
    }
    if c_len < required_c {
        return Err(GemmError::InsufficientC {
            required: required_c,
            actual: c_len,
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// AlignedVec — cache-line-aligned allocation
// ---------------------------------------------------------------------------

/// A block of 64 bytes (typical L1 cache line size on ARM/x86).
/// We use this to force the system allocator to return 64-byte aligned pointers.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct CacheBlock {
    pub _data: [u8; 64],
}

impl Default for CacheBlock {
    fn default() -> Self {
        Self { _data: [0; 64] }
    }
}

/// A heap-allocated vector whose data pointer is guaranteed aligned to 64 bytes
/// using a `#[repr(align(64))]` custom allocation wrapper.
///
/// This alignment guarantees that:
/// 1. SIMD loads/stores never cross cache line boundaries (no split-line penalty)
/// 2. Prefetch instructions bring in exactly the data we need
/// 3. The CPU cache tiling algorithm operates on clean cache line boundaries
///
/// Only available with the `std` feature (heap allocation requires the allocator).
#[cfg(feature = "std")]
pub struct AlignedVec<T: GemmElement> {
    blocks: Vec<CacheBlock>,
    rows: usize,
    cols: usize,
    /// Leading dimension (stride between rows). Always >= cols.
    ld: usize,
    _marker: core::marker::PhantomData<T>,
}

#[cfg(feature = "std")]
impl<T: GemmElement> AlignedVec<T> {
    /// Create a new zero-initialized aligned matrix.
    ///
    /// The leading dimension is rounded up to the next multiple of
    /// `64 / size_of::<T>()` to ensure each row starts on a cache line boundary.
    pub fn new(rows: usize, cols: usize) -> Self {
        let elements_per_block = 64 / core::mem::size_of::<T>();
        let ld = if cols % elements_per_block == 0 {
            cols
        } else {
            cols + (elements_per_block - cols % elements_per_block)
        };

        let total_elements = rows * ld;
        let num_blocks = (total_elements + elements_per_block - 1) / elements_per_block;
        
        let blocks = vec![CacheBlock::default(); num_blocks];

        let mut s = Self { blocks, rows, cols, ld, _marker: core::marker::PhantomData };
        // Zero initialize the T elements
        for elem in s.as_mut_slice() {
            *elem = T::zero();
        }
        s
    }

    /// Create from existing data with specified leading dimension.
    pub fn from_slice(data: &[T], rows: usize, cols: usize, ld: usize) -> Self {
        assert!(ld >= cols, "Leading dimension must be >= cols");
        let required = if rows == 0 { 0 } else { (rows - 1) * ld + cols };
        assert!(data.len() >= required, "Insufficient data for dimensions");

        let mut aligned = Self::new(rows, ld); // Use ld as cols to get exact ld
        
        let target_slice = aligned.as_mut_slice();
        for i in 0..rows {
            for j in 0..cols {
                target_slice[i * ld + j] = data[i * ld + j];
            }
        }

        // Correct the cols value (new() set it to ld)
        aligned.cols = cols;
        aligned
    }

    /// Get the underlying data slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let ptr = self.blocks.as_ptr() as *const T;
        unsafe { core::slice::from_raw_parts(ptr, self.rows * self.ld) }
    }

    /// Get the underlying data slice mutably.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let ptr = self.blocks.as_mut_ptr() as *mut T;
        unsafe { core::slice::from_raw_parts_mut(ptr, self.rows * self.ld) }
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Leading dimension (stride between rows).
    #[inline]
    pub fn ld(&self) -> usize {
        self.ld
    }
}

#[cfg(test)]
mod types_tests {
    use super::*;

    #[test]
    fn test_f32_gemm_element() {
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::one(), 1.0);
        assert_eq!(2.0f32.mul_add(3.0, 4.0), 10.0);
        assert_eq!(f32::from_f64(3.14), 3.14f32);
    }

    #[test]
    fn test_validate_dims_valid() {
        let result = validate_gemm_dims(2, 3, 4, 3, 4, 4, 6, 12, 8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_dims_zero_m() {
        let result = validate_gemm_dims(0, 3, 4, 3, 4, 4, 0, 12, 0);
        assert!(matches!(result, Err(GemmError::ZeroDimension { which: "m" })));
    }

    #[test]
    fn test_validate_dims_insufficient_a() {
        let result = validate_gemm_dims(2, 3, 4, 3, 4, 4, 5, 12, 8);
        assert!(matches!(result, Err(GemmError::InsufficientA { .. })));
    }

    #[test]
    fn test_validate_dims_bad_ld() {
        let result = validate_gemm_dims(2, 3, 4, 2, 4, 4, 6, 12, 8);
        assert!(matches!(
            result,
            Err(GemmError::InvalidLeadingDimension { matrix: "A", .. })
        ));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_aligned_vec_alignment() {
        let mat = AlignedVec::<f32>::new(4, 4);
        // ld should be a multiple of 16 (64 bytes / 4 bytes per f32)
        assert_eq!(mat.ld() % 16, 0);
        assert_eq!(mat.rows(), 4);
        assert_eq!(mat.cols(), 4);
    }
}
