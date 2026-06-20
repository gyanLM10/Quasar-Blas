//! CPU-based GEMM engines.
//!
//! Three implementations at increasing optimization levels:
//! - [`NaiveGemm`] — O(N³) baseline, no optimizations
//! - [`TiledGemm`] — const-generic cache-blocked with compile-time tile size
//! - [`SimdGemm`] — ARM Neon SIMD-accelerated tiled kernel (aarch64 only)

pub mod naive;
pub mod tiled;
pub mod simd;

pub use naive::NaiveGemm;
pub use tiled::TiledGemm;
pub use simd::SimdGemm;
