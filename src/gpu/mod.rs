//! GPU-accelerated GEMM engines using wgpu compute shaders.
//!
//! Supports Metal (macOS) and Vulkan (Linux) backends transparently.
//! Two shader variants:
//! - Naive: one thread per output element
//! - Tiled: shared-memory cooperative tiling for reduced global memory traffic

pub mod context;
pub mod buffers;
pub mod pipeline;

pub use context::GpuContext;
pub use pipeline::{GpuGemm, ShaderVariant};
