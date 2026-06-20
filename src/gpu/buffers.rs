//! GPU buffer management for GEMM operations.
//!
//! Manages three storage buffers (A, B, C) for the compute shader and a
//! staging buffer for reading results back to the CPU.
//!
//! ## Buffer Strategy
//!
//! - **Storage buffers** (A, B, C): `STORAGE | COPY_DST` (A, B) and
//!   `STORAGE | COPY_SRC | COPY_DST` (C) — live in GPU VRAM
//! - **Uniform buffer** (dims): Holds M, K, N dimensions for the shader
//! - **Staging buffer**: `MAP_READ | COPY_DST` — host-visible for CPU readback

use bytemuck::{Pod, Zeroable};
use wgpu;

use super::context::GpuContext;

/// Matrix dimensions passed to the compute shader as a uniform buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuDimensions {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub _padding: u32, // Align to 16 bytes (wgpu uniform alignment requirement)
}

/// Manages GPU buffers for a single GEMM operation.
pub struct GpuBuffers {
    pub dims_buffer: wgpu::Buffer,
    pub a_buffer: wgpu::Buffer,
    pub b_buffer: wgpu::Buffer,
    pub c_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,
    pub m: usize,
    pub n: usize,
}

impl GpuBuffers {
    /// Create GPU buffers sized for the given matrix dimensions.
    pub fn new(ctx: &GpuContext, m: usize, k: usize, n: usize) -> Self {
        let a_size = (m * k * std::mem::size_of::<f32>()) as u64;
        let b_size = (k * n * std::mem::size_of::<f32>()) as u64;
        let c_size = (m * n * std::mem::size_of::<f32>()) as u64;

        let dims_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dims_uniform"),
            size: std::mem::size_of::<GpuDimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let a_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matrix_a"),
            size: a_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matrix_b"),
            size: b_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matrix_c"),
            size: c_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size: c_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            dims_buffer,
            a_buffer,
            b_buffer,
            c_buffer,
            staging_buffer,
            m,
            n,
        }
    }

    /// Upload matrix data and dimensions to the GPU.
    pub fn upload(&self, ctx: &GpuContext, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) {
        let dims = GpuDimensions {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _padding: 0,
        };

        ctx.queue.write_buffer(&self.dims_buffer, 0, bytemuck::bytes_of(&dims));
        ctx.queue.write_buffer(&self.a_buffer, 0, bytemuck::cast_slice(a));
        ctx.queue.write_buffer(&self.b_buffer, 0, bytemuck::cast_slice(b));
    }

    /// Download the result matrix C from GPU to CPU.
    ///
    /// This is a synchronous operation that maps the staging buffer,
    /// copies the data, and unmaps it.
    pub fn download(&self, ctx: &GpuContext, c: &mut [f32]) {
        // Copy C buffer to staging buffer
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });

        let c_size = (self.m * self.n * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.c_buffer, 0, &self.staging_buffer, 0, c_size);
        ctx.queue.submit(Some(encoder.finish()));

        // Map the staging buffer and read back
        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        {
            let data = buffer_slice.get_mapped_range();
            let result: &[f32] = bytemuck::cast_slice(&data);
            c[..self.m * self.n].copy_from_slice(&result[..self.m * self.n]);
        }
        self.staging_buffer.unmap();
    }
}
