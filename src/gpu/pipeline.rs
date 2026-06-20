//! GPU compute pipeline for GEMM execution.
//!
//! Creates the wgpu compute pipeline from WGSL shader source, sets up
//! bind groups, and dispatches the compute workgroups. Implements `GemmEngine<f32>`
//! for seamless integration with the rest of the Quasar-BLAS API.

use wgpu;

use crate::types::{GemmError, validate_gemm_dims};
use crate::GemmEngine;

use super::buffers::GpuBuffers;
use super::context::GpuContext;

/// Which WGSL shader to use for the GPU compute pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderVariant {
    /// One thread per output element, no shared memory.
    Naive,
    /// Workgroup-cooperative tiling with shared memory.
    Tiled,
}

/// Naive WGSL shader source.
const SHADER_NAIVE: &str = include_str!("shaders/matmul_naive.wgsl");

/// Tiled WGSL shader source.
const SHADER_TILED: &str = include_str!("shaders/matmul_tiled.wgsl");

/// GPU GEMM engine — executes matrix multiplication on the GPU via wgpu.
pub struct GpuGemm {
    ctx: GpuContext,
    variant: ShaderVariant,
}

impl GpuGemm {
    /// Create a new GPU GEMM engine with the specified shader variant.
    pub fn new(variant: ShaderVariant) -> Self {
        let ctx = GpuContext::new();
        Self { ctx, variant }
    }

    /// Create from an existing GPU context (avoids re-initializing the device).
    pub fn with_context(ctx: GpuContext, variant: ShaderVariant) -> Self {
        Self { ctx, variant }
    }

    /// Get the workgroup size for the current shader variant.
    fn workgroup_size(&self) -> (u32, u32) {
        match self.variant {
            ShaderVariant::Naive => (8, 8),
            ShaderVariant::Tiled => (16, 16),
        }
    }

    /// Get the shader source for the current variant.
    fn shader_source(&self) -> &str {
        match self.variant {
            ShaderVariant::Naive => SHADER_NAIVE,
            ShaderVariant::Tiled => SHADER_TILED,
        }
    }

    /// Get adapter info string.
    pub fn adapter_info(&self) -> String {
        format!("{} ({})", self.ctx.adapter_name, self.ctx.backend)
    }
}

impl GemmEngine<f32> for GpuGemm {
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

        // GPU operates on contiguous row-major data.
        // If the caller provides non-tight strides, we pack into contiguous buffers.
        let (a_packed, b_packed) = if lda == k && ldb == n {
            // Already contiguous — no copy needed
            (a.to_vec(), b.to_vec())
        } else {
            // Pack with tight strides
            let mut a_packed = vec![0.0f32; m * k];
            for i in 0..m {
                for j in 0..k {
                    a_packed[i * k + j] = a[i * lda + j];
                }
            }
            let mut b_packed = vec![0.0f32; k * n];
            for i in 0..k {
                for j in 0..n {
                    b_packed[i * n + j] = b[i * ldb + j];
                }
            }
            (a_packed, b_packed)
        };

        // Create GPU buffers
        let buffers = GpuBuffers::new(&self.ctx, m, k, n);

        // Upload data to GPU
        buffers.upload(&self.ctx, m, k, n, &a_packed, &b_packed);

        // Create the compute shader module
        let shader = self
            .ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gemm_shader"),
                source: wgpu::ShaderSource::Wgsl(self.shader_source().into()),
            });

        // Create bind group layout
        let bind_group_layout =
            self.ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("gemm_bind_group_layout"),
                    entries: &[
                        // @binding(0): dims uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // @binding(1): A storage
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // @binding(2): B storage
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // @binding(3): C storage (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Create pipeline layout
        let pipeline_layout =
            self.ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("gemm_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Create compute pipeline
        let pipeline =
            self.ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("gemm_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Create bind group
        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.dims_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.c_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute workgroups
        let (wg_x, wg_y) = self.workgroup_size();
        let num_workgroups_x = (n as u32 + wg_x - 1) / wg_x;
        let num_workgroups_y = (m as u32 + wg_y - 1) / wg_y;

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemm_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        // Read back results
        let mut c_result = vec![0.0f32; m * n];
        buffers.download(&self.ctx, &mut c_result);

        // Unpack into caller's buffer (handling non-tight ldc)
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] = c_result[i * n + j];
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_naive_2x2() {
        let engine = GpuGemm::new(ShaderVariant::Naive);
        println!("GPU Adapter: {}", engine.adapter_info());

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];

        engine.gemm(2, 2, 2, &a, 2, &b, 2, &mut c, 2).unwrap();

        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //   = [[19, 22], [43, 50]]
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gpu_naive_non_square() {
        let engine = GpuGemm::new(ShaderVariant::Naive);

        // A (2×3) × B (3×2) = C (2×2)
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = [0.0f32; 4];

        engine.gemm(2, 3, 2, &a, 3, &b, 2, &mut c, 2).unwrap();

        assert_eq!(c, [22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_gpu_tiled_2x2() {
        let engine = GpuGemm::new(ShaderVariant::Tiled);

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];

        engine.gemm(2, 2, 2, &a, 2, &b, 2, &mut c, 2).unwrap();

        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gpu_matches_cpu() {
        use crate::cpu::NaiveGemm;

        let m = 33;
        let k = 17;
        let n = 25;

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 - 2.0).collect();

        let mut c_cpu = vec![0.0f32; m * n];
        let mut c_gpu = vec![0.0f32; m * n];

        NaiveGemm.gemm(m, k, n, &a, k, &b, n, &mut c_cpu, n).unwrap();
        GpuGemm::new(ShaderVariant::Naive)
            .gemm(m, k, n, &a, k, &b, n, &mut c_gpu, n)
            .unwrap();

        for idx in 0..m * n {
            assert!(
                (c_cpu[idx] - c_gpu[idx]).abs() < 1e-2,
                "Mismatch at {}: cpu={}, gpu={}",
                idx, c_cpu[idx], c_gpu[idx]
            );
        }
    }
}
