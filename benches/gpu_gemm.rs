//! Criterion benchmarks for GPU GEMM engines.
//!
//! Compares naive vs. tiled WGSL compute shaders targeting the local wgpu adapter.
//!
//! Reports throughput in GFLOPS: `2 * M * K * N / time_ns`
//! (the factor of 2 accounts for one multiply + one add per MAC operation).

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};

use quasar_blas::gpu::{GpuGemm, ShaderVariant};
use quasar_blas::GemmEngine;

/// Matrix sizes to benchmark on GPU.
const SIZES: &[usize] = &[64, 128, 256, 512, 1024];

/// Generate deterministic test data for a given matrix size.
fn generate_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((seed.wrapping_mul(i as u64 + 1).wrapping_add(7)) % 2000) as f32 / 100.0 - 10.0)
        .collect()
}

fn bench_gpu_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_naive");
    
    // Create the GPU engine once per group to avoid device initialization overhead
    let engine = GpuGemm::new(ShaderVariant::Naive);

    for &size in SIZES {
        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));

        let a = generate_matrix(size, size, 42);
        let b = generate_matrix(size, size, 137);
        let mut result = vec![0.0f32; size * size];

        group.bench_with_input(
            BenchmarkId::new("GpuGemm(Naive)", size),
            &size,
            |bench, &n| {
                bench.iter(|| {
                    engine.gemm(
                        black_box(n),
                        black_box(n),
                        black_box(n),
                        black_box(&a),
                        n,
                        black_box(&b),
                        n,
                        black_box(&mut result),
                        n,
                    ).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_gpu_tiled(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_tiled");
    
    // Create the GPU engine once per group
    let engine = GpuGemm::new(ShaderVariant::Tiled);

    for &size in SIZES {
        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));

        let a = generate_matrix(size, size, 42);
        let b = generate_matrix(size, size, 137);
        let mut result = vec![0.0f32; size * size];

        group.bench_with_input(
            BenchmarkId::new("GpuGemm(Tiled)", size),
            &size,
            |bench, &n| {
                bench.iter(|| {
                    engine.gemm(
                        black_box(n),
                        black_box(n),
                        black_box(n),
                        black_box(&a),
                        n,
                        black_box(&b),
                        n,
                        black_box(&mut result),
                        n,
                    ).unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gpu_naive, bench_gpu_tiled);
criterion_main!(benches);
