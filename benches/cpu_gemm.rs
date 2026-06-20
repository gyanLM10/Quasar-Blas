//! Criterion benchmarks for CPU GEMM engines.
//!
//! Compares naive vs. tiled vs. SIMD at multiple matrix sizes to measure
//! the real-world impact of cache tiling and SIMD vectorization.
//!
//! Reports throughput in GFLOPS: `2 * M * K * N / time_ns`
//! (the factor of 2 accounts for one multiply + one add per MAC operation).

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};

use quasar_blas::GemmEngine;
use quasar_blas::cpu::{NaiveGemm, TiledGemm, SimdGemm};

/// Matrix sizes to benchmark.
const SIZES: &[usize] = &[64, 128, 256, 512, 1024];

/// Generate deterministic test data for a given matrix size.
fn generate_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((seed.wrapping_mul(i as u64 + 1).wrapping_add(7)) % 2000) as f32 / 100.0 - 10.0)
        .collect()
}

fn bench_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_naive");

    for &size in SIZES {
        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));

        let a = generate_matrix(size, size, 42);
        let b = generate_matrix(size, size, 137);
        let mut result = vec![0.0f32; size * size];

        group.bench_with_input(
            BenchmarkId::new("NaiveGemm", size),
            &size,
            |bench, &n| {
                bench.iter(|| {
                    NaiveGemm
                        .gemm(
                            black_box(n),
                            black_box(n),
                            black_box(n),
                            black_box(&a),
                            n,
                            black_box(&b),
                            n,
                            black_box(&mut result),
                            n,
                        )
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_tiled(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_tiled");

    for &size in SIZES {
        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));

        let a = generate_matrix(size, size, 42);
        let b = generate_matrix(size, size, 137);
        let mut result = vec![0.0f32; size * size];

        group.bench_with_input(
            BenchmarkId::new("TiledGemm<64>", size),
            &size,
            |bench, &n| {
                bench.iter(|| {
                    TiledGemm::<64>
                        .gemm(
                            black_box(n),
                            black_box(n),
                            black_box(n),
                            black_box(&a),
                            n,
                            black_box(&b),
                            n,
                            black_box(&mut result),
                            n,
                        )
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_simd");

    for &size in SIZES {
        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));

        let a = generate_matrix(size, size, 42);
        let b = generate_matrix(size, size, 137);
        let mut result = vec![0.0f32; size * size];

        group.bench_with_input(
            BenchmarkId::new("SimdGemm<64>", size),
            &size,
            |bench, &n| {
                bench.iter(|| {
                    SimdGemm::<64>
                        .gemm(
                            black_box(n),
                            black_box(n),
                            black_box(n),
                            black_box(&a),
                            n,
                            black_box(&b),
                            n,
                            black_box(&mut result),
                            n,
                        )
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_comparison");

    // Compare all three at a medium size (256) for side-by-side analysis
    let size = 256;
    let flops = 2 * size * size * size;
    group.throughput(Throughput::Elements(flops as u64));

    let a = generate_matrix(size, size, 42);
    let b = generate_matrix(size, size, 137);
    let mut result = vec![0.0f32; size * size];

    group.bench_function("Naive_256", |bench| {
        bench.iter(|| {
            NaiveGemm
                .gemm(
                    black_box(size), black_box(size), black_box(size),
                    black_box(&a), size, black_box(&b), size,
                    black_box(&mut result), size,
                )
                .unwrap();
        });
    });

    group.bench_function("Tiled64_256", |bench| {
        bench.iter(|| {
            TiledGemm::<64>
                .gemm(
                    black_box(size), black_box(size), black_box(size),
                    black_box(&a), size, black_box(&b), size,
                    black_box(&mut result), size,
                )
                .unwrap();
        });
    });

    group.bench_function("SimdNeon_256", |bench| {
        bench.iter(|| {
            SimdGemm::<64>
                .gemm(
                    black_box(size), black_box(size), black_box(size),
                    black_box(&a), size, black_box(&b), size,
                    black_box(&mut result), size,
                )
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_naive, bench_tiled, bench_simd, bench_comparison);
criterion_main!(benches);
