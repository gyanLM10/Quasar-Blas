extern crate openblas_src;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cblas_sys::{cblas_sgemm, CblasRowMajor, CblasNoTrans};

const SIZES: &[usize] = &[64, 128, 256, 512, 1024];

fn generate_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((seed.wrapping_mul(i as u64 + 1).wrapping_add(7)) % 2000) as f32 / 100.0 - 10.0)
        .collect()
}

fn bench_openblas(c: &mut Criterion) {
    let mut group = c.benchmark_group("industry_openblas");

    for &size in SIZES {
        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));

        let a = generate_matrix(size, size, 42);
        let b = generate_matrix(size, size, 137);
        let mut result = vec![0.0f32; size * size];

        group.bench_with_input(
            BenchmarkId::new("OpenBLAS", size),
            &size,
            |bench, &n| {
                bench.iter(|| {
                    unsafe {
                        cblas_sgemm(
                            CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            n as i32,
                            n as i32,
                            n as i32,
                            1.0,
                            black_box(a.as_ptr()),
                            n as i32,
                            black_box(b.as_ptr()),
                            n as i32,
                            0.0,
                            black_box(result.as_mut_ptr()),
                            n as i32,
                        );
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_openblas);
criterion_main!(benches);
