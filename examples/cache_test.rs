use quasar_blas::GemmEngine;
use quasar_blas::cpu::{NaiveGemm, TiledGemm};

fn main() {
    let n = 256;
    let a = vec![1.0f32; n * n];
    let b = vec![2.0f32; n * n];
    let mut c = vec![0.0f32; n * n];

    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "naive" {
        NaiveGemm.gemm(n, n, n, &a, n, &b, n, &mut c, n).unwrap();
    } else if args.len() > 1 && args[1] == "tiled" {
        TiledGemm::<64>.gemm(n, n, n, &a, n, &b, n, &mut c, n).unwrap();
    } else {
        println!("Pass 'naive' or 'tiled'");
    }
}
