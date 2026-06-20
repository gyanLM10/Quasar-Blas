# Phase 6: Linux Hardware Validation Guide

This document is for the AI agent (or developer) operating in the Linux server environment. The Quasar-BLAS project (Phases 1-5) is feature-complete and has been built and tested on macOS (Apple Silicon + Metal).

Your objective in **Phase 6** is to validate the engine's performance using Linux-specific hardware performance counters and the Vulkan graphics backend. 

Follow these steps exactly:

## 1. Environment Preparation
Ensure the Linux server is ready for profiling and Vulkan compute:
1. **Toolchain:** Ensure the `nightly` Rust toolchain is installed (`rustup default nightly`), as the portable SIMD feature strictly requires it.
2. **Perf Tools:** Install the `linux-tools-common` and `linux-tools-generic` packages to get access to the `perf` command. Ensure the user has `perf` execution permissions (e.g., `sudo sysctl -w kernel.perf_event_paranoid=-1`).
3. **GPU Drivers:** Verify that Vulkan drivers are installed and functional (e.g., `vulkaninfo` or `clinfo` to check discrete GPU presence).

---

## 2. Hardware Cache Profiling (`perf stat`)
We need empirical proof that the `TiledGemm` eliminates the L1 cache misses seen in the `NaiveGemm`.

**Action:** Run both the Naive and Tiled benchmarks wrapped in `perf stat`, targeting the L1 data cache load misses.

```bash
# 1. Profile the Naive Engine (Expect HIGH cache misses)
perf stat -e L1-dcache-loads,L1-dcache-load-misses cargo bench --bench cpu_gemm -- "cpu_naive/NaiveGemm/1024"

# 2. Profile the Tiled Engine (Expect LOW cache misses, < 3%)
perf stat -e L1-dcache-loads,L1-dcache-load-misses cargo bench --bench cpu_gemm -- "cpu_tiled/TiledGemm<64>/1024"
```

**Expected Result:** The ratio of `L1-dcache-load-misses` to `L1-dcache-loads` should be drastically lower for the Tiled version, mathematically proving the cache optimization. Document these percentages in `results.md`.

---

## 3. GPU Vulkan Backend Validation
On macOS, `wgpu` automatically compiled our WGSL compute shaders to Apple Metal. On Linux, we need to prove it seamlessly cross-compiles to Vulkan and scales on a discrete GPU.

**Action:** Run the GPU benchmark suite.
```bash
cargo bench --bench gpu_gemm --features "gpu"
```

**Expected Result:** 
- The console should output `[Quasar-BLAS GPU] Adapter: <GPU Name> | Backend: Vulkan`.
- Ensure the `GpuGemm(Naive)` vs `GpuGemm(Tiled)` executes successfully without panic.
- Expect GPU GFLOPS to vastly outperform the CPU on the `1024` matrix sizes due to discrete GPU massive parallelization.

---

## 4. x86_64 AVX2 / FMA Validation
The `SimdGemm` uses `core::simd::f32x4`. On macOS, LLVM lowered this to ARM Neon instructions. On a modern Linux x86 box, LLVM should lower this to AVX2/FMA instructions.

**Action:** Run the SIMD benchmarks and observe throughput.
```bash
cargo bench --bench cpu_gemm -- "cpu_simd"
```

**Expected Result:** The SIMD benchmarks should match or closely rival the `TiledGemm` benchmarks, proving that the `#![feature(portable_simd)]` seamlessly generated native 256-bit wide hardware vector instructions for Intel/AMD chips without any code alterations.

---

## 5. Industry Benchmarking (OpenBLAS & Intel MKL)
To truly gauge the systems engineering success of Quasar-BLAS, it must be benchmarked against the gold standards of linear algebra.

**Action:**
1. Install `libopenblas-dev` on the Linux host.
2. If Intel MKL is available via Intel oneAPI, source its environment (`source /opt/intel/oneapi/setvars.sh`).
3. Create a new criterion benchmark file `benches/industry_gemm.rs` that imports the `cblas-sys` and `openblas-src` crates.
4. Write a benchmark wrapper that calls `cblas_sgemm` (Single-precision GEMM) using the same matrix sizes ($N=64$ to $1024$) and exactly the same row-major arrays used in `cpu_gemm.rs`.
5. Run the comparison benchmark:
   ```bash
   cargo bench --bench industry_gemm
   ```

**Expected Result:** 
- OpenBLAS/Intel MKL are written in highly optimized assembly and will serve as the absolute ceiling for single-threaded CPU throughput. 
- You should document the ratio of Quasar-BLAS `TiledGemm` / `SimdGemm` performance compared to OpenBLAS (e.g., "Quasar-BLAS achieves 85% of OpenBLAS throughput on $N=1024$").

---

## 6. Final Reporting
If all validations (Cache Counters, Vulkan, AVX2, and OpenBLAS comparisons) pass, append the new Linux findings and charts to the bottom of the existing `results.md` file to finalize the project documentation.
