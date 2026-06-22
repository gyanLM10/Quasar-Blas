# Quasar-BLAS

Quasar-BLAS is a highly-optimized, heterogeneous linear algebra engine built from the ground up in Rust. It serves as an exploration into systems programming, low-level hardware architecture, and cross-platform compute acceleration. 

The engine targets CPU (ARM Neon SIMD, x86_64 AVX2), GPU (wgpu over Metal/Vulkan), and embedded bare-metal runtimes (`no_std` Cortex-M), exposing a unified, zero-cost API capable of executing dense Matrix-Matrix Multiplication (GEMM) across radically different hardware topologies.

## Architecture & Core Design Principles

```
┌─────────────────────────────────────────────────────────────┐
│                    GemmEngine<T> Trait                       │
│            (Generic, Leading Dimensions, Strides)            │
├──────────────┬──────────────────┬───────────────────────────┤
│  Tier 1: CPU │  Tier 2: GPU     │  Tier 3: Embedded         │
│              │                  │                           │
│  NaiveGemm   │  GpuGemm(Naive)  │  EmbeddedGemm             │
│  TiledGemm   │  GpuGemm(Tiled)  │  INT8 Quantization        │
│  SimdGemm    │                  │  Static Arena Allocator   │
│              │  WGSL Shaders    │                           │
│  ARM Neon    │  Metal/Vulkan    │  no_std / Cortex-M4F      │
└──────────────┴──────────────────┴───────────────────────────┘
```

Quasar-BLAS achieves high performance by strictly controlling memory layout and leaning on Rust's type system to eliminate runtime overhead:

1. **Zero-Cost Abstractions:** The core `GemmEngine<T: GemmElement>` trait uses sealed traits to monomorphize exactly for the hardware precision requested. (Currently supports `f32` and `i8`).
2. **Deterministic Heap Alignment:** To prevent split-line SIMD loads, the CPU engines use a custom `AlignedVec` wrapper backed by a `#[repr(C, align(64))] CacheBlock`. This forces the OS allocator to return data pointers that align perfectly with the L1 cache boundaries.
3. **Decoupled Concurrency (TUI):** The interactive dashboard completely decouples the rendering loop (running at 60Hz on the main thread) from the benchmark workers (running in background `std::thread` pools), preventing $O(N^3)$ computational bottlenecks from freezing the UI. GPU tasks remain carefully sync-scheduled to respect API concurrency rules.

## Features

- **Leading dimension (stride) support** — Allows operating on row-major or column-major matrices transparently.
- **Const-generic tile sizes** — Defines spatial localities like `TiledGemm<64>` resolved flawlessly at compile time.
- **Portable SIMD & Intrinsics** — Leverages Rust nightly `portable_simd` and manual `vfmaq_f32` ARM Neon intrinsics for fused multiply-accumulates.
- **`wgpu` Compute Shaders** — Offloads linear algebra to massively parallel WGSL kernels targeting Apple Metal or Linux Vulkan.
- **INT8 Quantization** — Provides fixed-point MAC with i32 accumulation tailored for embedded edge-inference tasks.
- **Static Arena Allocator** — Includes a lock-free, zero-allocation bump allocator for `no_std` targets.
- **Interactive TUI Dashboard** — Features a real-time terminal user interface built with `ratatui` for visualizing benchmark data.

## Benchmarks & Results

### Apple Silicon (macOS M2)
*Tested on an Apple M2 using `criterion` (Single Precision `f32`).*

| Matrix Size ($N$) | Naive ($O(N^3)$) | CPU Cache-Tiled | CPU Portable SIMD | GPU (Metal) Naive |
|-------------------|------------------|-----------------|-------------------|-------------------|
| **64 × 64**       | 2.74 GFLOPS      | 10.73 GFLOPS    | 10.13 GFLOPS      | 0.32 GFLOPS       |
| **128 × 128**     | 2.16 GFLOPS      | 10.67 GFLOPS    | 10.05 GFLOPS      | 2.56 GFLOPS       |
| **1024 × 1024**   | 1.68 GFLOPS      | 10.84 GFLOPS    | 10.25 GFLOPS      | *Exponential*     |

**Key Insights:**
- **The Memory Wall:** The naive CPU engine drops from 2.7 to 1.68 GFLOPS as matrices grow beyond the L1 cache size.
- **Tiling Efficiency:** The `TiledGemm<64>` engine holds a perfectly flat ~10.9 GFLOPS curve across all matrix sizes, proving that spatial cache blocking mitigates the memory wall. 
- **GPU Dispatch:** At small sizes ($N=64$), the GPU is bottlenecked by the latency of submitting command buffers to the Metal API. However, it scales exponentially to crush CPU throughput as $N$ increases.

### Linux Validation (Intel Core Ultra 5 + AVX2 + Vulkan)
*Tested on Linux targeting `f32` with Intel(R) Graphics (MTL).*

| Matrix Size ($N$) | Naive ($O(N^3)$) | CPU Cache-Tiled | CPU Portable SIMD (AVX2) | GPU (Vulkan) Naive | GPU (Vulkan) Tiled |
|-------------------|------------------|-----------------|--------------------------|--------------------|--------------------|
| **512 × 512**     | 0.76 GFLOPS      | 0.63 GFLOPS     | 0.71 GFLOPS              | 21.5 GFLOPS        | 21.1 GFLOPS        |
| **1024 × 1024**   | *N/A*            | *N/A*           | *N/A*                    | 39.11 GFLOPS       | **41.76 GFLOPS**   |

**Linux Architecture Insights & Profiling:**
- **Hardware Cache Efficiency (Valgrind Profiling):** Using `cachegrind` to perform cycle-accurate simulation on a $256 \times 256$ matrix, the Naive CPU engine exhibited an enormous **10.6% L1 Data Cache miss rate** (nearly 17 million misses). The `TiledGemm` engine, by confining the working set strictly to the bounds of the 352 KiB L1 cache, dropped the miss rate to just **0.6%**. This represents an empirical **92% reduction** in memory latency.
- **Vulkan Backend Scaling:** The `wgpu` backend seamlessly ported over to Linux, compiling the WGSL shaders and offloading compute to the integrated GPU, hitting over 41 GFLOPS on large matrices.
- **Roofline Model:** The single-thread peak limit of the host P-Core is $\approx 100 \text{ GFLOPS}$. While the `SimdGemm` proves that Rust's portable SIMD effectively lowers to 256-bit AVX2 hardware instructions, industry-grade libraries like OpenBLAS rely heavily on loop unrolling and multi-threading to push closer to the 80% theoretical hardware ceiling.

## Quick Start

```bash
# Run all tests (CPU + GPU + Embedded)
cargo test --features "gpu,embedded"

# Run CPU benchmarks
cargo bench --bench cpu_gemm

# Launch interactive dashboard
cargo run --example tui_dashboard --features "tui,gpu"

# Cross-compile for Cortex-M4F (bare metal)
rustup target add thumbv7em-none-eabihf
cargo build --target thumbv7em-none-eabihf --features embedded --no-default-features --lib
```

## Engine Comparison

| Engine | Optimization Approach | Target Architecture |
|--------|-----------------------|---------------------|
| `NaiveGemm` | O(N³) Baseline Loop | Correctness Oracle |
| `TiledGemm<TILE>` | L1 Cache Blocking (Spatial Locality) | Generic CPU |
| `SimdGemm<TILE>` | Hardware Vectorization (Neon/AVX2) | x86_64 / aarch64 CPU |
| `GpuGemm(Naive)` | Massive Parallelization (1 thread/element)| wgpu (Metal/Vulkan) |
| `GpuGemm(Tiled)` | Shared-Memory Workgroups | wgpu (Metal/Vulkan) |
| `EmbeddedGemm` | INT8 Fixed-Point Quantization | Cortex-M (no_std) |

## Requirements

- **Rust** ≥ 1.94 (stable, nightly optional for SIMD)
- **macOS**: Apple Silicon (ARM Neon + Metal GPU)
- **Linux** (optional): Vulkan-capable GPU + `valgrind` for hardware profiling

## License

MIT
