# Quasar-BLAS ⚡

A heterogeneous, high-performance linear algebra engine in Rust, targeting CPU (ARM Neon SIMD), GPU (wgpu/Metal/Vulkan), and embedded bare-metal runtimes.

## Architecture

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

## Features

- **Generic `GemmEngine<T>`** trait with zero-cost abstractions
- **Leading dimension (stride) support** — row-major or column-major transparently
- **Const-generic tile sizes** — `TiledGemm<64>` resolved at compile time
- **ARM Neon SIMD** — `vfmaq_f32` fused multiply-accumulate on Apple Silicon
- **wgpu compute shaders** — naive and shared-memory tiled WGSL kernels
- **INT8 quantization** — fixed-point MAC with i32 accumulation for edge inference
- **Static arena allocator** — lock-free, zero-allocation bump allocator for `no_std`
- **Property-based testing** — 700+ randomized test cases via proptest + ndarray oracle
- **Interactive TUI dashboard** — ratatui-powered benchmark visualization

## Architecture

Quasar-BLAS achieves high performance by strictly controlling memory layout and leaning on Rust's type system to eliminate runtime overhead:

1. **Zero-Cost Abstractions:** The core `GemmEngine<T: GemmElement>` trait uses sealed traits to monomorphize exactly for the hardware precision requested (currently `f32`, extending to `i8`).
2. **Deterministic Heap Alignment:** To prevent split-line SIMD loads, the CPU engines use a custom `AlignedVec` wrapper backed by a `#[repr(C, align(64))] CacheBlock`. This forces the OS allocator to return data pointers that start perfectly on an L1 cache line boundary.
3. **Decoupled Concurrency (TUI):** The dashboard completely decouples the rendering loop (60Hz on main thread) from the benchmark workers (background `std::thread`), preventing the $O(N^3)$ CPU bottlenecks from freezing the UI. GPU tasks remain carefully sync-scheduled on the main thread to respect macOS Metal API concurrency rules.

## Benchmarks & Results

*Tested on Apple M2 (macOS) using `criterion` (Single Precision `f32`).*

| Matrix Size ($N$) | Naive ($O(N^3)$) | CPU Cache-Tiled | CPU Portable SIMD | GPU (Metal) Naive |
|-------------------|------------------|-----------------|-------------------|-------------------|
| **64 × 64**       | 2.74 GFLOPS      | 10.73 GFLOPS    | 10.13 GFLOPS      | 0.32 GFLOPS       |
| **128 × 128**     | 2.16 GFLOPS      | 10.67 GFLOPS    | 10.05 GFLOPS      | 2.56 GFLOPS       |
| **1024 × 1024**   | 1.68 GFLOPS      | 10.84 GFLOPS    | 10.25 GFLOPS      | *scaling*         |

### Key Takeaways
- **The Memory Wall:** The naive CPU engine drops from 2.7 to 1.68 GFLOPS as matrices grow beyond the L1 cache size.
- **Tiling Efficiency:** `TiledGemm<64>` holds a perfectly flat ~10.9 GFLOPS curve across all sizes, proving the cache blocking strategy works. LLVM auto-vectorization on the M2 makes it match/exceed manual `core::simd` intrinsics.
- **GPU Dispatch Overhead:** At small sizes ($N=64$), the GPU is bottlenecked by the latency of submitting command buffers to Metal (0.32 GFLOPS). However, it scales exponentially as $N$ increases (800% jump at $N=128$).

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

## Feature Flags

| Flag | Description |
|------|-------------|
| `std` (default) | Standard library support |
| `gpu` | wgpu compute pipeline (Metal/Vulkan) |
| `tui` | Ratatui interactive dashboard |
| `embedded` | `no_std` bare-metal runtime |

## Engine Comparison

| Engine | Optimization | Target |
|--------|-------------|--------|
| `NaiveGemm` | O(N³) baseline | Correctness oracle |
| `TiledGemm<64>` | L1 cache blocking | CPU (any) |
| `SimdGemm<64>` | Neon SIMD + tiling | CPU (aarch64) |
| `GpuGemm(Naive)` | 1 thread/element | GPU (Metal/Vulkan) |
| `GpuGemm(Tiled)` | Shared memory tiling | GPU (Metal/Vulkan) |
| `EmbeddedGemm` | INT8 quantized | Cortex-M4F |

## Requirements

- **Rust** ≥ 1.94 (stable)
- **macOS**: Apple Silicon (ARM Neon + Metal GPU)
- **Linux** (optional): Vulkan-capable GPU + `perf` for hardware validation

## License

MIT
