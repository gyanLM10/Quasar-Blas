# Quasar-BLAS: High-Performance CPU Linear Algebra Engine

Quasar-BLAS is a highly-optimized, CPU-native linear algebra engine built from the ground up in Rust. It serves as a deep dive into systems programming, focusing on how cache-aware memory access patterns and hardware SIMD (Single Instruction, Multiple Data) vectorization can dramatically accelerate dense matrix multiplication (GEMM) compared to a naive textbook implementation.

By carefully controlling memory layout and leveraging Rust's type system, Quasar-BLAS achieves significant speedups while providing a zero-cost, type-safe API.

---

## 🏗️ Architectural Decisions

### The Universal `GemmEngine<T>` Trait

At the core of the engine is the `GemmEngine<T: GemmElement>` trait, which provides a unified API for General Matrix Multiply: $C = A \times B$.

```rust
pub trait GemmEngine<T: GemmElement> {
    fn gemm(&self, m: usize, k: usize, n: usize, 
            a: &[T], lda: usize, 
            b: &[T], ldb: usize, 
            c: &mut [T], ldc: usize) -> Result<(), GemmError>;
}
```

**Design Choices:**
1.  **Strided Memory (Leading Dimensions):** Instead of forcing a specific matrix type, the API takes raw slices and `lda`, `ldb`, `ldc` (leading dimensions). This allows the caller to pass sub-matrices or transparently switch between row-major and column-major data without copying. Element $(i, j)$ is accessed as `data[i * ld + j]`.
2.  **Zero-Cost Abstractions:** `GemmElement` is a sealed trait implemented for `f32`. All operations (like `zero()`, `one()`, `mul_add()`) are marked `#[inline(always)]`. The compiler monomorphizes the trait away entirely, resulting in zero runtime dispatch overhead.

### Deterministic Heap Alignment

Modern CPUs load memory in 64-byte chunks (cache lines). If a SIMD vector load crosses a cache line boundary, the CPU suffers a "split-line penalty," requiring two separate L1 cache accesses.

**Solution (`AlignedVec`):**
We circumvent the standard Rust `Vec` allocator for matrix storage. Instead, we use a custom `CacheBlock` type:
```rust
#[repr(C, align(64))]
pub struct CacheBlock { pub _data: [u8; 64], }
```
The `AlignedVec` allocates these 64-byte blocks and transmutes them into `&[f32]`. This guarantees that every row in our matrix starts perfectly aligned to an L1 cache line boundary, ensuring optimal SIMD load/store performance and predictable prefetching.

---

## 🚀 Algorithmic Evolution & Optimizations

We built three distinct engines to measure the exact impact of different optimization techniques.

### 1. The Baseline: `NaiveGemm`
*The Correctness Oracle*

This is the standard $O(N^3)$ triple-nested loop (`i → j → k`). 
**The Problem (The Memory Wall):** In a row-major layout, the inner loop iterates over columns of $B$, striding by $ldb$. This causes constant cache misses because the CPU loads a full 64-byte cache line just to read one 4-byte `f32`, and then immediately jumps to the next row, evicting the line before it can be reused. Performance drops exponentially as matrices exceed the L1 cache size.

### 2. Spatial Locality: `TiledGemm<TILE>`
*Defeating the Memory Wall via Cache Blocking*

To solve cache thrashing, we break the matrix into small $TILE \times TILE$ blocks.
**Design Choices:**
- **Const-Generics:** We use `TiledGemm<64>` where `64` is a compile-time constant. This allows the compiler to fully unroll the tile loops and eliminate bounds checking within the micro-kernel.
- **Loop Reordering (`ii → kk → jj → i → k → j`):** 
  We load a block of $A$, a block of $B$, and accumulate into a block of $C$. The inner loops are reordered so we traverse $C$ and $B$ sequentially (maximizing spatial locality) while broadcasting a single scalar from $A$ across the row (maximizing temporal locality).
- **L1 Cache Math:** A $64 \times 64$ tile of `f32` is 16 KB. Holding three tiles ($A, B, C$) takes 48 KB, which fits perfectly within the 128 KB L1 Data Cache of an Apple M-series chip or the 32 KB/core L1d of an AMD/Intel processor, dropping cache misses by over 90%.

### 3. Hardware Vectorization: `SimdGemm<TILE>`
*Pushing the Silicon Limits*

This engine takes the cache-tiled layout and applies Rust's `#![feature(portable_simd)]` (`core::simd`) to process 4 elements per instruction cycle.
**Design Choices:**
- **Portable Architecture:** Instead of writing raw assembly or unsafe architecture-specific intrinsics (like `_mm256_fmadd_ps`), `core::simd` allows the code to transparently compile to ARM Neon on Apple Silicon or AVX2/AVX-512 on x86_64.
- **Fused Multiply-Add (FMA):** 
  ```rust
  let result = a_vec.mul_add(b_vec, c_vec); // c = a * b + c
  ```
  This line compiles to a single hardware FMA instruction, doing a multiply and an add in one clock cycle with no intermediate rounding loss.

---

## 📊 Benchmarking Methodology

We use [Criterion.rs](https://bheisler.github.io/criterion.rs/book/index.html) to perform rigorous, statistically sound benchmarking.

**How We Benchmark:**
1.  **Deterministic Generation:** Matrices are populated using a pseudo-random seed to ensure every benchmark run processes the exact same entropy and bit-patterns, preventing branch prediction anomalies between runs.
2.  **Warm-up & Sampling:** Criterion runs a warm-up phase to ensure CPU frequency scaling (Turbo Boost) has engaged and the instruction cache is hot. It then takes hundreds of samples and calculates confidence intervals.
3.  **Throughput Measurement (GFLOPS):** 
    We define the number of floating-point operations for an $M \times K \times N$ GEMM as $2 \times M \times K \times N$ (one multiply and one add per element). 
    Criterion divides this by the measured time to report **GFLOPS** (Giga-Floating Point Operations Per Second), allowing us to directly compare our software against the theoretical hardware limits of the CPU.

**Run the benchmarks:**
```bash
cargo bench --bench cpu_gemm
```
*Reports are generated in `target/criterion/report/index.html`.*

---

## 🧪 Testing Strategy & Verification

Correctness is verified via mathematical property-based testing using `proptest`. Instead of hardcoded assertions, we generate thousands of randomized matrices (including non-square and non-power-of-2 dimensions) and test structural invariants.

**1. The Oracle Test:**
We compare the output of our optimized engines (`TiledGemm`, `SimdGemm`) against the industry standard `ndarray::Array2::dot()` implementation. We assert that element-wise differences are within a strict $\epsilon = 10^{-3}$ tolerance.

**2. Algebraic Invariants:**
- **Identity Property:** Generates random matrices $A$, multiplies by an identity matrix $I$, and ensures $A \times I = A$.
- **Transpose Property:** Verifies that $(A \times B)^T = B^T \times A^T$.

**Run the verification suite:**
```bash
cargo test
```

---

## 💻 Quick Start

```bash
# 1. Ensure you have Rust nightly installed (required for portable_simd)
rustup default nightly

# 2. Run the property-based verification tests
cargo test

# 3. Run the statistical benchmarks
cargo bench --bench cpu_gemm

# 4. Run a quick interactive comparison
cargo run --example cache_test -- tiled

# 5. Run the Database Integration example (ClickHouse -> Quasar-BLAS -> Pinecone)
cargo run --example db_integration
```

---

## 🗄️ Database Integration (Analytics & ML)

While Quasar-BLAS is a computational engine, it pairs powerfully with databases for large-scale data workflows. Here are the three primary architectural patterns:

### 1. The Machine Learning Pipeline (Vector DBs)
Real-world dense matrices often represent relational embeddings or user/item features.
Quasar-BLAS functions as the high-performance computational core:
1. **Extraction (ClickHouse / DuckDB):** Pull raw relational feature data from a fast analytical database.
2. **Computation (Quasar-BLAS):** Use the `SimdGemm` engine to compute massive Similarity/Gram matrices ($A \times A^T$).
3. **Ingestion (Pinecone / Qdrant):** Upsert the computed similarity profiles directly into a Vector Database for downstream AI similarity search.

*We provide a complete, executable workflow demonstrating this pipeline in `examples/db_integration.rs`.*

### 2. Out-of-Core Processing (Huge Matrices)
For matrices that exceed system RAM (e.g., 100+ GB), you can use a database or memory-mapped key-value store (like **LMDB** or **RocksDB**) to store the matrix on disk.
By reading the matrix in small "chunks" (tiles) from the database into Quasar-BLAS's `AlignedVec`, multiplying them using our `TiledGemm` engine, and writing the results back to disk, you can process virtually infinite amounts of data without OOM (Out Of Memory) errors.

### 3. CI/CD Benchmark Telemetry
To prevent performance regressions (e.g., a code change accidentally destroying SIMD auto-vectorization), benchmark results should be tracked over time.
Instead of just viewing Criterion's HTML output, you can log the exact L1 cache miss rates and GFLOPS throughput to a time-series database (like **InfluxDB**) or **SQLite**. This allows CI/CD pipelines to automatically fail pull requests if matrix multiplication performance drops.

---

### Running the Example
To run the provided machine learning pipeline example with real databases, set the environment variables:
```bash
CLICKHOUSE_URL="http://localhost:8123" \
PINECONE_HOST="https://your-index-host.svc.pinecone.io" \
PINECONE_API_KEY="your-api-key" \
cargo run --example db_integration
```
*(If the variables are omitted, the example gracefully falls back to mock data to demonstrate the structure.)*

## License
MIT
