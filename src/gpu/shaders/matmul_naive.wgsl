// Quasar-BLAS: Naive GEMM compute shader
//
// One thread per output element. Each thread independently computes
// the full dot product for its assigned C[row, col] position.
// No shared memory, no cooperation between threads.

struct Dimensions {
    M: u32,
    K: u32,
    N: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;

    // Bounds check: threads outside the matrix dimensions are no-ops
    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var p: u32 = 0u; p < dims.K; p = p + 1u) {
        sum = sum + A[row * dims.K + p] * B[p * dims.N + col];
    }

    C[row * dims.N + col] = sum;
}
