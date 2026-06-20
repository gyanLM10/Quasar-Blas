// Quasar-BLAS: Tiled GEMM compute shader with shared memory
//
// Threads within a 16×16 workgroup cooperatively load tiles of A and B
// into ultra-fast GPU shared memory (SRAM), then compute the partial
// dot products from shared memory instead of slow global memory.
//
// This reduces global memory traffic by a factor of TILE_SIZE (16×),
// which is the dominant performance win for memory-bound GPU kernels.

const TILE_SIZE: u32 = 16u;

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

// Shared memory tiles — accessible to all threads in the workgroup
var<workgroup> tile_A: array<f32, 256>; // 16 × 16
var<workgroup> tile_B: array<f32, 256>; // 16 × 16

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;

    var sum: f32 = 0.0;

    // Number of tiles we need to iterate over the K dimension
    let num_tiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // --- Phase 1: Cooperative tile loading ---

        // Each thread loads one element of A into shared memory
        let a_col = t * TILE_SIZE + local_col;
        if (row < dims.M && a_col < dims.K) {
            tile_A[local_row * TILE_SIZE + local_col] = A[row * dims.K + a_col];
        } else {
            tile_A[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Each thread loads one element of B into shared memory
        let b_row = t * TILE_SIZE + local_row;
        if (b_row < dims.K && col < dims.N) {
            tile_B[local_row * TILE_SIZE + local_col] = B[b_row * dims.N + col];
        } else {
            tile_B[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // --- Synchronization barrier ---
        // All threads must finish loading before any thread starts computing
        workgroupBarrier();

        // --- Phase 2: Compute partial dot product from shared memory ---
        for (var p: u32 = 0u; p < TILE_SIZE; p = p + 1u) {
            sum = sum + tile_A[local_row * TILE_SIZE + p] * tile_B[p * TILE_SIZE + local_col];
        }

        // --- Synchronization barrier ---
        // All threads must finish computing before the next tile is loaded
        workgroupBarrier();
    }

    // Write result to global memory (with bounds check)
    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = sum;
    }
}
