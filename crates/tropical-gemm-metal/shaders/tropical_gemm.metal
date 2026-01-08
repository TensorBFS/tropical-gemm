// Tropical GEMM Metal Shaders
// High-performance tropical matrix multiplication for Apple GPUs

#include <metal_stdlib>
using namespace metal;

// Blocking parameters
constant int BLOCK_SIZE_M = 32;
constant int BLOCK_SIZE_N = 32;
constant int BLOCK_SIZE_K = 32;
constant int THREAD_SIZE_M = 4;
constant int THREAD_SIZE_N = 4;

// Helper macro for column-major indexing
#define OFFSET_COL(row, col, ld) ((col) * (ld) + (row))

// ============================================================================
// TropicalMaxPlus<f32>: C[i,j] = max_k(A[i,k] + B[k,j])
// ============================================================================

kernel void tropical_maxplus_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const int threads_per_group_m = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int threads_per_group_n = BLOCK_SIZE_N / THREAD_SIZE_N;

    // Shared memory for tiles
    threadgroup float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    threadgroup float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    // Accumulators in registers
    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    // Initialize with -infinity (tropical zero for MaxPlus)
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = -INFINITY;
    }

    const int linear_tid = tid.y * threads_per_group_m + tid.x;
    const int THREAD_NUM = threads_per_group_m * threads_per_group_n;

    const int A_TILE_ROW = linear_tid % BLOCK_SIZE_M;
    const int A_TILE_COL = linear_tid / BLOCK_SIZE_M;
    const int B_TILE_ROW = linear_tid % BLOCK_SIZE_K;
    const int B_TILE_COL = linear_tid / BLOCK_SIZE_K;

    const int A_STRIDE = THREAD_NUM / BLOCK_SIZE_M;
    const int B_STRIDE = THREAD_NUM / BLOCK_SIZE_K;

    // Loop over K dimension tiles
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        // Load A tile cooperatively
        for (int i = 0; i < BLOCK_SIZE_K; i += A_STRIDE) {
            int row = BLOCK_SIZE_M * gid.x + A_TILE_ROW;
            int col = A_TILE_COL + i + tile_idx;
            float val = -INFINITY;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        // Load B tile cooperatively
        for (int i = 0; i < BLOCK_SIZE_N; i += B_STRIDE) {
            int row = tile_idx + B_TILE_ROW;
            int col = BLOCK_SIZE_N * gid.y + i + B_TILE_COL;
            float val = -INFINITY;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute tile contribution
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            // Load A values into registers
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(tid.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }

            // Load B values into registers
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, tid.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }

            // Compute outer product with tropical operations
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    float prod = regs_a[tm] + regs_b[tn];
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    accum[idx] = max(accum[idx], prod);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
            int row = BLOCK_SIZE_M * gid.x + THREAD_SIZE_M * tid.x + tm;
            int col = BLOCK_SIZE_N * gid.y + THREAD_SIZE_N * tid.y + tn;
            if (row < M && col < N) {
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)];
            }
        }
    }
}

// ============================================================================
// TropicalMinPlus<f32>: C[i,j] = min_k(A[i,k] + B[k,j])
// ============================================================================

kernel void tropical_minplus_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const int threads_per_group_m = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int threads_per_group_n = BLOCK_SIZE_N / THREAD_SIZE_N;

    threadgroup float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    threadgroup float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    // Initialize with +infinity (tropical zero for MinPlus)
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = INFINITY;
    }

    const int linear_tid = tid.y * threads_per_group_m + tid.x;
    const int THREAD_NUM = threads_per_group_m * threads_per_group_n;

    const int A_TILE_ROW = linear_tid % BLOCK_SIZE_M;
    const int A_TILE_COL = linear_tid / BLOCK_SIZE_M;
    const int B_TILE_ROW = linear_tid % BLOCK_SIZE_K;
    const int B_TILE_COL = linear_tid / BLOCK_SIZE_K;

    const int A_STRIDE = THREAD_NUM / BLOCK_SIZE_M;
    const int B_STRIDE = THREAD_NUM / BLOCK_SIZE_K;

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        for (int i = 0; i < BLOCK_SIZE_K; i += A_STRIDE) {
            int row = BLOCK_SIZE_M * gid.x + A_TILE_ROW;
            int col = A_TILE_COL + i + tile_idx;
            float val = INFINITY;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        for (int i = 0; i < BLOCK_SIZE_N; i += B_STRIDE) {
            int row = tile_idx + B_TILE_ROW;
            int col = BLOCK_SIZE_N * gid.y + i + B_TILE_COL;
            float val = INFINITY;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(tid.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, tid.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    float prod = regs_a[tm] + regs_b[tn];
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    accum[idx] = min(accum[idx], prod);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
            int row = BLOCK_SIZE_M * gid.x + THREAD_SIZE_M * tid.x + tm;
            int col = BLOCK_SIZE_N * gid.y + THREAD_SIZE_N * tid.y + tn;
            if (row < M && col < N) {
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)];
            }
        }
    }
}

// ============================================================================
// TropicalMaxMul<f32>: C[i,j] = max_k(A[i,k] * B[k,j])
// ============================================================================

kernel void tropical_maxmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const int threads_per_group_m = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int threads_per_group_n = BLOCK_SIZE_N / THREAD_SIZE_N;

    threadgroup float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    threadgroup float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    // Initialize with 0 (tropical zero for MaxMul)
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = 0.0f;
    }

    const int linear_tid = tid.y * threads_per_group_m + tid.x;
    const int THREAD_NUM = threads_per_group_m * threads_per_group_n;

    const int A_TILE_ROW = linear_tid % BLOCK_SIZE_M;
    const int A_TILE_COL = linear_tid / BLOCK_SIZE_M;
    const int B_TILE_ROW = linear_tid % BLOCK_SIZE_K;
    const int B_TILE_COL = linear_tid / BLOCK_SIZE_K;

    const int A_STRIDE = THREAD_NUM / BLOCK_SIZE_M;
    const int B_STRIDE = THREAD_NUM / BLOCK_SIZE_K;

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        for (int i = 0; i < BLOCK_SIZE_K; i += A_STRIDE) {
            int row = BLOCK_SIZE_M * gid.x + A_TILE_ROW;
            int col = A_TILE_COL + i + tile_idx;
            float val = 0.0f;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        for (int i = 0; i < BLOCK_SIZE_N; i += B_STRIDE) {
            int row = tile_idx + B_TILE_ROW;
            int col = BLOCK_SIZE_N * gid.y + i + B_TILE_COL;
            float val = 0.0f;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(tid.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, tid.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    // MaxMul: tropical_mul = *, tropical_add = max
                    float prod = regs_a[tm] * regs_b[tn];
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    accum[idx] = max(accum[idx], prod);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
            int row = BLOCK_SIZE_M * gid.x + THREAD_SIZE_M * tid.x + tm;
            int col = BLOCK_SIZE_N * gid.y + THREAD_SIZE_N * tid.y + tn;
            if (row < M && col < N) {
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)];
            }
        }
    }
}
