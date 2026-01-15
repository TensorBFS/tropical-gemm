// Tropical GEMM Metal Shaders
// DRY implementation using Metal shader functions
// Adapted from CUDA implementation for Apple Silicon

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// CONSTANTS AND UTILITIES
// ============================================================================

// Memory layout helpers (column-major)
#define OFFSET_COL(row, col, ld) ((col) * (ld) + (row))

// Integer "infinity" constants (sentinel values for tropical zero)
constant int INF_I32 = 46340;
constant int NEG_INF_I32 = -46340;

// Saturating addition for MaxPlus (propagates -infinity)
inline int saturating_add_maxplus_i32(int a, int b) {
    if (a == NEG_INF_I32 || b == NEG_INF_I32) return NEG_INF_I32;
    return a + b;
}

// Saturating addition for MinPlus (propagates +infinity)
inline int saturating_add_minplus_i32(int a, int b) {
    if (a == INF_I32 || b == INF_I32) return INF_I32;
    return a + b;
}

// ============================================================================
// F32 MAXPLUS GEMM KERNEL
// ============================================================================
// Block sizes for f32: 64x32x64, Thread sizes: 4x4

kernel void tropical_maxplus_f32_nn(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    const uint BLOCK_SIZE_M = 64;
    const uint BLOCK_SIZE_K = 32;
    const uint BLOCK_SIZE_N = 64;
    const uint THREAD_SIZE_M = 4;
    const uint THREAD_SIZE_N = 4;

    const uint bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const uint bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const uint THREAD_NUM_PER_BLOCK = bszm * bszn;

    uint BLOCK_IDX = gid.x;
    uint BLOCK_IDY = gid.y;

    const uint thread_id = tid.y * bszm + tid.x;

    threadgroup float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    threadgroup float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    for (uint i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = -INFINITY;
    }

    const uint A_TILE_COL = thread_id / BLOCK_SIZE_M;
    const uint A_TILE_ROW = thread_id % BLOCK_SIZE_M;
    const uint B_TILE_COL = thread_id / BLOCK_SIZE_K;
    const uint B_TILE_ROW = thread_id % BLOCK_SIZE_K;
    const uint A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const uint B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (uint tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        for (uint i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            uint row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            uint col = A_TILE_COL + i + tile_idx;
            float val = -INFINITY;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        for (uint i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            uint row = tile_idx + B_TILE_ROW;
            uint col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            float val = -INFINITY;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BLOCK_SIZE_K; ++k) {
            for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(tid.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }
            for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, tid.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }
            for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
                for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    float prod = regs_a[tm] + regs_b[tn];
                    uint idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    accum[idx] = max(accum[idx], prod);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
        for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
            uint row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * tid.x + tm;
            uint col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * tid.y + tn;
            if (row < M && col < N) {
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)];
            }
        }
    }
}

// ============================================================================
// F32 MINPLUS GEMM KERNEL
// ============================================================================

kernel void tropical_minplus_f32_nn(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    const uint BLOCK_SIZE_M = 64;
    const uint BLOCK_SIZE_K = 32;
    const uint BLOCK_SIZE_N = 64;
    const uint THREAD_SIZE_M = 4;
    const uint THREAD_SIZE_N = 4;

    const uint bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const uint bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const uint THREAD_NUM_PER_BLOCK = bszm * bszn;

    uint BLOCK_IDX = gid.x;
    uint BLOCK_IDY = gid.y;

    const uint thread_id = tid.y * bszm + tid.x;

    threadgroup float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    threadgroup float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    for (uint i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = INFINITY;
    }

    const uint A_TILE_COL = thread_id / BLOCK_SIZE_M;
    const uint A_TILE_ROW = thread_id % BLOCK_SIZE_M;
    const uint B_TILE_COL = thread_id / BLOCK_SIZE_K;
    const uint B_TILE_ROW = thread_id % BLOCK_SIZE_K;
    const uint A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const uint B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (uint tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        for (uint i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            uint row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            uint col = A_TILE_COL + i + tile_idx;
            float val = INFINITY;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        for (uint i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            uint row = tile_idx + B_TILE_ROW;
            uint col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            float val = INFINITY;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BLOCK_SIZE_K; ++k) {
            for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(tid.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }
            for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, tid.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }
            for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
                for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    float prod = regs_a[tm] + regs_b[tn];
                    uint idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    accum[idx] = min(accum[idx], prod);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
        for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
            uint row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * tid.x + tm;
            uint col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * tid.y + tn;
            if (row < M && col < N) {
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)];
            }
        }
    }
}

// ============================================================================
// F32 MAXMUL GEMM KERNEL
// ============================================================================

kernel void tropical_maxmul_f32_nn(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    const uint BLOCK_SIZE_M = 64;
    const uint BLOCK_SIZE_K = 32;
    const uint BLOCK_SIZE_N = 64;
    const uint THREAD_SIZE_M = 4;
    const uint THREAD_SIZE_N = 4;

    const uint bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const uint bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const uint THREAD_NUM_PER_BLOCK = bszm * bszn;

    uint BLOCK_IDX = gid.x;
    uint BLOCK_IDY = gid.y;

    const uint thread_id = tid.y * bszm + tid.x;

    threadgroup float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    threadgroup float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    for (uint i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = 0.0f;
    }

    const uint A_TILE_COL = thread_id / BLOCK_SIZE_M;
    const uint A_TILE_ROW = thread_id % BLOCK_SIZE_M;
    const uint B_TILE_COL = thread_id / BLOCK_SIZE_K;
    const uint B_TILE_ROW = thread_id % BLOCK_SIZE_K;
    const uint A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const uint B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (uint tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        for (uint i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            uint row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            uint col = A_TILE_COL + i + tile_idx;
            float val = 0.0f;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        for (uint i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            uint row = tile_idx + B_TILE_ROW;
            uint col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            float val = 0.0f;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BLOCK_SIZE_K; ++k) {
            for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(tid.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }
            for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, tid.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }
            for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
                for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    float prod = regs_a[tm] * regs_b[tn];
                    uint idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    accum[idx] = max(accum[idx], prod);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
        for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
            uint row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * tid.x + tm;
            uint col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * tid.y + tn;
            if (row < M && col < N) {
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)];
            }
        }
    }
}

// ============================================================================
// F32 MAXPLUS GEMM WITH ARGMAX KERNEL
// ============================================================================

kernel void tropical_maxplus_f32_nn_with_argmax(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device int* argmax_out [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    const uint BLOCK_SIZE_M = 64;
    const uint BLOCK_SIZE_K = 32;
    const uint BLOCK_SIZE_N = 64;
    const uint THREAD_SIZE_M = 4;
    const uint THREAD_SIZE_N = 4;

    const uint bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const uint bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const uint THREAD_NUM_PER_BLOCK = bszm * bszn;

    uint BLOCK_IDX = gid.x;
    uint BLOCK_IDY = gid.y;

    const uint thread_id = tid.y * bszm + tid.x;

    threadgroup float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    threadgroup float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    int accum_idx[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    for (uint i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = -INFINITY;
        accum_idx[i] = 0;
    }

    const uint A_TILE_COL = thread_id / BLOCK_SIZE_M;
    const uint A_TILE_ROW = thread_id % BLOCK_SIZE_M;
    const uint B_TILE_COL = thread_id / BLOCK_SIZE_K;
    const uint B_TILE_ROW = thread_id % BLOCK_SIZE_K;
    const uint A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const uint B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (uint tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        for (uint i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            uint row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            uint col = A_TILE_COL + i + tile_idx;
            float val = -INFINITY;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        for (uint i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            uint row = tile_idx + B_TILE_ROW;
            uint col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            float val = -INFINITY;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BLOCK_SIZE_K; ++k) {
            int global_k = tile_idx + k;
            for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(tid.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }
            for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, tid.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }
            for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
                for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    float prod = regs_a[tm] + regs_b[tn];
                    uint idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    if (prod > accum[idx]) {
                        accum[idx] = prod;
                        accum_idx[idx] = global_k;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
        for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
            uint row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * tid.x + tm;
            uint col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * tid.y + tn;
            if (row < M && col < N) {
                uint out_idx = OFFSET_COL(row, col, M);
                uint local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                C[out_idx] = accum[local_idx];
                argmax_out[out_idx] = accum_idx[local_idx];
            }
        }
    }
}

// ============================================================================
// F32 MINPLUS GEMM WITH ARGMAX KERNEL
// ============================================================================

kernel void tropical_minplus_f32_nn_with_argmax(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device int* argmax_out [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    const uint BLOCK_SIZE_M = 64;
    const uint BLOCK_SIZE_K = 32;
    const uint BLOCK_SIZE_N = 64;
    const uint THREAD_SIZE_M = 4;
    const uint THREAD_SIZE_N = 4;

    const uint bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const uint bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const uint THREAD_NUM_PER_BLOCK = bszm * bszn;

    uint BLOCK_IDX = gid.x;
    uint BLOCK_IDY = gid.y;

    const uint thread_id = tid.y * bszm + tid.x;

    threadgroup float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    threadgroup float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    int accum_idx[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    for (uint i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = INFINITY;
        accum_idx[i] = 0;
    }

    const uint A_TILE_COL = thread_id / BLOCK_SIZE_M;
    const uint A_TILE_ROW = thread_id % BLOCK_SIZE_M;
    const uint B_TILE_COL = thread_id / BLOCK_SIZE_K;
    const uint B_TILE_ROW = thread_id % BLOCK_SIZE_K;
    const uint A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const uint B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (uint tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        for (uint i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            uint row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            uint col = A_TILE_COL + i + tile_idx;
            float val = INFINITY;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        for (uint i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            uint row = tile_idx + B_TILE_ROW;
            uint col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            float val = INFINITY;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BLOCK_SIZE_K; ++k) {
            int global_k = tile_idx + k;
            for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(tid.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }
            for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, tid.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }
            for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
                for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    float prod = regs_a[tm] + regs_b[tn];
                    uint idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    if (prod < accum[idx]) {
                        accum[idx] = prod;
                        accum_idx[idx] = global_k;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
        for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
            uint row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * tid.x + tm;
            uint col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * tid.y + tn;
            if (row < M && col < N) {
                uint out_idx = OFFSET_COL(row, col, M);
                uint local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                C[out_idx] = accum[local_idx];
                argmax_out[out_idx] = accum_idx[local_idx];
            }
        }
    }
}

// ============================================================================
// F32 MAXMUL GEMM WITH ARGMAX KERNEL
// ============================================================================

kernel void tropical_maxmul_f32_nn_with_argmax(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device int* argmax_out [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    const uint BLOCK_SIZE_M = 64;
    const uint BLOCK_SIZE_K = 32;
    const uint BLOCK_SIZE_N = 64;
    const uint THREAD_SIZE_M = 4;
    const uint THREAD_SIZE_N = 4;

    const uint bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const uint bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const uint THREAD_NUM_PER_BLOCK = bszm * bszn;

    uint BLOCK_IDX = gid.x;
    uint BLOCK_IDY = gid.y;

    const uint thread_id = tid.y * bszm + tid.x;

    threadgroup float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    threadgroup float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    int accum_idx[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    for (uint i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = 0.0f;
        accum_idx[i] = 0;
    }

    const uint A_TILE_COL = thread_id / BLOCK_SIZE_M;
    const uint A_TILE_ROW = thread_id % BLOCK_SIZE_M;
    const uint B_TILE_COL = thread_id / BLOCK_SIZE_K;
    const uint B_TILE_ROW = thread_id % BLOCK_SIZE_K;
    const uint A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const uint B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (uint tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        for (uint i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            uint row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            uint col = A_TILE_COL + i + tile_idx;
            float val = 0.0f;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        for (uint i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            uint row = tile_idx + B_TILE_ROW;
            uint col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            float val = 0.0f;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BLOCK_SIZE_K; ++k) {
            int global_k = tile_idx + k;
            for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(tid.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }
            for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, tid.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }
            for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
                for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    float prod = regs_a[tm] * regs_b[tn];
                    uint idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    if (prod > accum[idx]) {
                        accum[idx] = prod;
                        accum_idx[idx] = global_k;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint tm = 0; tm < THREAD_SIZE_M; ++tm) {
        for (uint tn = 0; tn < THREAD_SIZE_N; ++tn) {
            uint row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * tid.x + tm;
            uint col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * tid.y + tn;
            if (row < M && col < N) {
                uint out_idx = OFFSET_COL(row, col, M);
                uint local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                C[out_idx] = accum[local_idx];
                argmax_out[out_idx] = accum_idx[local_idx];
            }
        }
    }
}

// ============================================================================
// BACKWARD PASS KERNELS
// ============================================================================

kernel void tropical_backward_a_f32(
    device const float* grad_c [[buffer(0)]],
    device const int* argmax_in [[buffer(1)]],
    device atomic_float* grad_a [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = M * N;
    if (idx < total) {
        uint i = idx % M;
        int k = argmax_in[idx];
        if (k >= 0 && (uint)k < K) {
            atomic_fetch_add_explicit(&grad_a[i + k * M], grad_c[idx], memory_order_relaxed);
        }
    }
}

kernel void tropical_backward_b_f32(
    device const float* grad_c [[buffer(0)]],
    device const int* argmax_in [[buffer(1)]],
    device atomic_float* grad_b [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = M * N;
    if (idx < total) {
        uint j = idx / M;
        int k = argmax_in[idx];
        if (k >= 0 && (uint)k < K) {
            atomic_fetch_add_explicit(&grad_b[k + j * K], grad_c[idx], memory_order_relaxed);
        }
    }
}
