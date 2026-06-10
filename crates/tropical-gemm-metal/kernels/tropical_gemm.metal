// Tropical GEMM Metal kernels.
// Ported from crates/tropical-gemm-cuda/kernels/tropical_gemm.cu — keep the two
// files section-aligned when changing either. Differences from the CUDA side:
//  * one type-parameterized macro instead of per-type copies (MSL macro takes
//    SCALAR and block sizes as parameters);
//  * no f64 kernels (Metal has no double);
//  * 2D threadgroup position is reconstructed from a 1D threadgroup index the
//    same way the CUDA side folds blockIdx.x.
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// CONSTANTS AND UTILITIES (mirrors .cu lines 9-105)
// ============================================================================

// Integer "infinity" sentinels — identical values and data-range contract as
// the CUDA side (see the long comment in tropical_gemm.cu): keep |data| < |S|/4.
#define INF_I32 (1000000000)
#define NEG_INF_I32 (-1000000000)
// MSL `long` is always 64-bit (MSL spec table of scalar types), so `1L`
// here equals the CUDA side's `1LL`.
#define INF_I64 (1L << 60)
#define NEG_INF_I64 (-(1L << 60))

#define OFFSET_COL(row, col, ld) ((col) * (ld) + (row))

// Semiring ops, passed to the kernel macro as CMP_FN / MUL_FN.
// max/min resolve via <metal_stdlib> overloads for float/int/long/uint/ulong.
#define TADD(a, b) ((a) + (b))
#define TMUL(a, b) ((a) * (b))
#define TOR(a, b)  ((a) | (b))
#define TAND(a, b) ((a) & (b))
#define TMAX(a, b) (max((a), (b)))
#define TMIN(a, b) (min((a), (b)))

// Drifted tropical-zero detection for integer argmax canonicalization
// (write-out only; mirrors .cu zero_* helpers).
#define ZERO_MAXPLUS_I32(v) ((v) <= NEG_INF_I32 / 2)
#define ZERO_MINPLUS_I32(v) ((v) >= INF_I32 / 2)
#define ZERO_MAXPLUS_I64(v) ((v) <= NEG_INF_I64 / 2)
#define ZERO_MINPLUS_I64(v) ((v) >= INF_I64 / 2)
#define ZERO_NEVER(v) (false)

struct GemmParams {
    int M;
    int N;
    int K;
};

// ============================================================================
// GEMM KERNEL MACRO (mirrors .cu TROPICAL_GEMM_F32, lines 220-300)
// ============================================================================
// Column-major A (M×K, ld=M), B (K×N, ld=K), C (M×N, ld=M).
// Grid: 1D, DIM_GRID_X*DIM_GRID_Y threadgroups; threadgroup (BM/4)×(BN/4)=16×16.
// Interior threadgroups load tiles unguarded; only edge blocks / the final
// K-tile take the bounds-checked path (issue #40 structure).

#define TROPICAL_GEMM(SCALAR, KERNEL_NAME, INIT_VAL, CMP_FN, MUL_FN, BM, BK, BN) \
kernel void KERNEL_NAME(                                                       \
    device const SCALAR* A [[buffer(0)]],                                      \
    device const SCALAR* B [[buffer(1)]],                                      \
    device SCALAR* C [[buffer(2)]],                                            \
    constant GemmParams& p [[buffer(3)]],                                      \
    uint2 tgp [[threadgroup_position_in_grid]],                                \
    uint2 lid [[thread_position_in_threadgroup]])                              \
{                                                                              \
    const uint tg = tgp.x;                                                     \
    const int M = p.M, N = p.N, K = p.K;                                       \
    const int TM = 4, TN = 4;                                                  \
    const int bszm = (BM) / TM;                                                \
    const int bszn = (BN) / TN;                                                \
    const int THREADS = bszm * bszn;                                           \
                                                                               \
    const int DIM_GRID_X = (M + (BM) - 1) / (BM);                              \
    const int DIM_GRID_Y = (N + (BN) - 1) / (BN);                              \
    const int BLOCK_IDX = (int)tg % DIM_GRID_X;                                \
    const int BLOCK_IDY = (int)tg / DIM_GRID_X;                                \
    const int tx = (int)lid.x, ty = (int)lid.y;                                \
    const int tid = ty * bszm + tx;                                            \
                                                                               \
    threadgroup SCALAR As[(BM) * (BK)];                                        \
    threadgroup SCALAR Bs[(BK) * (BN)];                                        \
                                                                               \
    SCALAR accum[TM * TN];                                                     \
    SCALAR regs_a[TM];                                                         \
    SCALAR regs_b[TN];                                                         \
    for (int i = 0; i < TM * TN; ++i) { accum[i] = (INIT_VAL); }               \
                                                                               \
    const int A_TILE_COL = tid / (BM);                                         \
    const int A_TILE_ROW = tid % (BM);                                         \
    const int B_TILE_COL = tid / (BK);                                         \
    const int B_TILE_ROW = tid % (BK);                                         \
    const int A_COL_STRIDE = THREADS / (BM);                                   \
    const int B_COL_STRIDE = THREADS / (BK);                                   \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += (BK)) {                   \
        for (int i = 0; i < (BK); i += A_COL_STRIDE) {                         \
            int row = (BM) * BLOCK_IDX + A_TILE_ROW;                           \
            int col = A_TILE_COL + i + tile_idx;                               \
            int dst = OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, (BM));            \
            if (BLOCK_IDX == DIM_GRID_X - 1 || tile_idx >= K - (BK))           \
                As[dst] = (row < M && col < K) ? A[OFFSET_COL(row, col, M)] : (INIT_VAL); \
            else                                                               \
                As[dst] = A[OFFSET_COL(row, col, M)];                          \
        }                                                                      \
        for (int i = 0; i < (BN); i += B_COL_STRIDE) {                         \
            int row = tile_idx + B_TILE_ROW;                                   \
            int col = (BN) * BLOCK_IDY + i + B_TILE_COL;                       \
            int dst = OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, (BK));            \
            if (tile_idx >= K - (BK) || BLOCK_IDY == DIM_GRID_Y - 1)           \
                Bs[dst] = (row < K && col < N) ? B[OFFSET_COL(row, col, K)] : (INIT_VAL); \
            else                                                               \
                Bs[dst] = B[OFFSET_COL(row, col, K)];                          \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
                                                                               \
        for (int k = 0; k < (BK); ++k) {                                       \
            for (int tm = 0; tm < TM; ++tm)                                    \
                regs_a[tm] = As[OFFSET_COL(tx * TM + tm, k, (BM))];            \
            for (int tn = 0; tn < TN; ++tn)                                    \
                regs_b[tn] = Bs[OFFSET_COL(k, ty * TN + tn, (BK))];            \
            for (int tm = 0; tm < TM; ++tm) {                                  \
                for (int tn = 0; tn < TN; ++tn) {                              \
                    SCALAR prod = MUL_FN(regs_a[tm], regs_b[tn]);              \
                    int idx = OFFSET_COL(tm, tn, TM);                          \
                    accum[idx] = CMP_FN(accum[idx], prod);                     \
                }                                                              \
            }                                                                  \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
                                                                               \
    const bool edge = (BLOCK_IDX == DIM_GRID_X - 1 || BLOCK_IDY == DIM_GRID_Y - 1); \
    for (int tm = 0; tm < TM; ++tm) {                                          \
        for (int tn = 0; tn < TN; ++tn) {                                      \
            int row = (BM) * BLOCK_IDX + TM * tx + tm;                         \
            int col = (BN) * BLOCK_IDY + TN * ty + tn;                         \
            if (!edge || (row < M && col < N))                                 \
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, TM)];    \
        }                                                                      \
    }                                                                          \
}

// ============================================================================
// INSTANTIATIONS — keep names identical to the CUDA side
// ============================================================================

// f32: block 64x32x64, thread 4x4 (mirrors .cu lines 1205-1207)
TROPICAL_GEMM(float, tropical_maxplus_f32_nn, -INFINITY, TMAX, TADD, 64, 32, 64)
TROPICAL_GEMM(float, tropical_minplus_f32_nn,  INFINITY, TMIN, TADD, 64, 32, 64)
TROPICAL_GEMM(float, tropical_maxmul_f32_nn,       0.0f, TMAX, TMUL, 64, 32, 64)

// i32: block 64x32x64 (mirrors .cu lines 1225-1227)
TROPICAL_GEMM(int, tropical_maxplus_i32_nn, NEG_INF_I32, TMAX, TADD, 64, 32, 64)
TROPICAL_GEMM(int, tropical_minplus_i32_nn,     INF_I32, TMIN, TADD, 64, 32, 64)
TROPICAL_GEMM(int, tropical_maxmul_i32_nn,            0, TMAX, TMUL, 64, 32, 64)

// i64: block 32x16x32 — the 8-byte tier (mirrors .cu f64 tier, lines 1230-1232)
TROPICAL_GEMM(long, tropical_maxplus_i64_nn, NEG_INF_I64, TMAX, TADD, 32, 16, 32)
TROPICAL_GEMM(long, tropical_minplus_i64_nn,     INF_I64, TMIN, TADD, 32, 16, 32)
TROPICAL_GEMM(long, tropical_maxmul_i64_nn,           0L, TMAX, TMUL, 32, 16, 32)

// bool (AndOr): add = OR, mul = AND; false is a true absorbing zero, PAD = 0.
// uchar + bitwise ops, NOT logical && / || — keeps values in byte form instead
// of predicate round-trips (the 1.6x regression noted on the CUDA side).
TROPICAL_GEMM(uchar, tropical_andor_bool_nn, (uchar)0, TOR, TAND, 64, 32, 64)

// ============================================================================
// GEMM WITH ARGMAX (mirrors .cu TROPICAL_GEMM_ARGMAX_*, lines 808-1160)
// ============================================================================
// COMPARE_OP is the *strict* comparison (>, <): updates only on strictly
// better, so ties keep the smallest k. ZERO_FN canonicalizes drifted integer
// tropical-zero cells' argmax to 0 at write-out (ZERO_NEVER for floats).

#define TROPICAL_GEMM_ARGMAX(SCALAR, KERNEL_NAME, INIT_VAL, COMPARE_OP, MUL_FN, ZERO_FN, BM, BK, BN) \
kernel void KERNEL_NAME(                                                       \
    device const SCALAR* A [[buffer(0)]],                                      \
    device const SCALAR* B [[buffer(1)]],                                      \
    device SCALAR* C [[buffer(2)]],                                            \
    device uint* argmax_out [[buffer(3)]],                                     \
    constant GemmParams& p [[buffer(4)]],                                      \
    uint2 tgp [[threadgroup_position_in_grid]],                                \
    uint2 lid [[thread_position_in_threadgroup]])                              \
{                                                                              \
    const uint tg = tgp.x;                                                     \
    const int M = p.M, N = p.N, K = p.K;                                       \
    const int TM = 4, TN = 4;                                                  \
    const int bszm = (BM) / TM;                                                \
    const int bszn = (BN) / TN;                                                \
    const int THREADS = bszm * bszn;                                           \
                                                                               \
    const int DIM_GRID_X = (M + (BM) - 1) / (BM);                              \
    const int DIM_GRID_Y = (N + (BN) - 1) / (BN);                              \
    const int BLOCK_IDX = (int)tg % DIM_GRID_X;                                \
    const int BLOCK_IDY = (int)tg / DIM_GRID_X;                                \
    const int tx = (int)lid.x, ty = (int)lid.y;                                \
    const int tid = ty * bszm + tx;                                            \
                                                                               \
    threadgroup SCALAR As[(BM) * (BK)];                                        \
    threadgroup SCALAR Bs[(BK) * (BN)];                                        \
                                                                               \
    SCALAR accum[TM * TN];                                                     \
    uint accum_idx[TM * TN];                                                   \
    SCALAR regs_a[TM];                                                         \
    SCALAR regs_b[TN];                                                         \
    for (int i = 0; i < TM * TN; ++i) { accum[i] = (INIT_VAL); accum_idx[i] = 0u; } \
                                                                               \
    const int A_TILE_COL = tid / (BM);                                         \
    const int A_TILE_ROW = tid % (BM);                                         \
    const int B_TILE_COL = tid / (BK);                                         \
    const int B_TILE_ROW = tid % (BK);                                         \
    const int A_COL_STRIDE = THREADS / (BM);                                   \
    const int B_COL_STRIDE = THREADS / (BK);                                   \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += (BK)) {                   \
        for (int i = 0; i < (BK); i += A_COL_STRIDE) {                         \
            int row = (BM) * BLOCK_IDX + A_TILE_ROW;                           \
            int col = A_TILE_COL + i + tile_idx;                               \
            int dst = OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, (BM));            \
            if (BLOCK_IDX == DIM_GRID_X - 1 || tile_idx >= K - (BK))           \
                As[dst] = (row < M && col < K) ? A[OFFSET_COL(row, col, M)] : (INIT_VAL); \
            else                                                               \
                As[dst] = A[OFFSET_COL(row, col, M)];                          \
        }                                                                      \
        for (int i = 0; i < (BN); i += B_COL_STRIDE) {                         \
            int row = tile_idx + B_TILE_ROW;                                   \
            int col = (BN) * BLOCK_IDY + i + B_TILE_COL;                       \
            int dst = OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, (BK));            \
            if (tile_idx >= K - (BK) || BLOCK_IDY == DIM_GRID_Y - 1)           \
                Bs[dst] = (row < K && col < N) ? B[OFFSET_COL(row, col, K)] : (INIT_VAL); \
            else                                                               \
                Bs[dst] = B[OFFSET_COL(row, col, K)];                          \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
                                                                               \
        for (int k = 0; k < (BK); ++k) {                                       \
            uint global_k = (uint)(tile_idx + k);                              \
            for (int tm = 0; tm < TM; ++tm)                                    \
                regs_a[tm] = As[OFFSET_COL(tx * TM + tm, k, (BM))];            \
            for (int tn = 0; tn < TN; ++tn)                                    \
                regs_b[tn] = Bs[OFFSET_COL(k, ty * TN + tn, (BK))];            \
            for (int tm = 0; tm < TM; ++tm) {                                  \
                for (int tn = 0; tn < TN; ++tn) {                              \
                    SCALAR prod = MUL_FN(regs_a[tm], regs_b[tn]);              \
                    int idx = OFFSET_COL(tm, tn, TM);                          \
                    if (prod COMPARE_OP accum[idx]) {                          \
                        accum[idx] = prod;                                     \
                        accum_idx[idx] = global_k;                             \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
                                                                               \
    const bool edge = (BLOCK_IDX == DIM_GRID_X - 1 || BLOCK_IDY == DIM_GRID_Y - 1); \
    for (int tm = 0; tm < TM; ++tm) {                                          \
        for (int tn = 0; tn < TN; ++tn) {                                      \
            int row = (BM) * BLOCK_IDX + TM * tx + tm;                         \
            int col = (BN) * BLOCK_IDY + TN * ty + tn;                         \
            if (!edge || (row < M && col < N)) {                               \
                int out_idx = OFFSET_COL(row, col, M);                         \
                int li = OFFSET_COL(tm, tn, TM);                               \
                C[out_idx] = accum[li];                                        \
                argmax_out[out_idx] = ZERO_FN(accum[li]) ? 0u : accum_idx[li]; \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// f32 argmax (strict > / <; ZERO_NEVER)
TROPICAL_GEMM_ARGMAX(float, tropical_maxplus_f32_nn_with_argmax, -INFINITY, >, TADD, ZERO_NEVER, 64, 32, 64)
TROPICAL_GEMM_ARGMAX(float, tropical_minplus_f32_nn_with_argmax,  INFINITY, <, TADD, ZERO_NEVER, 64, 32, 64)
TROPICAL_GEMM_ARGMAX(float, tropical_maxmul_f32_nn_with_argmax,       0.0f, >, TMUL, ZERO_NEVER, 64, 32, 64)

// i32 argmax (drifted-zero canonicalization; maxmul's 0 never drifts)
TROPICAL_GEMM_ARGMAX(int, tropical_maxplus_i32_nn_with_argmax, NEG_INF_I32, >, TADD, ZERO_MAXPLUS_I32, 64, 32, 64)
TROPICAL_GEMM_ARGMAX(int, tropical_minplus_i32_nn_with_argmax,     INF_I32, <, TADD, ZERO_MINPLUS_I32, 64, 32, 64)
TROPICAL_GEMM_ARGMAX(int, tropical_maxmul_i32_nn_with_argmax,            0, >, TMUL, ZERO_NEVER,       64, 32, 64)

// i64 argmax (8-byte tier)
TROPICAL_GEMM_ARGMAX(long, tropical_maxplus_i64_nn_with_argmax, NEG_INF_I64, >, TADD, ZERO_MAXPLUS_I64, 32, 16, 32)
TROPICAL_GEMM_ARGMAX(long, tropical_minplus_i64_nn_with_argmax,     INF_I64, <, TADD, ZERO_MINPLUS_I64, 32, 16, 32)
TROPICAL_GEMM_ARGMAX(long, tropical_maxmul_i64_nn_with_argmax,           0L, >, TMUL, ZERO_NEVER,       32, 16, 32)
