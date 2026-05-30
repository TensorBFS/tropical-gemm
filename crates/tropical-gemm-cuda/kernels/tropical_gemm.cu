// Tropical GEMM CUDA Kernels
// DRY implementation using C preprocessor macros
// Adapted from CuTropicalGEMM.jl

// ============================================================================
// CONSTANTS AND UTILITIES
// ============================================================================

// Infinity constants (NVRTC doesn't have access to standard headers)
#define INF_F32 __int_as_float(0x7f800000)
#define NEG_INF_F32 __int_as_float(0xff800000)
#define INF_F64 __longlong_as_double(0x7ff0000000000000LL)
#define NEG_INF_F64 __longlong_as_double(0xfff0000000000000LL)

// Integer "infinity" constants (sentinel values for the tropical zero).
// Integers can't hold a true -inf/+inf, so we stand it in with a large-magnitude
// value S. Multiply is a bare add (no per-element guard, no output clamp): the
// tropical zero is just this big number, and `S + data` stays deep in sentinel
// territory, so it keeps absorbing in every max/min. The literal value drifts
// (you get `S + data`, not exactly `S`) — that is harmless; detect a tropical-zero
// result with a threshold (`<= NEG_INF_I32/2` / `>= INF_I32/2`), not `== S`.
//
// SAFE DATA RANGE (measured, see examples/int_range_limits.rs):
//   keep |data| < |S|/4  ->  i32: < 2.5e8,  i64: < 2^58 (~2.9e17).
// In this range every result is correct AND a tropical-zero output is reliably
// detected by the threshold test (value <= NEG_INF_I32/2, etc.). Two softer
// limits lie just beyond:
//   * |S|/4 <= |data| < |S|/3 : values are still correct, but a finite result can
//     dip past the S/2 threshold and be misread as the tropical zero.
//   * |data| >= |S|/3 : a real path (weight up to 2*|data|) can be masked by a
//     "no-edge + data" term (= S + data) and silently lost -> WRONG result.
// Overflow is never the issue: the largest intermediate is 2*S, and 2e9 < INT_MAX
// / 2^61 < INT64_MAX regardless of |data|. If your data exceeds the i32 bound,
// switch to i64 (~2.9e17 of headroom).
//
// These values are mirrored as SENTINEL_I32 / SENTINEL_I64 (and MAX_RELIABLE_DATA_*)
// in the Rust crate root (src/lib.rs) — keep the two in sync.
#define INF_I32 (1000000000)
#define NEG_INF_I32 (-1000000000)
#define INF_I64 (1LL << 60)
#define NEG_INF_I64 (-(1LL << 60))

// Memory layout helpers
#define OFFSET_COL(row, col, ld) ((col) * (ld) + (row))

// Integer max/min functions
__device__ __forceinline__ int max_i32(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int min_i32(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ long long max_i64(long long a, long long b) { return a > b ? a : b; }
__device__ __forceinline__ long long min_i64(long long a, long long b) { return a < b ? a : b; }

// Tropical multiply for integer MaxPlus/MinPlus is a bare add: no per-element
// guard, no output clamp. The tropical zero is the large sentinel above, so
// `S + data` stays in sentinel territory and keeps absorbing in every max/min
// (its literal value drifts to `S + data`, which is harmless — see the data-range
// note above). This is ~2.4x faster than the old per-multiply saturating guard.
__device__ __forceinline__ int add_i32(int a, int b) { return a + b; }
__device__ __forceinline__ long long add_i64(long long a, long long b) { return a + b; }

// Simple multiplication for MaxMul integer types
// For MaxMul: zero = 0, one = 1, add = max, mul = *
// Note: 0 * anything = 0 (correct absorbing element behavior)
__device__ __forceinline__ int mul_i32(int a, int b) {
    return a * b;
}
__device__ __forceinline__ long long mul_i64(long long a, long long b) {
    return a * b;
}

// Boolean (AndOr) semiring: add = OR, mul = AND, zero = false, one = true.
// false is a true absorbing zero (false AND x = false), so no sentinel/drift like
// the integer MaxPlus types -- the out-of-range tile PAD is simply false.
//
// Use *bitwise* &/| (not logical &&/||): for 0/1 bytes they are equivalent, but
// bitwise ops let ptxas keep the values in byte/integer form (LOP3) instead of
// round-tripping each byte through a predicate (ISETP -> PLOP3 -> P2R), which
// measured ~1.6x slower than the reference's fused byte-wise path on sm_86.
__device__ __forceinline__ bool or_bool(bool a, bool b)  { return a | b; }
__device__ __forceinline__ bool and_bool(bool a, bool b) { return a & b; }

// Bit-packed boolean (Bitwise) semiring: add = OR, mul = AND, zero = 0, one = ~0.
// Each bit-lane is an independent boolean problem. zero = 0 is a true absorbing
// zero for AND (0 & x = 0), so the out-of-range tile PAD is simply 0.
__device__ __forceinline__ unsigned int or_u32(unsigned int a, unsigned int b)  { return a | b; }
__device__ __forceinline__ unsigned int and_u32(unsigned int a, unsigned int b) { return a & b; }
__device__ __forceinline__ unsigned long long or_u64(unsigned long long a, unsigned long long b)  { return a | b; }
__device__ __forceinline__ unsigned long long and_u64(unsigned long long a, unsigned long long b) { return a & b; }

// Drifted tropical-zero detection for argmax canonicalization. A no-contribution
// output cell's value sits in "infinity territory" (past S/2) after the
// guard-free add drifts it (`S + data`). Used ONLY at the O(M*N) write-out (not
// in the inner loop) to reset such a cell's argmax to the seed 0, so the result
// is a deterministic, repo-wide value instead of a data-dependent k.
// maxmul's 0 is a true absorbing zero (0 * x = 0, never drifts) -> zero_never.
__device__ __forceinline__ bool zero_maxplus_i32(int v)        { return v <= NEG_INF_I32 / 2; }
__device__ __forceinline__ bool zero_minplus_i32(int v)        { return v >= INF_I32 / 2; }
__device__ __forceinline__ bool zero_never_i32(int v)          { (void)v; return false; }
__device__ __forceinline__ bool zero_maxplus_i64(long long v)  { return v <= NEG_INF_I64 / 2; }
__device__ __forceinline__ bool zero_minplus_i64(long long v)  { return v >= INF_I64 / 2; }
__device__ __forceinline__ bool zero_never_i64(long long v)    { (void)v; return false; }

// atomicAdd for double (not supported on all architectures)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ double atomicAddDouble(double* address, double val) {
    return atomicAdd(address, val);
}
#endif

// ============================================================================
// BOUNDARY-AWARE TILE I/O HELPERS
// ============================================================================
// The global<->shared tile copies are identical across every GEMM kernel, so
// they live here once. Interior blocks (not the last block-row/col, not the
// final K-tile) load and store WITHOUT a per-element bounds check; only edge
// blocks take the guarded path. That predicate -- which the old code ran on
// every element of every block -- was the entire ~5% slowdown vs the C
// reference on large matrices (issue #40); removing it for interior blocks
// closes and reverses the gap. Mirrors the reference NN kernel
// (TropicalGemm_Cuda/tropicalgemm_kernels.cu:625,638,683).
//
// These expand inside a kernel body and use its locals: As/Bs, BLOCK_IDX/IDY,
// DIM_GRID_X/Y, tile_idx, the tile-size constants, M/N/K, accum[/_idx].
// PAD is the tropical zero used to fill out-of-range cells.

// A tile -> As. A is column-major with leading dimension M.
#define LOAD_A_TILE(SRC, PAD)                                                  \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {                \
        int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;                       \
        int col = A_TILE_COL + i + tile_idx;                                   \
        int dst = OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M);        \
        if (BLOCK_IDX == DIM_GRID_X - 1 || tile_idx >= K - BLOCK_SIZE_K)       \
            As[dst] = (row < M && col < K) ? SRC[OFFSET_COL(row, col, M)] : (PAD); \
        else                                                                  \
            As[dst] = SRC[OFFSET_COL(row, col, M)];                           \
    }

// B tile -> Bs. B is column-major with leading dimension K.
#define LOAD_B_TILE(SRC, PAD)                                                  \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {                \
        int row = tile_idx + B_TILE_ROW;                                       \
        int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;                   \
        int dst = OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K);        \
        if (tile_idx >= K - BLOCK_SIZE_K || BLOCK_IDY == DIM_GRID_Y - 1)       \
            Bs[dst] = (row < K && col < N) ? SRC[OFFSET_COL(row, col, K)] : (PAD); \
        else                                                                  \
            Bs[dst] = SRC[OFFSET_COL(row, col, K)];                           \
    }

// Edge block iff last block-row or last block-col; interior blocks are fully in
// range so they store unconditionally.
#define TILE_IS_EDGE (BLOCK_IDX == DIM_GRID_X - 1 || BLOCK_IDY == DIM_GRID_Y - 1)

// C tile store (value only).
#define STORE_C_TILE(DST)                                                      \
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn; \
            if (!TILE_IS_EDGE || (row < M && col < N))                         \
                DST[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)]; \
        }                                                                      \
    }

// C + argmax store (no tropical-zero canonicalization; float/double argmax).
#define STORE_C_ARGMAX_TILE(DST, ARGMAX_DST)                                   \
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn; \
            if (!TILE_IS_EDGE || (row < M && col < N)) {                       \
                int out_idx = OFFSET_COL(row, col, M);                         \
                int local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);             \
                DST[out_idx] = accum[local_idx];                               \
                ARGMAX_DST[out_idx] = accum_idx[local_idx];                    \
            }                                                                  \
        }                                                                      \
    }

// C + argmax store with tropical-zero canonicalization (integer argmax): a
// drifted-zero cell's argmax is reset to the seed 0 via ZERO_FN.
#define STORE_C_ARGMAX_ZERO_TILE(DST, ARGMAX_DST, ZERO_FN)                     \
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn; \
            if (!TILE_IS_EDGE || (row < M && col < N)) {                       \
                int out_idx = OFFSET_COL(row, col, M);                         \
                int local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);             \
                DST[out_idx] = accum[local_idx];                               \
                ARGMAX_DST[out_idx] = (ZERO_FN)(accum[local_idx]) ? 0 : accum_idx[local_idx]; \
            }                                                                  \
        }                                                                      \
    }

// ============================================================================
// F32 GEMM KERNEL MACRO
// ============================================================================
// Block sizes for f32: 64x32x64, Thread sizes: 4x4
// Generates: tropical_{semiring}_f32_nn

#define TROPICAL_GEMM_F32(KERNEL_NAME, INIT_VAL, COMPARE_FN, MUL_OP)           \
extern "C" __global__ void KERNEL_NAME(                                        \
    const float* __restrict__ A,                                               \
    const float* __restrict__ B,                                               \
    float* __restrict__ C,                                                     \
    int M, int N, int K                                                        \
) {                                                                            \
    const int BLOCK_SIZE_M = 64;                                               \
    const int BLOCK_SIZE_K = 32;                                               \
    const int BLOCK_SIZE_N = 64;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;                    \
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;                    \
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];                          \
    __shared__ float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];                          \
                                                                               \
    float accum[THREAD_SIZE_M * THREAD_SIZE_N];                                \
    float regs_a[THREAD_SIZE_M];                                               \
    float regs_b[THREAD_SIZE_N];                                               \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        accum[i] = INIT_VAL;                                                   \
    }                                                                          \
                                                                               \
    const int A_TILE_COL = tid / BLOCK_SIZE_M;                                 \
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;                                 \
    const int B_TILE_COL = tid / BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;                                 \
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;         \
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        LOAD_A_TILE(A, INIT_VAL)                                          \
                                                                               \
        LOAD_B_TILE(B, INIT_VAL)                                          \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm,   \
                                           k, BLOCK_SIZE_M)];                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn,\
                                           BLOCK_SIZE_K)];                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    float prod = regs_a[tm] MUL_OP regs_b[tn];                 \
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);               \
                    accum[idx] = COMPARE_FN(accum[idx], prod);                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
        STORE_C_TILE(C)                                                   \
}

// ============================================================================
// F64 GEMM KERNEL MACRO
// ============================================================================
// Block sizes for f64: 32x16x32, Thread sizes: 4x4

#define TROPICAL_GEMM_F64(KERNEL_NAME, INIT_VAL, COMPARE_FN, MUL_OP)           \
extern "C" __global__ void KERNEL_NAME(                                        \
    const double* __restrict__ A,                                              \
    const double* __restrict__ B,                                              \
    double* __restrict__ C,                                                    \
    int M, int N, int K                                                        \
) {                                                                            \
    const int BLOCK_SIZE_M = 32;                                               \
    const int BLOCK_SIZE_K = 16;                                               \
    const int BLOCK_SIZE_N = 32;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;                    \
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;                    \
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ double As[BLOCK_SIZE_M * BLOCK_SIZE_K];                         \
    __shared__ double Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];                         \
                                                                               \
    double accum[THREAD_SIZE_M * THREAD_SIZE_N];                               \
    double regs_a[THREAD_SIZE_M];                                              \
    double regs_b[THREAD_SIZE_N];                                              \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        accum[i] = INIT_VAL;                                                   \
    }                                                                          \
                                                                               \
    const int A_TILE_COL = tid / BLOCK_SIZE_M;                                 \
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;                                 \
    const int B_TILE_COL = tid / BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;                                 \
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;         \
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        LOAD_A_TILE(A, INIT_VAL)                                          \
                                                                               \
        LOAD_B_TILE(B, INIT_VAL)                                          \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm,   \
                                           k, BLOCK_SIZE_M)];                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn,\
                                           BLOCK_SIZE_K)];                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    double prod = regs_a[tm] MUL_OP regs_b[tn];                \
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);               \
                    accum[idx] = COMPARE_FN(accum[idx], prod);                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
        STORE_C_TILE(C)                                                   \
}

// ============================================================================
// I32 GEMM KERNEL MACRO
// ============================================================================
// Block sizes for i32: 64x32x64, Thread sizes: 4x4 (same as f32)
// Multiply is a bare add (MUL_FN); the tropical zero is a large sentinel.

#define TROPICAL_GEMM_I32(KERNEL_NAME, INIT_VAL, COMPARE_FN, MUL_FN)           \
extern "C" __global__ void KERNEL_NAME(                                        \
    const int* __restrict__ A,                                                 \
    const int* __restrict__ B,                                                 \
    int* __restrict__ C,                                                       \
    int M, int N, int K                                                        \
) {                                                                            \
    const int BLOCK_SIZE_M = 64;                                               \
    const int BLOCK_SIZE_K = 32;                                               \
    const int BLOCK_SIZE_N = 64;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;                    \
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;                    \
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ int As[BLOCK_SIZE_M * BLOCK_SIZE_K];                            \
    __shared__ int Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];                            \
                                                                               \
    int accum[THREAD_SIZE_M * THREAD_SIZE_N];                                  \
    int regs_a[THREAD_SIZE_M];                                                 \
    int regs_b[THREAD_SIZE_N];                                                 \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        accum[i] = INIT_VAL;                                                   \
    }                                                                          \
                                                                               \
    const int A_TILE_COL = tid / BLOCK_SIZE_M;                                 \
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;                                 \
    const int B_TILE_COL = tid / BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;                                 \
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;         \
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        LOAD_A_TILE(A, INIT_VAL)                                          \
                                                                               \
        LOAD_B_TILE(B, INIT_VAL)                                          \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm,   \
                                           k, BLOCK_SIZE_M)];                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn,\
                                           BLOCK_SIZE_K)];                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    int prod = MUL_FN(regs_a[tm], regs_b[tn]);                  \
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);               \
                    accum[idx] = COMPARE_FN(accum[idx], prod);                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
        STORE_C_TILE(C)                                                   \
}

// ============================================================================
// BOOL GEMM KERNEL MACRO (AndOr semiring)
// ============================================================================
// Identical tiling to I32 (64x32x64, 4x4 threads) but on a 1-byte `bool`
// element. add = OR (COMPARE_FN), mul = AND (MUL_FN), tropical zero = false.
// A 1-byte element uses *less* shared memory than i32, so the block config is
// safe. false absorbs in AND/OR, so the out-of-range PAD is just INIT_VAL.

#define TROPICAL_GEMM_BOOL(KERNEL_NAME, INIT_VAL, COMPARE_FN, MUL_FN)           \
extern "C" __global__ void KERNEL_NAME(                                        \
    const bool* __restrict__ A,                                                \
    const bool* __restrict__ B,                                                \
    bool* __restrict__ C,                                                      \
    int M, int N, int K                                                        \
) {                                                                            \
    const int BLOCK_SIZE_M = 64;                                               \
    const int BLOCK_SIZE_K = 32;                                               \
    const int BLOCK_SIZE_N = 64;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;                    \
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;                    \
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ bool As[BLOCK_SIZE_M * BLOCK_SIZE_K];                           \
    __shared__ bool Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];                           \
                                                                               \
    bool accum[THREAD_SIZE_M * THREAD_SIZE_N];                                 \
    bool regs_a[THREAD_SIZE_M];                                                \
    bool regs_b[THREAD_SIZE_N];                                                \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        accum[i] = INIT_VAL;                                                   \
    }                                                                          \
                                                                               \
    const int A_TILE_COL = tid / BLOCK_SIZE_M;                                 \
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;                                 \
    const int B_TILE_COL = tid / BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;                                 \
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;         \
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        LOAD_A_TILE(A, INIT_VAL)                                          \
                                                                               \
        LOAD_B_TILE(B, INIT_VAL)                                          \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm,   \
                                           k, BLOCK_SIZE_M)];                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn,\
                                           BLOCK_SIZE_K)];                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);               \
                    accum[idx] = COMPARE_FN(MUL_FN(regs_a[tm], regs_b[tn]),    \
                                            accum[idx]);                       \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
        STORE_C_TILE(C)                                                   \
}

// ============================================================================
// U32 GEMM KERNEL MACRO (Bitwise semiring, bit-packed boolean)
// ============================================================================
// Identical tiling to I32 (64x32x64, 4x4) on a 4-byte `unsigned int` element.
// add = OR (COMPARE_FN), mul = AND (MUL_FN), tropical zero = 0; PAD = 0 is the
// absorbing zero for AND. Being a 4-byte element, this shares the i32 kernel's
// shared-load path rather than the 1-byte bool kernel's per-byte loads — verify
// the win on HW with `cuobjdump -sass` (expect LDS, not LDS.U8).

#define TROPICAL_GEMM_U32(KERNEL_NAME, INIT_VAL, COMPARE_FN, MUL_FN)            \
extern "C" __global__ void KERNEL_NAME(                                        \
    const unsigned int* __restrict__ A,                                        \
    const unsigned int* __restrict__ B,                                        \
    unsigned int* __restrict__ C,                                              \
    int M, int N, int K                                                        \
) {                                                                            \
    const int BLOCK_SIZE_M = 64;                                               \
    const int BLOCK_SIZE_K = 32;                                               \
    const int BLOCK_SIZE_N = 64;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;                    \
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;                    \
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ unsigned int As[BLOCK_SIZE_M * BLOCK_SIZE_K];                   \
    __shared__ unsigned int Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];                   \
                                                                               \
    unsigned int accum[THREAD_SIZE_M * THREAD_SIZE_N];                         \
    unsigned int regs_a[THREAD_SIZE_M];                                        \
    unsigned int regs_b[THREAD_SIZE_N];                                        \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        accum[i] = INIT_VAL;                                                   \
    }                                                                          \
                                                                               \
    const int A_TILE_COL = tid / BLOCK_SIZE_M;                                 \
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;                                 \
    const int B_TILE_COL = tid / BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;                                 \
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;         \
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        LOAD_A_TILE(A, INIT_VAL)                                          \
                                                                               \
        LOAD_B_TILE(B, INIT_VAL)                                          \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm,   \
                                           k, BLOCK_SIZE_M)];                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn,\
                                           BLOCK_SIZE_K)];                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);               \
                    accum[idx] = COMPARE_FN(MUL_FN(regs_a[tm], regs_b[tn]),    \
                                            accum[idx]);                       \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
        STORE_C_TILE(C)                                                   \
}

// ============================================================================
// U64 GEMM KERNEL MACRO (Bitwise semiring, 64 lanes)
// ============================================================================
// Identical tiling to I64 (32x16x32, 4x4) on an 8-byte `unsigned long long`.

#define TROPICAL_GEMM_U64(KERNEL_NAME, INIT_VAL, COMPARE_FN, MUL_FN)            \
extern "C" __global__ void KERNEL_NAME(                                        \
    const unsigned long long* __restrict__ A,                                  \
    const unsigned long long* __restrict__ B,                                  \
    unsigned long long* __restrict__ C,                                        \
    int M, int N, int K                                                        \
) {                                                                            \
    const int BLOCK_SIZE_M = 32;                                               \
    const int BLOCK_SIZE_K = 16;                                               \
    const int BLOCK_SIZE_N = 32;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;                    \
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;                    \
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ unsigned long long As[BLOCK_SIZE_M * BLOCK_SIZE_K];             \
    __shared__ unsigned long long Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];             \
                                                                               \
    unsigned long long accum[THREAD_SIZE_M * THREAD_SIZE_N];                   \
    unsigned long long regs_a[THREAD_SIZE_M];                                  \
    unsigned long long regs_b[THREAD_SIZE_N];                                  \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        accum[i] = INIT_VAL;                                                   \
    }                                                                          \
                                                                               \
    const int A_TILE_COL = tid / BLOCK_SIZE_M;                                 \
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;                                 \
    const int B_TILE_COL = tid / BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;                                 \
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;         \
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        LOAD_A_TILE(A, INIT_VAL)                                          \
                                                                               \
        LOAD_B_TILE(B, INIT_VAL)                                          \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm,   \
                                           k, BLOCK_SIZE_M)];                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn,\
                                           BLOCK_SIZE_K)];                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);               \
                    accum[idx] = COMPARE_FN(MUL_FN(regs_a[tm], regs_b[tn]),    \
                                            accum[idx]);                       \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
        STORE_C_TILE(C)                                                   \
}

// ============================================================================
// I64 GEMM KERNEL MACRO
// ============================================================================
// Block sizes for i64: 32x16x32, Thread sizes: 4x4 (same as f64)

#define TROPICAL_GEMM_I64(KERNEL_NAME, INIT_VAL, COMPARE_FN, MUL_FN)           \
extern "C" __global__ void KERNEL_NAME(                                        \
    const long long* __restrict__ A,                                           \
    const long long* __restrict__ B,                                           \
    long long* __restrict__ C,                                                 \
    int M, int N, int K                                                        \
) {                                                                            \
    const int BLOCK_SIZE_M = 32;                                               \
    const int BLOCK_SIZE_K = 16;                                               \
    const int BLOCK_SIZE_N = 32;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;                    \
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;                    \
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ long long As[BLOCK_SIZE_M * BLOCK_SIZE_K];                      \
    __shared__ long long Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];                      \
                                                                               \
    long long accum[THREAD_SIZE_M * THREAD_SIZE_N];                            \
    long long regs_a[THREAD_SIZE_M];                                           \
    long long regs_b[THREAD_SIZE_N];                                           \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        accum[i] = INIT_VAL;                                                   \
    }                                                                          \
                                                                               \
    const int A_TILE_COL = tid / BLOCK_SIZE_M;                                 \
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;                                 \
    const int B_TILE_COL = tid / BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;                                 \
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;         \
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        LOAD_A_TILE(A, INIT_VAL)                                          \
                                                                               \
        LOAD_B_TILE(B, INIT_VAL)                                          \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm,   \
                                           k, BLOCK_SIZE_M)];                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn,\
                                           BLOCK_SIZE_K)];                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    long long prod = MUL_FN(regs_a[tm], regs_b[tn]);           \
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);               \
                    accum[idx] = COMPARE_FN(accum[idx], prod);                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
        STORE_C_TILE(C)                                                   \
}

// ============================================================================
// F32 GEMM WITH ARGMAX KERNEL MACRO
// ============================================================================

#define TROPICAL_GEMM_ARGMAX_F32(KERNEL_NAME, INIT_VAL, COMPARE_OP, MUL_OP)    \
extern "C" __global__ void KERNEL_NAME(                                        \
    const float* __restrict__ A,                                               \
    const float* __restrict__ B,                                               \
    float* __restrict__ C,                                                     \
    unsigned int* __restrict__ argmax,                                         \
    int M, int N, int K                                                        \
) {                                                                            \
    const int BLOCK_SIZE_M = 64;                                               \
    const int BLOCK_SIZE_K = 32;                                               \
    const int BLOCK_SIZE_N = 64;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;                    \
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;                    \
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];                          \
    __shared__ float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];                          \
                                                                               \
    float accum[THREAD_SIZE_M * THREAD_SIZE_N];                                \
    int accum_idx[THREAD_SIZE_M * THREAD_SIZE_N];                              \
    float regs_a[THREAD_SIZE_M];                                               \
    float regs_b[THREAD_SIZE_N];                                               \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        accum[i] = INIT_VAL;                                                   \
        accum_idx[i] = 0;                                                      \
    }                                                                          \
                                                                               \
    const int A_TILE_COL = tid / BLOCK_SIZE_M;                                 \
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;                                 \
    const int B_TILE_COL = tid / BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;                                 \
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;         \
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        LOAD_A_TILE(A, INIT_VAL)                                          \
                                                                               \
        LOAD_B_TILE(B, INIT_VAL)                                          \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            int global_k = tile_idx + k;                                       \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm,   \
                                           k, BLOCK_SIZE_M)];                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn,\
                                           BLOCK_SIZE_K)];                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    float prod = regs_a[tm] MUL_OP regs_b[tn];                 \
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);               \
                    if (prod COMPARE_OP accum[idx]) {                          \
                        accum[idx] = prod;                                     \
                        accum_idx[idx] = global_k;                             \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
        STORE_C_ARGMAX_TILE(C, argmax)                                          \
}

// ============================================================================
// F64 GEMM WITH ARGMAX KERNEL MACRO
// ============================================================================

#define TROPICAL_GEMM_ARGMAX_F64(KERNEL_NAME, INIT_VAL, COMPARE_OP, MUL_OP)    \
extern "C" __global__ void KERNEL_NAME(                                        \
    const double* __restrict__ A,                                              \
    const double* __restrict__ B,                                              \
    double* __restrict__ C,                                                    \
    unsigned int* __restrict__ argmax,                                         \
    int M, int N, int K                                                        \
) {                                                                            \
    const int BLOCK_SIZE_M = 32;                                               \
    const int BLOCK_SIZE_K = 16;                                               \
    const int BLOCK_SIZE_N = 32;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;                    \
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;                    \
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ double As[BLOCK_SIZE_M * BLOCK_SIZE_K];                         \
    __shared__ double Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];                         \
                                                                               \
    double accum[THREAD_SIZE_M * THREAD_SIZE_N];                               \
    int accum_idx[THREAD_SIZE_M * THREAD_SIZE_N];                              \
    double regs_a[THREAD_SIZE_M];                                              \
    double regs_b[THREAD_SIZE_N];                                              \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        accum[i] = INIT_VAL;                                                   \
        accum_idx[i] = 0;                                                      \
    }                                                                          \
                                                                               \
    const int A_TILE_COL = tid / BLOCK_SIZE_M;                                 \
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;                                 \
    const int B_TILE_COL = tid / BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;                                 \
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;         \
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        LOAD_A_TILE(A, INIT_VAL)                                          \
                                                                               \
        LOAD_B_TILE(B, INIT_VAL)                                          \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            int global_k = tile_idx + k;                                       \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm,   \
                                           k, BLOCK_SIZE_M)];                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn,\
                                           BLOCK_SIZE_K)];                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    double prod = regs_a[tm] MUL_OP regs_b[tn];                \
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);               \
                    if (prod COMPARE_OP accum[idx]) {                          \
                        accum[idx] = prod;                                     \
                        accum_idx[idx] = global_k;                             \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
        STORE_C_ARGMAX_TILE(C, argmax)                                          \
}

// ============================================================================
// I32 GEMM WITH ARGMAX KERNEL MACRO
// ============================================================================

#define TROPICAL_GEMM_ARGMAX_I32(KERNEL_NAME, INIT_VAL, COMPARE_OP, MUL_FN, ZERO_FN) \
extern "C" __global__ void KERNEL_NAME(                                        \
    const int* __restrict__ A,                                                 \
    const int* __restrict__ B,                                                 \
    int* __restrict__ C,                                                       \
    unsigned int* __restrict__ argmax,                                         \
    int M, int N, int K                                                        \
) {                                                                            \
    const int BLOCK_SIZE_M = 64;                                               \
    const int BLOCK_SIZE_K = 32;                                               \
    const int BLOCK_SIZE_N = 64;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;                    \
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;                    \
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ int As[BLOCK_SIZE_M * BLOCK_SIZE_K];                            \
    __shared__ int Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];                            \
                                                                               \
    int accum[THREAD_SIZE_M * THREAD_SIZE_N];                                  \
    int accum_idx[THREAD_SIZE_M * THREAD_SIZE_N];                              \
    int regs_a[THREAD_SIZE_M];                                                 \
    int regs_b[THREAD_SIZE_N];                                                 \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        accum[i] = INIT_VAL;                                                   \
        accum_idx[i] = 0;                                                      \
    }                                                                          \
                                                                               \
    const int A_TILE_COL = tid / BLOCK_SIZE_M;                                 \
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;                                 \
    const int B_TILE_COL = tid / BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;                                 \
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;         \
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        LOAD_A_TILE(A, INIT_VAL)                                          \
                                                                               \
        LOAD_B_TILE(B, INIT_VAL)                                          \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            int global_k = tile_idx + k;                                       \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm,   \
                                           k, BLOCK_SIZE_M)];                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn,\
                                           BLOCK_SIZE_K)];                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    int prod = MUL_FN(regs_a[tm], regs_b[tn]);                  \
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);               \
                    if (prod COMPARE_OP accum[idx]) {                          \
                        accum[idx] = prod;                                     \
                        accum_idx[idx] = global_k;                             \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
        STORE_C_ARGMAX_ZERO_TILE(C, argmax, ZERO_FN)                                 \
}

// ============================================================================
// I64 GEMM WITH ARGMAX KERNEL MACRO
// ============================================================================

#define TROPICAL_GEMM_ARGMAX_I64(KERNEL_NAME, INIT_VAL, COMPARE_OP, MUL_FN, ZERO_FN) \
extern "C" __global__ void KERNEL_NAME(                                        \
    const long long* __restrict__ A,                                           \
    const long long* __restrict__ B,                                           \
    long long* __restrict__ C,                                                 \
    unsigned int* __restrict__ argmax,                                         \
    int M, int N, int K                                                        \
) {                                                                            \
    const int BLOCK_SIZE_M = 32;                                               \
    const int BLOCK_SIZE_K = 16;                                               \
    const int BLOCK_SIZE_N = 32;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;                    \
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;                    \
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ long long As[BLOCK_SIZE_M * BLOCK_SIZE_K];                      \
    __shared__ long long Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];                      \
                                                                               \
    long long accum[THREAD_SIZE_M * THREAD_SIZE_N];                            \
    int accum_idx[THREAD_SIZE_M * THREAD_SIZE_N];                              \
    long long regs_a[THREAD_SIZE_M];                                           \
    long long regs_b[THREAD_SIZE_N];                                           \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        accum[i] = INIT_VAL;                                                   \
        accum_idx[i] = 0;                                                      \
    }                                                                          \
                                                                               \
    const int A_TILE_COL = tid / BLOCK_SIZE_M;                                 \
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;                                 \
    const int B_TILE_COL = tid / BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;                                 \
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;         \
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        LOAD_A_TILE(A, INIT_VAL)                                          \
                                                                               \
        LOAD_B_TILE(B, INIT_VAL)                                          \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            int global_k = tile_idx + k;                                       \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm,   \
                                           k, BLOCK_SIZE_M)];                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn,\
                                           BLOCK_SIZE_K)];                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    long long prod = MUL_FN(regs_a[tm], regs_b[tn]);           \
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);               \
                    if (prod COMPARE_OP accum[idx]) {                          \
                        accum[idx] = prod;                                     \
                        accum_idx[idx] = global_k;                             \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
        STORE_C_ARGMAX_ZERO_TILE(C, argmax, ZERO_FN)                                 \
}

// ============================================================================
// BACKWARD PASS KERNEL MACROS
// ============================================================================

#define TROPICAL_BACKWARD_A(KERNEL_NAME, TYPE, ATOMIC_ADD)                     \
extern "C" __global__ void KERNEL_NAME(                                        \
    const TYPE* __restrict__ grad_c,                                           \
    const unsigned int* __restrict__ argmax,                                   \
    TYPE* __restrict__ grad_a,                                                 \
    int M, int N, int K                                                        \
) {                                                                            \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = M * N;                                                         \
    if (idx < total) {                                                         \
        int i = idx % M;                                                       \
        int k = (int)argmax[idx];                                              \
        if (k >= 0 && k < K) {                                                 \
            ATOMIC_ADD(&grad_a[i + k * M], grad_c[idx]);                       \
        }                                                                      \
    }                                                                          \
}

#define TROPICAL_BACKWARD_B(KERNEL_NAME, TYPE, ATOMIC_ADD)                     \
extern "C" __global__ void KERNEL_NAME(                                        \
    const TYPE* __restrict__ grad_c,                                           \
    const unsigned int* __restrict__ argmax,                                   \
    TYPE* __restrict__ grad_b,                                                 \
    int M, int N, int K                                                        \
) {                                                                            \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = M * N;                                                         \
    if (idx < total) {                                                         \
        int j = idx / M;                                                       \
        int k = (int)argmax[idx];                                              \
        if (k >= 0 && k < K) {                                                 \
            ATOMIC_ADD(&grad_b[k + j * K], grad_c[idx]);                       \
        }                                                                      \
    }                                                                          \
}

// ============================================================================
// KERNEL INSTANTIATIONS
// ============================================================================

// --- F32 Basic GEMM Kernels ---
TROPICAL_GEMM_F32(tropical_maxplus_f32_nn, NEG_INF_F32, fmaxf, +)
TROPICAL_GEMM_F32(tropical_minplus_f32_nn, INF_F32,     fminf, +)
TROPICAL_GEMM_F32(tropical_maxmul_f32_nn,  0.0f,        fmaxf, *)

// --- F64 Basic GEMM Kernels ---
TROPICAL_GEMM_F64(tropical_maxplus_f64_nn, NEG_INF_F64, fmax, +)
TROPICAL_GEMM_F64(tropical_minplus_f64_nn, INF_F64,     fmin, +)
TROPICAL_GEMM_F64(tropical_maxmul_f64_nn,  0.0,         fmax, *)

// --- F32 GEMM with Argmax Kernels ---
TROPICAL_GEMM_ARGMAX_F32(tropical_maxplus_f32_nn_with_argmax, NEG_INF_F32, >, +)
TROPICAL_GEMM_ARGMAX_F32(tropical_minplus_f32_nn_with_argmax, INF_F32,     <, +)
TROPICAL_GEMM_ARGMAX_F32(tropical_maxmul_f32_nn_with_argmax,  0.0f,        >, *)

// --- F64 GEMM with Argmax Kernels ---
TROPICAL_GEMM_ARGMAX_F64(tropical_maxplus_f64_nn_with_argmax, NEG_INF_F64, >, +)
TROPICAL_GEMM_ARGMAX_F64(tropical_minplus_f64_nn_with_argmax, INF_F64,     <, +)
TROPICAL_GEMM_ARGMAX_F64(tropical_maxmul_f64_nn_with_argmax,  0.0,         >, *)

// --- I32 Basic GEMM Kernels ---
TROPICAL_GEMM_I32(tropical_maxplus_i32_nn, NEG_INF_I32, max_i32, add_i32)
TROPICAL_GEMM_I32(tropical_minplus_i32_nn, INF_I32,     min_i32, add_i32)
TROPICAL_GEMM_I32(tropical_maxmul_i32_nn,  0,           max_i32, mul_i32)

// --- I64 Basic GEMM Kernels ---
TROPICAL_GEMM_I64(tropical_maxplus_i64_nn, NEG_INF_I64, max_i64, add_i64)
TROPICAL_GEMM_I64(tropical_minplus_i64_nn, INF_I64,     min_i64, add_i64)
TROPICAL_GEMM_I64(tropical_maxmul_i64_nn,  0LL,         max_i64, mul_i64)

// --- BOOL Basic GEMM Kernel (AndOr semiring) ---
TROPICAL_GEMM_BOOL(tropical_andor_bool_nn, false, or_bool, and_bool)

// --- U32/U64 Bitwise GEMM Kernels (bit-packed boolean) ---
TROPICAL_GEMM_U32(tropical_bitwise_u32_nn, 0u,   or_u32, and_u32)
TROPICAL_GEMM_U64(tropical_bitwise_u64_nn, 0ull, or_u64, and_u64)

// --- I32 GEMM with Argmax Kernels ---
// ZERO_FN canonicalizes a drifted tropical-zero cell's argmax to 0 at write-out.
TROPICAL_GEMM_ARGMAX_I32(tropical_maxplus_i32_nn_with_argmax, NEG_INF_I32, >, add_i32, zero_maxplus_i32)
TROPICAL_GEMM_ARGMAX_I32(tropical_minplus_i32_nn_with_argmax, INF_I32,     <, add_i32, zero_minplus_i32)
TROPICAL_GEMM_ARGMAX_I32(tropical_maxmul_i32_nn_with_argmax,  0,           >, mul_i32, zero_never_i32)

// --- I64 GEMM with Argmax Kernels ---
TROPICAL_GEMM_ARGMAX_I64(tropical_maxplus_i64_nn_with_argmax, NEG_INF_I64, >, add_i64, zero_maxplus_i64)
TROPICAL_GEMM_ARGMAX_I64(tropical_minplus_i64_nn_with_argmax, INF_I64,     <, add_i64, zero_minplus_i64)
TROPICAL_GEMM_ARGMAX_I64(tropical_maxmul_i64_nn_with_argmax,  0LL,         >, mul_i64, zero_never_i64)

// --- Backward Pass Kernels (float/double only, no integer gradients) ---
TROPICAL_BACKWARD_A(tropical_backward_a_f32, float,  atomicAdd)
TROPICAL_BACKWARD_B(tropical_backward_b_f32, float,  atomicAdd)
TROPICAL_BACKWARD_A(tropical_backward_a_f64, double, atomicAddDouble)
TROPICAL_BACKWARD_B(tropical_backward_b_f64, double, atomicAddDouble)

// ============================================================================
// BATCHED F32 GEMM WITH ARGMAX KERNEL MACRO
// ============================================================================
// Strided batched GEMM: processes batch_size independent GEMMs
// Uses blockIdx.z for batch index, strides for memory offsets

#define TROPICAL_GEMM_BATCHED_ARGMAX_F32(KERNEL_NAME, INIT_VAL, COMPARE_OP, MUL_OP) \
extern "C" __global__ void KERNEL_NAME(                                        \
    const float* __restrict__ A,                                               \
    const float* __restrict__ B,                                               \
    float* __restrict__ C,                                                     \
    unsigned int* __restrict__ argmax,                                         \
    int M, int N, int K,                                                       \
    int strideA, int strideB, int strideC                                      \
) {                                                                            \
    const int BLOCK_SIZE_M = 64;                                               \
    const int BLOCK_SIZE_K = 32;                                               \
    const int BLOCK_SIZE_N = 64;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int batch_idx = blockIdx.z;                                                \
    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;                    \
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;                    \
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const float* A_batch = A + batch_idx * strideA;                            \
    const float* B_batch = B + batch_idx * strideB;                            \
    float* C_batch = C + batch_idx * strideC;                                  \
    unsigned int* argmax_batch = argmax + batch_idx * strideC;                          \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];                          \
    __shared__ float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];                          \
                                                                               \
    float accum[THREAD_SIZE_M * THREAD_SIZE_N];                                \
    int accum_idx[THREAD_SIZE_M * THREAD_SIZE_N];                              \
    float regs_a[THREAD_SIZE_M];                                               \
    float regs_b[THREAD_SIZE_N];                                               \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        accum[i] = INIT_VAL;                                                   \
        accum_idx[i] = 0;                                                      \
    }                                                                          \
                                                                               \
    const int A_TILE_COL = tid / BLOCK_SIZE_M;                                 \
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;                                 \
    const int B_TILE_COL = tid / BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;                                 \
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;         \
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        LOAD_A_TILE(A_batch, INIT_VAL)                                          \
                                                                               \
        LOAD_B_TILE(B_batch, INIT_VAL)                                          \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            int global_k = tile_idx + k;                                       \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm,   \
                                           k, BLOCK_SIZE_M)];                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn,\
                                           BLOCK_SIZE_K)];                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    float prod = regs_a[tm] MUL_OP regs_b[tn];                 \
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);               \
                    if (prod COMPARE_OP accum[idx]) {                          \
                        accum[idx] = prod;                                     \
                        accum_idx[idx] = global_k;                             \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
        STORE_C_ARGMAX_TILE(C_batch, argmax_batch)                                          \
}

// --- Batched F32 GEMM with Argmax Kernels ---
TROPICAL_GEMM_BATCHED_ARGMAX_F32(tropical_maxplus_f32_nn_batched_with_argmax, NEG_INF_F32, >, +)
TROPICAL_GEMM_BATCHED_ARGMAX_F32(tropical_minplus_f32_nn_batched_with_argmax, INF_F32,     <, +)
TROPICAL_GEMM_BATCHED_ARGMAX_F32(tropical_maxmul_f32_nn_batched_with_argmax,  0.0f,        >, *)
