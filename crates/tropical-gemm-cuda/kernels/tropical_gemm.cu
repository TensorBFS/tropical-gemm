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

// Integer "infinity" constants (sentinel values for tropical zero)
// Use sqrt(typemin) to avoid overflow: sqrt(x) + sqrt(x) = 2*sqrt(x) << x
// sqrt(2^31) ≈ 46340, sqrt(2^63) ≈ 3037000499
#define INF_I32 46340
#define NEG_INF_I32 (-46340)
#define INF_I64 3037000499LL
#define NEG_INF_I64 (-3037000499LL)

// Memory layout helpers
#define OFFSET_COL(row, col, ld) ((col) * (ld) + (row))

// Integer max/min functions
__device__ __forceinline__ int max_i32(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int min_i32(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ long long max_i64(long long a, long long b) { return a > b ? a : b; }
__device__ __forceinline__ long long min_i64(long long a, long long b) { return a < b ? a : b; }

// Saturating addition for MaxPlus (propagates -infinity)
__device__ __forceinline__ int saturating_add_maxplus_i32(int a, int b) {
    if (a == NEG_INF_I32 || b == NEG_INF_I32) return NEG_INF_I32;
    return a + b;
}
__device__ __forceinline__ long long saturating_add_maxplus_i64(long long a, long long b) {
    if (a == NEG_INF_I64 || b == NEG_INF_I64) return NEG_INF_I64;
    return a + b;
}

// Saturating addition for MinPlus (propagates +infinity)
__device__ __forceinline__ int saturating_add_minplus_i32(int a, int b) {
    if (a == INF_I32 || b == INF_I32) return INF_I32;
    return a + b;
}
__device__ __forceinline__ long long saturating_add_minplus_i64(long long a, long long b) {
    if (a == INF_I64 || b == INF_I64) return INF_I64;
    return a + b;
}

// Simple multiplication for MaxMul integer types
// For MaxMul: zero = 0, one = 1, add = max, mul = *
// Note: 0 * anything = 0 (correct absorbing element behavior)
__device__ __forceinline__ int mul_i32(int a, int b) {
    return a * b;
}
__device__ __forceinline__ long long mul_i64(long long a, long long b) {
    return a * b;
}

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
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {            \
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;                   \
            int col = A_TILE_COL + i + tile_idx;                               \
            float val = INIT_VAL;                                              \
            if (row < M && col < K) {                                          \
                val = A[OFFSET_COL(row, col, M)];                              \
            }                                                                  \
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;    \
        }                                                                      \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {            \
            int row = tile_idx + B_TILE_ROW;                                   \
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;               \
            float val = INIT_VAL;                                              \
            if (row < K && col < N) {                                          \
                val = B[OFFSET_COL(row, col, K)];                              \
            }                                                                  \
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;    \
        }                                                                      \
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
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn; \
            if (row < M && col < N) {                                          \
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)]; \
            }                                                                  \
        }                                                                      \
    }                                                                          \
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
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {            \
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;                   \
            int col = A_TILE_COL + i + tile_idx;                               \
            double val = INIT_VAL;                                             \
            if (row < M && col < K) {                                          \
                val = A[OFFSET_COL(row, col, M)];                              \
            }                                                                  \
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;    \
        }                                                                      \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {            \
            int row = tile_idx + B_TILE_ROW;                                   \
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;               \
            double val = INIT_VAL;                                             \
            if (row < K && col < N) {                                          \
                val = B[OFFSET_COL(row, col, K)];                              \
            }                                                                  \
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;    \
        }                                                                      \
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
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn; \
            if (row < M && col < N) {                                          \
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)]; \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// ============================================================================
// I32 GEMM KERNEL MACRO
// ============================================================================
// Block sizes for i32: 64x32x64, Thread sizes: 4x4 (same as f32)
// Uses function-based multiply for saturating arithmetic

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
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {            \
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;                   \
            int col = A_TILE_COL + i + tile_idx;                               \
            int val = INIT_VAL;                                                \
            if (row < M && col < K) {                                          \
                val = A[OFFSET_COL(row, col, M)];                              \
            }                                                                  \
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;    \
        }                                                                      \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {            \
            int row = tile_idx + B_TILE_ROW;                                   \
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;               \
            int val = INIT_VAL;                                                \
            if (row < K && col < N) {                                          \
                val = B[OFFSET_COL(row, col, K)];                              \
            }                                                                  \
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;    \
        }                                                                      \
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
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn; \
            if (row < M && col < N) {                                          \
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)]; \
            }                                                                  \
        }                                                                      \
    }                                                                          \
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
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {            \
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;                   \
            int col = A_TILE_COL + i + tile_idx;                               \
            long long val = INIT_VAL;                                          \
            if (row < M && col < K) {                                          \
                val = A[OFFSET_COL(row, col, M)];                              \
            }                                                                  \
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;    \
        }                                                                      \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {            \
            int row = tile_idx + B_TILE_ROW;                                   \
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;               \
            long long val = INIT_VAL;                                          \
            if (row < K && col < N) {                                          \
                val = B[OFFSET_COL(row, col, K)];                              \
            }                                                                  \
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;    \
        }                                                                      \
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
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn; \
            if (row < M && col < N) {                                          \
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)]; \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// ============================================================================
// F32 GEMM WITH ARGMAX KERNEL MACRO
// ============================================================================

#define TROPICAL_GEMM_ARGMAX_F32(KERNEL_NAME, INIT_VAL, COMPARE_OP, MUL_OP)    \
extern "C" __global__ void KERNEL_NAME(                                        \
    const float* __restrict__ A,                                               \
    const float* __restrict__ B,                                               \
    float* __restrict__ C,                                                     \
    int* __restrict__ argmax,                                                  \
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
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {            \
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;                   \
            int col = A_TILE_COL + i + tile_idx;                               \
            float val = INIT_VAL;                                              \
            if (row < M && col < K) {                                          \
                val = A[OFFSET_COL(row, col, M)];                              \
            }                                                                  \
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;    \
        }                                                                      \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {            \
            int row = tile_idx + B_TILE_ROW;                                   \
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;               \
            float val = INIT_VAL;                                              \
            if (row < K && col < N) {                                          \
                val = B[OFFSET_COL(row, col, K)];                              \
            }                                                                  \
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;    \
        }                                                                      \
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
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn; \
            if (row < M && col < N) {                                          \
                int out_idx = OFFSET_COL(row, col, M);                         \
                int local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);             \
                C[out_idx] = accum[local_idx];                                 \
                argmax[out_idx] = accum_idx[local_idx];                        \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// ============================================================================
// F64 GEMM WITH ARGMAX KERNEL MACRO
// ============================================================================

#define TROPICAL_GEMM_ARGMAX_F64(KERNEL_NAME, INIT_VAL, COMPARE_OP, MUL_OP)    \
extern "C" __global__ void KERNEL_NAME(                                        \
    const double* __restrict__ A,                                              \
    const double* __restrict__ B,                                              \
    double* __restrict__ C,                                                    \
    int* __restrict__ argmax,                                                  \
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
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {            \
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;                   \
            int col = A_TILE_COL + i + tile_idx;                               \
            double val = INIT_VAL;                                             \
            if (row < M && col < K) {                                          \
                val = A[OFFSET_COL(row, col, M)];                              \
            }                                                                  \
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;    \
        }                                                                      \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {            \
            int row = tile_idx + B_TILE_ROW;                                   \
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;               \
            double val = INIT_VAL;                                             \
            if (row < K && col < N) {                                          \
                val = B[OFFSET_COL(row, col, K)];                              \
            }                                                                  \
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;    \
        }                                                                      \
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
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn; \
            if (row < M && col < N) {                                          \
                int out_idx = OFFSET_COL(row, col, M);                         \
                int local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);             \
                C[out_idx] = accum[local_idx];                                 \
                argmax[out_idx] = accum_idx[local_idx];                        \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// ============================================================================
// I32 GEMM WITH ARGMAX KERNEL MACRO
// ============================================================================

#define TROPICAL_GEMM_ARGMAX_I32(KERNEL_NAME, INIT_VAL, COMPARE_OP, MUL_FN)    \
extern "C" __global__ void KERNEL_NAME(                                        \
    const int* __restrict__ A,                                                 \
    const int* __restrict__ B,                                                 \
    int* __restrict__ C,                                                       \
    int* __restrict__ argmax,                                                  \
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
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {            \
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;                   \
            int col = A_TILE_COL + i + tile_idx;                               \
            int val = INIT_VAL;                                                \
            if (row < M && col < K) {                                          \
                val = A[OFFSET_COL(row, col, M)];                              \
            }                                                                  \
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;    \
        }                                                                      \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {            \
            int row = tile_idx + B_TILE_ROW;                                   \
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;               \
            int val = INIT_VAL;                                                \
            if (row < K && col < N) {                                          \
                val = B[OFFSET_COL(row, col, K)];                              \
            }                                                                  \
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;    \
        }                                                                      \
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
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn; \
            if (row < M && col < N) {                                          \
                int out_idx = OFFSET_COL(row, col, M);                         \
                int local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);             \
                C[out_idx] = accum[local_idx];                                 \
                argmax[out_idx] = accum_idx[local_idx];                        \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// ============================================================================
// I64 GEMM WITH ARGMAX KERNEL MACRO
// ============================================================================

#define TROPICAL_GEMM_ARGMAX_I64(KERNEL_NAME, INIT_VAL, COMPARE_OP, MUL_FN)    \
extern "C" __global__ void KERNEL_NAME(                                        \
    const long long* __restrict__ A,                                           \
    const long long* __restrict__ B,                                           \
    long long* __restrict__ C,                                                 \
    int* __restrict__ argmax,                                                  \
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
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {            \
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;                   \
            int col = A_TILE_COL + i + tile_idx;                               \
            long long val = INIT_VAL;                                          \
            if (row < M && col < K) {                                          \
                val = A[OFFSET_COL(row, col, M)];                              \
            }                                                                  \
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;    \
        }                                                                      \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {            \
            int row = tile_idx + B_TILE_ROW;                                   \
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;               \
            long long val = INIT_VAL;                                          \
            if (row < K && col < N) {                                          \
                val = B[OFFSET_COL(row, col, K)];                              \
            }                                                                  \
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;    \
        }                                                                      \
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
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn; \
            if (row < M && col < N) {                                          \
                int out_idx = OFFSET_COL(row, col, M);                         \
                int local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);             \
                C[out_idx] = accum[local_idx];                                 \
                argmax[out_idx] = accum_idx[local_idx];                        \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// ============================================================================
// BACKWARD PASS KERNEL MACROS
// ============================================================================

#define TROPICAL_BACKWARD_A(KERNEL_NAME, TYPE, ATOMIC_ADD)                     \
extern "C" __global__ void KERNEL_NAME(                                        \
    const TYPE* __restrict__ grad_c,                                           \
    const int* __restrict__ argmax,                                            \
    TYPE* __restrict__ grad_a,                                                 \
    int M, int N, int K                                                        \
) {                                                                            \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = M * N;                                                         \
    if (idx < total) {                                                         \
        int i = idx % M;                                                       \
        int k = argmax[idx];                                                   \
        if (k >= 0 && k < K) {                                                 \
            ATOMIC_ADD(&grad_a[i + k * M], grad_c[idx]);                       \
        }                                                                      \
    }                                                                          \
}

#define TROPICAL_BACKWARD_B(KERNEL_NAME, TYPE, ATOMIC_ADD)                     \
extern "C" __global__ void KERNEL_NAME(                                        \
    const TYPE* __restrict__ grad_c,                                           \
    const int* __restrict__ argmax,                                            \
    TYPE* __restrict__ grad_b,                                                 \
    int M, int N, int K                                                        \
) {                                                                            \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = M * N;                                                         \
    if (idx < total) {                                                         \
        int j = idx / M;                                                       \
        int k = argmax[idx];                                                   \
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
TROPICAL_GEMM_I32(tropical_maxplus_i32_nn, NEG_INF_I32, max_i32, saturating_add_maxplus_i32)
TROPICAL_GEMM_I32(tropical_minplus_i32_nn, INF_I32,     min_i32, saturating_add_minplus_i32)
TROPICAL_GEMM_I32(tropical_maxmul_i32_nn,  0,           max_i32, mul_i32)

// --- I64 Basic GEMM Kernels ---
TROPICAL_GEMM_I64(tropical_maxplus_i64_nn, NEG_INF_I64, max_i64, saturating_add_maxplus_i64)
TROPICAL_GEMM_I64(tropical_minplus_i64_nn, INF_I64,     min_i64, saturating_add_minplus_i64)
TROPICAL_GEMM_I64(tropical_maxmul_i64_nn,  0LL,         max_i64, mul_i64)

// --- I32 GEMM with Argmax Kernels ---
TROPICAL_GEMM_ARGMAX_I32(tropical_maxplus_i32_nn_with_argmax, NEG_INF_I32, >, saturating_add_maxplus_i32)
TROPICAL_GEMM_ARGMAX_I32(tropical_minplus_i32_nn_with_argmax, INF_I32,     <, saturating_add_minplus_i32)
TROPICAL_GEMM_ARGMAX_I32(tropical_maxmul_i32_nn_with_argmax,  0,           >, mul_i32)

// --- I64 GEMM with Argmax Kernels ---
TROPICAL_GEMM_ARGMAX_I64(tropical_maxplus_i64_nn_with_argmax, NEG_INF_I64, >, saturating_add_maxplus_i64)
TROPICAL_GEMM_ARGMAX_I64(tropical_minplus_i64_nn_with_argmax, INF_I64,     <, saturating_add_minplus_i64)
TROPICAL_GEMM_ARGMAX_I64(tropical_maxmul_i64_nn_with_argmax,  0LL,         >, mul_i64)

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
    int* __restrict__ argmax,                                                  \
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
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;                                   \
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;                                   \
                                                                               \
    const float* A_batch = A + batch_idx * strideA;                            \
    const float* B_batch = B + batch_idx * strideB;                            \
    float* C_batch = C + batch_idx * strideC;                                  \
    int* argmax_batch = argmax + batch_idx * strideC;                          \
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
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {            \
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;                   \
            int col = A_TILE_COL + i + tile_idx;                               \
            float val = INIT_VAL;                                              \
            if (row < M && col < K) {                                          \
                val = A_batch[OFFSET_COL(row, col, M)];                        \
            }                                                                  \
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;    \
        }                                                                      \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {            \
            int row = tile_idx + B_TILE_ROW;                                   \
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;               \
            float val = INIT_VAL;                                              \
            if (row < K && col < N) {                                          \
                val = B_batch[OFFSET_COL(row, col, K)];                        \
            }                                                                  \
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;    \
        }                                                                      \
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
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn; \
            if (row < M && col < N) {                                          \
                int out_idx = OFFSET_COL(row, col, M);                         \
                int local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);             \
                C_batch[out_idx] = accum[local_idx];                           \
                argmax_batch[out_idx] = accum_idx[local_idx];                  \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// --- Batched F32 GEMM with Argmax Kernels ---
TROPICAL_GEMM_BATCHED_ARGMAX_F32(tropical_maxplus_f32_nn_batched_with_argmax, NEG_INF_F32, >, +)
TROPICAL_GEMM_BATCHED_ARGMAX_F32(tropical_minplus_f32_nn_batched_with_argmax, INF_F32,     <, +)
TROPICAL_GEMM_BATCHED_ARGMAX_F32(tropical_maxmul_f32_nn_batched_with_argmax,  0.0f,        >, *)
