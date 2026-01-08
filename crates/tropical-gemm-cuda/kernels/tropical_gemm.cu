// Tropical GEMM CUDA Kernels
// Adapted from CuTropicalGEMM.jl

// Define infinity constants (NVRTC doesn't have access to standard headers)
#define INFINITY __int_as_float(0x7f800000)
#define INFINITY_F64 __longlong_as_double(0x7ff0000000000000LL)

//
// Blocking parameters:
// - BLOCK_SIZE_M = 64: rows of C per thread block
// - BLOCK_SIZE_K = 32: depth of tile (f32) or 16 (f64)
// - BLOCK_SIZE_N = 64: cols of C per thread block
// - THREAD_SIZE_M = 4: rows of C per thread
// - THREAD_SIZE_N = 4: cols of C per thread

#define OFFSET_ROW(row, col, ld) ((row) * (ld) + (col))
#define OFFSET_COL(row, col, ld) ((col) * (ld) + (row))

// ============================================================================
// TropicalMaxPlus<f32>: C[i,j] = max_k(A[i,k] + B[k,j])
// ============================================================================

extern "C" __global__ void tropical_maxplus_f32_nn(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;

    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;  // 16
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;  // 16
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;   // 256

    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    const int tid = threadIdx.y * bszm + threadIdx.x;

    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    // Initialize with -infinity (tropical zero for MaxPlus)
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = -INFINITY;
    }

    const int A_TILE_COL = tid / BLOCK_SIZE_M;
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;
    const int B_TILE_COL = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;

    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        // Load A tile (column-major)
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            int col = A_TILE_COL + i + tile_idx;
            float val = -INFINITY;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        // Load B tile (column-major)
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            int row = tile_idx + B_TILE_ROW;
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            float val = -INFINITY;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        __syncthreads();

        // Compute tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }

            #pragma unroll
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    // MaxPlus: tropical_mul = +, tropical_add = max
                    float prod = regs_a[tm] + regs_b[tn];
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    accum[idx] = fmaxf(accum[idx], prod);
                }
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm;
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn;
            if (row < M && col < N) {
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)];
            }
        }
    }
}

// ============================================================================
// TropicalMinPlus<f32>: C[i,j] = min_k(A[i,k] + B[k,j])
// ============================================================================

extern "C" __global__ void tropical_minplus_f32_nn(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;

    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;

    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    const int tid = threadIdx.y * bszm + threadIdx.x;

    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    // Initialize with +infinity (tropical zero for MinPlus)
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = INFINITY;
    }

    const int A_TILE_COL = tid / BLOCK_SIZE_M;
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;
    const int B_TILE_COL = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;

    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            int col = A_TILE_COL + i + tile_idx;
            float val = INFINITY;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            int row = tile_idx + B_TILE_ROW;
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            float val = INFINITY;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }

            #pragma unroll
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    // MinPlus: tropical_mul = +, tropical_add = min
                    float prod = regs_a[tm] + regs_b[tn];
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    accum[idx] = fminf(accum[idx], prod);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm;
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn;
            if (row < M && col < N) {
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)];
            }
        }
    }
}

// ============================================================================
// TropicalMaxMul<f32>: C[i,j] = max_k(A[i,k] * B[k,j])
// ============================================================================

extern "C" __global__ void tropical_maxmul_f32_nn(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;

    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;

    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    const int tid = threadIdx.y * bszm + threadIdx.x;

    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    // Initialize with 0 (tropical zero for MaxMul)
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = 0.0f;
    }

    const int A_TILE_COL = tid / BLOCK_SIZE_M;
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;
    const int B_TILE_COL = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;

    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            int col = A_TILE_COL + i + tile_idx;
            float val = 0.0f;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            int row = tile_idx + B_TILE_ROW;
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            float val = 0.0f;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }

            #pragma unroll
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    // MaxMul: tropical_mul = *, tropical_add = max
                    float prod = regs_a[tm] * regs_b[tn];
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    accum[idx] = fmaxf(accum[idx], prod);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm;
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn;
            if (row < M && col < N) {
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)];
            }
        }
    }
}

// ============================================================================
// TropicalMaxPlus<f64>: C[i,j] = max_k(A[i,k] + B[k,j])
// ============================================================================

extern "C" __global__ void tropical_maxplus_f64_nn(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 16;
    const int BLOCK_SIZE_N = 32;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;

    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;  // 8
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;  // 8
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;   // 64

    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int DIM_GRID_Y = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    const int tid = threadIdx.y * bszm + threadIdx.x;

    __shared__ double As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ double Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    double accum[THREAD_SIZE_M * THREAD_SIZE_N];
    double regs_a[THREAD_SIZE_M];
    double regs_b[THREAD_SIZE_N];

    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = -INFINITY_F64;
    }

    const int A_TILE_COL = tid / BLOCK_SIZE_M;
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;
    const int B_TILE_COL = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;

    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            int col = A_TILE_COL + i + tile_idx;
            double val = -INFINITY_F64;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            int row = tile_idx + B_TILE_ROW;
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            double val = -INFINITY_F64;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }

            #pragma unroll
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    double prod = regs_a[tm] + regs_b[tn];
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    accum[idx] = fmax(accum[idx], prod);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm;
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn;
            if (row < M && col < N) {
                C[OFFSET_COL(row, col, M)] = accum[OFFSET_COL(tm, tn, THREAD_SIZE_M)];
            }
        }
    }
}

// ============================================================================
// KERNELS WITH ARGMAX TRACKING FOR BACKWARD PROPAGATION
// ============================================================================
// These kernels also output the k-index that produced each optimal C[i,j]
// This is essential for computing gradients in tropical neural networks

// ============================================================================
// TropicalMaxPlus<f32> with argmax: C[i,j] = max_k(A[i,k] + B[k,j])
// ============================================================================

extern "C" __global__ void tropical_maxplus_f32_nn_with_argmax(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int* __restrict__ argmax,  // Output: k-index that produced each C[i,j]
    int M, int N, int K
) {
    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;

    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;

    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    const int tid = threadIdx.y * bszm + threadIdx.x;

    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    int accum_idx[THREAD_SIZE_M * THREAD_SIZE_N];  // Track winning k-index
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = -INFINITY;
        accum_idx[i] = 0;
    }

    const int A_TILE_COL = tid / BLOCK_SIZE_M;
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;
    const int B_TILE_COL = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;

    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        // Load A tile
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            int col = A_TILE_COL + i + tile_idx;
            float val = -INFINITY;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        // Load B tile
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            int row = tile_idx + B_TILE_ROW;
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            float val = -INFINITY;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        __syncthreads();

        // Compute tile with argmax tracking
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            int global_k = tile_idx + k;  // Global k-index

            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }

            #pragma unroll
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    float prod = regs_a[tm] + regs_b[tn];
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    // Update if new value is strictly greater (keeps first winner on tie)
                    if (prod > accum[idx]) {
                        accum[idx] = prod;
                        accum_idx[idx] = global_k;
                    }
                }
            }
        }

        __syncthreads();
    }

    // Store results and argmax indices
    #pragma unroll
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm;
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn;
            if (row < M && col < N) {
                int out_idx = OFFSET_COL(row, col, M);
                int local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                C[out_idx] = accum[local_idx];
                argmax[out_idx] = accum_idx[local_idx];
            }
        }
    }
}

// ============================================================================
// TropicalMinPlus<f32> with argmax: C[i,j] = min_k(A[i,k] + B[k,j])
// ============================================================================

extern "C" __global__ void tropical_minplus_f32_nn_with_argmax(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int* __restrict__ argmax,
    int M, int N, int K
) {
    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;

    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;

    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    const int tid = threadIdx.y * bszm + threadIdx.x;

    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    float accum[THREAD_SIZE_M * THREAD_SIZE_N];
    int accum_idx[THREAD_SIZE_M * THREAD_SIZE_N];
    float regs_a[THREAD_SIZE_M];
    float regs_b[THREAD_SIZE_N];

    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = INFINITY;
        accum_idx[i] = 0;
    }

    const int A_TILE_COL = tid / BLOCK_SIZE_M;
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;
    const int B_TILE_COL = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;

    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            int col = A_TILE_COL + i + tile_idx;
            float val = INFINITY;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            int row = tile_idx + B_TILE_ROW;
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            float val = INFINITY;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            int global_k = tile_idx + k;

            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }

            #pragma unroll
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    float prod = regs_a[tm] + regs_b[tn];
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    // For MinPlus: update if new value is strictly smaller
                    if (prod < accum[idx]) {
                        accum[idx] = prod;
                        accum_idx[idx] = global_k;
                    }
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm;
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn;
            if (row < M && col < N) {
                int out_idx = OFFSET_COL(row, col, M);
                int local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                C[out_idx] = accum[local_idx];
                argmax[out_idx] = accum_idx[local_idx];
            }
        }
    }
}

// ============================================================================
// TropicalMaxPlus<f64> with argmax: C[i,j] = max_k(A[i,k] + B[k,j])
// ============================================================================

extern "C" __global__ void tropical_maxplus_f64_nn_with_argmax(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int* __restrict__ argmax,
    int M, int N, int K
) {
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 16;
    const int BLOCK_SIZE_N = 32;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;

    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;

    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    const int tid = threadIdx.y * bszm + threadIdx.x;

    __shared__ double As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ double Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    double accum[THREAD_SIZE_M * THREAD_SIZE_N];
    int accum_idx[THREAD_SIZE_M * THREAD_SIZE_N];
    double regs_a[THREAD_SIZE_M];
    double regs_b[THREAD_SIZE_N];

    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = -INFINITY_F64;
        accum_idx[i] = 0;
    }

    const int A_TILE_COL = tid / BLOCK_SIZE_M;
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;
    const int B_TILE_COL = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;

    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            int col = A_TILE_COL + i + tile_idx;
            double val = -INFINITY_F64;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            int row = tile_idx + B_TILE_ROW;
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            double val = -INFINITY_F64;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            int global_k = tile_idx + k;

            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }

            #pragma unroll
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    double prod = regs_a[tm] + regs_b[tn];
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    if (prod > accum[idx]) {
                        accum[idx] = prod;
                        accum_idx[idx] = global_k;
                    }
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm;
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn;
            if (row < M && col < N) {
                int out_idx = OFFSET_COL(row, col, M);
                int local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                C[out_idx] = accum[local_idx];
                argmax[out_idx] = accum_idx[local_idx];
            }
        }
    }
}

// ============================================================================
// TropicalMinPlus<f64> with argmax: C[i,j] = min_k(A[i,k] + B[k,j])
// ============================================================================

extern "C" __global__ void tropical_minplus_f64_nn_with_argmax(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int* __restrict__ argmax,
    int M, int N, int K
) {
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 16;
    const int BLOCK_SIZE_N = 32;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;

    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;

    int DIM_GRID_X = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    const int tid = threadIdx.y * bszm + threadIdx.x;

    __shared__ double As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ double Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    double accum[THREAD_SIZE_M * THREAD_SIZE_N];
    int accum_idx[THREAD_SIZE_M * THREAD_SIZE_N];
    double regs_a[THREAD_SIZE_M];
    double regs_b[THREAD_SIZE_N];

    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {
        accum[i] = INFINITY_F64;
        accum_idx[i] = 0;
    }

    const int A_TILE_COL = tid / BLOCK_SIZE_M;
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;
    const int B_TILE_COL = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;

    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_COL_STRIDE) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW;
            int col = A_TILE_COL + i + tile_idx;
            double val = INFINITY_F64;
            if (row < M && col < K) {
                val = A[OFFSET_COL(row, col, M)];
            }
            As[OFFSET_COL(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = val;
        }

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            int row = tile_idx + B_TILE_ROW;
            int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;
            double val = INFINITY_F64;
            if (row < K && col < N) {
                val = B[OFFSET_COL(row, col, K)];
            }
            Bs[OFFSET_COL(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            int global_k = tile_idx + k;

            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                regs_a[tm] = As[OFFSET_COL(threadIdx.x * THREAD_SIZE_M + tm, k, BLOCK_SIZE_M)];
            }

            #pragma unroll
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                regs_b[tn] = Bs[OFFSET_COL(k, threadIdx.y * THREAD_SIZE_N + tn, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
                    double prod = regs_a[tm] + regs_b[tn];
                    int idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                    if (prod < accum[idx]) {
                        accum[idx] = prod;
                        accum_idx[idx] = global_k;
                    }
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {
            int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + tm;
            int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + tn;
            if (row < M && col < N) {
                int out_idx = OFFSET_COL(row, col, M);
                int local_idx = OFFSET_COL(tm, tn, THREAD_SIZE_M);
                C[out_idx] = accum[local_idx];
                argmax[out_idx] = accum_idx[local_idx];
            }
        }
    }
}
