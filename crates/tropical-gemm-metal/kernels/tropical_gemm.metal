// Tropical GEMM Metal kernels.
// Ported from crates/tropical-gemm-cuda/kernels/tropical_gemm.cu — keep the
// two files section-aligned when changing either.
#include <metal_stdlib>
using namespace metal;

// Placeholder used by context tests; removed when real kernels land (Task 4).
kernel void tropical_probe(device float* out [[buffer(0)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid == 0) { out[0] = 42.0f; }
}
