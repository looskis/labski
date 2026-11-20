// Build against the pinned TensorFold CUDA sources. Only the block layout and load pipeline differ.
#include "glm_mm.cu"

template<int Rows, int WarpRows, int Stages, bool RowsFirst>
__global__ void __launch_bounds__(WarpRows * 128) labski_q8_tile(
        const __nv_bfloat16* x, long long ldx, const uint32_t* weights, const __half* scales,
        void* output, float* partials, int fp32, int m, int n, int k, int slices) {
    mm_kernel<8, Rows, WarpRows, Stages, RowsFirst>(
        x, ldx, weights, scales, output, partials, fp32, m, n, k, slices);
}

extern "C" int tile_weights(const uint32_t* rows, const __half* scales, uint32_t* tiled,
                            __half* tiled_scales, int n, int k, cudaStream_t stream) {
    const int chunks = (n / 8) * (k / 64) * 32;
    glm_mm_tile8<<<(chunks + 255) / 256, 256, 0, stream>>>(rows, scales, tiled, tiled_scales, n, k);
    return cudaGetLastError();
}

using Kernel = void (*)(const __nv_bfloat16*, long long, const uint32_t*, const __half*,
                        void*, float*, int, int, int, int, int);
struct Variant {
    Kernel kernel;
    int rows, threads, stages;
    bool rows_first;
};
static const Variant variants[] = {
    {glm_mm8_128,                           128, 256, 3, false},
    {labski_q8_tile<128, 2, 3, true>,       128, 256, 3, true},
    {labski_q8_tile<128, 2, 2, false>,      128, 256, 2, false},
    {labski_q8_tile<128, 2, 2, true>,       128, 256, 2, true},
    {labski_q8_tile<256, 4, 2, false>,      256, 512, 2, false},
    {labski_q8_tile<256, 4, 2, true>,       256, 512, 2, true},
};

extern "C" int configure() {
    for (const Variant& v : variants) {
        const int bytes = v.stages * (v.rows * 128 + 8 * 32 * 4 * 4 + 128);
        cudaError_t status = cudaFuncSetAttribute(v.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}

extern "C" int project(int variant, const __nv_bfloat16* x, const uint32_t* weights,
                       const __half* scales, void* output, int fp32, int m, int n, int k, cudaStream_t stream) {
    if (variant < 0 || variant >= static_cast<int>(sizeof(variants) / sizeof(variants[0]))) return cudaErrorInvalidValue;
    const Variant& v = variants[variant];
    int cols = (n + 63) / 64, rows = (m + v.rows - 1) / v.rows;
    dim3 grid = v.rows_first ? dim3(rows, cols) : dim3(cols, rows);
    int bytes = v.stages * (v.rows * 128 + 8 * 32 * 4 * 4 + 128);
    long long ldx = k;
    float* partials = nullptr;
    int slices = 1;
    void* args[] = {&x, &ldx, &weights, &scales, &output, &partials, &fp32, &m, &n, &k, &slices};
    return cudaLaunchKernel(reinterpret_cast<const void*>(v.kernel), grid, dim3(v.threads), args, bytes, stream);
}
