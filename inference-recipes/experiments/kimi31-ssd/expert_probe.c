/* Exercise the unmodified Argodrive routed-expert CUDA entrypoint on layer 1. */
#include "ds4_gpu.h"
#include "ds4_kimi_k3.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

enum { LATENT = 3584, FF = 3072, USED = 16, PASSES = 6 };

static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: expert_probe EXPERT_DIRECTORY OUTPUT_F32\n");
        return 2;
    }
    float input[LATENT], output[LATENT], first[LATENT], probabilities[USED];
    int32_t selected[USED];
    for (int i = 0; i < LATENT; ++i) {
        input[i] = (float)((i * 16807 + 377) % 65521 - 32760) / 32768.0f;
    }
    for (int i = 0; i < USED; ++i) {
        selected[i] = (i * 53 + 7) % 896;
        probabilities[i] = (float)(i + 1) / 136.0f;
    }
    if (!ds4_gpu_init()) return 1;
    ds4_k3_expert_source_configure(argv[1], NULL, 17547264);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc(sizeof(output));
    if (!x || !y || !ds4_gpu_tensor_write(x, 0, input, sizeof(input)) || !ds4_gpu_synchronize()) return 1;
    /* Warm the CUDA and reader pools with a disjoint set of experts. */
    int32_t warm[USED];
    for (int i = 0; i < USED; ++i) warm[i] = (selected[i] + 1) % 896;
    if (!ds4_gpu_k3_moe_raw_tensor(y, x, warm, probabilities, USED, 1, FF, LATENT, 4.0f, 25.0f)
            || !ds4_gpu_synchronize()) return 1;
    for (int i = 0; i < PASSES; ++i) {
        uint64_t before, after;
        ds4_gpu_k3_moe_raw_stats(NULL, &before);
        const double start = seconds();
        if (!ds4_gpu_k3_moe_raw_tensor(y, x, selected, probabilities, USED, 1, FF, LATENT, 4.0f, 25.0f)
                || !ds4_gpu_synchronize()) return 1;
        const double elapsed = seconds() - start;
        ds4_gpu_k3_moe_raw_stats(NULL, &after);
        if (!ds4_gpu_tensor_read(y, 0, output, sizeof(output))) return 1;
        if (i == 0) {
            memcpy(first, output, sizeof(output));
        } else if (memcmp(first, output, sizeof(output)) != 0) {
            fprintf(stderr, "cold and cached expert output differ\n");
            return 1;
        }
        printf("{\"pass\":%d,\"seconds\":%.9f,\"engine_read_bytes\":%llu}\n",
               i + 1, elapsed, (unsigned long long)(after - before));
    }
    FILE *file = fopen(argv[2], "wb");
    if (!file || fwrite(first, sizeof(float), LATENT, file) != LATENT || fclose(file) != 0) return 1;
    ds4_gpu_tensor_free(y);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    return 0;
}
