// Project model, context and graph buffers without reading tensor payloads.
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "llama-ext.h"
#include "llama.h"

#include <array>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

static int positive(const char * arg) {
    int value = 0;
    const char * end = arg + std::strlen(arg);
    const auto parsed = std::from_chars(arg, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end || value < 1) {
        throw std::invalid_argument("expected a positive integer");
    }
    return value;
}

int main(int argc, char ** argv) try {
    if (argc < 11 || argc > 14) {
        std::fprintf(stderr, "usage: reserve FIRST_SHARD SLOTS CONTEXT_PER_SLOT SHARE0 SHARE1 SHARE2 SHARE3 RPC1 RPC2 RPC3 [f16|q8-k|q8-kv] [MICROBATCH] [unified|separate]\n");
        return 2;
    }
    const bool q8_k = argc >= 12 && std::strcmp(argv[11], "f16") != 0;
    const bool q8_v = q8_k && std::strcmp(argv[11], "q8-kv") == 0;
    if (q8_k && !q8_v && std::strcmp(argv[11], "q8-k") != 0) return 2;
    const int ubatch = argc >= 13 ? positive(argv[12]) : 256;
    if (ubatch > 512) return 2;
    const bool unified = argc < 14 || std::strcmp(argv[13], "unified") == 0;
    if (!unified && std::strcmp(argv[13], "separate") != 0) return 2;
    const int slots = positive(argv[2]), ctx_per_slot = positive(argv[3]);
    if (slots > std::numeric_limits<int>::max() / ctx_per_slot) return 2;
    std::vector<float> shares(llama_max_devices(), 0.0f);
    if (shares.size() < 4) return 1;
    for (int i = 0; i < 4; ++i) shares[i] = float(positive(argv[4 + i]));

    llama_backend_init();
    ggml_backend_load_all();
    std::array<ggml_backend_dev_t, 5> devices{};
    devices[0] = ggml_backend_dev_by_name("CUDA0");
    if (!devices[0]) throw std::runtime_error("missing CUDA0");
    for (int i = 1; i < 4; ++i) {
        auto * reg = ggml_backend_rpc_add_server(argv[7 + i]);
        if (!reg || ggml_backend_reg_dev_count(reg) != 1) throw std::runtime_error("expected one RPC device per endpoint");
        ggml_backend_register(reg);
        devices[i] = ggml_backend_reg_dev_get(reg, 0);
    }
    auto model_params = llama_model_default_params();
    model_params.no_alloc = true;
    model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    model_params.devices = devices.data();
    model_params.n_gpu_layers = 999;
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    model_params.tensor_split = shares.data();
    auto * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) return 1;

    auto context_params = llama_context_default_params();
    context_params.n_ctx = slots * ctx_per_slot;
    context_params.n_seq_max = slots;
    context_params.n_batch = 512;
    context_params.n_ubatch = ubatch;
    context_params.kv_unified = unified;
    if (q8_k) context_params.type_k = GGML_TYPE_Q8_0;
    if (q8_v) context_params.type_v = GGML_TYPE_Q8_0;
    auto * ctx = llama_init_from_model(model, context_params);
    if (!ctx) {
        llama_model_free(model);
        return 1;
    }
    // Index 4 collects the head's host buffers, which share GB10's physical RAM.
    std::array<llama_memory_breakdown_data, 5> totals{};
    for (const auto & [type, sizes] : llama_get_memory_breakdown(ctx)) {
        int rank = 4;
        if (!ggml_backend_buft_is_host(type)) {
            auto * dev = ggml_backend_buft_get_device(type);
            rank = 0;
            while (rank < 4 && devices[rank] != dev) ++rank;
            if (rank == 4) throw std::runtime_error("unexpected device buffer");
        }
        totals[rank].model += sizes.model;
        totals[rank].context += sizes.context;
        totals[rank].compute += sizes.compute;
    }
    std::puts("rank,model_bytes,context_bytes,compute_bytes");
    for (int i = 0; i < 5; ++i) {
        std::printf("%d,%lld,%lld,%lld\n", i, (long long) totals[i].model,
                    (long long) totals[i].context, (long long) totals[i].compute);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
