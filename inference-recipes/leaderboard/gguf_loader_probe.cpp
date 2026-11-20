// Check a GGUF header range or validate a complete model's tensor layout without allocating weights.
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <charconv>
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc == 4 && std::strcmp(argv[1], "header") == 0) {
        const char * end = argv[3] + std::strlen(argv[3]);
        size_t prefix_bytes = 0;
        const auto parsed = std::from_chars(argv[3], end, prefix_bytes);
        if (parsed.ec != std::errc{} || parsed.ptr != end) return 2;
        auto * header = gguf_init_from_file(argv[2], {true, nullptr});
        if (!header) return 1;
        const auto offset = gguf_get_data_offset(header);
        std::printf("GGUF_HEADER %zu %lld\n", offset, (long long) gguf_get_n_tensors(header));
        gguf_free(header);
        return offset <= prefix_bytes ? 0 : 1;
    }
    if (argc != 3 || std::strcmp(argv[1], "model") != 0) {
        std::fprintf(stderr, "usage: probe header FILE DOWNLOADED_PREFIX_BYTES | probe model FIRST_SHARD\n");
        return 2;
    }
    ggml_backend_load_all();
    llama_backend_init();
    auto params = llama_model_default_params();
    params.no_alloc = true;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[2], params);
    if (model) {
        std::printf("MODEL_SHAPE %d %llu\n", llama_model_n_layer(model),
                    (unsigned long long) llama_model_n_params(model));
        llama_model_free(model);
    }
    llama_backend_free();
    return model ? 0 : 1;
}
