# Thinking Machines Inkling on DGX Spark

The original Inkling has 947.03B parameters across 66 transformer layers. The lab compiled the Inkling llama.cpp branch for ARM64 and SM121a. It accepted all 1,512 tensor descriptors in both the 343.96 GB Q2_K and 451.04 GB Q3_K_M GGUFs. This page records prompt rendering and engine memory projections. Full-weight generation is pending. [Pinned conversions](https://huggingface.co/DevQuasar/thinkingmachines.Inkling-GGUF/tree/c79703c0c9f308b5f097a7082257a8e8603185eb) · [Architecture implementation](https://github.com/danielhanchen/llama.cpp/tree/eeac951be2038fc27e89613d8d1158c31d97d732) · [Loader checks](../../leaderboard/gguf-loader-2026-11-20.csv)

## Set thinking effort in the actual prompt

With `--reasoning off` alone, the original Q3_K_M chat template rendered **“Thinking effort level: 0.9”**. A request with `"reasoning_effort":"none"` also rendered 0.9. Passing numeric zero under `chat_template_kwargs` rendered **“Thinking effort level: 0”**. We checked the rendered system message through the pinned ARM64 server’s `/apply-template` endpoint on a Spark. The fixture loaded Qwen3 0.6B weights and the original Inkling template. These tests measured the text sent to the model. [All ten request bodies and rendered prompts](template-check.json)

For a request-level override use `"chat_template_kwargs":{"reasoning_effort":0.0}`. To set the default for ordinary requests, launch with `--chat-template-kwargs '{"reasoning_effort":0.0}'`. Sending top-level `"reasoning_effort":"none"` restored the template’s 0.9 default even on that launch. Readers can check their own running server with the same request they intend to benchmark.

```sh
curl -fsS http://127.0.0.1:8000/apply-template \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is seven plus five?"}],"chat_template_kwargs":{"reasoning_effort":0.0}}'
```

## Choose a four-Spark layer split

The first two Q3_K_M transformer blocks weigh 0.28 GiB each. Most later MoE blocks weigh about 6.51 GiB. Equal layer shares place 17, 17, 17 and 16 blocks on the four ranks, counting the output as the final block. We enumerated the contiguous four-way splits of the 67 blocks, including the input embedding on the head, and minimized the largest rank’s weight bytes. The split with **18, 16, 16 and 17 blocks** reached 106.42 GiB maximum, down from 110.74 GiB. The corresponding llama.cpp argument is `--tensor-split 35,32,32,35`. Its boundaries fall midway between two layer indices. [Tensor payloads by layer](tensor-bytes.csv) · [Pinned layer assignment code](https://github.com/danielhanchen/llama.cpp/blob/eeac951be2038fc27e89613d8d1158c31d97d732/src/llama-model.cpp)

We also instantiated the full model’s context and computation graph in the engine’s allocation-free mode. One GB10 exposed a local CUDA backend and three separate loopback RPC endpoints. The peak below is the largest rank’s projected model, context and compute buffers including host buffers on the head. CUDA driver allocations, operating-system memory and runtime peaks require an actual full-weight launch. [All precision, slot and context projections](loopback-graph.jsonl) · [Probe source](reserve.cpp)

| Q3_K_M configuration | Slots × context per slot | Peak projected GiB |
| --- | ---: | ---: |
| Equal layer shares, FP16 KV | 4 × 8,192 | 115.91 |
| Balanced shares, FP16 KV | 4 × 8,192 | 112.03 |
| Equal layer shares, FP16 KV | 4 × 16,384 | 120.87 |
| Balanced shares, FP16 KV | 4 × 16,384 | 117.42 |
| Balanced shares, FP16 KV | 4 × 32,768 | 128.20 |

## KV precision and prefill memory

At four slots of 16,384 tokens, using balanced shares and a unified KV buffer, quantizing the keys to Q8_0 reduced context storage but grew the compute scratch enough to raise the peak from **117.42 to 130.67 GiB** at microbatch 256. At microbatch 64, FP16 and Q8 keys projected 115.37 and 117.42 GiB. With separate slot buffers at that same microbatch, the ordering reversed, at 116.07 and 114.14 GiB. The engine rejected a quantized value cache for Inkling. These are candidate settings for a speed and quality comparison. [Per-rank model, context and compute sizes](loopback-kv.jsonl)

## Reproduce the graph check

Build the branch above with CUDA, RPC and RPC RDMA enabled using the same CMake flags as the lab’s ARM64 GGUF engine. The probe includes the branch’s memory-breakdown API. Compile it inside the build environment and run against the first shard. The remaining shards must be in the same directory. Complete headers in appropriately sized sparse files are sufficient for this probe. [Build flags](../mimo26-rpc/compile.sh) · [Header staging procedure](../../leaderboard/)

```sh
lib="$build/bin"
c++ -std=c++17 -I"$src/include" -I"$src/ggml/include" -I"$src/src" reserve.cpp \
  -L"$lib" -lllama -lggml -lggml-base -lggml-rpc -o reserve
export LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}"
# Start ggml-rpc-server --device CUDA0 --host "$FABRIC_IP" --port 50228 on each worker.
./reserve "$first_shard" 4 8192 35 32 32 35 "$RPC1" "$RPC2" "$RPC3" f16 256 unified
./reserve "$first_shard" 4 16384 35 32 32 35 "$RPC1" "$RPC2" "$RPC3" q8-k 64 separate
```

Each RPC argument is a worker address and port. To reproduce the one-GPU calculation, bind three workers to distinct loopback ports in the same network namespace. Rank 4 in the output denotes the head’s host buffers. The probe reads GGUF headers and reserves graphs without generating any tokens.
