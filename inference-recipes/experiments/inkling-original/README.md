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

We also instantiated the full model’s context and computation graph in the engine’s allocation-free mode. We ran the same 16 configurations twice, first with three loopback RPC backends on one GB10 and then with three physical workers on the four-Spark ring. Every per-rank model, context and compute byte count, graph node count and graph split count matched. The peak below is the largest rank’s projected model, context and compute buffers including host buffers on the head. CUDA driver allocations, operating-system memory and runtime peaks require an actual full-weight launch. [Four physical Sparks](four-spark-graph.jsonl) · [One-Spark loopback control](loopback-graph.jsonl) · [Probe source](reserve.cpp)

| Q3_K_M configuration | Slots × context per slot | Peak projected GiB |
| --- | ---: | ---: |
| Equal layer shares, FP16 KV | 4 × 8,192 | 115.91 |
| Balanced shares, FP16 KV | 4 × 8,192 | 112.03 |
| Equal layer shares, FP16 KV | 4 × 16,384 | 120.87 |
| Balanced shares, FP16 KV | 4 × 16,384 | 117.42 |
| Balanced shares, FP16 KV | 4 × 32,768 | 128.20 |

## KV precision and prefill memory

At four slots of 16,384 tokens, using balanced shares and a unified KV buffer, quantizing the keys to Q8_0 reduced context storage but grew the compute scratch enough to raise the peak from **117.42 to 130.67 GiB** at microbatch 256. At microbatch 64, FP16 and Q8 keys projected 115.37 and 117.42 GiB. With separate slot buffers at that same microbatch, the ordering reversed, at 116.07 and 114.14 GiB. The engine rejected a quantized value cache for Inkling. We repeated 21 cache configurations on four physical Sparks. All 18 successful projections and all three Q8-value rejections reproduced the one-GPU calculation. These are candidate settings for a speed and quality comparison. [Four-Spark cache trials](four-spark-kv.jsonl) · [Loopback microbatch sweep](loopback-kv.jsonl)

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

## Stage the original model for serving

From this recipe directory, build the pinned Inkling engine on one Spark. A build with six parallel jobs completed on the lab machine. The public launch candidate below uses the Q3_K_M checkpoint on one head and one runtime copy per worker. It selects numeric zero thinking effort, balanced layer shares, four client slots and a unified FP16 KV buffer. [Shared ARM64 builder](../mimo26-rpc/compile.sh) · [Shared RPC launcher](../mimo26-rpc/launch.py)

```sh
work="$PWD/inkling-build"
rev=eeac951be2038fc27e89613d8d1158c31d97d732
base=nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
mkdir -p "$work/src" "$work/engine"
curl -fsSL "https://codeload.github.com/danielhanchen/llama.cpp/tar.gz/$rev" -o "$work/source.tar.gz"
tar xzf "$work/source.tar.gz" --strip-components=1 -C "$work/src"
cp ../mimo26-rpc/compile.sh "$work/"
docker run --rm --gpus all --ipc=host -v "$work:/work" -e BUILD_JOBS=6 \
  -e LLAMA_BUILD_NUMBER=0 -e LLAMA_BUILD_COMMIT=eeac951be --entrypoint bash "$base" /work/compile.sh
tar xzf "$work/runtime.tar.gz" -C "$work/engine"
hf download DevQuasar/thinkingmachines.Inkling-GGUF --revision c79703c0c9f308b5f097a7082257a8e8603185eb \
  --include 'Q3_K_M/*.gguf' --local-dir "$work/weights"
```

Copy and extract the runtime on the three workers. Start each adjacent worker on its private link to the head. The opposite worker uses its reachable routed fabric address. The launcher binds the model server to loopback. The rank order for this split is head, adjacent worker, opposite worker and the other adjacent worker. [Worker commands and transport](../mimo26-rpc/#launch)

```sh
first="$work/weights/Q3_K_M/thinkingmachines.Inkling.Q3_K_M-00001-of-00033.gguf"
python3 ../mimo26-rpc/launch.py "$work/engine" serve "$first" \
  "$ADJACENT_B:50052" "$OPPOSITE_A:50052" "$ADJACENT_D:50052" \
  --alias inkling-original-q3km --split 35,32,32,35 --kv-unified --template-effort-zero
```

For the separate-slot Q8-key trial, replace `--kv-unified` with `--q8-keys --ubatch-size 64`. Keep `--template-effort-zero` and check its rendered prompt before comparing outputs.
