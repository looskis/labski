# Repeated edits, unrelated prose and disk-backed lookup rows

This replay uses four synthetic Python functions and three successive edits per function. Each edit replaces the previous turn's return constant. The twelve unrelated prose prompts cover different subjects. The probe sends the same messages at one or four clients and scores each edit by comparing the Python syntax tree with the expected tree. It retains hashes of the visible response and the actual generated token IDs, including any end-of-generation token the server returns. [Replay](replay.py) · [Research questions and prior work](../../research-leads.md)

Launch the model with prefix caching disabled for both sides of each comparison. Use a fresh output path. For proposers with cross-request history, start a fresh server between workloads and client counts. Each workload begins with the same excluded one-word warm-up and runs three groups of four requests.

```sh
python3 replay.py "$MODEL" --workload edits --clients 1 --out edits-c1.jsonl
python3 replay.py "$MODEL" --workload prose --clients 4 --out prose-c4.jsonl
```

The streaming receipt records the arrival time and token count of each output chunk. `first_token_s` measures first returned token ID. `mean_token_interval_s` divides the interval between the first and last token-bearing chunks by output tokens minus one. Speculative verification can deliver several tokens together. The receipt also records vLLM's draft and acceptance counter deltas, wall-clock trial boundaries and aggregate output tokens per second. [GPU energy integration](../../measurement.md#measure-energy-on-gb10)

## Build and select a Qwen 27B draft policy

The pinned ARM64 vLLM 0.33.0 image lacks Arctic Inference. Its suffix proposer imports the CPU suffix-tree package. On a Spark, build that extension from the verified 0.1.1 source archive using the serving image’s Python. The build container is disposable. From this experiment directory run:

```sh
mkdir -p "$PWD/arctic-build"
IMAGE=$(python3 -c 'import json; print(json.load(open("../../images.json"))["vllm-0.33.0"])')
docker run --rm --cpus 4 --memory 6g --volume "$PWD/arctic-build:/experiment" \
  --volume "$PWD/build_suffix.sh:/build_suffix.sh:ro" --entrypoint sh "$IMAGE" /build_suffix.sh
python3 serve_drafting.py suffix32 --extension "$PWD/arctic-build/python"
```

Prepare `qwen38-27b-nvfp4` with the main launcher first. Modes `none`, `ngram`, `suffix0` and `suffix32` all use the same target weights, 8K context, four scheduler slots and disabled prefix caching. N-gram lookup considers lengths 2–5. Both suffix modes use depth 16. Each proposer can offer up to five tokens. Zero or 32 completed requests may contribute suffix history. Pass the matching served name, such as `qwen27-draft-suffix32`, to the replay. [Build](build_suffix.sh) · [Serving commands](serve_drafting.py) · [Upstream configuration](https://docs.vllm.ai/en/v0.33.0/features/speculative_decoding/suffix/)

## Flash-Next PLE trace

In the pinned FP4-Local image, vLLM uses FlashInfer 0.7.1. Its disk-table Python module has SHA-256 `601348d52d488057911b5a0b22d36f0e922d80cb956c2658901bee56e63e7c9a` and its native io_uring reader source has SHA-256 `36bcce65d3e131a02749a21105eda1771f80ca0e55686f84c576240fa783f585`. The planner opens the registered files with `O_DIRECT`. It coalesces aligned fragments within a gather, while the native reader resets its byte and block counters at each call. The Python wrapper waits for IDs to be ready and passes the packed weight and scale destinations to the reader. [Pinned image](../../images.json) · [Public reader implementation](https://github.com/flashinfer-ai/flashinfer/blob/601cb12713fd73f1744b15fdea6f9ddaaf2b4ec3/flashinfer/experimental/b12x/loader/_ple_reader.c)

The diagnostic hook wraps the installed class's `_read_staged` method in a disposable serving container. It records IDs after the original method synchronizes them, sums native counters offline, and compares both packed row planes byte-for-byte for repeated IDs. It traces only while a workload marker is present. The trace adds CPU work and an unbounded reference dictionary for the duration of the small synthetic workload. Use a separate uninstrumented deployment for performance comparisons. [Observer](trace_ple.py)

```sh
python3 replay_lru.py edits-c1-READER_PID.jsonl
```

Offline replay counts hits on distinct row IDs per gather that were resident at the start of that gather. Its cold caches use 0, 64 and 256 MiB of packed weight-plus-scale *payload*. Index and eviction metadata would require additional memory. The recorded read bytes, unique blocks and whole-device NVMe sector counts have different meanings. An offline row hit establishes potential reuse. A serving speedup requires measured changes in the real read path. [LRU simulator](replay_lru.py)
