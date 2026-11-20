# MiMo-V2.6-Pro MOPD on four DGX Sparks

The 1.02-trillion-parameter MOPD checkpoint served four simultaneous clients on a switchless ring of four 128 GB Sparks. It delivered **5.85 output tokens/s for one client and 11.74 aggregate for four**. The head held the 418.77 GB checkpoint on disk. The workers received their layers over the fabric. [Build and launch](../../experiments/mimo26-rpc/) · [Pinned weights and SHA-256 hashes](../../experiments/mimo26-rpc/weights.json)

| Concurrent clients | Output tokens/s by trial | Median output tokens/s | Median first streamed delta, ms | Four-GPU J/output token |
| ---: | --- | ---: | ---: | ---: |
| 1 | 5.88, 5.80, 5.85 | 5.85 | 1,307 | 13.20 |
| 4 | 11.59, 11.74, 13.16 | 11.74 | 3,278 | 6.88 |

Each timed trial generated four 96-token prose responses with greedy sampling. One warm-up per concurrency was excluded. One client ran first and four clients followed in the same server lifetime. The throughput includes prefill, scheduling and decoding. Energy sums interpolated NVML counters across all four devices. These figures exclude wall-plug energy. [Request texts and timings](q2k-four/requests.jsonl.gz) · [Trials](q2k-four/trials.jsonl) · [GPU energy](q2k-four/mimo26-pro-mixed-energy.jsonl) · [Device counters](q2k-four/mimo26-pro-mixed-energy.csv.gz)

## Answer quality

The model scored **17/28** on the fixed MMLU-Pro short-answer diagnostic and **20/28** when a GBNF grammar forced the first token to be an answer letter. The 28 items cover 14 subjects. Free answers had a 16-token budget, and 6 reached that cap. Requiring a single letter gained 3 correct answers and lost 0 relative to free generation. The inventory lookup workload returned the exact expected value on **8/8** requests. These short tests leave the accuracy difference from Xiaomi’s original checkpoint unmeasured. [Free answers](q2k-four/quality.jsonl.gz) · [Constrained answers](q2k-four/grammar.jsonl.gz) · [Lookup prompts and answers](q2k-four/lookup.json)

## Placement and startup

The GGUF uses Q2_K routed gate and up projections and MXFP4 routed down projections. Eight of 384 routed experts are active per MoE layer. The engine reported 3.28 bits per weight across the text model and placed 71/71 layers, including the output, on the four GPUs. Using equal layer shares gave model buffers of 96.06, 101.11, 101.11, 90.80 GiB in head/worker order. The head also reported a 949.88 MiB host buffer. Each of the four slots had 8,192 tokens. The server disabled prompt caching and reasoning. [Device, KV and compute allocations](q2k-four/startup.json)

From head-container start to the first successful health poll took **18.28 minutes** and **13.31 GPU Wh across four devices**. The measurement began after checkpoint verification and worker startup. The head had recently read the weights for SHA-256 verification. After the single-client trials, Linux reported 13.1–23.4 GiB available across the four nodes. The Sparks share memory between CPU and GPU. [Startup timing](q2k-four/ready.json) · [Startup energy and interpolation samples](q2k-four/startup-energy.json) · [Per-rank memory](q2k-four/memory.json) · [Sampled memory through load and serving](q2k-four/memory-trace.csv.gz)

Two adjacent workers negotiated RoCEv2. Both registered bidirectional RDMA traffic around the workload. The opposite worker used TCP on the routed ConnectX ring and registered zero RDMA bytes over that interval. [Per-port counter deltas and sample times](q2k-four/rdma.csv) This run used ordinary decoding and one sequential layer partition per GPU. Drowzeys reports 24.2 prose tokens/s at one client and 46.3 aggregate at four using a 363.11 GB ARVQ/NVFP4 checkpoint, tensor parallelism and multi-token prediction. Their single-client test used 512 generated tokens at temperature 1.0. The lab is preparing the same local request suite for the [hybrid-21 Spark candidate](../../experiments/mimo26-arvq/). [Pinned hybrid weights](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21/tree/bb83094e0075076cb0b553c965412e20b061b9c2) · [External four-Spark ARVQ deployment](https://github.com/drowzeys/keys-MiMo-V2.6-Pro-MOPD-C3-Jarrelscy-ARVQ-4-DGX-Sparks-1M-Context/tree/d8a10997f2c6b1b6140b204fc5dddddd247d894b)

## Reproduce the workload

Use the linked four-Spark launcher and the pinned MMLU-Pro question preparation. From `inference-recipes/` on the head, choose a fresh output directory. [Question selection](../qwen38-27b-precision-2026-11-19/#reproduce) · [Recorded benchmark settings](q2k-four/settings.jsonl)

```sh
model=mimo26-pro-mopd-q2k
out=./mimo-pro-measurements
mkdir "$out"
python3 mmlu_probe.py run "$model" /tmp/mmlu-112.jsonl --per-subject 2 --out "$out/free.jsonl"
for c in 1 4; do
  python3 bench.py "$model" --thinking off --workload prose --requests 4 --repeats 3 \
    --concurrency "$c" --max-tokens 96 --out "$out/prose-c$c"
done
python3 bench.py "$model" --thinking off --workload lookup --records 32 --requests 8 \
  --repeats 1 --max-tokens 128 --out "$out/lookup"
python3 mmlu_probe.py run "$model" /tmp/mmlu-112.jsonl --per-subject 2 \
  --one-letter-gbnf --out "$out/grammar.jsonl"
```
