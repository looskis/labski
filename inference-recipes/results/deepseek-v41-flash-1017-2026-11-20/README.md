# DeepSeek V4.1 Flash 1017 on a four-Spark ring

**TensorFold's direct RoCE decode path delivered 43.61 tokens/s for one client and 72.21 aggregate for four.** Using NCCL over RoCE for decode on the same four Sparks delivered 37.63 and 61.79. The direct path gained 15.9% and 16.9% respectively. All 60 paired responses had identical visible text and output-token counts. The checkpoint, rank weights, prompt collective, drafting, context and sampling settings were the same. [Build, launch and transport selection](../../experiments/deepseek1017/)

| Decode transport | Output tok/s, 1 / 4 clients | First streamed delta ms, 1 / 4 | Four-GPU J/output token, 1 / 4 | Free answers | Exact lookups |
| --- | --- | --- | --- | ---: | ---: |
| NCCL over RoCE | 37.63 / 61.79 | 147 / 378 | 4.19 / 3.01 | 20/28 | 8/8 |
| TensorFold direct RoCE | 43.61 / 72.21 | 137 / 352 | 3.98 / 2.85 | 20/28 | 8/8 |

The direct path reduced GPU joules per output token by 4.8% at one client and 5.4% at four. Its active GPU power was higher. Each timing condition had an excluded warm-up and three trials of four prose prompts. Every timed response used its full 128-token budget. The table gives medians of the trial-level rates, median first-delta latencies and four-device energy per generated token. The energy measure interpolates cumulative NVML counters over each trial. It excludes wall-plug measurements. [NCCL trials](nccl/trials.jsonl) · [Direct trials](verbs/trials.jsonl) · [NCCL GPU energy](nccl/deepseek1017-energy.jsonl) · [Direct GPU energy](verbs/deepseek1017-verbs-energy.jsonl)

The 60 comparisons comprise 28 MMLU-Pro questions, 24 timed prose requests and eight deterministic table lookups. The free-answer probe requested a single letter with thinking off and allowed 128 output tokens. Four responses reached the cap in each run. The 28 visible answers and all correctness verdicts agreed. Lookup inputs were 307–308 tokens. This tests short-context inference. NCCL ran first, then direct RoCE after a service restart. The study covers one ordered comparison, three timing trials per condition and one checkpoint. [NCCL answers](nccl/quality.jsonl.gz) · [Direct answers](verbs/quality.jsonl.gz) · [NCCL prose](nccl/requests.jsonl.gz) · [Direct prose](verbs/requests.jsonl.gz) · [NCCL lookups](nccl/lookup.json) · [Direct lookups](verbs/lookup.json)

## What fits

The pinned original release has 510.30 GB of SafeTensors files. TensorFold baked four parity-split rank files totalling 321.32 GB, roughly 80.3 GB per Spark, and each Spark read the original 203.07 GB of Engram tables from its local NVMe. The Sparks use unified memory. A sample taken after the first timed, single-client trial showed 34.2–36.1 GiB of Linux-reported available memory across the four NCCL ranks and 34.3–36.0 GiB with direct RoCE. These are host snapshots. The cgroup snapshots and their file-cache accounting are included separately. [Rank file sizes and index hashes](rank-files.json) · [Image and kit verification](build.json) · [NCCL memory sample](nccl/memory.json) · [Direct memory sample](verbs/memory.json)

Both services used the same ARM64 image on four GB10s, native FP8/MXFP4 weights, native DSpark drafting, original Engram AIO, four 8,192-token slots and the upstream correctness setting `TF_DS_DECODE_DEC=0`. The physical ranks use logical roles 0, 1, 3, 2. Both runs used the same switchless NCCL library and two channels. The direct run additionally opened TensorFold's decode rings with one lane for each row and column peer and two relays per logical node. Every physical rank reported NCCL NET/IB. Both directions on all eight selected port endpoints registered RDMA traffic during each workload. [NCCL startup](nccl/startup.json) · [Direct startup and lane map](verbs/startup.json) · [NCCL port counters](nccl/rdma.csv) · [Direct port counters](verbs/rdma.csv)

Mia's AI Lab publishes a four-Spark reference that reaches 65.8 tok/s for one prose client and 130.5 for four with temperature 0.7, thinking enabled, 16 streams and a 1M-token context. The Labski comparison uses greedy requests, a shorter context and four slots. Its reported percentages compare the two local runs. [Upstream workloads and results](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/tree/8ffa67d53ef15bab997399135f85e8b0910f30be#performance)

## Reproduce the requests

Prepare and start all ranks with the linked launcher. For the direct condition add `--decode-transport verbs` to every rank. Select a GID for your own fabric if needed. From `inference-recipes/` on the head, prepare the same MMLU-Pro corpus and use a fresh results directory per transport. [Launch instructions](../../experiments/deepseek1017/) · [Pinned question preparation](../qwen38-27b-precision-2026-11-19/#reproduce)

```sh
model=deepseek1017-native
out=./deepseek1017-nccl
mkdir "$out"
python3 mmlu_probe.py run "$model" /tmp/mmlu-112.jsonl --enable-thinking off \
  --per-subject 2 --max-tokens 128 --out "$out/quality.jsonl"
for clients in 1 4; do
  python3 bench.py "$model" --workload prose --requests 4 --repeats 3 \
    --concurrency "$clients" --max-tokens 128 --out "$out/prose-c$clients"
done
python3 bench.py "$model" --workload lookup --records 32 --requests 8 \
  --repeats 1 --max-tokens 256 --out "$out/lookup"
```
