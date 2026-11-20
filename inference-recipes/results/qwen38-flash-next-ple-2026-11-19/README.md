# Qwen3.8 Flash-Next's disk embedding on one DGX Spark

**An exact row cache cut physical NVMe reads by 91% on single-client code edits, from 218.57 MB to 19.78 MB per twelve requests.** Throughput moved from 49.58 to 49.90 output tok/s and GPU energy from 0.756 to 0.754 J/token. The cache conserved SSD bandwidth. Serving rates changed by −1.7% to +0.6% across the four tested conditions. [Serving measurements](#live-cache-comparison) · [Run the experiment](../../experiments/qwen-replay/)

## Row reuse

**The editing workload reused 91% of distinct row lookups at one client and 78% at four.** Twelve unrelated prose prompts reused 24% and 20%, respectively. The installed B12X reader fetched packed weights and scales from the Spark’s own NVMe drive on every gather. A cold 64 MiB simulated cache retained every previously seen row across the synthetic workload. Increasing the payload budget to 256 MiB gave the same counts. [Trace summaries and whole-device counters](trace-summary.json) · [Experiment and pinned reader](../../experiments/qwen-replay/)

| Workload | Clients | Gathers | Cross-gather hits / distinct lookups | Native MB read | NVMe MB read | Repeated rows checked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated edits | 1 | 139 | 23,944 / 26,328 | 218.49 | 233.46 | 24,528 |
| Unrelated prose | 1 | 437 | 9,552 / 39,336 | 325.36 | 325.66 | 9,576 |
| Repeated edits | 4 | 37 | 8,888 / 11,344 | 94.05 | 103.74 | 24,536 |
| Unrelated prose | 4 | 123 | 7,656 / 37,864 | 313.13 | 317.28 | 9,872 |

Each cache hit counts a distinct ID within a gather that appeared in an earlier gather. Repeated-row checks include duplicates within a gather. All 68,512 checks matched the original packed weights and scales byte for byte. The checkpoint uses 90 bytes per row across its two planes. The largest workload saw about 2.6 MiB of distinct row payload. In a separate capacity sweep, 256 KiB of packed payload captured every available cross-gather hit on both editing traces. At 128 KiB the edit hit rates were 83.9% and 66.4%. Prose reached every available hit at 4 MiB. Longer sessions and larger document sets can require different budgets. [Capacity sweep](capacity-sweep.json) [One-client editing trace](edits-c1-rows.jsonl.gz) · [Four-client editing trace](edits-c4-rows.jsonl.gz) · [One-client prose trace](prose-c1-rows.jsonl.gz) · [Four-client prose trace](prose-c4-rows.jsonl.gz)

All four runs used the same 8K FP4-Local service, four-token native MTP and disabled prefix KV caching. The excluded warm-up preceded the trace. The observer retained packed bytes on the Spark and archived row IDs, per-call counters and a digest of the gathered planes. Native bytes and unique blocks are summed over calls. Device bytes multiply the change in the Linux whole-NVMe sectors-read counter by 512, including any other device traffic in that interval. The observer timed the original reader call, which includes waiting for CUDA to finish producing the IDs. That timer does not isolate storage latency. [Edit requests at one](edits-c1-turns.jsonl.gz) · [Edit requests at four](edits-c4-turns.jsonl.gz) · [Prose requests at one](prose-c1-turns.jsonl.gz) · [Prose requests at four](prose-c4-turns.jsonl.gz)

Recompute any cache budget comparison with the archived rows, for example:

```sh
python3 ../../experiments/qwen-replay/replay_lru.py edits-c1-rows.jsonl.gz --mib 0 .0625 .125 .25 .5 1 2 4 64 256
```

## Live cache comparison

The prototype keeps exact packed weight and scale bytes across gathers and reuses the stock native io_uring reader for misses. Every returned byte matched a separate full native read over 135,088 row positions on the verification pass. The four timed blocks then ran in stock, cache, cache, stock order within the same engine process. Each workload began with an excluded warm-up and a cold row cache. All runs used the same checkpoint, prefix-KV setting, native MTP, context, scheduler and synthetic prompts. [Cache implementation](../../experiments/qwen-replay/retained_rows.py) · [Raw per-reader and device counters](serving/readers.json)

| Workload | Clients | Output tok/s, stock → cache | GPU J/token, stock → cache | NVMe MB read, stock → cache | First token ms, stock → cache | Mean token interval ms, stock → cache |
| --- | ---: | --- | --- | --- | --- | --- |
| Edits | 1 | 49.58 → 49.90 | 0.756 → 0.754 | 218.57 → 19.78 | 290 → 283 | 13.72 → 13.65 |
| Prose | 1 | 39.78 → 39.57 | 0.977 → 0.983 | 324.46 → 247.08 | 198 → 199 | 24.61 → 24.85 |
| Edits | 4 | 129.90 → 128.83 | 0.310 → 0.314 | 93.87 → 19.64 | 361 → 363 | 21.91 → 21.58 |
| Prose | 4 | 72.69 → 71.49 | 0.532 → 0.545 | 311.49 → 251.59 | 325 → 337 | 49.32 → 50.40 |

Physical read volume fell by 91.0% and 79.1% on edits and by 23.8% and 19.2% on prose at one and four clients. Read volume is the mean of the two twelve-request blocks for each method. Rates are median aggregate throughput across six turns. First-token and mean-token-interval values are medians across 24 requests. GPU J/token sums NVML energy across the timed turns and divides by the number of output tokens. The largest closing cache payload was 2,756,880 bytes. Worker RSS at block boundaries includes the engine, allocator reuse and the index. It cannot isolate the cache's memory cost. [Paired streaming receipts](serving/turns.jsonl.gz) · [GPU energy samples](serving/energy.csv.gz)

Across 48 matched edit requests the cache and stock engine generated identical token IDs. Each passed 20/24 expected Python syntax trees at one client and 22/24 at four. On prose the paired token sequences matched on 22/24 requests at one client and 7/24 at four. Stock against itself repeated 11/12 and 6/12 sequences respectively. The byte oracle verifies the row substitution itself. These prose measurements capture variability across repeated serving runs and have no independent correctness labels.

Use the standard FP4-Local recipe for the measured serving baseline. For workloads where SSD traffic matters, the experimental launcher accepts `--retain-rows` to enable a 64 MiB payload limit. A bounded Python index and per-gather scratch require additional host memory. The prototype’s throughput and GPU energy were close to stock on the measured workloads. [Serving launcher](../../experiments/qwen-replay/serve_ple.py) · [One-Spark baseline](../qwen38-flash-next-2026-11-19/)
