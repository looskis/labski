# Qwen3.8 Flash-Next's disk embedding on one DGX Spark

**The editing workload reused 91% of distinct row lookups at one client and 78% at four.** Twelve unrelated prose prompts reused 24% and 20%, respectively. The installed B12X reader fetched packed weights and scales from the Spark’s own NVMe drive on every gather. A cold 64 MiB simulated cache retained every previously seen row across the synthetic workload. Increasing the payload budget to 256 MiB gave the same counts. [Trace summaries and whole-device counters](trace-summary.json) · [Experiment and pinned reader](../../experiments/qwen-replay/)

| Workload | Clients | Gathers | Cross-gather hits / distinct lookups | Native MB read | NVMe MB read | Repeated rows checked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated edits | 1 | 139 | 23,944 / 26,328 | 218.49 | 233.46 | 24,528 |
| Unrelated prose | 1 | 437 | 9,552 / 39,336 | 325.36 | 325.66 | 9,576 |
| Repeated edits | 4 | 37 | 8,888 / 11,344 | 94.05 | 103.74 | 24,536 |
| Unrelated prose | 4 | 123 | 7,656 / 37,864 | 313.13 | 317.28 | 9,872 |

Each cache hit counts a distinct ID within a gather that appeared in an earlier gather. Repeated-row checks include duplicates within a gather. All 68,512 checks matched the original packed weights and scales byte for byte. The checkpoint uses 90 bytes per row across its two planes. The largest workload saw about 2.6 MiB of distinct row payload. Exploring eviction at higher occupancy needs a longer workload. [One-client editing trace](edits-c1-rows.jsonl.gz) · [Four-client editing trace](edits-c4-rows.jsonl.gz) · [One-client prose trace](prose-c1-rows.jsonl.gz) · [Four-client prose trace](prose-c4-rows.jsonl.gz)

All four runs used the same 8K FP4-Local service, four-token native MTP and disabled prefix KV caching. The excluded warm-up preceded the trace. The observer retained packed bytes on the Spark and archived row IDs, per-call counters and a digest of the gathered planes. Native bytes and unique blocks are summed over calls. Device bytes multiply the change in the Linux whole-NVMe sectors-read counter by 512, including any other device traffic in that interval. The observer timed the original reader call, which includes waiting for CUDA to finish producing the IDs. That timer does not isolate storage latency. [Edit requests at one](edits-c1-turns.jsonl.gz) · [Edit requests at four](edits-c4-turns.jsonl.gz) · [Prose requests at one](prose-c1-turns.jsonl.gz) · [Prose requests at four](prose-c4-turns.jsonl.gz)

Recompute any cache budget comparison with the archived rows, for example:

```sh
python3 ../../experiments/qwen-replay/replay_lru.py edits-c1-rows.jsonl.gz
```
