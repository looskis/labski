# Serving Inkling 2 Small from rank-local weights

Four 128 GB DGX Sparks served the native mixed NVFP4/MXFP8 checkpoint from **431.96 GB of allocated indexed weights across the fleet**, down 72.31% from four 390.06 GB copies. The two-client run delivered **40.68 aggregate output tokens/s**. Under the 768-token answer budget the original and sparse layouts gave the same final text on all 14 questions, including the same incorrect answer. [Weight layout and staging commands](../../../experiments/inkling2-sparse-weights/) · [Pinned two-slot recipe](../../../recipes/inkling2-small-native-c2.json)

Each expert rank consumed its own 96 of 384 expert rows, along with the original shared tensors, headers and scale denominators. The 96 files retained their original logical sizes and offsets. Ten serving assets add 3.45 GB to each standalone snapshot. In this run one Spark hosted all four staged directories on ext4. Rank 2 read its directory locally. The other three ranks read their respective directories over read-only rclone HTTP range mounts on the ConnectX fabric. Each remote mount reported 100.16–100.50 GB read by service readiness and zero additional bytes across the subsequent requests. The four-rank service used the same NCCL RoCE ring for inference in both runs. [Allocation receipts](../../../experiments/inkling2-sparse-weights/staged-shards.jsonl) · [Mounted file checks](mounted-samples.json) · [Startup and transfer counters](startup.json)

## Matched answers

| Greedy workload | Paired requests | Nonempty final text per run | Strict correct, original | Strict correct, sparse |
| --- | ---: | ---: | ---: | ---: |
| MMLU-Pro, low effort, 256-token cap | 28 | 8 | 8/28 | 8/28 |
| First question per subject, low effort, 768-token cap | 14 | 14 | 13/14 | 13/14 |
| Six-digit inventory lookup, 519–3,597 input tokens | 6 | 6 | 6/6 | 6/6 |

The final-answer SHA-256 and finish reason matched for every pair. At the short budget 20 requests in each run exhausted their tokens before emitting a final answer. All 14 longer-budget requests stopped naturally. The 24 additional timed prose requests all reached their 96-token limits with empty final text. Their throughput counts generated reasoning tokens. The comparisons cover visible answers, finish reasons and recorded token counts. Hidden token sequences and logits were outside the measurement. [Short-answer pairs](low256-pairs.jsonl) · [Longer-answer pairs](low768-pairs.jsonl) · [Lookup pairs](lookup-pairs.jsonl) · [Prose pairs](prose-pairs.jsonl)

## Serving performance

| Indexed weights | Clients | Aggregate output tok/s | p95 first streamed delta, ms | Fleet GPU J/output token |
| --- | ---: | ---: | ---: | ---: |
| Original | 1 | 22.66 | 349 | 5.37 |
| Rank-local sparse | 1 | 22.40 | 313 | 5.12 |
| Original | 2 | 39.34 | 643 | 2.98 |
| Rank-local sparse | 2 | 40.68 | 611 | 2.78 |

Each cell is the median of three four-request trials after a warm-up at that concurrency. Energy sums all four GPUs’ counters over each timed trial. Both runs used two scheduler slots, a 4,096-token request limit, BF16 KV and the same prompts. The sparse two-client trials ranged from 34.75 to 41.36 tok/s. The original and sparse measurements came from separate launches several hours apart. Their timing spread limits conclusions about the small differences in rate and energy. [Original trials](../native/two-slot-trials.jsonl) · [Sparse trials](trials.jsonl) · [Request timings](requests.jsonl.gz) · [Settings](settings.jsonl) · [Integrated energy](inkling2-sparse-energy.jsonl) · [Counter samples](inkling2-sparse-energy.csv.gz)
