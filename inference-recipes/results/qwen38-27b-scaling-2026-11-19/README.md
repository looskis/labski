# How many Sparks for Qwen3.8 27B?

Two Sparks sharding the same 21.92 GB mixed NVFP4/FP8 checkpoint delivered **1.65× the single-client throughput** and **1.59× the four-client throughput** of one Spark. The single Spark used the hardware more efficiently at either load. For an interactive user, the pair generated the same 96-token response budget in about 4.7 seconds per request, compared with 7.7 seconds on one Spark. [Checkpoint and runtime settings](../../recipes/qwen38-27b-nvfp4.json)

| Sparks per endpoint | Interconnect | One client, output tok/s | Four clients, aggregate output tok/s | Four-client output tok/s per Spark | Correct / 112 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Local | 12.40 | 45.74 | 45.74 | 74 |
| 2 | Direct ConnectX pair, NCCL configured for both RoCE rails | 20.46 | 72.63 | 36.32 | 77 |

Both deployments used Spark A as the API host. The pair added Spark B as the second tensor-parallel rank. Each cell is the median of three four-request trials with an excluded warm-up. All timed requests generated 96 tokens. Wall time includes prefill and scheduling. The comparison uses 12 distinct prose prompts per concurrency, matched by hash, case and input length across deployments. The direct pair logged NCCL 2.31.2 and the PyNCCL all-reduce backend. [Per-trial measurements](trials.jsonl) · [Per-request timings and answers](requests.jsonl.gz) · [Runtime and benchmark settings](settings.jsonl)

On the matched MMLU-Pro diagnostic, five questions switched from incorrect to correct on the pair and two switched from correct to incorrect. Both deployments hit the 16-token limit four times. A separate single-Spark run on B scored 78/112. One greedy pass per deployment cannot establish an accuracy advantage for either topology. [Paired answers and finish reasons](quality.jsonl.gz) · [Single-Spark crossover](../qwen38-27b-precision-2026-11-19/)

## Reproduce

Prepare the same pinned checkpoint on each participating Spark. Launch `qwen38-27b-nvfp4` on one GPU or with `--nodes 2` on the direct pair, using one rank per host. On rank zero run the same three prose trials at concurrency one and four and the same 112-item quality probe. Use the default seed 42, greedy decoding and thinking off. The context is 8,192 tokens with four scheduler slots, a 2,048-token batch cap, BF16 KV cache and disabled prefix caching. [Distributed launch instructions](../../README.md#one-endpoint-on-multiple-sparks) · [Exact timing and scoring commands](../qwen38-27b-precision-2026-11-19/#reproduce)
