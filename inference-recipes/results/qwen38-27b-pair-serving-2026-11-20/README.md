# Two replicas or one tensor-parallel Qwen3.8 27B endpoint?

With the same two DGX Sparks and a 3 GiB KV pool per GPU, **two-way tensor parallelism won at all six tested client counts**. One client reached 20.50 output tok/s on the sharded endpoint against 12.22 on two round-robin replicas. A burst of 64 reached 279.40 against 209.08 aggregate tok/s. At 32 clients, p95 time to the first token was 0.58 seconds on the sharded endpoint and 10.19 seconds on replicas. [Launcher and replay](../../experiments/qwen-pair-serving/)

## 3 GiB KV per Spark

| Clients | Replicas tok/s | TP2 tok/s | Replicas p95 first token, ms | TP2 p95 first token, ms | Replicas GPU J/token | TP2 GPU J/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.22 | 20.50 | 148 | 128 | 3.86 | 3.52 |
| 4 | 46.13 | 72.64 | 258 | 287 | 1.50 | 1.00 |
| 8 | 88.09 | 130.31 | 282 | 351 | 0.78 | 0.56 |
| 16 | 168.01 | 229.80 | 353 | 415 | 0.41 | 0.32 |
| 32 | 167.89 | 233.33 | 10,189 | 579 | 0.43 | 0.32 |
| 64 | 209.08 | 279.40 | 20,466 | 8,999 | 0.35 | 0.27 |

Each row reports the median of three trials. Latency is the median of the three within-trial p95 values. At 4–64 clients the client issues one simultaneous burst per trial. At one client it sends four requests sequentially, alternating between replicas. Each prose request returns 96 tokens. Aggregate throughput includes queueing, prefill and decode. Short prompts, burst size and output length matter when applying these numbers to a live service. [Replica trials](3g/replicas/trials.jsonl) · [TP2 trials](3g/tp2/trials.jsonl) · [Replica requests](3g/replicas/requests.jsonl.gz) · [TP2 requests](3g/tp2/requests.jsonl.gz)

On a fresh replica launch we repeated 32 clients first, then eight. The pair reached 168.39 and 89.03 tok/s, against 167.89 and 88.09 in the original sweep. Each deployment used 32 total scheduler slots, 8K maximum context, disabled prefix caching, and the same mixed NVFP4/FP8 checkpoint. Each replica has its own 2,048-token batch budget. TP2 has one such budget. vLLM reported 36,278 KV tokens per 3 GiB replica. The reported slot ceiling alone does not determine queueing latency. [Repeat trials](3g/replicas-repeat/trials.jsonl) · [Effective settings](3g/replicas/settings.jsonl)

Replicas scored **19/28** free answers. TP2 scored **18/28**, gaining 1 correct and losing 2 relative to replicas. The two shapes produced identical answer text on 23/28 diagnostic questions and 2/384 prose requests. Exact lookups scored 8/8 and 8/8, respectively. One answer in each deployment reached the 16-token cap, on different questions. This single pass does not establish a general accuracy difference between placements. [Replica diagnostic](3g/replicas/quality.jsonl.gz) · [TP2 diagnostic](3g/tp2/quality.jsonl.gz) · [Replica lookups](3g/replicas/lookup.json) · [TP2 lookups](3g/tp2/lookup.json)

The benchmark client ran on the head. Replicas used one loopback API and one private ConnectX API. TP2 used a loopback API and two NCCL RoCE rails between ranks. Each TP2 rank sent and received about 126.5 GB of unicast RDMA payload over the workload. The independent replicas had zero RDMA payload. Other checkpoint transfers on the lab network continued during parts of the campaign. GPU energy integrates NVML counters from both Sparks over the client trial, including the idle replica at one client. Whole-system power requires an external meter. [TP2 counters](3g/tp2/rdma.json) · [Replica counters](3g/replicas/rdma.json) · [TP2 energy](3g/tp2/energy.jsonl) · [Replica energy](3g/replicas/energy.jsonl)
