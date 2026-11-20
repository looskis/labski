# How many Sparks for Qwen3.8 27B?

Two Sparks sharding the same 21.92 GB mixed NVFP4/FP8 checkpoint delivered **1.65× the single-client throughput** and **1.59× the four-client throughput** of one Spark. On the four-rank routed TCP ring the same model delivered **8.03 output tok/s at one client**. Among these three configurations the direct pair was fastest and one Spark delivered the most tokens per second per device. For an interactive user, the pair generated the same 96-token response budget in about 4.7 seconds per request, compared with 7.7 seconds on one Spark. [Checkpoint and runtime settings](../../recipes/qwen38-27b-nvfp4.json)

| Sparks per endpoint | Interconnect | One client, output tok/s | Four clients, aggregate output tok/s | Four-client output tok/s per Spark | Correct / 112 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Local | 12.40 | 45.74 | 45.74 | 74 |
| 2 | Direct ConnectX pair, NCCL configured for both RoCE rails | 20.46 | 72.63 | 36.32 | 77 |
| 4 | Switchless ConnectX ring, routed NCCL Socket | 8.03 | 25.72 | 6.43 | 74 |

All deployments used Spark A as the API host. The pair added Spark B as the second tensor-parallel rank. Each cell is the median of three four-request trials with an excluded warm-up. All timed requests generated 96 tokens. Wall time includes prefill and scheduling. The comparison uses 12 distinct prose prompts per concurrency, matched by hash, case and input length across deployments. The direct pair logged NCCL 2.31.2 and the PyNCCL all-reduce backend. [Per-trial measurements](trials.jsonl) · [Per-request timings and answers](requests.jsonl.gz) · [Runtime and benchmark settings](settings.jsonl)

On the matched MMLU-Pro diagnostic, five questions switched from incorrect to correct on the pair and two switched from correct to incorrect. Both deployments hit the 16-token limit four times. The four-rank run scored 74/112 with three truncations. Six answers correct on A alone became incorrect on four ranks and six changed in the other direction. A separate single-Spark run on B scored 78/112. One greedy pass per deployment cannot establish an accuracy advantage for a topology. [Paired answers and finish reasons](quality.jsonl.gz) · [Single-Spark crossover](../qwen38-27b-precision-2026-11-19/)

## What the four-rank TCP result measures

The four-node endpoint used one tensor-parallel rank on each Spark. Its NCCL payload traversed TCP over the ConnectX ring with a routed path to the opposite node. The three single-client trial rates were 5.47, 8.03 and 10.10 output tok/s in that order. The four-client trials were steadier at 25.00–27.18. Model-download containers paused around the timed requests. Container image preparation overlapped part of the run. All three single-client trials fell below both the one-Spark and direct-pair medians. [Ring setup](../../ring-tcp.md) · [Rank-level CUDA collective timings](ring-collectives.jsonl)

NVIDIA reports 32.82 **decode** tok/s with a four-Spark physical ring on this checkpoint using llama-benchy, 8,192 input tokens, 256 output tokens and one client. Our rates include end-to-end processing for a 41-token prompt. This provides a reference for a different ring transport and workload. [NVIDIA topology benchmark, Table 2](https://developer.nvidia.com/blog/running-frontier-class-ai-models-across-8-nvidia-dgx-spark-nodes/)

Across all four GB10s, the median NVML energy per generated token was **10.43 J at one client** and **3.33 J at four clients**. Each trial generated 384 tokens. We sampled the cumulative GPU energy counter on each time-synchronized rank every 200 ms and interpolated at the client trial boundaries. The ranks averaged 79.7–87.2 W combined across these six trials. This is GPU telemetry. Whole-system power needs an external measurement. [Per-rank and per-trial energy](qtp4-energy.jsonl) · [Raw counters](qtp4-energy.csv.gz) · [Energy methodology](../../measurement.md#measure-energy-on-gb10)

## Native BF16 on a pair

Keeping the original BF16 weights gave **8.35 output tok/s for one client** and **31.75 aggregate tok/s for four clients** on a direct Spark pair. It gave the same correct/incorrect verdict as both single-Spark BF16 runs on every one of the 112 items, and the same three item IDs reached the output limit. The single-machine baselines delivered 4.46–4.61 tok/s at one client and 17.54–17.77 at four. This gives a measured way to improve latency while preserving the original weights and the sampled quality. The BF16 pair used the other two Sparks, C and D. Three trials per concurrency returned all 2,304 requested output tokens. [BF16 pair recipe](../../recipes/qwen38-27b-bf16.json) · [Trial timings](bf16-pair-trials.jsonl) · [Paired answers](bf16-pair-quality.jsonl.gz) · [Request records](bf16-pair-requests.jsonl.gz) · [Settings](bf16-pair-settings.jsonl)

The BF16 pair used **9.64 GPU J/output token at one client** and **2.25 at four**. Both PCI-domain functions on each connected port carried unicast RDMA payload, verified by hardware counter deltas from after weight loading began until the benchmark completed. [Energy by rank and trial](qbf16-tp2-energy.jsonl) · [Raw energy samples](qbf16-tp2-energy.csv.gz) · [RDMA byte deltas](bf16-pair-rdma.csv)

## Energy cost of sharding

| Mixed checkpoint | One client, output tok/s | One client, GPU J/output token | Four clients, total output tok/s | Four clients, GPU J/output token |
| --- | ---: | ---: | ---: | ---: |
| Single Spark, repeat | 12.19 | 2.78 | 44.88 | 0.74 |
| Two Sparks, direct RoCE configuration, repeat | 20.45 | 3.61 | 72.68 | 0.99 |
| Four Sparks, routed TCP | 8.03 | 10.43 | 25.72 | 3.33 |

The direct pair saved time for an interactive request at about 30% more GPU energy per output token than the single Spark. At four clients its aggregate throughput grew by 62% and GPU energy per token grew by 33%. Use one Spark when minimizing measured GPU joules per token at these loads, or the direct pair when one user needs the shorter wait. The fresh pair launch delivered 20.45 and 72.68 tok/s, within 0.1% of its original two throughput measurements. It scored 75/112 free-response and 76/112 when constrained to a single letter. [Pair trials](controls/tp2/trials.jsonl) · [Pair energy](controls/tp2/energy.jsonl) · [Raw counters](controls/tp2/energy.csv.gz) · [Free responses](controls/tp2/quality.jsonl.gz) · [Constrained responses](controls/tp2/one-letter.jsonl.gz)

## Repeating the single-Spark run

After a fresh server launch on the same Spark A, mixed precision delivered 12.19 output tok/s at one client and 44.88 at four, within 2% of its first run. NVML measured **2.78 GPU J/output token** and **0.74 J/output token** respectively. Four Spark ranks over TCP consumed about 3.8× and 4.5× as many GPU joules per delivered token at these loads. Trials used the same 12 prose prompts per concurrency. [Single-Spark timings](controls/tp1/trials.jsonl) · [Request records](controls/tp1/requests.jsonl.gz) · [Integrated energy](controls/tp1/energy.jsonl) · [Raw counters](controls/tp1/energy.csv.gz) · [Settings](controls/tp1/settings.jsonl)

The greedy free-response score on the same GPU changed from **74/112 to 78/112** across launches. Five previously incorrect verdicts became correct and one changed the other way, with identical prompts, answer keys and input token counts. This establishes that changing machines is unnecessary to observe score variation on this setup. [Single-Spark repeat answers](controls/tp1/quality.jsonl.gz) · [First run](../qwen38-27b-precision-2026-11-19/)

We then kept the 112 prompts fixed and restricted the server's greedy decoder to one of ten uppercase letter tokens. The score changed from **78 to 80**, with two newly correct answers and every previously correct answer retained. Generated tokens fell from **294 to 112**. This comparison applies to callers that require a single multiple-choice letter. It does not measure open-ended reasoning. To reproduce, add `--one-letter` to the [MMLU-Pro probe](../../mmlu_probe.py) run command. [Constrained outputs and token IDs](controls/tp1/one-letter.jsonl.gz)

## Reproduce

Prepare the same pinned checkpoint on each participating Spark. Launch `qwen38-27b-nvfp4` on one GPU, with `--nodes 2` on the direct pair, or with `--nodes 4 --transport tcp --interface fabric0` on the routed ring. Use one rank per host. On rank zero run the same three prose trials at concurrency one and four and the same 112-item quality probe. Use the default seed 42, greedy decoding and thinking off. The context is 8,192 tokens with four scheduler slots, a 2,048-token batch cap, BF16 KV cache and disabled prefix caching. [Distributed launch instructions](../../README.md#one-endpoint-on-multiple-sparks) · [Exact timing and scoring commands](../qwen38-27b-precision-2026-11-19/#reproduce)
