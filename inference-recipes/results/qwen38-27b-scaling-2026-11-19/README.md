# How many Sparks for Qwen3.8 27B?

On one, two and four Sparks, the same 21.92 GB mixed NVFP4/FP8 checkpoint delivered **12.40, 20.46 and 33.94 output tok/s for one client**. At four clients the end-to-end aggregate reached **45.74, 72.63 and 110.77 tok/s**. The four-rank result uses stock NCCL 2.31.2 over the switchless ConnectX-7 RoCE ring. Four ranks over routed TCP on the same cables delivered 8.03 and 25.72 tok/s. One Spark gave the highest throughput per device and the lowest measured GPU energy per token. [Checkpoint and runtime settings](../../recipes/qwen38-27b-nvfp4.json)

| Sparks per endpoint | Interconnect | One client, output tok/s | Four clients, aggregate output tok/s | Four-client output tok/s per Spark | Correct / 112 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Local | 12.40 | 45.74 | 45.74 | 74 |
| 2 | Direct ConnectX pair, NCCL configured for both RoCE rails | 20.46 | 72.63 | 36.32 | 77 |
| 4 | Switchless ConnectX ring, stock NCCL RoCE | 33.94 | 110.77 | 27.69 | 77 |
| 4 | Switchless ConnectX ring, routed NCCL Socket | 8.03 | 25.72 | 6.43 | 74 |

All deployments used Spark A as the API host. The pair added Spark B as the second tensor-parallel rank. Each cell is the median of three four-request trials with an excluded warm-up. All timed requests generated 96 tokens. Wall time includes prefill and scheduling. The comparison uses 12 distinct prose prompts per concurrency, matched by hash, case and input length across deployments. The direct pair logged NCCL 2.31.2 and the PyNCCL all-reduce backend. [Per-trial measurements](trials.jsonl) · [Per-request timings and answers](requests.jsonl.gz) · [Runtime and benchmark settings](settings.jsonl)

On the matched MMLU-Pro diagnostic, five questions switched from incorrect to correct on the pair and two switched from correct to incorrect. Both deployments hit the 16-token limit four times. The four-rank TCP run scored 74/112 with three truncations. Six answers correct on A alone became incorrect on four ranks and six changed in the other direction. A separate single-Spark run on B scored 78/112. One greedy pass per deployment cannot establish an accuracy advantage for a topology. [Paired answers and finish reasons](quality.jsonl.gz) · [Single-Spark crossover](../qwen38-27b-precision-2026-11-19/)

## Four-Spark RoCE with stock NCCL

Changing the four ranks from routed TCP to the built-in IB transport on the same ring increased median throughput **4.23× at one client and 4.31× at four**. The RoCE single-client trials were 33.94, 32.92 and 34.05 tok/s. Four-client totals were 106.78, 110.90 and 110.77 tok/s. All 24 requests generated exactly 96 tokens. The original TCP single-client trial varied from 5.47 to 10.10 tok/s, so the ratio depends on the baseline run. RoCE scored 77/112 on the matched diagnostic, with five responses reaching its 16-token budget. [Stock ring launcher and counter verification](../../ring-tcp.md#use-roce-on-the-same-four-cables) · [Trials](controls/tp4-roce/trials.jsonl) · [Outputs](controls/tp4-roce/requests.jsonl.gz) · [Quality](controls/tp4-roce/quality.jsonl.gz) · [Settings](controls/tp4-roce/settings.jsonl)

At one client the four GPUs together used **3.38 J per generated token**, slightly below the direct pair’s 3.61 J at the same load. At four clients the four GPUs used **1.10 J/token**, slightly above the direct pair’s 0.99 J. The single Spark remained the lowest-energy option at both loads. The four-node RoCE trials averaged 114–122 W of combined GPU power. [Integrated energy by rank and trial](controls/tp4-roce/qtp4-roce-stock-energy.jsonl) · [Raw counters](controls/tp4-roce/qtp4-roce-stock-energy.csv.gz)

## What the four-rank TCP result measures

The four-node endpoint used one tensor-parallel rank on each Spark. Its NCCL payload traversed TCP over the ConnectX ring with a routed path to the opposite node. The three single-client trial rates were 5.47, 8.03 and 10.10 output tok/s in that order. The four-client trials were steadier at 25.00–27.18. Model-download containers paused around the timed requests. Container image preparation overlapped part of the run. All three single-client trials fell below both the one-Spark and direct-pair medians. [Ring setup](../../ring-tcp.md) · [Rank-level CUDA collective timings](ring-collectives.jsonl)

NVIDIA reports 32.82 **decode** tok/s with a four-Spark physical ring on this checkpoint using llama-benchy, 8,192 input tokens, 256 output tokens and one client. Our rates include end-to-end processing for a 41-token prompt. This provides a reference for a different ring transport and workload. [NVIDIA topology benchmark, Table 2](https://developer.nvidia.com/blog/running-frontier-class-ai-models-across-8-nvidia-dgx-spark-nodes/)

Across all four GB10s, the median NVML energy per generated token was **10.43 J at one client** and **3.33 J at four clients**. Each trial generated 384 tokens. We sampled the cumulative GPU energy counter on each time-synchronized rank every 200 ms and interpolated at the client trial boundaries. The ranks averaged 79.7–87.2 W combined across these six trials. This is GPU telemetry. Whole-system power needs an external measurement. [Per-rank and per-trial energy](qtp4-energy.jsonl) · [Raw counters](qtp4-energy.csv.gz) · [Energy methodology](../../measurement.md#measure-energy-on-gb10)

## Native BF16 on one, two and four Sparks

Keeping the original BF16 weights gave **8.35 output tok/s for one client** and **31.75 aggregate tok/s for four clients** on a direct Spark pair. It gave the same correct/incorrect verdict as both single-Spark BF16 runs on every one of the 112 items, and the same three item IDs reached the output limit. The single-machine baselines delivered 4.46–4.61 tok/s at one client and 17.54–17.77 at four. This gives a measured way to improve latency while preserving the original weights and the sampled quality. The BF16 pair used the other two Sparks, C and D. Three trials per concurrency returned all 2,304 requested output tokens. [BF16 pair recipe](../../recipes/qwen38-27b-bf16.json) · [Trial timings](bf16-pair-trials.jsonl) · [Paired answers](bf16-pair-quality.jsonl.gz) · [Request records](bf16-pair-requests.jsonl.gz) · [Settings](bf16-pair-settings.jsonl)

The BF16 pair used **9.64 GPU J/output token at one client** and **2.25 at four**. Both PCI-domain functions on each connected port carried unicast RDMA payload, verified by hardware counter deltas from after weight loading began until the benchmark completed. [Energy by rank and trial](qbf16-tp2-energy.jsonl) · [Raw energy samples](qbf16-tp2-energy.csv.gz) · [RDMA byte deltas](bf16-pair-rdma.csv)

Four Sparks with stock NCCL over RoCE sustained **13.92 output tok/s at one client** and **54.33 at four clients**, about 3.0–3.1× the two single-Spark BF16 baselines. All 112 correctness verdicts matched the BF16 pair and both single-Spark runs, yielding 83/112. Two four-rank responses reached the diagnostic’s 16-token limit. All 24 timed prose requests generated their 96-token budget. Four-rank GPU energy was **10.16 J/output token at one client** and **2.35 at four**, about 5% above the two-rank BF16 measurements. [Four-rank BF16 trials](controls/bf16-tp4/trials.jsonl) · [Energy](controls/bf16-tp4/qbf16-tp4-roce-energy.jsonl) · [Raw counters](controls/bf16-tp4/qbf16-tp4-roce-energy.csv.gz) · [Quality](controls/bf16-tp4/quality.jsonl.gz) · [Outputs](controls/bf16-tp4/requests.jsonl.gz) · [Settings](controls/bf16-tp4/settings.jsonl)

Holding the four Sparks and RoCE transport fixed, **mixed FP4 delivered 2.44× the BF16 throughput at one client and 2.04× at four**. GPU energy fell from 10.16 to 3.38 J/token at one client and from 2.35 to 1.10 J/token at four. The mixed checkpoint lost eight of the BF16-correct items and gained two, for 77/112 versus 83/112. Its repeated one-Spark runs had already shown variation in greedy answers. The exact item-level changes apply to these two four-rank launches. [Four-rank mixed answers](controls/tp4-roce/quality.jsonl.gz) · [Four-rank BF16 answers](controls/bf16-tp4/quality.jsonl.gz)

## Energy cost of sharding

| Mixed checkpoint | One client, output tok/s | One client, GPU J/output token | Four clients, total output tok/s | Four clients, GPU J/output token |
| --- | ---: | ---: | ---: | ---: |
| Single Spark, repeat | 12.19 | 2.78 | 44.88 | 0.74 |
| Two Sparks, direct RoCE configuration, repeat | 20.45 | 3.61 | 72.68 | 0.99 |
| Same two Sparks, TCP over their direct ConnectX cable | 8.88 | 5.99 | 27.68 | 1.86 |
| Four Sparks, switchless RoCE | 33.94 | 3.38 | 110.77 | 1.10 |
| Four Sparks, routed TCP | 8.03 | 10.43 | 25.72 | 3.33 |

The direct pair saved time for an interactive request at about 30% more GPU energy per output token than the single Spark. At four clients its aggregate throughput grew by 62% and GPU energy per token grew by 33%. Use one Spark when minimizing measured GPU joules per token at these loads, or four Sparks on RoCE for the fastest single endpoint in these trials. The fresh pair launch delivered 20.45 and 72.68 tok/s, within 0.1% of its original two throughput measurements. It scored 75/112 free-response and 76/112 when constrained to a single letter. [Pair trials](controls/tp2/trials.jsonl) · [Pair energy](controls/tp2/energy.jsonl) · [Raw counters](controls/tp2/energy.csv.gz) · [Free responses](controls/tp2/quality.jsonl.gz) · [Constrained responses](controls/tp2/one-letter.jsonl.gz)

### Holding the pair fixed and changing the transport

On the same two machines and checkpoint, the direct RoCE configuration was **2.30× faster for one client and 2.63× faster at four clients** than NCCL Socket over ConnectX. At one client, moving from TCP to RoCE reduced combined GPU energy from 5.99 to 3.61 J/output token. The corresponding four-client reduction was 1.86 to 0.99 J/token. The TCP route between these two adjacent ranks was a single physical hop. A CPU-only container command-line check briefly ran on the worker during the TCP block. The RoCE configuration enabled both PCI-domain functions, and TCP used one rail, so these numbers describe the complete transport configurations. The TCP run scored 79/112 with four responses reaching the budget, compared with 75/112 and three in the fresh RoCE run. [TCP trial timings](controls/tp2-tcp/trials.jsonl) · [TCP energy](controls/tp2-tcp/energy.jsonl) · [TCP request records](controls/tp2-tcp/requests.jsonl.gz) · [TCP answers](controls/tp2-tcp/quality.jsonl.gz)

## Repeating the single-Spark run

After a fresh server launch on the same Spark A, mixed precision delivered 12.19 output tok/s at one client and 44.88 at four, within 2% of its first run. NVML measured **2.78 GPU J/output token** and **0.74 J/output token** respectively. Four Spark ranks over TCP consumed about 3.8× and 4.5× as many GPU joules per delivered token at these loads. Trials used the same 12 prose prompts per concurrency. [Single-Spark timings](controls/tp1/trials.jsonl) · [Request records](controls/tp1/requests.jsonl.gz) · [Integrated energy](controls/tp1/energy.jsonl) · [Raw counters](controls/tp1/energy.csv.gz) · [Settings](controls/tp1/settings.jsonl)

The greedy free-response score on the same GPU changed from **74/112 to 78/112** across launches. Five previously incorrect verdicts became correct and one changed the other way, with identical prompts, answer keys and input token counts. This establishes that changing machines is unnecessary to observe score variation on this setup. [Single-Spark repeat answers](controls/tp1/quality.jsonl.gz) · [First run](../qwen38-27b-precision-2026-11-19/)

We then kept the 112 prompts fixed and restricted the server's greedy decoder to one of ten uppercase letter tokens. The score changed from **78 to 80**, with two newly correct answers and every previously correct answer retained. Generated tokens fell from **294 to 112**. This comparison applies to callers that require a single multiple-choice letter. It does not measure open-ended reasoning. To reproduce, add `--one-letter` to the [MMLU-Pro probe](../../mmlu_probe.py) run command. [Constrained outputs and token IDs](controls/tp1/one-letter.jsonl.gz)

## Reproduce

Prepare the same pinned checkpoint on each participating Spark. Launch `qwen38-27b-nvfp4` on one GPU, with `--nodes 2` on the direct pair, or with `--nodes 4 --transport ring-roce` and the address and HCA settings in the ring instructions. For the TCP baseline use `--nodes 4 --transport tcp --interface fabric0`. Use one rank per host. On rank zero run the same three prose trials at concurrency one and four and the same 112-item quality probe. Use the default seed 42, greedy decoding and thinking off. The context is 8,192 tokens with four scheduler slots, a 2,048-token batch cap, BF16 KV cache and disabled prefix caching. [Distributed launch instructions](../../README.md#one-endpoint-on-multiple-sparks) · [Exact timing and scoring commands](../qwen38-27b-precision-2026-11-19/#reproduce)
