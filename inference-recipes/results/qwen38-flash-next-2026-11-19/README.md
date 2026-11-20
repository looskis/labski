# Qwen3.8 Flash-Next on one or two DGX Sparks

**For these 8K text workloads, the one-Spark FP4-Local recipe delivered the highest throughput and lowest GPU energy per output token.** It reached 33.98 tok/s at one client and 72.86 aggregate at four. The two-Spark recipe returned the first streamed delta sooner for a lone client. [Single-Spark recipe](../../recipes/qwen38-flash-next-fp4-local.json) · [Matched comparison](#single-spark-fp4-local-versus-the-tuned-pair)

For the 132.68 GB checkpoint on two Sparks, use the 2 GiB KV pool and four scheduler slots. The same pair delivered **64.96 aggregate output tok/s** at four clients and **21.73 tok/s** at one. The one-slot control delivered 22.38 and 21.80 respectively. [Four-client recipe](../../recipes/qwen38-flash-next-kv2g-c4.json)

NVIDIA’s 132.68 GB mixed checkpoint served text on two 128 GB DGX Sparks with upstream vLLM 0.33.0, BF16 KV and no speculative decoding. A single client received **21.34 output tokens/s**. Four concurrent clients received **21.87 aggregate output tokens/s** with one scheduler slot, so the extra requests queued. Both physical RoCE functions on the direct link carried data. [Pinned inference recipe](../../recipes/qwen38-flash-next-nvfp4.json) · [RDMA byte counters](rdma.csv)

| Clients | Median aggregate output tok/s | Trial range | Median combined GPU J/output token |
| ---: | ---: | ---: | ---: |
| 1 | 21.34 | 21.22–21.60 | 2.52 |
| 4 | 21.87 | 20.66–22.06 | 2.46 |

The quality probe scored **75/112** in a single thinking-off, greedy pass. Fifteen answers hit the 16-token output limit. The task requires only a letter, and some responses began explaining the solution before returning one. This particular probe scores exact format and a limited answer budget. It does not estimate the model's leaderboard score. Each timed prose trial used four 41-token prompts and a 96-token output budget. All 24 requests reached the budget. Energy covers both GB10 counters, from request submission through receipt of the final response. [All quality answers](quality.jsonl.gz) · [Prose timings and answers](requests.jsonl.gz) · [Trial timings](trials.jsonl) · [Energy per trial](flash-next-tp2-energy.jsonl) · [Raw energy samples](flash-next-tp2-energy.csv.gz)

The checkpoint uses NVFP4 routed experts and an FP8 PLE table. Both rank logs reported 37.44 GiB for the model-loading allocation. Rank zero reported 27.79 GiB available for KV cache under the 0.8 utilization target. These are engine accounting figures in a shared-memory machine. The CPU-side embedding, operating system, file cache and other allocations also consume physical memory. Benchmarking overlapped host-side distribution of a separate container image. The native compilation job on rank zero was paused during the timing trials. [Complete settings](settings.jsonl)

## A smaller KV cache for short contexts

With an 8,192-token context and one scheduler slot, rank zero sized the default KV pool for **1,007,616 tokens**. Setting `--kv-cache-memory-bytes 2147483648` reduced the per-rank KV reservation from **27.79 GiB to 2 GiB**, leaving a 77,451-token pool. Two GB10s on the lab’s other direct pair completed all 24 timing requests and all 112 quality requests with the smaller cache. [2 GiB recipe](../../recipes/qwen38-flash-next-kv2g-c1.json)

| KV setting, one scheduler slot | One client, output tok/s | Four queued clients, total output tok/s | GPU J/output token, one / four clients | Correct / 112 |
| --- | ---: | ---: | --- | ---: |
| Automatic, pair A–B | 21.34 | 21.87 | 2.52 / 2.46 | 75 |
| 2 GiB per rank, pair C–D | 21.80 | 22.38 | 2.51 / 2.47 | 76 |

The smaller pool freed about 25.8 GiB of configured KV capacity per rank. The hardware pairs differed, so the roughly 2% timing difference cannot be attributed to the cache cap alone. Three quality verdicts changed. Both runs used the same checkpoint, BF16 KV dtype, 2,048-token prefill cap and single active sequence. [Small-pool timing trials](kv2g-c1/trials.jsonl) · [Energy by trial](kv2g-c1/flash-next-kv2g-c1-energy.jsonl) · [Free-response answers](kv2g-c1/quality.jsonl.gz)

**For applications that require one option letter**, constraining the greedy decoder to one uppercase A–J token improved the small-pool score from **76 to 84 out of 112** and reduced total generation from **448 tokens to 112**. Eight responses became correct and every previously correct verdict was retained. Free-response decoding had hit its 16-token limit on 16 questions. Both tests used identical prompts with thinking disabled. The constrained grammar applies to this output contract. [Constrained answers and allowed token IDs](kv2g-c1/one-letter.jsonl.gz) · [Scoring code](../../mmlu_probe.py)

## Letting four clients generate together

On the same C–D pair we changed only the number of active sequences from one to four, retaining the 2 GiB KV cap, checkpoint, context, no drafting and all other server and benchmark settings. With four clients the change **raised median throughput by 2.90×** and **cut GPU joules per output token by 61%**. One-client speed changed by 0.3%. Every timed response reached the same 96-token budget.

| Scheduler slots | One client, output tok/s | Four clients, aggregate output tok/s | Four-client GPU J/output token | Free letter score / 112 | Constrained letter score / 112 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 21.80 | 22.38 | 2.47 | 76 | 84 |
| 4 | 21.73 | 64.96 | 0.97 | 77 | 86 |

The three four-client trial rates with four slots were **50.31, 64.96 and 65.49** output tok/s, in run order after the excluded single-request warm-up. All three exceeded the best one-slot rate of 22.98. The first batched trial was visibly slower. Five of the 112 free-response verdicts changed across server launches, with two correct only at one slot and three correct only at four. The four-slot constrained run recovered ten misses and lost one previously correct answer, with 112 generated tokens versus 462 unconstrained. The quality probes were sequential in both deployments. [Four-slot timing and requests](kv2g-c4/trials.jsonl) · [All request records](kv2g-c4/requests.jsonl.gz) · [Energy](kv2g-c4/flash-next-kv2g-c4-energy.jsonl) · [Free answers](kv2g-c4/quality.jsonl.gz) · [Constrained answers](kv2g-c4/one-letter.jsonl.gz) · [Pinned settings](kv2g-c4/settings.jsonl)

## Reproduce

Prepare and serve `qwen38-flash-next-kv2g-c4` for four active clients, `qwen38-flash-next-kv2g-c1` for its single-slot control, or `qwen38-flash-next-nvfp4` for the original automatic-cache baseline on a direct ConnectX pair, with rank zero as the client. Use the pinned ARM64 image, the recipe’s 8,192-token window, the recipe’s scheduler slot count, a 2,048-token batch cap and eager execution. Prepare the 112-item corpus using the Qwen 27B instructions, then run its quality and prose commands with `MODEL` set to the served recipe ID. For energy, start the NVML sampler on each rank before the first timed trial. [Distributed launcher](../../README.md#one-endpoint-on-multiple-sparks) · [Exact prompts and benchmark commands](../qwen38-27b-precision-2026-11-19/#reproduce) · [Energy capture](../../measurement.md#measure-energy-on-gb10)

NVIDIA has also published a one-Spark configuration for a separate 102 GB FP4-Local checkpoint with its PLE table on disk and four-token MTP. The lab ran the one-Spark configuration at an 8K request limit and four scheduler slots. [Lab recipe](../../recipes/qwen38-flash-next-fp4-local.json) · [Pinned checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300) Its active-decode benchmark uses different weights, settings and timing boundaries. [NVIDIA single-Spark deployment](https://github.com/NVIDIA/local-ai-inference-recipes/tree/bb515086841e7a81e7d51925561338e7dee35023/models/qwen/qwen3.8-flash-next/dgx-spark/linux/vllm/fp4-local)

The one-Spark runtime reads its PLE rows using Linux io_uring. On the lab Spark a two-entry queue opened on the host, returned EPERM under Docker 29.6.2's built-in seccomp profile, and opened under the vLLM fork's io_uring profile with Docker's normal AppArmor confinement. The recipe selects the upstream default-deny profile. Its unmodified JSON has SHA-256 `823b66051a52180058aa04b4343eb29a9cc963c1cff306e5c81dd606324252dc`. This tests queue creation. [Profile from the vLLM image source revision](https://github.com/local-inference-lab/vllm/blob/8244d1c44348ae8a5bbfc9f1b607021652283ed4/scripts/seccomp/spark-io-uring.json) · [Profile used by the launcher](../../runtime/spark-io-uring.json)


## Single-Spark FP4-Local versus the tuned pair

| 8K text service, four slots | One client, output tok/s | Four clients, aggregate output tok/s | GPU J/output token, one / four clients | First delta in ms, one / four clients | Short answers, free / one letter |
| --- | ---: | ---: | --- | --- | --- |
| One Spark, NVIDIA FP4-Local, disk PLE, FP8 KV, MTP 4 | 33.98 | 72.86 | 1.08 / 0.52 | 306 / 253 | 82 / 85 of 112 |
| Two Sparks, NVIDIA mixed FP4, BF16 KV, no drafting | 21.73 | 64.96 | 2.51 / 0.97 | 134 / 353 | 77 / 86 of 112 |

At one client the single-Spark service generated **56.4% more tokens/s** and used **56.8% fewer GPU joules/token**. Its first delta took about 172 ms longer. At four clients its aggregate throughput was **12.2% higher** and energy per token **46.3% lower**. These are workload-level comparisons. The two recipes change the checkpoint, engine, placement of the PLE table, KV precision, prefix caching, drafting and node count together. The single-Spark recipe enables prefix caching. Each prompt includes a distinct case tag. Across the two deployments, all 24 timed requests matched on prompt hash, input tokens and 96-token output count. Each table entry for first delta is the median of the three within-trial medians. [Single-Spark trial settings](fp4-local/settings.jsonl) · [Trials](fp4-local/trials.jsonl) · [Streamed request records](fp4-local/requests.jsonl.gz) · [GPU energy](fp4-local/flash-one-local-energy.jsonl) · [Raw counter samples](fp4-local/flash-one-local-energy.csv.gz)

On the thinking-off short-answer probe, the single Spark gained six correct answers and lost one against the pair. Its free responses used 364 total output tokens and ten hit the 16-token limit. Under a one-uppercase-letter constraint, the single Spark gained four answers and lost five against the pair. All 112 tokenized prompt lengths matched in both comparisons. Within the single-Spark run, constraining the answer recovered three items and kept all 82 previous correct verdicts. The pair retains a one-item lead under this answer contract. [Free responses](fp4-local/quality.jsonl.gz) · [Constrained responses](fp4-local/one-letter.jsonl.gz)

The 102.19 GB checkpoint contains 21 SafeTensors files. B12X reported 71.41 GB selected for loading and 72.38 GB of physical reads during the 132.91-second weight-loading phase. The engine reported a separate 473.13 seconds for profiling, cache creation and warm-up. During inference the worker had seven checkpoint files open with Linux `O_DIRECT`, corresponding to shards 15–21. The two-Spark configuration uses the distinct 132.68 GB release. [Startup counters](fp4-local/startup.json) · [Direct-I/O file observations](fp4-local/direct-io.json)

On the single Spark run `python3 spark.py prepare qwen38-flash-next-fp4-local`, then `python3 spark.py serve qwen38-flash-next-fp4-local`. Use `qwen38-flash-next-fp4-local` for the served model in the same benchmark and quality commands used for the pair. [Single-Spark launcher settings](../../recipes/qwen38-flash-next-fp4-local.json) · [Shared prompt and measurement procedure](../qwen38-27b-precision-2026-11-19/#reproduce)
