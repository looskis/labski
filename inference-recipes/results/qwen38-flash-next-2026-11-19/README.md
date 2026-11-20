# Qwen3.8 Flash-Next on a direct Spark pair

NVIDIA’s 132.68 GB mixed checkpoint served text on two 128 GB DGX Sparks with upstream vLLM 0.33.0, BF16 KV and no speculative decoding. A single client received **21.34 output tokens/s**. Four concurrent clients received **21.87 aggregate output tokens/s** with one scheduler slot, so the extra requests queued. Both physical RoCE functions on the direct link carried data. [Pinned inference recipe](../../recipes/qwen38-flash-next-nvfp4.json) · [RDMA byte counters](rdma.csv)

| Clients | Median aggregate output tok/s | Trial range | Median combined GPU J/output token |
| ---: | ---: | ---: | ---: |
| 1 | 21.34 | 21.22–21.60 | 2.52 |
| 4 | 21.87 | 20.66–22.06 | 2.46 |

The quality probe scored **75/112** in a single thinking-off, greedy pass. Fifteen answers hit the 16-token output limit. The task requires only a letter, and some responses began explaining the solution before returning one. This particular probe scores exact format and a limited answer budget. It does not estimate the model's leaderboard score. Each timed prose trial used four 41-token prompts and a 96-token output budget. All 24 requests reached the budget. Energy covers both GB10 counters, from request submission through receipt of the final response. [All quality answers](quality.jsonl.gz) · [Prose timings and answers](requests.jsonl.gz) · [Trial timings](trials.jsonl) · [Energy per trial](flash-next-tp2-energy.jsonl) · [Raw energy samples](flash-next-tp2-energy.csv.gz)

The checkpoint uses NVFP4 routed experts and an FP8 PLE table. Both rank logs reported 37.44 GiB for the model-loading allocation. Rank zero reported 27.79 GiB available for KV cache under the 0.8 utilization target. These are engine accounting figures in a shared-memory machine. The CPU-side embedding, operating system, file cache and other allocations also consume physical memory. Benchmarking overlapped host-side distribution of a separate container image. The native compilation job on rank zero was paused during the timing trials. [Complete settings](settings.jsonl)

## Reproduce

Prepare and serve `qwen38-flash-next-nvfp4` on a direct ConnectX pair, with rank zero as the client. Use the pinned ARM64 image, the recipe’s 8,192-token window, one scheduler slot, a 2,048-token batch cap and eager execution. Prepare the 112-item corpus using the Qwen 27B instructions, then run its quality and prose commands with `MODEL=qwen38-flash-next-nvfp4`. For energy, start the NVML sampler on each rank before the first timed trial. [Distributed launcher](../../README.md#one-endpoint-on-multiple-sparks) · [Exact prompts and benchmark commands](../qwen38-27b-precision-2026-11-19/#reproduce) · [Energy capture](../../measurement.md#measure-energy-on-gb10)

NVIDIA has also published a one-Spark configuration for a separate 102 GB FP4-Local checkpoint with its PLE table on disk and four-token MTP. Its active-decode benchmark uses different weights, settings and timing boundaries. [NVIDIA single-Spark deployment](https://github.com/NVIDIA/local-ai-inference-recipes/tree/bb515086841e7a81e7d51925561338e7dee35023/models/qwen/qwen3.8-flash-next/dgx-spark/linux/vllm/fp4-local)
