# Qwen3.8 27B on one Spark

NVIDIA's mixed NVFP4/FP8 checkpoint delivered **12.84 output tokens/s at one client** against **4.46 for BF16**. At four clients the aggregate rates were **47.42 and 17.54 output tokens/s**. Mixed precision also answered five fewer questions correctly on a 112-item, short-answer MMLU-Pro diagnostic. Both configurations completed the workload on one 128 GB Spark. [BF16 recipe](../../recipes/qwen38-27b-bf16.json) · [Mixed-precision recipe](../../recipes/qwen38-27b-nvfp4.json)

| Configuration | Root weights | One client, output tok/s | Four clients, aggregate output tok/s | Correct / 112 | Truncated answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16, Spark A | 55.56 GB | 4.46 | 17.54 | 83 (74.1%) | 3 |
| Mixed NVFP4/FP8, Spark B | 21.92 GB | 12.84 | 47.42 | 78 (69.6%) | 4 |

The smaller checkpoint cut stored weight bytes by 60.5%. Batch throughput increased 2.88× at C1 and 2.70× at C4. Each cell is the median of three trials with four matched prose requests and an excluded warm-up. Every timed request hit the 96-token output budget. The checkpoints ran on separate GB10 units in this first pass. Each used ARM64 vLLM 0.33.0, driver 580.178.04, BF16 KV, an 8,192-token context, up to four active sequences, disabled prefix caching and the same 2,048-token batch limit. The benchmark measures client-observed throughput including prompt processing. [Per-trial timings](trials.jsonl)

**Same-device crossover.** After the first pass the lab swapped checkpoints. At one client, Spark A delivered 4.46 tok/s in BF16 and 12.40 in mixed precision (2.78×). Spark B delivered 4.61 and 12.84 (2.79×) with the run order reversed. These ratios compare each physical machine against itself. Each rate is the median of three four-request trials. [Crossover medians](crossover-c1.csv)

**Quality.** The paired outcome was 74 both correct, 25 both incorrect, nine BF16-only correct and four mixed-only correct. The observed difference is −4.5 percentage points on this sample. The two-sided exact McNemar test on the 13 discordant answers gives p = 0.267. This sample cannot establish a population-wide loss or equivalence. Each question had one vote, including truncations and formatting failures. [Per-question answers, prompt hashes and finish reasons](quality.jsonl.gz)

## Reproduce

Use the [one-Spark launch instructions](../../README.md#first-completion-on-one-spark). Run the commands below from `inference-recipes` on the API host, once per served model. The timing and quality workloads both use greedy decoding with the Qwen chat template's thinking mode disabled.

```sh
MODEL=qwen38-27b-bf16   # or qwen38-27b-nvfp4
for clients in 1 4; do
  python3 bench.py "$MODEL" --thinking off --workload prose --requests 4 \
    --concurrency "$clients" --repeats 3 --max-tokens 96 --out "runs/$MODEL-c$clients"
done
```

The quality probe selects eight test items from each of 14 subjects using a stable hash of each question ID. It scores the exact option letter with a 16-token limit. This is a zero-shot, thinking-off diagnostic. The leaderboard uses a different evaluation and lists Qwen's highest reasoning effort. Preparing the corpus requires PyArrow. Running it requires only Python's standard library. The generated 112-item JSONL has SHA-256 `7fe1d37bf67bd45046cd4f71333011f9024a1ba49e86ffe8875de867b3225e8a`. [Pinned dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be) · [Probe](../../mmlu_probe.py)

```sh
curl -fL -o /tmp/mmlu-pro.parquet \
  https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/b189ec765aa7ed75c8acfea42df31fdae71f97be/data/test-00000-of-00001.parquet
uv run --with pyarrow python mmlu_probe.py prepare /tmp/mmlu-pro.parquet --out /tmp/mmlu-112.jsonl
python3 mmlu_probe.py run "$MODEL" /tmp/mmlu-112.jsonl --out "runs/$MODEL-answers.jsonl"
```
