# Qwen3.8 27B on one Spark

On each of two 128 GB Sparks, NVIDIA’s mixed NVFP4/FP8 checkpoint generated **2.78–2.79× as many output tokens per second as BF16 at one client**, and **2.61–2.67× at four clients**. On the same 112 short-answer questions, BF16 scored 83 on both Sparks. Mixed precision scored 74 and 78. [BF16 recipe](../../recipes/qwen38-27b-bf16.json) · [Mixed-precision recipe](../../recipes/qwen38-27b-nvfp4.json)

| Spark | Precision | Run order | One client, output tok/s | Four clients, aggregate output tok/s | Correct / 112 | Truncated |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | BF16 | 1 | 4.46 | 17.54 | 83 | 3 |
| A | Mixed NVFP4/FP8 | 2 | 12.40 | 45.74 | 74 | 4 |
| B | Mixed NVFP4/FP8 | 1 | 12.84 | 47.42 | 78 | 4 |
| B | BF16 | 2 | 4.61 | 17.77 | 83 | 3 |

The smaller checkpoint cut stored weight bytes by 60.5%, from 55.56 GB to 21.92 GB. Each throughput cell is the median of three trials with four matched prose requests and an excluded warm-up. All 96 timed requests hit the 96-token output budget. Both checkpoints used ARM64 vLLM 0.33.0, driver 580.178.04, BF16 KV, an 8,192-token context, up to four active sequences, disabled prefix caching and a 2,048-token batch limit. Throughput includes prompt processing. [Trial timings](trials.jsonl) · [Request records](prose-requests.jsonl.gz)

**Quality.** On Spark A, ten answers were correct only in BF16 and one only in mixed precision. On Spark B those counts were nine and four. The BF16 verdict agreed on every item between machines. Eight mixed-precision verdicts changed between machines, including six correct only on B and two correct only on A. Identical rendered prompt lengths, question hashes and answer keys were checked for every pair. The experiment used one greedy pass per checkpoint per machine. A later fresh launch on Spark A scored 78/112 in mixed precision, compared with its first 74/112. Six item verdicts changed on the same GPU. [Same-machine repeat](../qwen38-27b-scaling-2026-11-19/#repeating-the-single-spark-run) Each truncation or malformed answer counted as incorrect. [Individual answers and finish reasons](quality.jsonl.gz)

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
