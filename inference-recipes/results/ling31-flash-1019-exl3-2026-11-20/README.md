# Ling 3.1 Flash 1019 on one DGX Spark

**One Spark served the 100.81 GB mixed EXL3 checkpoint at 15.77 output tok/s for one client and 36.74 aggregate for four.** The 560B parameter model ran in exllamav3 with 8,192 tokens per sequence and a 32,768-token eight-bit KV cache. On the 28-question diagnostic it scored 4/28 with a free short answer and 14/28 when the server constrained the answer to a single letter. [Build and serving recipe](../../experiments/ling31-exl3/) · [Weights, wheel and runtime versions](build.json)

| Concurrent clients | Output tok/s | Median first streamed delta | GPU J/output token |
| ---: | ---: | ---: | ---: |
| 1 | 15.77 | 508 ms | 3.41 |
| 4 | 36.74 | 1,513 ms | 1.87 |

Each condition used one warm-up and three timed trials of four greedy prose prompts. Every timed response reached the 96-token limit. Medians are across the three trials. The service loaded all 75 model modules in 84.3 seconds on its measured launch. Linux reported 18.5 GiB of available host memory after the first single-client trial. [Request settings](settings.jsonl) · [Prose requests](requests.jsonl.gz) · [Trials](trials.jsonl) · [GPU energy](ling31-single-energy.jsonl) · [NVML counters](ling31-single-energy.csv.gz) · [Memory snapshot](memory.json)

## Why the answer format matters

Twenty-four of the 28 free responses reached the 16-token cap, often while explaining the problem. The original four correct responses stayed correct under the one-token regex grammar, which recovered ten more. All 28 grammar responses emitted exactly one visible A–J letter and reported one output token. An attempted allowed-token-ID request returned `Let` on a controlled item even though its allowable IDs encoded only uppercase letters. The same prompt under TabbyAPI's `[A-J]` regex returned `H`. The grammar scores use the verified response contract. [Free answers](quality.jsonl.gz) · [Grammar answers](grammar.jsonl.gz) · [Constraint control](constraint-check.json) · [Probe implementation](../../mmlu_probe.py)

Six of eight 524-token table lookups met the exact numeric-answer rule. The other two returned the correct number inside an explanatory sentence. A 3.03 bpw TensorFold checkpoint has an external two-Spark recipe. The one-Spark checkpoint uses 1.0 or 1.5 bpw for routed experts and higher precision elsewhere. This study does not measure the quality difference between those checkpoints. [Raw lookup responses](lookup.json) · [Published two-Spark TensorFold configuration](https://github.com/bertholomus/ling-3.1-tensorfold-tp2-2xgb10/tree/6088f47241c434e4263043a97a25784b03b0d1df)

## Reproduce

Launch the pinned exllamav3 and TabbyAPI service and prepare the fixed question corpus. From `inference-recipes/`, use fresh output paths for each run. [Serving instructions](../../experiments/ling31-exl3/) · [Question preparation](../qwen38-27b-precision-2026-11-19/#reproduce)

```sh
model=ling31-exl3-low
out=./ling31-run
mkdir "$out"
python3 mmlu_probe.py run "$model" /tmp/mmlu-112.jsonl --per-subject 2 --out "$out/free.jsonl"
for clients in 1 4; do
  python3 bench.py "$model" --thinking off --workload prose --requests 4 --repeats 3 \
    --concurrency "$clients" --max-tokens 96 --out "$out/prose-c$clients"
done
python3 bench.py "$model" --thinking off --workload lookup --records 32 --requests 8 \
  --repeats 1 --max-tokens 128 --out "$out/lookup"
python3 mmlu_probe.py run "$model" /tmp/mmlu-112.jsonl --per-subject 2 \
  --one-letter-regex --out "$out/grammar.jsonl"
```
