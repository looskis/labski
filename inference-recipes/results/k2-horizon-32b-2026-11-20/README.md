# K2 Horizon 32B on one DGX Spark

The Institute of Foundation Models publishes BF16, FP8 and NVFP4 checkpoints of K2 Horizon 32B. The lab used native vLLM 0.33.0 on GB10 with the K2 Horizon reasoning parser, low reasoning effort, four 8,192-token slots and a 3 GiB KV pool. All runs used the same prompts and greedy decoding. [BF16 configuration](../../recipes/k2-horizon-32b-bf16.json) · [FP8 configuration](../../recipes/k2-horizon-32b-fp8.json) · [NVFP4 configuration](../../recipes/k2-horizon-32b-nvfp4.json)

| Checkpoint | Weights, GB | Output tok/s, 1 / 4 clients | GPU J/token, 1 / 4 | First streamed delta, ms, 1 / 4 | Lookups | WikiText prefix perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [NVFP4](nvfp4/) | 23.25 | 9.99 / 40.81 | 3.30 / 0.89 | 262 / 382 | 8/8 | 6.7826 |

Throughput uses three timed trials at each concurrency after a warm-up. Each trial generated four 96-token prose continuations. Time to first delta includes visible content or reasoning. GPU energy integrates the Spark’s NVML counter over each timed trial.

NVFP4 returned visible prose in 24/24 timed responses. [Requests](nvfp4/requests.jsonl.gz) · [Trial summaries](nvfp4/trials.jsonl) · [GPU energy](nvfp4/horizon32-nvfp4-energy.jsonl)

## Answer budget and token likelihood

The open-response diagnostic uses one question from each of the 14 MMLU-Pro subjects. We score an exact final letter only when generation stops normally. The separate 112-item diagnostic constrains the very first generated token to one of ten letter IDs. Each test sends a single question at a time. [Scoring procedure](../../mmlu_probe.py)

| Checkpoint | Free correct / 14 at 128 / 512 tokens | Capped answers at 128 / 512 | Tokens used at 128 / 512 | Seconds for 14 at 128 / 512 | First-token letters correct / 112 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVFP4 | 7 / 7 | 3 / 1 | 505 / 1201 | 52.2 / 122.4 | 26 |

Increasing the NVFP4 answer budget from 128 to 512 gained 0 correct and lost 0 correct on the same 14 questions. The forced letters appeared in the API’s visible answer on 112/112 items. [128-token answers](nvfp4/effort-low-128.jsonl.gz) · [512-token answers](nvfp4/effort-low-512.jsonl.gz) · [Constrained letters](nvfp4/grammar112.jsonl.gz)

The likelihood probe scores up to 511 next-token predictions per article on the opening body text of all 62 WikiText-2 raw test articles. It includes every ground-truth token after the first and archives a token sequence hash per article. NVFP4 scored 30,935 positions with a mean negative log likelihood of 1.91436 nats/token. [Per-article likelihood](nvfp4/wikitext.jsonl)

## Reproduce

From `inference-recipes/`, use the pinned runtime and checkpoint for each precision. Prepare the lab’s question corpus and the WikiText article file using the linked procedures. Use a fresh output directory per precision. [Checkpoint staging](../../README.md) · [MMLU-Pro corpus](../qwen38-27b-precision-2026-11-19/#reproduce) · [WikiText preparation](../../experiments/glm53-nvfp4-scales/#score-the-same-text-under-all-three-sets-of-weights)

```sh
model=k2-horizon-32b-nvfp4
python3 spark.py serve "$model"
```

With the endpoint ready, run the following in a second shell.

```sh
model=k2-horizon-32b-nvfp4
mkdir -p "$RUNS"
for c in 1 4; do
  python3 bench.py "$model" --concurrency "$c" --requests 4 --repeats 3 --thinking default --max-tokens 96 --out "$RUNS/prose-c$c"
done
python3 mmlu_probe.py run "$model" "$CORPUS" --one-letter-token --out "$RUNS/grammar112.jsonl"
for budget in 128 512; do
  python3 mmlu_probe.py run "$model" "$CORPUS" --reasoning-effort low --per-subject 1 --max-tokens "$budget" --out "$RUNS/effort-low-$budget.jsonl"
done
python3 bench.py "$model" --workload lookup --records 32 --requests 8 --repeats 1 --concurrency 1 --thinking default --max-tokens 512 --out "$RUNS/lookup"
python3 wikitext_probe.py score "$model" articles.jsonl "$RUNS/wikitext.jsonl" --url http://127.0.0.1:8000
```
