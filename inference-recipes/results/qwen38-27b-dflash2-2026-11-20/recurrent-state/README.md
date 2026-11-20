# Qwen3.8 27B recurrent-state precision on one DGX Spark

At a fixed 3 GiB cache budget, changing the recurrent state from FP32 to BF16 raised four-client throughput from **52.43 to 94.78 generated tokens/s** and cut GPU energy from **0.68 to 0.42 J/output token**. The BF16 run lost one correct answer on the 28-question panel. Both runs used NVIDIA’s mixed NVFP4/FP8 target, the same DFlash2 draft with seven proposals, an 8,192-token request limit and four configured slots. [BF16 recipe](../../../recipes/qwen38-27b-dflash-v033-7-ssmbf16.json) · [FP32 control recipe](../../../recipes/qwen38-27b-dflash-v033-7.json)

| Effective recurrent state | Client count | Aggregate output tok/s | p95 first text delta, ms | GPU J/output token |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 1 | 30.24 | 251 | 1.15 |
| BF16 | 1 | 31.92 | 246 | 1.09 |
| FP32 | 4 | 52.43 | 3,988 | 0.68 |
| BF16 | 4 | 94.78 | 289 | 0.42 |

These are medians of three four-request trials per client count, following an excluded warm-up. The requests had identical prompt hashes and lengths in both runs and generated 96 tokens each with thinking disabled. The FP32 control is an earlier launch on this Spark. Energy integrates its GPU counter over the timed request window. [BF16 trials](bf16/trials.jsonl) · [FP32 trials](../kv-budget/7-kv3-repeat/trials.jsonl) · [BF16 energy](bf16/qwen-dflash-7-ssmbf16-energy.jsonl) · [FP32 energy](../kv-budget/7-kv3-repeat/qwen-dflash-7-kv3-repeat-energy.jsonl)

The single recipe change was `--mamba-ssm-cache-dtype bfloat16`. vLLM reported 351 GPU blocks of 448 attention tokens apiece, versus 189 blocks of 832 with FP32. Its hybrid-cache capacity estimate increased from 12,002 to 16,815 tokens. Both runs retained the same 8,192-token recurrent-state block setting and 3,221,225,472-byte pool. The first BF16 run’s scheduler sampler polled the wrong port, so the exact number of requests active together is unavailable for that launch. [BF16 engine configuration](bf16/effective-cache.json) · [FP32 engine configuration](../kv-budget/7-kv3-repeat/effective-cache.json)

At a 16-token answer limit the BF16 run scored 18/28 against 19/28 for FP32. The visible answers matched on 25 questions. Question 3682 changed from correct to incorrect and no other correctness verdict changed. Both runs returned all eight exact numbers in the 515-token inventory prompts. BF16 also returned all eight at 7,715 input tokens and scored perplexity 6.2840 on 30,939 teacher-forced next-token predictions across 62 WikiText passages. [BF16 answers](bf16/answers.jsonl.gz) · [FP32 answers](../kv-budget/7-kv3-repeat/answers.jsonl.gz) · [Long retrieval](bf16/long-lookup.json) · [Per-passage negative log-likelihoods](bf16/wikitext.jsonl)

## Reproduce

From `inference-recipes/` on the Spark, prepare and serve the BF16 configuration. The parent experiment gives the matched throughput and short-answer commands. For the long inventory, use `bench.py` with `--thinking off --workload lookup --records 512 --requests 8 --concurrency 4 --max-tokens 48 --repeats 1`. [Matched benchmark commands](../) · [Recorded settings](bf16/settings.jsonl)

```sh
model=qwen38-27b-dflash-v033-7-ssmbf16
python3 spark.py prepare "$model"
python3 spark.py serve "$model"
```
