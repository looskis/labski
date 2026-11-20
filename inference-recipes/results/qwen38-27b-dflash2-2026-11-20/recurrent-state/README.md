# Qwen3.8 27B recurrent-state precision on one DGX Spark

At a fixed 3 GiB cache budget, changing the recurrent state from FP32 to BF16 raised four-client throughput from **53.73 to 94.78 generated tokens/s** (+76.4%) and cut GPU energy from **0.67 to 0.42 J/output token** (−37.5%). Fresh FP32 and BF16 launches returned the same 28 short answers and the same 16 lookup answers. Both runs used NVIDIA’s mixed NVFP4/FP8 target, the same DFlash2 draft with seven proposals, an 8,192-token request limit and four configured slots. [BF16 recipe](../../../recipes/qwen38-27b-dflash-v033-7-ssmbf16.json) · [FP32 control recipe](../../../recipes/qwen38-27b-dflash-v033-7.json)

| Effective recurrent state | Client count | Aggregate output tok/s | p95 first text delta, ms | GPU J/output token |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 1 | 30.72 | 247 | 1.17 |
| BF16 | 1 | 31.92 | 246 | 1.09 |
| FP32 | 4 | 53.73 | 3,956 | 0.67 |
| BF16 | 4 | 94.78 | 289 | 0.42 |

These are medians of three four-request trials per client count, following an excluded warm-up. The requests had identical prompt hashes and lengths in both runs and generated 96 tokens each with thinking disabled. The BF16 and explicit FP32 controls used separate launches on this Spark, in that order. Energy integrates its GPU counter over the timed request window. [BF16 trials](bf16/trials.jsonl) · [FP32 trials](fp32/trials.jsonl) · [BF16 energy](bf16/qwen-dflash-7-ssmbf16-energy.jsonl) · [FP32 energy](fp32/qwen-dflash-7-ssmfp32-energy.jsonl)

The comparison changed only `--mamba-ssm-cache-dtype` between `float32` and `bfloat16`. vLLM reported 351 GPU blocks of 448 attention tokens apiece, versus 189 blocks of 832 with FP32. Its hybrid-cache capacity estimate increased from 12,002 to 16,815 tokens. Both runs retained the same 8,192-token recurrent-state block setting and 3,221,225,472-byte pool. FP32 admitted at most two clients and queued two in each four-client trial. The first BF16 run’s scheduler sampler polled the wrong port, so the exact number of requests active together is unavailable for that launch. [BF16 engine configuration](bf16/effective-cache.json) · [FP32 engine configuration](fp32/effective-cache.json)

Both fresh launches scored **18/28** at the 16-token answer limit. All 28 visible answers matched, including the errors. Both returned the exact six-digit numbers on all eight 515-token inventory prompts and all eight 7,715-token prompts. On 62 WikiText passages every per-passage NLL sum was numerically identical, with perplexity **6.2840** over 30,939 scored positions. These teacher-forced passages used up to 512 input tokens, below the 2,048-token prefill chunk budget. Across the 24 timed free-form prose requests, zero full continuations matched byte for byte. The drafter used probabilistic proposals. [FP32 short answers](fp32/answers.jsonl.gz) · [BF16 short answers](bf16/answers.jsonl.gz) · [FP32 long retrieval](fp32/long-lookup.json) · [BF16 long retrieval](bf16/long-lookup.json) · [Paired likelihoods](paired-likelihood.json) · [FP32 prose](fp32/requests.jsonl.gz) · [BF16 prose](bf16/requests.jsonl.gz)

The earlier default-FP32 launch reached 52.43 tok/s at four clients and scored 19/28. Relative to that launch the BF16 answers matched on 25/28 questions and lost one correct letter. The subsequent explicit FP32 run reproduced the BF16 answers, so the first comparison alone cannot attribute that lost point to state precision. [Earlier control](../kv-budget/7-kv3-repeat/answers.jsonl.gz) · [Fresh FP32 scheduler samples](fp32/scheduler-trials.jsonl)

## Long-context follow-up fixed before scoring

The 512-token passages fit inside one prefill chunk. To test how state precision behaves across chunk boundaries, we fixed eight disjoint 6,144-token WikiText windows before scoring. Each window spans three nominal 2,048-token chunks. The test compares the target token’s log probability at the same 49,144 positions in FP32 and BF16. It also repeats the first window once per launch to measure within-launch variation. Windows and calls run in the same order with one request at a time. [Fixture preparation and scoring](long_likelihood.py)

Prepare the 62-article corpus with the shared WikiText script and obtain `tokenizer.json` from the pinned Qwen target. From this directory, run the following for each state setting, with `MODEL` set to the active served name. Both files supplied to the preparation command have pinned SHA-256 digests. [WikiText source and preparation](../../../experiments/glm53-nvfp4-scales/#reproduce) · [Qwen tokenizer](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/blob/482ca0f3832238542f8f5295dde86b5f22711d80/tokenizer.json)

```sh
uv run --with tokenizers python long_likelihood.py prepare articles.jsonl tokenizer.json windows.jsonl
python3 long_likelihood.py score "$MODEL" windows.jsonl "$MODEL-long.jsonl.gz"
```

## Reproduce

From `inference-recipes/` on the Spark, prepare and serve the BF16 configuration. The parent experiment gives the matched throughput and short-answer commands. For the long inventory, use `bench.py` with `--thinking off --workload lookup --records 512 --requests 8 --concurrency 4 --max-tokens 48 --repeats 1`. [Matched benchmark commands](../) · [Recorded settings](bf16/settings.jsonl)

```sh
model=qwen38-27b-dflash-v033-7-ssmbf16
python3 spark.py prepare "$model"
python3 spark.py serve "$model"
```
