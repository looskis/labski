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

## Independent BF16 repeat

A second BF16 launch reached **104.34 aggregate output tok/s at four clients**, with p95 first streamed delta of **269 ms** and **0.390 GPU J/token**. At one client it reached 30.74 tok/s and 1.148 GPU J/token. All three four-client trials showed four requests running at once. None of the 43 within-trial samples showed a request waiting. In the matched FP32 run, the corresponding maxima were two running and two waiting. The repeat hit 97.1% cache occupancy, with the same 351-block configuration as the first BF16 run. [Repeat trials](bf16-repeat/trials.jsonl) · [GPU energy](bf16-repeat/qwen-dflash-7-ssmbf16-r2-energy.jsonl) · [Scheduler observations](bf16-repeat/scheduler-trials.jsonl) · [Individual samples](bf16-repeat/scheduler-rows.jsonl.gz)

Across the two BF16 launches, the four-client rate ranged from 94.78 to 104.34 tok/s, versus 53.73 for the fresh FP32 control. The repeat again matched every short answer (18/28 correct), all 16 correct lookup strings and finish reasons, and all 62 short-passage NLL sums. None of its 24 full prose continuations matched either previous launch byte for byte. [Short answers](bf16-repeat/answers.jsonl.gz) · [Short retrieval](bf16-repeat/lookup.json) · [Long retrieval](bf16-repeat/long-lookup.json) · [Per-passage likelihoods](bf16-repeat/wikitext.jsonl) · [Free-form prose](bf16-repeat/requests.jsonl.gz)

## Long-context follow-up fixed before scoring

The 512-token passages fit inside one prefill chunk. To test how state precision behaves across chunk boundaries, we fixed eight disjoint 6,144-token WikiText windows before scoring. Each window spans three nominal 2,048-token chunks. The test compares the target token’s log probability at the same 49,144 positions in FP32 and BF16. It also repeats the first window once per launch to measure within-launch variation. Windows and calls run in the same order with one request at a time. [Fixture preparation and scoring](long_likelihood.py)

Prepare the 62-article corpus with the shared WikiText script and obtain `tokenizer.json` from the pinned Qwen target. From this directory, run the following for each state setting, with `MODEL` set to the active served name. Both files supplied to the preparation command have pinned SHA-256 digests. [WikiText source and preparation](../../../experiments/glm53-nvfp4-scales/#score-the-same-text-under-all-three-sets-of-weights) · [Qwen tokenizer](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/blob/482ca0f3832238542f8f5295dde86b5f22711d80/tokenizer.json)

```sh
uv run --with tokenizers python long_likelihood.py prepare articles.jsonl tokenizer.json windows.jsonl
python3 long_likelihood.py score "$MODEL" windows.jsonl "$MODEL-long.jsonl.gz"
```

## Long-context results

On the eight 6,144-token windows, **only 19 of 49,144 target-token log probabilities were exactly equal** across FP32 and BF16 state. Differences appeared within the first nominal 2,048-token chunk as well as in later chunks. Inside each launch, repeating the first window reproduced all 6,143 log probabilities exactly. The 62 short-passage NLL matches therefore do not extend to these longer inputs. [FP32 per-position scores](long-context/fp32.jsonl.gz) · [BF16 per-position scores](long-context/bf16.jsonl.gz) · [Paired results by window and position](long-context/comparison.json)

| Token position, zero-based | Positions scored | Numerically identical | FP32 mean NLL | BF16 mean NLL |
| --- | ---: | ---: | ---: | ---: |
| 1–2,047 | 16,376 | 12 | 1.92489 | 1.91570 |
| 2,048–4,095 | 16,384 | 3 | 1.93305 | 1.93438 |
| 4,096–6,143 | 16,384 | 4 | 2.03872 | 2.03401 |
| All scored positions | 49,144 | 19 | 1.96556 | 1.96137 |

Token-weighted perplexity was **7.1389 for FP32 and 7.1091 for BF16**, a 0.42% decrease on these windows. BF16 had lower mean NLL on four windows and higher mean NLL on four. The windows came from one corpus. The launches also used different attention block sizes as a consequence of the cache setting. This comparison measures the deployed configurations and leaves the source of the numerical differences open. Position 0 in each window has no preceding token and is excluded. The repeated window appears only in the repeatability calculation.

The probability assigned to the observed next token changed by at least a factor of two at **3,080 of 49,144 positions (6.27%)**. BF16 gave that token at least twice as much probability at 1,578 positions, and FP32 did so at 1,502. At a factor of ten, the counts were 169 and 130. These counts describe the probability of the observed token. Determining how often the highest-ranked token changed would require its ID from both runs. [Counts by window and chunk](long-context/comparison.json)

The first long window starts with the exact 512 token IDs of one short passage. In the **same BF16 launch**, scoring those first 511 target positions alone gave a summed NLL of 939.0655. Scoring them as the prefix of the 6,144-token request gave 949.5710. FP32 gave 939.0655 and 943.5751 in separate launches. This matched-prefix check shows that request length can change the reported scores even with the state dtype fixed. It does not identify which kernel or cache behavior caused the difference. [Verified prefix IDs and NLL sums](long-context/shared-prefix.json)

The BF16 scores came from the second BF16 launch, after the timing, retrieval and short-passage measurements. A fresh FP32 launch on the same Spark scored the identical window token IDs in the same order. Both used the seven-proposal recipe and disabled prefix caching. To recompute the tables from this directory, run the following.

```sh
python3 long_likelihood.py compare long-context/fp32.jsonl.gz long-context/bf16.jsonl.gz
python3 long_likelihood.py prefix windows.jsonl bf16-repeat/wikitext.jsonl long-context/bf16.jsonl.gz
```

## Next-token agreement test

We fixed eight prefix lengths per window to test whether the probability changes also change the preferred next token. The follow-up collects vLLM’s first-ranked token IDs at all 49,144 positions. It separately asks for one greedy output token with the prompt cut at 512, 2,047, 2,048, 2,049, 4,095, 4,096, 4,097 and 6,143 tokens. This gives 64 direct decoding comparisons. The first window and its eight prefixes repeat once in each launch. Returned ties are retained. [Test program](next_token.py) · [vLLM log probability ranks](https://docs.vllm.ai/en/v0.33.0/api/vllm/logprobs/)

```sh
python3 next_token.py "$MODEL" windows.jsonl "$MODEL-next-token.jsonl.gz"
```

## Reproduce

From `inference-recipes/` on the Spark, prepare and serve the BF16 configuration. The parent experiment gives the matched throughput and short-answer commands. For the long inventory, use `bench.py` with `--thinking off --workload lookup --records 512 --requests 8 --concurrency 4 --max-tokens 48 --repeats 1`. [Matched benchmark commands](../) · [Recorded settings](bf16/settings.jsonl)

```sh
model=qwen38-27b-dflash-v033-7-ssmbf16
python3 spark.py prepare "$model"
python3 spark.py serve "$model"
```
