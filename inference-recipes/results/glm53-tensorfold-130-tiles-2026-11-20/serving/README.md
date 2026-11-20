# TensorFold GLM-5.3-Flash prompt dispatch on two Sparks

On two 128 GB DGX Sparks, the patched TensorFold 1.3.0 engine processed a 32,768-token uncached prompt at **1,424.3 input tokens/s with upstream dispatch and 1,450.4 with the lab tiles** (+1.83%). The test used the 4 bpw EXL3 checkpoint with its 4-bit DFlash2 draft, FP8 KV, one active request and 4,096-token prefill chunks. Both dispatches ran in the same compiled binary. [Build and launch](../engine/) · [All request measurements](requests.jsonl)

| Exact prompt length | Control input tok/s | Tuned input tok/s | Combined change | Control → tuned | Tuned → control¹ | Same first output token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1,252.9 | 1,284.3 | +2.51% | +2.02% | +3.00% | 4/4 |
| 2,048 | 1,414.5 | 1,459.0 | +3.15% | +2.86% | +3.43% | 4/4 |
| 2,560 | 1,357.8 | 1,394.5 | +2.70% | +2.40% | +3.01% | 2/4 |
| 3,072 | 1,340.7 | 1,363.7 | +1.71% | +1.37% | +2.05% | 2/4 |
| 4,096 | 1,297.6 | 1,320.5 | +1.76% | +1.56% | +1.97% | 3/4 |
| 8,192 | 1,367.2 | 1,391.9 | +1.80% | +1.62% | +1.98% | 3/4 |
| 32,768 | 1,424.3 | 1,450.4 | +1.83% | +1.53% | +2.14% | 4/4 |

¹ Each pair’s percentage expresses tuned performance relative to its own control. Four successive fresh service launches used control, tuned, tuned, control order. Each pair saw the same prompts, with a different seed for the second pair. After an excluded 8,192-token warm-up, each launch measured two prompts at each of seven lengths and generated one token per request. Every response reported zero cached prompt tokens. Throughput divides total input tokens by total engine prefill seconds. It excludes queue time and decoding. The request file separately records time to the first streamed token. [Paired prompt hashes and times](pairs.jsonl) · [Exact settings](settings.jsonl) · [Both ranks’ dispatch traces](rank-traces.jsonl)

## Output comparison

On the first pair of launches, both modes returned identical text on all ten short questions and identical mean negative log-likelihood on each of four passages (token-weighted NLL **0.753646**). Those short probes fit below the first patched row count. On the timed long prompts, 22/28 pairs matched on both the first-token ID and the visible text. All six differences occurred in the first pair of launches and the second pair matched on all 14. The original two pairs used different prompts. This design cannot establish how often the same dispatch changes its answer across launches. These two checks exercise different inputs and answer lengths. The synthetic kernel tests used small exactly representable inputs. The serving results give direct evidence about greedy output differences on the full checkpoint. [Short answers and likelihoods](quality.jsonl) · [Measured first-token pairs](pairs.jsonl) · [Upstream quality probe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/blob/d8a7439e8c5aac8645c423ccd24189690359ab69/docs/quality.py)

## Follow-up fixed before collecting repeatability results

Replay all 14 first-pair prompts twice on a new tuned launch and twice on a new control launch, in that order. Compare within-launch repeats, previous same-dispatch launches and cross-dispatch pairs. Also test 24 fresh six-digit inventory queries at exactly 2,048, 4,096 and 8,192 input tokens. Each length has four inventories at each of two key depths. Require a stopped answer containing exactly the expected value and verify zero cached prompt tokens. [Fixed lookup cases and scoring](sized_lookup.py)

## Same-prompt repeatability on two fresh launches

The tuned and stock dispatches selected the **same first token on all 14 prompts in each matched pass**. When we repeated those exact prompts inside either launch, only **6/14** first tokens stayed the same. All six prompts behind the original control/tuned differences also changed between passes inside a single dispatch. The run order reproduced this pattern in both launches. These data establish sensitivity to the pass through the workload. They leave its cause unresolved. [First-token and visible-text comparisons for every pair](repeatability/comparisons.jsonl) · [Raw request hashes, times and cache counts](repeatability/requests.jsonl)

| Same 14 prompts | Stock pass 1 | Stock pass 2 |
| --- | ---: | ---: |
| Tuned pass 1 | 14/14 same | 6/14 same |
| Tuned pass 2 | 6/14 same | 14/14 same |

All 56 timed requests used temperature zero, produced a single token and reported zero cached prompt tokens. Visible-text matches gave the identical table. The tuned launch preceded the stock launch. Across seven prompt lengths the new launches measured **1.75–3.40% faster uncached prefill** with tuning. At 32,768 tokens, throughput was **1,419.3 stock and 1,444.1 tuned input tokens/s**. One earlier startup ended before recording any requests and is excluded. [Per-length rates and all six passes](repeatability/summary.json)

**Long inventory retrieval was poor in both modes.** Each returned the same one exact answer out of 24 with a 32-token output budget. Both scored 1/8 at 2,048 tokens, 0/8 at 4,096 and 0/8 at 8,192. Only three responses per mode contained the expected six digits anywhere as a standalone value. Fourteen tuned and fifteen stock responses hit the output cap. The two modes produced identical visible text on 20/24. This limits the case for either serving configuration on this retrieval workload. [Per-request answers, expected values and stopping reasons](repeatability/lookups.jsonl)

To reproduce, launch the tuned service and run the command below the next heading twice with the same `--prompt-seed labski-tf130-pair1`, using `--run-id replay1` and `--run-id replay2`. Run `python3 sized_lookup.py lookups.jsonl` on that service. Repeat the same sequence in a new directory on a fresh stock launch.

## Reproduce the paired timing

Start the ranks with the published launcher. Run the following on the head for each service launch from this results directory. Set `label` to the arm name and `pair` to 1 or 2. Use both control and tuned dispatch for the same pair. The program adjusts the synthetic input to hit the exact prompt token count. [Launcher](../engine/serve-rank.sh) · [Timing harness](../../glm53-tensorfold-wide-2026-11-18/prefill.py) · [API and binary metadata](api.jsonl)

```sh
python3 ../../glm53-tensorfold-wide-2026-11-18/prefill.py \
  --base http://127.0.0.1:18220 --label "$label" --exact-contexts \
  --contexts 1024,2048,2560,3072,4096,8192,32768 --warmup-tokens 8192 \
  --concurrency 1 --trials 2 --prompt-seed "labski-tf130-pair$pair" \
  --run-id prefill --out "runs/$label"
```
