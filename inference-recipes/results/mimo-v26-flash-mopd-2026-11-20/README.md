# MiMo-V2.6-Flash MOPD on DGX Spark

The 309B model served four 8,192-token slots on one 128 GB Spark from a **105.86 GB Q2_K_S text checkpoint**. Its 48-layer mixture of experts activates eight of 256 experts in each MoE layer. Ordinary decoding delivered **20.23 output tokens/s for one client and 38.72 aggregate for four**. [Engine build and launch](../../experiments/mimo26-rpc/) · [Checkpoint checksums](../../experiments/mimo26-rpc/weights.json)

| Configuration | Output tok/s, 1 / 4 clients | First streamed delta, ms, 1 / 4 | GPU J/output token, 1 / 4 | Free / constrained out of 28 | Exact lookups |
| --- | ---: | ---: | ---: | ---: | ---: |
| [One Spark, Q2_K_S](q2ks-one/) | 20.23 / 38.72 | 419 / 1,071 | 2.59 / 1.50 | 17 / 19 | 8/8 |
| [One Spark, Q2_K_S + MTP2](q2ks-one-mtp2/) | 22.37 / 43.32 | 434 / 1,248 | 2.49 / 1.46 | 17 / 19 | 8/8 |

Each rate is the median of three end-to-end trials. Each trial generated four 96-token prose responses, with an excluded warm-up at each concurrency. One client ran first and four clients followed. GPU energy sums NVML counters on participating ranks over the timed request windows. Each configuration used greedy sampling with reasoning and prompt caching disabled. The checks use a fixed 28-question MMLU-Pro subset, a 16-token free-answer budget, a one-token GBNF letter constraint, and eight 32-record inventory lookups. [Request workload](../../bench.py) · [Answer scoring](../../mmlu_probe.py)

## One-Spark placement

The text model occupied 97.97 GiB in the GPU model buffer. The engine reported 49/49 layers offloaded including the output. Linux reported 15.2 GiB available after the one-client trials. From container start to the first successful health poll took 100.5 s and used 0.286 GPU Wh. This interval began after the files had been verified on disk. [Engine allocations](q2ks-one/startup.json) · [Host and cgroup memory](q2ks-one/memory.json) · [Health timing](q2ks-one/ready.json) · [Startup energy](q2ks-one/startup-energy.json)

## Two-token MTP on one Spark

The second launch added the 2.38 GB Q8_0 draft head to the same Q2_K_S target. End-to-end throughput was **10.6% higher at one client and 11.9% higher at four**. GPU joules/token were 4.1% lower and 2.9% lower respectively. The ordinary launch ran first. The same prompt hashes appeared in both conditions. Visible answer text and output length matched on 1/24 timed prose requests, 27/28 free answers and 8/8 lookups. The two runs selected the same constrained letter on 28/28 items. [Ordinary answers](q2ks-one/quality.jsonl.gz) · [Drafted answers](q2ks-one-mtp2/quality.jsonl.gz) · [Ordinary prose](q2ks-one/requests.jsonl.gz) · [Drafted prose](q2ks-one-mtp2/requests.jsonl.gz)

All 28 free-answer correctness verdicts matched. For each configuration independently, we also compared the 12 matched prose prompts across concurrency levels. Identical answers occurred on 0/12 for ordinary decoding and 0/12 for MTP. The short-answer scores therefore do not establish greedy text equivalence.

The server accepted 1,212 of 2,102 proposed tokens across the 24 timed prose responses (57.7%). [Per-response draft counts](q2ks-one-mtp2/draft-acceptance.json)

## Reproduce

Prepare the pinned weights and use the launch instructions for the desired topology. For the one-Spark draft run, add the Q8_0 MTP sidecar as shown there. From `inference-recipes/` on the serving head, use the same pinned question file and a fresh output directory per run. [Source revision and command line](../../experiments/mimo26-rpc/) · [Prepare the question corpus](../qwen38-27b-precision-2026-11-19/#reproduce)

```sh
model=mimo26-flash-mopd-q2ks
out=./mimo-flash-results
mkdir "$out"
python3 mmlu_probe.py run "$model" /tmp/mmlu-112.jsonl --per-subject 2 --out "$out/free.jsonl"
for c in 1 4; do
  python3 bench.py "$model" --thinking off --workload prose --requests 4 --repeats 3 \
    --concurrency "$c" --max-tokens 96 --out "$out/prose-c$c"
done
python3 bench.py "$model" --thinking off --workload lookup --records 32 --requests 8 \
  --repeats 1 --max-tokens 128 --out "$out/lookup"
python3 mmlu_probe.py run "$model" /tmp/mmlu-112.jsonl --per-subject 2 \
  --one-letter-gbnf --out "$out/grammar.jsonl"
```
