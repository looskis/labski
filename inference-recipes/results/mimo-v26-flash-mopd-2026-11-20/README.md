# MiMo-V2.6-Flash MOPD on DGX Spark

The 309B model served four 8,192-token slots on one 128 GB Spark from a **105.86 GB Q2_K_S text checkpoint**. Its 48-layer mixture of experts activates eight of 256 experts in each MoE layer. Ordinary decoding delivered **20.23 output tokens/s for one client and 38.72 aggregate for four**. [Engine build and launch](../../experiments/mimo26-rpc/) · [Checkpoint checksums](../../experiments/mimo26-rpc/weights.json)

| Configuration | Output tok/s, 1 / 4 clients | First streamed delta, ms, 1 / 4 | GPU J/output token, 1 / 4 | Free / constrained out of 28 | Exact lookups |
| --- | ---: | ---: | ---: | ---: | ---: |
| [One Spark, Q2_K_S](q2ks-one/) | 20.23 / 38.72 | 419 / 1,071 | 2.59 / 1.50 | 17 / 19 | 8/8 |
| [One Spark, Q2_K_S + MTP1](q2ks-one-mtp1/) | 22.44 / 43.22 | 433 / 1,229 | 2.26 / 1.52 | 17 / 19 | 8/8 |
| [One Spark, Q2_K_S + MTP2](q2ks-one-mtp2/) | 22.37 / 43.32 | 434 / 1,248 | 2.49 / 1.46 | 17 / 19 | 8/8 |
| [One Spark, Q2_K_S + MTP3](q2ks-one-mtp3/) | 19.12 / 39.92 | 435 / 1,181 | 2.94 / 1.62 | 17 / 19 | 8/8 |
| [One Spark, ordinary repeat](q2ks-one-plain-repeat/) | 20.47 / 41.53 | 418 / 1,276 | 2.59 / 1.47 | 17 / 19 | 8/8 |
| [Two Sparks, Q2_K_S](q2ks-two/) | 19.35 / 38.82 | 445 / 1,000 | 3.11 / 1.75 | 17 / 19 | 8/8 |
| [Two Sparks, MXFP4](mxfp4-two/) | 16.49 / 28.00 | 481 / 1,084 | 3.37 / 2.02 | 18 / 19 | 8/8 |

Each rate is the median of three end-to-end trials. Each trial generated four 96-token prose responses, with an excluded warm-up at each concurrency. One client ran first and four clients followed. GPU energy sums NVML counters on participating ranks over the timed request windows. Each configuration used greedy sampling with reasoning and prompt caching disabled. The checks use a fixed 28-question MMLU-Pro subset, a 16-token free-answer budget, a one-token GBNF letter constraint, and eight 32-record inventory lookups. [Request workload](../../bench.py) · [Answer scoring](../../mmlu_probe.py)

## One-Spark placement

The text model occupied 97.97 GiB in the GPU model buffer. The engine reported 49/49 layers offloaded including the output. Linux reported 15.2 GiB available after the one-client trials. From container start to the first successful health poll took 100.5 s and used 0.286 GPU Wh. This interval began after the files had been verified on disk. [Engine allocations](q2ks-one/startup.json) · [Host and cgroup memory](q2ks-one/memory.json) · [Health timing](q2ks-one/ready.json) · [Startup energy](q2ks-one/startup-energy.json)

## Two-token MTP on one Spark

The second launch added the 2.38 GB Q8_0 draft head to the same Q2_K_S target. End-to-end throughput was **10.6% higher at one client and 11.9% higher at four**. GPU joules/token were 4.1% lower and 2.9% lower respectively. The ordinary launch ran first. The same prompt hashes appeared in both conditions. Visible answer text and output length matched on 1/24 timed prose requests, 27/28 free answers and 8/8 lookups. The two runs selected the same constrained letter on 28/28 items. [Ordinary answers](q2ks-one/quality.jsonl.gz) · [Drafted answers](q2ks-one-mtp2/quality.jsonl.gz) · [Ordinary prose](q2ks-one/requests.jsonl.gz) · [Drafted prose](q2ks-one-mtp2/requests.jsonl.gz)

All 28 free-answer correctness verdicts matched. For each configuration independently, we also compared the 12 matched prose prompts across concurrency levels. Identical answers occurred on 0/12 for ordinary decoding and 0/12 for MTP. The short-answer scores therefore do not establish greedy text equivalence.

The server accepted 1,212 of 2,102 proposed tokens across the 24 timed prose responses (57.7%). [Per-response draft counts](q2ks-one-mtp2/draft-acceptance.json)

## How far to draft

The hardware ran ordinary decoding, then MTP2, MTP1 and MTP3. Each draft launch used the same 2.38 GB head. The table counts proposals on the 24 timed prose requests and excludes both warm-ups.

| Draft tokens | Accepted / proposed | Accepted | Correctness verdicts matching ordinary, free / constrained | Identical prose responses to ordinary |
| ---: | ---: | ---: | ---: | ---: |
| [1](q2ks-one-mtp1/draft-acceptance.json) | 983 / 1,279 | 76.9% | 28/28 · 28/28 | 2/24 |
| [2](q2ks-one-mtp2/draft-acceptance.json) | 1,212 / 2,102 | 57.7% | 28/28 · 28/28 | 1/24 |
| [3](q2ks-one-mtp3/draft-acceptance.json) | 1,256 / 3,007 | 41.8% | 28/28 · 28/28 | 0/24 |

One- and two-token proposals delivered nearly identical throughput at both client counts in these trials. Three-token proposals delivered less throughput and used more GPU joules per token than either shorter setting.

We repeated the ordinary launch after MTP3. It reached 20.47 / 41.53 tok/s. The repeat reproduced 12/12 single-client and 3/12 four-client prose responses exactly. Against this second baseline, MTP2 was 9.3% faster at one client and 4.3% faster at four. [Repeated baseline outputs](q2ks-one-plain-repeat/requests.jsonl.gz)

## Routed expert precision on two Sparks

On the same head and direct RoCE worker, ordinary decoding with the 167.37 GB conversion that retained MXFP4 routed experts scored **81/112** constrained answers. The 105.86 GB Q2_K_S conversion scored **81/112**. Switching to MXFP4 gained 7 correct items and lost 7. The Q2_K_S launch ran first. Both used the same device split, concurrency, cache policy and prompt hashes. These two conversions also differ in other stored tensor types. [Q2_K_S full panel](q2ks-two/grammar112.jsonl.gz) · [MXFP4 full panel](mxfp4-two/grammar112.jsonl.gz)

On these two Sparks the smaller conversion delivered 19.35 vs 16.49 output tok/s at one client and 38.82 vs 28.00 at four. That is 38.6% more aggregate throughput at four clients, with GPU energy falling from 2.02 to 1.75 J/token. MXFP4 answered one more of the 28 open-response questions correctly. Both answered all eight exact lookups. The equal 112-item scores mask 14 changes in correctness, so this panel does not establish equivalent accuracy.

The additional Spark mainly made room for the larger checkpoint. With Q2_K_S and ordinary decoding, moving from one Spark to two raised four-client GPU energy from 1.50 to 1.75 J/token and yielded 38.72 vs 38.82 tok/s. For the sampled workload, one Spark with the one-token draft served 22.44 / 43.22 tok/s and left the second machine available. [Single-Spark trials](q2ks-one-mtp1/trials.jsonl) · [Two-Spark Q2_K_S trials](q2ks-two/trials.jsonl) · [Two-Spark MXFP4 trials](mxfp4-two/trials.jsonl)

The two ranks allocated 51,049 / 49,272 MiB in Q2_K_S model buffers and 81,001 / 77,976 MiB for MXFP4. Time from container start to observed readiness was 192 s and 302 s respectively, after all checkpoint files had been staged. During the sampled serving intervals, the head’s ConnectX port recorded 0.49 / 2.76 GB sent / received for Q2_K_S and 0.52 / 2.79 GB for MXFP4. [Q2_K_S allocations](q2ks-two/startup.json) · [MXFP4 allocations](mxfp4-two/startup.json) · [Q2_K_S RDMA counters](q2ks-two/rdma.csv) · [MXFP4 RDMA counters](mxfp4-two/rdma.csv)

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
python3 mmlu_probe.py run "$model" /tmp/mmlu-112.jsonl \
  --one-letter-gbnf --out "$out/grammar112.jsonl"
```
