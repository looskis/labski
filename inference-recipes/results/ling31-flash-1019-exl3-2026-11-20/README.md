# Ling 3.1 Flash 1019 on one DGX Spark

**A one-token draft served four simultaneous clients at 46.65 aggregate output tok/s. A two-token draft was fastest for one client at 22.67 tok/s.** Ordinary decoding on the same 100.81 GB EXL3 checkpoint reached 36.74 and 15.77 respectively. All three draft depths scored 15/28 on constrained letters and 4/28 on free responses. The 560B parameter model ran with 8,192 tokens per sequence and a 32,768-token eight-bit KV cache. [Build and serving recipe](../../experiments/ling31-exl3/) · [Weights and versions](build.json)

| Draft tokens | Output tok/s, 1 / 4 clients | Median first streamed delta ms, 1 / 4 | GPU J/output token, 1 / 4 | Proposed tokens accepted, 1 / 4 |
| ---: | --- | --- | --- | --- |
| 0 | 15.77 / 36.74 | 508 / 1,513 | 3.41 / 1.87 | — |
| 1 | 22.34 / 46.65 | 555 / 1,342 | 2.61 / 1.52 | 81.1% / 82.6% |
| 2 | 22.67 / 42.56 | 600 / 1,423 | 2.75 / 1.55 | 58.9% / 56.7% |
| 4 | 20.61 / 34.09 | 665 / 2,505 | 3.12 / 1.87 | 34.5% / 35.1% |

Each concurrency used one excluded warm-up and three trials of four greedy prose prompts. Every timed response reached the 96-token limit. Each table cell reports the median across the three trials, except acceptance, which pools the engine’s accepted and proposed token counts. At four clients, proposing one token gained 27.0% throughput over ordinary decoding. Proposing four cost 7.2%, and the first streamed delta took nearly a second longer. [Ordinary requests and timing](trials.jsonl) · [Depth one](mtp1/) · [Depth two](mtp/) · [Depth four](mtp4/)

## Four-client repeat with all slots warmed

Four fresh service launches ran in depth order 1, 2, 2, 1. A preliminary launch was excluded after the disk counter showed an unrelated checkpoint transfer was still active. The transfer process was suspended for all four reported launches. Each first served a batch of four separate warm-up prompts, then ran the same 12 measured requests across three trials. Median throughput for the four launches was 45.51, 43.43, 45.54 and 47.23 tok/s in order. Pooling six trials per setting gave 46.37 tok/s at depth one and 44.02 at depth two, a 5.3% advantage. Median first delta was 1,430 versus 1,507 ms. GPU energy was close at 1.485 versus 1.470 J/token. The repeated launches narrow the original throughput gap and show run-to-run variation. [Trial and warm-up timings](abba/trials.jsonl) · [Requests](abba/requests.jsonl.gz) · [Power-counter measurements](abba/energy.jsonl) · [Raw NVML counters](abba/nvml.csv.gz)

## Quality and memory

All three draft depths preserved the 28 free-answer verdicts and the eight lookup texts and lengths. Each gained one correct constrained letter over ordinary decoding and lost zero. On the 24 greedy prose responses, five, six and six texts respectively matched ordinary decoding byte for byte at depths one, two and four. The best throughput settings have different visible generation on this corpus. [Ordinary prose](requests.jsonl.gz) · [Depth-one prose](mtp1/requests.jsonl.gz) · [Depth-two prose](mtp/requests.jsonl.gz) · [Depth-four prose](mtp4/requests.jsonl.gz)

The ordinary service loaded all 75 modules in 84.3 seconds and Linux reported 18.5 GiB available after the first single-client trial. At depth two, loading took 81.5 seconds and 14.9 GiB remained available. Across timed requests, depth two accepted 623 of 1,058 proposed tokens at one client and 612 of 1,080 at four. [Ordinary memory](memory.json) · [Depth-two memory](mtp/memory.json) · [Engine-reported acceptance](mtp/startup.json)

## Why the answer format matters

With ordinary decoding, twenty-four of the 28 free responses reached the 16-token cap, often while explaining the problem. The original four correct responses stayed correct under the one-token regex grammar, which recovered ten more. All 28 grammar responses emitted exactly one visible A–J letter and reported one output token. An attempted allowed-token-ID request returned `Let` on a controlled item even though its allowable IDs encoded only uppercase letters. The same prompt under TabbyAPI's `[A-J]` regex returned `H`. The grammar scores use the verified response contract. [Free answers](quality.jsonl.gz) · [Grammar answers](grammar.jsonl.gz) · [Constraint control](constraint-check.json) · [Probe implementation](../../mmlu_probe.py)

Six of eight 524-token table lookups met the exact numeric-answer rule. The other two returned the correct number inside an explanatory sentence. A 3.03 bpw TensorFold checkpoint has an external two-Spark recipe. The one-Spark checkpoint uses 1.0 or 1.5 bpw for routed experts and higher precision elsewhere. This study does not measure the quality difference between those checkpoints. [Raw lookup responses](lookup.json) · [Published two-Spark TensorFold configuration](https://github.com/bertholomus/ling-3.1-tensorfold-tp2-2xgb10/tree/6088f47241c434e4263043a97a25784b03b0d1df)

## Reproduce

Launch the pinned service and prepare the fixed question corpus. Use fresh output paths for each run. Omit `--mtp` for ordinary decoding or pass `--mtp 1`, `--mtp 2` or `--mtp 4` on the service command. For the four-launch repeat, use order 1, 2, 2, 1. Run one extra four-client prose batch with `--seed 94701 --repeats 1` before the standard four-client trials in each launch. [Serving instructions](../../experiments/ling31-exl3/) · [Question preparation](../qwen38-27b-precision-2026-11-19/#reproduce)

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
