# DFlash2 with Qwen3.8 27B on one DGX Spark

A single 128 GB Spark served NVIDIA’s 21.92 GB mixed NVFP4/FP8 target with z-lab’s 3.85 GB DFlash2 drafter using stock ARM64 vLLM 0.33.0. At seven proposed tokens per step it generated **30.90 output tok/s for one client and 48.20 aggregate for four**. The median time to first streamed delta increased from 232 ms at one client to 2.10 s at four. Both model revisions and the runtime digest are pinned. [Three-token recipe](../../recipes/qwen38-27b-dflash-v033-3.json) · [Seven-token recipe](../../recipes/qwen38-27b-dflash-v033-7.json) · [Target checkpoint](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/tree/482ca0f3832238542f8f5295dde86b5f22711d80) · [Draft checkpoint](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4)

With default FP32 recurrent state and a fixed **3 GiB KV budget**, seven proposals were fastest for a single client (30.90 tok/s), while three were fastest for four clients (**86.54 aggregate tok/s**, 0.42 GPU J/token). Three proposals delivered 1.80× the four-client throughput of seven and reduced the median first streamed delta from 2.10 s to 309 ms. All three settings scored 19/28 on the short-answer panel and returned eight exact lookups. With 5.5 GiB of KV memory, seven proposals reached 93.86 tok/s at four clients with a 296 ms first delta and the same 19/28 questions correct as the seven-proposal 3 GiB control. At 9 GiB the engine reached 92.29 tok/s with a 264 ms first delta and 18/28 correct. [KV budget comparison](kv-budget/)

| Proposed tokens | Output tok/s, 1 / 4 clients | GPU J/output token, 1 / 4 | First delta, ms, 1 / 4 | Free answers / 28 | Exact lookups / 8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| [0](0/) | 12.06 / 44.44 | 2.76 / 0.83 | 147 / 262 | 19 | 8 |
| [3](3/) | 25.34 / 86.54 | 1.46 / 0.42 | 174 / 309 | 19 | 8 |
| [7](7/) | 30.90 / 48.20 | 1.16 / 0.80 | 232 / 2105 | 19 | 8 |

Each concurrency used three timed trials of four 96-token prose continuations after a warm-up. Reported rates and first-delta times are medians of trial summaries. GPU energy uses differences in the timestamped cumulative NVML energy counter over each trial. The target used BF16 KV, an 8K request limit, four scheduler slots, 3 GiB of KV memory and disabled prefix caching. The questions used greedy, thinking-off decoding with a 16-token answer budget. Each lookup asked for one value among 32 records.

At one client, seven proposals produced **2.56×** the ordinary decoding rate and used **58.1% less GPU energy per generated token**. At four clients the median first streamed delta was 8.0× longer.

Within the 0-proposal condition, 0/12 prose prompts gave identical visible continuations when replayed at one and four clients. Both runs used temperature zero.

Within the 3-proposal condition, 0/12 prose prompts gave identical visible continuations when replayed at one and four clients. Both runs used temperature zero.

Within the 7-proposal condition, 0/12 prose prompts gave identical visible continuations when replayed at one and four clients. Both runs used temperature zero.

Against ordinary decoding at the same 3 GiB KV budget, 3 draft tokens changed one-client throughput by +110.1% and four-client throughput by +94.7%. 0/24 prose continuations had identical visible text and finish reasons. 26/28 short answers had identical visible text and finish reasons. 8/8 lookup answers had identical visible text and finish reasons. The short-answer panel gained 1 correct and lost 1.

Against ordinary decoding at the same 3 GiB KV budget, 7 draft tokens changed one-client throughput by +156.2% and four-client throughput by +8.5%. 0/24 prose continuations had identical visible text and finish reasons. 26/28 short answers had identical visible text and finish reasons. 8/8 lookup answers had identical visible text and finish reasons. The short-answer panel gained 1 correct and lost 1.

## Draft acceptance

With 3 proposals, vLLM verified 912 draft blocks and accepted 1,720 of 2,736 proposed tokens (62.9%). The fraction of all blocks that accepted each successive position was 83.8%, 62.4%, 42.4%. The counters cover the server lifetime, including warm-ups, prose, questions and lookups. [Engine counters](3/spec-metrics.json) · [Timing and quality archive](3/)

With 7 proposals, vLLM verified 761 draft blocks and accepted 1,886 of 5,327 proposed tokens (35.4%). The fraction of all blocks that accepted each successive position was 80.9%, 58.7%, 38.4%, 24.0%, 19.1%, 15.1%, 11.6%. The counters cover the server lifetime, including warm-ups, prose, questions and lookups. [Engine counters](7/spec-metrics.json) · [Timing and quality archive](7/)

## KV capacity and output agreement

At the same 3 GiB KV budget the engine reported 36,278 token slots with ordinary decoding, 17,264 with three proposals and 12,002 with seven. For the three-proposal run, polling the engine every 250 ms during the four-client trials observed all four requests running, zero waiting and peak KV use of 92.3% in each repeat. The prompts and their 96-token generations were shorter than the configured 8K maximum. [Ordinary capacity](0/kv-capacity.json) · [Three-proposal capacity](3/kv-capacity.json) · [Seven-proposal capacity](7/kv-capacity.json) · [Three-proposal scheduler observations](3/scheduler-c4.jsonl.gz) · [Metrics sampler](sample_metrics.py)

Changing concurrency alone changed the text on every 96-token prose prompt even at temperature zero with ordinary decoding. Retokenizing all visible outputs with the pinned target tokenizer gave 96 tokens each. The median shared prefix between the one- and four-client runs was 32 tokens in each of the three modes. Against ordinary decoding at four clients, three proposals shared a median 31 tokens and seven shared 16. These open-ended continuations had no reference answer. [Per-case prefixes](prose-prefixes.jsonl) · [Comparison code](compare_prose.py) · [Pinned target tokenizer](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/blob/482ca0f3832238542f8f5295dde86b5f22711d80/tokenizer.json)

## Reproduce

Run these commands from `inference-recipes/` on a Spark. The launcher stages both checkpoints and resolves the image to its recorded digest. The lab used NVIDIA driver 580.178.04.

```sh
model=qwen38-27b-dflash-v033-3
python3 spark.py prepare "$model"
python3 spark.py serve "$model"
```

With the endpoint ready, run the following in a second shell. Set `CORPUS` to the prepared 112-question JSONL file. [Question selection and preparation](../qwen38-27b-precision-2026-11-19/#reproduce)

```sh
model=qwen38-27b-dflash-v033-3
for c in 1 4; do
  python3 bench.py "$model" --thinking off --requests 4 --repeats 3 --max-tokens 96 \
    --concurrency "$c" --out "runs/$model/prose-c$c"
done
python3 mmlu_probe.py run "$model" "$CORPUS" --per-subject 2 --max-tokens 16 --out "runs/$model/answers.jsonl"
python3 bench.py "$model" --thinking off --workload lookup --records 32 --requests 8 \
  --max-tokens 128 --concurrency 4 --repeats 1 --out "runs/$model/lookup"
```
