# Nanbeige4.3-33B precision on one DGX Spark

**On one Spark, the community NVFP4 checkpoint delivered 3.08× BF16 throughput at four clients and consumed 68.4% less GPU energy per generated token.** It scored 60/112 on the free-answer probe against 63/112 for official BF16 and FP8. Official FP8 kept the BF16 total while gaining one question and losing one. [BF16 recipe](../../recipes/nanbeige43-33b-bf16.json) · [FP8 recipe](../../recipes/nanbeige43-33b-fp8.json) · [NVFP4 recipe](../../recipes/nanbeige43-33b-nvfp4.json)

| Checkpoint | Weight GB | Output tok/s at 1 / 4 clients | GPU J/token at 1 / 4 clients | First delta in ms at 1 / 4 clients | Free answers | Constrained A–J |
| --- | ---: | --- | --- | --- | ---: | ---: |
| Official BF16 | 66.24 | 3.71 / 14.10 | 11.14 / 2.54 | 332 / 692 | 63/112 | 83/112 |
| Official FP8 | 35.52 | 6.17 / 24.14 | 5.39 / 1.41 | 177 / 392 | 63/112 | 84/112 |
| Community NVFP4 | 22.07 | 11.10 / 43.50 | 3.09 / 0.80 | 108 / 223 | 60/112 | 81/112 |

The single-Spark vLLM 0.33.0 service used BF16 KV, a 3 GiB KV budget, 2,048-token prefill chunks, disabled prefix caching and the publisher’s chat template with thinking off. Each timing condition had one excluded warm-up followed by three turns of four deterministic prose prompts with a 96-token output budget. Every timed request reached that limit. Rates and GPU energy are medians across the three turns. First delta is the median of the three within-turn medians. [Serving and request settings](bf16/settings.jsonl) · [Throughput trials](bf16/trials.jsonl) · [Raw requests](bf16/requests.jsonl.gz) · [GPU energy](bf16/nanbeige-one-bf16-energy.jsonl) · [NVML samples](bf16/nanbeige-one-bf16-energy.csv.gz)

The free-answer probe scores an exact, completed letter under greedy decoding. Some responses started explaining their answer despite the request for one letter. Nanbeige’s tokenizer uses a metaspace prefix for isolated letters. The strict visible-character control stopped on the first BF16 case. We added a distinct probe that requests and verifies the emitted token ID, scores its allowed A–J choice and saves the actual visible response. [BF16 answers](bf16/quality.jsonl.gz) · [Probe and scoring rules](../../mmlu_probe.py)

The FP8 run used the same Spark and runtime settings after BF16. Its GPU energy per output token fell 51.7% at one client and 44.3% at four. All 24 timing prompts returned 96 tokens. On the free-answer probe 110/112 correctness verdicts and 98/112 visible answers matched BF16. Forty FP8 requests reached the 16-token limit. [FP8 settings](fp8/settings.jsonl) · [Trial timing](fp8/trials.jsonl) · [Requests](fp8/requests.jsonl.gz) · [Energy](fp8/nanbeige-one-fp8-energy.jsonl) · [NVML samples](fp8/nanbeige-one-fp8-energy.csv.gz) · [Free answers](fp8/quality.jsonl.gz)

Restricting the first generated token to A–J raised scores from 63 to 83 on BF16, 63 to 84 on FP8 and 60 to 81 on NVFP4. All 336 responses emitted the recorded ID for the selected letter and rendered as a space followed by that letter. Relative to constrained BF16, FP8 gained two questions and lost one. NVFP4 gained two and lost four. The answer-format change had a larger effect on this diagnostic than the difference between checkpoints. The BF16 constrained pass ran after the other checkpoints on the same Spark with the original BF16 serving settings. [BF16 constrained answers](bf16/one-letter-token.jsonl.gz) · [FP8 constrained answers](fp8/one-letter-token.jsonl.gz) · [NVFP4 constrained answers](nvfp4/one-letter-token.jsonl.gz)

The NVFP4 checkpoint ran 1.80× as fast as FP8 at both client counts. Relative to BF16 it gained three free answers and lost six. Forty-three NVFP4 responses reached the free-answer cap, compared with 41 for BF16. Under the A–J mask, NVFP4 gained one FP8 item and lost four, with 101/112 identical choices. These results describe this 112-question probe and the two answer formats. [NVFP4 settings](nvfp4/settings.jsonl) · [Trial timing](nvfp4/trials.jsonl) · [Requests](nvfp4/requests.jsonl.gz) · [Energy](nvfp4/nanbeige-one-nvfp4-energy.jsonl) · [NVML samples](nvfp4/nanbeige-one-nvfp4-energy.csv.gz) · [Free answers](nvfp4/quality.jsonl.gz) · [Constrained tokens](nvfp4/one-letter-token.jsonl.gz)

## Reproduce on one Spark

From `inference-recipes/`, choose one of `nanbeige43-33b-bf16`, `nanbeige43-33b-fp8` or `nanbeige43-33b-nvfp4`. Run the serving command in a separate terminal. Prepare the fixed 112 questions once using the pinned dataset and script. [Corpus preparation](../qwen38-27b-precision-2026-11-19/#reproduce)

```sh
recipe=nanbeige43-33b-fp8
python3 spark.py prepare "$recipe"
python3 spark.py serve "$recipe"
```

Once the service is ready, set the same recipe name in the benchmark terminal and use a fresh output directory.

```sh
recipe=nanbeige43-33b-fp8
out="./run-$recipe"
mkdir "$out"
python3 mmlu_probe.py run "$recipe" /tmp/mmlu-112.jsonl --out "$out/free.jsonl"
for clients in 1 4; do
  python3 bench.py "$recipe" --workload prose --thinking off --requests 4 --repeats 3 \
    --concurrency "$clients" --max-tokens 96 --out "$out/prose-c$clients"
done
python3 mmlu_probe.py run "$recipe" /tmp/mmlu-112.jsonl --one-letter-token --out "$out/constrained.jsonl"
```
