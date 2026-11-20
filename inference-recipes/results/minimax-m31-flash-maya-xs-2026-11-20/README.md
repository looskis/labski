# MiniMax M3.1 Flash Maya-XS on one DGX Spark

**A single 128 GB Spark served the 93.00 GB Maya-XS release and reached 30.25 aggregate output tokens/s across four clients.** With four slots and 7,012 layer–expert pairs resident on the GPU, it ran 26.4% faster and used 8.1% less GPU energy per token than the same four-slot service capped at 90 GPU-resident experts per MoE layer. Both runs scored 14/28 on free answers with identical correctness verdicts, and both passed all eight exact lookups. [Patched ARM64 build and launcher](../../experiments/minimax-maya/) · [Four-slot requests](four-slots-repeat/requests.jsonl.gz) · [Capped requests](four-slots-cap90/requests.jsonl.gz)

| Launch in execution order | GPU-resident layer–expert pairs | Output tok/s, 1 / 4 clients | First streamed delta ms, 1 / 4 | GPU J/output token, 1 / 4 |
| --- | ---: | --- | --- | --- |
| Four slots, automatic placement | 5,383 | 12.70 / 24.65 | 597 / 2,370 | 3.13 / 1.71 |
| One slot, automatic placement | 7,034 | 14.65 / 14.84 | 506 / 10,218 | 2.99 / 3.05 |
| Four slots, automatic placement again | 7,012 | 14.71 / 30.25 | 506 / 1,997 | 3.01 / 1.62 |
| Four slots, 90 experts per MoE layer | 5,130 | 12.42 / 23.93 | 611 / 2,416 | 3.24 / 1.76 |

**Record the resident count at startup.** Two launches with identical four-slot arguments differed by 1,629 GPU-resident layer–expert pairs and delivered 24.65 and 30.25 aggregate tok/s. Maya sizes its resident expert pool using the free CUDA memory at launch. The experiment did not independently vary initial memory or the operating system's file cache. With similar residency, the one-slot and repeated four-slot runs gave nearly the same single-client speed. At four clients the one-slot service queued requests, increasing median first-delta latency to 10.22 seconds. [First four-slot placement](four-slots/startup.json) · [One-slot placement](one-slot/startup.json) · [Repeated four-slot placement](four-slots-repeat/startup.json) · [Capped placement](four-slots-cap90/startup.json) · [Upstream placement code](https://github.com/mw00/project-maya/blob/e75b3ef7c5f725ef50fafe8ae74b16f77982f85e/src/mm3/mm3_model.cu#L1539)

## Model and output quality

The two text shards contain 92.98 GB of mixed GGUF weights, primarily IQ1_S and IQ2_XXS for routed experts. The separate 12.54 MB sparse-attention indexer handles the first three dense layers. All 57 MoE layers retain 128 routed experts each, giving 7,296 layer–expert pairs. Maya reported CUDA placement across all 60 layers, 48 staging slots and an 18-thread CPU expert lane. Each run used BF16 KV, an 8,192-token window per slot, greedy decoding, text-only inference and no drafts. The CPU and GPU both contributed to service. The energy column uses the GB10 NVML counter and does not measure whole-system electricity. [Checkpoint files and pinned hashes](../../experiments/minimax-maya/weights.sha256) · [Four-slot GPU counters](four-slots-repeat/minimax-maya-four-repeat-energy.csv.gz) · [Capped GPU counters](four-slots-cap90/minimax-maya-cap90-energy.csv.gz)

All four runs scored 14/28 on the same zero-shot, 16-token MMLU-Pro subset. Between the repeated four-slot and capped runs, 27/28 visible short answers matched, all 28 correctness verdicts matched, and only 1/24 timed prose responses matched exactly. Every run recovered all eight lookup values. The release publisher reports 5.0% greater next-token loss than the original checkpoint on its evaluation. The local study used only Maya-XS, so the local cost of weight quantization requires a higher-precision comparison. [Repeated short answers](four-slots-repeat/quality.jsonl.gz) · [Capped short answers](four-slots-cap90/quality.jsonl.gz) · [Exact lookups](four-slots-repeat/lookup.json) · [Publisher's loss evaluation](https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF/blob/0ac574071c3322f02b540f3547f9ef1bc48eb494/README.md)

## Reproduce

Build the ARM64 engine and prepare the checkpoint with the linked recipe. Run `python3 serve.py "$work" --slots 4` for automatic placement, `--slots 1` for one slot or `--slots 4 --resident-cap 90` for the capped experiment. Save the CUDA placement line on every launch. Startup includes a calibration of CPU execution versus staging. The portable host kernels passed the recorded dot, RMS and accumulation arithmetic checks before serving. [Build and launch](../../experiments/minimax-maya/) · [Measured CPU arithmetic errors](four-slots-repeat/startup.json)

Run the following from `inference-recipes/` once each service is ready. Use a fresh output directory and prepare the pinned question corpus first. Each prose condition includes a warm-up and three timed trials of four requests. Rates and energy are trial medians. First delta is the median of the three within-trial medians. [Question preparation](../qwen38-27b-precision-2026-11-19/#reproduce) · [First four-slot trials](four-slots/trials.jsonl) · [One-slot trials](one-slot/trials.jsonl) · [Repeated four-slot trials](four-slots-repeat/trials.jsonl) · [Capped trials](four-slots-cap90/trials.jsonl)

```sh
model=minimax-m31-flash-maya-xs
out=./maya-four-slots
mkdir "$out"
python3 mmlu_probe.py run "$model" /tmp/mmlu-112.jsonl --per-subject 2 --out "$out/quality.jsonl"
for clients in 1 4; do
  python3 bench.py "$model" --thinking off --workload prose --requests 4 --repeats 3 \
    --concurrency "$clients" --max-tokens 96 --out "$out/prose-c$clients"
done
python3 bench.py "$model" --thinking off --workload lookup --records 32 --requests 8 \
  --repeats 1 --max-tokens 128 --out "$out/lookup"
```
