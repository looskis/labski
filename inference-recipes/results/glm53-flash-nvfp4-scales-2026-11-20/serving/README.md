# Serving GLM-5.3-Flash with refitted NVFP4 dense weights

On two 128 GB Sparks we measured the original NVIDIA checkpoint as the control for the dense-prefix refits. The candidate tensors replace the corresponding tensors before vLLM shards and repacks them. The activation scales, tensor shapes and B12X inference kernels are the same. Each model was launched in a fresh process. Both physical ranks logged the source key count, replacement count and sidecar digest. [Offline fit and matrix errors](../) · [Build the sidecars and serve](../../../experiments/glm53-nvfp4-scales/)

The control assigned a perplexity of **2.6859** to 30,938 token predictions, answered **22/28** free questions correctly at low reasoning effort and retrieved the correct values on **8/8** prompts containing 1,500 records each. The refitted weight comparisons are pending.

| Weights | NLL, nats/token | Perplexity | Free letters / 28 | Lookups / 8 | Forced letters / 112 | Output tok/s, 1 / 4 clients | Fleet GPU J/token, 1 / 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [NVIDIA](published/) | 0.98802 | 2.6859 | 22 | 8 | 11 | 13.65 / 13.64 | 5.28 / 5.28 |

Each text condition scored the same 62 WikiText-2 raw test article prefixes, up to 511 token predictions per article. The published checkpoint scored 30,938 positions. We use a paired bootstrap over articles for differences in mean negative log likelihood. Lower values indicate a higher probability assigned to the reference tokens. [Likelihood probe](../../../wikitext_probe.py)

The server retained the original 73,728-token, single-sequence configuration with 8,192-token prefill chunks. Four simultaneous clients therefore queued for one execution slot. Each throughput figure is the median of three trials of four matched prose prompts, with a 96-token completion limit and an excluded warm-up. Each rank’s cumulative NVML energy counter was interpolated at the start and end of each trial. The free-answer and record-lookup panels use the checkpoint’s low-effort template. The 112 constrained choices measure the first generated token.

## Reproduce

Prepare both local snapshots, generate the dense-prefix sidecars on GB10 and launch the same mode on both ranks. Use `published`, `scale_only` and `scale_and_codes` in separate runs. Prepare the fixed question and article files using the linked procedures. With the endpoint ready, collect the same trials for each mode. [Quantization and launch procedure](../../../experiments/glm53-nvfp4-scales/) · [Question corpus](../../qwen38-27b-precision-2026-11-19/#reproduce)

From `inference-recipes/`, set `URL` to the head endpoint and `RUN` to a fresh output directory for this condition. The 28 free answers and eight lookups each have a 256-token budget. [Generate the eight 1,500-record lookup prompts](../../glm53-effort-2026-11-19/#reproduce)

```sh
model=glm53-nvfp4-prefill
mkdir -p "$RUN"
python3 glm_effort.py "$model" "$CORPUS" --url "$URL" --effort low --per-subject 2 --max-tokens 256 --out "$RUN/answers.jsonl"
python3 glm_effort.py "$model" "$LOOKUPS" --url "$URL" --effort low --per-subject 8 --max-tokens 256 --out "$RUN/lookup.jsonl"
python3 mmlu_probe.py run "$model" "$CORPUS" --url "$URL" --one-letter-token --out "$RUN/forced112.jsonl"
for c in 1 4; do
  python3 bench.py "$model" --endpoint "$URL" --thinking default --workload prose --concurrency "$c" --requests 4 --repeats 3 --max-tokens 96 --out "$RUN/prose-c$c"
done
python3 wikitext_probe.py score "$model" articles.jsonl "$RUN/wikitext.jsonl" --url "$URL"
```
