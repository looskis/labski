# Serving GLM-5.3-Flash with refitted NVFP4 dense weights

On two 128 GB Sparks we measured NVIDIA’s original checkpoint and the completed dense-prefix variants below. The candidate tensors replace the corresponding tensors before vLLM shards and repacks them. The activation scales, tensor shapes and B12X inference kernels are the same. Each model was launched in a fresh process. Both physical ranks logged the source key count, replacement count and sidecar digest. [Offline fit and matrix errors](../) · [Build the sidecars and serve](../../../experiments/glm53-nvfp4-scales/)

| Weights | NLL, nats/token | Perplexity | Free letters / 28 | Lookups / 8 | Forced letters / 112 | Output tok/s, 1 / 4 clients | Fleet GPU J/token, 1 / 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [NVIDIA](published/) | 0.98802 | 2.6859 | 22 | 8 | 11 | 13.65 / 13.64 | 5.28 / 5.28 |
| [Refit scales](scale_only/) | 0.98200 | 2.6698 | 22 | 8 | 10 | 13.84 / 13.85 | 5.13 / 5.13 |
| [Refit scales and codes](scale_and_codes/) | 0.98162 | 2.6688 | 23 | 7 | 10 | 13.64 / 13.66 | 5.26 / 5.27 |

Each text condition scored the same 62 WikiText-2 raw test article prefixes, up to 511 token predictions per article. The published checkpoint scored 30,938 positions. We use a paired bootstrap over articles for differences in mean negative log likelihood. Lower values indicate a higher probability assigned to the reference tokens. [Likelihood probe](../../../wikitext_probe.py)

The server retained the original 73,728-token, single-sequence configuration with 8,192-token prefill chunks. Four simultaneous clients therefore queued for one execution slot. Each throughput figure is the median of three trials of four matched prose prompts, with a 96-token completion limit and an excluded warm-up. Each rank’s cumulative NVML energy counter was interpolated at the start and end of each trial. The free-answer and record-lookup panels use the checkpoint’s low-effort template. The 112 constrained choices measure the first generated token.

## Refit scales compared with NVIDIA

Mean NLL changed by **-0.00602 nats/token** (95% paired article-bootstrap interval -0.01250 to +0.00033). Perplexity changed by -0.600%. The refit assigned higher total likelihood to 36/62 article prefixes. The interval reflects these prefixes and their lengths.

For free letters, 22/28 visible final answers and finish reasons matched. The refit gained 2 correct answers and lost 2.

For lookup values, 8/8 visible final answers and finish reasons matched. The refit gained 0 correct answers and lost 0.

For constrained first-token letters, 103/112 visible final answers and finish reasons matched. The refit gained 0 correct answers and lost 1.

The key-aware six-digit parser returned the requested value for 8/8 candidate lookups. It accepts either the bare value or the exact requested key followed by an equals sign and the value. The table retains the original strict score. [Parser](../../../experiments/step5-reasoning/lookup_value.py) · [Per-request check](scale_only-lookup-values.jsonl)

## Refit scales and codes compared with NVIDIA

Mean NLL changed by **-0.00640 nats/token** (95% paired article-bootstrap interval -0.01313 to +0.00024). Perplexity changed by -0.637%. The refit assigned higher total likelihood to 34/62 article prefixes. The interval reflects these prefixes and their lengths.

For free letters, 24/28 visible final answers and finish reasons matched. The refit gained 2 correct answers and lost 1.

For lookup values, 7/8 visible final answers and finish reasons matched. The refit gained 0 correct answers and lost 1.

For constrained first-token letters, 100/112 visible final answers and finish reasons matched. The refit gained 2 correct answers and lost 3.

The key-aware six-digit parser returned the requested value for 8/8 candidate lookups. It accepts either the bare value or the exact requested key followed by an equals sign and the value. The table retains the original strict score. [Parser](../../../experiments/step5-reasoning/lookup_value.py) · [Per-request check](scale_and_codes-lookup-values.jsonl)

## What changed when the 4-bit codes were refitted as well?

Holding the original 4-bit codes fixed and changing 6,447,486 scale bytes moved perplexity from 2.6859 to 2.6698. The scale-only run retained all eight strict lookups. Its 22/28 free-answer total included two gains and two losses against the original. Refitting the codes as well moved perplexity a further 0.038%, to 2.6688. The joint-minus-scale-only NLL difference was −0.00038 nats/token, with a paired 95% interval of −0.00681 to +0.00595. On these passages both refits had lower measured perplexity than NVIDIA’s original. All three pairwise NLL intervals include zero. [Joint versus scale-only article comparison](joint-vs-scales-likelihood.json)

## Reproduce

Prepare both local snapshots, generate the dense-prefix sidecars on GB10 and launch the same mode on both ranks. Use `published`, `scale_only` and `scale_and_codes` in separate runs. Prepare the fixed question and article files using the linked procedures. With the endpoint ready, collect the same trials for each mode. [Quantization and launch procedure](../../../experiments/glm53-nvfp4-scales/) · [Question corpus](../../qwen38-27b-precision-2026-11-19/#reproduce)

From `inference-recipes/`, set `URL` to the head endpoint and `RUN` to a fresh output directory for this condition. The 28 free answers and eight lookups each have a 256-token budget. [Generate the eight 1,500-record lookup prompts](../../glm53-effort-2026-11-19/#reproduce-the-effort-comparison)

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
