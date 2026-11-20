# Nemotron 3.5 Super VL on one DGX Spark

The official NVFP4 release stores the 120B-parameter vision-language model in 88 weight shards. Each launch used one 128 GB Spark, vLLM 0.33, FP8 KV cache, a 7 GiB cache budget and an 8,192-token request limit. [Checkpoint layout and runtime registration](../../experiments/nemotron35-vl/)

| Launch recipe | Output tok/s, one / four clients | p95 first streamed delta, ms, one / four | Short answers | Text lookups | Image lookups | WikiText perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Ordinary decoding](../../recipes/nemotron35-vl-plain.json) | 18.99 / 57.17 | 229 / 475 | 23/28 | 8/8 | 5/8 | 3.4170 |

Rates are medians of three trials per client count, with four requests per trial and one excluded warm-up. Each timed request asked for at most 96 output tokens. The short-answer panel used a 32-token cap. Text and image lookups required an exact six-digit answer. The vision panel contains eight synthetic 12-row inventory cards. Each image record includes the expected value, the image SHA-256 and the response hash. [Image fixtures](cards/) · [Vision harness](vision_check.py)

## Measurements

- **Ordinary decoding**. [Timing](plain/trials.jsonl) · [Settings](plain/settings.jsonl) · [Individual requests](plain/requests.jsonl.gz) · [Short answers](plain/short.jsonl.gz) · [Text lookups](plain/lookup.json) · [Image lookups](plain/vision.jsonl) · [Per-passage likelihoods](plain/wikitext.jsonl)

## Reproduce

From `inference-recipes/` on the Spark, select a measured recipe and run the following commands. The timing and text panels use the shared benchmark harness. The vision harness sends the eight published PNG files to the local endpoint. [Text benchmark procedure](../../measurement.md)

```sh
model=nemotron35-vl-plain
python3 spark.py prepare "$model"
python3 spark.py serve "$model"
```

From a second terminal, run the image panel.

```sh
model=nemotron35-vl-plain
python3 results/nemotron35-vl-2026-11-20/vision_check.py "$model" vision-scores.jsonl
```
