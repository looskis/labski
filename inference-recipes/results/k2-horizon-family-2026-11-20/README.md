# K2 Horizon model size and precision on one DGX Spark

The 7B BF16 model produced **55.00 generated tokens/s at four clients**, compared with **40.81** for 32B NVFP4. It consumed 0.60 versus 0.89 GPU J/token. On the same 139,378 UTF-8 bytes it assigned lower probability to the text, with **0.7140 versus 0.5936 bits/byte**. The small model also exposed a low-effort response parsing issue. Its raw API score and the final letters recovered from its outputs are reported separately below. [Matched-byte method](../../experiments/k2-horizon-family/)

| Checkpoint | Generated tok/s, 1 / 4 | GPU J/token, 1 / 4 | Bits/byte | Open answers / 14, 128 / 512 tokens | First-token choices / 112 | Strict streamed lookups / 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [32B BF16](../k2-horizon-32b-2026-11-20/bf16/) | 3.36 / 13.15 | 11.49 / 2.62 | 0.5692 | 5 / 5 | 21 | 8 |
| [32B FP8](../k2-horizon-32b-2026-11-20/fp8/) | 5.22 / 21.04 | 5.89 / 1.49 | 0.5723 | 5 / 5 | 22 | 8 |
| [32B NVFP4](../k2-horizon-32b-2026-11-20/nvfp4/) | 9.99 / 40.81 | 3.30 / 0.89 | 0.5936 | 7 / 7 | 26 | 8 |
| [7B BF16](7b/) | 12.91 / 55.00 | 2.80 / 0.60 | 0.7140 | 0 / 0 | 23 | 0 |
| [3.7B BF16](3.7b/) | 20.07 / 97.85 | 1.72 / 0.34 | 0.7796 | 1 / 1 | 24 | 7 |

Each endpoint used the same Spark, pinned vLLM 0.33 image, 8K context, 3 GiB KV budget and four slots. Throughput includes reasoning and visible tokens, with four 96-token requests per trial and three trials after warm-up. The 112-question forced choice measures the first token before reasoning. The open-answer score requires a stopped response containing the expected letter in the API content field. The lookup benchmark reads that field from the streamed response. GPU energy uses the cumulative NVML counter. [32B three-precision comparison](../k2-horizon-32b-2026-11-20/)

## The 7B low-effort delimiter

The tokenizer defines three native reasoning delimiters. The low-effort chat template opens `<ifm|think_faster>`. In all eight naturally stopped short answers at the 128-token budget and all 13 at the 512-token budget, the 7B checkpoint closed with `</ifm|think>`. The stock vLLM parser expects the closing marker for the requested effort. The nonstreamed API consequently put both the preceding text and the final letter into the answer field. Strict scoring gave 0/14 at both budgets. Reading only the text after the observed closing marker and requiring a stopped answer with exactly one letter recovered **7/14 at 128 tokens and 9/14 at 512**. Eight streamed lookup replies stopped with empty visible content. These records contain hashes of the original output and the parsed letter. [128-token pairs](7b/effort-low-128.jsonl.gz) · [512-token pairs](7b/effort-low-512.jsonl.gz) · [Stock vLLM parser](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/reasoning/k2_horizon_reasoning_parser.py) · [7B chat template](https://huggingface.co/IFM/K2-Horizon-7B/blob/1ad5223c80f635ede5fb926f6304dc7228246a4b/chat_template.jinja)

On the 3.7B model the stock endpoint reached **97.85 generated tok/s and 0.34 GPU J/token** at four clients, with seven exact streamed lookups. On the 512-token question panel, one returned answer passed strict scoring. Five outputs contained the high-effort closing marker, and counting exact final letters after those delimiters brought the total to four. The 3.7B checkpoint scored 0.7796 bits/byte on the matched text. [3.7B responses](3.7b/effort-low-512.jsonl.gz) · [3.7B throughput](3.7b/trials.jsonl)

A live one-Spark repeat with a parser that accepts all three native K2 closing markers served **7/14, 9/14 and 8/8** on the same two question budgets and streamed lookups. On 16 fresh values in both API modes, the stock service returned 0/32 exact answers and the plugin returned 32/32. All 32 pairs had identical generated token IDs. [Corrected 7B recipe and paired outputs](7b-parser/)

## Reproduce

From `inference-recipes/`, prepare and serve `k2-horizon-7b-bf16`. Follow the 32B throughput, forced-choice, lookup and two open-answer commands with this model name. Use the matched-byte procedure for cross-tokenizer likelihood. [7B launch configuration](../../recipes/k2-horizon-7b-bf16.json) · [Benchmark commands](../k2-horizon-32b-2026-11-20/#reproduce) · [Build and score the byte corpus](../../experiments/k2-horizon-family/#reproduce)
