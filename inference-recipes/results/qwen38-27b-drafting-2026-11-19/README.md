# Qwen3.8 27B drafting on one DGX Spark

**Built-in n-gram drafting accelerated exact code edits by 3.93× at one client and 3.52× at four.** The 24 edits matched the ordinary decoder token for token and all passed the expected Python syntax-tree check. On unrelated prose, n-gram drafting was 3.4% slower at one client and 13.9% slower at four. Each run used the same mixed NVFP4/FP8 target, 8K context, BF16 KV, four scheduler slots and disabled prefix caching. [Run the experiment](../../experiments/qwen-replay/)

| Workload | Clients | Ordinary tok/s | N-gram tok/s | Ordinary GPU J/token | N-gram GPU J/token | Accepted / proposed draft tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated Python edits | 1 | 12.07 | 47.40 | 2.88 | 0.86 | 435 / 540 |
| Repeated Python edits | 4 | 43.57 | 153.20 | 0.79 | 0.29 | 435 / 540 |
| Unrelated prose | 1 | 12.37 | 11.95 | 2.81 | 3.30 | 43 / 5,485 |
| Unrelated prose | 4 | 45.47 | 39.15 | 0.73 | 1.05 | 43 / 5,505 |

The editing workload gave the model a short function and requested one literal change. Four functions each went through three successive edits. The model could copy most of its output from the current prompt. The prose workload used twelve questions on different subjects. The n-gram proposer searched lengths 2–5 and offered up to five tokens. It accepted 80.6% of proposed tokens on edits and under 0.8% on prose. At a single client the median edit request's first token arrived in 196 ms with drafting and 184 ms without it. The estimated mean interval between subsequent tokens fell from 80.61 to 17.32 ms. The gain came during generation. [Editing prompts, scoring and streaming harness](../../experiments/qwen-replay/replay.py)

All twelve prose outputs at each concurrency reached the 96-token budget. Their generated token sequences differed from the ordinary decoder's on every prompt. Prose has no answer key in this replay. The exact-edit result applies to the tested functions. Both services used greedy target decoding. [Ordinary request and trial records](none-turns.jsonl.gz) · [N-gram request and trial records](ngram-turns.jsonl.gz)

Rates are median aggregate output tokens per second across three turns of four requests. GPU J/token sums the NVML differences over the three measured intervals and divides by all generated tokens. It includes prefill and scheduling on one GB10. The trial omits one warm-up request per workload. The checkpoints stayed local, with the lab's checkpoint download on this host paused during timing. [Ordinary GPU counters](none-energy.csv.gz) · [N-gram GPU counters](ngram-energy.csv.gz)

## Suffix drafting from the current request

| Workload | Clients | Output tok/s | GPU J/token | First token, ms | Mean token interval, ms | Accepted / proposed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated Python edits | 1 | 50.74 | 0.70 | 115 | 17.51 | 435 / 480 |
| Repeated Python edits | 4 | 151.76 | 0.28 | 297 | 20.49 | 435 / 480 |
| Unrelated prose | 1 | 12.60 | 2.73 | 124 | 78.26 | 78 / 446 |
| Unrelated prose | 4 | 44.18 | 0.79 | 237 | 84.71 | 61 / 423 |

With zero previous requests retained, suffix drafting delivered 4.20× ordinary output throughput at one client on edits and 3.48× at four. Each of the 24 edits matched ordinary decoding token for token and passed the syntax-tree check. It accepted the same 435 edit tokens per concurrency as the n-gram proposer while requesting 60 fewer draft tokens. The four-client suffix edit turns ranged from 130.45 to 151.98 tok/s. At one client on prose it was 1.9% faster than ordinary decoding, and at four it was 2.8% slower. All 24 prose sequences differed from the ordinary decoder's. Those small throughput differences come from three turns per condition. [Suffix request and trial records](suffix0-turns.jsonl.gz) · [GPU counters](suffix0-energy.csv.gz)

The ARM64 build used Arctic Inference 0.1.1’s CPU suffix tree, loaded into the same vLLM 0.33.0 serving image. The proposer used depth 16 and up to five draft tokens. The current request’s prompt contains a copy of most of each desired function. The model can therefore achieve high edit acceptance with the cross-request history size set to zero. [Build and serving settings](../../experiments/qwen-replay/)

## Does retaining earlier requests help?

| Workload | Clients | Output tok/s with 32-request history | GPU J/token | First token, ms | Mean token interval, ms | Accepted / proposed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated Python edits | 1 | 47.77 | 0.726 | 116 | 19.39 | 426 / 521 |
| Repeated Python edits | 4 | 139.07 | 0.288 | 296 | 22.77 | 428 / 511 |
| Unrelated prose | 1 | 13.33 | 2.473 | 125 | 73.00 | 148 / 697 |
| Unrelated prose | 4 | 45.60 | 0.704 | 249 | 78.47 | 164 / 720 |

Retaining up to 32 requests slowed the editing workload by 5.9% at one client and 8.4% at four relative to current-request-only suffix drafting. It still produced the exact ordinary-decoder token sequence for every edit. The additional history offered more edit tokens but accepted fewer. On prose, history increased single-client throughput by 5.8% against current-request-only suffix and 7.8% against ordinary decoding. At four clients it produced 45.60 tok/s against 44.18 and 45.47, respectively. All 24 history-enabled prose outputs differed from the ordinary decoder's. [32-request history receipts](suffix32-turns.jsonl.gz) · [GPU counters](suffix32-energy.csv.gz)

Each history-enabled workload and concurrency started in a fresh process, received one excluded common warm-up, then ran twelve requests in the same order as the other methods. The 32-request limit exceeded the total number of requests in each replay. Within a four-client turn the server scheduled arrivals concurrently. Runs took place in method order on the same Spark, and the small prose deltas warrant further replication. For this edit task the ordinary prompt already supplied the copyable text. [Reproduce all four methods](../../experiments/qwen-replay/)
