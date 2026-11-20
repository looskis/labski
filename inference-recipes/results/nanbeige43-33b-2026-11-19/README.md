# Nanbeige4.3-33B precision on one DGX Spark

**The official FP8 release generated 1.66× as many tokens per second as BF16 at one client and 1.71× at four.** Both scored 63/112 on the free-answer probe, with one gain and one loss. [FP8 recipe](../../recipes/nanbeige43-33b-fp8.json)

The publisher’s BF16 checkpoint ran on a 128 GB GB10 with four scheduler slots and an 8K context. One client generated 3.71 output tok/s and four clients generated 14.10 aggregate. The same 112-question short-answer probe used for the Qwen precision study returned 63 correct responses. Forty-one responses reached the 16-token cap. [BF16 recipe](../../recipes/nanbeige43-33b-bf16.json) · [Publisher checkpoint](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/tree/df919e1e2cf077a22204c78b6c826bf3db18bc08)

| Checkpoint | Weight GB | Output tok/s at 1 / 4 clients | GPU J/token at 1 / 4 clients | First delta in ms at 1 / 4 clients | Free short answers |
| --- | ---: | --- | --- | --- | --- |
| Official BF16 | 66.24 | 3.71 / 14.10 | 11.14 / 2.54 | 332 / 692 | 63/112 |
| Official FP8 | 35.52 | 6.17 / 24.14 | 5.39 / 1.41 | 177 / 392 | 63/112 |

The single-Spark vLLM 0.33.0 service used BF16 KV, a 3 GiB KV budget, 2,048-token prefill chunks, disabled prefix caching and the publisher’s chat template with thinking off. Each timing condition had one excluded warm-up followed by three turns of four deterministic prose prompts with a 96-token output budget. Every timed request reached that limit. Rates and GPU energy are medians across the three turns. First delta is the median of the three within-turn medians. [Serving and request settings](bf16/settings.jsonl) · [Throughput trials](bf16/trials.jsonl) · [Raw requests](bf16/requests.jsonl.gz) · [GPU energy](bf16/nanbeige-one-bf16-energy.jsonl) · [NVML samples](bf16/nanbeige-one-bf16-energy.csv.gz)

The free-answer probe scores an exact, completed letter under greedy decoding. Some responses started explaining their answer despite the request for one letter. Nanbeige’s tokenizer uses a metaspace prefix for isolated letters. The strict visible-character control stopped on the first BF16 case. We added a distinct probe that requests and verifies the emitted token ID, scores its allowed A–J choice and saves the actual visible response. [BF16 answers](bf16/quality.jsonl.gz) · [Probe and scoring rules](../../mmlu_probe.py)

The FP8 run used the same Spark and runtime settings after BF16. Its GPU energy per output token fell 51.7% at one client and 44.3% at four. All 24 timing prompts returned 96 tokens. On the free-answer probe 110/112 correctness verdicts and 98/112 visible answers matched BF16. Forty FP8 requests reached the 16-token limit. [FP8 settings](fp8/settings.jsonl) · [Trial timing](fp8/trials.jsonl) · [Requests](fp8/requests.jsonl.gz) · [Energy](fp8/nanbeige-one-fp8-energy.jsonl) · [NVML samples](fp8/nanbeige-one-fp8-energy.csv.gz) · [Free answers](fp8/quality.jsonl.gz)

Under the ten-token allowed-choice mask, FP8 scored **84/112**. It recovered 22 free-answer errors and lost one correct answer. Every constrained response rendered as a space followed by the selected letter, and every returned token ID matched that letter. This is a separate output contract. [Constrained FP8 tokens and visible responses](fp8/one-letter-token.jsonl.gz)
