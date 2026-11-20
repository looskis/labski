# Nanbeige4.3-33B precision on one DGX Spark

The publisher’s BF16 checkpoint ran on a 128 GB GB10 with four scheduler slots and an 8K context. One client generated 3.71 output tok/s and four clients generated 14.10 aggregate. The same 112-question short-answer probe used for the Qwen precision study returned 63 correct responses. Forty-one responses reached the 16-token cap. [BF16 recipe](../../recipes/nanbeige43-33b-bf16.json) · [Publisher checkpoint](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/tree/df919e1e2cf077a22204c78b6c826bf3db18bc08)

| Checkpoint | Weight GB | Output tok/s at 1 / 4 clients | GPU J/token at 1 / 4 clients | First delta in ms at 1 / 4 clients | Free short answers |
| --- | ---: | --- | --- | --- | --- |
| Official BF16 | 66.24 | 3.71 / 14.10 | 11.14 / 2.54 | 332 / 692 | 63/112 |

The single-Spark vLLM 0.33.0 service used BF16 KV, a 3 GiB KV budget, 2,048-token prefill chunks, disabled prefix caching and the publisher’s chat template with thinking off. Each timing condition had one excluded warm-up followed by three turns of four deterministic prose prompts with a 96-token output budget. Every timed request reached that limit. Rates and GPU energy are medians across the three turns. First delta is the median of the three within-turn medians. [Serving and request settings](bf16/settings.jsonl) · [Throughput trials](bf16/trials.jsonl) · [Raw requests](bf16/requests.jsonl.gz) · [GPU energy](bf16/nanbeige-one-bf16-energy.jsonl) · [NVML samples](bf16/nanbeige-one-bf16-energy.csv.gz)

The free-answer probe scores an exact, completed letter under greedy decoding. Some responses started explaining their answer despite the request for one letter. Nanbeige’s tokenizer uses a metaspace prefix for isolated letters. The strict visible-character control stopped on the first BF16 case. We added a distinct probe that requests and verifies the emitted token ID, scores its allowed A–J choice and saves the actual visible response. [BF16 answers](bf16/quality.jsonl.gz) · [Probe and scoring rules](../../mmlu_probe.py)
