# K2 Horizon MoVA 36B A4B on one DGX Spark

The official 48.36 GB FP8 checkpoint served four concurrent requests on one 128 GB GB10. At four clients it delivered **18.51 aggregate output tokens/s** and used **2.12 GPU joules per output token**. For an interactive client the corresponding figures were **5.38 tokens/s** and **7.01 GPU J/token**. The checkpoint quantizes the feed-forward routed experts in layers 3–47. It retains BF16 for the MoVA value experts, attention, shared experts, dense layers, embeddings and output head. [Pinned recipe](../../recipes/k2-horizon-mova-36b-fp8.json) · [Publisher model card and quantization method](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B-FP8/blob/deaeefb1bb9e0c080be6d8844f3b222f90fd4bc7/README.md)

| Concurrent clients | Output tokens/s | Median first streamed delta | P95 first streamed delta | GPU J/output token |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.38 | 533 ms | 536 ms | 7.01 |
| 4 | 18.51 | 806 ms | 909 ms | 2.12 |

Each entry is the median of three trial statistics. Each trial served four distinct prompts with a 96-token output cap. All 24 timed requests reached the cap. GPU energy uses cumulative device counters over each trial wall-clock interval. The Spark ran vLLM 0.33.0 with an 8,192-token context, four sequence slots, 3 GiB of KV memory, and prefix caching disabled. Its runtime selected FlashAttention for attention and DeepGEMM for the routed FP8 MoE. [Timing](fp8/trials.jsonl) · [Individual requests](fp8/requests.jsonl.gz) · [Benchmark configuration](fp8/settings.jsonl) · [GPU energy](fp8/k2-mova-fp8-energy.jsonl) · [Raw energy samples](fp8/k2-mova-fp8-energy.csv.gz)

On the lab’s 112-question MMLU-Pro diagnostic, constraining the first output token to an answer letter yielded **19/112**. The model had one token in which to choose. The lab’s low-effort, deterministic application protocol differs from IFM’s published high-effort sampling evaluation. [Questions, labels and per-item results](fp8/grammar112.jsonl.gz) · [Publisher evaluation settings](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B-FP8/blob/deaeefb1bb9e0c080be6d8844f3b222f90fd4bc7/README.md)

## Reproduce

Run these commands from `inference-recipes/` on one Spark. Use the matched K2 32B benchmark procedure with the MoVA served name for the two client counts and diagnostic panels. [Throughput, question and retrieval commands](../k2-horizon-32b-2026-11-20/#reproduce)

```sh
model=k2-horizon-mova-36b-fp8
python3 spark.py prepare "$model"
python3 spark.py serve "$model"
```
