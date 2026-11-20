# K2 Horizon MoVA 36B A4B on one DGX Spark

The official 48.36 GB FP8 checkpoint served four concurrent requests on one 128 GB GB10. At four clients it delivered **18.51 aggregate output tokens/s** and used **2.12 GPU joules per output token**. For an interactive client the corresponding figures were **5.38 tokens/s** and **7.01 GPU J/token**. The checkpoint quantizes the feed-forward routed experts in layers 3–47. It retains BF16 for the MoVA value experts, attention, shared experts, dense layers, embeddings and output head. [Pinned recipe](../../recipes/k2-horizon-mova-36b-fp8.json) · [Publisher model card and quantization method](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B-FP8/blob/deaeefb1bb9e0c080be6d8844f3b222f90fd4bc7/README.md)

| Concurrent clients | Output tokens/s | Median first streamed delta | P95 first streamed delta | GPU J/output token |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.38 | 533 ms | 536 ms | 7.01 |
| 4 | 18.51 | 806 ms | 909 ms | 2.12 |

Each entry is the median of three trial statistics. Each trial served four distinct prompts with a 96-token output cap. All 24 timed requests reached the cap. GPU energy uses cumulative device counters over each trial wall-clock interval. The Spark ran vLLM 0.33.0 with an 8,192-token context, four sequence slots, 3 GiB of KV memory, and prefix caching disabled. Its runtime selected FlashAttention for attention and DeepGEMM for the routed FP8 MoE. [Timing](fp8/trials.jsonl) · [Individual requests](fp8/requests.jsonl.gz) · [Benchmark configuration](fp8/settings.jsonl) · [GPU energy](fp8/k2-mova-fp8-energy.jsonl) · [Raw energy samples](fp8/k2-mova-fp8-energy.csv.gz)

On the lab’s 112-question MMLU-Pro diagnostic, constraining the first output token to an answer letter yielded **19/112**. The model had one token in which to choose. The lab’s low-effort, deterministic application protocol differs from IFM’s published high-effort sampling evaluation. [Questions, labels and per-item results](fp8/grammar112.jsonl.gz) · [Publisher evaluation settings](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B-FP8/blob/deaeefb1bb9e0c080be6d8844f3b222f90fd4bc7/README.md)

## Open answers, retrieval and likelihood

With room to reason, the model returned **7/14 correct final answers at 128 output tokens** and **9/14 at 512**. Two 128-token responses reached their cap. All 14 requests at 512 finished before the cap. The model also returned **8/8 exact six-digit inventory values**. These are deterministic tests on the official FP8 release with low reasoning effort. [128-token answers](fp8/effort-low-128.jsonl.gz) · [512-token answers](fp8/effort-low-512.jsonl.gz) · [Streamed lookups](fp8/lookup.json)

On the 60 matched WikiText continuations (139,378 UTF-8 bytes), MoVA FP8 scored **0.6238 bits/byte**. On these same bytes K2 Horizon 32B scored 0.5692 in BF16, 0.5723 in FP8 and 0.5936 in NVFP4. MoVA FP8 assigned lower probability to the observed continuation on all 60 passages compared with 32B FP8, and on 54 of 60 compared with 32B NVFP4. This comparison changes the model architecture and training as well as the deployed precision. The separate token-based MoVA diagnostic gave perplexity 7.4741 on 30,935 predictions from 62 passages. [MoVA scores per continuation](fp8/bytes.jsonl) · [32B controls on the identical bytes](../../experiments/k2-horizon-family/reference/) · [MoVA token-level totals](fp8/wikitext.jsonl)

The FP8 files also contain original BF16 tensors. In a staging experiment, two Spark-built BF16 shards matched the official hashes after fetching only the missing tensor bytes and headers. The mixed-shard test used a verified source on the ConnectX peer. [Per-shard measurements and restoration commands](../../experiments/k2-horizon-family/#reuse-the-official-mova-fp8-cache-when-staging-bf16)

## Reproduce

Run these commands from `inference-recipes/` on one Spark. Use the matched K2 32B benchmark procedure with the MoVA served name for the two client counts and diagnostic panels. [Throughput, question and retrieval commands](../k2-horizon-32b-2026-11-20/#reproduce)

```sh
model=k2-horizon-mova-36b-fp8
python3 spark.py prepare "$model"
python3 spark.py serve "$model"
```
