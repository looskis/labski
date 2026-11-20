# Qwen3 0.6B on one, two and four independent Sparks

One Mac client sent the same 96 deterministic prose requests to each deployment at each concurrency. Each Spark ran one copy of the [BF16 recipe](../../recipes/qwen3-0.6b.json). The four GB10s each reported 121 GiB of unified memory and driver 580.178.04. One Spark had wired management, and the additional endpoints used Wi‑Fi. The client reached their loopback APIs through SSH tunnels. Changes in transport and batching are part of these end-to-end results.

| Replicas | Concurrent clients | Median aggregate output tok/s | Relative to one Spark at this load | Median first visible text (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 492.8 | 1.00× | 57.9 |
| 2 | 4 | 520.8 | 1.06× | 46.4 |
| 4 | 4 | 427.3 | 0.87× | 40.5 |
| 1 | 16 | 538.3 | 1.00× | 2843.7 |
| 2 | 16 | 1070.1 | 1.99× | 954.4 |
| 4 | 16 | 1894.5 | 3.52× | 56.3 |

Throughput is the median of three trial rates. Each rate divides actual generated tokens by the wall time for 32 requests. The latency column pools the 96 request timings in that cell. All 576 measured requests completed and delivered visible answer text. [Trial timings and token counts](trials.csv) · [All request records](requests.jsonl.gz)

## Reproduction

On each participating 128 GB Spark, prepare and launch the same vLLM 0.33.0 recipe with its 1 GiB KV budget and four scheduler slots. Forward the endpoints to the measurement client. Use the command below for four replicas, then repeat with one or two endpoint arguments and a fresh output directory. Run once with concurrency 4 and once with 16. An excluded warm-up hits each endpoint. The measured runs use a 128-token budget, seed 42, greedy sampling and thinking disabled.

```sh
python3 bench.py qwen3-0.6b --thinking off --workload prose --requests 32 \
  --repeats 3 --concurrency 16 --max-tokens 128 --out runs/replicas-4-c16 \
  --endpoint http://127.0.0.1:18101 --endpoint http://127.0.0.1:18102 \
  --endpoint http://127.0.0.1:18103 --endpoint http://127.0.0.1:18104
```

This measures independent replicas with round-robin request placement. Tensor-parallel inference uses a different communication pattern and requires its own measurement.
