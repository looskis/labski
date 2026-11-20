# Can low-confidence Qwen FP4 answers benefit from BF16?

The live policy reproduced all 112 held-out BF16 answer letters with 42 BF16 calls. [Measured outcome](../../results/qwen38-27b-precision-routing-2026-11-19/)

The existing Qwen3.8 27B crossover measured a throughput advantage and some changed correctness verdicts with the mixed checkpoint. This experiment asks whether the mixed model's first-token confidence can identify which questions merit the cost of a second inference. It tests a one-token, ten-choice workload. [Original precision comparison](../../results/qwen38-27b-precision-2026-11-19/)

## Fixed policy and held-out evaluation

Start the published one-Spark `qwen38-27b-nvfp4` and `qwen38-27b-bf16` recipes on separate Sparks. Both are greedy 8K text services with four slots, BF16 KV and disabled prefix caching. Specify the ten allowed answer IDs and request a log probability for each. The policy uses the mixed answer directly when its highest log probability exceeds the second by at least **1.0 natural-log unit**. Below that threshold it sends the same question to the BF16 service and uses its answer. This is a fixed threshold chosen before collecting the scores.

Prepare the pinned MMLU-Pro corpus with sixteen items from each of its fourteen subjects. The first eight are the original lab diagnostic. The next eight are held out from the original study. The two checkpoints must report identical rendered input token hashes on every matched case. Both pure-precision controls return the same set of label log probabilities as the mixed probe. [Dataset and preparation](../../results/qwen38-27b-precision-2026-11-19/#reproduce)

```sh
python3 ../../mmlu_probe.py prepare /path/to/test.parquet --per-subject 16 --out qwen-224.jsonl
python3 probe.py qwen38-27b-nvfp4 qwen-224.jsonl --out mixed.jsonl
python3 probe.py qwen38-27b-bf16 qwen-224.jsonl --out bf16.jsonl
```

Run the commands against their respective endpoints, setting `--url` as needed. The batch receipts include all ten label scores, the selected token ID, its margin, the expected answer and the digest of the actual model input. [Probe](probe.py)

The live replay sends each of the 112 held-out questions through ordinary mixed precision, ordinary BF16, and conditional BF16. It makes the second model request only when the primary margin triggers it. It shuffles case order with a fixed seed and varies the order of the three policies across groups of sixteen. Both models stay resident. Each worker reads case IDs on standard input and returns a scored request on standard output.

```sh
python3 replay.py qwen-224.jsonl \
  --mixed-worker 'python3 probe.py qwen38-27b-nvfp4 qwen-224.jsonl --worker --url http://127.0.0.1:8001' \
  --bf16-worker 'python3 probe.py qwen38-27b-bf16 qwen-224.jsonl --worker --url http://127.0.0.1:8002' \
  --out held-out-live.jsonl
```

The command example uses two local forwarded ports. A worker can also run on its Spark over an existing SSH connection. The measured client wall time then includes the request transport. [Matched live replay](replay.py)
