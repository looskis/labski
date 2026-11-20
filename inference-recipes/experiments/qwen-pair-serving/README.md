# Serve Qwen3.8 27B on a pair of DGX Sparks

Use two independent copies of the 21.92 GB NVIDIA mixed-precision checkpoint for concurrent requests, or shard one copy over the direct ConnectX‑7 link for a single endpoint. Both launches use the same pinned ARM64 vLLM image, 8,192-token context, BF16 KV, 3 GiB of KV per Spark by default, and disabled prefix caching. Each replica advertises 16 scheduler slots. The tensor-parallel endpoint advertises 32. The batch-token ceiling is 2,048 per engine, so the replica pair has two independent batching budgets. [Base recipe](../../recipes/qwen38-27b-nvfp4.json) · [Checkpoint staging](../../README.md)

From this directory on each Spark, set `HF_HUB_CACHE` to the local cache containing the pinned snapshot. For replicas, start the first command on the client’s machine and the second on its peer. Set `PEER_API_IP` to the peer’s private fabric address reachable by the client. This launcher uses host networking.

```sh
python3 deploy.py replicas --rank 0 --cache "$HF_HUB_CACHE" --port 18100
python3 deploy.py replicas --rank 1 --cache "$HF_HUB_CACHE" --port 18100 --api-host "$PEER_API_IP"
```

For one tensor-parallel endpoint, run the following on each Spark with its own rank, fabric IP, connected interface and comma-separated RoCE HCAs. Use the rank-zero fabric address for `HEAD_IP` on both. Run the benchmark on rank zero. [Direct-pair networking](../../README.md#one-endpoint-on-multiple-sparks)

```sh
python3 deploy.py tp2 --rank "$RANK" --cache "$HF_HUB_CACHE" --port 18100 \
  --head "$HEAD_IP" --node-ip "$FABRIC_IP" --interface "$INTERFACE" --hcas "$HCAS"
```

To reproduce the higher-throughput 32-client run, pass `--kv-gib 9` on both ranks. The 3.25 GiB follow-up used `--kv-mib 3328` and admitted all 32 short requests. Replay it with `--clients 32 64 --timing-only`. At each budget use the same value for both replicas and the tensor-parallel pair. [Throughput, latency and GPU energy](../../results/qwen38-27b-pair-serving-2026-11-20/)

For each deployment, give the benchmark a new output directory. The shared client uses round-robin dispatch across the supplied endpoints. Prepare the diagnostic corpus with the lab’s pinned MMLU-Pro procedure. [Corpus preparation](../../results/qwen38-27b-precision-2026-11-19/#reproduce)

```sh
python3 replay.py --endpoint http://127.0.0.1:18100 --endpoint "http://$PEER_API_IP:18100" \
  --corpus "$CORPUS" --out "$RUNS/replicas"
python3 replay.py --endpoint http://127.0.0.1:18100 --corpus "$CORPUS" --out "$RUNS/tp2"
```

Stop each deployment with `docker stop labski-qwen-pair-serving` on both Sparks before launching the next. To repeat the two replica loads in reverse order, run the replica command with `--clients 32 8 --timing-only` and another new output directory.
