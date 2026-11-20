# Full GLM-5.3 on four DGX Sparks

The 753B GLM-5.3 served text on four 128 GB DGX Sparks in a switchless ConnectX-7 ring. With the 2.75 bpw EXL3 checkpoint, the 6-bit output head and two-token native MTP, it delivered **32.85 generated tokens/s for one client and 63.33 aggregate tokens/s for four**. A separate retrieval run returned the exact value in all eight inventory lookups. [Recipe](../../recipes/full-glm53.json) · [Lookup requests](q6-lookup.json) · [Upstream TensorFold implementation](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark/tree/ea05ec696d1c408268a3d1de39b090b154c1ebff)

## Six-bit output head

| Clients | Output tokens/s across the four Sparks, three trials | Median | GPU joules per generated token, median |
| ---: | --- | ---: | ---: |
| 1 | 33.28, 32.85, 32.32 | 32.85 | 6.12 |
| 4 | 66.08, 63.33, 60.87 | 63.33 | 3.60 |

The test ran four prose prompts per trial, with an excluded warm-up at each concurrency. Each of the 24 timed requests generated 128 tokens, including reasoning, with the template's default behavior. Timing includes prefill, decoding and scheduling. Energy sums NVML counters from all four GPUs within each trial. [Trial data](q6-trials.jsonl) · [Request timing and hashes](q6-requests.jsonl.gz) · [Client settings](q6-settings.jsonl) · [Energy by rank](full-glm53-q6-energy.jsonl) · [Raw counters](full-glm53-q6-energy.csv.gz)

On a fixed 28-item MMLU-Pro short-answer diagnostic, the endpoint scored **21/28**. Three replies reached the 128-token limit. The client sent top-level `enable_thinking=false`, but the server logged `thinking=true` and 18 responses included reasoning. The rubric requires a stopped response containing exactly the correct final letter. These results describe that request shape and budget. The arithmetic smoke test returned exactly 42 in 0.61 s. [Item-level results](q6-quality.jsonl.gz) · [Arithmetic](q6-first.json)

## Runtime and reproduction

The pinned checkpoint contains 276.93 GB of SafeTensors including the auxiliary head. Each rank loaded all 78 layers, reported 64.4 GiB of device weights at the final layer-loading milestone, then reported four 32,768-token stream caches at 2.95 GiB each and 312 captured decode graphs. The container's version file identifies TensorFold `7765bda0` and mlx-serve `a2072091`. The rank logs confirm a 6.0-bit output projection split across the four GPUs. Both neighbor ports on each Spark recorded unicast RDMA transmit and receive traffic during the requests. [Per-rank startup](q6-startup.jsonl) · [NIC counters](q6-rdma.csv) · [Checkpoint files](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW/tree/ae7b69671a94cf130fd1360a3b9213539ef3446f)

Prepare the checkpoint and pinned runtime on each Spark. Set up the routed ring and build the lab's NCCL 2.31.2 library. Launch this command on each rank with its own addresses and interface names. The recipe uses NCCL over RoCE, caps NCCL at two CTAs, disables the engine's separate custom RoCE path and disables prompt reuse. The API binds to the head's loopback interface. [Ring setup](../../ring-tcp.md) · [Library build](../../transport/nccl-switchless/)

```sh
python3 spark.py prepare full-glm53
python3 spark.py serve full-glm53 --nodes 4 --rank "$RANK" --head "$HEAD_IP" \
  --node-ip "$NODE_IP" --interface "$FABRIC_IF" --transport ring-roce \
  --hcas "$CCW_HCA,$CW_HCA" --fabric-range "$LINK_CIDR" --link-prefix 30 \
  --ring-nccl "$NCCL_LIBRARY" --ring-only-init --output-head q6
```

On the API host, get the model ID from `/v1/models`. Run the following from the inference-recipes directory. The same corpus generator fixes the 28 questions across output heads. [Prepare the question file](../qwen38-27b-precision-2026-11-19/#reproduce)

```sh
MODEL=$(python3 -c 'import json,urllib.request; print(json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models"))["data"][0]["id"])')
python3 mmlu_probe.py run "$MODEL" labski-mmlu-112.jsonl --enable-thinking off \
  --per-subject 2 --max-tokens 128 --out q6-quality.jsonl
for c in 1 4; do
  python3 bench.py "$MODEL" --workload prose --requests 4 --repeats 3 \
    --concurrency "$c" --max-tokens 128 --out "q6-c$c"
done
python3 bench.py "$MODEL" --workload lookup --records 32 --requests 8 \
  --repeats 1 --max-tokens 512 --out q6-lookup
```
