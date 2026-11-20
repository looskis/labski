# Step-5-Preview on four switchless DGX Sparks

Four 128 GB Sparks served the 341.38 GB hybrid EXL3/BF16 checkpoint with two MTP draft tokens. With INT8 GEMV enabled, the fleet generated **20.37 tokens/s for one client and 57.88 aggregate for four**. GPU energy fell from 6.58 to 2.62 J per generated token as concurrency increased. These rates include reasoning tokens. The run used a physical ConnectX-7 ring, an 8,192-token request limit, four sequence slots and 4 GiB of KV memory per rank. [Lab recipe](../../recipes/step5-preview-exl3.json) · [Pinned model](https://huggingface.co/0xSero/Step-5-Preview-Spark/tree/51a46a7f62eee1225d124170da903bba23d04a26) · [Upstream implementation](https://github.com/0xSero/Step-5-Preview-Four-Sparks/tree/662949a6c1244c1e8a5c44ed62cedb89a207cbcc)

| INT8 GEMV | Generated tok/s, 1 / 4 clients | Fleet GPU J/token, 1 / 4 | First streamed delta, ms, 1 / 4 | Final answers / 28 | Capped / 28 | Forced letters / 112 | Exact final lookups / 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [On](gemv1/) | 20.37 / 57.88 | 6.58 / 2.62 | 287 / 490 | 7 | 21 | 30 | 8 |

Each throughput cell is the median of three trials with four matched prose requests and a 96-token limit, following an excluded warm-up at each concurrency. The first-delta timer stops on the first generated text, which can be reasoning. Energy integrates NVML readings on all four GB10s. The four-rank runs used eager execution, a 2,048-token prefill batch limit and the lab's ring-only NCCL 2.31.2 build. [Fabric and NCCL setup](../../transport/nccl-switchless/)

The four full snapshots occupied 1,365.52 GB of safetensors across the fleet. Keeping shared weights and each rank’s owned experts takes 485.27 GB, or 121.32 GB per Spark. All 371,712 routed expert tensors matched the source byte for byte. [Partition and verify the rank-local layout](../../experiments/step5-rank-weights/)

## Reading Step’s answers

The pinned template opens a reasoning block for every new answer. The benchmark’s `enable_thinking=false` hint does not close it. The measured endpoint returned reasoning and final text in a single content field with a literal `</think>` boundary. For the free-answer and lookup tests we require a normal stop and an exact match after the first closing delimiter. The archive contains the final answer, original-field SHA-256, completion length and finish reason. [Checkpoint template](https://huggingface.co/0xSero/Step-5-Preview-Spark/blob/51a46a7f62eee1225d124170da903bba23d04a26/chat_template.jinja)

With INT8 GEMV, seven questions returned a complete final letter within 128 tokens, and all seven were correct. The other 21 hit the cap. All eight lookup values were exact after the delimiter. Matching the entire raw field would score zero on both tests. The 112-item diagnostic constrains the first token to one of ten letters before the model can reason. The upstream interactive launcher supplies the `step3p5` reasoning parser.

## Reproduce

Stage the pinned model and image on each rank. Set the head and rank addresses, the interface, the two neighbor HCAs and the link subnets as in the lab’s four-node RoCE procedure. Set `RING_NCCL` to the patched library’s absolute path on each Spark. Run from `inference-recipes/` on every node. [Four-rank addressing](../../ring-tcp.md#use-roce-on-the-same-four-cables)

```sh
model=step5-preview-exl3
python3 spark.py prepare "$model"
python3 spark.py serve "$model" --nodes 4 --rank "$RANK" --head "$HEAD_IP" \
  --node-ip "$NODE_IP" --interface "$CX_IF" --hcas "$HCAS" --transport ring-roce \
  --fabric-range "$FABRIC_CIDR" --link-prefix 30 --ring-nccl "$RING_NCCL" --ring-only-init
```

Run the workloads on the head with the prepared corpus. To run the GEMV control, set `EXL3_INT8_GEMV` to `0` in a copy of the recipe on every rank, use the new recipe name and restart the endpoint. [Question corpus](../qwen38-27b-precision-2026-11-19/#reproduce)

```sh
model=step5-preview-exl3
mkdir -p "runs/$model"
for c in 1 4; do
  python3 bench.py "$model" --thinking off --workload prose --requests 4 --repeats 3 \
    --concurrency "$c" --max-tokens 96 --out "runs/$model/prose-c$c"
done
python3 mmlu_probe.py run "$model" "$CORPUS" --per-subject 2 --max-tokens 128 --out "runs/$model/free.jsonl"
python3 mmlu_probe.py run "$model" "$CORPUS" --one-letter --out "runs/$model/forced.jsonl"
python3 bench.py "$model" --thinking off --workload lookup --records 32 --requests 8 \
  --max-tokens 256 --concurrency 1 --repeats 1 --out "runs/$model/lookup"
```
