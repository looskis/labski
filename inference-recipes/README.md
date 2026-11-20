# Inference recipes for DGX Spark

The lab has four 128 GB NVIDIA DGX Sparks cabled in a ConnectX‑7 ring. Each reports one GB10 and approximately 121 GiB of system memory. Start with one for interactive serving, use a pair for GLM, or shard MiniMax across all four. The matched Qwen recipes let you measure what an extra Spark or a lower-precision checkpoint buys on the same workload.

[Open-weight leaderboard and Spark fit inventory](leaderboard/)

**Measured on the lab fleet**

- **Qwen3.8 27B precision.** Mixed FP4 delivered 2.78–2.79× the BF16 throughput at one client in a two-machine crossover. The mixed checkpoint scored 74/112 and 78/112 on the paired short-answer diagnostic against 83/112 on both BF16 runs. [Timing, answers and settings](results/qwen38-27b-precision-2026-11-19/)
- **Four replicas.** At 16 concurrent prose requests, Qwen3 0.6B delivered 1,894 aggregate output tok/s across four Sparks versus 538 on one. At four clients, the single Spark delivered 493 versus 427 across four. [Workload, raw requests and trial results](results/qwen3-four-replicas-2026-11-19/)
- **GB10 kernel tuning.** A wider Q8 projection tile added 4.3% to GLM cold-prefill throughput at 32K. The sampled likelihoods and greedy answers matched the control. A separate INT8 activation setting was faster and changed a correct answer. [Code, timing and quality checks](results/glm53-tensorfold-wide-2026-11-18/)
- **Two-node NVFP4.** GLM‑5.3‑Flash returned the first output token for a 32,768-token input in 13.07 s on the November 19 replay. The November 18 comparison recorded 13.15 s for NVFP4/vLLM and 17.22 s for EXL3/TensorFold on matching input IDs. [Token corpus, timing and quality results](results/glm53-prefill-2026-11-18/)

Additional catalog entries are candidates unless a result is linked. Node counts and RAM below describe target configurations. [TensorFold comparison](tensorfold.md) · [Benchmark and publication protocol](measurement.md)

| Recipe | Sparks × nominal RAM per Spark | Weight format | Experiment |
| --- | --- | --- | --- |
| [Qwen3 0.6B](recipes/qwen3-0.6b.json) | 1 × 64 GB or larger | BF16 | About 1.5 GB of weights, [verified on a 128 GB Spark](results/qwen3-0.6b-2026-11-19/) |
| [Qwen3 8B](recipes/qwen3-8b.json) | 1 × 64 GB or larger | BF16 | Dense chat at one and four clients |
| [GPT-OSS 20B](recipes/gpt-oss-20b.json) | 1 × 64 GB or larger | Native MXFP4 experts | Small reasoning service with up to eight active sequences |
| [Nemotron 3.5 Lightning 30B A3B](recipes/nemotron35-lightning.json) | 1 × 64 GB or larger | Mixed NVFP4 | DSpark drafting with a hybrid attention/Mamba cache |
| [Qwen3 Coder 30B A3B](recipes/qwen3-coder-30b.json) | 1 × 128 GB | BF16 | Sparse coding model with tool calling |
| [Qwen3.8 27B control](recipes/qwen38-27b-bf16.json) | 1 × 128 GB | BF16 | Reference for precision and output quality |
| [Qwen3.8 27B precision and scaling](recipes/qwen38-27b-nvfp4.json) | 1, 2 or 4 × 128 GB | Mixed NVFP4/FP8 | Same scheduler, KV dtype and context as the BF16 control |
| [Qwen3.8 27B DFlash2](recipes/qwen38-27b-dflash.json) | 1 × 128 GB | Mixed NVFP4/FP8 | Seven-token draft, FP8 KV cache, GB10 dynamic linear backend |
| [Gemma 4 31B IT](recipes/gemma4-31b-nvfp4.json) | 1 × 128 GB | NVFP4 | Tool calling with an assistant drafter |
| [GLM 5.3 Flash measured prefill](recipes/glm53-nvfp4-prefill.json) | 2 × 128 GB | NVFP4 | 8K prefill chunks, C1, [archived measurements](results/glm53-prefill-2026-11-18/) |
| [Qwen3.8 Flash-Next](recipes/qwen38-flash-next-nvfp4.json) | 2 × 128 GB | Mixed NVFP4/FP8/BF16 | Text-only fit trial with a 133 GB checkpoint |
| [Inkling 2 Small](recipes/inkling2-small-native.json) | 4 × 128 GB | Native NVFP4/MXFP8 experts | Four-node text fit trial with the SGLang GB10 hooks and BF16 KV |
| [GLM 5.3 Flash 4K chunk control](recipes/glm53-chunk4096.json) | 2 × 128 GB | NVFP4 | Same image and runtime settings as the 8K recipe, with [one parameter varied](results/glm53-prefill-2026-11-18/#isolating-prefill-chunk-size) |
| [GLM 5.3 Flash chat](recipes/glm53-flash-fp4.json) | 2 × 128 GB | Mixed FP4-Local | 320B MoE, B12X loader, MTP and 64K context |
| [MiniMax M3.1 Flash](recipes/minimax31-flash-tp4.json) | 4 × 128 GB | NVFP4 experts + MXFP8 | One large endpoint using all four Sparks and the bundled DSpark head |

## First completion on one Spark

From this directory on a GB10 Spark running DGX OS, use Python 3.9+, Docker with NVIDIA GPU support. Use a driver compatible with the selected image. Each recipe's `source` points to its model or platform reference. On the lab’s 580.178.04 driver, the NGC 26.10.pre1 image hit a PTX toolchain error serving Qwen3 0.6B. Its verified bring-up recipe uses vLLM 0.33.0 in eager mode. The DFlash2 candidate still uses the NGC build for its GB10-specific kernel flags and needs a separate driver check. The launcher resolves container tags to the immutable digests in [images.json](images.json).

```sh
python3 spark.py list
python3 spark.py prepare qwen3-0.6b --dry-run
python3 spark.py prepare qwen3-0.6b
python3 spark.py serve qwen3-0.6b
```

In a second terminal on the same Spark, wait for `/health` to return HTTP 200 and then request a completion.

```sh
curl -f http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"Name one primary color."}],"temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

`prepare` pulls the ARM64 image and downloads the pinned target and any separate draft checkpoint to the local Hugging Face cache. Set `HF_HUB_CACHE` before both commands to use a larger volume. Allow space for the full checkpoint on **each** participating machine, plus the image and runtime scratch space. GLM needs roughly 177 GB of target shards per node, MiniMax roughly 250 GB per node before its draft. Downloads resume through the Hub cache. `serve` uses the cached snapshot paths with the weight mount read-only. Stop the foreground container with Ctrl-C or run `docker stop labski-qwen3-0.6b` in another terminal.

## One endpoint on multiple Sparks

Set up the dedicated ConnectX fabric and verify all participating ranks can communicate over RoCE. The two-node example uses a direct pair. The four-node TP examples require routed reachability between all ranks on the selected fabric, as in a switched network. A four-node cable ring only gives each Spark direct links to its two neighbors. The independent-replica experiment works on the lab’s ring using the management network for requests. Use the management LAN for SSH. On each Spark, `ip -br -4 addr` and `ibdev2netdev` identify its local address, interface and HCA names. Use NCCL all-reduce tests across the intended nodes before interpreting scaling results. [NVIDIA cluster setup](https://build.nvidia.com/playbooks/connect-multiple-sparks) · [NVIDIA multi-Spark vLLM playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/playbook-vllm-cluster)

Copy this scope to the same revision on every node. Run `python3 spark.py prepare minimax31-flash-tp4` on each of the four. Then set these values in a terminal on **each** node. The addresses and interface names below are examples.

```sh
# Rank 0 uses the head's address. On ranks 1, 2 and 3 set their own RANK and NODE_IP.
export RANK=0 NODE_IP=192.0.2.1
export HEAD_IP=192.0.2.1 CX_IF=enp1s0f0np0 CX_HCAS=rocep1s0f0,roceP2p1s0f0

python3 spark.py serve minimax31-flash-tp4 --nodes 4 --rank "$RANK" \
  --head "$HEAD_IP" --node-ip "$NODE_IP" --interface "$CX_IF" --hcas "$CX_HCAS"
```

The [switchless ring setup](ring-tcp.md) gives a worked four-node IP and routing example. For a routed TCP reference, supply `--transport tcp --interface fabric0` in place of the RoCE interface and HCA arguments on every rank. Each node's `--node-ip` must be local to that interface. All four addresses must route to each other over ConnectX. This selects NCCL Socket and disables IB transport. Verify the interface selection in the NCCL initialization logs and retain the transport with the result.

To test collectives before loading weights, run the following from this directory on every rank, with the same rendezvous address and port, and its own `RANK`. Set `CX_IF` to the interface on the routed ConnectX fabric. Each rank checks the exact all-reduce of rank-specific values at three payload sizes. Eight calls at each size include one warm-up.

```sh
IMAGE=$(python3 -c 'import json; print(json.load(open("images.json"))["vllm-0.33.0"])')
docker run --rm -i --gpus all --network host --ipc host \
  -e RANK="$RANK" -e WORLD_SIZE=4 -e MASTER_ADDR="$HEAD_IP" -e MASTER_PORT=29522 \
  -e NCCL_NET=Socket -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME="=$CX_IF" \
  -e GLOO_SOCKET_IFNAME="$CX_IF" -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
  --entrypoint python3 "$IMAGE" - < fabric_probe.py
```

Start ranks 1–3 in separate terminals, then rank 0. The workers wait for rendezvous on port 29501. Only rank 0 serves the loopback API. Use `--master-port` on every rank and `--port` on the head if needed. Stop the container on **every** rank after the run. For two Sparks, use `glm53-flash-fp4 --nodes 2` and ranks 0–1. For a scaling comparison, use `qwen38-27b-nvfp4` at one, two and four nodes. GPU memory budgets apply to each rank and include KV cache and workspace in addition to its share of the weights.

## Four independent endpoints

For aggregate throughput, launch the same one-Spark recipe separately on each box. This topology needs no distributed rendezvous. Forward the four loopback APIs to distinct ports on the measurement client. Replace the host aliases with your own and keep the tunnels open.

```sh
ssh -N -L 8101:127.0.0.1:8000 spark-a
# In separate terminals repeat for 8102/spark-b, 8103/spark-c and 8104/spark-d.
```

Run the same prompt set and total client concurrency through the four endpoints, then through one four-rank endpoint. The benchmark uses round-robin request placement and a single wall clock for the fleet. [Measurement commands](measurement.md#compare-topologies)

## Publishing a recipe

Add a JSON file under [recipes](recipes/) with a purpose, checkpoint and immutable revision, precision, supported node counts, nominal memory per node, runtime, serving limits, model-specific options and primary source. Register any new image by digest in [images.json](images.json). Add the catalog row and link the measured run when available. The [Apache 2.0 license](LICENSE) covers the lab's original material in this scope. Checkpoints and containers are fetched from their publishers under their own terms.
