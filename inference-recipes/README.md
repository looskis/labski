# Inference recipes for DGX Spark

The lab has four 128 GB NVIDIA DGX Sparks cabled in a ConnectX‑7 ring. Each reports one GB10 and approximately 121 GiB of system memory. One Spark serves Qwen3.8 27B in BF16 or mixed precision. Qwen3.8 Flash-Next runs on a single Spark with its disk-backed FP4-Local checkpoint and on a direct pair with the larger checkpoint. A direct pair also serves GLM-5.3-Flash. Four Sparks serve DeepSeek V4.1 Flash 1017, full GLM-5.3 and native Inkling 2 Small. The lab also measured a single Qwen endpoint sharded across all four Sparks. Each result records its transport and workload.

[Open-weight leaderboard and Spark fit inventory](leaderboard/) · [Experiments suggested by the literature](research-leads.md)

**Measured on the lab fleet**

- **Qwen3.8 27B drafting.** Target-verified suffix drafts from the current request raised single-client exact-edit throughput from 12.07 to 50.74 tok/s. Retaining earlier requests slowed edits but helped the unrelated-prose workload at one client. [Four draft policies on the same GPU](results/qwen38-27b-drafting-2026-11-19/)
- **Qwen3.8 27B precision.** Mixed FP4 delivered 2.78–2.79× the BF16 throughput at one client in a two-machine crossover. The mixed checkpoint scored 74/112 and 78/112 on the paired short-answer diagnostic against 83/112 on both BF16 runs. [Timing, answers and settings](results/qwen38-27b-precision-2026-11-19/)
- **Qwen3.8 27B tensor parallelism.** The same mixed checkpoint delivered 12.40, 20.46 and 33.94 output tok/s for one client on one, two and four Sparks. Four-client totals were 45.74, 72.63 and 110.77. The four-Spark switchless ring used stock NCCL over RoCE. Routed TCP on those four machines reached 8.03 and 25.72 tok/s. [Scaling and quality](results/qwen38-27b-scaling-2026-11-19/) At 32 clients with 9 GiB KV per GPU, TP2 reached 354.39 tok/s at 0.21 GPU J/token. Two replicas reached 296.47 tok/s with a shorter p95 first-token wait (455 vs 587 ms). On the second pair, 3.25 GiB per GPU was enough for all 32 short requests and 351.58 tok/s on TP2. [Replica comparison](results/qwen38-27b-pair-serving-2026-11-20/)
- **Qwen3.8 Flash-Next.** The one-Spark FP4-Local recipe reached 33.98 and 72.86 output tok/s at one and four clients. On a direct pair, four scheduler slots with 2 GiB of KV per rank delivered 64.96 output tok/s at four clients, 2.90× the one-slot control, at 61% fewer GPU joules per output token. One client delivered 21.73 tok/s. [Quality, timing and energy](results/qwen38-flash-next-2026-11-19/)
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
| [K2 Horizon MoVA 36B A4B](recipes/k2-horizon-mova-36b-fp8.json) | 1 × 128 GB | Official FP8 routed experts with BF16 value experts and attention | 5.38 output tok/s at one client and 18.51 aggregate at four. [Quality and energy](results/k2-horizon-mova-36b-2026-11-20/) |
| [K2 Horizon 7B with native end markers](recipes/k2-horizon-7b-effort-parser.json) | 1 × 128 GB | BF16 | [7/14 and 9/14 open answers at 128 and 512 tokens, all eight streamed lookups](results/k2-horizon-family-2026-11-20/7b-parser/) |
| [Kimi K3.1 Pro storage pilot](experiments/kimi31-ssd/) | 1 × 128 GB plus local NVMe | Q8_0 spine and original MXFP4 routed experts | GB10 runtime compiled. One verified 896-expert layer delivered 10.59 GB/s direct reads at 16-way concurrency. Full generation is pending. |
| [DeepSeek V4.1 Flash 1017](experiments/deepseek1017/) | 4 × 128 GB | Native FP8/MXFP4 plus disk-backed Engram | Direct RoCE decode reached 43.61 tok/s at one client and 72.21 at four, a 15.9–16.9% gain over NCCL decode. [Paired timing and quality](results/deepseek-v41-flash-1017-2026-11-20/) |
| [MiniMax M3.1 Flash Maya-XS](experiments/minimax-maya/) | 1 × 128 GB | Mixed IQ1_S/IQ2_XXS GGUF and separate sparse indexer | At four slots, 14.71 tok/s for one client and 30.25 for four with 7,012 GPU-resident layer–expert pairs. [One-Spark study](results/minimax-m31-flash-maya-xs-2026-11-20/) |
| [Ling 3.1 Flash 1019](experiments/ling31-exl3/) | 1 × 128 GB | Mixed EXL3 with 1.0/1.5 bpw routed experts | Two draft tokens reached 22.67 tok/s at one client. One draft token reached 46.65 aggregate at four and beat two by 5.3% in the warmed repeat. All draft depths scored 15/28 constrained letters. [Measured recipe](results/ling31-flash-1019-exl3-2026-11-20/) |
| Nanbeige4.3-33B precision sweep | 1 × 128 GB | [Official BF16](recipes/nanbeige43-33b-bf16.json), [official FP8](recipes/nanbeige43-33b-fp8.json), [community NVFP4](recipes/nanbeige43-33b-nvfp4.json) | At four clients, 14.10 / 24.14 / 43.50 tok/s and 63 / 63 / 60 free answers out of 112. [Measurements](results/nanbeige43-33b-2026-11-19/) |
| [Qwen3.8 27B control](recipes/qwen38-27b-bf16.json) | 1, 2 or 4 × 128 GB | BF16 | 83/112 with identical verdicts at every topology, up to 13.92 tok/s C1 and 54.33 aggregate at C4 |
| [Qwen3.8 27B precision and scaling](recipes/qwen38-27b-nvfp4.json) | 1, 2 or 4 × 128 GB | Mixed NVFP4/FP8 | Same scheduler, KV dtype and context as the BF16 control |
| [Qwen3.8 27B DFlash2 with BF16 state](recipes/qwen38-27b-dflash-v033-7-ssmbf16.json) | 1 × 128 GB | Mixed NVFP4/FP8, BF16 recurrent state | With 3 GiB cache, four clients reached 94.78 and 104.34 tok/s on separate launches, against 53.73 for FP32 state. Both matched on 28 short answers and 16 lookups. [Matched measurements](results/qwen38-27b-dflash2-2026-11-20/recurrent-state/) |
| [Qwen3.8 27B DFlash2 with 5.5 GiB KV](recipes/qwen38-27b-dflash-v033-7-kv5632m.json) | 1 × 128 GB | Mixed NVFP4/FP8, BF16 KV | [30.38 / 93.86 tok/s with one / four clients, 19/28 short answers and all eight lookups](results/qwen38-27b-dflash2-2026-11-20/kv-budget/) |
| [Qwen3.8 27B DFlash2, stock vLLM](results/qwen38-27b-dflash2-2026-11-20/) | 1 × 128 GB | Mixed NVFP4/FP8 | With 3 GiB KV, three proposals led at four clients and seven at one |
| [Qwen3.8 27B DFlash2 on NGC](recipes/qwen38-27b-dflash.json) | 1 × 128 GB | Mixed NVFP4/FP8 | Seven-token draft, FP8 KV cache, GB10 dynamic linear backend |
| [Gemma 4 31B IT](recipes/gemma4-31b-nvfp4.json) | 1 × 128 GB | NVFP4 | Tool calling with an assistant drafter |
| [GLM 5.3 Flash measured prefill](recipes/glm53-nvfp4-prefill.json) | 2 × 128 GB | NVFP4 | 8K prefill chunks, C1, [archived measurements](results/glm53-prefill-2026-11-18/) |
| [Qwen3.8 Flash-Next](recipes/qwen38-flash-next-nvfp4.json) | 2 × 128 GB | Mixed NVFP4/FP8/BF16 | Measured text-only service at 8K context with a 133 GB checkpoint |
| [Inkling 2 Small, two slots](recipes/inkling2-small-native-c2.json) | 4 × 128 GB | Native mixed NVFP4/MXFP8 | [39.34 aggregate tok/s at two clients and six exact lookups up to 3,597 input tokens](results/inkling2-gb10-runtime-2026-11-19/#two-slot-throughput-and-shared-prefix-context) |
| [Inkling 2 Small, one slot](recipes/inkling2-small-native.json) | 4 × 128 GB | Native mixed NVFP4/MXFP8 | [Served at 4K context with one scheduler slot](results/inkling2-gb10-runtime-2026-11-19/) |
| [GLM 5.3 Flash 4K chunk control](recipes/glm53-chunk4096.json) | 2 × 128 GB | NVFP4 | Same image and runtime settings as the 8K recipe, with [one parameter varied](results/glm53-prefill-2026-11-18/#isolating-prefill-chunk-size) |
| [Qwen3.8 Flash-Next FP4-Local](recipes/qwen38-flash-next-fp4-local.json) | 1 × 128 GB | Mixed FP4-Local, FP8 KV | [33.98 output tok/s at one client and 72.86 aggregate at four, 1.08/0.52 GPU J/token](results/qwen38-flash-next-2026-11-19/#single-spark-fp4-local-versus-the-tuned-pair) |
| [Qwen3.8 Flash-Next, four clients](recipes/qwen38-flash-next-kv2g-c4.json) | 2 × 128 GB | Mixed NVFP4/FP8/BF16 | Tuned 8K service with 2 GiB KV per rank and four scheduler slots |
| [Qwen3.8 Flash-Next, small KV pool](recipes/qwen38-flash-next-kv2g-c1.json) | 2 × 128 GB | Mixed NVFP4/FP8/BF16 | 2 GiB BF16 KV per rank and 21.80 output tok/s at one client |
| [GLM-5.3-Flash on TensorFold 1.3.0](results/glm53-tensorfold-130-tiles-2026-11-20/engine/) | 2 × 128 GB | 4 bpw EXL3 | The lab kernel raised uncached prefill throughput 1.71–3.15% across seven input lengths. Both fresh dispatches agreed on 14/14 first tokens in each matched pass. Repeating the prompts within either launch reproduced 6/14. [Serving measurements](results/glm53-tensorfold-130-tiles-2026-11-20/serving/) |
| [Step-5 Preview with rank-local experts](experiments/step5-rank-weights/) | 4 × 128 GB | Hybrid EXL3/BF16 | 121.32 GB of weights per Spark, 19.67 / 53.49 output tok/s with one / four clients, and 8/8 exact lookups. [Placement comparison](results/step5-preview-exl3-2026-11-20/rank-local/) |
| [Full GLM-5.3](recipes/full-glm53.json) | 4 × 128 GB | TensorFold EXL3 2.75 bpw | [Measured at 32K with the native MTP layer and 6-bit output head, 3.8% higher throughput than the BF16 head at one client and 2.0% higher at four](results/glm53-full-tp4-2026-11-19/) |
| [GLM 5.3 Flash chat](recipes/glm53-flash-fp4.json) | 2 × 128 GB | Mixed FP4-Local | 320B MoE, B12X loader, MTP and 64K context |
| [MiniMax M3.1 Flash](recipes/minimax31-flash-tp4.json) | 4 × 128 GB | NVFP4 experts + MXFP8 | One large endpoint using all four Sparks and the bundled DSpark head |

For full GLM-5.3, `--output-head q6` selects the pinned auxiliary 6-bit output projection. Use `--output-head bf16` on all four ranks for the control. Both use the same EXL3 checkpoint and MTP setting. This launcher invokes the final image layer’s rank-specific entrypoint with its API bound to loopback. [Full GLM-5.3 recipe](recipes/full-glm53.json) · [Upstream head comparison](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark/tree/ea05ec696d1c408268a3d1de39b090b154c1ebff)

## First completion on one Spark

From this directory on a GB10 Spark running DGX OS, use Python 3.9+, Docker with NVIDIA GPU support. Use a driver compatible with the selected image. Each recipe's `source` points to its model or platform reference. On the lab’s 580.178.04 driver, the NGC 26.10.pre1 image hit a PTX toolchain error serving Qwen3 0.6B. Its verified bring-up recipe uses vLLM 0.33.0 in eager mode. The NGC DFlash2 candidate uses GB10-specific kernel flags and needs a separate driver check. The measured DFlash2 recipe runs on the same driver with vLLM 0.33.0 and BF16 KV. The launcher resolves container tags to the immutable digests in [images.json](images.json).

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

`prepare` pulls the ARM64 image and downloads the pinned target and any separate draft checkpoint to the local Hugging Face cache. Set `HF_HUB_CACHE` before both commands to use a larger Spark volume. Downloads resume through the Hub cache. `serve` uses the cached snapshot paths with the weight mount read-only. Stop the foreground container with Ctrl-C or run `docker stop labski-qwen3-0.6b` in another terminal.

### Stage weights for a fleet run

Choose a Spark with space for the missing weights and the recipe’s temporary files. Check its existing target and draft snapshots against the pinned revision and digests. Run the download on that Spark. The management workstation can launch it over SSH and collect the small result files.

For a full-copy runtime, each participating Spark needs its own complete target snapshot. Transfer the missing files from the source Spark over ConnectX, keeping Hugging Face blob symlinks resolvable on the receiver. For a runtime with a verified rank-local or remote-weight loader, follow its placement instructions. Step-5 and Inkling use only the expert rows assigned to each rank. K2 Horizon 375B can build a compact shard from upstream byte ranges directly on its destination Spark. Preserve verified shards and partial downloads across retries. Schedule checkpoint traffic outside measured serving intervals on the same machines and links. [Step-5 compact weights](experiments/step5-rank-weights/) · [Inkling sparse weights](experiments/inkling2-sparse-weights/) · [K2 per-rank downloader](experiments/k2-moe-weights/#download-one-expert-partition-directly-to-a-spark)

## One endpoint on multiple Sparks

Set up the dedicated ConnectX fabric and verify all participating ranks can communicate over RoCE. The two-node example uses a direct pair. On the four-node cable ring, give each pair of neighbors a separate link subnet and give each rank a routed address for rendezvous. Stock NCCL can then carry tensor-parallel payloads over RoCE between adjacent peers. The independent-replica experiment used the management network for requests. Use the management LAN for SSH. On each Spark, `ip -br -4 addr` and `ibdev2netdev` identify its local address, interface and HCA names. Use NCCL all-reduce tests across the intended nodes before interpreting scaling results. [NVIDIA cluster setup](https://build.nvidia.com/playbooks/connect-multiple-sparks) · [NVIDIA multi-Spark vLLM playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/playbook-vllm-cluster)

Copy this scope at the same revision to every node. For the MiniMax example, run `python3 spark.py prepare minimax31-flash-tp4` on the source Spark. Stage the full target and draft snapshots on the other three Sparks over ConnectX. Set these values in a terminal on **each** node. The addresses and interface names below are examples.

```sh
# Rank 0 uses the head's address. On ranks 1, 2 and 3 set their own RANK and NODE_IP.
export RANK=0 NODE_IP=192.0.2.1
export HEAD_IP=192.0.2.1 CX_IF=enp1s0f0np0 CX_HCAS=rocep1s0f0,roceP2p1s0f0

python3 spark.py serve minimax31-flash-tp4 --nodes 4 --rank "$RANK" \
  --head "$HEAD_IP" --node-ip "$NODE_IP" --interface "$CX_IF" --hcas "$CX_HCAS"
```

The [switchless ring setup](ring-tcp.md) gives a worked four-node IP and routing example. For the measured four-node RoCE configuration use `--transport ring-roce`, supply both neighbor HCAs and select the link subnets with `--fabric-range` and `--link-prefix`. For a routed TCP reference, supply `--transport tcp --interface fabric0` in place of the RoCE interface and HCA arguments on every rank. Each node's `--node-ip` must be local to that interface. All four addresses must route to each other over ConnectX. This selects NCCL Socket and disables IB transport. Verify the interface selection in the NCCL initialization logs and retain the transport with the result.

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
