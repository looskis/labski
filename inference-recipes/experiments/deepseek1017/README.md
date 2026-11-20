# DeepSeek V4.1 Flash 1017 on four DGX Sparks

Four DGX Sparks served the original FP8/MXFP4 release with TensorFold and both disk-backed Engram shards. Direct RoCE decode delivered 43.61 output tok/s at one client and 72.21 at four. Its 60 paired outputs matched the NCCL control. [Local timing, energy and quality](../../results/deepseek-v41-flash-1017-2026-11-20/) · [Upstream engine and published benchmark](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/tree/8ffa67d53ef15bab997399135f85e8b0910f30be) · [Original model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/08c8657f02291e11b6f6b63e0e16014f8e748518)

## Prepare rank-local files

On the head, download the official snapshot at the recorded revision. Use a Hugging Face client with large-file/Xet support. Its last two shards are each over 101 GB. The regular HTTP path in the lab's vLLM download image has a 50 GB per-file limit. Preparation checks for 48 complete weight files totalling 510,296,708,312 bytes.

```sh
revision=08c8657f02291e11b6f6b63e0e16014f8e748518
snapshot=$(hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision "$revision")
stage=/path/on/checkpoint-volume/deepseek1017
bash prepare.sh "$snapshot" "$stage"
```

The script builds the pinned engine, verifies the kernel kit, creates the token map, and bakes each parity-split rank into its own directory. Hardlinks keep the source weights and both Engram files from taking a second allocation on the head's NVMe. A separate serving view carries the upstream TensorFold chat template. The original snapshot remains suitable for other engines. [Prepare script](prepare.sh) · [Pinned kernel kit](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/tree/6033ea48a27004b2715db82190e2a61f33411d92)

The lab reproduced 355 of 356 kit manifest entries on the downloaded checkpoint. The one difference was the regenerated vision routing bias. All 351 tracked assets and four regenerated RoPE files matched. This recipe measures text inference. [Upstream kit verifier](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/blob/6033ea48a27004b2715db82190e2a61f33411d92/kit/verify_kit.sh)

Distribute the built image and ring-specific NCCL library to all ranks. Each follower needs `common/` and just its own logical `rankN/` directory from the staged tree. The physical rank order 0, 1, 2, 3 maps to logical rank 0, 1, 3, 2. The head opens its complete model view. Followers use the configuration and tokenizer files in `common/` and read Engram locally. Two original Engram files total 203,073,077,576 bytes per follower. [Build the NCCL library](../../transport/nccl-switchless/) · [Ring interface and addressing](../../ring-tcp.md#use-roce-on-the-same-four-cables)

## Launch

Start one command on each Spark from this experiment directory. Fill in the head address, routed interface and the two RoCE device names for that Spark. Use the same head address and fabric CIDR on all ranks. The graph builder assumes the first local NCCL net device faces the previous rank and the second faces the next. The device IDs can be set explicitly. Run the head on the machine with the complete snapshot view.

```sh
logical=(0 1 3 2)
model="$stage/common"
if [ "$RANK" = 0 ]; then model="$stage/head-model"; fi
python3 serve.py --rank "$RANK" --head "$HEAD_IP" --address-range "$FABRIC_CIDR" \
  --interface "$FABRIC_INTERFACE" --hcas "$PREVIOUS_HCA,$NEXT_HCA" \
  --model "$model" --common "$stage/common" --rank-cache "$stage/rank${logical[$RANK]}" \
  --ring-nccl "$RING_NCCL" --graph-dir "$stage/graphs"
```

The defaults are 8,192 tokens per slot, four slots and a head API on loopback port 8000 under `deepseek1017-native`. The draft scheduler uses the published GB10 costs for rounds of 1–48 rows and a 25 ms draft step. Each rank uses the original Engram lookup, native drafting, two ring channels and the upstream correctness setting `TF_DS_DECODE_DEC=0`. Use `--dry-run` after provisioning a rank to inspect the Docker command and generated graph files. [Launcher](serve.py) · [Upstream scheduler measurements](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/blob/8ffa67d53ef15bab997399135f85e8b0910f30be/scripts/config.sh)

### Select the decode transport

The default uses NCCL over RoCE for both prompt and decode collectives. Passing `--decode-transport verbs` keeps NCCL for the larger collectives and uses TensorFold's direct RoCE rings for decode-size exchanges. The engine pairs ports that share a /30 subnet and relays traffic to the opposite Spark. On the lab ring each selected device had a RoCE v2 IPv4 GID at index 3 that paired with its physical neighbor, so all ranks added `--rdma-gid 3`. Check each port on your own ring with `show_gids`. Omitting the index lets the upstream engine select the first RoCE v2 IPv4 GID per port. The worker log reports the number of lanes to each logical rank, the relay count and the column and row lanes. [Upstream direct ring implementation](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/blob/8ffa67d53ef15bab997399135f85e8b0910f30be/zig/patches/0001-dsv41-native-fp8-mxfp4-and-ring-tp4.patch)
