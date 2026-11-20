# MiMo V2.6 MOPD on DGX Spark

The one-Spark Flash deployment uses the Q2_K_S text checkpoint and can load a separate Q8_0 multi-token prediction head. The four-Spark Pro deployment uses Q2_K text weights. In Pro, the routed expert gate and up projections are Q2_K and the down projections retain MXFP4. These are Xiaomi's MOPD variants. The leaderboard also lists the releases without a post-training suffix. [Flash conversion](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-MOPD-GGUF/blob/c9d8697904d2564708b5197d176543983b9d8db5/README.md) · [Pro conversion](https://huggingface.co/ggml-org/MiMo-V2.6-Pro-MOPD-GGUF/blob/cd70ac158ea81ca857eb0e6feb9fb46d4996a242/README.md)

The lab served the 418.77 GB MOPD Q2_K conversion on four Sparks at 5.85 / 11.74 output tok/s for one / four clients. It scored 17/28 free and 20/28 constrained, with 8/8 exact lookups. [Measurements and reproduction](../../results/mimo-v26-pro-mopd-rpc-2026-11-20/)


The lab’s Flash measurements include the 105.86 GB target on one Spark. [Flash timings, quality and placement](../../results/mimo-v26-flash-mopd-2026-11-20/)

## Runtime qualification

A 639 MB Qwen3 0.6B checkpoint exercised the distributed path on the four physical Sparks. The engine reported all 29 layers offloaded, and all three workers executed CUDA graphs. The two adjacent workers negotiated RoCEv2 and the routed worker used TCP. The service answered the arithmetic prompt on the head. The same pinned engine accepted the real MiMo Pro and Flash tensor metadata in allocation-free mode, reporting 70 and 48 layers respectively. [Hardware and loader receipts](preflight.json) · [Fixture checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/23749fefcc72300e3a2ad315e1317431b06b590a)

## Build once on ARM64

Run from this directory on a 128 GB Spark with Docker, the NVIDIA Container Toolkit, Git and access to NGC. The build targets SM121a with CUDA 13.3 and enables the llama.cpp RPC backend and libibverbs. It produces a relocatable archive of executables and libraries. Each serving node runs it inside the same pinned base image. [Build flags](compile.sh) · [Pinned llama.cpp source](https://github.com/ggml-org/llama.cpp/tree/f498f864fbc0472004ee1c3616c1188c68eb157f)

```sh
work=$PWD/mimo-build
rev=f498f864fbc0472004ee1c3616c1188c68eb157f
base=nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
mkdir -p "$work/src" "$work/engine"
curl -fsSL "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$rev" -o "$work/source.tar.gz"
tar xzf "$work/source.tar.gz" --strip-components=1 -C "$work/src"
cp compile.sh "$work/"
docker run --rm --gpus all --ipc=host -v "$work:/work" --entrypoint bash "$base" /work/compile.sh
tar xzf "$work/runtime.tar.gz" -C "$work/engine"
```

## Stage the weights

For Pro, only the head needs a copy of the 418.77 GB checkpoint. For Flash, stage the 105.86 GB target on the serving Spark. Its optional draft head adds 2.38 GB. The checksums cover the exact files used by these recipes. [Source files and SHA-256 hashes](weights.json)

```sh
hf download ggml-org/MiMo-V2.6-Pro-MOPD-GGUF \
  --revision cd70ac158ea81ca857eb0e6feb9fb46d4996a242 \
  --include 'MiMo-V2.6-Pro-MOPD-Q2_K-*.gguf' --local-dir "$work/pro"
hf download ggml-org/MiMo-V2.6-Flash-MOPD-GGUF \
  --revision c9d8697904d2564708b5197d176543983b9d8db5 \
  --include 'MiMo-V2.6-Flash-MOPD-Q2_K_S-*.gguf' 'mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf' \
  --local-dir "$work/flash"
```

## Launch

On one Spark, the default Flash command below uses ordinary decoding. Append `--mtp "$work/flash/mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf"` to request a two-token draft. With the sidecar loaded, `--draft-tokens 1` and `--draft-tokens 3` select the other tested proposal limits.

```sh
python3 launch.py "$work/engine" serve "$work/flash/MiMo-V2.6-Flash-MOPD-Q2_K_S-00001-of-00002.gguf" \
  --alias mimo26-flash-mopd-q2ks
```

For four Sparks, copy the runtime archive and launcher to the three workers and extract the archive into an engine directory on each. Each worker exposes its GB10 over the trusted fabric address supplied to the launcher. ggml RPC has no authentication. On the lab's switchless ring the two adjacent workers use direct RoCE. The opposite worker binds to its routed fabric address with TCP forced. These commands can each be run from a management machine with SSH access to the corresponding node. [Upstream RPC transport](https://github.com/ggml-org/llama.cpp/blob/f498f864fbc0472004ee1c3616c1188c68eb157f/tools/rpc/README.md)

```sh
# On each adjacent worker, set FABRIC_IP to the address on its link to the head.
python3 launch.py "$work/engine" worker "$FABRIC_IP"
# On the opposite worker, use its reachable routed address.
python3 launch.py "$work/engine" --tcp worker "$FABRIC_IP"
```

On the head, supply worker endpoints in layer order. The default uses four equal layer shares, four slots, 8,192 tokens per slot and loopback port 8000. Prompt caching and reasoning are disabled. For a TCP control on all links, pass `--tcp` before `serve`. The worker processes accept either transport.

```sh
python3 launch.py "$work/engine" serve "$work/pro/MiMo-V2.6-Pro-MOPD-Q2_K-00001-of-00007.gguf" \
  "$ADJACENT_B:50052" "$OPPOSITE_C:50052" "$ADJACENT_D:50052"
```

## Compare Flash weight precision on two Sparks

Stage both Flash conversions on the head. The MXFP4 text checkpoint contains 167.37 GB. Start one adjacent worker with the command above. For each run, give the head its two equal layer shares and use the same four slots and 8,192-token context per slot. Stop the head between runs and keep the worker running.

```sh
hf download ggml-org/MiMo-V2.6-Flash-MOPD-GGUF \
  --revision c9d8697904d2564708b5197d176543983b9d8db5 \
  --include 'MiMo-V2.6-Flash-MOPD-MXFP4-*.gguf' --local-dir "$work/flash"
python3 launch.py "$work/engine" serve "$work/flash/MiMo-V2.6-Flash-MOPD-Q2_K_S-00001-of-00002.gguf" \
  "$ADJACENT_WORKER:50052" --alias mimo26-flash-mopd-q2ks
# After completing the Q2_K_S workload, stop the head and start MXFP4.
python3 launch.py "$work/engine" serve "$work/flash/MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf" \
  "$ADJACENT_WORKER:50052" --alias mimo26-flash-mopd-mxfp4
```

Run the prose, lookup and short-answer workloads with each alias. For the longer constrained panel, run `python3 mmlu_probe.py run "$model" /tmp/mmlu-112.jsonl --one-letter-gbnf --out "$out/grammar112.jsonl"` from the inference recipe directory. [Workload and pinned question set](../../results/mimo-v26-flash-mopd-2026-11-20/#reproduce)
