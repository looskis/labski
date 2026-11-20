# Kimi K3.1 Pro with NVMe expert streaming

The original checkpoint has 1.56 TB of SafeTensors. Even the 592.68 GB IQ1_S text GGUF exceeds the physical memory of four lab Sparks. Argonaut Labs’ CUDA runtime offers another layout. Its Q8_0 dense matrices and floating-point vectors reside in memory, while a cache fetches the selected native MXFP4 experts from NVMe. The checkpoint has 92 MoE layers, 896 experts per layer and 16 active experts per layer per token. The runtime’s authors demonstrated their CUDA implementation on a 96 GB GPU. [Pinned engine and weight layout](https://github.com/argonautlabsai/argodrive-ds4-kimi-k3.1/blob/4673fc5ea0649371842f551eaffedd8da50380f1/tools/kimi-k3/README.md) · [Original checkpoint](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/tree/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e)

## What the lab verified

On one GB10 the pinned engine compiled to an AArch64 executable containing SM121a CUDA kernels and loaded its streaming CLI. Two original checkpoint shards totaling 19.33 GB passed SHA-256 verification. The upstream extractor produced 51 resident planes and all 896 experts in the first MoE layer. Every packed expert matched the original six MXFP4 planes in the engine’s required order. Full-model generation and quality remain unmeasured here. [Build and packing receipts](build-check.json) · [Per-file checksums](first-shards.sha256.gz)

The Spark’s local NVMe returned **10.59 GB/s** with 16 concurrent readers on those real expert files. Four readers reached 10.49 GB/s. This isolates the cost of cache misses. One hypothetical token missing all 16 experts in every one of the 92 MoE layers needs 25.83 GB of reads, about 2.44 seconds at the measured 16-reader rate. A running model’s cache hits, computation and read overlap change its latency. [Direct-I/O harness](read_trace.py) · [All 16 trials and device counters](read-trace.jsonl.gz)

| Concurrent readers | GB/s | Median time to read one set of 16 experts |
| ---: | ---: | ---: |
| 1 | 9.23 | 30.29 ms |
| 4 | 10.49 | 26.71 ms |
| 8 | 10.42 | 26.83 ms |
| 16 | 10.59 | 26.42 ms |

The harness reads a fixed synthetic schedule of 64 routing steps, picking 16 distinct 17,547,264-byte expert files per step from the verified layer. It performs one full aligned `O_DIRECT` read per selected file and waits for all 16 before advancing. Each setting used four excluded warm-up steps and four trials with counterbalanced order. Every trial’s process-level physical reads equaled its requested bytes. The whole-device counter differed by at most 692,224 bytes per trial. The table reports medians of the four trial throughputs and four per-trial median step times. Reproduce with `python3 read_trace.py "$work/experts" --device nvme0n1 --out disk-trials.jsonl`, substituting the block device that contains the expert store.

## Build for a Spark

Run on Linux ARM64 with Docker and the NVIDIA Container Toolkit. The engine’s Makefile includes a native SM121a target for its MXFP4 kernels. Use an ext4 or other sparse-file-capable filesystem for the final model.

```sh
work=$PWD/kimi31-build
base=nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
rev=4673fc5ea0649371842f551eaffedd8da50380f1
mkdir -p "$work/src"
curl -fsSL "https://codeload.github.com/argonautlabsai/argodrive-ds4-kimi-k3.1/tar.gz/$rev" -o "$work/source.tar.gz"
tar xzf "$work/source.tar.gz" --strip-components=1 -C "$work/src"
docker run --rm --gpus all -v "$work/src:/engine" -w /engine --entrypoint make "$base" -j5 ds4 CUDA_ARCH=sm_121a
docker run --rm --gpus all -v "$work/src:/engine:ro" -w /engine --entrypoint /engine/ds4 "$base" --help
```

## Prepare one source shard at a time

The pinned index maps every expert’s six planes to a single source file. This lets a Spark pack and verify one shard before downloading the next. The 94 text shards total 1,560,398,422,888 bytes. The packed expert store contains 82,432 files and occupies 1,446,456,066,048 bytes. Allow additional space for the 2,580 resident tensor planes, the roughly 62 GB converted spine, and one source shard. The two remaining upstream files hold the vision tower and multimodal projector. [SHA-256 manifest of all text shards](source.json)

Stage the pinned metadata, then run the following from the recipe directory. If you already have the source files, pass each one directly to `pack_shard.py`. Each invocation verifies the source SHA-256 and model index, runs the pinned upstream extractor, and compares every packed file against the original planes. It writes per-file checksums and a per-shard receipt. Completed shards can be resumed individually. [Packing and verification](pack_shard.py)

```sh
export work
rev=d9e99b0792a5e93b78ea9bea62a68167e36ddd9e
hf download moonshotai/Kimi-K3.1-Pro --revision "$rev" --local-dir "$work/indexer-input" \
  --include model.safetensors.index.json config.json generation_config.json tokenizer_config.json \
  tiktoken.model LICENSE tokenization_kimi.py encoding_k3.py README.md
python3 - <<'PY'
import json, os, subprocess, sys, tempfile
from pathlib import Path

work = Path(os.environ["work"]).resolve()
spec = json.loads(Path("source.json").read_text())
for name in spec["text_shards"]:
    if (work / "receipts" / (name + ".json")).exists():
        continue
    with tempfile.TemporaryDirectory(dir=work) as scratch:
        subprocess.run(["hf", "download", spec["repository"], name, "--revision", spec["revision"],
                        "--local-dir", scratch], check=True)
        subprocess.run([sys.executable, "pack_shard.py", str(Path(scratch) / name),
                        str(work / "indexer-input/model.safetensors.index.json"), str(work),
                        str(work / "src/tools/kimi-k3/k31_extract.py")], check=True)
PY
```

The `indexer-input` directory retains the exact headers and sparse indexer planes at their source offsets. After all 94 receipts exist, finalization verifies every packed file, runs the upstream DSA indexer extractor, builds the Q8 spine, hashes its compact byte extent and extends the file to the required sparse length. A complete indexer has five tensors in each of the 24 full-attention layers. Both intermediate and final products are local. [Finalization](finalize.py)

```sh
cp pack_shard.py finalize.py source.json "$work/"
docker run --rm -v "$work:/model" --entrypoint python3 "$base" \
  /model/finalize.py /model /model/src/tools/kimi-k3
```

## CUDA check with 16 real Kimi experts

The engine's native routed-expert CUDA function processed a fixed 3,584-dimensional input, 16 selected experts from layer 1 and fixed routing weights. Each of three fresh processes first warmed the kernels and reader pools on 16 disjoint experts. The first call on the measured set read 280,756,224 bytes and took a median 27.84 ms. The next five calls per process read zero additional expert bytes and took a median 3.20 ms. All 18 calls returned the same 3,584 float32 values byte for byte. This measures one routed-expert branch with a synthetic input and route. [All calls and reference errors](expert-check.json) · [CUDA harness](expert_probe.c)

An independent NumPy float64 calculation decoded the original E2M1 nibbles and E8M0 scales, applied the gate, up and down projections and combined the same routing weights. Across the output vector, maximum absolute error was 3.08 × 10⁻⁶ and relative L2 error was 3.69 × 10⁻⁶. [Reference implementation](check_expert.py)

After packing the first MoE layer and building the engine, run these commands from the recipe directory. The executable links against the same engine objects as the server. The host-cache setting requests zero GiB, and the runtime retains its internal minimum of 64 host expert slots.

```sh
mkdir -p "$work/probe"
cp expert_probe.c expert-probe.mk check_expert.py "$work/probe/"
docker run --rm --gpus all -v "$work/src:/engine" -v "$work/probe:/probe" -w /engine \
  --entrypoint make "$base" -f Makefile -f /probe/expert-probe.mk expert-probe CUDA_ARCH=sm_121a
for run in 1 2 3; do
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --memory 16g --memory-swap 16g \
    -e DS4_K3_CACHE_RAM_GB=0 -e DS4_K3_CACHE_VRAM_INIT_SLABS=32 \
    -v "$work/probe:/probe" -v "$work/experts:/experts:ro" --entrypoint /probe/expert-probe \
    "$base" /experts "/probe/output-$run.f32" > "$work/probe/run-$run.jsonl"
done
docker run --rm -e OPENBLAS_NUM_THREADS=4 -v "$work/probe:/probe:ro" -v "$work/experts:/experts:ro" \
  --entrypoint python3 "$base" /probe/check_expert.py /experts /probe/output-1.f32
```

## Full-model request measurement

The native streaming API supplies visible text and an end marker. It writes prompt and generation token counts to the server log. The serial request probe joins each stream with its timing line and records first-delta time, first visible text, total request time, exact answer, process physical reads and whole-device reads. The pinned server runs its own one-token warm-up before listening. Repeated prompts can benefit from both expert residency and prompt reuse. [Request probe and fixed prompts](serve_probe.py) · [Pinned server implementation](https://github.com/argonautlabsai/argodrive-ds4-kimi-k3.1/blob/4673fc5ea0649371842f551eaffedd8da50380f1/ds4_k3_serve.c)

On the Linux serving host, set `server_pid` to the host PID of the native engine and `server_log` to its combined output file. Use an exclusive local endpoint during the measurement so each response corresponds to one native timing line. Supply the block device containing the expert files and a new results path. The example issues three questions twice each.

```sh
python3 serve_probe.py --url http://127.0.0.1:8000 --pid "$server_pid" \
  --server-log "$server_log" --device nvme0n1 --max-tokens 16 \
  --order sum sum city city lookup lookup --out requests.jsonl
```
