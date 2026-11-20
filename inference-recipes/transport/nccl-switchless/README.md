# Switchless NCCL build for the Inkling SGLang runtime

On the lab’s four-node ConnectX ring the pinned SGLang image bundled NCCL 2.30.7. Its CUDA all-reduce failed at queue-pair setup with a connection timeout. With the same four ranks, replacing that library with a ring-specific NCCL 2.31.2 build completed all 96 rank-level checks across three message sizes. An additional control preloaded the unmodified NCCL 2.31.2 from the vLLM image into SGLang. All four ranks exited before completing the first collective. The passing run used the patched library and ring-only initialization. The Qwen vLLM recipe uses the stock NCCL 2.31.2 from its own image. [Inkling runtime measurements](../../results/inkling2-gb10-runtime-2026-11-19/) · [Four-node vLLM results](../../results/qwen38-27b-scaling-2026-11-19/)

## Build on a Spark

Use an ARM64 toolchain and the CUDA Toolkit. The lab built this on GB10 with CUDA 13.0 and compiled SM121 device code. Run from this directory.

```sh
patch_file="$PWD/nccl-2.31.2.patch"
git clone https://github.com/NVIDIA/nccl.git nccl-2.31.2
cd nccl-2.31.2
git checkout --detach 7b83616df3ae082a1f32bb74c27458bfe8153a13
git apply --check "$patch_file"
git apply "$patch_file"
make -j4 src.build CUDA_HOME=/usr/local/cuda \
  'NVCC_GENCODE=-gencode=arch=compute_121,code=sm_121' \
  NCCL_GIT_COMMIT_HASH=7b83616df3ae082a1f32bb74c27458bfe8153a13
sha256sum build/lib/libnccl.so.2.31.2
```

Install the resulting library at a local path on each rank. Verify that its SHA-256 matches on all four. The lab binary’s digest was `9271a924253b2fff89f5ef82a61a7244c455c230b77d33a24ce95a288fd74d3b`. Add `--ring-nccl /absolute/path/libnccl.so.2.31.2 --ring-only-init` to the SGLang `--transport ring-roce` command. This mounts the library read-only, sets the SGLang loader path, preloads it for PyTorch and selects ring-only initialization. [Four-rank launch and addressing](../../ring-tcp.md#use-roce-on-the-same-four-cables)

## Source and attribution

Labski adapted Sparkring’s switchless-cycle changes for NVIDIA NCCL 2.31.2. The patch advertises the two eligible RoCE devices when matching the remote link subnet and adds an opt-in flag to skip tree and PAT connection setup. Use the ring algorithm with this flag. The affected NCCL files carry Copyright © 2015–2026 and 2016–2026 NVIDIA Corporation & Affiliates respectively. The patch and upstream code use Apache License 2.0. [Original Sparkring patch](https://github.com/FujitsuPolycom/sparkring/blob/dcfc97430ea8095dd31233b6595bdd14a4b63097/spark_transport/nccl/nccl-2.30.7-switchless-cycle.patch) · [NVIDIA source revision](https://github.com/NVIDIA/nccl/tree/7b83616df3ae082a1f32bb74c27458bfe8153a13) · [License](LICENSE) · [Sparkring notice](Sparkring-NOTICE.txt)
