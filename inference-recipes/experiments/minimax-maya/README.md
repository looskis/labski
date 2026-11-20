# MiniMax M3.1 Flash Maya-XS on GB10

**Hardware status.** The patched ARM64 engine compiled for SM121a and started its CLI on a DGX Spark. The host arithmetic checks passed. This is a one-Spark inference candidate pending a full checkpoint run. The two text shards and sparse-attention indexer total 92,997,076,736 bytes. The quantizer retains all routed experts and mixes IQ1_S with higher-precision tensors. The upstream model card reports a 5.0% increase in its next-token loss for XS. [Pinned weights and quality evaluation](https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF/blob/0ac574071c3322f02b540f3547f9ef1bc48eb494/README.md) · [Upstream Maya source](https://github.com/mw00/project-maya/tree/e75b3ef7c5f725ef50fafe8ae74b16f77982f85e)

## Build and serve

Run from this directory on the Spark. Use a new working directory on the NVMe drive. The build fetches pinned Maya and ggml source revisions. Preparation verifies all three SHA-256 hashes, indexes the two GGUF shards in place and attaches the dense layers’ indexers. The text-only configuration uses BF16 KV, four 8K slots, no vision and no drafting. The endpoint binds to loopback on port 8000.

```sh
work="$HOME/.cache/labski/minimax-maya"
sh build.sh "$work"
sh prepare.sh "$work"
python3 serve.py "$work" --slots 4
```

The service name is `minimax-m31-flash-maya-xs`. The build and serving commands use the same pinned ARM64 vLLM image as a CUDA toolkit and runtime container. Maya supplies the inference engine. The build extracts NVIDIA’s hash-verified CUDA 13.0 profiler development header, which its MiniMax entrypoint includes. On the lab Spark the resulting executable had SHA-256 `1aeea0f8308afe83202f36c373dbc5db9fd462f05ba9ee168009c76a70f1972c`. [Build](build.sh) · [Weight preparation](prepare.sh) · [Launch configuration](serve.py) · [File hashes](weights.sha256)

## ARM64 changes and checks

At the pinned upstream revision, the MiniMax CPU lane unconditionally included the x86 intrinsic header. The host helpers also used x86-only vector instructions. The patch reuses Maya’s architecture-specific spin hint and adds a portable implementation of the fixed-order dot product, RMS normalization and accumulation. It preserves the existing four groups of eight fused accumulators in the dot product. [Patch](arm64.patch) · [Host arithmetic check](host_check.cpp)

On GB10 the maximum absolute difference against a double-precision reference was 1.26 × 10⁻⁶ over 13 dot-product shapes and 1.24 × 10⁻⁷ over 3,072 normalized outputs. All 6,144 accumulation outputs matched the reference’s float addition order exactly. To rerun the check from this directory:

```sh
c++ -std=c++20 -O3 -ffp-contract=off -I "$work/source/include" -I "$work/source/src/mm3" \
  "$work/source/src/mm3/mm3_host.cpp" host_check.cpp -o "$work/host-check"
"$work/host-check"
```
