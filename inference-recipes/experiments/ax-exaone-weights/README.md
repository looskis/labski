# Four-Spark weight budgets for A.X K2 and K-EXAONE 2.0

A physical ARM64 Spark read the official indexes and SafeTensors headers for the two NVFP4 targets and their DSpark releases. It fetched zero weight payload bytes. The files reveal two different deployment constraints. A.X K2’s 398.28 GB weights have a four-Spark budget, but the pinned stock vLLM image lacks its native model mapping. K-EXAONE’s native model is present in that image, but its 525.25 GB of target tensor data exceed the fleet’s reported physical memory before runtime allocations. Neither target has a measured serving result in this experiment. [A.X K2 inventory](ax.json) · [K-EXAONE inventory](exaone.json) · [Runtime registry evidence](registry.json)

## A.X K2

All 186,942 names in the 346 indexed files matched their headers. Routed experts occupy **380.51 GB** including scales. With contiguous assignment of the 256 experts to four ranks, each rank owns 95.13 GB of expert data. Keeping every common tensor on each rank adds 17.75 GB, for **112.88 GB per rank** before headers. That calculation describes one possible disk layout. A serving engine can shard common tensors and needs additional memory for KV, temporary buffers and its runtime. SKT’s separate DSpark file contains 44 tensors and adds 6.84 GB. [Target files, dtypes and rank byte counts](ax.json) · [Draft header](ax-draft.json)

The stock ARM64 vLLM 0.33.0 image registers `DSparkDraftModel` to a DeepSeek V4 implementation and has no explicit `AXK2ForCausalLM` entry. The SKT fork registers the A.X target to `axk2` and the same draft architecture name to a Qwen3 DSpark implementation. The lab checked those class definitions at the pinned fork revision. A matching architecture string on the draft alone is insufficient to select a compatible target/draft runtime. SKT documents its four-B200 NVFP4 recipe on that fork. [Registry comparison](registry.json) · [Pinned SKT registry](https://github.com/SKT-AI/vllm/blob/2146b39e34e1438b1e0b5e5f4aa4163fe95d909a/vllm/model_executor/models/registry.py) · [Publisher deployment](https://huggingface.co/skt/A.X-K2-NVFP4/blob/9e2e804e80f8d1b3afba5d7938173cec1ed46b49/README.md)

## K-EXAONE 2.0 750B A37B

The official NVFP4 release has 53 indexed files and 220,676 tensors. Its 526.33 GB includes 1.057 GB of native MTP tensors and 26.36 MB of headers. Even after subtracting both, the **525.251 GB target exceeds the four nodes’ combined 522.644 GB of Linux-reported physical memory by 2.607 GB**. The last shard mixes 5.56 GB of target data with the MTP tensors, so omitting that whole shard would discard target weights. [Shard inventory](exaone.json) · [Official NVFP4 release](https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B-NVFP4/tree/360716a7e9cb1161906061d23fb9c68fe3767d6c)

Six routed MLP layers (2, 3, 4, 75, 76 and 77) retain **115.96 GB of BF16 routed weights**. Each of those layers costs 19.327 GB. Each of the other routed layers costs 5.436 GB in the publisher’s packed layout including its scales. Substituting that measured byte layout would save **13.892 GB per converted layer**. The table assumes even target placement across four GPUs. Its headroom includes all memory the OS and runtime would also need. Requantization quality, runtime support for the modified weights and generation speed remain unmeasured. [Per-layer dtype and scale bytes](exaone.json)

| BF16 routed layers converted | Target weight data per rank | Physical memory remaining per rank |
| ---: | ---: | ---: |
| 0 | 122.29 GiB | 0.61 GiB short |
| 1 | 119.06 GiB | 2.63 GiB |
| 3 | 112.59 GiB | 9.10 GiB |
| 6 | 102.89 GiB | 18.80 GiB |

The separate DSpark repository indexes 17 weight files totaling **1.503 TB**. Its 62 draft tensors all reside in the **5.065 GB** `model-dspark.safetensors` file. A draft-only inspection therefore needs just that file’s 6,912-byte header. The compatibility of these five draft layers with the NVFP4 target needs a live test. [Draft component and index](exaone-draft.json) · [Pinned DSpark snapshot](https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B-DSpark/tree/e42409dbd652ade0bc043fd85ef0c86044f97b46)

## Expert conversion test fixed before scoring

The first precision test samples one expert in each of the six BF16 layers, selecting the expert by a fixed SHA-256 rule. It reads the complete gate, up and down matrices for those six experts. The 18 tensors contain **452,984,832 source bytes**, fetched as individual ranges on a Spark. Two conversions use the original BF16 values as their reference.

- E4M3 FP8 uses one FP32 maximum-derived scale per 128 × 128 tile.
- NVFP4 uses signed E2M1 values, one E4M3 scale per 16 adjacent weights and one FP32 scale from the maximum of the complete tensor. Magnitude ties select the lower value.

The program records squared weight error and squared projection error for 32 fixed Gaussian input vectors rounded to BF16. Reference and reconstructed weights both use FP32 matrix multiplication with those same vectors. This measures weight-only error on sampled experts. Native low-precision activation error, task accuracy, and serving speed require separate tests. [Sampling and conversion code](expert_error.py)

After preparing the EXAONE headers, run the following on a Spark with PyTorch installed. The sampling step uses only the Python standard library. Keep the downloaded matrices on the Spark used for scoring.

```sh
python3 expert_error.py fetch ./header-cache ./expert-samples
python3 expert_error.py evaluate ./expert-samples ./expert-scores.json
```

## Reproduce on a Spark

The Python standard-library probe fetches the pinned metadata and two exact header ranges for each selected file. It checks the name and byte extent of each tensor, retains a local header cache and records per-header SHA-256. The DSpark run for K-EXAONE reads the full index and then only the draft shard’s header. Use new result filenames.

```sh
for model in ax exaone ax-draft exaone-draft; do
  python3 probe.py "$model" ./header-cache "$model-replay.json"
done
```
