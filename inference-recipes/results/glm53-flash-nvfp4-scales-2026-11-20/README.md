# Refitting GLM-5.3-Flash NVFP4 scales on DGX Spark

On one GB10, an offline change to GLM-5.3-Flash’s existing 4-bit weights reduced sampled weight error from **9.21% to 7.75%** relative to the original BF16 weights. The same native B12X matrix multiply reduced output error from **13.42% to 12.41%** on Gaussian inputs and from **13.06% to 12.09%** on inputs with sparse outliers. All 18 sampled projections improved on both input sets. The packed weights and scales occupied exactly the same number of bytes in all three configurations. [Per-projection measurements](projections.jsonl) · [Source byte ranges and digests](panels.json)

| Offline weight preparation | Weight error | Native output error, Gaussian | Native output error, sparse outliers |
| --- | ---: | ---: | ---: |
| Published NVIDIA NVFP4 | 9.21% | 13.42% | 13.06% |
| Change only block scales | 8.79% | 13.13% | 12.80% |
| Change block scales and 4-bit codes | 7.75% | 12.41% | 12.09% |

Each entry is the square root of summed squared error divided by summed squared BF16 reference norm. We summed both quantities across panels before taking the square root. Each output reference used FP32 multiplication of the same BF16 inputs and original BF16 weights with TF32 disabled. The three native paths used identical activation quantization, input and weight global scales, BF16 output dtype and B12X kernel. These measurements cover matrix output fidelity on generated inputs. Model-level answer quality and serving speed remain unmeasured for the refitted weights. [Pinned NVIDIA checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/1d00738646cbb0c6d0a9eb403dac5f366f1473a7) · [Pinned Z.ai BF16 reference](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/tree/f9fed3bbc82291073373d67452e9f25317ef7577)

## Which bytes changed?

Each group shares one E4M3 scale across 16 E2M1 weights. With the published 4-bit codes fixed, least squares gives an optimal group scale. We rounded that scale to the nearest representable value and kept the original on ties. This changed 62,994 of 245,760 scale bytes and preserved all 3,932,160 weight codes. The second fit tested all 126 finite positive E4M3 scales, chose the nearest signed E2M1 value for every weight at each scale, and kept the group with the smallest squared distance to BF16. It changed 162,889 scale bytes and 1,380,834 weight codes. The representation, block size and tensor-global scales stayed fixed. [NVIDIA Model Optimizer’s packing and global-scale convention](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py)

We sampled early, middle and final routed layers 3, 24 and 44. At each layer a fixed hash selected two of 288 experts. Another hash selected a contiguous 64-row panel from each gate, up and down projection. Every panel used all input columns and 64 input vectors per distribution. For the outlier workload we amplified every 64th coordinate eightfold. Both workloads used the same seeded draws and were normalized to 80% of the activation quantizer’s maximum representable magnitude. These are six experts and two synthetic input distributions. [Sampling and native comparison](../../experiments/glm53-nvfp4-scales/)

## All nine dense projections in the first three layers

We then refitted the complete dense prefix that every text token traverses before the first routed layer. It contains 452,984,832 gate, up and down weights. On all nine full matrices, each fit reduced both weight error and native GB10 output error for both input distributions. The same quantized input and native B12X kernel were used for each comparison. [Per-matrix results](dense-prefix/projections.jsonl) · [Source byte ranges](dense-prefix/matrices.json)

| Offline weight preparation | Weight error | Native output error, Gaussian | Native output error, sparse outliers |
| --- | ---: | ---: | ---: |
| Published NVIDIA NVFP4 | 9.24% | 13.32% | 13.19% |
| Change only block scales | 8.89% | 13.08% | 12.93% |
| Change block scales and 4-bit codes | 7.85% | 12.38% | 12.23% |

The fixed-code candidate changed 6,447,486 of 28,311,552 block-scale bytes. The joint fit changed 18,408,671 block-scale bytes and 160,540,200 4-bit codes. We verified both resulting patches by digest on the two serving ranks. These are full-weight and synthetic-activation measurements. The effect on token likelihood and generated answers still needs the end-to-end comparison. [Scale-only receipt](dense-prefix/scale_only.json) · [Joint-fit receipt](dense-prefix/scale_and_codes.json) · [Reproduction and serving commands](../../experiments/glm53-nvfp4-scales/)

The full original checkpoint served on two GB10s with the replacement hook in verification-only mode. It scored 22/28 free answers and 8/8 on 1,500-record lookups. [Original checkpoint control and per-article likelihood](serving/)

## Reproduce on one Spark

Run from `inference-recipes/experiments/glm53-nvfp4-scales`. The sampler uses pinned Hub byte ranges. To reuse a local copy of NVIDIA’s checkpoint add `--nvfp4 /path/to/its/pinned/snapshot`. The sampler records a digest for every source slice. The comparison checks those digests before loading the samples.

```sh
python3 sample.py samples
image=vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
docker run --rm --gpus all --ipc=host --network none --memory 16g \
  -v "$PWD:/work" -w /work --entrypoint python3 "$image" \
  compare.py samples --out comparison.jsonl
```
