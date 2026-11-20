# Nemotron 3 Ultra on four DGX Sparks

The 352.31 GB official NVFP4 release splits cleanly at a shard boundary. The target uses the first **108 files (329.88 GB)** and the native MTP drafter uses the last **five (22.43 GB)**. The draft contains **21.47 GB of BF16 routed experts** while the target stores its routed experts as packed NVFP4 with FP8 scales. Copying the complete weights to four machines costs 1,409.23 GB. The five draft files account for 89.73 GB of that footprint. [Pinned model](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/tree/02462641f13d3af838b904f48195b9bb8a1e4ebc) · [Per-file classification and dtype totals](spark-layout.json)

| Consumer | Files | Indexed tensors | Routed expert payload | Other payload | Headers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target | 108 | 197,847 | 289.91 GB | 39.94 GB | 24.50 MB |
| MTP drafter | 5 | 1,040 | 21.47 GB | 0.96 GB | 0.13 MB |

We fetched every complete SafeTensors header at the pinned revision, then opened all 113 as sparse metadata fixtures on an ARM64 DGX Spark. The installed vLLM 0.33.0 target weight mapper and draft name filter assigned all 198,887 tensor names to exactly one consumer. We checked names, shapes, dtypes, offsets and file lengths. The probe reads zero weight payload bytes. A target-only server using 108 files still needs a generation test. [Executable probe](probe.py)

## Recurrent state at four ranks

NVIDIA’s four-Spark guide uses tensor parallelism, FP32 recurrent state, FP8 KV and four MTP proposals. We passed the pinned model configuration to the same vLLM 0.33.0 state-shape and dtype methods used by the Nemotron target. Across 48 Mamba layers, one complete set of temporal states per rank costs **96 MiB in FP32 and 48 MiB in BF16**. With four proposals the separate BF16 convolution states cost another 2.95 MiB per rank in either mode. With drafting disabled those convolution states cost 1.27 MiB. Actual pool size depends on the scheduler, alignment and number of blocks. These are allocation calculations. The lab has yet to measure Ultra generation, accuracy or the effect of changing its state dtype. [NVIDIA’s four-Spark deployment](https://github.com/NVIDIA-NeMo/Nemotron/blob/fe2187ce63f0f872be413c2c673787b9cb99ad31/usage-cookbook/Nemotron-3-Ultra/SparkDeploymentGuide/README.md) · [Reported shapes and dtypes](spark-layout.json)

Run from this experiment directory with `SNAPSHOT` pointing to the pinned checkpoint. The probe uses CPU and writes its measurements to a new output file.

```sh
image=vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
docker run --rm --network none --volume "$SNAPSHOT:/model:ro" \
  --volume "$PWD:/work" --workdir /work --entrypoint python3 "$image" \
  probe.py /model observed.json
```
