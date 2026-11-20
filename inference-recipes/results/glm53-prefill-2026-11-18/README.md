# GLM‑5.3‑Flash cold prefill on two Sparks

On November 18, 2026 the lab ran the same rendered token IDs through two serving stacks on a pair of 128 GB GB10 Sparks. At 32,768 input tokens, the faster NVFP4 configuration returned its first token in a median **13.15 s**, compared with **17.22 s** for the TensorFold EXL3 reference. Across the three tested prompt lengths, the ratio of effective prefill rates was 1.31–1.32×.

| Input tokens | TensorFold EXL3, input tok/s | vLLM NVFP4, 4K chunk | vLLM NVFP4, 8K chunk | EXL3 TTFT | NVFP4 8K TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1,657 | 2,163 | 2,180 | 2.47 s | 1.88 s |
| 32,768 | 1,903 | 2,224 | 2,492 | 17.22 s | 13.15 s |
| 65,536 | 1,894 | 2,207 | 2,476 | 34.60 s | 26.47 s |

Each cell is the median of three single-client cold requests after an excluded warm-up. Each request generated exactly one token. Effective prefill rate is input tokens divided by client time to first output token. The comparison changes both runtime and checkpoint. The 8K profile also changes NCCL and CPU thread settings together. Decode throughput and the effects of individual settings need separate experiments. [Per-request timings](requests.csv) · [Precision, quality and run summary](summary.json)

The faster configuration answered 10/10 short questions in one pass and 9/10 on repeat. Both long-retrieval checks passed. An identical greedy reversal prompt produced correct and incorrect answers across runs. This result supports a cold-prefill speed claim with a measured quality caveat.

## November 19 hardware replay

The portable launcher served the same pinned NVFP4 weights and image on the two lab Sparks. All nine timed requests and the excluded warm-up passed the cold-cache and token-count checks. [New request timings](replay-2026-11-19.csv)

| Input tokens | Median client TTFT | Median effective prefill |
| ---: | ---: | ---: |
| 4,096 | 1.87 s | 2,188 input tok/s |
| 32,768 | 13.07 s | 2,507 input tok/s |
| 65,536 | 26.42 s | 2,481 input tok/s |

The November 18 TensorFold reference is the historical comparison. On November 19 only the NVFP4 side was rerun.

## Reproduce the NVFP4 configuration

Use the [prefill recipe](../../recipes/glm53-nvfp4-prefill.json) on two 128 GB Sparks with dual 200 Gb/s ConnectX‑7 rails. The measured transport was NCCL NET/IB with GPU Direct RDMA disabled, driver 580.178.04. The serving settings include TP2, eager execution, 2 GiB of FP8 KV cache per rank, an 8,192-token chunk, one active sequence and disabled prefix caching. The checkpoint and ARM64 vLLM 0.33.0 image are pinned. The portable launcher uses the recipe ID as the API alias and omits the archive’s profiling-only output mount. [Launch instructions](../../README.md#one-endpoint-on-multiple-sparks)

Use the ordinary two-node launch commands with `glm53-nvfp4-prefill` as the recipe ID. On the API host, from this result directory:

```sh
python3 replay.py --model glm53-nvfp4-prefill --out /tmp/glm53-nvfp4-replay.csv
```

The replay verifies the SHA-256 of the original [compressed token corpus](corpus.jsonl.gz) after decompression. It checks the actual input and output counts and cold-cache evidence before reporting a timing. Use the same corpus and change the API alias and URL for the reference engine. The archived reference used TensorFold 1.1.9 (engine commit `f350cd3e`) with Mia’s EXL3 weights at `76c0b5173166d2795dd48860f45d8224817f894c`, DFlash2, FP8 KV, eight server slots and prompt caching disabled. The lab enabled its additional wide-prompt patch (`TF_GLM_PROMPT_WIDE=1`, patch SHA-256 `c82d6718216d909f7802735f0c29804ebbcc0bff3f19d7a1d37391a01ddcc88e`). The comparison measures that tuned build. An upstream-only rebuild is a different reference configuration. [Pinned upstream recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/tree/80be4d6882837d8e0840827ae2aab6124fd48da3)

The records in this directory come from the lab’s archived experiment. They were checked against the saved request rows during import. The [GLM FP4-Local serving candidate](../../recipes/glm53-flash-fp4.json) uses a different checkpoint, loader and workload.

## Isolating prefill chunk size

A second November 19 run changed only the maximum batched token count from 8,192 to 4,096. It used the same two Sparks, checkpoint, image, dual-rail RoCE transport, greedy one-token responses, 2 GiB FP8 KV allocation, NCCL settings, CPU threads and cold token-ID corpus as the 8K replay above. Each median covers three timed requests after the excluded warm-up. Run order was 8K then 4K in separate server lifetimes. [4K control recipe](../../recipes/glm53-chunk4096.json) · [4K control request timings](replay-4096-2026-11-19.csv)

| Input tokens | 4K chunk, input tok/s | 8K chunk, input tok/s | Change with 8K |
| ---: | ---: | ---: | ---: |
| 4,096 | 2,199 | 2,188 | −0.5% |
| 32,768 | 2,263 | 2,507 | +10.8% |
| 65,536 | 2,248 | 2,481 | +10.4% |

The larger chunk helped once the prompt spanned multiple chunks. The 4K result is within 1% between settings, with three trials per cell. To reproduce the control, run the same replay with `--model glm53-chunk4096`. These timings measure prefill only.
