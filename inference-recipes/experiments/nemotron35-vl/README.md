# Nemotron 3.5 Super VL on one 128 GB Spark

NVIDIA’s NVFP4 release contains **75.01 GB** in 88 indexed SafeTensors files. The checkpoint includes the text backbone, native multi-token prediction (MTP) and vision weights. We matched all 169,505 tensor names to the pinned index and checked every payload offset against the shard length reported by the Hub’s range responses. This inventory read headers only. [Per-component totals and all 88 header hashes](weight-layout.json) · [Official checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/tree/ae1368329b1fc76b71534c5d04b87b35d9cc6e15)

| Component | Tensors | Stored tensor bytes |
| --- | ---: | ---: |
| Text backbone and output head | 164,989 | 71,532,437,000 |
| Native MTP | 4,120 | 1,769,654,296 |
| Vision encoder | 391 | 1,307,223,080 |
| Vision projection | 5 | 377,502,720 |

The tensor payload totals 74,986,817,096 bytes. The 88 headers add 23,371,144 bytes. The index metadata reports 75,010,189,968 bytes, 1,728 bytes above the combined file lengths. The table uses the SafeTensors offsets. Run `python3 components.py "$SNAPSHOT"` against the complete pinned checkpoint to reproduce it. [Accounting program](components.py)

NVIDIA’s separate LiLiCorr draft adds one **1,317,952,596-byte** weight file. On a physical ARM64 Spark we checked that the pinned vLLM 0.33.0 image registers the target architecture, its text backbone and the LiLiCorr draft. The lab’s serving comparison is queued for ordinary decoding, five native MTP proposals and seven external draft proposals at a common 8K context. It includes text generation and eight image-only inventory lookups, each with a known answer. Local throughput and task scores are pending. [Spark registry receipt](spark-registry.json) · [Pinned external draft](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/tree/292bf1efc06ba39567da7351f2ad7a522afa3959) · [NVIDIA’s one-Spark LiLiCorr recipe](https://github.com/NVIDIA/local-ai-inference-recipes/tree/bb515086841e7a81e7d51925561338e7dee35023/models/nvidia/nemotron-3.5-super-vl-120b-a12b/dgx-spark/linux/vllm/nvfp4)
