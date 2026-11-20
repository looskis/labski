# Inkling 2 Small on GB10

A single-rank hardware probe found an architecture mismatch before loading the 393 GB checkpoint. The designated vLLM ARM64 image recognizes Inkling 2, but its routed-expert implementation loads the FlashInfer SM100 module and requests kernels for CUDA major architecture 10. The fleet's GB10 reports compute capability **12.1**. Executing that architecture-selection call on a Spark raised `RuntimeError: No supported CUDA architectures found for major versions [10].` PyTorch reported `2.14.0+cu130` and vLLM reported `0.1.dev25249+gb974b306f.cu130`. This is a kernel compatibility result. It does not measure model memory, quality or throughput. [vLLM deployment guide](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Small) · [Probe source](probe.py)

```sh
docker run --rm -i --gpus all --entrypoint python3 \
  vllm/vllm-openai:inkling-2-small@sha256:bc9f13e3fca71bea6a8603cc880fc705b6ebe3c21b438371d744ace7de07f0e7 \
  - < probe.py
```

The four-node candidate uses Aiden Le's pinned GB10 SGLang image and its FlashInfer expert hooks. It retains the official checkpoint, a BF16 KV cache, and a short 4K text workload. Expert parallelism distributes whole experts. Plain decoding and an 8K total token pool keep the initial fit experiment small. For the lab's routed ConnectX ring the launcher offers a TCP collective baseline. The original author's four-node deployment uses switched RoCE and a different KV and drafting configuration. [Lab candidate](../../recipes/inkling2-small-native.json) · [Upstream GB10 implementation](https://github.com/aidendle94/sparkrun-recipes/tree/3d7e798cfaecf1298576b79c945455bb5a75ac46/recipes/inkling-2-small/sglang)

The SGLang image runs on all four lab hosts. Its OCI revision label is `7832fd9+image`, and the baked `/patches/sitecustomize.py` has SHA-256 `9545ffa8280ce5210df0b0134802adae323307b12978ffc54b146e78e3419ac8`. The ARM64 image includes the FlashInfer MoE and paged-decode hooks on its Python path. The installed launcher advertises every flag used by the lab recipe. The later upstream recipe reference describes configuration changes since that image build. These checks cover the image and command line. A full-checkpoint completion is the serving test. [Source revision recorded by the image](https://github.com/aidendle94/sparkrun-recipes/tree/7832fd9d9d123837973b05c16197cac98370804b/recipes/inkling-2-small/sglang)
