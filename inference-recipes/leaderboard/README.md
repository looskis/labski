# Open-weight inference queue

The November 19, 2026 snapshot contains 44 releases with a current Artificial Analysis Intelligence Index score. The queue retains the highest-scoring effort setting per release and sorts by its unrounded score. A leaderboard result belongs to the provider configuration named in the row. The Spark experiments use their own prompts, reasoning settings and metrics. [Ranked queue](2026-11-19.csv) · [Source leaderboard](https://artificialanalysis.ai/leaderboards/models?is_open_weights=open_source) · [Snapshot extractor](snapshot.py)

## Which checkpoints merit a hardware run?

| Rank | Model | Root weight files | Spark experiment |
| ---: | --- | ---: | --- |
| 1 | Kimi K3.1 Pro | 1,561 GB upstream | The upstream and NVIDIA NVFP4 downloads both exceed the four-node resident weight budget. An offload experiment needs a separate latency and quality target. |
| 2 | DeepSeek V4.1 Flash 1017 | 510 GB upstream | TensorFold's checkpoint converter produces about 320 GB of rank files and serves the native checkpoint on a four-Spark ring. This is an external four-node baseline. [Upstream recipe and measurements](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/tree/8ffa67d53ef15bab997399135f85e8b0910f30be) |
| 3 | GLM-5.3 | 756 GB upstream, 464 GB NVIDIA NVFP4 | Four-node fit trial for NVFP4. At an even split the weight files consume about 108 GiB per rank, so runtime overhead can decide the outcome. |
| 4 | Thinking Machines Inkling 2 Small | 393 GB native, 623 GB MXFP8 | Trial on four nodes at native precision, 4K context, C1. The publisher specifies 500 GB of aggregate GPU memory. The Sparks share their memory with Linux. Use the GB10 SGLang candidate. The stock vLLM image’s expert kernel requires a different GPU architecture. [Runtime probe](../results/inkling2-gb10-runtime-2026-11-19/) [Checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Small/tree/4d564489295e30e9341a9822a8046f0c9ded1a7b) · [Candidate](../recipes/inkling2-small-native.json) |
| 7 | MiniMax M3.1 Flash | 250 GB upstream | Four-node launch candidate with a bundled draft head. [Recipe](../recipes/minimax31-flash-tp4.json) |
| 8 | Qwen3.8 Flash-Next | 360 GB upstream, 133 GB NVIDIA mixed FP4 | Four nodes for the upstream payload under the planning allowance, two for the smaller checkpoint. Start with no drafting at 8K context. [Pair recipe](../recipes/qwen38-flash-next-nvfp4.json) |
| 10 | GLM-5.3-Flash | 328 GB upstream, 204 GB NVIDIA NVFP4 | Two-node NVFP4 serving and cold-prefill timing verified. This is a distinct release from rank 3. [Measured recipe](../results/glm53-prefill-2026-11-18/) |
| 16 | Qwen3.8 27B | 55.6 GB BF16, 21.9 GB NVIDIA mixed FP4 | Both served on one Spark. Mixed precision delivered 2.78–2.79× the BF16 throughput at one client and scored five to nine fewer correct answers on the 112-item probe. [Measurements](../results/qwen38-27b-precision-2026-11-19/) [BF16 recipe](../recipes/qwen38-27b-bf16.json) · [Mixed FP4 recipe](../recipes/qwen38-27b-nvfp4.json) |

The [checkpoint inventory](weights-2026-11-19.csv) covers all 44 ranked releases. Each row links an immutable Hub revision. Sizes sum the root SafeTensors files, including auxiliary or draft files in the root. The count omits duplicate consolidated copies where the conventional weight files are also present. Model variants such as Xiaomi's RL and MOPD are listed separately because the leaderboard label alone does not identify the locally downloaded variant.

The planning column picks the smallest of one, two or four Sparks allowing 110 GiB per node for weight files. Each lab Spark reports about 121 GiB total. The allowance is a consistent way to triage downloads. An engine may repack, replicate or offload tensors, and individual ranks may have different memory demands. Runtime support, KV cache, working memory, interconnect and a successful completion determine actual serving fit. Rank 2 illustrates the difference.

Refresh the scores with `python3 snapshot.py > YYYY-MM-DD.csv`. Use the linked manifests to recheck byte counts at the recorded checkpoints. Record successful configurations and measured limitations alongside the [inference results](../results/).
