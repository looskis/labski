# labski

labski publishes reproducible experiments on local AI hardware. Each research scope keeps its own code, hardware assumptions and results.

| Research scope | First platform | What you can reproduce |
| --- | --- | --- |
| [Inference recipes](inference-recipes/) | Four NVIDIA DGX Sparks | Serve open-weight models and measure how checkpoint precision, speculative decoding and the number of Sparks affect inference. |

## Measured on DGX Spark

- **Qwen3.8 27B on one Spark.** Mixed FP4 delivered 2.78–2.79× the BF16 throughput on the same machines at one client. BF16 scored 83/112 on a short-answer diagnostic and mixed FP4 scored 78/112. [Timing, paired answers and recipes](inference-recipes/results/qwen38-27b-precision-2026-11-19/)
- **GLM‑5.3‑Flash on two Sparks.** An 8K prefill chunk improved 32K-input throughput by 10.8% over a 4K chunk. A separate TensorFold kernel experiment gained 4.3% on the same-length workload while matching the control's sampled answers and likelihoods. [Chunk-size experiment](inference-recipes/results/glm53-prefill-2026-11-18/) · [Kernel patch and quality checks](inference-recipes/results/glm53-tensorfold-wide-2026-11-18/)
- **Qwen3 0.6B on four replicas.** Sixteen concurrent clients reached 1,894 aggregate output tok/s across four Sparks. At four clients a single Spark was faster. [Replication crossover](inference-recipes/results/qwen3-four-replicas-2026-11-19/)

The inference queue tracks 44 ranked open-weight releases, with pinned checkpoints and one-, two- and four-Spark capacity estimates. [Leaderboard and weight inventory](inference-recipes/leaderboard/)
