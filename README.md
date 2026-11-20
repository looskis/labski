# labski

labski publishes reproducible experiments on local AI hardware. Each research scope keeps its own code, hardware assumptions and results.

| Research scope | First platform | What you can reproduce |
| --- | --- | --- |
| [Inference recipes](inference-recipes/) | Four NVIDIA DGX Sparks | Serve open-weight models and measure how checkpoint precision, speculative decoding and the number of Sparks affect inference. |

## Measured on DGX Spark

- **Qwen3.8 Flash-Next on two Sparks.** Four scheduler slots and 2 GiB of KV per rank reached 64.96 aggregate output tok/s for four clients, 2.90× the one-slot control with 61% lower GPU energy per token. [Recipe, timings and paired quality](inference-recipes/results/qwen38-flash-next-2026-11-19/)
- **Qwen3.8 27B on one Spark.** Mixed FP4 delivered 2.78–2.79× the BF16 throughput on the same machines at one client. BF16 scored 83/112 on a short-answer diagnostic and mixed FP4 scored 74/112 and 78/112 on the two machines. [Timing, paired answers and recipes](inference-recipes/results/qwen38-27b-precision-2026-11-19/)
- **One, two or four Sparks for Qwen3.8 27B.** The direct pair delivered 20.46 output tok/s for one client against 12.40 on one Spark and 8.03 on a four-node routed TCP ring. [Transport, quality and GPU energy](inference-recipes/results/qwen38-27b-scaling-2026-11-19/)
- **GLM‑5.3‑Flash on two Sparks.** Two 8K-chunk runs improved 32K-input throughput by 10.8–11.2% over a 4K chunk. The two chunk settings scored 22/28 and 23/28 respectively on a short-answer check. A separate TensorFold kernel experiment gained 4.3% on the same-length workload while matching the control's sampled answers and likelihoods. [Chunk-size experiment](inference-recipes/results/glm53-prefill-2026-11-18/) · [Kernel patch and quality checks](inference-recipes/results/glm53-tensorfold-wide-2026-11-18/)
- **Qwen3 0.6B on four replicas.** Sixteen concurrent clients reached 1,894 aggregate output tok/s across four Sparks. At four clients a single Spark was faster. [Replication crossover](inference-recipes/results/qwen3-four-replicas-2026-11-19/)

The inference queue tracks 44 ranked open-weight releases, with pinned checkpoints and one-, two- and four-Spark capacity estimates. [Leaderboard and weight inventory](inference-recipes/leaderboard/)
