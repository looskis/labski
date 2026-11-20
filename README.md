# labski

labski publishes reproducible experiments on local AI hardware. Each research scope keeps its own code, hardware assumptions and results.

| Research scope | First platform | What you can reproduce |
| --- | --- | --- |
| [Inference recipes](inference-recipes/) | Four NVIDIA DGX Sparks | Serve open-weight models and measure how checkpoint precision, speculative decoding and the number of Sparks affect inference. |

## Measured on DGX Spark

- **Full GLM-5.3 on four Sparks.** The 753B model with a 6-bit output head delivered 32.85 generated tok/s for one client and 63.33 aggregate for four. Both it and the BF16 output head scored 21/28, with one item gained and one lost. [Recipe and measurements](inference-recipes/results/glm53-full-tp4-2026-11-19/)
- **Thinking Machines Inkling 2 Small on four Sparks.** The native 393.48 GB checkpoint delivered 39.34 generated tokens/s across two clients and 2.98 GPU J/output token on the switchless ring. Six inventory lookups were exact up to 3,597 input tokens. [Reproduce the run](inference-recipes/results/inkling2-gb10-runtime-2026-11-19/)
- **Qwen3.8 Flash-Next on one or two Sparks.** The one-Spark NVIDIA FP4-Local checkpoint reached 33.98 tok/s for one client and 72.86 aggregate for four, using 1.08 and 0.52 GPU J/token. The tuned two-Spark recipe reached 21.73 and 64.96 tok/s and returned the first delta sooner at one client. [Recipes, accuracy and raw timing](inference-recipes/results/qwen38-flash-next-2026-11-19/)
- **Code editing with Qwen3.8 27B.** On one Spark, current-request suffix drafting reached 50.74 tok/s against 12.07 for ordinary decoding, with all 12 single-client edits matching token for token. Four clients reached 151.76 aggregate tok/s. Unrelated prose showed much smaller gains or a loss, depending on the proposer. [Four draft policies, paired outputs and GPU energy](inference-recipes/results/qwen38-27b-drafting-2026-11-19/)
- **Qwen3.8 27B on one Spark.** Mixed FP4 delivered 2.78–2.79× the BF16 throughput on the same machines at one client. BF16 scored 83/112 on a short-answer diagnostic and mixed FP4 scored 74/112 and 78/112 on the two machines. [Timing, paired answers and recipes](inference-recipes/results/qwen38-27b-precision-2026-11-19/)
- **One, two or four Sparks for Qwen3.8 27B.** One client reached 12.40, 20.46 and 33.94 output tok/s on one, two and four Sparks. Four clients reached 110.77 tok/s on the switchless RoCE ring using stock NCCL. With native BF16 those four delivered 54.33 tok/s and retained every sampled correctness verdict from the single-Spark BF16 baseline. [Transport, quality and GPU energy](inference-recipes/results/qwen38-27b-scaling-2026-11-19/)
- **GLM‑5.3‑Flash on two Sparks.** Two 8K-chunk runs improved 32K-input throughput by 10.8–11.2% over a 4K chunk. The two chunk settings scored 22/28 and 23/28 respectively on a short-answer check. A separate TensorFold kernel experiment gained 4.3% on the same-length workload while matching the control's sampled answers and likelihoods. [Chunk-size experiment](inference-recipes/results/glm53-prefill-2026-11-18/) · [Kernel patch and quality checks](inference-recipes/results/glm53-tensorfold-wide-2026-11-18/)
- **Qwen3 0.6B on four replicas.** Sixteen concurrent clients reached 1,894 aggregate output tok/s across four Sparks. At four clients a single Spark was faster. [Replication crossover](inference-recipes/results/qwen3-four-replicas-2026-11-19/)

The inference queue tracks 44 ranked open-weight releases, with pinned checkpoints and one-, two- and four-Spark capacity estimates. [Leaderboard and weight inventory](inference-recipes/leaderboard/)
