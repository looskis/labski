# labski

labski publishes reproducible experiments on local AI hardware. Each research scope keeps its own code, hardware assumptions and results.

| Research scope | First platform | What you can reproduce |
| --- | --- | --- |
| [Inference recipes](inference-recipes/) | Four NVIDIA DGX Sparks | Serve open-weight models and measure how checkpoint precision, speculative decoding and the number of Sparks affect inference. |

The first hardware results cover [Qwen3 on one, two and four replicas](inference-recipes/results/qwen3-four-replicas-2026-11-19/) and [GLM‑5.3‑Flash cold prefill on two Sparks](inference-recipes/results/glm53-prefill-2026-11-18/).
