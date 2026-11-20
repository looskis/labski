# Return K2 Horizon 7B final answers at low reasoning effort

On one Spark, changing the vLLM response parser recovered **7/14 open answers at 128 tokens, 9/14 at 512 tokens and 8/8 streamed record lookups**. The stock parser returned 0/14, 0/14 and 0/8 on the same requests. The model, chat template, input hashes, token budgets, observed finish reasons, and generated token counts matched. Each corrected question answer also matched the text following the closing marker in the original output. [128-token pairs](paired-128.jsonl) · [512-token pairs](paired-512.jsonl) · [Streamed lookup pairs](paired-lookups.jsonl)

The 7B template opens `<ifm|think_faster>` at low effort. The checkpoint can close with `</ifm|think>`, the native high-effort closing marker. The stock parser looks for the specific closing marker associated with the requested effort. The lab parser accepts any of the tokenizer’s three native closing markers, preserves tool-call boundaries, and exposes the text following the first boundary as content. As with streamed reasoning, a response truncated before any boundary has no final answer. [Parser implementation](../../../experiments/k2-horizon-family/k2_effort_parser.py) · [Stock parser](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/reasoning/k2_horizon_reasoning_parser.py)

We also submitted **16 fresh lookup prompts** in both streamed and nonstreamed modes, alternating the mode order. All 32 replies stopped with the correct six digits. For every prompt the two modes returned the same full generated token-ID hash, including reasoning. Those hashes compare the two modes of the corrected serving recipe. The tokenizer-level test in the pinned ARM64 image passed 537 text-fragment split points as well as complete-text and token-ID boundary checks. [Fresh paired requests](heldout.jsonl) · [Tokenizer test](../../../experiments/k2-horizon-family/verify_parser.py)

## Reproduce

Start in `inference-recipes/` on the Spark. Both configurations use the same pinned BF16 snapshot and vLLM 0.33 image. [Fixed recipe](../../../recipes/k2-horizon-7b-effort-parser.json) · [Stock configuration](../../../recipes/k2-horizon-7b-bf16.json)

```sh
python3 spark.py prepare k2-horizon-7b-effort-parser
python3 spark.py serve k2-horizon-7b-effort-parser
```

From a second terminal in the same directory, run the fresh paired lookup probe. For the 14 opening MMLU-Pro questions, use the low-effort 128- and 512-token commands from the family benchmark with this alias. [Question setup](../../k2-horizon-32b-2026-11-20/#reproduce)

```sh
python3 experiments/k2-horizon-family/check_endpoint.py k2-horizon-7b-effort-parser heldout.jsonl
```
