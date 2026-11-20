# GLM-5.3-Flash reasoning under a 256-token budget

On two Sparks, native low effort reached a final choice on **28/28** questions and scored **23/28**. Maximum effort reached a final choice on **16/28** and scored **14/28**. Each request had the same 256-token output limit. The difference matters for applications that require an answer within a fixed generation budget. [Per-request answers, token counts and timings](requests.jsonl.gz)

| Native effort | Correct / 28 | Final answers reached | Total output tokens | Median output tokens | Median request time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low | 23 | 28 | 1,327 | 23.5 | 1.95 s |
| Max | 14 | 16 | 5,203 | 206 | 14.59 s |

The model and engine were identical across runs. The client used the checkpoint's own tokenizer and rendered each question with the requested effort, then submitted the exact token IDs to completions. Both runs used greedy decoding on the same two GB10s, one request at a time, with no prefix reuse. Each unanswered or malformed final choice counted as incorrect. Thirteen questions were correct at both settings, ten only at low effort and one only at max effort. Twelve max-effort requests reached the token limit before closing the reasoning span. A larger budget is a separate experiment. [Measured two-node recipe](../../recipes/glm53-chunk4096.json)

On a second workload, low effort returned the correct value in the requested exact format in seven of eight approximately 17K-token lookup prompts. The eighth returned the correct value with its bin label. That distinction matters for a caller that parses the response as a number.

## Reproduce the effort comparison

Follow the [corpus preparation instructions](../qwen38-27b-precision-2026-11-19/#reproduce). The probe selects the first two of the eight deterministic items in each of the 14 subjects. The low and max runs have matching question hashes. The rendered token hashes capture the effort-specific template.

```sh
for effort in low max; do
  python3 glm_effort.py glm53-chunk4096 /tmp/mmlu-112.jsonl \
    --effort "$effort" --per-subject 2 --max-tokens 256 --out "glm-$effort.jsonl"
done
```

For the long prompts, generate seeds 42–49 with `bench.workload("lookup", seed, 1500)`. Store each prompt and expected value in one JSONL row with `id` equal to the seed and `subject` equal to `lookup`, then run the same probe with `--effort low --per-subject 8`. [Prompt generator](../../bench.py) · [Effort probe](../../glm_effort.py)
