# Closing Step-5’s reasoning block in the prompt

On four Sparks we compared two greedy 384-token completions for each of 14 MMLU-Pro questions and eight record lookups. Both received the same tokenized official low-effort prompt ending in `<think>`. The second condition appended the single token `</think>` (ID 128799) before generation. The model weights, runtime, context and two-token MTP setting were identical. INT8 GEMV was off.

| Workload | Prompt ending | Exact completed answers | Hit length cap | Generated tokens, total | Median latency, s | Mean fleet GPU J/request |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MMLU-Pro | Open | 9/14 | 4 | 3,082 | 7.95 | 1,348.3 |
| MMLU-Pro | Closed | 7/14 | 0 | 294 | 0.64 | 221.3 |
| Lookup | Open | 8/8 | 0 | 296 | 2.00 | 320.3 |
| Lookup | Closed | 2/8 | 0 | 60 | 0.88 | 158.7 |

On the questions, preclosing the block gained 2 and lost 4 exact answers relative to leaving the block open. Total fleet GPU energy changed by -83.6%. The model generated a new `<think>` marker on 0 of the 14 preclosed prompts.

On the lookups, preclosing the block gained 0 and lost 6 exact answers relative to leaving the block open. Total fleet GPU energy changed by -50.4%. The model generated a new `<think>` marker on 0 of the 8 preclosed prompts.

All eight preclosed replies contained the right value. Six echoed the exact requested key in the form `bin-NNNN = DDDDDD`, and two returned the six digits alone. A deterministic reader that accepts those two forms and checks the requested key recovered **8/8 in each condition**. It rejects other keys, extra prose, multiple records and unfinished replies. This parser was designed after inspecting the eight responses. Use the strict 2/8 result for clients that require a bare value. [Value extractor](../../../experiments/step5-reasoning/lookup_value.py)

The paired order alternated by case. Each condition ran once per prompt at one client. The 14 questions cover one item per subject and do not estimate full-benchmark accuracy. Lookups each contained 32 records. The exact scorer requires a normal stop and a single correct letter or six-digit value. Native answers begin after a generated closing token. In the preclosed condition the entire new response is the answer. Raw token-ID hashes, output lengths, matched prompt hashes, timestamps and per-rank joules accompany the results. The energy figures cover the four GPUs during each HTTP request. [Paired records](requests.jsonl.gz) · [Cumulative GPU counters](fleet-energy.csv.gz)

The same reader subsequently recovered 63/64 preclosed answers on fresh inventories. It rejected one full sentence containing the right value. [Held-out evaluation and retry policy](../rank-local/held-out-lookups/)

## Reproduce

Launch the full four-rank model using the GEMV-off configuration. Run the following from `inference-recipes/` with the fixed MMLU-Pro corpus. The probe checks both special token IDs, the last rendered prompt token, and server-reported counts. It issues token-ID completions and alternates which condition runs first. [Serving configuration](../) · [Probe](../../../experiments/step5-reasoning/pair.py)

```sh
python3 experiments/step5-reasoning/pair.py step5-preview-exl3 "$CORPUS" paired.jsonl --limit 384
```
