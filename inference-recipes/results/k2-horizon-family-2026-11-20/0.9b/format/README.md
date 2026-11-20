# Returning a six-digit record value with K2 Horizon 0.9B

A one-Spark BF16 service found the right value on **64/64 fresh inventory lookups** when it could generate an ordinary answer. Four answers were already bare digits. The other 60 included extra text. Extracting a single standalone six-digit value from a stopped response recovered all 64. This rule came from the first eight outputs and was fixed before these 64 prompts were submitted. [Original eight checks](../lookup-values.jsonl) · [All paired requests](paired.jsonl)

Forcing the first generated character to start a six-digit answer had a different effect. The native vLLM regex constraint produced exactly six digits on every request and **0/64 correct values**. It returned `123455` on 32 prompts and `123500` on 19. Both conditions saw the same prompt and had the same decoding budget. The constraint accepted every six-digit number, with no access to the expected value.

| Records per prompt | Answer format | Correct bare replies | Correct single extracted value | Median generated tokens | Median request time |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | Free | 1/32 | 32/32 | 19 | 256 ms |
| 32 | Six-digit regex | 0/32 | 0/32 | 6 | 95 ms |
| 128 | Free | 3/32 | 32/32 | 19 | 286 ms |
| 128 | Six-digit regex | 0/32 | 0/32 | 6 | 123 ms |

These measurements use one client, 128 output tokens, greedy decoding, low reasoning effort and a 2.16 GB BF16 checkpoint. The free-answer extraction accepts one standalone ASCII six-digit substring from a completed generation. It rejects multiple candidates, a longer run of digits and unfinished generations. The prompt supplied 32 or 128 possible values. Each seed was submitted in both modes, alternating which mode ran first.

## Reproduce

From `inference-recipes/`, start the stock service on a DGX Spark, then run the paired client in a second terminal. The records include visible-output hashes, token counts, latencies and all six-digit matches. [Recipe](../../../../recipes/k2-horizon-0.9b-bf16.json) · [Client and extraction function](../../../../experiments/k2-horizon-family/lookup_format.py)

```sh
python3 spark.py serve k2-horizon-0.9b-bf16
python3 experiments/k2-horizon-family/lookup_format.py k2-horizon-0.9b-bf16 paired.jsonl
```
