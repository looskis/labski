# Step-5 inventory retrieval on 64 fresh prompts

Closing the reasoning block before generation cut mean four-GPU energy per request from **347.8 to 155.9 joules** and median generation-request latency from **2.03 to 0.88 seconds**. Native reasoning returned 64/64 bare six-digit values. Preclosing returned 26/64 bare values. A reader fixed on eight earlier prompts accepted 63/64 preclosed replies and returned the correct value on all 63. [Original eight-case experiment](../../reasoning-384/) · [Fixed reader](../../../../experiments/step5-reasoning/lookup_value.py)

| Prompt ending | Exact bare value | Accepted and correct under fixed reader | Total generated tokens | Mean fleet GPU J/request |
| --- | ---: | ---: | ---: | ---: |
| Native open reasoning block | 64/64 | 64/64 | 2,587 | 347.8 |
| Prefilled closing token | 26/64 | 63/64 | 424 | 155.9 |

Both arms used the same four-Spark compact layout, INT8 GEMV, 384-token budget, low reasoning effort and single-client raw completions. The 64 new prompts used seeds 1000–1063 with 32 records each. We alternated call order within pairs. All 128 requests stopped normally. Of the 38 preclosed responses that failed the bare-value contract, 37 returned the requested key and its exact value in the accepted form. The remaining response read `The value of bin-0022 is 608745.` and the fixed reader rejected it. [Per-request token hashes, normalized values and GPU joules](requests.jsonl.gz) · [Four cumulative energy counters](fleet-energy.csv.gz)

### Retry when the reader declines

We replayed a two-attempt policy over the measured pairs. It tries a preclosed completion first, checks the response with the fixed reader, and requests native reasoning only if the reader declines. The decision uses the requested key and the reply format. A well-formed wrong value would pass that check.

The replay returned the correct six digits on **64/64** queries, with one retry. Summing the actual calls selected per pair gives **160.7 fleet GPU J/query**, 53.8% below always using native reasoning, and 461 generated tokens. The two arms were timed as separate requests. These figures estimate the policy’s cost on this workload and omit prompt tokenization and application overhead. [Runnable client](../../../../experiments/step5-reasoning/lookup_policy.py)

From `inference-recipes/`, generate the paired measurements or serve a single lookup. Set `MODEL` to the active Step-5 endpoint name. For the client, `inventory.txt` contains the records and a question requesting the value of the specified key. [Compact serving configuration](../) · [Paired workload](../../../../experiments/step5-reasoning/pair.py)

```sh
python3 experiments/step5-reasoning/pair.py "$MODEL" "$CORPUS" pairs.jsonl \
  --lookups-only --lookup-seed 1000 --lookup-count 64 --limit 384
python3 experiments/step5-reasoning/lookup_policy.py "$MODEL" bin-0022 inventory.txt
```

## Live retry comparison fixed before collection

Run both the complete retry client and a native-reasoning client on 64 new inventories, seeds 2000–2063. Each client receives the same key and input text. Alternate client order by seed. The retry decision sees only the requested key and completion. Time the full client call, including prompt tokenization and any second generation. Compare the final six-digit values, number of retries, total generated tokens and fleet GPU energy per query. [Live paired benchmark](../../../../experiments/step5-reasoning/benchmark_policy.py)

```sh
python3 experiments/step5-reasoning/benchmark_policy.py "$MODEL" live-pairs.jsonl \
  --seed 2000 --count 64 --limit 384
```
