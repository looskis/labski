# KV memory and DFlash2 at four clients

On one GB10 we tested whether allocating more KV memory lets a seven-proposal drafter serve four short requests together. Every run used the same target, draft, runtime and 8,192-token request limit. Four sequence slots were enabled. Both original controls used a 3 GiB pool. [Depth comparison and model revisions](../)

With seven proposals, increasing KV memory from 3 to 9 GiB raised four-client throughput from **48.20 to 92.29 tok/s** and brought the median first streamed delta from **2.10 s to 264 ms**. GPU energy fell from 0.80 to 0.42 J/output token. The 9 GiB run scored 18/28 on short answers and 8/8 on lookups. At one client throughput stayed near 31 tok/s at both capacities.

At **5.5 GiB**, seven proposals delivered **93.86 tok/s at four clients**, with a 296 ms median first delta and 0.42 GPU J/token. Across the three trials, throughput ranged from 86.68 to 95.09 tok/s. All four requests ran together in every trial, with peak KV usage of 98.6%. The 5.5 GiB and original 3 GiB runs answered the same 19/28 questions correctly. The answer strings matched on 26/28 questions. Both passed all eight lookups. These capacity measurements apply to the short prompts and 96-token completions in this workload. [5.5 GiB recipe](../../../recipes/qwen38-27b-dflash-v033-7-kv5632m.json)

![Four-client throughput and first streamed delta for Qwen on one Spark](four-clients.png)

[Plot from per-trial measurements](plot.py)

| Run | KV GiB | Proposals | Reported KV token slots | Output tok/s, 1 / 4 | GPU J/token, 1 / 4 | Median first delta, ms, 1 / 4 | p95 first delta, ms at 4 | Short answers / 28 | Lookups / 8 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Original three](../3/) | 3 | 3 | 17,264 | 25.34 / 86.54 | 1.46 / 0.42 | 174 / 309 | 310 | 19 | 8 |
| [Original seven](../7/) | 3 | 7 | 12,002 | 30.90 / 48.20 | 1.16 / 0.80 | 232 / 2105 | 4308 | 19 | 8 |
| [Larger KV pool](7-kv9/) | 9 | 7 | 36,006 | 30.85 / 92.29 | 1.14 / 0.42 | 235 / 264 | 265 | 18 | 8 |
| [5.5 GiB pool](7-kv5632m/) | 5.5 | 7 | 21,972 | 30.38 / 93.86 | 1.13 / 0.42 | 237 / 296 | 296 | 19 | 8 |

## Larger KV pool

At four clients this run delivered 1.91× the throughput of the original seven and changed GPU energy per generated token by -48.0%. It gained 0 and lost 1 correct short answers. The maximum observed number of running requests in the three repeats was 4, 4, 4. Maximum waiting counts were 0, 0, 0. [Per-trial scheduler samples](7-kv9/scheduler-trials.jsonl) · [Timed trials](7-kv9/trials.jsonl)

Peak sampled KV usage across the four-client trials reached 60.1% of the allocated pool.

## 5.5 GiB pool

At four clients this run delivered 1.95× the throughput of the original seven and changed GPU energy per generated token by -47.5%. It gained 0 and lost 0 correct short answers. The maximum observed number of running requests in the three repeats was 4, 4, 4. Maximum waiting counts were 0, 0, 0. [Per-trial scheduler samples](7-kv5632m/scheduler-trials.jsonl) · [Timed trials](7-kv5632m/trials.jsonl)

Peak sampled KV usage across the four-client trials reached 98.6% of the allocated pool.

Inside the sampled trial windows, the engine reported 7.00 at 1 client, 7.00 at 4 clients proposed tokens per draft block. The counters use the first and last in-window samples and can omit boundary steps. [Observed proposal widths](7-kv5632m/observed-draft-widths.jsonl)

Each rate and latency is a median across three trials. Each trial submitted four 96-token prose requests after an excluded warm-up. GPU energy uses the cumulative NVML counter. The scheduler was polled every 250 ms during the new runs. [Sampler](../sample_metrics.py)

## Reproduce

From `inference-recipes/`, prepare and serve `qwen38-27b-dflash-v033-7-kv9`. Use the matched prose, question and lookup commands from the depth comparison with that alias. [9 GiB launch configuration](../../../recipes/qwen38-27b-dflash-v033-7-kv9.json) · [Benchmark commands](../#reproduce)
