# Shortening K2 Horizon 0.9B lookup answers with an assistant prefix

On one Spark, an answer prefix let the 0.9B BF16 model complete all **64 fresh inventory queries with the correct six digits followed by `**`**. The prefix provided the requested bin name and an opening bold marker. The model supplied the value. Unconstrained generation took a median of **5 output tokens and 97 ms**, compared with **19 tokens and 276 ms** for ordinary chat. The stock chat response contained a single correct six-digit value on 62/64 queries. [Prefilled completions](fresh-prefix.jsonl.gz) · [Stock chat](fresh-chat.jsonl.gz)

| Fresh cases, same 64 inventories | Stop with bare correct digits | Exactly one standalone six-digit value, correct | Median output tokens | Median generation request |
| --- | ---: | ---: | ---: | ---: |
| Stock chat, free | 10/64 | 62/64 | 19 | 276 ms |
| Stock chat, six-digit regex | 0/64 | 0/64 | 6 | 118 ms |
| Raw completion, unmodified chat prompt, free | 0/64 | 52/64 | 19 | 269 ms |
| Raw completion, unmodified chat prompt, regex requested | 0/64 | 52/64 | 19 | 273 ms |
| Raw completion, answer prefix, free | 0/64 | 64/64 | 5 | 97 ms |
| Raw completion, answer prefix, regex requested | 0/64 | 64/64 | 5 | 98 ms |

The raw-completion controls used the same tokenized low-effort chat prompt. The prefix added 14 input tokens, closing the reasoning block and starting the visible answer. For example, for `bin-0021` the assistant answer began with `The value of **bin-0021** is:`, then a blank line and `**`. We checked every output hash against the expected six digits followed by two asterisks. The same pattern held on the previous 64 queries. The raw endpoint produced identical bytes with and without the regex on each paired input, while the stock chat endpoint changed output in all 64 cases. [Unmodified raw completions](fresh-raw.jsonl.gz) · [Previous 64 queries replayed with the prefix](original-prefix.jsonl.gz)

Each mode used one client, greedy decoding, a 128-token cap and 32 inventories of each size (32 or 128 records). Free and regex request order alternated by seed. The reported times cover the generation API call. Raw-completion timings exclude client-side template assembly and the separate tokenization calls. A client that needs plain digits can check the continuation format before returning the number. [Paired client](../../../../../experiments/k2-horizon-family/lookup_format.py) · [Pinned recipe](../../../../../recipes/k2-horizon-0.9b-bf16.json)

Run the following from `inference-recipes/` against the one-Spark service. Omit the mode flag for stock chat or use `--raw-control` for the unchanged tokenized prompt.

```sh
python3 experiments/k2-horizon-family/lookup_format.py \
  k2-horizon-0.9b-bf16 fresh-prefix.jsonl --seed 4100 --assistant-prefix
```

## Return the six digits directly

On a third set of 64 new inventories, adding `"stop": ["**"]` to the prefilled completions request made the endpoint return **64/64 exact bare values**. Leaving the stop out gave 0/64 bare values and 64/64 correct values followed by two asterisks. Both arms used identical input token IDs and the 0.9B BF16 model. SHA-256 checks confirmed the complete visible response bytes for every query. The free-completion median fell from 5 to **3 output tokens** and from 91 to **73 ms**. Tokenization and prefix construction are outside those timings. [Fresh stop requests](held-out-stop.jsonl.gz) · [Matched open-ended requests](held-out-open.jsonl.gz)

The stop condition also returned 64/64 bare values when we replayed the previous 64 inventories. On both sets the six-digit regex produced the same output bytes as unconstrained generation. The stop was chosen from the previous set and then held fixed for seeds 5100–5163. [Previous inputs with the stop](repeat-stop.jsonl.gz)

From `inference-recipes/`, repeat the fresh test with the served 0.9B model. Run the same command without `--stop-markdown` and with a new output path for the matched control.

```sh
python3 experiments/k2-horizon-family/lookup_format.py \
  k2-horizon-0.9b-bf16 stop.jsonl --assistant-prefix --stop-markdown --seed 5100
```
