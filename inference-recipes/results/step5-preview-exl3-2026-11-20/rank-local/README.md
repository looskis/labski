# Step-5 Preview with one expert partition per Spark

Four 128 GB Sparks served the complete Step-5 Preview model from **485.27 GB of safetensors across the fleet**, down from 1,365.52 GB for four full copies. Each rank held all common weights and 88 of 352 routed experts per layer. The compact run generated **19.67 tokens/s for one client and 53.49 aggregate for four**. All eight inventory lookups returned the exact requested value. [Partition, verify and launch](../../../experiments/step5-rank-weights/) · [Full-copy control](../)

| Layout | Weight files per Spark | Output tok/s at 1 / 4 clients | Fleet GPU J/output token at 1 / 4 | Median first streamed delta at 1 / 4 |
| --- | ---: | ---: | ---: | ---: |
| Full copies | 341.38 GB | 20.37 / 57.88 | 6.58 / 2.62 | 287 / 490 ms |
| One expert partition per Spark | 121.32 GB | 19.67 / 53.49 | 6.78 / 2.81 | 336 / 706 ms |

These are separate four-rank launches on the same physical ring. Each used INT8 GEMV, two MTP proposals, an 8,192-token request limit and four sequence slots. The compact launch also enabled the upstream `step3p5` reasoning parser. At the 96-token limit all 24 timed prose requests in each layout ended within reasoning with empty final answers. The rates include the generated reasoning tokens. Each cell is the median of three four-request trials following a warm-up. The observed compact rates were 3.4% and 7.6% lower. The comparison does not isolate storage placement from the parser or launch conditions. [Prompt-matched trials](trials.jsonl) · [Requests](requests.jsonl.gz) · [Settings](settings.jsonl) · [GPU energy](step5-ranklocal-energy.jsonl) · [Raw energy counters](step5-ranklocal-energy.csv.gz)

With a 128-token budget, both layouts returned **7/28** exact final letters. The compact run gained one correct answer and lost one on the matched questions. Final answers agreed on 26/28, and final answer, finish reason and output-token count all agreed on 23/28. Both layouts returned **8/8** exact six-digit lookups with a 256-token budget. Only three lookup pairs took the same number of generated tokens. The full-copy scorer extracted the answer after the closing reasoning delimiter. The compact scorer read the API’s final-answer field. [28 questions](quality.jsonl.gz) · [Eight inventories](lookup.json)

All four ranks reported 81.47 GiB of model memory in both launches. Each launch reported 74,631 KV tokens. Model loading took 550–558 seconds per rank with the compact files and 295–343 seconds with the full copies. The two launches had uncontrolled filesystem cache state. [Per-rank loading](../../../experiments/step5-rank-weights/startup.json) · [Post-load memory snapshot](memory.json) · [ConnectX traffic deltas](rdma.json)

On 14 questions with low reasoning effort, raising the limit from 128 to 512 tokens increased exact final answers from 5/14 to 8/14. Nine and five requests respectively reached the limit. With generation restricted to a single A–J token, 30/112 matched the reference, the same total as the full-copy run. The two runs selected the same token on 102/112 items. The compact endpoint put each forced letter in the reasoning field, and the scorer verified it from the returned token ID. [Low effort, 128](low-128.jsonl.gz) · [Low effort, 512](low-512.jsonl.gz) · [Forced-token records](grammar112.jsonl.gz) · [Per-rank file sizes](weights.jsonl)

On a separate 64-prompt inventory workload, a preclosed prompt and the previously fixed reader returned 63 correct values and declined one reply. Replaying native reasoning only for that reply recovered all 64 at an estimated 53.8% lower fleet GPU energy than always using native reasoning. [Held-out lookup results and retry client](held-out-lookups/)

The subsequent end-to-end client test returned 64/64 fresh inventory values with one retry, using 159.6 versus 340.9 GPU J/query and a median 0.875 versus 2.004 seconds. [Live paired replay](held-out-lookups/#live-end-to-end-result)
